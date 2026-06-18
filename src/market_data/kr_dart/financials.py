from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pandas as pd

from market_data.kr_dart.client import DartClient, dumps_json
from market_data.utils import now_utc_iso

REPORT_CODES = ("11013", "11012", "11014", "11011")
_MISSING_TEXT = {"", "-", "n/a", "na", "none", "null", "nan"}
_NS_LINK = "http://www.xbrl.org/2003/linkbase"
_NS_XBRLI = "http://www.xbrl.org/2003/instance"
_NS_XLINK = "http://www.w3.org/1999/xlink"
_ALLOWED_CONTEXT_DIMENSION_MARKERS = (
    "consolidatedandseparatefinancialstatementsaxis",
    "statementinformationaxis",
)
_STATEMENT_HINTS = {
    "StatementOfFinancialPositionAbstract": "BS",
    "StatementOfComprehensiveIncomeAbstract": "CIS",
    "IncomeStatementAbstract": "IS",
    "StatementOfCashFlowsAbstract": "CF",
    "StatementOfChangesInEquityAbstract": "SCE",
}
_EMPTY_FINANCIAL_FRAME_COLUMNS = [
    "corp_code",
    "ticker",
    "market",
    "bsns_year",
    "reprt_code",
    "fs_div",
    "sj_div",
    "sj_nm",
    "account_id",
    "account_nm",
    "account_detail",
    "account_key",
    "currency",
    "thstrm_amount",
    "thstrm_add_amount",
    "frmtrm_amount",
    "frmtrm_add_amount",
    "frmtrm_q_amount",
    "bfefrmtrm_amount",
    "receipt_no",
    "ord",
    "raw_payload",
    "filing_date",
    "period_end",
    "source",
    "collected_at",
]


def _to_number(value: object) -> float | None:
    text = str(value or "").strip()
    if not text or text in {"-", "N/A", "nan"}:
        return None
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def _series_or_empty(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([None] * len(frame), index=frame.index, dtype=object)


def _clean_text(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in _MISSING_TEXT else text


def _empty_financial_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_FINANCIAL_FRAME_COLUMNS)


def _account_key(account_id: object, account_nm: object, sj_div: object) -> str:
    account_id_text = _clean_text(account_id)
    if account_id_text:
        return account_id_text
    account_name_text = _clean_text(account_nm)
    account_name_text = re.sub(r"\s+", "_", account_name_text)
    account_name_text = account_name_text.replace("/", "_").replace("\\", "_")
    if account_name_text:
        return f"{_clean_text(sj_div)}::{account_name_text}"
    return ""


def _capture_namespaces(raw: bytes) -> dict[str, str]:
    namespaces: dict[str, str] = {}
    for _event, item in ET.iterparse(io.BytesIO(raw), events=("start-ns",)):
        prefix, uri = item
        namespaces[str(prefix or "")] = str(uri)
    return namespaces


def _local_name(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _prefix_for_uri(namespaces: dict[str, str], uri: str) -> str | None:
    for prefix, value in namespaces.items():
        if value == uri:
            return prefix or None
    return None


def _concept_id(tag: str, namespaces: dict[str, str]) -> str:
    if tag.startswith("{"):
        uri, local = tag[1:].split("}", 1)
        prefix = _prefix_for_uri(namespaces, uri)
        if prefix:
            return f"{prefix}_{local}"
        return local
    return tag


def _parse_contexts(root: ET.Element) -> dict[str, dict[str, object]]:
    contexts: dict[str, dict[str, object]] = {}
    ns = {"xbrli": _NS_XBRLI}
    explicit_member_tag = "{http://xbrl.org/2006/xbrldi}explicitMember"
    typed_member_tag = "{http://xbrl.org/2006/xbrldi}typedMember"
    for context in root.findall("xbrli:context", ns):
        context_id = str(context.attrib.get("id") or "").strip()
        if not context_id:
            continue

        period = context.find("xbrli:period", ns)
        instant = pd.NaT
        start_date = pd.NaT
        end_date = pd.NaT
        if period is not None:
            instant_text = _clean_text(period.findtext("xbrli:instant", default="", namespaces=ns))
            start_text = _clean_text(period.findtext("xbrli:startDate", default="", namespaces=ns))
            end_text = _clean_text(period.findtext("xbrli:endDate", default="", namespaces=ns))
            instant = pd.to_datetime(instant_text, errors="coerce")
            start_date = pd.to_datetime(start_text, errors="coerce")
            end_date = pd.to_datetime(end_text, errors="coerce")

        dimension_members: list[str] = []
        members_lower: list[str] = []
        for member in context.iter():
            if member.tag not in {explicit_member_tag, typed_member_tag}:
                continue
            dimension_value = str(member.attrib.get("dimension") or "").strip()
            member_value = _clean_text(member.text)
            marker = " ".join([dimension_value, member_value]).strip()
            if marker:
                dimension_members.append(marker)
                members_lower.append(marker.lower())

        disallowed_dimensions = [
            value
            for value in members_lower
            if not any(marker in value for marker in _ALLOWED_CONTEXT_DIMENSION_MARKERS)
        ]
        actual_fs_div: str | None = None
        if any("separatemember" in value for value in members_lower):
            actual_fs_div = "OFS"
        elif any("consolidatedmember" in value for value in members_lower):
            actual_fs_div = "CFS"

        contexts[context_id] = {
            "instant": instant,
            "start": start_date,
            "end": end_date,
            "dimension_members": dimension_members,
            "dimension_count": len(dimension_members),
            "disallowed_dimensions": disallowed_dimensions,
            "actual_fs_div": actual_fs_div,
        }
    return contexts


def _parse_preferred_labels(label_root: ET.Element) -> dict[str, str]:
    loc_to_concept: dict[str, str] = {}
    label_text_by_id: dict[str, str] = {}
    label_role_by_id: dict[str, str] = {}
    labels_by_concept: dict[str, list[tuple[str, str]]] = {}
    xlink = f"{{{_NS_XLINK}}}"
    link = f"{{{_NS_LINK}}}"

    for label_link in label_root.findall(f"{link}labelLink"):
        for child in label_link:
            if child.tag == f"{link}loc":
                label = str(child.attrib.get(f"{xlink}label") or "").strip()
                href = str(child.attrib.get(f"{xlink}href") or "").strip()
                concept = href.split("#", 1)[-1].strip()
                if label and concept:
                    loc_to_concept[label] = concept
            elif child.tag == f"{link}label":
                label_id = str(child.attrib.get(f"{xlink}label") or "").strip()
                label_role = str(child.attrib.get(f"{xlink}role") or "").strip()
                text = _clean_text(child.text)
                if label_id and text:
                    label_text_by_id[label_id] = text
                    label_role_by_id[label_id] = label_role

        for child in label_link.findall(f"{link}labelArc"):
            loc_label = str(child.attrib.get(f"{xlink}from") or "").strip()
            label_id = str(child.attrib.get(f"{xlink}to") or "").strip()
            concept = loc_to_concept.get(loc_label)
            text = label_text_by_id.get(label_id)
            if concept and text:
                role = label_role_by_id.get(label_id, "")
                labels_by_concept.setdefault(concept, []).append((role, text))

    preferred: dict[str, str] = {}
    for concept, values in labels_by_concept.items():
        values = sorted(
            values,
            key=lambda item: (
                0 if "dart_label" in item[0] else 1,
                0 if item[0].endswith("/label") else 1,
                len(item[1]),
            ),
        )
        preferred[concept] = values[0][1]
    return preferred


def _statement_code_from_presentation_link(presentation_link: ET.Element) -> str | None:
    xlink = f"{{{_NS_XLINK}}}"
    link = f"{{{_NS_LINK}}}"
    for locator in presentation_link.findall(f"{link}loc"):
        href = str(locator.attrib.get(f"{xlink}href") or "").strip()
        concept = href.split("#", 1)[-1]
        for marker, sj_div in _STATEMENT_HINTS.items():
            if concept.endswith(marker):
                return sj_div
    return None


def _parse_statement_roles(pre_root: ET.Element) -> dict[str, str]:
    xlink = f"{{{_NS_XLINK}}}"
    link = f"{{{_NS_LINK}}}"
    concept_statements: dict[str, tuple[int, str]] = {}
    statement_rank = {"BS": 0, "IS": 1, "CIS": 2, "CF": 3, "SCE": 4}

    for presentation_link in pre_root.findall(f"{link}presentationLink"):
        sj_div = _statement_code_from_presentation_link(presentation_link)
        if not sj_div:
            continue
        for locator in presentation_link.findall(f"{link}loc"):
            href = str(locator.attrib.get(f"{xlink}href") or "").strip()
            concept = href.split("#", 1)[-1].strip()
            if not concept:
                continue
            existing = concept_statements.get(concept)
            candidate = (statement_rank.get(sj_div, 99), sj_div)
            if existing is None or candidate < existing:
                concept_statements[concept] = candidate

    return {concept: sj_div for concept, (_rank, sj_div) in concept_statements.items()}


def _open_xbrl_archive(zip_bytes: bytes) -> tuple[bytes, bytes | None, bytes | None]:
    with ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = archive.namelist()
        instance_name = next((name for name in names if name.lower().endswith(".xbrl")), None)
        if instance_name is None:
            raise RuntimeError("DART fnlttXbrl archive did not contain an .xbrl instance")
        lab_name = next((name for name in names if name.lower().endswith("_lab-ko.xml")), None)
        pre_name = next((name for name in names if name.lower().endswith("_pre.xml")), None)
        return (
            archive.read(instance_name),
            archive.read(lab_name) if lab_name else None,
            archive.read(pre_name) if pre_name else None,
        )


def _is_xbrl_fact_candidate(
    element: ET.Element,
    *,
    period_end: pd.Timestamp | None,
    context: dict[str, object],
) -> tuple[bool, str, int]:
    if context.get("disallowed_dimensions"):
        return False, "", 0

    instant = pd.to_datetime(context.get("instant"), errors="coerce")
    start_date = pd.to_datetime(context.get("start"), errors="coerce")
    end_date = pd.to_datetime(context.get("end"), errors="coerce")
    if period_end is not None and pd.notna(period_end):
        target = pd.Timestamp(period_end).normalize()
        if pd.notna(instant) and instant.normalize() == target:
            return True, "instant", 0
        if pd.notna(end_date) and end_date.normalize() == target:
            duration_days = 0
            if pd.notna(start_date):
                duration_days = max(int((end_date - start_date).days), 0)
            return True, "duration", duration_days
        return False, "", 0

    if pd.notna(instant):
        return True, "instant", 0
    if pd.notna(end_date):
        duration_days = 0
        if pd.notna(start_date):
            duration_days = max(int((end_date - start_date).days), 0)
        return True, "duration", duration_days
    return False, "", 0


def _parse_xbrl_financials(
    *,
    zip_bytes: bytes,
    corp_code: str,
    ticker: str,
    bsns_year: int,
    reprt_code: str,
    requested_fs_div: str,
    receipt_no: str,
    filing_date: object = None,
    period_end: object = None,
) -> pd.DataFrame:
    instance_bytes, label_bytes, pre_bytes = _open_xbrl_archive(zip_bytes)
    namespaces = _capture_namespaces(instance_bytes)
    root = ET.fromstring(instance_bytes)
    contexts = _parse_contexts(root)
    labels = _parse_preferred_labels(ET.fromstring(label_bytes)) if label_bytes else {}
    statements = _parse_statement_roles(ET.fromstring(pre_bytes)) if pre_bytes else {}

    normalized_period_end = pd.to_datetime(period_end, errors="coerce")
    requested_fs_div = str(requested_fs_div or "").strip().upper() or "CFS"
    collected_at = pd.Timestamp(now_utc_iso())

    candidates: list[dict[str, object]] = []
    for element in root.iter():
        context_ref = str(element.attrib.get("contextRef") or "").strip()
        if not context_ref:
            continue
        value = _to_number(element.text)
        if value is None:
            continue
        context = contexts.get(context_ref)
        if context is None:
            continue
        eligible, period_kind, duration_days = _is_xbrl_fact_candidate(
            element,
            period_end=normalized_period_end if pd.notna(normalized_period_end) else None,
            context=context,
        )
        if not eligible:
            continue

        account_id = _concept_id(element.tag, namespaces)
        concept_name = account_id.split("_", 1)[-1]
        account_nm = labels.get(concept_name) or labels.get(account_id) or _local_name(element.tag)
        statement_code = statements.get(concept_name) or statements.get(account_id) or ""
        actual_fs_div = str(context.get("actual_fs_div") or "").strip().upper() or requested_fs_div
        candidates.append(
            {
                "corp_code": corp_code,
                "ticker": ticker,
                "market": "kr",
                "bsns_year": int(bsns_year),
                "reprt_code": str(reprt_code),
                "fs_div": actual_fs_div,
                "sj_div": statement_code,
                "sj_nm": statement_code,
                "account_id": account_id,
                "account_nm": account_nm,
                "account_detail": "",
                "account_key": _account_key(account_id, account_nm, statement_code),
                "currency": _clean_text(element.attrib.get("unitRef")),
                "thstrm_amount": value,
                "thstrm_add_amount": None,
                "frmtrm_amount": None,
                "frmtrm_add_amount": None,
                "frmtrm_q_amount": None,
                "bfefrmtrm_amount": None,
                "receipt_no": receipt_no,
                "ord": None,
                "raw_payload": dumps_json(
                    {
                        "tag": element.tag,
                        "contextRef": context_ref,
                        "unitRef": element.attrib.get("unitRef"),
                        "decimals": element.attrib.get("decimals"),
                        "fs_div": actual_fs_div,
                        "period_kind": period_kind,
                        "duration_days": duration_days,
                        "dimensions": context.get("dimension_members"),
                    }
                ),
                "filing_date": pd.to_datetime(filing_date, errors="coerce"),
                "period_end": normalized_period_end,
                "source": "dart:fnlttXbrl",
                "collected_at": collected_at,
                "__period_kind": period_kind,
                "__duration_days": duration_days,
                "__dimension_count": int(context.get("dimension_count") or 0),
                "__fs_priority": 0 if actual_fs_div == requested_fs_div else 1,
            }
        )

    if not candidates:
        return _empty_financial_frame()

    frame = pd.DataFrame(candidates)
    requested_subset = frame.loc[frame["fs_div"].astype(str).str.upper() == requested_fs_div].copy()
    if not requested_subset.empty:
        requested_statement_subset = requested_subset.loc[requested_subset["sj_div"].astype(str).str.strip().ne("")].copy()
        any_statement_subset = frame.loc[frame["sj_div"].astype(str).str.strip().ne("")].copy()
        if not any_statement_subset.empty and requested_statement_subset.empty:
            requested_subset = pd.DataFrame()
    if not requested_subset.empty:
        frame = requested_subset

    if frame["sj_div"].astype(str).str.strip().ne("").any():
        frame = frame.loc[frame["sj_div"].astype(str).str.strip().ne("")].copy()

    frame = frame.sort_values(
        [
            "__fs_priority",
            "account_id",
            "sj_div",
            "__period_kind",
            "__duration_days",
            "__dimension_count",
        ],
        ascending=[True, True, True, True, False, True],
    )

    selected_rows: list[pd.Series] = []
    for (_, account_id, sj_div), chunk in frame.groupby(
        ["fs_div", "account_id", "sj_div"],
        sort=False,
        dropna=False,
    ):
        exact_statement = chunk.loc[chunk["sj_div"].astype(str).ne("")]
        if not exact_statement.empty:
            chunk = exact_statement
        selected_rows.append(chunk.iloc[0].copy())

    out = pd.DataFrame(selected_rows).reset_index(drop=True)
    return out[_EMPTY_FINANCIAL_FRAME_COLUMNS].drop_duplicates(
        subset=["corp_code", "bsns_year", "reprt_code", "fs_div", "sj_div", "account_key"],
        keep="last",
    ).reset_index(drop=True)


def fetch_single_account_financials(
    *,
    corp_code: str,
    ticker: str,
    bsns_year: int,
    reprt_code: str,
    fs_div: str = "CFS",
    receipt_no: str | None = None,
    filing_date: object = None,
    period_end: object = None,
    client: DartClient | None = None,
) -> pd.DataFrame:
    dart = client or DartClient()
    payload: dict[str, object] | None = None
    rows: list[dict[str, object]] = []
    source = "dart:fnlttSinglAcnt"

    if hasattr(dart, "financials_all_accounts"):
        try:
            payload = dart.financials_all_accounts(
                corp_code=corp_code,
                bsns_year=bsns_year,
                reprt_code=reprt_code,
                fs_div=fs_div,
            )
            source = "dart:fnlttSinglAcntAll"
            candidate_rows = payload.get("list") or []
            if isinstance(candidate_rows, list):
                rows = candidate_rows
        except RuntimeError as exc:
            if "status=013" in str(exc):
                source = "dart:fnlttSinglAcntAll"
            else:
                payload = None

    # Some KR filings return no rows from fnlttSinglAcntAll but do return
    # usable Korean account-name rows from fnlttSinglAcnt. Keep probing before
    # we fall back to the XBRL archive, which can be metadata-only for older
    # filings such as LIG넥스원 2018Q2~2019Q2.
    if not rows:
        try:
            payload = dart.financials_single_account(
                corp_code=corp_code,
                bsns_year=bsns_year,
                reprt_code=reprt_code,
                fs_div=fs_div,
            )
            source = "dart:fnlttSinglAcnt"
            candidate_rows = payload.get("list") or []
            if isinstance(candidate_rows, list):
                rows = candidate_rows
        except RuntimeError as exc:
            if "status=013" in str(exc):
                rows = []
            else:
                raise
    if not isinstance(rows, list) or not rows:
        if receipt_no and hasattr(dart, "financials_xbrl"):
            try:
                xbrl_bytes = dart.financials_xbrl(rcept_no=str(receipt_no).strip(), reprt_code=str(reprt_code))
                xbrl_frame = _parse_xbrl_financials(
                    zip_bytes=xbrl_bytes,
                    corp_code=corp_code,
                    ticker=ticker,
                    bsns_year=bsns_year,
                    reprt_code=reprt_code,
                    requested_fs_div=fs_div,
                    receipt_no=str(receipt_no).strip(),
                    filing_date=filing_date,
                    period_end=period_end,
                )
                if not xbrl_frame.empty:
                    return xbrl_frame
            except Exception:
                pass
        return _empty_financial_frame()

    out = pd.DataFrame(rows)
    out["corp_code"] = corp_code
    out["ticker"] = ticker
    out["market"] = "kr"
    out["bsns_year"] = int(bsns_year)
    out["reprt_code"] = reprt_code
    out["fs_div"] = fs_div
    out["sj_div"] = _series_or_empty(out, "sj_div").map(_clean_text)
    out["sj_nm"] = _series_or_empty(out, "sj_nm").map(_clean_text)
    out["account_id"] = _series_or_empty(out, "account_id").map(_clean_text)
    out["account_nm"] = _series_or_empty(out, "account_nm").map(_clean_text)
    out["account_detail"] = _series_or_empty(out, "account_detail").map(_clean_text)
    out["account_key"] = [
        _account_key(account_id, account_nm, sj_div)
        for account_id, account_nm, sj_div in zip(out["account_id"], out["account_nm"], out["sj_div"], strict=False)
    ]
    out["currency"] = _series_or_empty(out, "currency")
    out["thstrm_amount"] = _series_or_empty(out, "thstrm_amount").map(_to_number)
    out["thstrm_add_amount"] = _series_or_empty(out, "thstrm_add_amount").map(_to_number)
    out["frmtrm_amount"] = _series_or_empty(out, "frmtrm_amount").map(_to_number)
    out["frmtrm_add_amount"] = _series_or_empty(out, "frmtrm_add_amount").map(_to_number)
    out["frmtrm_q_amount"] = _series_or_empty(out, "frmtrm_q_amount").map(_to_number)
    out["bfefrmtrm_amount"] = _series_or_empty(out, "bfefrmtrm_amount").map(_to_number)
    out["receipt_no"] = _series_or_empty(out, "rcept_no").map(_clean_text)
    out["ord"] = pd.to_numeric(out.get("ord"), errors="coerce")
    out["raw_payload"] = out.apply(lambda row: dumps_json(row.to_dict()), axis=1)
    out["filing_date"] = pd.to_datetime(filing_date, errors="coerce")
    out["period_end"] = pd.to_datetime(period_end, errors="coerce")
    out["source"] = source
    out["collected_at"] = pd.Timestamp(now_utc_iso())
    return out[_EMPTY_FINANCIAL_FRAME_COLUMNS].drop_duplicates(
        subset=["corp_code", "bsns_year", "reprt_code", "fs_div", "sj_div", "account_key"],
        keep="last",
    ).reset_index(drop=True)
