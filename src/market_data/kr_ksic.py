from __future__ import annotations

from collections import deque
import html
import json
import re

import pandas as pd
import requests
import urllib3

from market_data.config import KSSC_KSIC_BROWSE_URL, KSSC_KSIC_REVISION, KSSC_KSIC_TREE_URL
from market_data.utils import now_utc_iso, retry_call

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_TREE_BASE_PARAMS = {
    "strCategoryNameCode": "001",
    "strCategoryDegree": str(KSSC_KSIC_REVISION),
    "strCategoryCode": "",
    "strCategoryCodeName": "",
}
_TAG_RE = re.compile(r"<[^>]+>")
_TD_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.IGNORECASE | re.DOTALL)
_DIV_RE = re.compile(r"<div[^>]*>(.*?)</div>", re.IGNORECASE | re.DOTALL)
_SPACE_RE = re.compile(r"\s+")

_KSIC_DIM_SCHEMA = [
    "ksic_code",
    "name_ko",
    "name_en",
    "level",
    "depth",
    "parent_code",
    "parent_name_ko",
    "section_code",
    "section_name_ko",
    "division_code",
    "division_name_ko",
    "group_code",
    "group_name_ko",
    "class_code",
    "class_name_ko",
    "subclass_code",
    "subclass_name_ko",
    "revision",
    "source_url",
    "collected_at",
]


def normalize_ksic_code(value: object) -> str | None:
    text = str(value or "").strip().replace(".", "")
    if not text:
        return None
    if re.fullmatch(r"[A-Z]", text):
        return text
    match = re.search(r"\d{2,5}", text)
    if match is None:
        return None
    return match.group(0)


def _clean_text(value: str) -> str:
    return _SPACE_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", value)).replace("\xa0", " ")).strip()


def _split_label(label: str) -> tuple[str, str | None]:
    text = _clean_text(label)
    if text.endswith(")") and " (" in text:
        ko, en = text.rsplit(" (", 1)
        return ko.strip(), en[:-1].strip() or None
    return text, None


def _level_for_code(code: str | None) -> tuple[str, int]:
    if code is None:
        return "unknown", 0
    if re.fullmatch(r"[A-Z]", code):
        return "section", 1
    if re.fullmatch(r"\d{2}", code):
        return "division", 2
    if re.fullmatch(r"\d{3}", code):
        return "group", 3
    if re.fullmatch(r"\d{4}", code):
        return "class", 4
    if re.fullmatch(r"\d{5}", code):
        return "subclass", 5
    return "unknown", len(code)


def _parse_tree_item(item: dict[str, object]) -> dict[str, object]:
    raw_text = str(item.get("text") or "")
    cells = _TD_RE.findall(raw_text)
    code = normalize_ksic_code(_clean_text(cells[0])) if cells else None
    if code is None:
        code = normalize_ksic_code(item.get("id"))
    label_raw = _DIV_RE.search(raw_text)
    name_ko, name_en = _split_label(label_raw.group(1) if label_raw else raw_text)
    level, depth = _level_for_code(code)
    return {
        "ksic_code": code,
        "name_ko": name_ko,
        "name_en": name_en,
        "level": level,
        "depth": depth,
        "has_children": bool(item.get("hasChildren")),
    }


def _extend_lineage(lineage: dict[str, object], parsed: dict[str, object]) -> dict[str, object]:
    out = dict(lineage)
    level = str(parsed.get("level") or "")
    code = parsed.get("ksic_code")
    name_ko = parsed.get("name_ko")
    if level == "section":
        out["section_code"] = code
        out["section_name_ko"] = name_ko
    elif level == "division":
        out["division_code"] = code
        out["division_name_ko"] = name_ko
    elif level == "group":
        out["group_code"] = code
        out["group_name_ko"] = name_ko
    elif level == "class":
        out["class_code"] = code
        out["class_name_ko"] = name_ko
    elif level == "subclass":
        out["subclass_code"] = code
        out["subclass_name_ko"] = name_ko
    return out


def _fetch_tree_nodes(
    session: requests.Session,
    *,
    root: str,
    revision: int,
    timeout: int,
) -> list[dict[str, object]]:
    params = dict(_TREE_BASE_PARAMS)
    params["strCategoryDegree"] = str(int(revision))
    params["root"] = root
    response = retry_call(
        lambda: session.get(KSSC_KSIC_TREE_URL, params=params, timeout=timeout, verify=False),
        retries=3,
        backoff_base=1.0,
        label=f"ksic-tree:{root}",
    )
    response.raise_for_status()
    payload = str(response.text or "").strip()
    if not payload:
        return []
    rows = json.loads(payload)
    return rows if isinstance(rows, list) else []


def fetch_ksic_dim(*, revision: int = KSSC_KSIC_REVISION, timeout: int = 30) -> pd.DataFrame:
    session = requests.Session()
    retry_call(
        lambda: session.get(
            KSSC_KSIC_BROWSE_URL,
            params={"gubun": "1", "strCategoryNameCode": "001", "categoryMenu": "002"},
            timeout=timeout,
            verify=False,
        ),
        retries=2,
        backoff_base=1.0,
        label="ksic-browse",
    )

    rows: list[dict[str, object]] = []
    queue: deque[tuple[str, dict[str, object], str | None, str | None]] = deque(
        [("source", {}, None, None)]
    )
    seen: set[str] = set()

    while queue:
        root, lineage, parent_code, parent_name_ko = queue.popleft()
        for item in _fetch_tree_nodes(session, root=root, revision=revision, timeout=timeout):
            parsed = _parse_tree_item(item)
            code = normalize_ksic_code(parsed.get("ksic_code"))
            if not code or code in seen:
                continue
            seen.add(code)
            node_lineage = _extend_lineage(lineage, parsed)
            row = {
                "ksic_code": code,
                "name_ko": parsed.get("name_ko"),
                "name_en": parsed.get("name_en"),
                "level": parsed.get("level"),
                "depth": parsed.get("depth"),
                "parent_code": parent_code,
                "parent_name_ko": parent_name_ko,
                "section_code": node_lineage.get("section_code"),
                "section_name_ko": node_lineage.get("section_name_ko"),
                "division_code": node_lineage.get("division_code"),
                "division_name_ko": node_lineage.get("division_name_ko"),
                "group_code": node_lineage.get("group_code"),
                "group_name_ko": node_lineage.get("group_name_ko"),
                "class_code": node_lineage.get("class_code"),
                "class_name_ko": node_lineage.get("class_name_ko"),
                "subclass_code": node_lineage.get("subclass_code"),
                "subclass_name_ko": node_lineage.get("subclass_name_ko"),
                "revision": int(revision),
                "source_url": KSSC_KSIC_TREE_URL,
                "collected_at": pd.Timestamp(now_utc_iso()),
            }
            rows.append(row)
            if bool(parsed.get("has_children")):
                queue.append((code, node_lineage, code, str(parsed.get("name_ko") or "")))

    if not rows:
        return pd.DataFrame(columns=_KSIC_DIM_SCHEMA)
    out = pd.DataFrame(rows)
    for column in _KSIC_DIM_SCHEMA:
        if column not in out.columns:
            out[column] = pd.NA
    return out[_KSIC_DIM_SCHEMA].drop_duplicates(subset=["ksic_code"], keep="last").reset_index(drop=True)


def empty_ksic_dim() -> pd.DataFrame:
    return pd.DataFrame(columns=_KSIC_DIM_SCHEMA)
