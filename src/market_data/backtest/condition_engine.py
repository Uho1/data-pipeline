from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from market_data.backtest.models import ConditionClause, ConditionSet


_COMPARE_OPS = {
    "<": ast.Lt,
    "<=": ast.LtE,
    ">": ast.Gt,
    ">=": ast.GtE,
    "==": ast.Eq,
    "!=": ast.NotEq,
}

_FUNC_ALIASES = {
    "sma": "sma",
    "이동평균": "sma",
    "rank_desc": "rank_desc",
    "랭크내림차순": "rank_desc",
    "rank_asc": "rank_asc",
    "랭크오름차순": "rank_asc",
    "percentile": "percentile",
    "백분위": "percentile",
    "zscore": "zscore",
    "abs": "abs",
    "min": "min",
    "max": "max",
    "clip": "clip",
}


@dataclass
class ConditionValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]


def _normalize_expr(expr: str) -> str:
    text = str(expr or "").strip()
    text = re.sub(r"\{([^}]+)\}", r"\1", text)
    return text


def _col_lookup(frame: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in frame.columns:
        out[str(c)] = str(c)
        out[str(c).lower()] = str(c)
        out[str(c).upper()] = str(c)
    return out


def _ensure_series(value: Any, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.reindex(index)
    return pd.Series([value] * len(index), index=index)


def _to_bool_series(value: Any, index: pd.Index, na_policy: str = "fail") -> pd.Series:
    out = _ensure_series(value, index)
    out = out.astype("boolean")
    if str(na_policy).lower() == "pass":
        out = out.fillna(True)
    else:
        out = out.fillna(False)
    return out.astype(bool)


def _apply_compare(left: Any, right: Any, op: str, index: pd.Index) -> pd.Series:
    l = _ensure_series(left, index)
    r = _ensure_series(right, index)
    if op == "<":
        return l < r
    if op == "<=":
        return l <= r
    if op == ">":
        return l > r
    if op == ">=":
        return l >= r
    if op == "==":
        return l == r
    if op == "!=":
        return l != r
    raise ValueError(f"Unsupported op: {op}")


def _resolve_name_series(name: str, frame: pd.DataFrame, col_map: dict[str, str]) -> pd.Series:
    if name in col_map:
        return pd.to_numeric(frame[col_map[name]], errors="coerce")
    raise ValueError(f"Unknown factor/column: {name}")


def _eval_series_expr(
    node: ast.AST,
    frame: pd.DataFrame,
    col_map: dict[str, str],
    panel: pd.DataFrame | None = None,
    signal_date: pd.Timestamp | None = None,
) -> pd.Series:
    idx = frame.index
    if isinstance(node, ast.Name):
        return _resolve_name_series(node.id, frame, col_map)
    if isinstance(node, ast.Constant):
        return pd.Series([float(node.value)] * len(idx), index=idx, dtype=float)
    if isinstance(node, ast.UnaryOp):
        val = _eval_series_expr(node.operand, frame, col_map, panel=panel, signal_date=signal_date)
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.UAdd):
            return val
        raise ValueError(f"Unsupported unary: {type(node.op).__name__}")
    if isinstance(node, ast.BinOp):
        l = _eval_series_expr(node.left, frame, col_map, panel=panel, signal_date=signal_date)
        r = _eval_series_expr(node.right, frame, col_map, panel=panel, signal_date=signal_date)
        if isinstance(node.op, ast.Add):
            return l + r
        if isinstance(node.op, ast.Sub):
            return l - r
        if isinstance(node.op, ast.Mult):
            return l * r
        if isinstance(node.op, ast.Div):
            return l / r
        if isinstance(node.op, ast.Mod):
            return l % r
        if isinstance(node.op, ast.Pow):
            return l**r
        raise ValueError(f"Unsupported binary: {type(node.op).__name__}")
    if isinstance(node, ast.Call):
        return _eval_call(node, frame, col_map, panel=panel, signal_date=signal_date)
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def _eval_series_expr_history(
    node: ast.AST,
    hist_frame: pd.DataFrame,
    hist_col_map: dict[str, str],
) -> pd.Series:
    idx = hist_frame.index
    if isinstance(node, ast.Name):
        return pd.to_numeric(hist_frame[hist_col_map.get(node.id, node.id)], errors="coerce")
    if isinstance(node, ast.Constant):
        return pd.Series([float(node.value)] * len(idx), index=idx, dtype=float)
    if isinstance(node, ast.UnaryOp):
        s = _eval_series_expr_history(node.operand, hist_frame, hist_col_map)
        if isinstance(node.op, ast.USub):
            return -s
        if isinstance(node.op, ast.UAdd):
            return s
        raise ValueError(f"Unsupported unary in sma: {type(node.op).__name__}")
    if isinstance(node, ast.BinOp):
        l = _eval_series_expr_history(node.left, hist_frame, hist_col_map)
        r = _eval_series_expr_history(node.right, hist_frame, hist_col_map)
        if isinstance(node.op, ast.Add):
            return l + r
        if isinstance(node.op, ast.Sub):
            return l - r
        if isinstance(node.op, ast.Mult):
            return l * r
        if isinstance(node.op, ast.Div):
            return l / r
        if isinstance(node.op, ast.Mod):
            return l % r
        if isinstance(node.op, ast.Pow):
            return l**r
        raise ValueError(f"Unsupported binary in sma: {type(node.op).__name__}")
    if isinstance(node, ast.Call):
        # Keep it conservative for timeseries mode.
        if isinstance(node.func, ast.Name):
            fn = _FUNC_ALIASES.get(node.func.id, "")
            if fn == "abs" and len(node.args) == 1:
                return _eval_series_expr_history(node.args[0], hist_frame, hist_col_map).abs()
            if fn == "clip" and len(node.args) in {2, 3}:
                src = _eval_series_expr_history(node.args[0], hist_frame, hist_col_map)
                lo = float(ast.literal_eval(node.args[1])) if len(node.args) >= 2 else None
                hi = float(ast.literal_eval(node.args[2])) if len(node.args) >= 3 else None
                return src.clip(lower=lo, upper=hi)
        raise ValueError("Unsupported nested function in sma() expression")
    raise ValueError(f"Unsupported node in sma() expression: {type(node).__name__}")


def _eval_call(
    node: ast.Call,
    frame: pd.DataFrame,
    col_map: dict[str, str],
    panel: pd.DataFrame | None = None,
    signal_date: pd.Timestamp | None = None,
) -> pd.Series:
    if not isinstance(node.func, ast.Name):
        raise ValueError("Only simple function names are allowed")
    fn = _FUNC_ALIASES.get(node.func.id, "")
    if not fn:
        raise ValueError(f"Unsupported function: {node.func.id}")

    idx = frame.index
    if fn == "sma":
        if len(node.args) != 2:
            raise ValueError("sma(expr, window) requires 2 args")
        if panel is None or panel.empty or signal_date is None:
            return pd.Series(np.nan, index=idx)
        try:
            window = int(ast.literal_eval(node.args[1]))
        except Exception as exc:  # noqa: BLE001
            raise ValueError("sma window must be an integer") from exc
        if window <= 0:
            raise ValueError("sma window must be > 0")

        out = pd.Series(np.nan, index=idx, dtype=float)
        for sym in idx:
            key = str(sym).upper()
            try:
                hist = panel.xs(key, level=1).sort_index()
            except Exception:
                continue
            hist = hist.loc[hist.index <= pd.Timestamp(signal_date)]
            if hist.empty:
                continue
            hist_col_map = _col_lookup(hist)
            try:
                series_hist = _eval_series_expr_history(node.args[0], hist, hist_col_map)
            except Exception:
                continue
            if series_hist.dropna().empty:
                continue
            out.loc[sym] = float(series_hist.rolling(window=window, min_periods=window).mean().iloc[-1])
        return out

    src = _eval_series_expr(node.args[0], frame, col_map, panel=panel, signal_date=signal_date)

    if fn == "rank_desc":
        return pd.to_numeric(src, errors="coerce").rank(method="average", ascending=False)
    if fn == "rank_asc":
        return pd.to_numeric(src, errors="coerce").rank(method="average", ascending=True)
    if fn == "percentile":
        return pd.to_numeric(src, errors="coerce").rank(method="average", ascending=True, pct=True) * 100.0
    if fn == "zscore":
        s = pd.to_numeric(src, errors="coerce")
        mu = float(s.mean(skipna=True))
        sd = float(s.std(skipna=True))
        if not np.isfinite(sd) or sd == 0.0:
            return pd.Series(0.0, index=s.index)
        return (s - mu) / sd
    if fn == "abs":
        return pd.to_numeric(src, errors="coerce").abs()
    if fn == "min":
        if len(node.args) != 2:
            raise ValueError("min(a,b) requires 2 args")
        other = _eval_series_expr(node.args[1], frame, col_map, panel=panel, signal_date=signal_date)
        return pd.Series(np.minimum(pd.to_numeric(src, errors="coerce"), pd.to_numeric(other, errors="coerce")), index=idx)
    if fn == "max":
        if len(node.args) != 2:
            raise ValueError("max(a,b) requires 2 args")
        other = _eval_series_expr(node.args[1], frame, col_map, panel=panel, signal_date=signal_date)
        return pd.Series(np.maximum(pd.to_numeric(src, errors="coerce"), pd.to_numeric(other, errors="coerce")), index=idx)
    if fn == "clip":
        lo = float(ast.literal_eval(node.args[1])) if len(node.args) >= 2 else None
        hi = float(ast.literal_eval(node.args[2])) if len(node.args) >= 3 else None
        return pd.to_numeric(src, errors="coerce").clip(lower=lo, upper=hi)
    raise ValueError(f"Unsupported function: {fn}")


def validate_condition_set(
    condition_set: ConditionSet,
    available_columns: list[str] | None = None,
) -> ConditionValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    labels = {str(c.label).strip().upper() for c in condition_set.conditions}

    if not condition_set.conditions:
        return ConditionValidationResult(ok=True, errors=[], warnings=["No conditions defined"])

    allowed_cols = {str(c) for c in (available_columns or [])}
    allowed_cols_low = {str(c).lower() for c in (available_columns or [])}

    for clause in condition_set.conditions:
        if clause.op not in _COMPARE_OPS:
            errors.append(f"[{clause.label}] unsupported operator: {clause.op}")
        text = _normalize_expr(clause.left)
        try:
            node = ast.parse(text, mode="eval")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"[{clause.label}] parse error: {exc}")
            continue
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                if not isinstance(n.func, ast.Name):
                    errors.append(f"[{clause.label}] invalid function call syntax")
                    continue
                if n.func.id not in _FUNC_ALIASES:
                    errors.append(f"[{clause.label}] unsupported function: {n.func.id}")
            elif isinstance(n, ast.Name):
                if n.id in _FUNC_ALIASES:
                    continue
                if available_columns:
                    if n.id not in allowed_cols and n.id.lower() not in allowed_cols_low and n.id.upper() not in allowed_cols:
                        warnings.append(f"[{clause.label}] unknown factor name: {n.id}")
            elif isinstance(n, (ast.Expression, ast.Load, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.USub, ast.UAdd)):
                continue
            else:
                errors.append(f"[{clause.label}] disallowed token: {type(n).__name__}")

    logic = str(condition_set.logical_expression or "").strip()
    if not logic:
        logic = " and ".join(sorted(labels))
    try:
        logic_ast = ast.parse(logic, mode="eval")
        logic_labels: set[str] = set()
        for n in ast.walk(logic_ast):
            if isinstance(n, ast.Name):
                logic_labels.add(n.id.upper())
            elif isinstance(n, (ast.Expression, ast.BoolOp, ast.UnaryOp, ast.And, ast.Or, ast.Not, ast.Load)):
                continue
            else:
                errors.append(f"[logical] disallowed token: {type(n).__name__}")
        for name in sorted(logic_labels):
            if name not in labels:
                errors.append(f"[logical] unknown label reference: {name}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"[logical] parse error: {exc}")

    return ConditionValidationResult(ok=(len(errors) == 0), errors=errors, warnings=warnings)


def evaluate_condition_set(
    condition_set: ConditionSet,
    frame: pd.DataFrame,
    *,
    panel: pd.DataFrame | None = None,
    signal_date: pd.Timestamp | None = None,
    na_policy: str | None = None,
) -> tuple[pd.Series, dict[str, dict[str, bool]]]:
    if frame is None or frame.empty:
        return pd.Series(dtype=bool), {}
    if not condition_set.conditions:
        out = pd.Series(True, index=frame.index, dtype=bool)
        return out, {str(i): {} for i in frame.index}

    policy = str(na_policy or condition_set.na_policy or "fail").strip().lower() or "fail"
    col_map = _col_lookup(frame)
    clause_results: dict[str, pd.Series] = {}

    for clause in condition_set.conditions:
        expr = _normalize_expr(clause.left)
        node = ast.parse(expr, mode="eval").body
        left_series = _eval_series_expr(node, frame, col_map, panel=panel, signal_date=signal_date)
        right_series = pd.Series(float(clause.right), index=frame.index)
        mask = _apply_compare(left_series, right_series, clause.op, frame.index)
        clause_results[clause.label.strip().upper()] = _to_bool_series(mask, frame.index, na_policy=policy)

    logic = str(condition_set.logical_expression or "").strip()
    if not logic:
        logic = " and ".join([c.label.strip().upper() for c in condition_set.conditions])
    logic_ast = ast.parse(logic, mode="eval").body

    def _eval_logic(node: ast.AST) -> pd.Series:
        if isinstance(node, ast.Name):
            key = node.id.strip().upper()
            if key not in clause_results:
                raise ValueError(f"Unknown logical label: {key}")
            return clause_results[key]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return ~_eval_logic(node.operand)
        if isinstance(node, ast.BoolOp):
            if not node.values:
                return pd.Series(False, index=frame.index)
            out = _eval_logic(node.values[0])
            for val in node.values[1:]:
                if isinstance(node.op, ast.And):
                    out = out & _eval_logic(val)
                elif isinstance(node.op, ast.Or):
                    out = out | _eval_logic(val)
                else:
                    raise ValueError("Unsupported logical operator")
            return out
        raise ValueError(f"Unsupported logical token: {type(node).__name__}")

    mask = _to_bool_series(_eval_logic(logic_ast), frame.index, na_policy=policy)
    detail: dict[str, dict[str, bool]] = {}
    for sym in frame.index:
        per = {}
        for label, series in clause_results.items():
            per[label] = bool(series.get(sym, False))
        detail[str(sym).upper()] = per
    return mask.astype(bool), detail


def dump_condition_set_dict(condition_set: ConditionSet) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "logical_expression": condition_set.logical_expression,
        "na_policy": condition_set.na_policy,
        "conditions": [
            {"label": c.label, "left": c.left, "op": c.op, "right": float(c.right)}
            for c in condition_set.conditions
        ],
    }


def load_condition_set_dict(payload: dict[str, Any] | None) -> ConditionSet:
    if not isinstance(payload, dict):
        return ConditionSet()
    clauses: list[ConditionClause] = []
    for item in payload.get("conditions", []) or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip().upper()
        left = str(item.get("left", "")).strip()
        op = str(item.get("op", "")).strip()
        try:
            right = float(item.get("right"))
        except Exception:
            continue
        if not label or not left or op not in _COMPARE_OPS:
            continue
        clauses.append(ConditionClause(label=label, left=left, op=op, right=right))
    return ConditionSet(
        conditions=clauses,
        logical_expression=str(payload.get("logical_expression", "")).strip(),
        na_policy=str(payload.get("na_policy", "fail")).strip().lower() or "fail",
    )

