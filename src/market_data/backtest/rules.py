from __future__ import annotations

import ast
import re
from typing import Any

import numpy as np
import pandas as pd


_ALLOWED_COMPARE_OPS = (
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


def _normalize_rule_expression(expr: str) -> str:
    text = str(expr or "").strip()
    text = re.sub(r"\bAND\b", " and ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bOR\b", " or ", text, flags=re.IGNORECASE)
    text = text.replace("&&", " and ").replace("||", " or ")
    text = text.replace("&", " and ").replace("|", " or ")
    return text


def _ensure_series(value: Any, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.reindex(index)
    if isinstance(value, pd.Index):
        return pd.Series(value, index=index)
    return pd.Series([value] * len(index), index=index)


def _to_bool_series(value: Any, index: pd.Index) -> pd.Series:
    out = _ensure_series(value, index)
    if out.dtype != bool:
        out = out.astype("boolean")
    return out.astype("boolean")


def _compare(left: Any, right: Any, op: ast.cmpop, index: pd.Index) -> pd.Series:
    l = _ensure_series(left, index)
    r = _ensure_series(right, index)

    if isinstance(op, ast.Eq):
        return l == r
    if isinstance(op, ast.NotEq):
        return l != r
    if isinstance(op, ast.Lt):
        return l < r
    if isinstance(op, ast.LtE):
        return l <= r
    if isinstance(op, ast.Gt):
        return l > r
    if isinstance(op, ast.GtE):
        return l >= r
    raise ValueError(f"Unsupported comparison operator: {type(op).__name__}")


def _eval_node(node: ast.AST, context: dict[str, Any], index: pd.Index) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, context, index)

    if isinstance(node, ast.Name):
        key = node.id
        if key in context:
            return context[key]
        key_low = key.lower()
        if key_low in context:
            return context[key_low]
        raise ValueError(f"Unknown field in rule: {key}")

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, context, index)
        if isinstance(node.op, ast.Not):
            return ~_to_bool_series(operand, index)
        if isinstance(node.op, ast.Invert):
            return ~_to_bool_series(operand, index)
        if isinstance(node.op, ast.USub):
            return -_ensure_series(operand, index)
        if isinstance(node.op, ast.UAdd):
            return _ensure_series(operand, index)
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

    if isinstance(node, ast.BoolOp):
        values = [_to_bool_series(_eval_node(v, context, index), index) for v in node.values]
        if not values:
            return pd.Series(False, index=index)
        out = values[0]
        for v in values[1:]:
            if isinstance(node.op, ast.And):
                out = out & v
            elif isinstance(node.op, ast.Or):
                out = out | v
            else:
                raise ValueError(f"Unsupported boolean operator: {type(node.op).__name__}")
        return out

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, context, index)
        right = _eval_node(node.right, context, index)
        l = _ensure_series(left, index)
        r = _ensure_series(right, index)
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
        if isinstance(node.op, ast.BitAnd):
            return _to_bool_series(l, index) & _to_bool_series(r, index)
        if isinstance(node.op, ast.BitOr):
            return _to_bool_series(l, index) | _to_bool_series(r, index)
        raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")

    if isinstance(node, ast.Compare):
        for op in node.ops:
            if not isinstance(op, _ALLOWED_COMPARE_OPS):
                raise ValueError(f"Unsupported comparison operator: {type(op).__name__}")

        left = _eval_node(node.left, context, index)
        out = pd.Series(True, index=index)
        for op, right_node in zip(node.ops, node.comparators):
            right = _eval_node(right_node, context, index)
            out = out & _compare(left, right, op, index)
            left = right
        return out

    raise ValueError(f"Unsupported rule syntax: {type(node).__name__}")


def evaluate_rule(
    expr: str,
    frame: pd.DataFrame,
    na_policy: str = "fail",
) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=bool)

    text = _normalize_rule_expression(expr)
    if not text:
        return pd.Series(True, index=frame.index, dtype=bool)

    parsed = ast.parse(text, mode="eval")
    context: dict[str, Any] = {}
    for col in frame.columns:
        context[col] = frame[col]
        context[col.lower()] = frame[col]
        context[col.upper()] = frame[col]

    raw = _eval_node(parsed, context=context, index=frame.index)
    out = _to_bool_series(raw, frame.index)

    policy = str(na_policy or "fail").strip().lower()
    if policy == "pass":
        out = out.fillna(True)
    else:
        out = out.fillna(False)

    return out.astype(bool)


def evaluate_rule_detailed(
    expr: str,
    frame: pd.DataFrame,
    na_policy: str = "fail",
) -> tuple[pd.Series, dict[str, dict[str, bool]] | None]:
    mask = evaluate_rule(expr, frame, na_policy=na_policy)
    if frame is None or frame.empty:
        return mask, {}

    text = _normalize_rule_expression(expr)
    if not text:
        return mask, {}

    text_low = text.lower()
    if " or " in text_low or "(" in text or ")" in text:
        # Fallback for complex expressions: keep mask only.
        return mask, None

    atoms = [part.strip() for part in re.split(r"\band\b", text, flags=re.IGNORECASE) if part.strip()]
    if not atoms:
        return mask, {}

    detail_by_symbol: dict[str, dict[str, bool]] = {}
    for atom in atoms:
        try:
            atom_mask = evaluate_rule(atom, frame, na_policy=na_policy)
        except Exception:
            return mask, None
        token = re.sub(r"\s+", "", atom)
        for sym, passed in atom_mask.items():
            key = str(sym).upper()
            bucket = detail_by_symbol.setdefault(key, {})
            bucket[token] = bool(passed)

    return mask, detail_by_symbol


def build_screen_candidates(
    date_t: pd.Timestamp,
    panel_t: pd.DataFrame,
    rule_expr: str,
    universe_filter: pd.Series | None = None,
    na_policy: str = "fail",
) -> pd.DataFrame:
    if panel_t is None or panel_t.empty:
        return pd.DataFrame()

    frame = panel_t.copy()
    mask = evaluate_rule(rule_expr, frame, na_policy=na_policy)
    if universe_filter is not None:
        uf = universe_filter.reindex(frame.index).fillna(False).astype(bool)
        mask = mask & uf

    out = frame.loc[mask].copy()
    out["signal_date"] = pd.Timestamp(date_t)
    return out


def evaluate_buy_and_sell_rules(
    frame: pd.DataFrame,
    buy_expr: str,
    sell_expr: str | None,
    na_policy: str = "fail",
) -> tuple[pd.Series, pd.Series]:
    buy_mask = evaluate_rule(buy_expr, frame, na_policy=na_policy)
    if sell_expr is None or not str(sell_expr).strip():
        sell_mask = ~buy_mask
    else:
        sell_mask = evaluate_rule(sell_expr, frame, na_policy=na_policy)
    return buy_mask, sell_mask
