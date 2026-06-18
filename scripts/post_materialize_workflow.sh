#!/usr/bin/env bash
# Post-materialize workflow: compare butler financials → segment backfill
# Run AFTER batch_materialize_kr.py completes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/logs"
TS="$(date +%Y%m%d_%H%M%S)"
PYTHON="$ROOT/.venv/bin/python"

echo "[$(date)] === Post-materialize workflow start ===" | tee "$LOG_DIR/post_materialize_${TS}.log"

# 1) Compare butler financials
echo "[$(date)] Step 1: Compare butler financials" | tee -a "$LOG_DIR/post_materialize_${TS}.log"
"$PYTHON" "$ROOT/scripts/compare_butler_financials.py" 2>&1 \
  | tee "$LOG_DIR/compare_butler_${TS}.log" \
  | grep -E "Overall|±5%|±10%|±15%|CAPEX|매출|영업|순이익|segment|Total" || true

echo "[$(date)] Step 1 done. Log: $LOG_DIR/compare_butler_${TS}.log" | tee -a "$LOG_DIR/post_materialize_${TS}.log"

# 2) Segment backfill (519 remaining Butler tickers)
echo "[$(date)] Step 2: Segment backfill" | tee -a "$LOG_DIR/post_materialize_${TS}.log"
"$PYTHON" "$ROOT/scripts/kr_dart_segments_backfill.py" --start-year 2013 2>&1 \
  | tee "$LOG_DIR/segment_backfill_${TS}.log" || true

echo "[$(date)] Step 2 done. Log: $LOG_DIR/segment_backfill_${TS}.log" | tee -a "$LOG_DIR/post_materialize_${TS}.log"

echo "[$(date)] === Post-materialize workflow complete ===" | tee -a "$LOG_DIR/post_materialize_${TS}.log"
