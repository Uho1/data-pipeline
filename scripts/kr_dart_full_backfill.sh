#!/usr/bin/env zsh

set -euo pipefail

cd /Users/uho/projects/market_data_lake

.venv/bin/python -u scripts/kr_dart_failover_backfill.py "$@"
