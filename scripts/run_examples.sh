#!/usr/bin/env bash
set -euo pipefail

python -m market_data ingest --universe sp500 --start 2000-01-01
python -m market_data ingest --universe kospi --tickers-file tickers_kospi.csv --start 2000-01-01
python -m market_data validate
python -m market_data sample --ticker AAPL
