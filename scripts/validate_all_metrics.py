import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# src 디렉토리를 경로에 추가하여 market_data 모듈을 임포트할 수 있도록 함
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from market_data.db import get_connection
from market_data.db_reader import load_financials_from_db

# 1. 지표 매핑(Mapping) 딕셔너리 자동 생성
# SEC의 DuckDB 컬럼명(METRIC_SPECS / FLOW_COLUMNS / STOCK_COLUMNS 기준) -> FMP API 키값
METRIC_MAPPING = {
    # Income Statement
    "Revenue": "revenue",
    "COGS": "costOfRevenue",
    "Gross Profit": "grossProfit",
    "SG&A": "sellingGeneralAndAdministrativeExpenses",
    "Operating Income": "operatingIncome",
    "Net Income": "netIncome",
    "EPS": "eps",
    "Diluted EPS": "epsdiluted",
    "D&A": "depreciationAndAmortization",
    "Interest": "interestExpense",
    "Pretax Income": "incomeBeforeTax",
    "Tax": "incomeTaxExpense",
    
    # Balance Sheet
    "Total Assets": "totalAssets",
    "Total Liabilities": "totalLiabilities",
    "Shareholders Equity": "totalStockholdersEquity",
    "Current Assets": "totalCurrentAssets",
    "Current Liabilities": "totalCurrentLiabilities",
    "AR": "netReceivables",
    "AP": "accountPayables",
    "Inventory": "inventory",
    "Cash": "cashAndCashEquivalents",
    "Debt Short": "shortTermDebt",
    "Debt Long": "longTermDebt",
    "Deferred Revenue": "deferredRevenue",
    "Goodwill": "goodwill",
    "Intangibles": "intangibleAssets",
    
    # Cash Flow Statement
    "Operating Cash Flow": "operatingCashFlow",
    "Investing Cash Flow": "netCashUsedForInvestingActivites",  # FMP API legacy typo handle
    "Financing Cash Flow": "netCashUsedProvidedByFinancingActivities",
    "Capital Expenditure": "capitalExpenditure",
    "Dividends Paid": "dividendsPaid",
    "Repurchases": "commonStockRepurchased",
    
    # Shares
    "Shares": "weightedAverageShsOut",
    "Diluted Shares": "weightedAverageShsOutDil",
    "Basic Shares": "weightedAverageShsOut",
}

# FMP API key alternative typo mapping just in case
FMP_ALT_KEYS = {
    "netCashUsedForInvestingActivites": "netCashUsedForInvestingActivities"
}

# 테스트 대상
DEFAULT_TICKERS = ['AAPL', 'MSFT', 'TSLA', 'JPM', 'JNJ']

def init_cache_db(con):
    """DuckDB에 FMP 데이터 캐싱을 위한 테이블을 생성합니다."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS fmp_financial_cache (
            ticker VARCHAR,
            year INT,
            quarter INT,
            has_data BOOLEAN,
            fmp_data VARCHAR,
            updated_at TIMESTAMP,
            UNIQUE(ticker, year, quarter)
        )
    """)

def fetch_fmp_api(ticker: str, api_key: str):
    """FMP API에서 분기별 재무제표를 가져와 병합합니다 (최대 40분기 = 10년)."""
    base_url = "https://financialmodelingprep.com/api/v3"
    endpoints = {
        "income": f"{base_url}/income-statement/{ticker}?period=quarter&apikey={api_key}&limit=40",
        "balance": f"{base_url}/balance-sheet-statement/{ticker}?period=quarter&apikey={api_key}&limit=40",
        "cashflow": f"{base_url}/cash-flow-statement/{ticker}?period=quarter&apikey={api_key}&limit=40"
    }
    
    dfs = []
    for name, url in endpoints.items():
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f"    [{ticker}] FMP API 호출 실패 ({name}): HTTP {resp.status_code}")
            continue
        
        data = resp.json()
        if not data:
            continue
            
        df = pd.DataFrame(data)
        if 'calendarYear' in df.columns and 'period' in df.columns:
            df['year'] = df['calendarYear'].astype(str).str.extract(r'(\d+)')[0].astype(float)
            df['quarter'] = df['period'].astype(str).str.extract(r'(\d+)')[0].astype(float)
            df = df.dropna(subset=['year', 'quarter'])
            df['year'] = df['year'].astype(int)
            df['quarter'] = df['quarter'].astype(int)
            dfs.append(df)
        
    if not dfs:
        return pd.DataFrame()
        
    # year, quarter 기준으로 병합 (동일 분기 중복시 가장 최근(index 0) 사용)
    for i in range(len(dfs)):
        dfs[i] = dfs[i].drop_duplicates(subset=['year', 'quarter'])

    merged = dfs[0]
    for i in range(1, len(dfs)):
        merged = pd.merge(merged, dfs[i], on=['year', 'quarter'], how='outer', suffixes=('', '_drop'))
        
    # 중복 컬럼 제거
    merged = merged.loc[:, ~merged.columns.str.endswith('_drop')]
    return merged

def get_cached_fmp_data(con, ticker: str, target_periods: list[tuple[int, int]], api_key: str):
    """
    지정된 (year, quarter) 목록에 대해 DuckDB 캐시를 우선 조회하고,
    누락된 데이터가 있으면 FMP API를 호출하여 캐시를 업데이트한 뒤 반환합니다.
    """
    init_cache_db(con)
    
    # 1. 현재 캐시에 있는 데이터 확인
    placeholders = ", ".join(["(?, ?)"] * len(target_periods))
    params = [ticker]
    for y, q in target_periods:
        params.extend([y, q])
        
    query = f"""
        SELECT year, quarter, has_data, fmp_data 
        FROM fmp_financial_cache 
        WHERE ticker = ? AND (year, quarter) IN ({placeholders})
    """
    cached_rows = con.execute(query, params).fetchall()
    
    cached_dict = {(row[0], row[1]): (row[2], row[3]) for row in cached_rows}
    missing_periods = [p for p in target_periods if p not in cached_dict]
    
    # 2. 누락된 데이터가 있다면 API 호출
    if missing_periods:
        print(f"    [Cache Miss] {ticker}의 일부 기간 누락으로 FMP API를 호출합니다. (누락: {len(missing_periods)}건)")
        api_df = fetch_fmp_api(ticker, api_key)
        
        now = datetime.now()
        upsert_query = """
            INSERT INTO fmp_financial_cache (ticker, year, quarter, has_data, fmp_data, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (ticker, year, quarter) DO UPDATE SET
                has_data = EXCLUDED.has_data,
                fmp_data = EXCLUDED.fmp_data,
                updated_at = EXCLUDED.updated_at
        """
        
        fetched_periods = set()
        if not api_df.empty:
            for _, row in api_df.iterrows():
                y = int(row['year'])
                q = int(row['quarter'])
                fetched_periods.add((y, q))
                row_dict = row.to_dict()
                row_json = json.dumps(row_dict)
                con.execute(upsert_query, [ticker, y, q, True, row_json, now])
                
        # API에서 응답이 오지 않은 타겟 기간은 Negative Cache(데이터 없음) 처리
        for (y, q) in target_periods:
            if (y, q) not in fetched_periods and (y, q) not in cached_dict:
                con.execute(upsert_query, [ticker, y, q, False, None, now])
                
        # API 호출 후 캐시 다시 조회
        cached_rows = con.execute(query, params).fetchall()

    # 3. 캐시 데이터를 DataFrame으로 변환
    results = []
    for row in cached_rows:
        y, q, has_data, fmp_data_json = row
        if has_data and fmp_data_json:
            data_dict = json.loads(fmp_data_json)
            # 확실히 year와 quarter 타입 보장
            data_dict['year'] = y
            data_dict['quarter'] = q
            results.append(data_dict)
            
    if not results:
        return pd.DataFrame()
        
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="SEC vs FMP Metrics Validator with DuckDB Caching")
    parser.add_argument("--start-year", type=int, default=2022, help="검증 시작 연도")
    parser.add_argument("--end-year", type=int, default=2023, help="검증 종료 연도")
    parser.add_argument("--start-quarter", type=int, default=1, help="검증 시작 분기")
    parser.add_argument("--end-quarter", type=int, default=4, help="검증 종료 분기")
    parser.add_argument("--tickers", type=str, nargs='+', default=DEFAULT_TICKERS, help="검증할 종목들 (띄어쓰기로 구분)")
    args = parser.parse_args()

    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        print("에러: FMP_API_KEY 환경변수가 설정되지 않았습니다.")
        print("export FMP_API_KEY='your_api_key_here' 를 실행한 후 다시 시도해주세요.")
        return

    print(f"검증 시작...")
    print(f"대상 종목: {args.tickers}")
    print(f"기간: {args.start_year} Q{args.start_quarter} ~ {args.end_year} Q{args.end_quarter}")
    
    # 검증 대상 (연도, 분기) 리스트 생성
    target_periods = []
    for y in range(args.start_year, args.end_year + 1):
        q_start = args.start_quarter if y == args.start_year else 1
        q_end = args.end_quarter if y == args.end_year else 4
        for q in range(q_start, q_end + 1):
            target_periods.append((y, q))
            
    if not target_periods:
        print("검증할 기간이 올바르지 않습니다.")
        return

    results = []
    con = get_connection()
    
    for ticker in args.tickers:
        print(f"\n-> {ticker} 데이터 처리 중...")
        
        # DuckDB SEC 데이터 가져오기
        try:
            duck_df = load_financials_from_db(ticker)
        except Exception as e:
            print(f"  [DuckDB] {ticker} SEC 데이터 로드 실패: {e}")
            continue
            
        if duck_df is None or duck_df.empty:
            print(f"  [DuckDB] {ticker} SEC 데이터가 비어있습니다.")
            continue
            
        if 'term' not in duck_df.columns:
            print(f"  [DuckDB] {ticker} SEC 데이터에 'term' 컬럼이 없습니다.")
            continue
            
        duck_df['year'] = pd.to_numeric(duck_df['term'].astype(str).str.slice(0, 4), errors='coerce').fillna(0).astype(int)
        duck_df['quarter'] = pd.to_numeric(duck_df['term'].astype(str).str.slice(-1), errors='coerce').fillna(0).astype(int)
        
        # 목표 기간에 맞는 데이터만 필터링
        period_df = pd.DataFrame(target_periods, columns=['year', 'quarter'])
        duck_df = pd.merge(duck_df, period_df, on=['year', 'quarter'], how='inner')
        
        if duck_df.empty:
            print(f"  [DuckDB] {ticker} 목표 기간에 해당하는 SEC 데이터가 없습니다.")
            continue
            
        # FMP 캐시 및 API 데이터 가져오기
        fmp_df = get_cached_fmp_data(con, ticker, target_periods, api_key)
        if fmp_df.empty:
            print(f"  [FMP] {ticker} 유효한 데이터를 가져오지 못했습니다.")
            continue
            
        # 3. 검증 로직 및 오차 계산
        merged = pd.merge(duck_df, fmp_df, on=['year', 'quarter'], how='inner')
        
        if merged.empty:
            print(f"  {ticker} DuckDB와 FMP 간에 일치하는 분기 데이터가 없습니다.")
            continue
            
        for _, row in merged.iterrows():
            year = row['year']
            quarter = row['quarter']
            
            for sec_metric, fmp_key in METRIC_MAPPING.items():
                if sec_metric not in row:
                    continue
                
                # FMP 키 대안 처리
                actual_fmp_key = fmp_key
                if fmp_key not in row and fmp_key in FMP_ALT_KEYS:
                    actual_fmp_key = FMP_ALT_KEYS[fmp_key]
                
                if actual_fmp_key not in row:
                    continue
                    
                sec_val = row[sec_metric]
                fmp_val = row[actual_fmp_key]
                
                if pd.isna(sec_val) or pd.isna(fmp_val):
                    continue
                
                sec_val = float(sec_val)
                fmp_val = float(fmp_val)
                
                # 회계상 부호가 다른 경우 (절댓값으로 비교)
                if sec_metric in ["Dividends Paid", "Repurchases", "Capital Expenditure", "Investing Cash Flow", "Financing Cash Flow"]:
                    sec_val = abs(sec_val)
                    fmp_val = abs(fmp_val)
                
                diff_abs = abs(sec_val - fmp_val)
                
                if fmp_val == 0 and sec_val == 0:
                    diff_pct = 0.0
                elif fmp_val == 0:
                    diff_pct = 100.0 if sec_val != 0 else 0.0
                else:
                    diff_pct = (diff_abs / abs(fmp_val)) * 100.0
                
                # 예외 처리: 미세한 차이는 무시
                if sec_metric in ["EPS", "Diluted EPS"]:
                    if diff_abs < 0.05:
                        continue
                else:
                    if diff_abs < 1_000_000:
                        continue
                        
                # 3% 이상 차이나는 항목만 기록
                if diff_pct >= 3.0:
                    results.append({
                        "Ticker": ticker,
                        "Year": int(year),
                        "Quarter": int(quarter),
                        "Metric_Name": sec_metric,
                        "SEC_Value": sec_val,
                        "FMP_Value": fmp_val,
                        "Diff_Percent": round(diff_pct, 2)
                    })
                    
    # 4. 결과 리포트 출력 (CSV 저장)
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by=['Ticker', 'Year', 'Quarter', 'Metric_Name'])
        out_path = "validation_results_all.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\n[완료] 총 {len(results)}건의 3% 이상 오차 항목이 발견되었습니다.")
        print(f"결과 파일: {out_path}")
    else:
        print("\n[완료] 지정한 모든 지표가 3% 오차 범위 내에서 일치합니다.")

if __name__ == "__main__":
    main()
