import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# src 디렉토리를 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from market_data.db_reader import load_financials_from_db

# 1. 지표 매핑(Mapping) - DuckDB SEC vs YFinance
# YFinance는 quarterly_financials, quarterly_balance_sheet에서 가져옴.
METRIC_MAPPING = {
    # DuckDB SEC 열 이름 : yfinance index 이름 (우선순위 리스트)
    "Revenue": ["Total Revenue", "Operating Revenue"],
    "Operating Income": ["Operating Income"],
    "Net Income": ["Net Income", "Net Income Common Stockholders"],
    "Total Assets": ["Total Assets"],
    "Total Liabilities": ["Total Liabilities Net Minority Interest", "Total Liabilities"]
}

DEFAULT_TICKERS = ['AAPL', 'MSFT', 'TSLA', 'JPM', 'JNJ']
START_YEAR = 2016
END_YEAR = 2026

def get_yfinance_data(ticker: str):
    """yfinance에서 분기별 핵심 5개 지표를 가져와 DataFrame으로 반환합니다."""
    t = yf.Ticker(ticker)
    
    # yfinance 데이터 가져오기
    try:
        q_fin = t.quarterly_financials
        q_bal = t.quarterly_balance_sheet
    except Exception as e:
        print(f"    [{ticker}] YFinance API 호출 실패: {e}")
        return pd.DataFrame()
        
    if q_fin.empty and q_bal.empty:
        return pd.DataFrame()
        
    # 두 DataFrame을 합침 (인덱스는 지표명, 컬럼은 Timestamp)
    # 중복되는 행(지표명)이 있을 수 있으므로 drop_duplicates 수행
    combined = pd.concat([q_fin, q_bal], axis=0)
    combined = combined[~combined.index.duplicated(keep='first')]
    
    # Timestamp(열)을 행으로 변경
    df = combined.T
    df.index.name = 'Date'
    df = df.reset_index()
    
    # Date를 기반으로 year, quarter 추출
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['quarter'] = df['Date'].dt.quarter
    
    # 중복된 분기가 있을 수 있으므로 최신 순으로 정렬 후 첫 번째 유지
    df = df.sort_values(by='Date', ascending=False).drop_duplicates(subset=['year', 'quarter'])
    
    return df
    
def main():
    parser = argparse.ArgumentParser(description="SEC vs YFinance 핵심 5대 지표 검증")
    parser.add_argument("--start-year", type=int, default=START_YEAR, help="검증 시작 연도")
    parser.add_argument("--end-year", type=int, default=END_YEAR, help="검증 종료 연도")
    parser.add_argument("--tickers", type=str, nargs='+', default=DEFAULT_TICKERS, help="검증할 종목들 (띄어쓰기로 구분)")
    args = parser.parseargs() if hasattr(parser, 'parseargs') else parser.parse_args()

    print(f"검증 시작...")
    print(f"대상 종목: {args.tickers}")
    print(f"기간: {args.start_year} ~ {args.end_year}")
    
    results = []
    
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
            
        # term에서 year, quarter 강제 추출
        duck_df['year'] = pd.to_numeric(duck_df['term'].astype(str).str.slice(0, 4), errors='coerce').fillna(0).astype(int)
        duck_df['quarter'] = pd.to_numeric(duck_df['term'].astype(str).str.slice(-1), errors='coerce').fillna(0).astype(int)
        
        duck_df = duck_df[(duck_df['year'] >= args.start_year) & (duck_df['year'] <= args.end_year)]
        
        if duck_df.empty:
            print(f"  [DuckDB] {ticker} 지정된 기간에 해당하는 SEC 데이터가 없습니다.")
            continue
            
        # YFinance 데이터 가져오기
        yf_df = get_yfinance_data(ticker)
        if yf_df.empty:
            print(f"  [YFinance] {ticker} 데이터를 가져오지 못했습니다. (Yahoo Finance 제공 범위 제한일 수 있음)")
            continue
            
        yf_df = yf_df[(yf_df['year'] >= args.start_year) & (yf_df['year'] <= args.end_year)]
        
        # 병합
        merged = pd.merge(duck_df, yf_df, on=['year', 'quarter'], how='inner')
        
        if merged.empty:
            print(f"  {ticker} DuckDB와 YFinance 간에 일치하는 분기 데이터가 없습니다.")
            continue
            
        print(f"  {ticker} 일치하는 분기 {len(merged)}건에 대해 지표 비교 시작...")
        
        for _, row in merged.iterrows():
            year = row['year']
            quarter = row['quarter']
            
            for sec_metric, yf_candidates in METRIC_MAPPING.items():
                if sec_metric not in row:
                    continue
                sec_val = row[sec_metric]
                
                if pd.isna(sec_val):
                    continue
                    
                # YFinance 컬럼 찾기 (우선순위에 따라)
                yf_val = np.nan
                actual_yf_key = ""
                for cand in yf_candidates:
                    if cand in row and pd.notna(row[cand]):
                        yf_val = row[cand]
                        actual_yf_key = cand
                        break
                        
                if pd.isna(yf_val):
                    continue
                    
                sec_val = float(sec_val)
                yf_val = float(yf_val)
                
                diff_abs = abs(sec_val - yf_val)
                
                if yf_val == 0 and sec_val == 0:
                    diff_pct = 0.0
                elif yf_val == 0:
                    diff_pct = 100.0 if sec_val != 0 else 0.0
                else:
                    diff_pct = (diff_abs / abs(yf_val)) * 100.0
                
                # 금액 지표는 100만 달러(1M) 미만 차이 무시 (스케일/반올림 차이)
                if diff_abs < 1_000_000:
                    continue
                        
                # 3% 이상 차이나는 항목만 기록
                if diff_pct >= 3.0:
                    results.append({
                        "Ticker": ticker,
                        "Year": int(year),
                        "Quarter": int(quarter),
                        "Metric_Name": sec_metric,
                        "YF_Key_Used": actual_yf_key,
                        "SEC_Value": sec_val,
                        "YF_Value": yf_val,
                        "Diff_Percent": round(diff_pct, 2)
                    })
                    
    # 결과 리포트 출력 (CSV 저장)
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by=['Ticker', 'Year', 'Quarter', 'Metric_Name'])
        out_path = "validation_results_yfinance.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\n[완료] 총 {len(results)}건의 3% 이상 오차 항목이 발견되었습니다.")
        print(f"결과 파일: {out_path}")
    else:
        print("\n[완료] 지정한 5대 핵심 지표가 3% 오차 범위 내에서 모두 일치합니다.")

if __name__ == "__main__":
    main()
