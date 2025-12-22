"""
한국 주식 데이터 수집 스크립트
FinanceDataReader와 pykrx를 사용하여 KOSPI/KOSDAQ 데이터 수집
"""
import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any
import time
import traceback

import pandas as pd

# 상위 디렉토리 import를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from calculate_indicators import process_stock, categorize_recommendations, get_kr_fundamentals


def get_market_stocks(market: str = 'KOSPI', top_n: int = 200) -> List[Dict[str, Any]]:
    """시장별 상위 종목 리스트 가져오기"""
    print(f"[INFO] {market} 종목 리스트 조회 중...")

    stocks = []

    # 방법 1: FinanceDataReader 시도
    try:
        import FinanceDataReader as fdr

        if market == 'KOSPI':
            stocks_df = fdr.StockListing('KOSPI')
        else:
            stocks_df = fdr.StockListing('KOSDAQ')

        print(f"[DEBUG] FDR 컬럼: {list(stocks_df.columns)}")
        print(f"[DEBUG] FDR 행 수: {len(stocks_df)}")

        if stocks_df.empty:
            raise Exception("FDR 데이터가 비어있음")

        # 시가총액 기준 상위 종목 선택 (다양한 컬럼명 처리)
        marcap_cols = ['Marcap', 'MarCap', 'marcap', 'MarketCap', 'Market Cap']
        marcap_col = None
        for col in marcap_cols:
            if col in stocks_df.columns:
                marcap_col = col
                break

        if marcap_col:
            stocks_df = stocks_df.nlargest(top_n, marcap_col)
        else:
            print(f"[WARN] 시가총액 컬럼 없음, 상위 {top_n}개 선택")
            stocks_df = stocks_df.head(top_n)

        for _, row in stocks_df.iterrows():
            # 다양한 컬럼명 처리
            code = None
            for col in ['Code', 'Symbol', 'code', 'symbol', 'Ticker']:
                if col in row.index and pd.notna(row.get(col)):
                    code = str(row.get(col))
                    break

            name = None
            for col in ['Name', 'name', 'Company', 'company']:
                if col in row.index and pd.notna(row.get(col)):
                    name = row.get(col)
                    break

            sector = ''
            for col in ['Sector', 'sector', 'Industry', 'industry']:
                if col in row.index and pd.notna(row.get(col)):
                    sector = row.get(col)
                    break

            if code and name:
                code = code.zfill(6) if len(code) < 6 else code
                stocks.append({
                    'code': code,
                    'name': name,
                    'market': market,
                    'sector': sector
                })

        print(f"[INFO] {market} {len(stocks)}개 종목 조회 완료 (FDR)")
        return stocks

    except Exception as e:
        print(f"[WARN] FDR 조회 실패: {e}")
        traceback.print_exc()

    # 방법 2: pykrx 시도
    try:
        from pykrx import stock as pykrx_stock

        today = datetime.now().strftime('%Y%m%d')

        if market == 'KOSPI':
            tickers = pykrx_stock.get_market_ticker_list(today, market='KOSPI')
        else:
            tickers = pykrx_stock.get_market_ticker_list(today, market='KOSDAQ')

        print(f"[DEBUG] pykrx 종목 수: {len(tickers)}")

        for ticker in tickers[:top_n]:
            try:
                name = pykrx_stock.get_market_ticker_name(ticker)
                stocks.append({
                    'code': ticker,
                    'name': name,
                    'market': market,
                    'sector': ''
                })
            except Exception as e:
                print(f"[WARN] {ticker} 이름 조회 실패: {e}")
                continue

        print(f"[INFO] {market} {len(stocks)}개 종목 조회 완료 (pykrx)")
        return stocks

    except Exception as e:
        print(f"[ERROR] pykrx 조회 실패: {e}")
        traceback.print_exc()

    return stocks


def get_stock_data(code: str, days: int = 100) -> pd.DataFrame:
    """개별 종목의 OHLCV 데이터 가져오기"""
    try:
        import FinanceDataReader as fdr

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        df = fdr.DataReader(code, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        if df.empty:
            return pd.DataFrame()

        # 컬럼명 표준화 (대소문자 구분 없이)
        col_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'open' in col_lower:
                col_mapping[col] = 'Open'
            elif 'high' in col_lower:
                col_mapping[col] = 'High'
            elif 'low' in col_lower:
                col_mapping[col] = 'Low'
            elif 'close' in col_lower:
                col_mapping[col] = 'Close'
            elif 'volume' in col_lower:
                col_mapping[col] = 'Volume'

        if col_mapping:
            df = df.rename(columns=col_mapping)

        # 필수 컬럼 확인
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in df.columns:
                print(f"[WARN] {code}: {col} 컬럼 없음")
                return pd.DataFrame()

        return df[required]

    except Exception as e:
        print(f"[WARN] {code} FDR 데이터 조회 실패: {e}")

    # pykrx 대체 시도
    try:
        from pykrx import stock as pykrx_stock

        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')

        df = pykrx_stock.get_market_ohlcv(start_date, end_date, code)

        if df.empty:
            return pd.DataFrame()

        # 컬럼명 표준화
        col_mapping = {
            '시가': 'Open',
            '고가': 'High',
            '저가': 'Low',
            '종가': 'Close',
            '거래량': 'Volume'
        }
        df = df.rename(columns=col_mapping)

        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    except Exception as e:
        print(f"[WARN] {code} pykrx 데이터 조회 실패: {e}")
        return pd.DataFrame()


def get_market_index(market: str = 'KOSPI') -> Dict[str, Any]:
    """시장 지수 정보 가져오기"""
    try:
        import FinanceDataReader as fdr

        if market == 'KOSPI':
            index_code = 'KS11'
        else:
            index_code = 'KQ11'

        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)

        df = fdr.DataReader(index_code, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        if df.empty or len(df) < 2:
            return {}

        # Close 컬럼 찾기
        close_col = None
        for col in df.columns:
            if 'close' in col.lower():
                close_col = col
                break

        if close_col is None:
            close_col = 'Close'

        current = df[close_col].iloc[-1]
        previous = df[close_col].iloc[-2]
        change_pct = ((current - previous) / previous) * 100

        return {
            'index': round(current, 2),
            'change_pct': round(change_pct, 2)
        }

    except Exception as e:
        print(f"[WARN] {market} 지수 조회 실패: {e}")
        return {}


def determine_market_sentiment(kospi_change: float, kosdaq_change: float) -> str:
    """시장 심리 판단"""
    avg_change = (kospi_change + kosdaq_change) / 2

    if avg_change > 1.0:
        return 'bullish'
    elif avg_change < -1.0:
        return 'bearish'
    else:
        return 'neutral'


def collect_and_analyze(market: str = 'KOSPI', top_n: int = 200) -> Dict[str, Any]:
    """데이터 수집 및 분석 실행"""
    print(f"\n{'='*50}")
    print(f"[START] {market} 데이터 수집 시작")
    print(f"{'='*50}\n")

    # 1. 종목 리스트 가져오기
    stocks = get_market_stocks(market, top_n)

    if not stocks:
        print(f"[ERROR] {market} 종목이 없습니다.")
        return {}

    # 2. 각 종목 데이터 수집 및 분석
    analyzed_stocks = []
    total = len(stocks)
    failed_count = 0

    for i, stock in enumerate(stocks):
        code = stock['code']
        name = stock['name']

        print(f"[{i+1}/{total}] {name} ({code}) 분석 중...", end=' ')

        # 데이터 가져오기
        df = get_stock_data(code)

        if df.empty or len(df) < 30:
            print("데이터 부족, 건너뜀")
            failed_count += 1
            continue

        try:
            # 재무제표 가져오기 (KR: 간단 - PER, PBR, EPS, BPS)
            fundamentals = get_kr_fundamentals(code)

            # 분석 수행 (재무제표 포함)
            result = process_stock(stock, df, fundamentals=fundamentals, region='KR')
            analyzed_stocks.append(result)

            # 재무 건전성 표시
            health = result.get('fundamental_health', '')
            health_emoji = '💚' if health == 'good' else ('⚠️' if health == 'warning' else '')
            print(f"점수: {result['score']}, 등급: {result['grade']} {health_emoji}")
        except Exception as e:
            print(f"분석 실패: {e}")
            failed_count += 1

        # API 부하 방지를 위한 딜레이
        time.sleep(0.3)

    print(f"\n[INFO] 총 {len(analyzed_stocks)}개 종목 분석 완료 (실패: {failed_count}개)")

    if not analyzed_stocks:
        print("[ERROR] 분석된 종목이 없습니다!")
        return {}

    # 3. 추천 분류
    recommendations = categorize_recommendations(analyzed_stocks)

    return {
        'market': market,
        'stocks': analyzed_stocks,
        'recommendations': recommendations
    }


def save_results(data: Dict[str, Any], output_dir: str):
    """결과 저장"""
    market = data.get('market', 'KR').lower()

    # 개별 종목 데이터 저장
    stocks_file = os.path.join(output_dir, f'{market}_stocks.json')
    stocks_data = {
        'updated_at': datetime.now().isoformat(),
        'market': data['market'],
        'count': len(data['stocks']),
        'stocks': data['stocks']
    }

    with open(stocks_file, 'w', encoding='utf-8') as f:
        json.dump(stocks_data, f, ensure_ascii=False, indent=2)

    print(f"[SAVED] {stocks_file}")

    return data['recommendations']


def main():
    parser = argparse.ArgumentParser(description='한국 주식 데이터 수집')
    parser.add_argument('--market', type=str, default='all',
                        choices=['kospi', 'kosdaq', 'all'],
                        help='수집할 시장 (kospi, kosdaq, all)')
    parser.add_argument('--top', type=int, default=200,
                        help='시장별 상위 종목 수')
    parser.add_argument('--output', type=str, default='../data/kr',
                        help='출력 디렉토리')

    args = parser.parse_args()

    print(f"\n[CONFIG] market={args.market}, top={args.top}, output={args.output}")
    print(f"[CONFIG] 현재 시간: {datetime.now().isoformat()}")

    # 출력 디렉토리 생성
    output_dir = os.path.join(os.path.dirname(__file__), args.output)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[CONFIG] 출력 디렉토리: {output_dir}")

    all_recommendations = {
        'strong_buy': [],
        'buy': [],
        'hold': [],
        'sell': [],
        'strong_sell': []
    }

    markets_to_collect = ['KOSPI', 'KOSDAQ'] if args.market == 'all' else [args.market.upper()]

    # 시장 지수 정보 수집
    print("\n[INFO] 시장 지수 조회 중...")
    kospi_index = get_market_index('KOSPI')
    kosdaq_index = get_market_index('KOSDAQ')

    print(f"[INFO] KOSPI: {kospi_index}")
    print(f"[INFO] KOSDAQ: {kosdaq_index}")

    market_summary = {
        'kospi_index': kospi_index.get('index', 0),
        'kospi_change_pct': kospi_index.get('change_pct', 0),
        'kosdaq_index': kosdaq_index.get('index', 0),
        'kosdaq_change_pct': kosdaq_index.get('change_pct', 0),
        'market_sentiment': determine_market_sentiment(
            kospi_index.get('change_pct', 0),
            kosdaq_index.get('change_pct', 0)
        )
    }

    # 각 시장 데이터 수집
    success_count = 0
    for market in markets_to_collect:
        result = collect_and_analyze(market, args.top)

        if result and result.get('stocks'):
            recommendations = save_results(result, output_dir)
            success_count += 1

            # 전체 추천에 병합
            for grade in all_recommendations:
                all_recommendations[grade].extend(recommendations.get(grade, []))

    if success_count == 0:
        print("\n[ERROR] 모든 시장 데이터 수집 실패!")
        sys.exit(1)

    # 전체 추천 결과 정렬 (점수순)
    for grade in ['strong_buy', 'buy']:
        all_recommendations[grade].sort(key=lambda x: x['score'], reverse=True)
    for grade in ['hold', 'sell', 'strong_sell']:
        all_recommendations[grade].sort(key=lambda x: x['score'])

    # 전체 추천 결과 저장
    recommendations_file = os.path.join(output_dir, 'kr_recommendations.json')
    recommendations_data = {
        'updated_at': datetime.now().isoformat(),
        'region': 'KR',
        'market_summary': market_summary,
        'recommendations': all_recommendations,
        'stats': {
            'strong_buy': len(all_recommendations['strong_buy']),
            'buy': len(all_recommendations['buy']),
            'hold': len(all_recommendations['hold']),
            'sell': len(all_recommendations['sell']),
            'strong_sell': len(all_recommendations['strong_sell'])
        }
    }

    with open(recommendations_file, 'w', encoding='utf-8') as f:
        json.dump(recommendations_data, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVED] {recommendations_file}")
    print(f"\n{'='*50}")
    print("[COMPLETE] 한국 주식 데이터 수집 완료")
    print(f"적극매수: {len(all_recommendations['strong_buy'])}개")
    print(f"매수: {len(all_recommendations['buy'])}개")
    print(f"보유: {len(all_recommendations['hold'])}개")
    print(f"매도: {len(all_recommendations['sell'])}개")
    print(f"적극매도: {len(all_recommendations['strong_sell'])}개")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()
