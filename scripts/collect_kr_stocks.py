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

import pandas as pd
import FinanceDataReader as fdr
from pykrx import stock as pykrx_stock

# 상위 디렉토리 import를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from calculate_indicators import process_stock, categorize_recommendations


def get_market_stocks(market: str = 'KOSPI', top_n: int = 200) -> List[Dict[str, Any]]:
    """시장별 상위 종목 리스트 가져오기"""
    print(f"[INFO] {market} 종목 리스트 조회 중...")

    try:
        # FinanceDataReader로 종목 리스트 가져오기
        if market == 'KOSPI':
            stocks_df = fdr.StockListing('KOSPI')
        else:
            stocks_df = fdr.StockListing('KOSDAQ')

        # 시가총액 기준 상위 종목 선택 (시가총액 컬럼이 있는 경우)
        if 'Marcap' in stocks_df.columns:
            stocks_df = stocks_df.nlargest(top_n, 'Marcap')
        else:
            stocks_df = stocks_df.head(top_n)

        stocks = []
        for _, row in stocks_df.iterrows():
            code = str(row.get('Code', row.get('Symbol', ''))).zfill(6)
            name = row.get('Name', '')
            sector = row.get('Sector', row.get('Industry', ''))

            if code and name:
                stocks.append({
                    'code': code,
                    'name': name,
                    'market': market,
                    'sector': sector
                })

        print(f"[INFO] {market} {len(stocks)}개 종목 조회 완료")
        return stocks

    except Exception as e:
        print(f"[ERROR] 종목 리스트 조회 실패: {e}")
        return []


def get_stock_data(code: str, days: int = 100) -> pd.DataFrame:
    """개별 종목의 OHLCV 데이터 가져오기"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        df = fdr.DataReader(code, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        if df.empty:
            return pd.DataFrame()

        # 컬럼명 표준화
        df = df.rename(columns={
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })

        return df

    except Exception as e:
        print(f"[WARN] {code} 데이터 조회 실패: {e}")
        return pd.DataFrame()


def get_market_index(market: str = 'KOSPI') -> Dict[str, Any]:
    """시장 지수 정보 가져오기"""
    try:
        if market == 'KOSPI':
            index_code = 'KS11'  # KOSPI 지수
        else:
            index_code = 'KQ11'  # KOSDAQ 지수

        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)

        df = fdr.DataReader(index_code, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        if df.empty or len(df) < 2:
            return {}

        current = df['Close'].iloc[-1]
        previous = df['Close'].iloc[-2]
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

    for i, stock in enumerate(stocks):
        code = stock['code']
        name = stock['name']

        print(f"[{i+1}/{total}] {name} ({code}) 분석 중...", end=' ')

        # 데이터 가져오기
        df = get_stock_data(code)

        if df.empty or len(df) < 30:
            print("데이터 부족, 건너뜀")
            continue

        try:
            # 분석 수행
            result = process_stock(stock, df)
            analyzed_stocks.append(result)
            print(f"점수: {result['score']}, 등급: {result['grade']}")
        except Exception as e:
            print(f"분석 실패: {e}")

        # API 부하 방지를 위한 딜레이
        time.sleep(0.3)

    print(f"\n[INFO] 총 {len(analyzed_stocks)}개 종목 분석 완료")

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

    # 출력 디렉토리 생성
    output_dir = os.path.join(os.path.dirname(__file__), args.output)
    os.makedirs(output_dir, exist_ok=True)

    all_recommendations = {
        'strong_buy': [],
        'buy': [],
        'hold': [],
        'sell': [],
        'strong_sell': []
    }

    markets_to_collect = ['KOSPI', 'KOSDAQ'] if args.market == 'all' else [args.market.upper()]

    # 시장 지수 정보 수집
    kospi_index = get_market_index('KOSPI')
    kosdaq_index = get_market_index('KOSDAQ')

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
    for market in markets_to_collect:
        result = collect_and_analyze(market, args.top)

        if result:
            recommendations = save_results(result, output_dir)

            # 전체 추천에 병합
            for grade in all_recommendations:
                all_recommendations[grade].extend(recommendations.get(grade, []))

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
