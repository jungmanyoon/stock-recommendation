"""
미국 주식 데이터 수집 스크립트
yfinance를 사용하여 S&P500/NASDAQ100 데이터 수집
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
import yfinance as yf

# 상위 디렉토리 import를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from calculate_indicators import process_stock, categorize_recommendations


# S&P 500 상위 종목 (시가총액 기준)
SP500_TOP = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
    'V', 'XOM', 'JPM', 'WMT', 'PG', 'MA', 'CVX', 'HD', 'LLY', 'ABBV',
    'MRK', 'PEP', 'KO', 'AVGO', 'PFE', 'COST', 'TMO', 'MCD', 'CSCO', 'ACN',
    'ABT', 'DHR', 'BAC', 'CRM', 'NKE', 'CMCSA', 'NEE', 'LIN', 'ADBE', 'TXN',
    'WFC', 'PM', 'AMD', 'VZ', 'ORCL', 'RTX', 'HON', 'COP', 'UPS', 'QCOM',
    'LOW', 'MS', 'SPGI', 'T', 'ELV', 'INTU', 'CAT', 'IBM', 'GE', 'DE',
    'BA', 'AMGN', 'GS', 'ISRG', 'BLK', 'SBUX', 'AXP', 'AMAT', 'MDT', 'NOW',
    'PLD', 'GILD', 'BKNG', 'SYK', 'ADI', 'MDLZ', 'VRTX', 'ADP', 'TJX', 'MMC',
    'CVS', 'LRCX', 'C', 'REGN', 'ZTS', 'MO', 'CI', 'TMUS', 'CB', 'SO'
]

# NASDAQ 100 주요 종목
NASDAQ100_TOP = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'COST', 'ASML',
    'PEP', 'CSCO', 'ADBE', 'NFLX', 'AMD', 'CMCSA', 'TXN', 'INTC', 'QCOM', 'HON',
    'INTU', 'AMGN', 'AMAT', 'ISRG', 'BKNG', 'SBUX', 'ADI', 'MDLZ', 'VRTX', 'ADP',
    'GILD', 'REGN', 'LRCX', 'PYPL', 'MU', 'CSX', 'SNPS', 'KLAC', 'PANW', 'CDNS',
    'CHTR', 'MAR', 'ORLY', 'MNST', 'MRVL', 'ADSK', 'ABNB', 'FTNT', 'AEP', 'KDP',
    'CTAS', 'PAYX', 'DXCM', 'KHC', 'MCHP', 'EXC', 'BIIB', 'PCAR', 'AZN', 'ROST',
    'ODFL', 'LULU', 'CPRT', 'IDXX', 'WDAY', 'EA', 'FAST', 'VRSK', 'XEL', 'CTSH',
    'DLTR', 'ANSS', 'GEHC', 'FANG', 'ZS', 'BKR', 'TEAM', 'CSGP', 'DDOG', 'ILMN'
]


def get_index_stocks(index: str = 'sp500') -> List[Dict[str, Any]]:
    """지수별 종목 리스트 가져오기"""
    print(f"[INFO] {index.upper()} 종목 리스트 준비 중...")

    if index.lower() == 'sp500':
        symbols = SP500_TOP
        market = 'NYSE/NASDAQ'
    else:
        symbols = NASDAQ100_TOP
        market = 'NASDAQ'

    stocks = []
    for symbol in symbols:
        stocks.append({
            'code': symbol,
            'name': symbol,  # 이름은 나중에 업데이트
            'market': market,
            'sector': ''
        })

    print(f"[INFO] {index.upper()} {len(stocks)}개 종목 준비 완료")
    return stocks


def get_stock_info(symbol: str) -> Dict[str, Any]:
    """종목 기본 정보 가져오기"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        if info and isinstance(info, dict):
            return {
                'name': info.get('shortName', info.get('longName', symbol)),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', '')
            }
    except Exception as e:
        print(f"[DEBUG] {symbol} 정보 조회 실패: {e}")

    return {'name': symbol, 'sector': '', 'industry': ''}


def get_stock_data(symbol: str, days: int = 100) -> pd.DataFrame:
    """개별 종목의 OHLCV 데이터 가져오기"""
    try:
        ticker = yf.Ticker(symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        df = ticker.history(start=start_date.strftime('%Y-%m-%d'),
                           end=end_date.strftime('%Y-%m-%d'))

        if df.empty:
            # 기간을 늘려서 재시도
            df = ticker.history(period='3mo')

        if df.empty:
            print(f"[DEBUG] {symbol}: 데이터 없음")
            return pd.DataFrame()

        # 컬럼명 표준화 (대소문자 처리)
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
        missing = [col for col in required if col not in df.columns]
        if missing:
            print(f"[DEBUG] {symbol}: 누락 컬럼 {missing}")
            return pd.DataFrame()

        return df[required]

    except Exception as e:
        print(f"[WARN] {symbol} 데이터 조회 실패: {e}")
        return pd.DataFrame()


def get_market_index() -> Dict[str, Any]:
    """미국 시장 지수 정보 가져오기"""
    indices = {
        'sp500': '^GSPC',
        'nasdaq': '^IXIC',
        'dow': '^DJI'
    }

    result = {}

    for name, symbol in indices.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='5d')

            if hist.empty or len(hist) < 2:
                print(f"[WARN] {name} 지수 데이터 부족")
                continue

            # Close 컬럼 찾기
            close_col = None
            for col in hist.columns:
                if 'close' in col.lower():
                    close_col = col
                    break
            if close_col is None:
                close_col = 'Close'

            current = hist[close_col].iloc[-1]
            previous = hist[close_col].iloc[-2]
            change_pct = ((current - previous) / previous) * 100

            result[f'{name}_index'] = round(current, 2)
            result[f'{name}_change_pct'] = round(change_pct, 2)

            print(f"[INFO] {name}: {round(current, 2)} ({round(change_pct, 2)}%)")

        except Exception as e:
            print(f"[WARN] {name} 지수 조회 실패: {e}")

    return result


def determine_market_sentiment(index_data: Dict[str, Any]) -> str:
    """시장 심리 판단"""
    changes = []
    for key in ['sp500_change_pct', 'nasdaq_change_pct', 'dow_change_pct']:
        if key in index_data:
            changes.append(index_data[key])

    if not changes:
        return 'neutral'

    avg_change = sum(changes) / len(changes)

    if avg_change > 1.0:
        return 'bullish'
    elif avg_change < -1.0:
        return 'bearish'
    else:
        return 'neutral'


def collect_and_analyze(index: str = 'sp500') -> Dict[str, Any]:
    """데이터 수집 및 분석 실행"""
    print(f"\n{'='*50}")
    print(f"[START] {index.upper()} 데이터 수집 시작")
    print(f"{'='*50}\n")

    # 1. 종목 리스트 가져오기
    stocks = get_index_stocks(index)

    if not stocks:
        print(f"[ERROR] {index} 종목이 없습니다.")
        return {}

    # 2. 각 종목 데이터 수집 및 분석
    analyzed_stocks = []
    total = len(stocks)
    failed_count = 0

    for i, stock in enumerate(stocks):
        symbol = stock['code']

        print(f"[{i+1}/{total}] {symbol} 분석 중...", end=' ')

        # 종목 정보 가져오기
        info = get_stock_info(symbol)
        stock['name'] = info['name']
        stock['sector'] = info['sector']

        # 데이터 가져오기
        df = get_stock_data(symbol)

        if df.empty or len(df) < 30:
            print("데이터 부족, 건너뜀")
            failed_count += 1
            continue

        try:
            # 분석 수행
            result = process_stock(stock, df)
            analyzed_stocks.append(result)
            print(f"점수: {result['score']}, 등급: {result['grade']}")
        except Exception as e:
            print(f"분석 실패: {e}")
            traceback.print_exc()
            failed_count += 1

        # API 부하 방지를 위한 딜레이
        time.sleep(0.5)

    print(f"\n[INFO] 총 {len(analyzed_stocks)}개 종목 분석 완료 (실패: {failed_count}개)")

    if not analyzed_stocks:
        print("[ERROR] 분석된 종목이 없습니다!")
        return {}

    # 3. 추천 분류
    recommendations = categorize_recommendations(analyzed_stocks)

    return {
        'index': index.upper(),
        'stocks': analyzed_stocks,
        'recommendations': recommendations
    }


def save_results(data: Dict[str, Any], output_dir: str):
    """결과 저장"""
    index_name = data.get('index', 'US').lower()

    # 개별 종목 데이터 저장
    stocks_file = os.path.join(output_dir, f'{index_name}_stocks.json')
    stocks_data = {
        'updated_at': datetime.now().isoformat(),
        'index': data['index'],
        'count': len(data['stocks']),
        'stocks': data['stocks']
    }

    with open(stocks_file, 'w', encoding='utf-8') as f:
        json.dump(stocks_data, f, ensure_ascii=False, indent=2)

    print(f"[SAVED] {stocks_file}")

    return data['recommendations']


def main():
    parser = argparse.ArgumentParser(description='미국 주식 데이터 수집')
    parser.add_argument('--index', type=str, default='all',
                        choices=['sp500', 'nasdaq100', 'all'],
                        help='수집할 지수 (sp500, nasdaq100, all)')
    parser.add_argument('--output', type=str, default='../data/us',
                        help='출력 디렉토리')

    args = parser.parse_args()

    print(f"\n[CONFIG] index={args.index}, output={args.output}")
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

    indices_to_collect = ['sp500', 'nasdaq100'] if args.index == 'all' else [args.index]

    # 시장 지수 정보 수집
    print("\n[INFO] 시장 지수 조회 중...")
    index_data = get_market_index()

    market_summary = {
        **index_data,
        'market_sentiment': determine_market_sentiment(index_data)
    }

    print(f"[INFO] 시장 심리: {market_summary.get('market_sentiment', 'unknown')}")

    # 각 지수 데이터 수집
    success_count = 0
    for index in indices_to_collect:
        result = collect_and_analyze(index)

        if result and result.get('stocks'):
            recommendations = save_results(result, output_dir)
            success_count += 1

            # 전체 추천에 병합 (중복 제거)
            existing_codes = set()
            for grade in all_recommendations:
                existing_codes.update(s['code'] for s in all_recommendations[grade])

            for grade in all_recommendations:
                for stock in recommendations.get(grade, []):
                    if stock['code'] not in existing_codes:
                        all_recommendations[grade].append(stock)
                        existing_codes.add(stock['code'])

    if success_count == 0:
        print("\n[ERROR] 모든 지수 데이터 수집 실패!")
        sys.exit(1)

    # 전체 추천 결과 정렬 (점수순)
    for grade in ['strong_buy', 'buy']:
        all_recommendations[grade].sort(key=lambda x: x['score'], reverse=True)
    for grade in ['hold', 'sell', 'strong_sell']:
        all_recommendations[grade].sort(key=lambda x: x['score'])

    # 전체 추천 결과 저장
    recommendations_file = os.path.join(output_dir, 'us_recommendations.json')
    recommendations_data = {
        'updated_at': datetime.now().isoformat(),
        'region': 'US',
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
    print("[COMPLETE] 미국 주식 데이터 수집 완료")
    print(f"적극매수: {len(all_recommendations['strong_buy'])}개")
    print(f"매수: {len(all_recommendations['buy'])}개")
    print(f"보유: {len(all_recommendations['hold'])}개")
    print(f"매도: {len(all_recommendations['sell'])}개")
    print(f"적극매도: {len(all_recommendations['strong_sell'])}개")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()
