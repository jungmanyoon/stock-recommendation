"""
기술적 지표 계산 및 추천 점수 산출 모듈
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import json


def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """RSI (Relative Strength Index) 계산"""
    if len(prices) < period + 1:
        return 50.0  # 데이터 부족시 중립값

    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    result = rsi.iloc[-1]
    return round(result, 2) if not pd.isna(result) else 50.0


def calculate_macd(prices: pd.Series) -> Dict[str, float]:
    """MACD 계산 (12, 26, 9)"""
    if len(prices) < 35:
        return {"macd_line": 0, "signal_line": 0, "histogram": 0}

    ema_12 = prices.ewm(span=12, adjust=False).mean()
    ema_26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line

    return {
        "macd_line": round(macd_line.iloc[-1], 2) if not pd.isna(macd_line.iloc[-1]) else 0,
        "signal_line": round(signal_line.iloc[-1], 2) if not pd.isna(signal_line.iloc[-1]) else 0,
        "histogram": round(histogram.iloc[-1], 2) if not pd.isna(histogram.iloc[-1]) else 0
    }


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
    """볼린저 밴드 계산"""
    if len(prices) < period:
        current = prices.iloc[-1] if len(prices) > 0 else 0
        return {"upper": current, "middle": current, "lower": current, "bandwidth": 0}

    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)

    middle_val = sma.iloc[-1]
    bandwidth = ((upper.iloc[-1] - lower.iloc[-1]) / middle_val) if middle_val > 0 else 0

    return {
        "upper": round(upper.iloc[-1], 2) if not pd.isna(upper.iloc[-1]) else 0,
        "middle": round(middle_val, 2) if not pd.isna(middle_val) else 0,
        "lower": round(lower.iloc[-1], 2) if not pd.isna(lower.iloc[-1]) else 0,
        "bandwidth": round(bandwidth, 4) if not pd.isna(bandwidth) else 0
    }


def calculate_sma(prices: pd.Series, periods: List[int] = [20, 50, 200]) -> Dict[str, float]:
    """이동평균선 계산"""
    result = {}
    for period in periods:
        if len(prices) >= period:
            sma = prices.rolling(window=period).mean().iloc[-1]
            result[f"sma_{period}"] = round(sma, 2) if not pd.isna(sma) else 0
        else:
            result[f"sma_{period}"] = 0
    return result


def calculate_volume_ratio(volumes: pd.Series, period: int = 20) -> Dict[str, Any]:
    """거래량 비율 계산"""
    if len(volumes) < period + 1:
        return {"current": 0, "avg": 0, "ratio": 1.0}

    current = volumes.iloc[-1]
    avg = volumes.iloc[-period-1:-1].mean()
    ratio = current / avg if avg > 0 else 1.0

    return {
        "current": int(current),
        "avg": int(avg),
        "ratio": round(ratio, 2)
    }


def calculate_all_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """모든 기술적 지표 계산"""
    prices = df['Close']
    volumes = df['Volume']

    indicators = {
        "rsi_14": calculate_rsi(prices, 14),
        "macd": calculate_macd(prices),
        "bollinger": calculate_bollinger_bands(prices),
        **calculate_sma(prices),
        "volume": calculate_volume_ratio(volumes)
    }

    return indicators


def generate_signals(indicators: Dict[str, Any], current_price: float) -> Dict[str, str]:
    """기술적 지표 기반 신호 생성"""
    signals = {}

    # RSI 신호
    rsi = indicators.get('rsi_14', 50)
    if rsi < 30:
        signals['rsi'] = 'oversold'
    elif rsi < 40:
        signals['rsi'] = 'oversold_recovery'
    elif rsi > 70:
        signals['rsi'] = 'overbought'
    elif rsi > 60:
        signals['rsi'] = 'approaching_overbought'
    else:
        signals['rsi'] = 'neutral'

    # MACD 신호
    macd = indicators.get('macd', {})
    histogram = macd.get('histogram', 0)
    macd_line = macd.get('macd_line', 0)
    signal_line = macd.get('signal_line', 0)

    if histogram > 0 and macd_line > signal_line:
        signals['macd'] = 'bullish_crossover'
    elif histogram < 0 and macd_line < signal_line:
        signals['macd'] = 'bearish_crossover'
    else:
        signals['macd'] = 'neutral'

    # 볼린저 밴드 신호
    bb = indicators.get('bollinger', {})
    lower = bb.get('lower', 0)
    upper = bb.get('upper', 0)

    if lower > 0 and current_price <= lower:
        signals['bollinger'] = 'lower_band_touch'
    elif lower > 0 and current_price <= lower * 1.02:
        signals['bollinger'] = 'lower_band_bounce'
    elif upper > 0 and current_price >= upper:
        signals['bollinger'] = 'upper_band_touch'
    elif upper > 0 and current_price >= upper * 0.98:
        signals['bollinger'] = 'upper_band_approach'
    else:
        signals['bollinger'] = 'within_bands'

    # 거래량 신호
    volume = indicators.get('volume', {})
    ratio = volume.get('ratio', 1.0)

    if ratio > 2.0:
        signals['volume'] = 'volume_surge'
    elif ratio > 1.5:
        signals['volume'] = 'above_average'
    elif ratio < 0.5:
        signals['volume'] = 'below_average'
    else:
        signals['volume'] = 'normal'

    # 이동평균선 신호
    sma_20 = indicators.get('sma_20', 0)
    sma_50 = indicators.get('sma_50', 0)
    sma_200 = indicators.get('sma_200', 0)

    if sma_20 > 0 and sma_50 > 0:
        if current_price > sma_20 > sma_50:
            signals['trend'] = 'strong_uptrend'
        elif current_price > sma_20:
            signals['trend'] = 'uptrend'
        elif current_price < sma_20 < sma_50:
            signals['trend'] = 'strong_downtrend'
        elif current_price < sma_20:
            signals['trend'] = 'downtrend'
        else:
            signals['trend'] = 'sideways'
    else:
        signals['trend'] = 'unknown'

    return signals


def calculate_recommendation_score(signals: Dict[str, str]) -> int:
    """추천 점수 계산 (0-100)"""
    score = 50  # 기본 점수

    # RSI 점수
    rsi_signal = signals.get('rsi', 'neutral')
    rsi_scores = {
        'oversold': 15,
        'oversold_recovery': 10,
        'neutral': 0,
        'approaching_overbought': -5,
        'overbought': -15
    }
    score += rsi_scores.get(rsi_signal, 0)

    # MACD 점수
    macd_signal = signals.get('macd', 'neutral')
    macd_scores = {
        'bullish_crossover': 15,
        'neutral': 0,
        'bearish_crossover': -15
    }
    score += macd_scores.get(macd_signal, 0)

    # 볼린저 밴드 점수
    bb_signal = signals.get('bollinger', 'within_bands')
    bb_scores = {
        'lower_band_touch': 15,
        'lower_band_bounce': 10,
        'within_bands': 0,
        'upper_band_approach': -5,
        'upper_band_touch': -15
    }
    score += bb_scores.get(bb_signal, 0)

    # 거래량 점수
    volume_signal = signals.get('volume', 'normal')
    volume_scores = {
        'volume_surge': 10,
        'above_average': 5,
        'normal': 0,
        'below_average': -5
    }
    score += volume_scores.get(volume_signal, 0)

    # 추세 점수
    trend_signal = signals.get('trend', 'sideways')
    trend_scores = {
        'strong_uptrend': 10,
        'uptrend': 5,
        'sideways': 0,
        'downtrend': -5,
        'strong_downtrend': -10,
        'unknown': 0
    }
    score += trend_scores.get(trend_signal, 0)

    # 점수 범위 제한 (0-100)
    return max(0, min(100, score))


def get_recommendation_grade(score: int) -> str:
    """점수에 따른 추천 등급 반환"""
    if score >= 80:
        return 'strong_buy'
    elif score >= 65:
        return 'buy'
    elif score >= 45:
        return 'hold'
    elif score >= 30:
        return 'sell'
    else:
        return 'strong_sell'


def generate_summary(signals: Dict[str, str]) -> str:
    """신호 기반 요약 생성"""
    summaries = []

    signal_descriptions = {
        'rsi': {
            'oversold': 'RSI 과매도',
            'oversold_recovery': 'RSI 과매도 회복',
            'overbought': 'RSI 과매수',
            'approaching_overbought': 'RSI 과매수 접근'
        },
        'macd': {
            'bullish_crossover': 'MACD 골든크로스',
            'bearish_crossover': 'MACD 데드크로스'
        },
        'bollinger': {
            'lower_band_touch': '볼린저 하단 터치',
            'lower_band_bounce': '볼린저 하단 반등',
            'upper_band_touch': '볼린저 상단 터치'
        },
        'volume': {
            'volume_surge': '거래량 급증',
            'above_average': '거래량 증가'
        },
        'trend': {
            'strong_uptrend': '강한 상승추세',
            'uptrend': '상승추세',
            'strong_downtrend': '강한 하락추세',
            'downtrend': '하락추세'
        }
    }

    for category, descriptions in signal_descriptions.items():
        signal = signals.get(category, 'neutral')
        if signal in descriptions:
            summaries.append(descriptions[signal])

    return ' + '.join(summaries) if summaries else '특별한 신호 없음'


def process_stock(stock_data: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """개별 종목 처리 및 추천 정보 생성"""
    indicators = calculate_all_indicators(df)
    current_price = df['Close'].iloc[-1]
    signals = generate_signals(indicators, current_price)
    score = calculate_recommendation_score(signals)
    grade = get_recommendation_grade(score)
    summary = generate_summary(signals)

    return {
        'code': stock_data['code'],
        'name': stock_data['name'],
        'market': stock_data.get('market', ''),
        'sector': stock_data.get('sector', ''),
        'price': {
            'current': round(current_price, 2),
            'open': round(df['Open'].iloc[-1], 2),
            'high': round(df['High'].iloc[-1], 2),
            'low': round(df['Low'].iloc[-1], 2),
            'volume': int(df['Volume'].iloc[-1]),
            'change_pct': round(((current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100, 2) if len(df) > 1 else 0
        },
        'indicators': indicators,
        'signals': signals,
        'score': score,
        'grade': grade,
        'summary': summary
    }


def categorize_recommendations(stocks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """추천 등급별로 종목 분류"""
    categories = {
        'strong_buy': [],
        'buy': [],
        'hold': [],
        'sell': [],
        'strong_sell': []
    }

    for stock in stocks:
        grade = stock.get('grade', 'hold')
        # 간소화된 추천 정보만 포함
        recommendation = {
            'code': stock['code'],
            'name': stock['name'],
            'market': stock['market'],
            'score': stock['score'],
            'signals': stock['signals'],
            'summary': stock['summary'],
            'price': stock['price']['current'],
            'change_pct': stock['price']['change_pct']
        }
        categories[grade].append(recommendation)

    # 각 카테고리 내에서 점수순 정렬
    for grade in categories:
        if grade in ['strong_buy', 'buy']:
            categories[grade].sort(key=lambda x: x['score'], reverse=True)
        else:
            categories[grade].sort(key=lambda x: x['score'])

    return categories
