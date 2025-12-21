"""
기술적 지표 계산 및 추천 점수 산출 모듈
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import json


def convert_to_native(obj):
    """numpy 타입을 Python 네이티브 타입으로 변환"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(i) for i in obj]
    return obj


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
    """
    개선된 추천 점수 계산 (0-100)

    개선사항:
    1. 추세 맥락 반영: 하락추세에서 RSI 과매도는 위험 신호
    2. 신호 상관관계 보정: RSI + 볼린저 중복 신호 감점
    3. 거래량은 승수로 적용: 신호 확신도 조절
    4. 더 보수적인 점수 체계
    """

    # === 1. 추세 점수 (기본 컨텍스트) ===
    trend_signal = signals.get('trend', 'sideways')
    trend_scores = {
        'strong_uptrend': 12,
        'uptrend': 6,
        'sideways': 0,
        'downtrend': -6,
        'strong_downtrend': -12,
        'unknown': 0
    }
    trend_score = trend_scores.get(trend_signal, 0)

    # === 2. RSI 점수 (추세 맥락 반영) ===
    rsi_signal = signals.get('rsi', 'neutral')

    # 상승추세에서는 과매도가 좋은 진입점
    if trend_signal in ['uptrend', 'strong_uptrend']:
        rsi_scores = {
            'oversold': 15,           # 상승추세 + 과매도 = 최고 매수
            'oversold_recovery': 10,
            'neutral': 0,
            'approaching_overbought': -3,  # 상승추세에서 과매수는 덜 나쁨
            'overbought': -8
        }
    # 하락추세에서는 과매도가 "떨어지는 칼날"
    elif trend_signal in ['downtrend', 'strong_downtrend']:
        rsi_scores = {
            'oversold': 5,            # 하락추세 + 과매도 = 주의
            'oversold_recovery': 8,   # 회복 신호는 긍정적
            'neutral': 0,
            'approaching_overbought': -8,
            'overbought': -15         # 하락추세 + 과매수 = 최악
        }
    else:  # 횡보
        rsi_scores = {
            'oversold': 12,
            'oversold_recovery': 8,
            'neutral': 0,
            'approaching_overbought': -5,
            'overbought': -12
        }
    rsi_score = rsi_scores.get(rsi_signal, 0)

    # === 3. MACD 점수 ===
    macd_signal = signals.get('macd', 'neutral')
    macd_scores = {
        'bullish_crossover': 12,
        'neutral': 0,
        'bearish_crossover': -12
    }
    macd_score = macd_scores.get(macd_signal, 0)

    # === 4. 볼린저 밴드 점수 (RSI와 중복 시 감점) ===
    bb_signal = signals.get('bollinger', 'within_bands')

    # RSI 과매도 + 볼린저 하단 = 중복 신호 (같은 상황)
    if bb_signal in ['lower_band_touch', 'lower_band_bounce'] and rsi_signal in ['oversold', 'oversold_recovery']:
        bb_scores = {
            'lower_band_touch': 5,    # 중복 → 낮은 추가 점수
            'lower_band_bounce': 3,
        }
    # RSI 과매수 + 볼린저 상단 = 중복 신호
    elif bb_signal in ['upper_band_touch', 'upper_band_approach'] and rsi_signal in ['overbought', 'approaching_overbought']:
        bb_scores = {
            'upper_band_touch': -5,
            'upper_band_approach': -3,
        }
    else:
        bb_scores = {
            'lower_band_touch': 12,
            'lower_band_bounce': 8,
            'within_bands': 0,
            'upper_band_approach': -5,
            'upper_band_touch': -12
        }
    bb_score = bb_scores.get(bb_signal, 0)

    # === 5. 기본 점수 합산 ===
    raw_delta = trend_score + rsi_score + macd_score + bb_score

    # === 6. 거래량 승수 적용 (확신도 조절) ===
    volume_signal = signals.get('volume', 'normal')
    volume_multiplier = {
        'volume_surge': 1.20,      # 거래량 급증 → 신호 확신도 20% 증가
        'above_average': 1.10,     # 평균 이상 → 10% 증가
        'normal': 1.0,
        'below_average': 0.85      # 거래량 부족 → 신호 신뢰도 15% 감소
    }.get(volume_signal, 1.0)

    adjusted_delta = int(raw_delta * volume_multiplier)

    # === 7. 최종 점수 ===
    final_score = 50 + adjusted_delta

    return max(0, min(100, final_score))


def get_recommendation_grade(score: int) -> str:
    """
    점수에 따른 추천 등급 반환 (보수적 기준)

    - strong_buy: 75점 이상 (기존 80 → 75)
    - buy: 60~74점 (기존 65~79 → 60~74)
    - hold: 40~59점 (기존 45~64 → 40~59, 중립 범위 확대)
    - sell: 25~39점 (기존 30~44 → 25~39)
    - strong_sell: 25점 미만 (기존 30 → 25)
    """
    if score >= 75:
        return 'strong_buy'
    elif score >= 60:
        return 'buy'
    elif score >= 40:
        return 'hold'
    elif score >= 25:
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
    current_price = float(df['Close'].iloc[-1])
    signals = generate_signals(indicators, current_price)
    score = calculate_recommendation_score(signals)
    grade = get_recommendation_grade(score)
    summary = generate_summary(signals)

    result = {
        'code': stock_data['code'],
        'name': stock_data['name'],
        'market': stock_data.get('market', ''),
        'sector': stock_data.get('sector', ''),
        'price': {
            'current': round(float(current_price), 2),
            'open': round(float(df['Open'].iloc[-1]), 2),
            'high': round(float(df['High'].iloc[-1]), 2),
            'low': round(float(df['Low'].iloc[-1]), 2),
            'volume': int(df['Volume'].iloc[-1]),
            'change_pct': round(float((current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100, 2) if len(df) > 1 else 0.0
        },
        'indicators': convert_to_native(indicators),
        'signals': signals,
        'score': int(score),
        'grade': grade,
        'summary': summary
    }

    return convert_to_native(result)


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
