"""
기술적 지표 계산 및 추천 점수 산출 모듈
하이브리드 재무제표 분석 포함 (US: 상세, KR: 간단)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
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


# ============================================================
# 하이브리드 재무제표 분석 (US: 상세, KR: 간단)
# ============================================================

def get_us_fundamentals(ticker_obj) -> Dict[str, Optional[float]]:
    """
    미국 주식 상세 재무제표 분석 (yfinance ticker 객체 사용)

    Returns:
        PER, PBR, ROE, EPS, 매출성장률, 이익률, 부채비율 등
    """
    try:
        info = ticker_obj.info
        if not info or not isinstance(info, dict):
            return _empty_us_fundamentals()

        return {
            'per': _safe_round(info.get('trailingPE')),
            'forward_per': _safe_round(info.get('forwardPE')),
            'pbr': _safe_round(info.get('priceToBook')),
            'roe': _safe_round(info.get('returnOnEquity'), 4),
            'roa': _safe_round(info.get('returnOnAssets'), 4),
            'eps': _safe_round(info.get('trailingEps')),
            'forward_eps': _safe_round(info.get('forwardEps')),
            'revenue_growth': _safe_round(info.get('revenueGrowth'), 4),
            'earnings_growth': _safe_round(info.get('earningsGrowth'), 4),
            'profit_margin': _safe_round(info.get('profitMargins'), 4),
            'operating_margin': _safe_round(info.get('operatingMargins'), 4),
            'debt_to_equity': _safe_round(info.get('debtToEquity')),
            'current_ratio': _safe_round(info.get('currentRatio')),
            'dividend_yield': _safe_round(info.get('dividendYield'), 4),
            'peg_ratio': _safe_round(info.get('pegRatio')),
            'beta': _safe_round(info.get('beta')),
            'market_cap': info.get('marketCap'),
            '52w_high': _safe_round(info.get('fiftyTwoWeekHigh')),
            '52w_low': _safe_round(info.get('fiftyTwoWeekLow')),
        }
    except Exception as e:
        print(f"[WARN] US 재무제표 조회 실패: {e}")
        return _empty_us_fundamentals()


def get_kr_fundamentals(code: str) -> Dict[str, Optional[float]]:
    """
    한국 주식 간단 재무제표 분석 (pykrx 사용)

    Returns:
        PER, PBR, EPS, BPS, DIV (배당수익률)
    """
    try:
        from pykrx import stock as pykrx_stock
        from datetime import datetime, timedelta

        # 최근 영업일 찾기 (최대 10일 전까지)
        for i in range(10):
            target_date = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
            try:
                fundamentals = pykrx_stock.get_market_fundamental(target_date, target_date, code)
                if not fundamentals.empty:
                    return {
                        'per': _safe_round(fundamentals['PER'].iloc[-1]),
                        'pbr': _safe_round(fundamentals['PBR'].iloc[-1]),
                        'eps': _safe_round(fundamentals['EPS'].iloc[-1]),
                        'bps': _safe_round(fundamentals['BPS'].iloc[-1]),
                        'div_yield': _safe_round(fundamentals['DIV'].iloc[-1], 4),
                    }
            except Exception:
                continue

        return _empty_kr_fundamentals()

    except Exception as e:
        print(f"[WARN] KR 재무제표 조회 실패 ({code}): {e}")
        return _empty_kr_fundamentals()


def _safe_round(value, decimals: int = 2) -> Optional[float]:
    """안전한 반올림 (None, NaN 처리)"""
    if value is None:
        return None
    try:
        if pd.isna(value) or np.isinf(value):
            return None
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return None


def _empty_us_fundamentals() -> Dict[str, None]:
    """빈 US 재무제표 데이터"""
    return {
        'per': None, 'forward_per': None, 'pbr': None,
        'roe': None, 'roa': None, 'eps': None, 'forward_eps': None,
        'revenue_growth': None, 'earnings_growth': None,
        'profit_margin': None, 'operating_margin': None,
        'debt_to_equity': None, 'current_ratio': None,
        'dividend_yield': None, 'peg_ratio': None, 'beta': None,
        'market_cap': None, '52w_high': None, '52w_low': None,
    }


def _empty_kr_fundamentals() -> Dict[str, None]:
    """빈 KR 재무제표 데이터"""
    return {
        'per': None, 'pbr': None, 'eps': None, 'bps': None, 'div_yield': None,
    }


def analyze_fundamentals(fundamentals: Dict[str, Any], region: str = 'US') -> Dict[str, Any]:
    """
    재무제표 기반 투자 신호 분석

    Args:
        fundamentals: 재무제표 데이터
        region: 'US' 또는 'KR'

    Returns:
        재무 건전성 점수 및 신호
    """
    signals = []
    score_adjustment = 0

    per = fundamentals.get('per')
    pbr = fundamentals.get('pbr')

    # PER 분석
    if per is not None:
        if per < 0:
            signals.append('적자 기업')
            score_adjustment -= 5
        elif per < 10:
            signals.append('저PER (저평가 가능)')
            score_adjustment += 5
        elif per < 20:
            signals.append('적정 PER')
        elif per < 40:
            signals.append('고PER')
            score_adjustment -= 2
        else:
            signals.append('매우 고PER (고평가 주의)')
            score_adjustment -= 5

    # PBR 분석
    if pbr is not None:
        if pbr < 1:
            signals.append('저PBR (자산가치 대비 저평가)')
            score_adjustment += 3
        elif pbr > 5:
            signals.append('고PBR (프리미엄)')
            score_adjustment -= 2

    # US 전용 추가 분석
    if region == 'US':
        roe = fundamentals.get('roe')
        profit_margin = fundamentals.get('profit_margin')
        debt_to_equity = fundamentals.get('debt_to_equity')
        revenue_growth = fundamentals.get('revenue_growth')

        # ROE 분석
        if roe is not None:
            if roe > 0.20:
                signals.append('높은 ROE (20%+)')
                score_adjustment += 5
            elif roe > 0.15:
                signals.append('양호한 ROE (15-20%)')
                score_adjustment += 3
            elif roe < 0:
                signals.append('음의 ROE')
                score_adjustment -= 5

        # 이익률 분석
        if profit_margin is not None:
            if profit_margin > 0.20:
                signals.append('높은 이익률 (20%+)')
                score_adjustment += 3
            elif profit_margin < 0:
                signals.append('적자')
                score_adjustment -= 5

        # 부채비율 분석
        if debt_to_equity is not None:
            if debt_to_equity > 200:
                signals.append('높은 부채비율 (주의)')
                score_adjustment -= 3
            elif debt_to_equity < 50:
                signals.append('낮은 부채비율 (안정)')
                score_adjustment += 2

        # 매출 성장률
        if revenue_growth is not None:
            if revenue_growth > 0.20:
                signals.append('높은 매출 성장 (20%+)')
                score_adjustment += 3
            elif revenue_growth < -0.10:
                signals.append('매출 감소')
                score_adjustment -= 3

    return {
        'signals': signals,
        'score_adjustment': max(-15, min(15, score_adjustment)),  # -15 ~ +15 제한
        'health': 'good' if score_adjustment > 3 else ('warning' if score_adjustment < -3 else 'neutral')
    }


def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """RSI (Relative Strength Index) 계산"""
    if len(prices) < period + 1:
        return 50.0  # 데이터 부족시 중립값

    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # Division by zero 보호: loss가 0이면 RSI = 100 (극단적 상승)
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100)

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


def process_stock(
    stock_data: Dict[str, Any],
    df: pd.DataFrame,
    fundamentals: Optional[Dict[str, Any]] = None,
    region: str = 'KR'
) -> Dict[str, Any]:
    """
    개별 종목 처리 및 추천 정보 생성

    Args:
        stock_data: 종목 기본 정보
        df: OHLCV 데이터프레임
        fundamentals: 재무제표 데이터 (선택)
        region: 'US' 또는 'KR'

    Returns:
        종목 분석 결과
    """
    indicators = calculate_all_indicators(df)
    current_price = float(df['Close'].iloc[-1])
    signals = generate_signals(indicators, current_price)

    # 기술적 지표 기반 점수
    tech_score = calculate_recommendation_score(signals)

    # 재무제표 분석 및 점수 조정
    fundamental_analysis = None
    final_score = tech_score

    if fundamentals:
        fundamental_analysis = analyze_fundamentals(fundamentals, region)
        # 재무제표 기반 점수 조정 (-15 ~ +15)
        final_score = max(0, min(100, tech_score + fundamental_analysis['score_adjustment']))

    grade = get_recommendation_grade(final_score)
    summary = generate_summary(signals)

    # 재무제표 신호가 있으면 요약에 추가
    if fundamental_analysis and fundamental_analysis['signals']:
        fund_summary = ' | '.join(fundamental_analysis['signals'][:2])  # 최대 2개
        summary = f"{summary} | {fund_summary}" if summary != '특별한 신호 없음' else fund_summary

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
        'score': int(final_score),
        'tech_score': int(tech_score),  # 기술적 지표만 점수
        'grade': grade,
        'summary': summary
    }

    # 재무제표 데이터 추가 (있는 경우)
    if fundamentals:
        result['fundamentals'] = convert_to_native(fundamentals)
        if fundamental_analysis:
            result['fundamental_health'] = fundamental_analysis['health']

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
            'tech_score': stock.get('tech_score', stock['score']),
            'signals': stock['signals'],
            'summary': stock['summary'],
            'price': stock['price']['current'],
            'change_pct': stock['price']['change_pct']
        }

        # 재무제표 핵심 지표 추가 (있는 경우)
        if 'fundamentals' in stock:
            fund = stock['fundamentals']
            recommendation['fundamentals'] = {
                'per': fund.get('per'),
                'pbr': fund.get('pbr'),
                'roe': fund.get('roe'),
                'health': stock.get('fundamental_health', 'neutral')
            }

        categories[grade].append(recommendation)

    # 각 카테고리 내에서 점수순 정렬
    for grade in categories:
        if grade in ['strong_buy', 'buy']:
            categories[grade].sort(key=lambda x: x['score'], reverse=True)
        else:
            categories[grade].sort(key=lambda x: x['score'])

    return categories
