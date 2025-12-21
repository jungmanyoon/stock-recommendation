/**
 * 주식 자동 추천 - Claude 아티팩트
 *
 * 사용법:
 * 1. Claude.ai에서 새 대화 시작
 * 2. 이 코드를 붙여넣고 "React 아티팩트로 만들어줘" 요청
 * 3. Publish 버튼으로 링크 생성
 * 4. 지인들에게 링크 공유
 *
 * 중요: 아래 CONFIG의 GITHUB_USER와 REPO_NAME을 본인 것으로 변경하세요!
 */

import React, { useState, useEffect } from 'react';

// ========================================
// 설정 - 본인의 GitHub 정보로 변경하세요!
// ========================================
const CONFIG = {
  GITHUB_USER: 'jungmanyoon',  // GitHub 사용자명
  REPO_NAME: 'stock-recommendation',  // 저장소 이름
  BRANCH: 'main'
};

const BASE_URL = `https://cdn.jsdelivr.net/gh/${CONFIG.GITHUB_USER}/${CONFIG.REPO_NAME}@${CONFIG.BRANCH}`;

// 캐시 무효화용 날짜
const CACHE_BUSTER = new Date().toISOString().split('T')[0];

// ========================================
// 메인 앱 컴포넌트
// ========================================
export default function StockRecommendation() {
  const [data, setData] = useState(null);
  const [region, setRegion] = useState('kr');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedStock, setSelectedStock] = useState(null);

  useEffect(() => {
    fetchData(region);
  }, [region]);

  const fetchData = async (reg) => {
    setLoading(true);
    setError(null);
    try {
      const url = `${BASE_URL}/data/${reg}/${reg}_recommendations.json?v=${CACHE_BUSTER}`;
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`데이터를 불러올 수 없습니다 (${response.status})`);
      }
      const json = await response.json();
      setData(json);
    } catch (err) {
      console.error('Fetch error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <LoadingSpinner />;
  if (error) return <ErrorMessage message={error} onRetry={() => fetchData(region)} />;
  if (!data) return <ErrorMessage message="데이터가 없습니다" onRetry={() => fetchData(region)} />;

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto p-4">
        <Header />
        <RegionSelector region={region} onChange={(r) => { setRegion(r); setSelectedStock(null); }} />
        <MarketSummary summary={data.market_summary} region={region} />
        <Stats stats={data.stats} />
        <RecommendationList
          recommendations={data.recommendations}
          selectedStock={selectedStock}
          onSelectStock={setSelectedStock}
        />
        <Footer updatedAt={data.updated_at} />
      </div>
    </div>
  );
}

// ========================================
// 헤더 컴포넌트
// ========================================
function Header() {
  return (
    <div className="text-center py-6">
      <h1 className="text-3xl font-bold text-gray-800 mb-2">주식 자동 추천</h1>
      <p className="text-gray-500">기술적 지표 기반 자동 분석 시스템</p>
    </div>
  );
}

// ========================================
// 지역 선택 컴포넌트
// ========================================
function RegionSelector({ region, onChange }) {
  return (
    <div className="flex justify-center gap-3 mb-6">
      <button
        onClick={() => onChange('kr')}
        className={`px-6 py-3 rounded-lg font-medium transition-all ${
          region === 'kr'
            ? 'bg-blue-600 text-white shadow-lg'
            : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'
        }`}
      >
        🇰🇷 한국 주식
      </button>
      <button
        onClick={() => onChange('us')}
        className={`px-6 py-3 rounded-lg font-medium transition-all ${
          region === 'us'
            ? 'bg-blue-600 text-white shadow-lg'
            : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'
        }`}
      >
        🇺🇸 미국 주식
      </button>
    </div>
  );
}

// ========================================
// 시장 요약 컴포넌트
// ========================================
function MarketSummary({ summary, region }) {
  if (!summary) return null;

  const getSentimentBadge = (sentiment) => {
    const styles = {
      bullish: 'bg-green-100 text-green-700',
      bearish: 'bg-red-100 text-red-700',
      neutral: 'bg-gray-100 text-gray-700'
    };
    const labels = {
      bullish: '강세',
      bearish: '약세',
      neutral: '중립'
    };
    return (
      <span className={`px-3 py-1 rounded-full text-sm font-medium ${styles[sentiment] || styles.neutral}`}>
        {labels[sentiment] || sentiment}
      </span>
    );
  };

  const formatChange = (value) => {
    if (value === undefined || value === null) return '-';
    const prefix = value >= 0 ? '+' : '';
    return `${prefix}${value.toFixed(2)}%`;
  };

  const getChangeColor = (value, isKorean = true) => {
    if (value === undefined || value === null) return 'text-gray-500';
    if (isKorean) {
      return value >= 0 ? 'text-red-500' : 'text-blue-500';
    } else {
      return value >= 0 ? 'text-green-500' : 'text-red-500';
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-sm p-5 mb-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-800">시장 현황</h2>
        {getSentimentBadge(summary.market_sentiment)}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        {region === 'kr' ? (
          <>
            <IndexCard
              name="KOSPI"
              value={summary.kospi_index}
              change={summary.kospi_change_pct}
              isKorean={true}
            />
            <IndexCard
              name="KOSDAQ"
              value={summary.kosdaq_index}
              change={summary.kosdaq_change_pct}
              isKorean={true}
            />
          </>
        ) : (
          <>
            <IndexCard
              name="S&P 500"
              value={summary.sp500_index}
              change={summary.sp500_change_pct}
              isKorean={false}
            />
            <IndexCard
              name="NASDAQ"
              value={summary.nasdaq_index}
              change={summary.nasdaq_change_pct}
              isKorean={false}
            />
            <IndexCard
              name="DOW"
              value={summary.dow_index}
              change={summary.dow_change_pct}
              isKorean={false}
            />
          </>
        )}
      </div>
    </div>
  );
}

function IndexCard({ name, value, change, isKorean }) {
  if (!value) return null;

  const getChangeColor = (val, korean) => {
    if (val === undefined || val === null) return 'text-gray-500';
    if (korean) {
      return val >= 0 ? 'text-red-500' : 'text-blue-500';
    }
    return val >= 0 ? 'text-green-500' : 'text-red-500';
  };

  return (
    <div className="bg-gray-50 rounded-lg p-3">
      <div className="text-sm text-gray-500">{name}</div>
      <div className="text-xl font-bold text-gray-800">
        {typeof value === 'number' ? value.toLocaleString() : value}
      </div>
      <div className={`text-sm font-medium ${getChangeColor(change, isKorean)}`}>
        {change !== undefined && change !== null ? (
          `${change >= 0 ? '▲' : '▼'} ${Math.abs(change).toFixed(2)}%`
        ) : '-'}
      </div>
    </div>
  );
}

// ========================================
// 통계 컴포넌트
// ========================================
function Stats({ stats }) {
  if (!stats) return null;

  return (
    <div className="grid grid-cols-5 gap-2 mb-6">
      <StatBadge label="적극매수" count={stats.strong_buy} color="bg-green-600" />
      <StatBadge label="매수" count={stats.buy} color="bg-green-400" />
      <StatBadge label="보유" count={stats.hold} color="bg-yellow-400" />
      <StatBadge label="매도" count={stats.sell} color="bg-orange-400" />
      <StatBadge label="적극매도" count={stats.strong_sell} color="bg-red-600" />
    </div>
  );
}

function StatBadge({ label, count, color }) {
  return (
    <div className="text-center">
      <div className={`${color} text-white text-lg font-bold rounded-lg py-2`}>
        {count || 0}
      </div>
      <div className="text-xs text-gray-500 mt-1">{label}</div>
    </div>
  );
}

// ========================================
// 추천 리스트 컴포넌트
// ========================================
function RecommendationList({ recommendations, selectedStock, onSelectStock }) {
  const gradeConfig = {
    strong_buy: { label: '적극 매수', color: 'bg-green-600', emoji: '🚀' },
    buy: { label: '매수', color: 'bg-green-400', emoji: '📈' },
    hold: { label: '보유', color: 'bg-yellow-400', emoji: '⏸️' },
    sell: { label: '매도', color: 'bg-orange-400', emoji: '📉' },
    strong_sell: { label: '적극 매도', color: 'bg-red-600', emoji: '🔻' }
  };

  const gradeOrder = ['strong_buy', 'buy', 'hold', 'sell', 'strong_sell'];

  return (
    <div className="space-y-4">
      {gradeOrder.map(grade => {
        const stocks = recommendations[grade];
        if (!stocks || stocks.length === 0) return null;

        const config = gradeConfig[grade];

        return (
          <div key={grade} className="bg-white rounded-xl shadow-sm overflow-hidden">
            <div className={`${config.color} text-white px-4 py-3 flex items-center justify-between`}>
              <span className="font-semibold">
                {config.emoji} {config.label}
              </span>
              <span className="bg-white/20 px-3 py-1 rounded-full text-sm">
                {stocks.length}개
              </span>
            </div>
            <div className="divide-y divide-gray-100">
              {stocks.slice(0, 10).map((stock) => (
                <StockCard
                  key={stock.code}
                  stock={stock}
                  isSelected={selectedStock?.code === stock.code}
                  onSelect={() => onSelectStock(selectedStock?.code === stock.code ? null : stock)}
                />
              ))}
              {stocks.length > 10 && (
                <div className="px-4 py-3 text-center text-gray-500 text-sm">
                  +{stocks.length - 10}개 더 있음
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ========================================
// 종목 카드 컴포넌트
// ========================================
function StockCard({ stock, isSelected, onSelect }) {
  const signalLabels = {
    oversold: { label: 'RSI 과매도', color: 'bg-green-100 text-green-700' },
    oversold_recovery: { label: 'RSI 과매도 회복', color: 'bg-green-100 text-green-700' },
    overbought: { label: 'RSI 과매수', color: 'bg-red-100 text-red-700' },
    approaching_overbought: { label: 'RSI 과매수 접근', color: 'bg-orange-100 text-orange-700' },
    bullish_crossover: { label: 'MACD 골든크로스', color: 'bg-green-100 text-green-700' },
    bearish_crossover: { label: 'MACD 데드크로스', color: 'bg-red-100 text-red-700' },
    lower_band_touch: { label: '볼린저 하단', color: 'bg-green-100 text-green-700' },
    lower_band_bounce: { label: '볼린저 하단 반등', color: 'bg-green-100 text-green-700' },
    upper_band_touch: { label: '볼린저 상단', color: 'bg-red-100 text-red-700' },
    volume_surge: { label: '거래량 급증', color: 'bg-purple-100 text-purple-700' },
    above_average: { label: '거래량 증가', color: 'bg-purple-100 text-purple-700' },
    strong_uptrend: { label: '강한 상승추세', color: 'bg-green-100 text-green-700' },
    uptrend: { label: '상승추세', color: 'bg-green-100 text-green-700' },
    strong_downtrend: { label: '강한 하락추세', color: 'bg-red-100 text-red-700' },
    downtrend: { label: '하락추세', color: 'bg-red-100 text-red-700' }
  };

  const getChangeStyle = (value) => {
    if (!value) return 'text-gray-500';
    return value >= 0 ? 'text-red-500' : 'text-blue-500';
  };

  const formatPrice = (price) => {
    if (!price) return '-';
    return typeof price === 'number' ? price.toLocaleString() : price;
  };

  const activeSignals = stock.signals
    ? Object.entries(stock.signals)
        .filter(([key, value]) => signalLabels[value])
        .map(([key, value]) => ({ key, ...signalLabels[value] }))
    : [];

  return (
    <div
      className={`px-4 py-3 cursor-pointer transition-colors ${isSelected ? 'bg-blue-50' : 'hover:bg-gray-50'}`}
      onClick={onSelect}
    >
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-gray-800">{stock.name}</span>
            <span className="text-xs text-gray-400">{stock.code}</span>
            {stock.market && (
              <span className="text-xs bg-gray-100 text-gray-500 px-2 py-0.5 rounded">
                {stock.market}
              </span>
            )}
          </div>
          {stock.summary && (
            <div className="text-sm text-gray-500 mt-1">{stock.summary}</div>
          )}
        </div>
        <div className="text-right">
          <div className="font-bold text-gray-800">{formatPrice(stock.price)}</div>
          <div className={`text-sm ${getChangeStyle(stock.change_pct)}`}>
            {stock.change_pct !== undefined ? (
              `${stock.change_pct >= 0 ? '+' : ''}${stock.change_pct.toFixed(2)}%`
            ) : '-'}
          </div>
        </div>
        <div className="ml-4 flex items-center gap-2">
          <div className="bg-blue-600 text-white px-3 py-1 rounded-full text-sm font-bold">
            {stock.score}
          </div>
          <span className="text-gray-400">{isSelected ? '▲' : '▼'}</span>
        </div>
      </div>

      {isSelected && activeSignals.length > 0 && (
        <div className="mt-3 pt-3 border-t border-gray-100">
          <div className="flex flex-wrap gap-2">
            {activeSignals.map(signal => (
              <span
                key={signal.key}
                className={`px-2 py-1 rounded-full text-xs font-medium ${signal.color}`}
              >
                {signal.label}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ========================================
// 로딩 스피너
// ========================================
function LoadingSpinner() {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center">
      <div className="animate-spin rounded-full h-16 w-16 border-4 border-blue-600 border-t-transparent"></div>
      <p className="mt-4 text-gray-500 text-lg">데이터 로딩 중...</p>
      <p className="mt-2 text-gray-400 text-sm">잠시만 기다려주세요</p>
    </div>
  );
}

// ========================================
// 에러 메시지
// ========================================
function ErrorMessage({ message, onRetry }) {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center p-4">
      <div className="bg-white rounded-xl shadow-sm p-8 text-center max-w-md">
        <div className="text-red-500 text-5xl mb-4">⚠️</div>
        <h2 className="text-xl font-bold text-gray-800 mb-2">데이터를 불러올 수 없습니다</h2>
        <p className="text-gray-500 mb-4">{message}</p>
        <p className="text-sm text-gray-400 mb-6">
          GitHub 저장소 설정을 확인하거나,<br/>
          데이터가 아직 수집되지 않았을 수 있습니다.
        </p>
        <button
          onClick={onRetry}
          className="bg-blue-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-700 transition-colors"
        >
          다시 시도
        </button>
      </div>
    </div>
  );
}

// ========================================
// 푸터
// ========================================
function Footer({ updatedAt }) {
  const formatDate = (isoString) => {
    if (!isoString) return '알 수 없음';
    try {
      const date = new Date(isoString);
      return date.toLocaleString('ko-KR', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        timeZone: 'Asia/Seoul'
      });
    } catch {
      return isoString;
    }
  };

  return (
    <div className="mt-8 py-6 border-t border-gray-200 text-center">
      <p className="text-sm text-gray-500 mb-2">
        마지막 업데이트: {formatDate(updatedAt)}
      </p>
      <p className="text-xs text-gray-400">
        본 서비스는 투자 참고용이며, 투자 결정에 대한 책임은 본인에게 있습니다.
      </p>
      <p className="text-xs text-gray-400 mt-1">
        RSI, MACD, 볼린저밴드 등 기술적 지표 기반 자동 분석
      </p>
    </div>
  );
}
