# 주식 자동 추천 시스템

기술적 지표(RSI, MACD, 볼린저밴드) 기반 주식 자동 추천 시스템입니다.

## 특징

- **완전 무료**: API 비용 0원
- **자동 업데이트**: 매일 장 마감 후 자동 데이터 수집
- **Discord 알림**: 매일 지정 시간에 추천 종목 자동 알림
- **설정 불필요**: 링크 클릭만 하면 즉시 사용
- **한국 + 미국**: KOSPI, KOSDAQ, S&P500, NASDAQ100 지원

## 알림 시간

| 구분 | 알림 시간 | 설명 |
|------|----------|------|
| 🇰🇷 한국 주식 | 오전 8:00 | 장 시작 전 |
| 🇺🇸 미국 주식 | 오후 10:30 | 미국장 개장 1시간 전 |

## 사용 방법

### 지인에게 공유하기

1. Claude 아티팩트 링크를 공유
2. 지인은 Claude 계정으로 로그인
3. 끝! 설정 없이 바로 사용

## 아키텍처

```
[GitHub Actions] ──▶ [JSON 데이터] ──▶ [jsDelivr CDN]
     (매일 자동)         (저장)          (CORS 해결)
                                              │
                                              ▼
[사용자] ──▶ [Claude.ai] ──▶ [아티팩트] ──▶ [추천 결과]
```

## 추천 알고리즘

### 기술적 지표

| 지표 | 매수 신호 | 매도 신호 |
|------|----------|----------|
| RSI | < 30 (과매도) | > 70 (과매수) |
| MACD | 골든크로스 | 데드크로스 |
| 볼린저밴드 | 하단 터치/반등 | 상단 터치 |
| 거래량 | 1.5배 이상 증가 | - |

### 추천 등급

| 등급 | 점수 | 의미 |
|------|------|------|
| 적극 매수 | 80+ | 강한 매수 신호 |
| 매수 | 65-79 | 매수 고려 |
| 보유 | 45-64 | 관망 |
| 매도 | 30-44 | 매도 고려 |
| 적극 매도 | 0-29 | 강한 매도 신호 |

## 데이터 소스

| 구분 | 라이브러리 | 비용 |
|------|-----------|------|
| 한국 주식 | FinanceDataReader, pykrx | 무료 |
| 미국 주식 | yfinance | 무료 |
| 호스팅 | GitHub Pages, jsDelivr | 무료 |

## 업데이트 주기

- **한국 주식**: 평일 16:30 KST (장 마감 후)
- **미국 주식**: 평일 05:30 KST (장 마감 후)

## 설치 및 배포

### 1. 저장소 포크/클론

```bash
git clone https://github.com/YOUR_USERNAME/stock-recommendation.git
cd stock-recommendation
```

### 2. GitHub 설정

1. Settings > Pages > Source: `main` branch, `/` (root)
2. Settings > Actions > Workflow permissions: `Read and write`

### 3. Discord 웹훅 설정 (알림용)

1. **Discord 서버 생성** (또는 기존 서버 사용)

2. **웹훅 생성**:
   - 서버 설정 > 연동 > 웹후크 > 새 웹후크
   - 이름: `주식 추천 봇`
   - 채널: 알림 받을 채널 선택
   - **웹후크 URL 복사**

3. **GitHub Secrets 설정**:
   - 저장소 > Settings > Secrets and variables > Actions
   - `New repository secret` 클릭
   - Name: `DISCORD_WEBHOOK_URL`
   - Value: 복사한 웹훅 URL 붙여넣기

4. **지인 초대**:
   - Discord 서버에 지인들 초대
   - 알림 채널에서 매일 추천 종목 확인!

### 4. 첫 데이터 수집 (수동)

Actions 탭에서 워크플로우 수동 실행:
- `Collect Korean Stocks Data` > Run workflow
- `Collect US Stocks Data` > Run workflow
- `Discord Stock Notification` > Run workflow (테스트)

### 5. Claude 아티팩트 생성 (선택사항)

> Discord 알림만으로 충분하다면 이 단계는 건너뛰어도 됩니다.

1. `artifact/stock-recommendation.jsx` 파일의 CONFIG 수정:
   ```javascript
   const CONFIG = {
     GITHUB_USER: 'YOUR_USERNAME',  // 본인 GitHub 사용자명
     REPO_NAME: 'stock-recommendation',
     BRANCH: 'main'
   };
   ```

2. Claude.ai에서 새 대화 시작
3. 코드를 붙여넣고 "React 아티팩트로 만들어줘" 요청
4. Publish 버튼으로 링크 생성
5. 지인들에게 링크 공유!

## 폴더 구조

```
├── .github/workflows/      # GitHub Actions
├── scripts/                # Python 데이터 수집
├── data/                   # 수집된 JSON 데이터
├── artifact/               # Claude 아티팩트 코드
└── docs/                   # 문서
```

## 면책 조항

본 서비스는 **투자 참고용**입니다. 투자 결정에 대한 모든 책임은 사용자 본인에게 있습니다.

## 라이선스

MIT License
