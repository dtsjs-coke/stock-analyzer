# 📈 Streamlit 주식 분석 웹 앱

## 🚀 빠른 시작

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

또는 개별 설치:

```bash
pip install streamlit yfinance pandas ta plotly scipy numpy openpyxl
```

### 2. 앱 실행

```bash
streamlit run streamlit_stock_app.py
```

브라우저가 자동으로 열리며 `http://localhost:8501`에서 앱이 실행됩니다.

### 3. 사용 방법

#### 🔍 단일 종목 분석
1. 사이드바에서 "단일 종목 분석" 선택
2. 티커 입력 (예: AAPL, TSLA, 005930.KS)
3. 분석 기간 선택
4. "분석 시작" 버튼 클릭

#### 📊 종목 비교
1. "종목 비교" 메뉴 선택
2. 쉼표로 구분하여 여러 티커 입력 (예: AAPL,MSFT,GOOGL)
3. "비교 분석" 버튼 클릭

#### ⭐ 즐겨찾기
1. "즐겨찾기" 메뉴 선택
2. 자주 보는 티커 추가
3. 즐겨찾기 목록에서 빠른 분석

#### 📜 분석 히스토리
- 과거 분석 기록 확인
- 티커별 필터링 가능

## 🎨 주요 기능

### ✨ 인터랙티브 차트
- **Plotly 기반**: 확대/축소, 패닝, 호버 정보
- **5개 패널**: 가격, RSI, MACD, Stochastic, ADX
- **실시간 업데이트**: 차트 조작 시 즉시 반영

### 📊 종합 대시보드
- **가격 정보**: 현재가, 52주 고저, 거래량
- **기술적 지표**: 10+ 지표 실시간 계산
- **목표가 계산**: ATR 기반 진입가/목표가/손절가
- **위험도 분석**: 변동성, 샤프비율, 최대낙폭

### 🎯 스마트 분석
- **하이브리드 스코어링**: 레벨 + 모멘텀 종합 분석
- **자동 추천**: 강력매수/매수/관망/매도/강력매도
- **지표별 점수**: 각 지표의 기여도 시각화

### 💾 데이터 관리
- **즐겨찾기**: 자주 보는 종목 저장
- **분석 히스토리**: 과거 분석 기록 추적
- **캐시 기능**: 1시간 동안 데이터 캐싱으로 빠른 재조회

## 🎨 UI 특징

### 📱 반응형 디자인
- 다양한 화면 크기 지원
- 모바일 친화적

### 🎨 색상 코딩
- 🟢 녹색: 매수 신호, 긍정적 지표
- 🔴 빨강: 매도 신호, 부정적 지표
- 🟡 노랑: 중립 신호

### 📊 시각화
- 막대 그래프: 점수 비교
- 산점도: 위험도 vs 수익성
- 라인 차트: 가격 추이
- 프로그레스 바: RSI, Stochastic 수준

## 🔧 커스터마이징

### 테마 변경
Streamlit 설정 파일 생성 (`.streamlit/config.toml`):

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"
```

### 포트 변경
```bash
streamlit run streamlit_stock_app.py --server.port 8502
```

### 자동 새로고침 비활성화
```bash
streamlit run streamlit_stock_app.py --server.runOnSave false
```

## 💡 사용 팁

### 한국 주식 분석
```
티커 입력: 005930.KS (삼성전자)
티커 입력: 000660.KS (SK하이닉스)
티커 입력: 035420.KS (NAVER)
```

### 포트폴리오 비교
```
종목 비교 입력: AAPL,MSFT,GOOGL,AMZN,NVDA
```

### 섹터별 비교
```
반도체: NVDA,AMD,INTC,TSM
자동차: TSLA,F,GM,TM
에너지: XOM,CVX,COP,SLB
```

## 📊 성능 최적화

### 캐시 활용
- 데이터는 1시간 동안 캐시됨
- 같은 종목 재조회 시 즉시 로딩

### 병렬 처리
- 여러 종목 비교 시 순차 처리
- 진행 상황 실시간 표시

## 🐛 문제 해결

### 데이터 로딩 실패
```
✓ 티커 철자 확인
✓ 인터넷 연결 확인
✓ Yahoo Finance 서버 상태 확인
```

### 차트가 안 보임
```
✓ 브라우저 새로고침 (F5)
✓ 캐시 지우기
✓ 다른 브라우저 시도
```

### 느린 속도
```
✓ 분석 기간 단축 (1y → 6mo)
✓ 차트 표시 일수 감소 (180 → 90)
✓ 비교 종목 수 감소
```

## 🚀 배포

### Streamlit Cloud
1. GitHub에 코드 푸시
2. [share.streamlit.io](https://share.streamlit.io) 접속
3. 저장소 연결 및 배포

### Docker
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY streamlit_stock_app.py .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_stock_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

빌드 및 실행:
```bash
docker build -t stock-analyzer .
docker run -p 8501:8501 stock-analyzer
```

## 📝 주의사항

1. **데이터 지연**: Yahoo Finance 데이터는 실시간이 아님
2. **API 제한**: 과도한 요청 시 제한될 수 있음
3. **투자 판단**: 참고용이며 실제 투자는 본인 책임

## 🆚 기존 코드 vs Streamlit 앱

| 기능 | 기존 코드 | Streamlit 앱 |
|------|-----------|--------------|
| **인터페이스** | 콘솔 | 웹 브라우저 |
| **차트** | 정적 PNG | 인터랙티브 Plotly |
| **사용성** | 코딩 필요 | 클릭만으로 사용 |
| **접근성** | 로컬만 | 어디서나 접근 가능 |
| **공유** | 어려움 | URL 공유 |
| **비교 분석** | 기본 | 4가지 차트 |
| **실시간성** | 수동 실행 | 자동 캐싱 |

## 🎯 다음 단계

- [ ] 실시간 가격 업데이트
- [ ] 알림 기능 (가격/지표 알림)
- [ ] 백테스팅 시뮬레이션
- [ ] 뉴스 통합
- [ ] 포트폴리오 관리
- [ ] 모바일 앱

## 📞 지원

문제가 있거나 제안사항이 있으시면 이슈를 남겨주세요!

---

**면책조항**: 이 앱은 교육 및 정보 제공 목적입니다. 투자 조언이 아닙니다.
