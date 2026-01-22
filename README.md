# 📈 주식 분석 대시보드 v2.5

AI 기반 기술적 분석을 통한 주식 투자 의사결정 지원 도구

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ 주요 기능

### 📊 포괄적인 시장 데이터
- 🇰🇷 **한국 주식**: 2,500개+ (코스피, 코스닥, 코넥스)
- 🇺🇸 **미국 주식**: 8,000개+ (NASDAQ, NYSE, AMEX)
- 🇯🇵 **일본 주식**: 검색 지원
- 🌐 **기타 시장**: 홍콩, 중국 등

### 🤖 AI 기반 종합 분석
- **6개 기술적 지표** 자동 계산 및 분석
  - MACD (25% 가중치)
  - 이동평균선 (20% 가중치)
  - RSI (20% 가중치)
  - 볼린저밴드 (15% 가중치)
  - 스토캐스틱 (12% 가중치)
  - ADX (8% 가중치)
- **종합 점수** (-100 ~ +100) 및 매매 추천
- **목표가/손절가** 자동 계산 (ATR 기반)

### 📈 강력한 시각화
- **6단 인터랙티브 차트**
  - 주가 캔들스틱 + 이동평균선 + 볼린저밴드
  - RSI + RSI Signal
  - MACD + Signal + Histogram
  - Stochastic (%K, %D)
  - ADX + DI
  - 거래량 + OBV
- **게이지 차트**: RSI, Stochastic, 볼린저밴드
- **미니 추세 차트**: 30일 추세 시각화
- **점수 막대 그래프**: 지표별 기여도

### 🎯 사용자 중심 기능
- ⚡ **즉시 분석**: 인기 종목/검색 결과 클릭 시 즉시 분석
- 📝 **투자 메모**: 종목별 전략 및 관찰 사항 기록
- 🔔 **가격 알림**: 목표가/손절가 알림 설정 (준비 중)
- 📖 **유저 가이드**: 사이드바에 상세한 사용 설명

---

## 🚀 빠른 시작

### 온라인 사용 (권장)
웹 브라우저에서 바로 사용 가능합니다. 설치 필요 없음!

👉 **[여기를 클릭하여 바로 시작](https://your-app-name.streamlit.app)**

### 로컬 설치

#### 1. 저장소 클론
```bash
git clone https://github.com/yourusername/stock-analyzer.git
cd stock-analyzer
```

#### 2. 가상환경 생성 (권장)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

#### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

#### 4. 실행
```bash
streamlit run stock_analyzer_v2.5_enhanced.py
```

#### 5. 브라우저 접속
자동으로 브라우저가 열립니다. 수동 접속: `http://localhost:8501`

---

## 📖 사용 방법

### 1️⃣ 종목 검색

#### 방법 A: 인기 종목 바로가기 (가장 빠름!)
1. "💡 인기 종목 바로가기" 클릭
2. 원하는 종목 버튼 클릭
3. 즉시 분석 결과 표시! ⚡

#### 방법 B: 검색창 사용
**한국 주식:**
```
예시: "삼성", "005930", "005930.KS"
```

**미국 주식:**
```
예시: "AAPL", "Apple", "TSLA"
```

#### 방법 C: 수동 입력
1. 국가 선택
2. 티커 입력
3. 분석 기간 선택 (1개월 ~ 5년)
4. "🔍 분석 시작" 클릭

### 2️⃣ 분석 결과 해석

#### 종합 점수
| 점수 | 추천 | 의미 |
|------|------|------|
| **+6 이상** | 🟢 강력 매수 | 매우 강한 상승 신호 |
| **+3 ~ +6** | 🟢 매수 | 상승 신호 |
| **+1 ~ +3** | 🟡 약한 매수 | 약한 상승 신호 |
| **-1 ~ +1** | 🟡 관망 | 중립 |
| **-3 ~ -1** | 🔴 약한 매도 | 약한 하락 신호 |
| **-6 ~ -3** | 🔴 매도 | 하락 신호 |
| **-6 이하** | 🔴 강력 매도 | 매우 강한 하락 신호 |

#### 5개 탭 구성
1. **📈 차트**: 6단 인터랙티브 차트
2. **📊 지표**: 게이지, 미니 차트, 상세 정보
3. **🎯 목표가**: 진입가, 목표가, 손절가
4. **⚠️ 위험도**: 변동성, 샤프비율, 낙폭
5. **📝 메모 & 알림**: 투자 메모, 알림 설정

### 3️⃣ 투자 전략 예시

#### 단기 매매 (1일 ~ 1주)
```
분석 기간: 1개월
차트 일수: 30-60일
주목 지표: RSI, Stochastic
진입: RSI < 30 + 상승 전환
청산: 목표가 1 도달
손절: 엄격히 적용
```

#### 스윙 트레이딩 (1주 ~ 3개월)
```
분석 기간: 6개월 ~ 1년
차트 일수: 120-180일
주목 지표: MACD, 이동평균선
진입: 정배열 + MACD 골든크로스
청산: 목표가 2-3 도달
손절: -5% 이하
```

#### 장기 투자 (6개월 이상)
```
분석 기간: 2-5년
차트 일수: 365일
주목 지표: 장기 MA, ADX
진입: 종합점수 +5 이상
청산: 목표가 3 또는 펀더멘털 변화
손절: -15% 또는 추세 반전
```

---

## 🛠️ 기술 스택

### 프론트엔드
- **Streamlit**: 웹 인터페이스
- **Plotly**: 인터랙티브 차트
- **Custom CSS**: 반응형 디자인

### 백엔드
- **Python 3.8+**
- **yfinance**: 주가 데이터
- **FinanceDataReader**: 한국/미국 주식 목록
- **ta (Technical Analysis)**: 기술적 지표 계산
- **pandas**: 데이터 처리
- **numpy**: 수치 계산
- **scipy**: 통계 분석

### 배포
- **Streamlit Community Cloud**
- **GitHub**: 버전 관리

---

## 📊 프로젝트 구조

```
stock-analyzer/
├── stock_analyzer_v2.5_enhanced.py  # 메인 애플리케이션
├── requirements.txt                  # 의존성 패키지
├── README.md                         # 프로젝트 설명
├── STREAMLIT_GUIDE.md               # Streamlit 배포 가이드
├── .gitignore                        # Git 제외 파일
└── streamlit_data/                   # 데이터 저장 폴더 (자동 생성)
    ├── favorites.json               # 즐겨찾기
    ├── history.json                 # 분석 히스토리
    ├── notes.json                   # 투자 메모
    └── alerts.json                  # 알림 설정
```

---

## ⚙️ 설정 및 커스터마이징

### 분석 기간 변경
`stock_analyzer_v2.5_enhanced.py` 파일에서:
```python
period = st.selectbox("분석 기간", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
```

### 지표 가중치 조정
```python
weights = {
    'MACD': 0.25,      # 25%
    'MA': 0.20,        # 20%
    'RSI': 0.20,       # 20%
    'BB': 0.15,        # 15%
    'STOCH': 0.12,     # 12%
    'ADX': 0.08        # 8%
}
```

### 차트 높이 조정
```python
fig.update_layout(height=1600)  # 픽셀 단위
```

---

## 🤝 기여하기

기여를 환영합니다! 다음 절차를 따라주세요:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### 개선 아이디어
- [ ] 실시간 가격 알림 기능 구현
- [ ] 포트폴리오 관리 기능
- [ ] 백테스팅 시뮬레이터
- [ ] 종목 비교 분석 기능
- [ ] 모바일 앱 버전
- [ ] 다국어 지원 (영어, 중국어 등)
- [ ] 펀더멘털 분석 추가
- [ ] AI 기반 예측 모델

---

## 📝 버전 히스토리

### v2.5 (2025-01-22) - Enhanced
- ✨ 게이지 차트 추가 (RSI, Stochastic, 볼린저밴드)
- ✨ 미니 추세 차트 추가 (30일 추세)
- ✨ 사이드바 유저 가이드 추가
- ✨ 투자 메모 기능 추가
- ✨ 가격 알림 UI 추가
- 🎨 지표 탭 시각화 대폭 개선

### v2.4 (2025-01-21) - Complete
- ✨ 6단 차트 구조 (거래량 + OBV 추가)
- ✨ RSI Signal 추가
- 📊 상세 지표 정보 표시
- 🎯 목표가/손절가 계산 개선

### v2.3 (2025-01-20) - Final
- ✨ 즉시 분석 실행 기능
- 🔍 검색창 항상 유지
- 🇰🇷🇺🇸 한국/미국 주식 캐시 통일
- 📊 로드 개수 표시

### v2.0 (2025-01-15) - Major Update
- 🌏 FinanceDataReader 통합
- 🇰🇷 2,500개+ 한국 주식 지원
- 🇺🇸 8,000개+ 미국 주식 지원
- 🤖 AI 기반 종합 점수 시스템

### v1.0 (2025-01-10) - Initial Release
- 📈 기본 차트 기능
- 📊 주요 기술적 지표
- 🔍 종목 검색

---

## ⚠️ 면책 조항

**중요: 반드시 읽어주세요**

이 프로그램은 **교육 및 정보 제공 목적**으로만 제작되었습니다.

### 주의사항
- ❌ **투자 조언이 아닙니다**
- ❌ **수익을 보장하지 않습니다**
- ❌ **실시간 데이터가 아닙니다** (일봉 기준)
- ❌ **펀더멘털 분석이 포함되지 않았습니다**

### 투자 원칙
1. ✅ 모든 투자는 본인 책임입니다
2. ✅ 과거 데이터 ≠ 미래 수익
3. ✅ 분산 투자를 권장합니다
4. ✅ 손실 감내 가능 금액만 투자하세요
5. ✅ 반드시 추가 조사를 수행하세요

### 권장사항
- 전문가 상담
- 펀더멘털 분석 병행
- 뉴스 및 공시 확인
- 리스크 관리 철저

**투자 손실에 대한 책임은 투자자 본인에게 있습니다.**

---

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

```
MIT License

Copyright (c) 2025 Stock Analyzer Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📞 문의 및 지원

### 문제 보고
GitHub Issues를 통해 버그나 개선 사항을 제안해주세요.
- [Issues 페이지](https://github.com/yourusername/stock-analyzer/issues)

### 토론 및 질문
- [Discussions 페이지](https://github.com/yourusername/stock-analyzer/discussions)

### 이메일
- 개발자: your.email@example.com

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/stock-analyzer&type=Date)](https://star-history.com/#yourusername/stock-analyzer&Date)

---

## 🙏 감사의 말

이 프로젝트는 다음 오픈소스 프로젝트들을 활용하여 제작되었습니다:

- [Streamlit](https://streamlit.io/) - 웹 인터페이스
- [yfinance](https://github.com/ranaroussi/yfinance) - 주가 데이터
- [FinanceDataReader](https://github.com/FinanceData/FinanceDataReader) - 한국/미국 주식 목록
- [ta](https://github.com/bukosabino/ta) - 기술적 분석
- [Plotly](https://plotly.com/) - 인터랙티브 차트

---

<div align="center">

**⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요! ⭐**

Made with ❤️ by Stock Analyzer Team

</div>
