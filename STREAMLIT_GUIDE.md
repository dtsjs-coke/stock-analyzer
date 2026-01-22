# 🚀 Streamlit Community Cloud 배포 가이드

이 문서는 주식 분석 대시보드를 Streamlit Community Cloud에 배포하는 방법을 단계별로 설명합니다.

---

## 📋 목차

1. [사전 준비](#사전-준비)
2. [GitHub 저장소 준비](#github-저장소-준비)
3. [Streamlit Community Cloud 설정](#streamlit-community-cloud-설정)
4. [배포 및 확인](#배포-및-확인)
5. [업데이트 및 관리](#업데이트-및-관리)
6. [문제 해결](#문제-해결)
7. [고급 설정](#고급-설정)

---

## 🎯 사전 준비

### 필요한 계정

1. **GitHub 계정** (무료)
   - https://github.com/signup
   - 저장소 생성 및 코드 관리

2. **Streamlit Community Cloud 계정** (무료)
   - https://streamlit.io/cloud
   - GitHub 계정으로 로그인 가능

### 로컬 환경 확인

#### Python 버전
```bash
python --version
# Python 3.8 이상 필요
```

#### Git 설치 확인
```bash
git --version
# git version 2.x.x
```

---

## 📦 GitHub 저장소 준비

### 1. 새 저장소 생성

#### GitHub에서 생성
1. GitHub 로그인
2. 우측 상단 `+` → `New repository` 클릭
3. 저장소 정보 입력:
   ```
   Repository name: stock-analyzer
   Description: AI-powered stock analysis dashboard
   Public/Private: Public (권장)
   Initialize: Add README (선택 사항)
   ```
4. `Create repository` 클릭

### 2. 로컬 프로젝트 준비

#### 프로젝트 구조 확인
```
stock-analyzer/
├── stock_analyzer_v2.5_enhanced.py  # 메인 파일
├── requirements.txt                  # 의존성
├── README.md                         # 프로젝트 설명
├── STREAMLIT_GUIDE.md               # 이 파일
└── .gitignore                        # Git 제외 파일
```

#### .gitignore 파일 생성
```bash
# .gitignore
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
ENV/
.env
.venv
streamlit_data/
*.json
.DS_Store
.streamlit/secrets.toml
```

### 3. 파일 확인 체크리스트

✅ **필수 파일**
- [ ] `stock_analyzer_v2.5_enhanced.py` - 메인 애플리케이션
- [ ] `requirements.txt` - 패키지 의존성
- [ ] `README.md` - 프로젝트 설명

✅ **권장 파일**
- [ ] `.gitignore` - Git 제외 파일
- [ ] `STREAMLIT_GUIDE.md` - 배포 가이드

### 4. Git 저장소 초기화 및 푸시

```bash
# 1. 프로젝트 폴더로 이동
cd stock-analyzer

# 2. Git 초기화
git init

# 3. 원격 저장소 연결
git remote add origin https://github.com/yourusername/stock-analyzer.git

# 4. 파일 추가
git add .

# 5. 커밋
git commit -m "Initial commit: Stock Analyzer v2.5"

# 6. 푸시 (main 브랜치)
git branch -M main
git push -u origin main
```

---

## ☁️ Streamlit Community Cloud 설정

### 1. Streamlit Cloud 접속

1. https://streamlit.io/cloud 방문
2. `Sign in with GitHub` 클릭
3. GitHub 계정으로 로그인
4. 권한 승인

### 2. 새 앱 배포

#### 단계별 설정

1. **대시보드에서 `New app` 클릭**

2. **저장소 선택**
   ```
   Repository: yourusername/stock-analyzer
   Branch: main
   Main file path: stock_analyzer_v2.5_enhanced.py
   ```

3. **앱 URL 설정 (선택)**
   ```
   App URL: stock-analyzer (또는 원하는 이름)
   최종 URL: https://stock-analyzer.streamlit.app
   ```

4. **고급 설정 (Advanced settings)** - 선택 사항
   ```
   Python version: 3.11 (권장)
   ```

5. **Deploy!** 클릭

### 3. 배포 프로세스

배포가 시작되면 다음 단계가 자동으로 진행됩니다:

```
1. 📦 저장소 클론
2. 🐍 Python 환경 설정
3. 📚 패키지 설치 (requirements.txt)
4. 🚀 앱 시작
5. ✅ 배포 완료!
```

**예상 소요 시간**: 3-5분

---

## 🎉 배포 및 확인

### 배포 완료 확인

배포가 성공하면:
1. ✅ 상태가 `Running`으로 변경
2. 🌐 앱 URL 활성화: `https://your-app-name.streamlit.app`
3. 📊 대시보드 접속 가능

### 첫 실행 시 주의사항

**최초 로딩 시간**: 약 30-60초
- 한국 주식 2,500개+ 로딩
- 미국 주식 8,000개+ 로딩
- 이후 24시간 캐시 유지

### 테스트 체크리스트

- [ ] 앱이 정상적으로 로드되는가?
- [ ] 한국 주식 데이터가 로드되는가?
- [ ] 미국 주식 데이터가 로드되는가?
- [ ] 검색 기능이 작동하는가?
- [ ] 차트가 정상적으로 표시되는가?
- [ ] 모든 탭이 작동하는가?

---

## 🔄 업데이트 및 관리

### 코드 업데이트

#### 로컬에서 수정 후 배포

```bash
# 1. 파일 수정 (예: stock_analyzer_v2.5_enhanced.py)

# 2. 변경사항 확인
git status

# 3. 변경 파일 추가
git add stock_analyzer_v2.5_enhanced.py

# 4. 커밋
git commit -m "Update: 새로운 기능 추가"

# 5. 푸시
git push origin main
```

**자동 재배포**: GitHub에 푸시하면 Streamlit Cloud가 자동으로 감지하고 재배포합니다!

### 의존성 업데이트

`requirements.txt` 수정 후:
```bash
git add requirements.txt
git commit -m "Update dependencies"
git push origin main
```

### 앱 재시작

Streamlit Cloud 대시보드에서:
1. 앱 선택
2. 우측 상단 `⋮` (메뉴)
3. `Reboot app` 클릭

---

## 🐛 문제 해결

### 일반적인 오류

#### 1. ModuleNotFoundError
```
❌ ModuleNotFoundError: No module named 'xxx'
```

**해결책:**
```bash
# requirements.txt에 패키지 추가
echo "package-name==version" >> requirements.txt
git add requirements.txt
git commit -m "Add missing package"
git push
```

#### 2. 메모리 부족
```
❌ Memory limit exceeded
```

**해결책:**
- 데이터 로딩 최적화
- 캐시 적극 활용 (`@st.cache_data`)
- 불필요한 데이터 제거

#### 3. 타임아웃
```
❌ App took too long to start
```

**해결책:**
- 초기 로딩 시간 단축
- 데이터 미리 로드
- `@st.cache_data` 사용

#### 4. 파일 경로 오류
```
❌ FileNotFoundError
```

**해결책:**
```python
# 절대 경로 대신 상대 경로 사용
from pathlib import Path

data_dir = Path("streamlit_data")
data_dir.mkdir(exist_ok=True)
```

### 로그 확인

Streamlit Cloud 대시보드에서:
1. 앱 선택
2. `Logs` 탭 클릭
3. 실시간 로그 확인

---

## ⚙️ 고급 설정

### 1. Secrets 관리

민감한 정보 (API 키 등)를 안전하게 저장:

#### Streamlit Cloud에서 설정
1. 앱 대시보드 → `Settings` → `Secrets`
2. TOML 형식으로 입력:
   ```toml
   [api_keys]
   alpha_vantage = "your-api-key"
   ```

#### 코드에서 사용
```python
import streamlit as st

api_key = st.secrets["api_keys"]["alpha_vantage"]
```

### 2. 커스텀 도메인 (Pro 플랜)

Pro 플랜 사용 시 커스텀 도메인 연결 가능:
```
your-app.streamlit.app → stocks.yourdomain.com
```

### 3. 환경 변수 설정

`.streamlit/config.toml` 파일 생성:
```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableXsrfProtection = true
enableCORS = false
```

### 4. 성능 최적화

#### 캐시 전략
```python
@st.cache_data(ttl=86400)  # 24시간
def load_data():
    # 데이터 로드
    return data

@st.cache_resource  # 세션 간 공유
def get_database_connection():
    return connection
```

#### 리소스 관리
```python
# 사용하지 않는 임포트 제거
# 큰 데이터셋은 필요할 때만 로드
# 메모리 사용량 모니터링
```

---

## 📊 모니터링 및 분석

### Analytics 확인

Streamlit Cloud 대시보드:
1. `Analytics` 탭
2. 확인 가능한 지표:
   - 일일 방문자 수
   - 세션 지속 시간
   - 지역별 접속
   - 오류율

### 사용량 제한 (무료 플랜)

| 항목 | 제한 |
|------|------|
| 앱 개수 | 1개 (Public) |
| 동시 사용자 | 무제한 |
| 리소스 | 1GB RAM |
| 스토리지 | 제한 없음 |

**팁**: 복잡한 연산은 최적화 필요

---

## 🔒 보안 모범 사례

### 1. 민감한 정보 보호
```python
# ❌ 절대 하지 말 것
API_KEY = "sk-1234567890abcdef"

# ✅ Secrets 사용
api_key = st.secrets["api_keys"]["openai"]
```

### 2. 입력 검증
```python
# 사용자 입력 검증
ticker = st.text_input("티커 입력")
if ticker:
    ticker = ticker.strip().upper()
    if len(ticker) > 10:  # 길이 제한
        st.error("티커가 너무 깁니다")
```

### 3. 에러 처리
```python
try:
    data = get_stock_data(ticker)
except Exception as e:
    st.error(f"오류 발생: {e}")
    # 민감한 정보는 로그에만 기록
```

---

## 🎯 체크리스트: 배포 전 최종 확인

### 코드
- [ ] 모든 `requirements.txt`에 패키지 포함
- [ ] 하드코딩된 경로 제거
- [ ] API 키 등 민감 정보 Secrets로 이동
- [ ] 에러 처리 추가
- [ ] 캐시 적절히 사용

### GitHub
- [ ] `.gitignore` 설정
- [ ] 불필요한 파일 제외
- [ ] README.md 작성
- [ ] 라이선스 추가

### Streamlit Cloud
- [ ] 저장소 연결 확인
- [ ] Python 버전 설정
- [ ] Secrets 설정 (필요시)
- [ ] 앱 이름 확인

### 테스트
- [ ] 로컬에서 정상 작동 확인
- [ ] 모든 기능 테스트
- [ ] 다양한 브라우저 확인
- [ ] 모바일 반응형 확인

---

## 📚 추가 리소스

### 공식 문서
- [Streamlit 공식 문서](https://docs.streamlit.io/)
- [Streamlit Cloud 문서](https://docs.streamlit.io/streamlit-community-cloud)
- [배포 가이드](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)

### 커뮤니티
- [Streamlit Forum](https://discuss.streamlit.io/)
- [GitHub Discussions](https://github.com/streamlit/streamlit/discussions)
- [Discord](https://discord.gg/streamlit)

### 튜토리얼
- [Streamlit 30일 챌린지](https://30days.streamlit.app/)
- [예제 갤러리](https://streamlit.io/gallery)

---

## 💡 팁과 트릭

### 1. 빠른 디버깅
```python
# 배포 환경 감지
import os
is_cloud = "STREAMLIT_RUNTIME_ENV" in os.environ

if is_cloud:
    # 클라우드 전용 설정
    pass
else:
    # 로컬 전용 설정
    pass
```

### 2. 진행 상황 표시
```python
with st.spinner("데이터 로딩 중..."):
    data = load_large_dataset()
```

### 3. 상태 관리
```python
# 세션 상태 활용
if 'counter' not in st.session_state:
    st.session_state.counter = 0

st.session_state.counter += 1
```

### 4. 레이아웃 최적화
```python
# 컬럼 사용
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("지표 1", value1)
```

---

## 🎓 다음 단계

배포가 완료되었다면:

1. **📢 공유하기**
   - 앱 URL을 README.md에 추가
   - 소셜 미디어에 공유
   - Badge 추가 (위 참고)

2. **📊 모니터링**
   - Analytics 주기적 확인
   - 사용자 피드백 수집
   - 오류 로그 검토

3. **🔄 업데이트**
   - 정기적인 업데이트
   - 새로운 기능 추가
   - 버그 수정

4. **💰 업그레이드 고려** (선택)
   - Pro 플랜 ($20/월)
   - 무제한 앱
   - 커스텀 도메인
   - 우선 지원

---

## ❓ FAQ

### Q1: 배포가 실패했어요
**A**: 로그를 확인하세요. 대부분 `requirements.txt` 또는 파일 경로 문제입니다.

### Q2: 앱이 느려요
**A**: 캐시를 적극 활용하고, 불필요한 연산을 줄이세요.

### Q3: 데이터가 사라져요
**A**: Streamlit Cloud는 stateless입니다. 영구 데이터는 외부 DB를 사용하세요.

### Q4: 무료 플랜 제한이 있나요?
**A**: Public 앱 1개, 리소스 1GB RAM 제한이 있습니다.

### Q5: 프라이빗 저장소도 가능한가요?
**A**: 네, Pro 플랜에서 가능합니다.

---

## 📞 지원

문제가 해결되지 않으면:

1. [Streamlit Forum](https://discuss.streamlit.io/)에 질문
2. [GitHub Issues](https://github.com/streamlit/streamlit/issues) 검색
3. 공식 문서 참조

---

<div align="center">

**🎉 축하합니다! 배포 완료! 🎉**

앱 URL을 친구들과 공유하세요!

Made with ❤️ using Streamlit

</div>
