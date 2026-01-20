"""
주식 분석 Streamlit 웹 애플리케이션
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import linregress
from datetime import datetime, timedelta
import json
from pathlib import Path

# 페이지 설정
st.set_page_config(
    page_title="📈 주식 분석 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .buy-signal {
        color: #00ff00;
        font-weight: bold;
    }
    .sell-signal {
        color: #ff0000;
        font-weight: bold;
    }
    .neutral-signal {
        color: #ffa500;
        font-weight: bold;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
    }
    </style>
    """, unsafe_allow_html=True)


class StreamlitStockAnalyzer:
    """Streamlit용 주식 분석 클래스"""
    
    def __init__(self):
        self.data_dir = Path("streamlit_data")
        self.data_dir.mkdir(exist_ok=True)
        self.favorites_file = self.data_dir / "favorites.json"
        self.history_file = self.data_dir / "history.json"
    
    @st.cache_data(ttl=3600)  # 1시간 캐시
    def get_stock_data(_self, ticker, period='1y'):
        """주가 데이터 가져오기 (캐시 적용)"""
        try:
            data = yf.download(ticker, period=period, progress=False)
            
            if data.empty:
                return None
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            data = data.loc[:, ~data.columns.duplicated()].copy()
            
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                return None
            
            if 'Adj Close' not in data.columns:
                data['Adj Close'] = data['Close']
            
            for col in data.columns:
                if col != 'Volume':
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce').fillna(0)
            
            data.dropna(subset=['Close'], inplace=True)
            
            return data if not data.empty else None
            
        except Exception as e:
            st.error(f"데이터 로딩 오류: {e}")
            return None
    
    def calculate_indicators(self, data):
        """기술적 지표 계산"""
        if len(data) < 120:
            st.warning("데이터가 부족합니다 (최소 120일 필요)")
            return data
        
        close = data['Close'].astype(float)
        high = data['High'].astype(float)
        low = data['Low'].astype(float)
        volume = data['Volume'].astype(float)
        
        # RSI
        data['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
        data['RSI_Signal'] = data['RSI'].rolling(window=9).mean()
        
        # MACD
        macd = ta.trend.MACD(close)
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Diff'] = macd.macd_diff()
        
        # 이동평균선
        data['MA_5'] = ta.trend.SMAIndicator(close, window=5).sma_indicator()
        data['MA_20'] = ta.trend.SMAIndicator(close, window=20).sma_indicator()
        data['MA_60'] = ta.trend.SMAIndicator(close, window=60).sma_indicator()
        data['MA_120'] = ta.trend.SMAIndicator(close, window=120).sma_indicator()
        
        # 볼린저 밴드
        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        data['BB_MA'] = bb.bollinger_mavg()
        data['BB_Upper'] = bb.bollinger_hband()
        data['BB_Lower'] = bb.bollinger_lband()
        data['BB_Width'] = bb.bollinger_wband()
        data['BB_Percent'] = bb.bollinger_pband()
        
        # 스토캐스틱
        stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
        data['STOCH_K'] = stoch.stoch()
        data['STOCH_D'] = stoch.stoch_signal()
        
        # ADX
        adx = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
        data['ADX'] = adx.adx()
        data['Plus_DI'] = adx.adx_pos()
        data['Minus_DI'] = adx.adx_neg()
        
        # ATR
        data['ATR'] = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
        
        # OBV
        data['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
        
        data.dropna(inplace=True)
        return data
    
    def calculate_slope(self, series, period):
        """추세선 기울기 계산"""
        if len(series) < period:
            return 0
        
        y = series.tail(period).values
        x = np.arange(len(y))
        
        if not np.all(np.isfinite(y)):
            return 0
        
        slope, _, _, _, _ = linregress(x, y)
        return slope if np.isfinite(slope) else 0
    
    def score_hybrid(self, data):
        """하이브리드 스코어링"""
        if len(data) < 10:
            return 0, "데이터 부족", 0, {}
        
        dynamic_period = int(max(3, min(10, len(data) * 0.05)))
        latest = data.iloc[-1]
        score = 0
        details = {}
        
        # RSI
        rsi_slope = self.calculate_slope(data['RSI'], dynamic_period)
        rsi_score = 0
        if latest['RSI'] < 30: rsi_score += 2
        if latest['RSI'] > 70: rsi_score -= 2
        if rsi_slope > 1.5: rsi_score += 2
        if rsi_slope < -1.5: rsi_score -= 2
        score += rsi_score
        details['RSI'] = rsi_score
        
        # MACD
        hist_slope = self.calculate_slope(data['MACD_Diff'], dynamic_period)
        macd_score = 0
        if latest['MACD'] > latest['MACD_Signal']: macd_score += 2
        if latest['MACD'] < latest['MACD_Signal']: macd_score -= 2
        if hist_slope > 0.1: macd_score += 2
        if hist_slope < -0.1: macd_score -= 2
        score += macd_score
        details['MACD'] = macd_score
        
        # 이동평균
        ma_spread_slope = self.calculate_slope(data['MA_5'] - data['MA_20'], dynamic_period)
        ma_score = 0
        if latest['MA_5'] > latest['MA_20']: ma_score += 1
        if latest['MA_5'] < latest['MA_20']: ma_score -= 1
        if ma_spread_slope > 0.5: ma_score += 2
        if ma_spread_slope < -0.5: ma_score -= 2
        score += ma_score
        details['MA'] = ma_score
        
        # 스토캐스틱
        stoch_slope = self.calculate_slope(data['STOCH_K'], dynamic_period)
        stoch_score = 0
        if latest['STOCH_K'] < 20: stoch_score += 2
        if latest['STOCH_K'] > 80: stoch_score -= 2
        if stoch_slope > 5: stoch_score += 2
        if stoch_slope < -5: stoch_score -= 2
        score += stoch_score
        details['STOCH'] = stoch_score
        
        # ADX
        adx_slope = self.calculate_slope(data['ADX'], dynamic_period)
        adx_score = 0
        if latest['ADX'] > 20:
            if latest['Plus_DI'] > latest['Minus_DI']: adx_score += 1
            else: adx_score -= 1
            if adx_slope > 0.5: adx_score += 2
        score += adx_score
        details['ADX'] = adx_score
        
        # 볼린저 밴드
        bb_score = 0
        if latest['BB_Percent'] < 0.2: bb_score += 2
        elif latest['BB_Percent'] > 0.8: bb_score -= 2
        score += bb_score
        details['BB'] = bb_score
        
        # 추천
        if score >= 7:
            recommendation = "강력 매수"
        elif score >= 3:
            recommendation = "매수"
        elif score > -3:
            recommendation = "관망"
        elif score > -7:
            recommendation = "매도"
        else:
            recommendation = "강력 매도"
        
        return score, recommendation, dynamic_period, details
    
    def calculate_risk_metrics(self, data):
        """위험도 지표 계산"""
        returns = data['Close'].pct_change().dropna()
        
        return {
            'volatility': returns.std() * np.sqrt(252) * 100,
            'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
            'max_drawdown': ((data['Close'] / data['Close'].cummax() - 1).min()) * 100,
            'current_drawdown': ((data['Close'].iloc[-1] / data['Close'].max() - 1)) * 100,
        }
    
    def calculate_target_price(self, data, latest):
        """목표가 계산"""
        atr = latest['ATR']
        current_price = latest['Close']
        
        return {
            'conservative_buy': current_price - (atr * 1),
            'aggressive_buy': current_price - (atr * 2),
            'target_1': current_price + (atr * 1),
            'target_2': current_price + (atr * 2),
            'target_3': current_price + (atr * 3),
            'stop_loss': current_price - (atr * 1.5),
        }
    
    def create_plotly_chart(self, data, ticker):
        """Plotly 인터랙티브 차트 생성"""
        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.4, 0.15, 0.15, 0.15, 0.15],
            subplot_titles=(f'{ticker} 주가', 'RSI', 'MACD', 'Stochastic', 'ADX')
        )
        
        # 캔들스틱
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # 이동평균선
        colors = {'MA_5': 'blue', 'MA_20': 'green', 'MA_60': 'red', 'MA_120': 'purple'}
        for ma, color in colors.items():
            fig.add_trace(
                go.Scatter(x=data.index, y=data[ma], name=ma, line=dict(color=color, width=1)),
                row=1, col=1
            )
        
        # 볼린저 밴드
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_Upper'], name='BB Upper', 
                      line=dict(color='orange', dash='dash', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_Lower'], name='BB Lower',
                      line=dict(color='orange', dash='dash', width=1), fill='tonexty'),
            row=1, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI_Signal'], name='RSI Signal', 
                      line=dict(color='orange', dash='dash')),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
        
        # MACD
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='orange')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal', 
                      line=dict(color='gray', dash='dash')),
            row=3, col=1
        )
        fig.add_trace(
            go.Bar(x=data.index, y=data['MACD_Diff'], name='Histogram', marker_color='darkred'),
            row=3, col=1
        )
        
        # Stochastic
        fig.add_trace(
            go.Scatter(x=data.index, y=data['STOCH_K'], name='%K', line=dict(color='blue')),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['STOCH_D'], name='%D', 
                      line=dict(color='red', dash='dash')),
            row=4, col=1
        )
        fig.add_hline(y=80, line_dash="dot", line_color="red", row=4, col=1)
        fig.add_hline(y=20, line_dash="dot", line_color="green", row=4, col=1)
        
        # ADX
        fig.add_trace(
            go.Scatter(x=data.index, y=data['ADX'], name='ADX', line=dict(color='orange')),
            row=5, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Plus_DI'], name='+DI', 
                      line=dict(color='green', dash='dot')),
            row=5, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Minus_DI'], name='-DI', 
                      line=dict(color='red', dash='dot')),
            row=5, col=1
        )
        fig.add_hline(y=20, line_dash="dot", line_color="gray", row=5, col=1)
        
        fig.update_layout(
            height=1200,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )
        
        return fig
    
    def load_favorites(self):
        """즐겨찾기 로드"""
        if self.favorites_file.exists():
            with open(self.favorites_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_favorites(self, favorites):
        """즐겨찾기 저장"""
        with open(self.favorites_file, 'w') as f:
            json.dump(favorites, f, indent=2)
    
    def add_to_history(self, ticker, score, recommendation):
        """히스토리에 추가"""
        history = []
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                history = json.load(f)
        
        history.append({
            'ticker': ticker,
            'date': datetime.now().isoformat(),
            'score': score,
            'recommendation': recommendation
        })
        
        history = history[-50:]  # 최근 50개만
        
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)


def main():
    """메인 애플리케이션"""
    
    # 사이드바
    st.sidebar.title("📊 주식 분석 대시보드")
    st.sidebar.markdown("---")
    
    analyzer = StreamlitStockAnalyzer()
    
    # 페이지 선택
    page = st.sidebar.radio(
        "메뉴",
        ["🔍 단일 종목 분석", "📊 종목 비교", "⭐ 즐겨찾기", "📜 분석 히스토리"]
    )
    
    if page == "🔍 단일 종목 분석":
        show_single_analysis(analyzer)
    
    elif page == "📊 종목 비교":
        show_comparison(analyzer)
    
    elif page == "⭐ 즐겨찾기":
        show_favorites(analyzer)
    
    elif page == "📜 분석 히스토리":
        show_history(analyzer)


def show_single_analysis(analyzer):
    """단일 종목 분석 페이지"""
    st.title("🔍 단일 종목 분석")
    
    # 입력 섹션
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        ticker = st.text_input(
            "티커 입력",
            value="TSLA",
            placeholder="예: AAPL, TSLA, 005930.KS"
        ).upper()
    
    with col2:
        period = st.selectbox(
            "분석 기간",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3
        )
    
    with col3:
        num_days = st.number_input(
            "차트 표시 일수",
            min_value=30,
            max_value=500,
            value=180,
            step=30
        )
    
    if st.button("🔍 분석 시작", type="primary", use_container_width=True):
        with st.spinner(f"{ticker} 데이터를 가져오는 중..."):
            data = analyzer.get_stock_data(ticker, period)
        
        if data is None:
            st.error("❌ 데이터를 가져올 수 없습니다. 티커를 확인해주세요.")
            return
        
        with st.spinner("지표를 계산하는 중..."):
            data = analyzer.calculate_indicators(data)
        
        if data.empty:
            st.error("❌ 지표 계산 실패")
            return
        
        # 분석 수행
        score, recommendation, period_used, details = analyzer.score_hybrid(data)
        risk_metrics = analyzer.calculate_risk_metrics(data)
        latest = data.iloc[-1]
        previous = data.iloc[-2]
        targets = analyzer.calculate_target_price(data, latest)
        
        # 히스토리에 추가
        analyzer.add_to_history(ticker, score, recommendation)
        
        # 결과 표시
        st.markdown("---")
        
        # 가격 정보
        st.subheader("💰 가격 정보")
        
        price_change = latest['Close'] - previous['Close']
        price_change_pct = (price_change / previous['Close']) * 100
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "현재가",
                f"${latest['Close']:,.2f}",
                f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
            )
        
        with col2:
            st.metric("52주 최고", f"${data['Close'].tail(252).max():,.2f}")
        
        with col3:
            st.metric("52주 최저", f"${data['Close'].tail(252).min():,.2f}")
        
        with col4:
            st.metric("거래량", f"{latest['Volume']:,.0f}")
        
        with col5:
            # 추천 색상
            if "매수" in recommendation:
                delta_color = "normal"
            elif "매도" in recommendation:
                delta_color = "inverse"
            else:
                delta_color = "off"
            
            st.metric(
                "종합 점수",
                f"{score}",
                recommendation,
                delta_color=delta_color
            )
        
        st.markdown("---")
        
        # 탭으로 정보 구성
        tab1, tab2, tab3, tab4 = st.tabs(["📈 차트", "📊 지표", "🎯 목표가", "⚠️ 위험도"])
        
        with tab1:
            st.subheader(f"{ticker} 인터랙티브 차트")
            data_plot = data.tail(num_days)
            fig = analyzer.create_plotly_chart(data_plot, ticker)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("📊 기술적 지표 현황")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 모멘텀 지표")
                
                # RSI
                rsi_status = "과매도" if latest['RSI'] < 30 else "과매수" if latest['RSI'] > 70 else "중립"
                rsi_color = "🟢" if latest['RSI'] < 30 else "🔴" if latest['RSI'] > 70 else "🟡"
                st.markdown(f"**RSI (14):** {rsi_color} {latest['RSI']:.2f} - {rsi_status}")
                st.progress(latest['RSI'] / 100)
                
                # Stochastic
                stoch_status = "과매도" if latest['STOCH_K'] < 20 else "과매수" if latest['STOCH_K'] > 80 else "중립"
                stoch_color = "🟢" if latest['STOCH_K'] < 20 else "🔴" if latest['STOCH_K'] > 80 else "🟡"
                st.markdown(f"**Stochastic:** {stoch_color} %K={latest['STOCH_K']:.2f} - {stoch_status}")
                st.progress(latest['STOCH_K'] / 100)
                
                st.markdown("---")
                st.markdown("##### 추세 지표")
                
                # MACD
                macd_signal = "상승" if latest['MACD'] > latest['MACD_Signal'] else "하락"
                macd_color = "🟢" if latest['MACD'] > latest['MACD_Signal'] else "🔴"
                st.markdown(f"**MACD:** {macd_color} {macd_signal}")
                st.markdown(f"- MACD: {latest['MACD']:.2f}")
                st.markdown(f"- Signal: {latest['MACD_Signal']:.2f}")
                st.markdown(f"- Histogram: {latest['MACD_Diff']:.2f}")
                
                # ADX
                adx_strength = "강함" if latest['ADX'] > 25 else "보통" if latest['ADX'] > 20 else "약함"
                trend_dir = "상승" if latest['Plus_DI'] > latest['Minus_DI'] else "하락"
                st.markdown(f"**ADX:** {latest['ADX']:.2f} - 추세 {adx_strength}, 방향 {trend_dir}")
            
            with col2:
                st.markdown("##### 변동성 지표")
                
                # 볼린저 밴드
                bb_position = "과매도" if latest['BB_Percent'] < 0.2 else "과매수" if latest['BB_Percent'] > 0.8 else "중립"
                bb_color = "🟢" if latest['BB_Percent'] < 0.2 else "🔴" if latest['BB_Percent'] > 0.8 else "🟡"
                st.markdown(f"**볼린저 밴드:** {bb_color} %B={latest['BB_Percent']:.2f} - {bb_position}")
                st.markdown(f"- Upper: ${latest['BB_Upper']:.2f}")
                st.markdown(f"- Middle: ${latest['BB_MA']:.2f}")
                st.markdown(f"- Lower: ${latest['BB_Lower']:.2f}")
                
                st.markdown("---")
                st.markdown("##### 이동평균선")
                
                ma_trend = "정배열" if (latest['MA_5'] > latest['MA_20'] > latest['MA_60']) else "역배열"
                ma_color = "🟢" if ma_trend == "정배열" else "🔴"
                st.markdown(f"**배열:** {ma_color} {ma_trend}")
                st.markdown(f"- MA 5: ${latest['MA_5']:.2f}")
                st.markdown(f"- MA 20: ${latest['MA_20']:.2f}")
                st.markdown(f"- MA 60: ${latest['MA_60']:.2f}")
                st.markdown(f"- MA 120: ${latest['MA_120']:.2f}")
            
            st.markdown("---")
            st.markdown("##### 지표별 점수")
            
            score_cols = st.columns(len(details))
            for i, (indicator, ind_score) in enumerate(details.items()):
                with score_cols[i]:
                    color = "🟢" if ind_score > 0 else "🔴" if ind_score < 0 else "🟡"
                    st.metric(indicator, f"{color} {ind_score:+d}")
        
        with tab3:
            st.subheader("🎯 목표가 설정 (ATR 기반)")
            
            st.markdown(f"**ATR (14일):** ${latest['ATR']:.2f}")
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 🟢 매수 진입가")
                st.metric("보수적 매수가", f"${targets['conservative_buy']:.2f}", 
                         f"{((targets['conservative_buy']/latest['Close']-1)*100):.1f}%")
                st.metric("공격적 매수가", f"${targets['aggressive_buy']:.2f}",
                         f"{((targets['aggressive_buy']/latest['Close']-1)*100):.1f}%")
                
                st.markdown("##### 🔴 손절가")
                st.metric("손절가", f"${targets['stop_loss']:.2f}",
                         f"{((targets['stop_loss']/latest['Close']-1)*100):.1f}%")
            
            with col2:
                st.markdown("##### 🎯 목표가")
                for i in range(1, 4):
                    target_key = f'target_{i}'
                    gain_pct = ((targets[target_key]/latest['Close']-1)*100)
                    st.metric(f"목표가 {i}", f"${targets[target_key]:.2f}", f"+{gain_pct:.1f}%")
        
        with tab4:
            st.subheader("⚠️ 위험도 분석")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("연간 변동성", f"{risk_metrics['volatility']:.2f}%")
            
            with col2:
                st.metric("샤프 비율", f"{risk_metrics['sharpe_ratio']:.2f}")
            
            with col3:
                st.metric("최대 낙폭", f"{risk_metrics['max_drawdown']:.2f}%")
            
            with col4:
                st.metric("현재 낙폭", f"{risk_metrics['current_drawdown']:.2f}%")
            
            st.markdown("---")
            
            # 위험도 해석
            st.markdown("##### 위험도 해석")
            
            vol_level = "매우 높음" if risk_metrics['volatility'] > 60 else \
                       "높음" if risk_metrics['volatility'] > 40 else \
                       "보통" if risk_metrics['volatility'] > 20 else "낮음"
            
            sharpe_level = "우수" if risk_metrics['sharpe_ratio'] > 1.5 else \
                          "양호" if risk_metrics['sharpe_ratio'] > 1.0 else \
                          "보통" if risk_metrics['sharpe_ratio'] > 0.5 else "미흡"
            
            st.markdown(f"- **변동성 수준:** {vol_level}")
            st.markdown(f"- **수익성 평가:** {sharpe_level}")
            st.markdown(f"- **최대 손실 경험:** 고점 대비 {abs(risk_metrics['max_drawdown']):.1f}% 하락")
            st.markdown(f"- **현재 위치:** 고점 대비 {abs(risk_metrics['current_drawdown']):.1f}% 하락")


def show_comparison(analyzer):
    """종목 비교 페이지"""
    st.title("📊 종목 비교 분석")
    
    # 티커 입력
    tickers_input = st.text_input(
        "비교할 티커들을 쉼표로 구분하여 입력하세요",
        value="AAPL,MSFT,GOOGL",
        placeholder="예: AAPL,MSFT,GOOGL,TSLA"
    )
    
    period = st.selectbox("분석 기간", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
    
    if st.button("📊 비교 분석", type="primary"):
        tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
        
        if not tickers:
            st.warning("티커를 입력해주세요.")
            return
        
        results = []
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(tickers):
            with st.spinner(f"{ticker} 분석 중... ({i+1}/{len(tickers)})"):
                data = analyzer.get_stock_data(ticker, period)
                
                if data is not None:
                    data = analyzer.calculate_indicators(data)
                    if not data.empty:
                        score, rec, _, details = analyzer.score_hybrid(data)
                        risk = analyzer.calculate_risk_metrics(data)
                        latest = data.iloc[-1]
                        
                        results.append({
                            'ticker': ticker,
                            'price': latest['Close'],
                            'score': score,
                            'recommendation': rec,
                            'rsi': latest['RSI'],
                            'volatility': risk['volatility'],
                            'sharpe': risk['sharpe_ratio'],
                            'max_dd': risk['max_drawdown'],
                            'data': data
                        })
                
                progress_bar.progress((i + 1) / len(tickers))
        
        progress_bar.empty()
        
        if not results:
            st.error("분석 가능한 종목이 없습니다.")
            return
        
        # 비교 테이블
        st.subheader("📋 종목 비교 요약")
        
        df_comparison = pd.DataFrame([{
            '티커': r['ticker'],
            '현재가': f"${r['price']:,.2f}",
            '점수': r['score'],
            '추천': r['recommendation'],
            'RSI': f"{r['rsi']:.1f}",
            '변동성': f"{r['volatility']:.1f}%",
            '샤프비율': f"{r['sharpe']:.2f}",
            '최대낙폭': f"{r['max_dd']:.1f}%"
        } for r in results])
        
        df_comparison = df_comparison.sort_values('점수', ascending=False)
        st.dataframe(df_comparison, use_container_width=True)
        
        # 비교 차트
        st.markdown("---")
        st.subheader("📊 비교 차트")
        
        tab1, tab2, tab3, tab4 = st.tabs(["점수 비교", "위험/수익", "가격 추이", "RSI 비교"])
        
        with tab1:
            # 점수 비교 막대 그래프
            fig = go.Figure()
            colors = ['green' if r['score'] >= 3 else 'red' if r['score'] <= -3 else 'orange' 
                     for r in results]
            
            fig.add_trace(go.Bar(
                x=[r['ticker'] for r in results],
                y=[r['score'] for r in results],
                marker_color=colors,
                text=[r['score'] for r in results],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="종합 점수 비교",
                xaxis_title="티커",
                yaxis_title="점수",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # 위험도 vs 수익성 산점도
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=[r['volatility'] for r in results],
                y=[r['sharpe'] for r in results],
                mode='markers+text',
                text=[r['ticker'] for r in results],
                textposition='top center',
                marker=dict(
                    size=15,
                    color=[r['score'] for r in results],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="점수")
                )
            ))
            
            fig.update_layout(
                title="위험도 vs 수익성",
                xaxis_title="연간 변동성 (%)",
                yaxis_title="샤프 비율",
                height=500
            )
            
            fig.add_hline(y=0, line_dash="dot", line_color="gray")
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # 정규화된 가격 추이
            fig = go.Figure()
            
            for r in results:
                data_60 = r['data'].tail(60)
                normalized = (data_60['Close'] / data_60['Close'].iloc[0] - 1) * 100
                fig.add_trace(go.Scatter(
                    x=data_60.index,
                    y=normalized,
                    name=r['ticker'],
                    mode='lines'
                ))
            
            fig.update_layout(
                title="최근 60일 가격 변화 (정규화)",
                xaxis_title="날짜",
                yaxis_title="변화율 (%)",
                height=500,
                hovermode='x unified'
            )
            
            fig.add_hline(y=0, line_dash="dot", line_color="gray")
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # RSI 비교
            fig = go.Figure()
            
            colors_rsi = ['green' if r['rsi'] < 30 else 'red' if r['rsi'] > 70 else 'orange' 
                         for r in results]
            
            fig.add_trace(go.Bar(
                x=[r['ticker'] for r in results],
                y=[r['rsi'] for r in results],
                marker_color=colors_rsi,
                text=[f"{r['rsi']:.1f}" for r in results],
                textposition='outside'
            ))
            
            fig.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="과매수")
            fig.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="과매도")
            
            fig.update_layout(
                title="RSI 비교",
                xaxis_title="티커",
                yaxis_title="RSI",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)


def show_favorites(analyzer):
    """즐겨찾기 페이지"""
    st.title("⭐ 즐겨찾기 관리")
    
    favorites = analyzer.load_favorites()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        new_ticker = st.text_input("즐겨찾기에 추가할 티커", placeholder="예: AAPL")
    
    with col2:
        st.write("")  # 간격
        st.write("")  # 간격
        if st.button("➕ 추가", use_container_width=True):
            if new_ticker:
                ticker = new_ticker.upper().strip()
                if ticker not in favorites:
                    favorites.append(ticker)
                    analyzer.save_favorites(favorites)
                    st.success(f"✅ {ticker}를 즐겨찾기에 추가했습니다!")
                    st.rerun()
                else:
                    st.warning(f"⚠️ {ticker}는 이미 즐겨찾기에 있습니다.")
    
    st.markdown("---")
    
    if favorites:
        st.subheader("📋 즐겨찾기 목록")
        
        for i, ticker in enumerate(favorites):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"### {ticker}")
            
            with col2:
                if st.button("🔍 분석", key=f"analyze_{i}"):
                    st.session_state['analyze_ticker'] = ticker
                    st.rerun()
            
            with col3:
                if st.button("🗑️ 삭제", key=f"delete_{i}"):
                    favorites.remove(ticker)
                    analyzer.save_favorites(favorites)
                    st.success(f"✅ {ticker}를 삭제했습니다!")
                    st.rerun()
        
        st.markdown("---")
        
        if st.button("📊 전체 즐겨찾기 비교 분석", type="primary"):
            st.session_state['compare_favorites'] = True
            st.rerun()
    else:
        st.info("즐겨찾기가 비어있습니다. 위에서 티커를 추가해주세요.")
    
    # 세션 상태 처리
    if 'analyze_ticker' in st.session_state:
        ticker = st.session_state['analyze_ticker']
        del st.session_state['analyze_ticker']
        
        with st.spinner(f"{ticker} 분석 중..."):
            data = analyzer.get_stock_data(ticker, '1y')
            if data is not None:
                data = analyzer.calculate_indicators(data)
                # 간단한 요약만 표시
                if not data.empty:
                    score, rec, _, _ = analyzer.score_hybrid(data)
                    latest = data.iloc[-1]
                    st.success(f"**{ticker}** - 현재가: ${latest['Close']:.2f} | 점수: {score} | {rec}")


def show_history(analyzer):
    """분석 히스토리 페이지"""
    st.title("📜 분석 히스토리")
    
    if not analyzer.history_file.exists():
        st.info("아직 분석 히스토리가 없습니다.")
        return
    
    with open(analyzer.history_file, 'r') as f:
        history = json.load(f)
    
    if not history:
        st.info("아직 분석 히스토리가 없습니다.")
        return
    
    # 최근 순으로 정렬
    history = list(reversed(history))
    
    # 필터
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticker_filter = st.text_input("티커로 필터", placeholder="예: TSLA")
    
    with col2:
        limit = st.number_input("표시 개수", min_value=5, max_value=50, value=20)
    
    # 필터링
    if ticker_filter:
        filtered_history = [h for h in history if ticker_filter.upper() in h['ticker']]
    else:
        filtered_history = history
    
    filtered_history = filtered_history[:limit]
    
    # 표시
    st.markdown("---")
    
    for record in filtered_history:
        date = datetime.fromisoformat(record['date']).strftime('%Y-%m-%d %H:%M:%S')
        
        col1, col2, col3, col4 = st.columns([2, 2, 1, 2])
        
        with col1:
            st.markdown(f"**{record['ticker']}**")
        
        with col2:
            st.markdown(f"📅 {date}")
        
        with col3:
            score_color = "🟢" if record['score'] >= 3 else "🔴" if record['score'] <= -3 else "🟡"
            st.markdown(f"{score_color} **{record['score']}**")
        
        with col4:
            st.markdown(f"_{record['recommendation']}_")
        
        st.markdown("---")


if __name__ == "__main__":
    main()
