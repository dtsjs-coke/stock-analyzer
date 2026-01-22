"""
주식 분석 Streamlit 웹 애플리케이션 v2.5 Enhanced
✅ 지표 시각화 대폭 강화 (게이지 차트, 미니 차트)
✅ 사이드바 유저 가이드 추가
✅ 추가 기능: 알림 설정, 메모, 비교 분석
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import linregress
from datetime import datetime
import json
from pathlib import Path
import FinanceDataReader as fdr

st.set_page_config(page_title="📈 주식 분석 대시보드", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
    div[data-testid="stMetricValue"] { font-size: 28px; }
    .indicator-box { 
        background-color: #f8f9fa; 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 4px solid #4CAF50;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data(ttl=86400)
def load_kr_stocks_cached():
    """한국 주식 로드"""
    try:
        with st.spinner("🇰🇷 한국 주식 데이터 로딩 중..."):
            # krx = fdr.StockListing('KRX')
            print('before_loading_krx')
            krx_kospi = fdr.StockListing('KOSPI')
            krx_kosdaq = fdr.StockListing('KOSDAQ')
            # krx_konex = fdr.StockListing('KONEX')
            # krx_kosd_glb = fdr.StockListing('KOSDAQ GLOBAL')
            krx = pd.concat([krx_kospi,krx_kosdaq])
            print('after_loading_krx')

            def add_suffix(row):
                if row['Market'] == 'KOSPI': return f"{row['Code']}.KS"
                elif row['Market'] == 'KOSDAQ': return f"{row['Code']}.KQ"
                # elif row['Market'] == 'KONEX': return f"{row['Code']}.KN"
                else: return f"{row['Code']}.KS"

            krx['Ticker'] = krx.apply(add_suffix, axis=1)
            kr_stock_names = dict(zip(krx['Ticker'], krx['Name']))

            kr_name_to_tickers = {}
            print('before for문')
            for ticker, name in kr_stock_names.items():
                if name not in kr_name_to_tickers:
                    kr_name_to_tickers[name] = []
                kr_name_to_tickers[name].append(ticker)

                for i in range(2, len(name) + 1):
                    partial_name = name[:i]
                    if partial_name not in kr_name_to_tickers:
                        kr_name_to_tickers[partial_name] = []
                    if ticker not in kr_name_to_tickers[partial_name]:
                        kr_name_to_tickers[partial_name].append(ticker)
            print('after for문')
            total = len(krx)
            kospi = len(krx[krx['Market'] == 'KOSPI'])
            kosdaq = len(krx[krx['Market'] == 'KOSDAQ'])
            print('after counting')

            st.success(f"✅ 한국 주식: {total:,}개 (코스피 {kospi:,}, 코스닥 {kosdaq:,})")
            return {'names': kr_stock_names, 'index': kr_name_to_tickers, 'total': total}
    except Exception as e:
        st.error(f"❌ 한국 주식 로드 실패: {e}")
        fallback = {"005930.KS": "삼성전자", "000660.KS": "SK하이닉스"}
        return {'names': fallback, 'index': {}, 'total': len(fallback)}


@st.cache_data(ttl=86400)
def load_us_stocks_cached():
    """미국 주식 로드"""
    try:
        with st.spinner("🇺🇸 미국 주식 데이터 로딩 중..."):
            df_nasdaq = fdr.StockListing('NASDAQ')
            df_nyse = fdr.StockListing('NYSE')
            df_amex = fdr.StockListing('AMEX')

            df_us = pd.concat([df_nasdaq, df_nyse, df_amex], ignore_index=True)
            df_us = df_us[['Symbol', 'Name']].drop_duplicates()
            us_names = dict(zip(df_us['Symbol'], df_us['Name']))

            total = len(df_us)
            nasdaq = len(df_nasdaq)
            nyse = len(df_nyse)
            amex = len(df_amex)

            st.success(f"✅ 미국 주식: {total:,}개 (NASDAQ {nasdaq:,}, NYSE {nyse:,}, AMEX {amex:,})")
            return {'df': df_us, 'names': us_names, 'total': total}
    except Exception as e:
        st.error(f"❌ 미국 주식 로드 실패: {e}")
        return {'df': pd.DataFrame(columns=['Symbol', 'Name']), 'names': {}, 'total': 0}


class StreamlitStockAnalyzer:
    def __init__(self):
        self.data_dir = Path("streamlit_data")
        self.data_dir.mkdir(exist_ok=True)
        self.favorites_file = self.data_dir / "favorites.json"
        self.history_file = self.data_dir / "history.json"
        self.notes_file = self.data_dir / "notes.json"
        self.alerts_file = self.data_dir / "alerts.json"

        kr_data = load_kr_stocks_cached()
        self.kr_stock_names = kr_data['names']
        self.kr_name_to_tickers = kr_data['index']
        self.kr_total = kr_data['total']

        us_data = load_us_stocks_cached()
        self.us_stock_df = us_data['df']
        self.us_stock_names = us_data['names']
        self.us_total = us_data['total']

    def get_stock_name(self, ticker):
        ticker_upper = ticker.upper()
        if ticker_upper in self.kr_stock_names:
            return self.kr_stock_names[ticker_upper]
        if ticker_upper in self.us_stock_names:
            return self.us_stock_names[ticker_upper]
        return ticker_upper

    def search_kr_stock(self, query, max_results=20):
        query = query.strip()
        results, seen = [], set()

        ticker_query = query.upper()
        if ticker_query in self.kr_stock_names:
            results.append({'ticker': ticker_query, 'name': self.kr_stock_names[ticker_query]})
            seen.add(ticker_query)

        if query.isdigit() and len(query) == 6:
            for suffix in ['.KS', '.KQ', '.KN']:
                full_ticker = query + suffix
                if full_ticker in self.kr_stock_names and full_ticker not in seen:
                    results.append({'ticker': full_ticker, 'name': self.kr_stock_names[full_ticker]})
                    seen.add(full_ticker)

        if query in self.kr_name_to_tickers:
            for ticker in self.kr_name_to_tickers[query]:
                if ticker not in seen:
                    results.append({'ticker': ticker, 'name': self.kr_stock_names[ticker]})
                    seen.add(ticker)

        results.sort(key=lambda x: (0 if query == x['name'] else 1 if x['name'].startswith(query) else 2, x['name']))
        return results[:max_results]

    def search_us_stock(self, query):
        if not query or self.us_stock_df.empty:
            return []
        query = query.upper().strip()
        mask = (self.us_stock_df['Symbol'].str.startswith(query, na=False)) | \
               (self.us_stock_df['Name'].str.contains(query, case=False, na=False))
        filtered = self.us_stock_df[mask].head(12)
        return [{'ticker': row['Symbol'], 'name': row['Name']} for _, row in filtered.iterrows()]

    def detect_currency(self, ticker):
        t = ticker.upper()
        if t.endswith(('.KS', '.KQ')): return 'KRW'
        elif t.endswith('.T'): return 'JPY'
        elif t.endswith('.HK'): return 'HKD'
        else: return 'USD'

    def format_currency(self, value, ticker):
        curr = self.detect_currency(ticker)
        if curr == 'KRW': return f"₩{value:,.0f}"
        elif curr == 'JPY': return f"¥{value:,.0f}"
        elif curr == 'HKD': return f"HK${value:,.2f}"
        else: return f"${value:,.2f}"

    @st.cache_data(ttl=3600)
    def get_stock_data(_self, ticker, period='1y'):
        try:
            data = yf.download(ticker.strip().upper(), period=period, progress=False)
            if data.empty: return None
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            data = data.loc[:, ~data.columns.duplicated()].copy()
            if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                return None
            if 'Adj Close' not in data.columns:
                data['Adj Close'] = data['Close']
            for col in data.columns:
                if col != 'Volume':
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce').fillna(0)
            data.dropna(subset=['Close'], inplace=True)
            return data if len(data) >= 60 else None
        except:
            return None

    def calculate_indicators(self, data):
        if len(data) < 120: return data

        close = data['Close'].astype(float)
        high = data['High'].astype(float)
        low = data['Low'].astype(float)
        volume = data['Volume'].astype(float)

        data['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
        data['RSI_Signal'] = data['RSI'].rolling(window=9).mean()

        macd = ta.trend.MACD(close)
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Diff'] = macd.macd_diff()

        for w, col in [(5, 'MA_5'), (20, 'MA_20'), (60, 'MA_60'), (120, 'MA_120')]:
            data[col] = ta.trend.SMAIndicator(close, window=w).sma_indicator()

        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        data['BB_MA'] = bb.bollinger_mavg()
        data['BB_Upper'] = bb.bollinger_hband()
        data['BB_Lower'] = bb.bollinger_lband()
        data['BB_Width'] = bb.bollinger_wband()
        data['BB_Percent'] = bb.bollinger_pband()

        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        data['STOCH_K'] = stoch.stoch()
        data['STOCH_D'] = stoch.stoch_signal()

        adx = ta.trend.ADXIndicator(high, low, close, window=14)
        data['ADX'] = adx.adx()
        data['Plus_DI'] = adx.adx_pos()
        data['Minus_DI'] = adx.adx_neg()

        data['ATR'] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
        data['OBV'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()

        data.dropna(inplace=True)
        return data

    def calculate_slope(self, series, period):
        if len(series) < period: return 0
        y = series.tail(period).values
        x = np.arange(len(y))
        if not np.all(np.isfinite(y)): return 0
        slope, *_ = linregress(x, y)
        return slope if np.isfinite(slope) else 0

    def score_hybrid(self, data):
        if len(data) < 10: return 0, "데이터 부족", 0, {}, {}

        period = int(max(3, min(10, len(data) * 0.05)))
        latest = data.iloc[-1]
        raw, weighted = {}, {}
        weights = {'MACD': 0.25, 'MA': 0.20, 'RSI': 0.20, 'BB': 0.15, 'STOCH': 0.12, 'ADX': 0.08}

        # RSI
        rsi_slope = self.calculate_slope(data['RSI'], period)
        rsi_score = 0
        if latest['RSI'] < 20: rsi_score += 5
        elif latest['RSI'] < 30: rsi_score += 3
        elif latest['RSI'] < 40: rsi_score += 1
        elif latest['RSI'] > 80: rsi_score -= 5
        elif latest['RSI'] > 70: rsi_score -= 3
        elif latest['RSI'] > 60: rsi_score -= 1

        if rsi_slope > 2: rsi_score += 5
        elif rsi_slope > 1.5: rsi_score += 3
        elif rsi_slope > 0.5: rsi_score += 1
        elif rsi_slope < -2: rsi_score -= 5
        elif rsi_slope < -1.5: rsi_score -= 3
        elif rsi_slope < -0.5: rsi_score -= 1

        rsi_score = max(-10, min(10, rsi_score))
        raw['RSI'] = rsi_score
        weighted['RSI'] = rsi_score * weights['RSI']

        # MACD
        hist_slope = self.calculate_slope(data['MACD_Diff'], period)
        macd_score = 0
        if latest['MACD'] > latest['MACD_Signal']:
            macd_score += 5 if latest['MACD_Diff'] > 0 else 3
        else:
            macd_score -= 5 if latest['MACD_Diff'] < 0 else 3

        if hist_slope > 0.2: macd_score += 5
        elif hist_slope > 0.1: macd_score += 3
        elif hist_slope > 0: macd_score += 1
        elif hist_slope < -0.2: macd_score -= 5
        elif hist_slope < -0.1: macd_score -= 3
        elif hist_slope < 0: macd_score -= 1

        macd_score = max(-10, min(10, macd_score))
        raw['MACD'] = macd_score
        weighted['MACD'] = macd_score * weights['MACD']

        # MA
        ma_slope = self.calculate_slope(data['MA_5'] - data['MA_20'], period)
        ma_score = 0
        if latest['MA_5'] > latest['MA_20'] > latest['MA_60'] > latest['MA_120']: ma_score += 5
        elif latest['MA_5'] > latest['MA_20'] > latest['MA_60']: ma_score += 4
        elif latest['MA_5'] > latest['MA_20']: ma_score += 2
        elif latest['MA_5'] < latest['MA_20'] < latest['MA_60'] < latest['MA_120']: ma_score -= 5
        elif latest['MA_5'] < latest['MA_20'] < latest['MA_60']: ma_score -= 4
        elif latest['MA_5'] < latest['MA_20']: ma_score -= 2

        if ma_slope > 1: ma_score += 5
        elif ma_slope > 0.5: ma_score += 3
        elif ma_slope > 0: ma_score += 1
        elif ma_slope < -1: ma_score -= 5
        elif ma_slope < -0.5: ma_score -= 3
        elif ma_slope < 0: ma_score -= 1

        ma_score = max(-10, min(10, ma_score))
        raw['MA'] = ma_score
        weighted['MA'] = ma_score * weights['MA']

        # BB, STOCH, ADX
        bb_score = 5 if latest['BB_Percent'] < 0.1 else -5 if latest['BB_Percent'] > 0.9 else 0
        raw['BB'] = max(-10, min(10, bb_score))
        weighted['BB'] = bb_score * weights['BB']

        stoch_score = 5 if latest['STOCH_K'] < 10 else -5 if latest['STOCH_K'] > 90 else 0
        raw['STOCH'] = max(-10, min(10, stoch_score))
        weighted['STOCH'] = stoch_score * weights['STOCH']

        strength = 5 if latest['ADX'] > 40 else 3 if latest['ADX'] > 25 else 0
        adx_score = strength if latest['Plus_DI'] > latest['Minus_DI'] else -strength
        raw['ADX'] = max(-10, min(10, adx_score))
        weighted['ADX'] = adx_score * weights['ADX']

        final = round(sum(weighted.values()) * 10, 1)
        rec = ("강력 매수" if final >= 6 else "매수" if final >= 3 else "약한 매수" if final >= 1 else
               "관망" if final > -1 else "약한 매도" if final > -3 else "매도" if final > -6 else "강력 매도")

        return final, rec, period, raw, weights

    def calculate_risk_metrics(self, data):
        returns = data['Close'].pct_change().dropna()
        return {
            'volatility': returns.std() * np.sqrt(252) * 100,
            'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
            'max_drawdown': ((data['Close'] / data['Close'].cummax() - 1).min()) * 100,
            'current_drawdown': ((data['Close'].iloc[-1] / data['Close'].max() - 1)) * 100,
        }

    def calculate_target_price(self, data, latest):
        atr = latest['ATR']
        price = latest['Close']
        return {
            'conservative_buy': price - atr,
            'aggressive_buy': price - atr * 2,
            'target_1': price + atr,
            'target_2': price + atr * 2,
            'target_3': price + atr * 3,
            'stop_loss': price - atr * 1.5
        }

    def create_gauge_chart(self, value, title, min_val=0, max_val=100,
                          thresholds=[30, 70], colors=['#26a69a', '#FFB74D', '#ef5350']):
        """게이지 차트 생성"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': title, 'font': {'size': 16}},
            gauge={
                'axis': {'range': [min_val, max_val], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [min_val, thresholds[0]], 'color': colors[0]},
                    {'range': [thresholds[0], thresholds[1]], 'color': colors[1]},
                    {'range': [thresholds[1], max_val], 'color': colors[2]}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': value
                }
            }
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        return fig

    def create_mini_trend_chart(self, data, column, title, color='#2962FF'):
        """미니 추세 차트"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index[-30:],
            y=data[column].tail(30),
            mode='lines',
            line=dict(color=color, width=2),
            fill='tozeroy',
            fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}'
        ))
        fig.update_layout(
            title=title,
            height=150,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig

    def create_plotly_chart(self, data, ticker):
        """6단 차트"""
        fig = make_subplots(
            rows=6, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.35, 0.13, 0.13, 0.13, 0.13, 0.13],
            subplot_titles=(f'{ticker} 주가', 'RSI', 'MACD', 'Stochastic', 'ADX', '거래량 & OBV')
        )

        # 1. 캔들스틱
        fig.add_trace(go.Candlestick(
            x=data.index, open=data['Open'], high=data['High'],
            low=data['Low'], close=data['Close'],
            name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
        ), row=1, col=1)

        for ma, name, color, width in [('MA_5', 'MA5', '#2962FF', 1.5), ('MA_20', 'MA20', '#FF6D00', 1.5),
                                       ('MA_60', 'MA60', '#D50000', 1.2), ('MA_120', 'MA120', '#AA00FF', 1.0)]:
            fig.add_trace(go.Scatter(x=data.index, y=data[ma], name=name,
                                    line=dict(color=color, width=width), opacity=0.8), row=1, col=1)

        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], name='BB Upper',
                                line=dict(color='rgba(255,152,0,0.5)', dash='dash'), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], name='BB Lower',
                                line=dict(color='rgba(255,152,0,0.5)', dash='dash'),
                                fill='tonexty', fillcolor='rgba(255,152,0,0.1)', showlegend=False), row=1, col=1)

        # 2. RSI
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI',
                                line=dict(color='#9C27B0', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI_Signal'], name='RSI Signal',
                                line=dict(color='#FF9800', dash='dash', width=1.5), opacity=0.7), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="rgba(239,83,80,0.6)", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="rgba(38,166,154,0.6)", row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1)

        # 3. MACD
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD',
                                line=dict(color='#FF6D00', width=2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal',
                                line=dict(color='#1976D2', dash='dash', width=1.5)), row=3, col=1)
        colors = ['#26a69a' if v > 0 else '#ef5350' for v in data['MACD_Diff']]
        fig.add_trace(go.Bar(x=data.index, y=data['MACD_Diff'], name='Histogram',
                            marker_color=colors, opacity=0.5), row=3, col=1)

        # 4. Stochastic
        fig.add_trace(go.Scatter(x=data.index, y=data['STOCH_K'], name='%K',
                                line=dict(color='#2196F3', width=2)), row=4, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['STOCH_D'], name='%D',
                                line=dict(color='#F44336', dash='dash', width=1.5)), row=4, col=1)
        fig.add_hline(y=80, line_dash="dot", line_color="rgba(239,83,80,0.6)", row=4, col=1)
        fig.add_hline(y=20, line_dash="dot", line_color="rgba(38,166,154,0.6)", row=4, col=1)
        fig.update_yaxes(range=[0, 100], row=4, col=1)

        # 5. ADX
        fig.add_trace(go.Scatter(x=data.index, y=data['ADX'], name='ADX',
                                line=dict(color='#FF6D00', width=2.5)), row=5, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['Plus_DI'], name='+DI',
                                line=dict(color='#4CAF50', dash='dot', width=1.5), opacity=0.7), row=5, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['Minus_DI'], name='-DI',
                                line=dict(color='#F44336', dash='dot', width=1.5), opacity=0.7), row=5, col=1)
        fig.update_yaxes(range=[0, 60], row=5, col=1)

        # 6. 거래량 + OBV
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='거래량',
                            marker_color='rgba(100,149,237,0.5)'), row=6, col=1)

        obv_normalized = (data['OBV'] - data['OBV'].min()) / (data['OBV'].max() - data['OBV'].min()) * data['Volume'].max()
        fig.add_trace(go.Scatter(x=data.index, y=obv_normalized, name='OBV',
                                line=dict(color='#FF6B6B', width=2)), row=6, col=1)

        fig.update_layout(
            height=1600,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        for i in range(1, 7):
            fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.2)', row=i, col=1)
            fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.2)', row=i, col=1)

        return fig

    def add_to_history(self, ticker, score, recommendation):
        history = []
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                history = json.load(f)
        history.append({'ticker': ticker, 'date': datetime.now().isoformat(), 'score': score, 'recommendation': recommendation})
        with open(self.history_file, 'w') as f:
            json.dump(history[-50:], f, indent=2)

    def save_note(self, ticker, note):
        """메모 저장"""
        notes = {}
        if self.notes_file.exists():
            with open(self.notes_file, 'r', encoding='utf-8') as f:
                notes = json.load(f)
        notes[ticker] = {'note': note, 'date': datetime.now().isoformat()}
        with open(self.notes_file, 'w', encoding='utf-8') as f:
            json.dump(notes, f, indent=2, ensure_ascii=False)

    def get_note(self, ticker):
        """메모 가져오기"""
        if self.notes_file.exists():
            with open(self.notes_file, 'r', encoding='utf-8') as f:
                notes = json.load(f)
                return notes.get(ticker, {}).get('note', '')
        return ''


def show_user_guide_sidebar():
    """사이드바 유저 가이드"""
    with st.sidebar.expander("📖 점수 배점 시스템", expanded=False):
        st.markdown("""
        ### 📊 지표별 가중치
        
        | 지표 | 가중치 | 역할 |
        |------|--------|------|
        | MACD | **25%** | 추세 전환 |
        | 이동평균 | **20%** | 추세 방향 |
        | RSI | **20%** | 과매수/도 |
        | 볼린저밴드 | **15%** | 변동성 |
        | 스토캐스틱 | **12%** | 모멘텀 |
        | ADX | **8%** | 추세 강도 |
        
        ### 📈 각 지표 점수 범위
        - 최소: **-10점**
        - 최대: **+10점**
        - 가중치 적용 후 합산
        """)

    with st.sidebar.expander("🎯 종합 점수 구간", expanded=False):
        st.markdown("""
        ### 점수별 추천 및 의미
        
        | 점수 | 추천 | 의미 |
        |------|------|------|
        | **+6 이상** | 🟢 강력 매수 | 매우 강한 상승 신호 |
        | **+3 ~ +6** | 🟢 매수 | 상승 신호 |
        | **+1 ~ +3** | 🟡 약한 매수 | 약한 상승 신호 |
        | **-1 ~ +1** | 🟡 관망 | 중립, 대기 |
        | **-3 ~ -1** | 🔴 약한 매도 | 약한 하락 신호 |
        | **-6 ~ -3** | 🔴 매도 | 하락 신호 |
        | **-6 이하** | 🔴 강력 매도 | 매우 강한 하락 신호 |
        
        ### 💡 활용 팁
        - **+5 이상**: 적극적 매수 고려
        - **-5 이하**: 매도 또는 관망
        - **-3 ~ +3**: 신중한 판단 필요
        """)

    with st.sidebar.expander("📚 주요 지표 설명", expanded=False):
        st.markdown("""
        ### RSI (상대강도지수)
        - **30 이하**: 과매도 (매수 기회)
        - **70 이상**: 과매수 (매도 신호)
        
        ### MACD
        - **골든크로스**: MACD > Signal (매수)
        - **데드크로스**: MACD < Signal (매도)
        
        ### 이동평균선
        - **정배열**: 단기 > 장기 (상승 추세)
        - **역배열**: 단기 < 장기 (하락 추세)
        
        ### 볼린저밴드
        - **하단**: 과매도 구간
        - **상단**: 과매수 구간
        
        ### 스토캐스틱
        - **20 이하**: 과매도
        - **80 이상**: 과매수
        
        ### ADX (추세 강도)
        - **25 이상**: 강한 추세
        - **20 이하**: 약한 추세 (횡보)
        """)

    with st.sidebar.expander("⚠️ 투자 유의사항", expanded=False):
        st.markdown("""
        ### 🚫 주의사항
        
        1. **참고용 도구**
           - 투자 조언 아님
           - 수익 보장 없음
        
        2. **추가 분석 필수**
           - 펀더멘털 분석
           - 뉴스 및 공시 확인
           - 시장 환경 고려
        
        3. **리스크 관리**
           - 손절가 엄수
           - 분산 투자
           - 여유 자금만 투자
        
        4. **감정 배제**
           - 데이터 기반 판단
           - 규칙 준수
           - 매매 일지 작성
        """)


def run_analysis_enhanced(analyzer, ticker, period='1y', num_days=180):
    """강화된 분석 실행"""
    stock_name = analyzer.get_stock_name(ticker)
    if stock_name != ticker:
        st.info(f"📌 분석 종목: **{stock_name}** ({ticker})")

    with st.spinner(f"{ticker} 분석 중..."):
        data = analyzer.get_stock_data(ticker, period)

    if data is None:
        st.error("❌ 데이터를 가져올 수 없습니다.")
        return

    data = analyzer.calculate_indicators(data)
    if data.empty:
        st.error("❌ 지표 계산 실패")
        return

    score, rec, _, raw, weights = analyzer.score_hybrid(data)
    risk = analyzer.calculate_risk_metrics(data)
    latest = data.iloc[-1]
    previous = data.iloc[-2]
    targets = analyzer.calculate_target_price(data, latest)
    analyzer.add_to_history(ticker, score, rec)

    # 헤더
    st.markdown("---")
    st.title(f"📊 {stock_name} ({ticker})" if stock_name != ticker else f"📊 {ticker}")

    # 가격 정보
    st.subheader("💰 가격 정보")
    change = latest['Close'] - previous['Close']
    pct = (change / previous['Close']) * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("현재가", analyzer.format_currency(latest['Close'], ticker), f"{change:+,.2f} ({pct:+.2f}%)")
    col2.metric("52주 최고", analyzer.format_currency(data['Close'].tail(252).max(), ticker))
    col3.metric("52주 최저", analyzer.format_currency(data['Close'].tail(252).min(), ticker))
    col4.metric("거래량", f"{latest['Volume']:,.0f}")
    col5.metric("종합 점수", f"{score:.1f}", rec,
               delta_color="normal" if "매수" in rec else "inverse" if "매도" in rec else "off")

    st.markdown("---")

    # 5개 탭
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 차트", "📊 지표", "🎯 목표가", "⚠️ 위험도", "📝 메모 & 알림"])

    with tab1:
        st.subheader(f"📈 {stock_name} ({ticker}) 인터랙티브 차트" if stock_name != ticker else f"📈 {ticker} 인터랙티브 차트")
        data_plot = data.tail(num_days)
        fig = analyzer.create_plotly_chart(data_plot, ticker)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("📊 기술적 지표 현황 (시각화)")

        # 게이지 차트 3개
        col1, col2, col3 = st.columns(3)

        with col1:
            st.plotly_chart(analyzer.create_gauge_chart(
                latest['RSI'], "RSI", 0, 100, [30, 70],
                ['#26a69a', '#FFB74D', '#ef5350']
            ), use_container_width=True)

        with col2:
            st.plotly_chart(analyzer.create_gauge_chart(
                latest['STOCH_K'], "Stochastic %K", 0, 100, [20, 80],
                ['#26a69a', '#FFB74D', '#ef5350']
            ), use_container_width=True)

        with col3:
            st.plotly_chart(analyzer.create_gauge_chart(
                latest['BB_Percent'] * 100, "볼린저밴드 %B", 0, 100, [20, 80],
                ['#26a69a', '#FFB74D', '#ef5350']
            ), use_container_width=True)

        st.markdown("---")

        # 미니 차트 2개
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(analyzer.create_mini_trend_chart(
                data, 'RSI', 'RSI 30일 추세', '#9C27B0'
            ), use_container_width=True)

            st.plotly_chart(analyzer.create_mini_trend_chart(
                data, 'MACD_Diff', 'MACD 히스토그램 30일 추세', '#FF6D00'
            ), use_container_width=True)

        with col2:
            st.plotly_chart(analyzer.create_mini_trend_chart(
                data, 'STOCH_K', 'Stochastic 30일 추세', '#2196F3'
            ), use_container_width=True)

            st.plotly_chart(analyzer.create_mini_trend_chart(
                data, 'ADX', 'ADX 30일 추세', '#FF6D00'
            ), use_container_width=True)

        st.markdown("---")

        # 상세 지표 정보
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📈 추세 & 모멘텀")

            # MACD
            macd_signal = "상승" if latest['MACD'] > latest['MACD_Signal'] else "하락"
            macd_color = "🟢" if latest['MACD'] > latest['MACD_Signal'] else "🔴"
            st.markdown(f"""
            <div class="indicator-box">
            <b>MACD</b> {macd_color} {macd_signal}<br>
            • MACD: {latest['MACD']:.3f}<br>
            • Signal: {latest['MACD_Signal']:.3f}<br>
            • Histogram: {latest['MACD_Diff']:.3f}<br>
            • 원점수: <b>{raw['MACD']}</b> / 가중점수: <b>{raw['MACD'] * weights['MACD']:.2f}</b>
            </div>
            """, unsafe_allow_html=True)

            # 이동평균
            ma_trend = "정배열" if (latest['MA_5'] > latest['MA_20'] > latest['MA_60']) else "역배열"
            ma_color = "🟢" if ma_trend == "정배열" else "🔴"
            st.markdown(f"""
            <div class="indicator-box">
            <b>이동평균선</b> {ma_color} {ma_trend}<br>
            • MA5: {analyzer.format_currency(latest['MA_5'], ticker)}<br>
            • MA20: {analyzer.format_currency(latest['MA_20'], ticker)}<br>
            • MA60: {analyzer.format_currency(latest['MA_60'], ticker)}<br>
            • 원점수: <b>{raw['MA']}</b> / 가중점수: <b>{raw['MA'] * weights['MA']:.2f}</b>
            </div>
            """, unsafe_allow_html=True)

            # ADX
            adx_strength = "매우 강함" if latest['ADX'] > 40 else "강함" if latest['ADX'] > 25 else "보통"
            trend_dir = "상승" if latest['Plus_DI'] > latest['Minus_DI'] else "하락"
            st.markdown(f"""
            <div class="indicator-box">
            <b>ADX (추세 강도)</b><br>
            • ADX: {latest['ADX']:.2f} - {adx_strength}<br>
            • 방향: {trend_dir}<br>
            • +DI: {latest['Plus_DI']:.2f} / -DI: {latest['Minus_DI']:.2f}<br>
            • 원점수: <b>{raw['ADX']}</b> / 가중점수: <b>{raw['ADX'] * weights['ADX']:.2f}</b>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("### 📊 변동성 & 모멘텀")

            # RSI
            rsi_status = "과매도" if latest['RSI'] < 30 else "과매수" if latest['RSI'] > 70 else "중립"
            rsi_color = "🟢" if latest['RSI'] < 30 else "🔴" if latest['RSI'] > 70 else "🟡"
            st.markdown(f"""
            <div class="indicator-box">
            <b>RSI</b> {rsi_color} {rsi_status}<br>
            • RSI: {latest['RSI']:.2f}<br>
            • Signal: {latest['RSI_Signal']:.2f}<br>
            • 원점수: <b>{raw['RSI']}</b> / 가중점수: <b>{raw['RSI'] * weights['RSI']:.2f}</b>
            </div>
            """, unsafe_allow_html=True)

            # Stochastic
            stoch_status = "과매도" if latest['STOCH_K'] < 20 else "과매수" if latest['STOCH_K'] > 80 else "중립"
            stoch_color = "🟢" if latest['STOCH_K'] < 20 else "🔴" if latest['STOCH_K'] > 80 else "🟡"
            st.markdown(f"""
            <div class="indicator-box">
            <b>Stochastic</b> {stoch_color} {stoch_status}<br>
            • %K: {latest['STOCH_K']:.2f}<br>
            • %D: {latest['STOCH_D']:.2f}<br>
            • 원점수: <b>{raw['STOCH']}</b> / 가중점수: <b>{raw['STOCH'] * weights['STOCH']:.2f}</b>
            </div>
            """, unsafe_allow_html=True)

            # 볼린저밴드
            bb_position = "과매도" if latest['BB_Percent'] < 0.2 else "과매수" if latest['BB_Percent'] > 0.8 else "중립"
            bb_color = "🟢" if latest['BB_Percent'] < 0.2 else "🔴" if latest['BB_Percent'] > 0.8 else "🟡"
            st.markdown(f"""
            <div class="indicator-box">
            <b>볼린저밴드</b> {bb_color} {bb_position}<br>
            • %B: {latest['BB_Percent']:.2f}<br>
            • Upper: {analyzer.format_currency(latest['BB_Upper'], ticker)}<br>
            • Lower: {analyzer.format_currency(latest['BB_Lower'], ticker)}<br>
            • 원점수: <b>{raw['BB']}</b> / 가중점수: <b>{raw['BB'] * weights['BB']:.2f}</b>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📊 점수 요약")

        # 점수 막대 그래프
        df_scores = pd.DataFrame([{
            '지표': indicator,
            '원점수': raw.get(indicator, 0),
            '가중치': f"{weights.get(indicator, 0)*100:.0f}%",
            '최종점수': round(raw.get(indicator, 0) * weights.get(indicator, 0), 2)
        } for indicator in ['MACD', 'MA', 'RSI', 'BB', 'STOCH', 'ADX']])

        fig_scores = go.Figure()
        colors_score = ['#26a69a' if s > 0 else '#ef5350' if s < 0 else '#FFB74D'
                       for s in df_scores['최종점수']]

        fig_scores.add_trace(go.Bar(
            x=df_scores['지표'],
            y=df_scores['최종점수'],
            marker_color=colors_score,
            text=[f"{s:.2f}" for s in df_scores['최종점수']],
            textposition='outside',
            name='최종점수'
        ))

        fig_scores.update_layout(
            title=f"지표별 최종 점수 (가중치 적용) - 총합: {score:.1f}",
            xaxis_title="지표",
            yaxis_title="최종 점수",
            height=350,
            showlegend=False,
            plot_bgcolor='white'
        )

        st.plotly_chart(fig_scores, use_container_width=True)

        st.dataframe(df_scores, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("🎯 목표가 설정 (ATR 기반)")

        st.info(f"**ATR (14일 평균 변동폭):** {analyzer.format_currency(latest['ATR'], ticker)}")
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🟢 매수 진입가")
            st.metric("보수적 매수가", analyzer.format_currency(targets['conservative_buy'], ticker),
                     f"{((targets['conservative_buy']/latest['Close']-1)*100):.1f}%")
            st.metric("공격적 매수가", analyzer.format_currency(targets['aggressive_buy'], ticker),
                     f"{((targets['aggressive_buy']/latest['Close']-1)*100):.1f}%")

            st.markdown("### 🔴 손절가")
            st.metric("손절가", analyzer.format_currency(targets['stop_loss'], ticker),
                     f"{((targets['stop_loss']/latest['Close']-1)*100):.1f}%",
                     delta_color="inverse")

        with col2:
            st.markdown("### 🎯 목표가")
            for i in range(1, 4):
                target_key = f'target_{i}'
                gain_pct = ((targets[target_key]/latest['Close']-1)*100)
                st.metric(f"목표가 {i} ({'단기' if i==1 else '중기' if i==2 else '장기'})",
                         analyzer.format_currency(targets[target_key], ticker),
                         f"+{gain_pct:.1f}%")

        st.markdown("---")
        st.markdown("### 💡 활용 가이드")
        st.markdown("""
        - **보수적 매수가**: 안전한 진입점, 리스크 낮음
        - **공격적 매수가**: 적극적 진입점, 큰 수익 기대
        - **손절가**: 반드시 지켜야 할 손실 제한선
        - **목표가 1**: 단기 수익 실현 (빠른 청산)
        - **목표가 2**: 중기 수익 목표 (균형)
        - **목표가 3**: 장기 수익 목표 (욕심)
        """)

    with tab4:
        st.subheader("⚠️ 위험도 분석")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("연간 변동성", f"{risk['volatility']:.2f}%")
        with col2:
            st.metric("샤프 비율", f"{risk['sharpe_ratio']:.2f}")
        with col3:
            st.metric("최대 낙폭", f"{risk['max_drawdown']:.2f}%")
        with col4:
            st.metric("현재 낙폭", f"{risk['current_drawdown']:.2f}%")

        st.markdown("---")

        # 위험도 해석
        vol_level = "매우 높음" if risk['volatility'] > 60 else \
                   "높음" if risk['volatility'] > 40 else \
                   "보통" if risk['volatility'] > 20 else "낮음"

        sharpe_level = "우수" if risk['sharpe_ratio'] > 1.5 else \
                      "양호" if risk['sharpe_ratio'] > 1.0 else \
                      "보통" if risk['sharpe_ratio'] > 0.5 else "미흡"

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 위험도 해석")
            st.markdown(f"""
            - **변동성 수준:** {vol_level}
            - **수익성 평가:** {sharpe_level}
            - **최대 손실 경험:** 고점 대비 {abs(risk['max_drawdown']):.1f}% 하락
            - **현재 위치:** 고점 대비 {abs(risk['current_drawdown']):.1f}% 하락
            """)

        with col2:
            st.markdown("### 💡 투자 가이드")

            if risk['volatility'] < 20:
                st.success("✅ 안정적인 종목 - 보수적 투자자에게 적합")
            elif risk['volatility'] < 40:
                st.info("📊 보통 수준 - 균형잡힌 투자 필요")
            else:
                st.warning("⚠️ 높은 변동성 - 리스크 관리 필수")

            if risk['sharpe_ratio'] > 1.0:
                st.success("✅ 위험 대비 수익률 양호")
            else:
                st.warning("⚠️ 위험 대비 수익률 낮음")

    with tab5:
        st.subheader("📝 투자 메모 & 알림 설정")

        # 메모 기능
        st.markdown("### 📝 투자 메모")
        current_note = analyzer.get_note(ticker)
        note = st.text_area(
            "이 종목에 대한 메모를 작성하세요 (전략, 관찰 사항 등)",
            value=current_note,
            height=150,
            key="note_input"
        )

        if st.button("💾 메모 저장"):
            analyzer.save_note(ticker, note)
            st.success("✅ 메모가 저장되었습니다!")

        st.markdown("---")

        # 간단한 알림 설정 (UI만)
        st.markdown("### 🔔 가격 알림 설정")

        col1, col2 = st.columns(2)

        with col1:
            alert_upper = st.number_input(
                "목표가 도달 알림",
                min_value=0.0,
                value=float(targets['target_1']),
                format="%.2f",
                key="alert_upper"
            )

        with col2:
            alert_lower = st.number_input(
                "손절가 도달 알림",
                min_value=0.0,
                value=float(targets['stop_loss']),
                format="%.2f",
                key="alert_lower"
            )

        if st.button("🔔 알림 설정 (준비 중)"):
            st.info("📱 알림 기능은 향후 업데이트에서 제공될 예정입니다.")

        st.markdown("---")

        # 분석 요약
        st.markdown("### 📊 분석 요약")
        st.markdown(f"""
        - **분석 일시:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        - **종합 점수:** {score:.1f} ({rec})
        - **현재가:** {analyzer.format_currency(latest['Close'], ticker)}
        - **추천 진입가:** {analyzer.format_currency(targets['conservative_buy'], ticker)} ~ {analyzer.format_currency(targets['aggressive_buy'], ticker)}
        - **손절가:** {analyzer.format_currency(targets['stop_loss'], ticker)}
        - **목표가:** {analyzer.format_currency(targets['target_1'], ticker)} → {analyzer.format_currency(targets['target_2'], ticker)} → {analyzer.format_currency(targets['target_3'], ticker)}
        """)


def main():
    st.sidebar.title("📊 주식 분석 대시보드")
    st.sidebar.markdown("---")
    st.sidebar.caption("Version 2.5 Enhanced")

    analyzer = StreamlitStockAnalyzer()

    page = st.sidebar.radio("메뉴", ["🔍 단일 종목 분석"])

    if page == "🔍 단일 종목 분석":
        show_single_analysis_enhanced(analyzer)

    # 유저 가이드 표시
    show_user_guide_sidebar()

    # 데이터베이스 정보
    with st.sidebar.expander("💾 데이터베이스 정보", expanded=False):
        st.markdown(f"""
        **로드된 주식:**
        - 🇰🇷 한국: **{analyzer.kr_total:,}개**
        - 🇺🇸 미국: **{analyzer.us_total:,}개**
        
        **업데이트:**
        - 캐시 유효기간: 24시간
        - 마지막 업데이트: 앱 시작 시
        """)



def show_single_analysis_enhanced(analyzer):
    """강화된 단일 종목 분석"""
    st.title("🔍 단일 종목 분석")

    if 'analyze_ticker' not in st.session_state:
        st.session_state.analyze_ticker = None
    if 'selected_country' not in st.session_state:
        st.session_state.selected_country = '🇺🇸 미국'
    if 'show_analysis' not in st.session_state:
        st.session_state.show_analysis = False

    # 인기 종목
    with st.expander("💡 인기 종목 바로가기 (클릭 시 즉시 분석!)", expanded=False):
        st.markdown("##### 🇺🇸 미국 주식")
        us = {"AAPL": "애플", "MSFT": "마이크로소프트", "GOOGL": "구글", "AMZN": "아마존",
              "NVDA": "엔비디아", "TSLA": "테슬라", "META": "메타", "AMD": "AMD", "NFLX": "넷플릭스"}
        cols = st.columns(3)
        for i, (t, n) in enumerate(us.items()):
            with cols[i % 3]:
                if st.button(f"{t}\n{n}", key=f"us_{t}", use_container_width=True):
                    st.session_state.analyze_ticker = t
                    st.session_state.show_analysis = True
                    st.rerun()

        st.markdown("##### 🇰🇷 한국 주식")
        kr = {"005930.KS": "삼성전자", "000660.KS": "SK하이닉스", "035420.KS": "NAVER",
              "005380.KS": "현대차", "051910.KS": "LG화학", "035720.KS": "카카오"}
        cols = st.columns(3)
        for i, (t, n) in enumerate(kr.items()):
            with cols[i % 3]:
                if st.button(f"{n}", key=f"kr_{t}", use_container_width=True):
                    st.session_state.analyze_ticker = t
                    st.session_state.show_analysis = True
                    st.rerun()

    st.markdown("---")

    # 검색
    col1, col2 = st.columns([1, 3])
    with col1:
        opts = ["🇺🇸 미국", "🇰🇷 한국", "🇯🇵 일본", "🌐 기타"]
        country = st.selectbox("🌍 국가", opts, index=opts.index(st.session_state.selected_country))
        st.session_state.selected_country = country

    with col2:
        placeholders = {"🇰🇷 한국": "예: 삼성, 005930", "🇺🇸 미국": "예: AAPL, TSLA",
                       "🇯🇵 일본": "예: 7203.T", "🌐 기타": "예: 0700.HK"}
        ticker_input = st.text_input("🔍 종목 검색", placeholder=placeholders[country])

    # 검색 결과
    if ticker_input and len(ticker_input) >= 2:
        st.markdown("##### 🔍 검색 결과 (클릭 시 즉시 분석!)")

        results = analyzer.search_kr_stock(ticker_input) if country == "🇰🇷 한국" else \
                  analyzer.search_us_stock(ticker_input) if country == "🇺🇸 미국" else []

        if results:
            cols = st.columns(3)
            for idx, r in enumerate(results):
                with cols[idx % 3]:
                    short = r['name'][:12] + ".." if len(r['name']) > 12 else r['name']
                    if st.button(f"**{short}**\n`{r['ticker']}`", key=f"s_{r['ticker']}", use_container_width=True):
                        st.session_state.analyze_ticker = r['ticker']
                        st.session_state.show_analysis = True
                        st.rerun()

    # 수동 분석
    st.markdown("---")
    col1, col2 = st.columns(2)
    period = col1.selectbox("분석 기간", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    num_days = col2.number_input("차트 일수", 30, 500, 180, 30)

    if st.button("🔍 분석 시작", type="primary", use_container_width=True, disabled=(not ticker_input)):
        if ticker_input:
            t = ticker_input.upper() if not any('\uac00' <= c <= '\ud7a3' for c in ticker_input) else ticker_input
            st.session_state.analyze_ticker = t
            st.session_state.show_analysis = True

    # 분석 결과 표시
    if st.session_state.show_analysis and st.session_state.analyze_ticker:
        ticker = st.session_state.analyze_ticker
        st.session_state.analyze_ticker = None

        st.markdown("---")
        st.markdown("---")
        run_analysis_enhanced(analyzer, ticker, period, num_days)


if __name__ == "__main__":
    main()
