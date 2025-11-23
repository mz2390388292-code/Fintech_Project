# app.py
import streamlit as st
import plotly.graph_objects as go
from src.data_loader import load_data
from src.rl_agent import train_rl_model, run_backtest

# --- 1. 页面配置 ---
st.set_page_config(page_title="Fintech AI Trader", layout="wide")

st.title("📈 Fintech Group Project: AI Stock Trading Platform")
st.markdown("""
该平台整合了 **实时数据获取**、**技术指标分析** 以及 **强化学习(RL)自动交易**。
""")

# --- 2. 侧边栏设置 ---
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Stock Ticker (e.g., AAPL, NVDA)", "AAPL")
period = st.sidebar.selectbox("Data Period", ["1y", "2y", "5y"])

# --- 3. 获取数据 ---
# 使用缓存装饰器，避免每次点击按钮都重新下载数据
@st.cache_data
def get_data(t, p):
    return load_data(t, p)

with st.spinner('Fetching data from Yahoo Finance...'):
    df = get_data(ticker, period)

if df is not None:
    # --- 4. Part D: 可视化与分析 ---
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(f"{ticker} Price & Technical Indicators")
        # 绘制 K 线图
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index,
                        open=df['Open'], high=df['High'],
                        low=df['Low'], close=df['Close'], name='OHLC'))
        # 叠加 SMA
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='orange', width=1), name='SMA 20'))
        
        fig.update_layout(height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Latest Data")
        st.dataframe(df[['Close', 'RSI', 'SMA_20']].tail(10))
        st.metric("Latest Close", f"${df['Close'].iloc[-1]:.2f}")
        st.metric("Latest RSI", f"{df['RSI'].iloc[-1]:.2f}")

    st.markdown("---")

    # --- 5. Part D Advanced: 强化学习交易 ---
    st.header("🤖 Reinforcement Learning Agent")
    st.write("点击下方按钮训练 AI 交易员。AI 将基于价格和技术指标学习买卖策略。")

    if st.button("Start AI Training & Backtest"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 训练模型
        status_text.text("Step 1/2: Training PPO Model (This may take a moment)...")
        model, env = train_rl_model(df)
        progress_bar.progress(50)
        
        # 回测
        status_text.text("Step 2/2: Running Backtest Strategy...")
        profit = run_backtest(model, df)
        progress_bar.progress(100)
        status_text.text("Done!")
        
        # 展示结果
        st.success(f"AI Trading Completed!")
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            color = "green" if profit > 1 else "red"
            st.metric("Total Return (Profit Factor)", f"{profit:.4f}", delta=f"{(profit-1)*100:.2f}%")
            st.caption("注：> 1.0 表示盈利 (e.g. 1.10 = 10% Profit)")
        
        with col_r2:
            st.info("Strategy Logic: Reinforcement Learning (PPO) using [Open, Close, RSI, SMA]")

else:
    st.error("无法获取数据，请检查股票代码是否正确 (注意: 中国股票需加后缀，如 600519.SS)。")
