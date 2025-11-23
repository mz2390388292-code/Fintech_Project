# src/data_loader.py
import yfinance as yf
import pandas as pd
import talib # 替换了 pandas_ta
import numpy as np

def load_data(ticker, period="2y", interval="1d"):
    """
    获取数据并使用 TA-Lib 计算技术指标
    """
    try:
        # 1. 获取数据
        print(f"Downloading data for {ticker}...")
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        
        if df.empty:
            print(f"Warning: No data found for {ticker}")
            return None
            
        # 2. 数据清洗：重置索引
        df.reset_index(inplace=True)
        
        # === 强制统一列名 (Open, High, Low, Close, Volume) ===
        df.columns = [str(c).strip().capitalize() for c in df.columns]
        
        # 处理日期列
        if 'Date' not in df.columns and 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'Date'}, inplace=True)
            
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        else:
            print("Error: Could not find Date/Datetime column")
            return None

        # 3. 数据工程：使用 TA-Lib 计算指标
        # 确保 'Close' 列存在且是浮点数类型
        if 'Close' not in df.columns:
            print("Error: 'Close' column is missing.")
            return None
        
        # TA-Lib 通常需要 numpy array 或 pandas Series 作为输入
        close_prices = df['Close'].values
        
        # --- Indicator 1: SMA (Simple Moving Average) ---
        # 对应 pandas_ta: ta.sma(df['Close'], length=20)
        df['SMA_20'] = talib.SMA(close_prices, timeperiod=20)
        
        # --- Indicator 2: RSI (Relative Strength Index) ---
        # 对应 pandas_ta: ta.rsi(df['Close'], length=14)
        df['RSI'] = talib.RSI(close_prices, timeperiod=14)
        
        # --- Indicator 3: MACD ---
        # talib.MACD 返回三个数组: macd, signal, hist
        # 对应 pandas_ta: ta.macd(df['Close'])
        macd, macd_signal, macd_hist = talib.MACD(
            close_prices, 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        df['MACD'] = macd
        df['MACD_SIGNAL'] = macd_signal
        df['MACD_HIST'] = macd_hist

        # 删除因为计算指标产生的空值 (TA-Lib 计算初期会产生 NaN)
        df.dropna(inplace=True)
        
        print(f"Data loaded successfully with TA-Lib features. Shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
