# src/rl_agent.py
import gymnasium as gym
import gym_anytrading
from gym_anytrading.envs import StocksEnv
from stable_baselines3 import A2C, PPO
import pandas as pd
import numpy as np

# 自定义处理函数：让 AI 不仅仅看价格，还能看到我们计算的 RSI 和 MACD
def my_process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Close'].to_numpy()[start:end]
    
    # 提取我们在 data_loader 中计算的指标作为信号
    # 注意：这里需要确保你的 dataframe 里确实有这些列
    signal_features = env.df.loc[:, ['Close', 'Open', 'High', 'Low', 'RSI', 'SMA_20']].to_numpy()[start:end]
    
    return prices, signal_features

def train_rl_model(df):
    """
    训练 PPO 模型
    """
    # 1. 创建环境
    # window_size=20: AI 每次看过去 20 天的数据来做决定
    # frame_bound: 训练数据的范围
    
    # 这里的 trick 是让 gym_anytrading 使用我们的自定义指标
    class MyStocksEnv(StocksEnv):
        _process_data = my_process_data

    # 创建自定义环境
    env = MyStocksEnv(df=df, window_size=20, frame_bound=(20, len(df)))

    # 2. 创建并训练模型 (使用 PPO 算法，适合初学者，稳定)
    model = PPO("MlpPolicy", env, verbose=0) 
    model.learn(total_timesteps=5000) # 训练步数，演示用 5000，实际可用 100000+
    
    return model, env

def run_backtest(model, df):
    """
    运行回测
    """
    class MyStocksEnv(StocksEnv):
        _process_data = my_process_data

    # 创建环境 (重用同一个数据进行演示，实际应使用测试集)
    env = MyStocksEnv(df=df, window_size=20, frame_bound=(20, len(df)))
    
    observation, info = env.reset()
    
    total_profit = []
    actions = []
    
    while True:
        action, _states = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            # 记录最后的信息
            total_profit = info['total_profit']
            break
            
    return total_profit
