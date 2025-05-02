# src/data/processor.py

import pandas as pd
import numpy as np
import logging

# 设置日志
logger = logging.getLogger(__name__)

def calculate_technical_indicators(data, **kwargs):
    """
    计算各种技术指标
    
    参数:
        data (DataFrame): 原始价格数据
        
    返回:
        DataFrame: 添加了技术指标的数据
    """
    if data is None or len(data) < 20:
        logger.warning("数据不足，无法计算技术指标")
        return data
    
    # 创建副本，避免修改原始数据
    df = data.copy()
    
    try:
        # 1. RSI - 相对强弱指标 (14天)
        df['rsi'] = calculate_rsi(df, window=14)
        
        # 2. MACD - 移动平均线收敛/发散
        macd_line, signal_line, macd_hist = calculate_macd(df)
        df['macd'] = macd_line
        df['signal'] = signal_line
        df['hist'] = macd_hist
        
        # 3. 布林带
        df['ma20'], df['upper_band'], df['lower_band'] = calculate_bollinger_bands(df)
        
        # 4. ATR - 平均真实范围
        df['atr'] = calculate_atr(df)
        
        # 5. 移动平均线
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma50'] = df['close'].rolling(window=50).mean()
        
        # 6. 趋势指标
        df['trend_5d'] = df['close'].pct_change(periods=5)
        df['trend_10d'] = df['close'].pct_change(periods=10)
        df['trend_20d'] = df['close'].pct_change(periods=20)
        
        # 7. 移动均线比率
        df['ma10_ratio'] = df['close'] / df['ma10']
        df['ma50_ratio'] = df['close'] / df['ma50']
        
        # 8. 波动率
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        logger.info("技术指标计算完成")
        return df
    
    except Exception as e:
        logger.error(f"计算技术指标失败: {str(e)}")
        return data

def calculate_rsi(data, window=14):
    """计算RSI指标"""
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    # 计算快线与慢线
    ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
    
    # MACD线 = 快线 - 慢线
    macd_line = ema_fast - ema_slow
    
    # 信号线 = MACD的EMA
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # MACD柱状图 = MACD线 - 信号线
    macd_histogram = macd_line - signal_line
    
    return macd_line, signal_line, macd_histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """计算布林带指标"""
    rolling_mean = data['close'].rolling(window=window).mean()
    rolling_std = data['close'].rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    
    return rolling_mean, upper_band, lower_band

def calculate_atr(data, window=14):
    """计算平均真实范围(ATR)指标"""
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    atr = true_range.rolling(window=window).mean()
    return atr

def prepare_features(data, target_horizon=1, drop_na=True):
    """
    准备机器学习特征
    
    参数:
        data (DataFrame): 包含技术指标的数据
        target_horizon (int): 目标预测周期（天数）
        drop_na (bool): 是否删除包含NaN的行
        
    返回:
        tuple: (X, y) 特征和目标变量
    """
    if data is None or len(data) < target_horizon + 10:
        logger.warning("数据不足，无法准备特征")
        return None, None
    
    try:
        # 创建副本，避免修改原始数据
        df = data.copy()
        
        # 创建目标变量 - 未来n天的涨跌
        df['future_return'] = df['close'].pct_change(periods=target_horizon).shift(-target_horizon)
        df['target'] = np.where(df['future_return'] > 0, 1, 0)
        
        # 选择特征
        features = [
            'rsi', 'macd', 'signal', 'hist', 
            'ma20', 'atr', 'trend_5d', 'trend_10d', 'trend_20d',
            'ma10_ratio', 'ma50_ratio', 'volatility'
        ]
        
        # 检查所有特征是否存在
        available_features = [f for f in features if f in df.columns]
        
        if len(available_features) < len(features):
            missing = [f for f in features if f not in available_features]
            logger.warning(f"缺少以下特征: {missing}")
            
        # 删除包含NaN的行
        if drop_na:
            df = df.dropna(subset=available_features + ['target'])
        
        # 分离特征和目标
        X = df[available_features]
        y = df['target']
        
        logger.info(f"特征准备完成，共{len(X)}条数据，{len(available_features)}个特征")
        return X, y
    
    except Exception as e:
        logger.error(f"准备特征失败: {str(e)}")
        return None, None

def normalize_data(X_train, X_test=None):
    """
    标准化特征数据
    
    参数:
        X_train (DataFrame): 训练数据特征
        X_test (DataFrame, optional): 测试数据特征
        
    返回:
        tuple: 标准化后的训练和测试数据
    """
    from sklearn.preprocessing import StandardScaler
    
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            return X_train_scaled, X_test_scaled, scaler
        
        return X_train_scaled, scaler
    
    except Exception as e:
        logger.error(f"标准化数据失败: {str(e)}")
        if X_test is not None:
            return X_train, X_test, None
        return X_train, None