# src/data/processor.py

import pandas as pd
import numpy as np
import logging

def setup_logger():
    """设置日志配置"""
    # 清除之前的处理器
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
            
    # 创建新的处理器
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # 设置日志级别并添加处理器
    root.setLevel(logging.INFO)
    root.addHandler(handler)

# 初始化日志
setup_logger()
logger = logging.getLogger(__name__)

def verify_features(df):
    """验证特征是否正确计算"""
    required_features = [
        'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower',
        'rsi', 'atr', 'ma10', 'ma50',
        'trend_5d', 'trend_10d', 'trend_20d',
        'ma10_ratio', 'ma50_ratio', 'volatility'
    ]
    
    missing = [f for f in required_features if f not in df.columns]
    if missing:
        logger.error(f"缺失的必需特征: {missing}")
        return False
        
    logger.info("特征验证通过")
    return True

def calculate_technical_indicators(data, use_short_term=True):
    """
    计算各种技术指标
    
    参数:
        data (DataFrame): 包含 OHLCV 数据的 DataFrame
        use_short_term (bool): 是否使用短期指标
        
    返回:
        DataFrame: 添加了技术指标的数据
    """
    if data is None or len(data) < 20:
        logger.warning("数据不足，无法计算技术指标")
        return data
    
    # 创建副本，避免修改原始数据
    df = data.copy()
    
    try:
        # 验证必要的列是否存在
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必要的列: {missing_columns}")
        
        # RSI计算
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD计算
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = macd_line - signal_line
        
        # 布林带计算
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (std * 2)
        df['bb_lower'] = df['bb_middle'] - (std * 2)
        
        # ATR计算
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # 移动平均线
        df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
        df['ma50'] = df['close'].rolling(window=50, min_periods=1).mean()
        
        # 趋势指标
        df['trend_5d'] = df['close'].pct_change(periods=5)
        df['trend_10d'] = df['close'].pct_change(periods=10)
        df['trend_20d'] = df['close'].pct_change(periods=20)
        
        # 移动均线比率
        df['ma10_ratio'] = df['close'] / df['ma10'].replace(0, np.nan)
        df['ma50_ratio'] = df['close'] / df['ma50'].replace(0, np.nan)
        
        # 波动率
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        # 处理无穷大和NaN值
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
        
    except Exception as e:
        logger.error(f"计算技术指标失败: {str(e)}")
        if 'df' in locals():
            available_features = "\n".join([
                "当前可用的特征列:",
                *[f"- {col}" for col in sorted(df.columns)]
            ])
            logger.info(available_features)
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
    
    return rolling_mean, upper_band, lower_band  # 修复了返回值

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
    """准备机器学习特征"""
    if data is None or len(data) < target_horizon + 10:
        logger.warning("数据不足，无法准备特征")
        return None, None
    
    try:
        df = data.copy()
        
        # 创建目标变量
        df['future_return'] = df['close'].pct_change(periods=target_horizon).shift(-target_horizon)
        df['target'] = np.where(df['future_return'] > 0, 1, 0)
        
        # 修改特征列表，使用与calculate_technical_indicators一致的名称
        features = [
            'rsi',                  # RSI
            'macd', 'macd_signal', 'macd_hist',  # MACD相关指标
            'bb_upper', 'bb_middle', 'bb_lower',  # 布林带
            'atr',                  # ATR
            'ma10', 'ma50',        # 移动平均线
            'trend_5d', 'trend_10d', 'trend_20d',  # 趋势指标
            'ma10_ratio', 'ma50_ratio',  # 移动均线比率
            'volatility'           # 波动率
        ]
        
        # 检查所有特征是否存在
        available_features = [f for f in features if f in df.columns]
        
        if len(available_features) < len(features):
            missing = [f for f in features if f not in df.columns]
            logger.error(f"缺少以下特征: {missing}")
        
        # 删除包含NaN的行
        if drop_na:
            df = df.dropna(subset=available_features + ['target'])
        
        # 分离特征和目标
        X = df[available_features]
        y = df['target']
        
        logger.info(f"特征准备完成，共{len(X)}条数据，{len(available_features)}个特征")
        return X, y
    
    except Exception as e:
        logger.error(f"准备预测数据失败: {str(e)}")
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