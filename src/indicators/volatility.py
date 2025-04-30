# src/indicators/volatility.py

import pandas as pd
import numpy as np
import logging

# 设置日志
logger = logging.getLogger(__name__)

def calculate_bollinger_bands(data, column='close', window=20, num_std=2):
    """
    计算布林带指标
    
    参数:
        data (DataFrame): 价格数据
        column (str): 要计算布林带的列名，默认是'close'
        window (int): 窗口大小
        num_std (float): 标准差的倍数
        
    返回:
        tuple: (middle_band, upper_band, lower_band)
    """
    try:
        if column not in data.columns:
            logger.error(f"列 {column} 不存在于数据中")
            return None, None, None
            
        # 计算中轨(移动平均线)
        middle_band = data[column].rolling(window=window).mean()
        
        # 计算标准差
        std = data[column].rolling(window=window).std()
        
        # 计算上轨和下轨
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        return middle_band, upper_band, lower_band
        
    except Exception as e:
        logger.error(f"计算布林带失败: {str(e)}")
        return None, None, None

def calculate_atr(data, window=14):
    """
    计算平均真实范围 (ATR)
    
    参数:
        data (DataFrame): 价格数据，必须有high, low, close列
        window (int): 窗口大小
        
    返回:
        Series: ATR数据
    """
    try:
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                logger.error(f"列 {col} 不存在于数据中")
                return None
                
        # 计算三种情况的范围
        high_low = data['high'] - data['low']
        high_close_prev = np.abs(data['high'] - data['close'].shift(1))
        low_close_prev = np.abs(data['low'] - data['close'].shift(1))
        
        # 计算真实范围 (取三者最大值)
        ranges = pd.concat([high_low, high_close_prev, low_close_prev], axis=1)
        true_range = ranges.max(axis=1)
        
        # 计算ATR (真实范围的移动平均)
        atr = true_range.rolling(window=window).mean()
        
        return atr
        
    except Exception as e:
        logger.error(f"计算ATR失败: {str(e)}")
        return None

def calculate_keltner_channel(data, window=20, atr_factor=2):
    """
    计算肯特纳通道 (Keltner Channel)
    
    参数:
        data (DataFrame): 价格数据，必须有high, low, close列
        window (int): 窗口大小
        atr_factor (float): ATR乘数
        
    返回:
        tuple: (middle_line, upper_line, lower_line)
    """
    try:
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                logger.error(f"列 {col} 不存在于数据中")
                return None, None, None
                
        # 计算中轨 (EMA)
        middle_line = data['close'].ewm(span=window, adjust=False).mean()
        
        # 计算ATR
        atr = calculate_atr(data, window)
        
        # 计算上轨和下轨
        upper_line = middle_line + (atr * atr_factor)
        lower_line = middle_line - (atr * atr_factor)
        
        return middle_line, upper_line, lower_line
        
    except Exception as e:
        logger.error(f"计算Keltner Channel失败: {str(e)}")
        return None, None, None

def calculate_historical_volatility(data, column='close', window=20, trading_days=252):
    """
    计算历史波动率
    
    参数:
        data (DataFrame): 价格数据
        column (str): 要计算波动率的列名，默认是'close'
        window (int): 窗口大小
        trading_days (int): 一年的交易日数，用于年化
        
    返回:
        Series: 历史波动率
    """
    try:
        if column not in data.columns:
            logger.error(f"列 {column} 不存在于数据中")
            return None
            
        # 计算对数收益率
        log_returns = np.log(data[column] / data[column].shift(1))
        
        # 计算移动标准差
        rolling_std = log_returns.rolling(window=window).std()
        
        # 年化波动率
        annualized_vol = rolling_std * np.sqrt(trading_days)
        
        return annualized_vol
        
    except Exception as e:
        logger.error(f"计算历史波动率失败: {str(e)}")
        return None