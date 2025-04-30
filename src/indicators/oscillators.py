# src/indicators/oscillators.py

import pandas as pd
import numpy as np
import logging

# 设置日志
logger = logging.getLogger(__name__)

def calculate_rsi(data, column='close', window=14):
    """
    计算相对强弱指数 (RSI)
    
    参数:
        data (DataFrame): 价格数据
        column (str): 要计算RSI的列名，默认是'close'
        window (int): RSI周期
        
    返回:
        Series: RSI数据
    """
    try:
        if column not in data.columns:
            logger.error(f"列 {column} 不存在于数据中")
            return None
            
        # 计算价格变化
        delta = data[column].diff()
        
        # 区分上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 计算平均上涨和下跌
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # 计算相对强度
        rs = avg_gain / avg_loss
        
        # 计算RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    except Exception as e:
        logger.error(f"计算RSI失败: {str(e)}")
        return None

def calculate_stochastic(data, k_period=14, d_period=3):
    """
    计算随机指标 (Stochastic Oscillator)
    
    参数:
        data (DataFrame): 价格数据，必须有high, low, close列
        k_period (int): %K周期
        d_period (int): %D周期
        
    返回:
        tuple: (%K, %D)
    """
    try:
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                logger.error(f"列 {col} 不存在于数据中")
                return None, None
                
        # 计算最低价和最高价
        low_min = data['low'].rolling(window=k_period).min()
        high_max = data['high'].rolling(window=k_period).max()
        
        # 计算%K
        k = 100 * ((data['close'] - low_min) / (high_max - low_min))
        
        # 计算%D (K的移动平均)
        d = k.rolling(window=d_period).mean()
        
        return k, d
        
    except Exception as e:
        logger.error(f"计算Stochastic失败: {str(e)}")
        return None, None

def calculate_williams_r(data, period=14):
    """
    计算威廉指标 (Williams %R)
    
    参数:
        data (DataFrame): 价格数据，必须有high, low, close列
        period (int): 周期
        
    返回:
        Series: Williams %R数据
    """
    try:
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                logger.error(f"列 {col} 不存在于数据中")
                return None
                
        # 计算最高和最低价
        highest_high = data['high'].rolling(window=period).max()
        lowest_low = data['low'].rolling(window=period).min()
        
        # 计算Williams %R
        williams_r = -100 * (highest_high - data['close']) / (highest_high - lowest_low)
        
        return williams_r
        
    except Exception as e:
        logger.error(f"计算Williams %R失败: {str(e)}")
        return None

def calculate_cci(data, period=20):
    """
    计算商品通道指数 (CCI)
    
    参数:
        data (DataFrame): 价格数据，必须有high, low, close列
        period (int): 周期
        
    返回:
        Series: CCI数据
    """
    try:
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                logger.error(f"列 {col} 不存在于数据中")
                return None
                
        # 计算典型价格
        tp = (data['high'] + data['low'] + data['close']) / 3
        
        # 计算典型价格的移动平均
        tp_ma = tp.rolling(window=period).mean()
        
        # 计算平均绝对偏差
        tp_mad = np.abs(tp - tp_ma).rolling(window=period).mean()
        
        # 计算CCI
        cci = (tp - tp_ma) / (0.015 * tp_mad)
        
        return cci
        
    except Exception as e:
        logger.error(f"计算CCI失败: {str(e)}")
        return None