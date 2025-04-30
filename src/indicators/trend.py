# src/indicators/trend.py

import pandas as pd
import numpy as np
import logging

# 设置日志
logger = logging.getLogger(__name__)

def calculate_ma(data, column='close', window=20):
    """
    计算简单移动平均线
    
    参数:
        data (DataFrame): 价格数据
        column (str): 要计算MA的列名，默认是'close'
        window (int): 窗口大小
        
    返回:
        Series: 移动平均线数据
    """
    try:
        if column not in data.columns:
            logger.error(f"列 {column} 不存在于数据中")
            return None
            
        ma = data[column].rolling(window=window).mean()
        return ma
        
    except Exception as e:
        logger.error(f"计算MA失败: {str(e)}")
        return None

def calculate_ema(data, column='close', span=20):
    """
    计算指数移动平均线
    
    参数:
        data (DataFrame): 价格数据
        column (str): 要计算EMA的列名，默认是'close'
        span (int): EMA的周期
        
    返回:
        Series: 指数移动平均线数据
    """
    try:
        if column not in data.columns:
            logger.error(f"列 {column} 不存在于数据中")
            return None
            
        ema = data[column].ewm(span=span, adjust=False).mean()
        return ema
        
    except Exception as e:
        logger.error(f"计算EMA失败: {str(e)}")
        return None

def calculate_macd(data, column='close', fast=12, slow=26, signal=9):
    """
    计算MACD (移动平均线收敛/发散)
    
    参数:
        data (DataFrame): 价格数据
        column (str): 要计算MACD的列名，默认是'close'
        fast (int): 快线周期
        slow (int): 慢线周期
        signal (int): 信号线周期
        
    返回:
        tuple: (macd_line, signal_line, histogram)
    """
    try:
        if column not in data.columns:
            logger.error(f"列 {column} 不存在于数据中")
            return None, None, None
            
        # 计算快线与慢线
        ema_fast = data[column].ewm(span=fast, adjust=False).mean()
        ema_slow = data[column].ewm(span=slow, adjust=False).mean()
        
        # MACD线 = 快线 - 慢线
        macd_line = ema_fast - ema_slow
        
        # 信号线 = MACD的EMA
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # MACD柱状图 = MACD线 - 信号线
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
        
    except Exception as e:
        logger.error(f"计算MACD失败: {str(e)}")
        return None, None, None

def calculate_tema(data, column='close', period=20):
    """
    计算三重指数移动平均线 (TEMA)
    
    参数:
        data (DataFrame): 价格数据
        column (str): 要计算TEMA的列名，默认是'close'
        period (int): 周期
        
    返回:
        Series: TEMA数据
    """
    try:
        if column not in data.columns:
            logger.error(f"列 {column} 不存在于数据中")
            return None
            
        # 计算一次EMA
        ema1 = data[column].ewm(span=period, adjust=False).mean()
        
        # 计算二次EMA (EMA的EMA)
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        
        # 计算三次EMA (EMA的EMA的EMA)
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        
        # TEMA = 3 * EMA1 - 3 * EMA2 + EMA3
        tema = 3 * ema1 - 3 * ema2 + ema3
        
        return tema
        
    except Exception as e:
        logger.error(f"计算TEMA失败: {str(e)}")
        return None

def calculate_ppo(data, column='close', fast=12, slow=26):
    """
    计算价格震荡百分比 (PPO)
    
    参数:
        data (DataFrame): 价格数据
        column (str): 要计算PPO的列名，默认是'close'
        fast (int): 快线周期
        slow (int): 慢线周期
        
    返回:
        Series: PPO数据
    """
    try:
        if column not in data.columns:
            logger.error(f"列 {column} 不存在于数据中")
            return None
            
        # 计算快线与慢线
        ema_fast = data[column].ewm(span=fast, adjust=False).mean()
        ema_slow = data[column].ewm(span=slow, adjust=False).mean()
        
        # PPO = (快线 - 慢线) / 慢线 * 100
        ppo = (ema_fast - ema_slow) / ema_slow * 100
        
        return ppo
        
    except Exception as e:
        logger.error(f"计算PPO失败: {str(e)}")
        return None