# src/data/fetcher.py

import os
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
from alpaca_trade_api import REST

# 设置日志
logger = logging.getLogger(__name__)

def get_api_client():
    """获取Alpaca API客户端"""
    is_paper = os.getenv('ALPACA_PAPER', 'true').lower() == 'true'
    
    if is_paper:
        API_KEY = os.getenv('ALPACA_API_KEY')
        API_SECRET = os.getenv('ALPACA_API_SECRET')
        BASE_URL = 'https://paper-api.alpaca.markets'
    else:
        API_KEY = os.getenv('ALPACA_LIVE_API_KEY')
        API_SECRET = os.getenv('ALPACA_LIVE_API_SECRET')
        BASE_URL = 'https://api.alpaca.markets'
    
    # Add this debug line
    logger.info(f"Using API with KEY: {API_KEY[:5]}..., BASE_URL: {BASE_URL}, PAPER: {is_paper}")
    
    return tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def get_stock_data(symbol, days=120, timeframe='1D'):
    """
    获取股票历史数据
    
    参数:
        symbol (str): 股票代码
        days (int): 获取的天数
        timeframe (str): 时间框架，如'1D', '1H', '15Min'等
        
    返回:
        DataFrame 或 None: 包含历史数据的DataFrame，获取失败时返回None
    """
    # 获取API客户端
    api = get_api_client()
    
    # 计算开始和结束日期
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    start = start_date.strftime('%Y-%m-%d')
    end = end_date.strftime('%Y-%m-%d')
    
    try:
        # 获取股票数据
        bars = api.get_bars(symbol, timeframe, start=start, end=end, adjustment='raw', feed='iex').df
        
        if len(bars) > 0:
            logger.info(f"成功获取{symbol}的数据，共{len(bars)}条记录")
            return bars
        else:
            logger.warning(f"未找到{symbol}的数据")
            return None
    except Exception as e:
        logger.error(f"获取{symbol}数据失败: {str(e)}")
        return None

def get_market_data(symbols=['SPY', 'QQQ', 'DIA'], days=30):
    """
    获取市场数据（多个指数）
    
    参数:
        symbols (list): 需要获取的市场指数代码列表
        days (int): 获取的天数
    
    返回:
        dict: 每个指数的数据
    """
    result = {}
    
    for symbol in symbols:
        data = get_stock_data(symbol, days)
        if data is not None:
            result[symbol] = data
        # 防止API限制
        time.sleep(0.5)
            
    return result

def get_latest_quotes(symbols):
    """
    获取一组股票的最新报价
    
    参数:
        symbols (list): 需要获取报价的股票代码列表
        
    返回:
        dict: 每个股票的最新价格数据
    """
    api = get_api_client()
    results = {}
    
    # 如果输入是单个字符串，转换为列表
    if isinstance(symbols, str):
        symbols = [symbols]
    
    for symbol in symbols:
        try:
            # 获取最新报价
            quote = api.get_latest_quote(symbol)
            
            if hasattr(quote, 'ap') and hasattr(quote, 'bp'):
                results[symbol] = {
                    'ask_price': quote.ap,
                    'bid_price': quote.bp,
                    'ask_size': quote.as_,
                    'bid_size': quote.bs,
                    'timestamp': quote.t
                }
            else:
                logger.warning(f"获取{symbol}的报价信息不完整")
                
        except Exception as e:
            logger.error(f"获取{symbol}的报价失败: {str(e)}")
            
        # 防止API限制
        time.sleep(0.1)
            
    return results

def get_account_info():
    """
    获取账户信息
    
    返回:
        dict: 账户信息
    """
    api = get_api_client()
    
    try:
        account = api.get_account()
        return {
            'cash': float(account.cash),
            'portfolio_value': float(account.portfolio_value),
            'equity': float(account.equity),
            'buying_power': float(account.buying_power),
            'initial_margin': float(account.initial_margin),
            'last_equity': float(account.last_equity),
            'long_market_value': float(account.long_market_value),
            'short_market_value': float(account.short_market_value),
            'daytrade_count': int(account.daytrade_count),
            'status': account.status
        }
    except Exception as e:
        logger.error(f"获取账户信息失败: {str(e)}")
        return None

def get_positions():
    """
    获取当前持仓
    
    返回:
        list: 持仓信息列表
    """
    api = get_api_client()
    
    try:
        positions = api.list_positions()
        return [
            {
                'symbol': position.symbol,
                'qty': int(position.qty),
                'avg_entry_price': float(position.avg_entry_price),
                'current_price': float(position.current_price),
                'market_value': float(position.market_value),
                'cost_basis': float(position.cost_basis),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc) * 100,
                'side': position.side
            }
            for position in positions
        ]
    except Exception as e:
        logger.error(f"获取持仓信息失败: {str(e)}")
        return []