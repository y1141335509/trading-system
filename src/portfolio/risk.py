# src/portfolio/risk.py

import os
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta

from ..data.fetcher import get_stock_data, get_api_client

# 设置日志
logger = logging.getLogger(__name__)

def assess_stock_risk(symbol, days=120):
    """
    评估股票风险，返回风险类别和风险指标
    
    参数:
        symbol (str): 股票代码
        days (int): 评估周期（天数）
        
    返回:
        tuple: (风险类别, 风险指标字典)
    """
    try:
        # 获取股票数据
        data = get_stock_data(symbol, days=days)
        
        if data is None or len(data) < 20:
            logger.warning(f"{symbol}无足够数据进行风险评估")
            return "unknown", None
        
        # 计算风险指标
        # 1. 波动率 (标准差 * sqrt(交易日))
        volatility = data['close'].pct_change().std() * np.sqrt(252)
        
        # 2. 获取SPY数据作为市场基准
        market_data = get_stock_data('SPY', days=days)
        
        # 3. 计算贝塔系数
        beta = 1.0  # 默认值
        if market_data is not None and len(market_data) >= 20:
            stock_returns = data['close'].pct_change().dropna()
            market_returns = market_data['close'].pct_change().dropna()
            
            # 确保两个序列长度相同
            min_length = min(len(stock_returns), len(market_returns))
            stock_returns = stock_returns.iloc[-min_length:]
            market_returns = market_returns.iloc[-min_length:]
            
            # 计算协方差和市场方差
            covariance = stock_returns.cov(market_returns)
            market_variance = market_returns.var()
            
            # 计算贝塔系数
            beta = covariance / market_variance if market_variance != 0 else 1.0
        
        # 4. 最大回撤
        roll_max = data['close'].cummax()
        drawdown = (data['close'] - roll_max) / roll_max
        max_drawdown = drawdown.min()
        
        # 5. 夏普比率
        risk_free_rate = 0.02  # 假设无风险利率为2%
        daily_returns = data['close'].pct_change().dropna()
        excess_returns = daily_returns - risk_free_rate/252  # 每日无风险收益率
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0
        
        # 风险指标集合
        risk_metrics = {
            'volatility': volatility,
            'beta': beta,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
        
        # 风险分类
        if volatility > 0.3 or beta > 1.5 or max_drawdown < -0.3:
            risk_category = "high"
        elif volatility > 0.15 or beta > 1.0 or max_drawdown < -0.2:
            risk_category = "medium"
        else:
            risk_category = "low"
            
        logger.info(f"{symbol} 风险评估: {risk_category} (波动率: {volatility:.2f}, Beta: {beta:.2f}, 最大回撤: {max_drawdown:.2f})")
        return risk_category, risk_metrics
        
    except Exception as e:
        logger.error(f"评估{symbol}风险时出错: {str(e)}")
        return "unknown", None

def set_stop_loss(symbol, stop_percent=0.05):
    """
    为已有持仓设置止损单
    
    参数:
        symbol (str): 股票代码
        stop_percent (float): 止损百分比
        
    返回:
        dict or None: 止损订单信息
    """
    try:
        api = get_api_client()
        
        # 获取持仓信息
        position = api.get_position(symbol)
        current_price = float(position.current_price)
        qty = float(position.qty)
        
        # 检查是否已有止损单
        existing_orders = api.list_orders(status='open', limit=100)
        for order in existing_orders:
            if order.symbol == symbol and order.side == 'sell' and order.type == 'stop':
                logger.info(f"{symbol} 已有止损单，ID: {order.id}, 止损价: ${order.stop_price}")
                return {
                    'order_id': order.id,
                    'symbol': symbol,
                    'stop_price': float(order.stop_price),
                    'qty': float(order.qty),
                    'status': 'existing'
                }
        
        # 计算止损价格
        stop_price = round(current_price * (1 - stop_percent), 2)
        
        # 创建止损单
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side='sell',
            type='stop',
            time_in_force='gtc',  # 一直有效，直到取消
            stop_price=stop_price
        )
        
        logger.info(f"已为 {symbol} 设置止损单，止损价: ${stop_price}")
        
        # 记录交易日志
        from .construction import log_portfolio_transaction
        log_portfolio_transaction(symbol, "设置止损", stop_price, qty, "自动风险管理")
        
        return {
            'order_id': order.id,
            'symbol': symbol,
            'stop_price': stop_price,
            'qty': float(qty),
            'status': 'new'
        }
        
    except Exception as e:
        logger.error(f"设置止损单失败: {str(e)}")
        return None

def update_stop_loss(symbol, trail_percent=0.05):
    """
    更新止损价格为移动止损，并更早触发
    
    参数:
        symbol (str): 股票代码
        trail_percent (float): 移动止损百分比
        
    返回:
        dict or None: 更新后的止损订单信息
    """
    try:
        api = get_api_client()
        
        # 获取持仓信息
        position = api.get_position(symbol)
        current_price = float(position.current_price)
        entry_price = float(position.avg_entry_price)
        qty = float(position.qty)
        
        # 如果当前价格高于买入价，则计算移动止损
        if current_price > entry_price:
            profit_percent = (current_price - entry_price) / entry_price
            
            # 降低移动止损触发阈值从10%到5%
            if profit_percent > 0.05:
                # 取消现有止损单
                existing_orders = api.list_orders(status='open', limit=100)
                for order in existing_orders:
                    if order.symbol == symbol and order.side == 'sell' and order.type == 'stop':
                        api.cancel_order(order.id)
                        logger.info(f"已取消 {symbol} 的现有止损单")
                
                # 计算新止损价格 - 根据盈利水平调整保护幅度
                if profit_percent > 0.20:  # 超过20%利润，保护更多
                    stop_price = max(current_price * (1 - trail_percent * 0.5), entry_price * 1.10)  # 确保至少锁定10%利润
                    logger.info(f"{symbol} 盈利超过20%，设置更紧的移动止损以保护利润")
                elif profit_percent > 0.10:  # 10-20%利润区间
                    stop_price = max(current_price * (1 - trail_percent * 0.7), entry_price * 1.05)  # 确保至少锁定5%利润
                    logger.info(f"{symbol} 盈利超过10%，设置中等程度的移动止损")
                else:  # 5-10%利润区间
                    stop_price = max(current_price * (1 - trail_percent), entry_price * 1.01)  # 确保至少锁定1%利润
                
                # 提交新的止损单
                order = api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='sell',
                    type='stop',
                    time_in_force='gtc',
                    stop_price=stop_price
                )
                
                logger.info(f"已为 {symbol} 更新移动止损单，止损价: ${stop_price:.2f}")
                
                # 记录交易日志
                from .construction import log_portfolio_transaction
                log_portfolio_transaction(symbol, "更新止损", stop_price, qty, "移动止损")
                
                return {
                    'order_id': order.id,
                    'symbol': symbol,
                    'stop_price': stop_price,
                    'qty': float(qty),
                    'status': 'updated'
                }
                
            else:
                logger.info(f"{symbol} 盈利 {profit_percent:.2%}，尚未达到移动止损触发阈值(5%)")
                return {
                    'symbol': symbol,
                    'status': 'unchanged',
                    'profit_percent': profit_percent
                }
        else:
            logger.info(f"{symbol} 当前亏损中，不更新止损")
            return {
                'symbol': symbol,
                'status': 'unchanged',
                'profit_percent': (current_price - entry_price) / entry_price
            }
    
    except Exception as e:
        logger.error(f"更新止损单失败: {str(e)}")
        return None

def calculate_portfolio_risk(positions):
    """
    计算投资组合总体风险
    
    参数:
        positions (list): 持仓列表
        
    返回:
        dict: 风险指标
    """
    try:
        if not positions:
            logger.warning("无持仓，无法计算投资组合风险")
            return None
            
        # 获取持仓股票的历史数据
        symbols = [p['symbol'] for p in positions]
        historical_data = {}
        
        for symbol in symbols:
            data = get_stock_data(symbol, days=120)
            if data is not None and len(data) > 20:
                historical_data[symbol] = data
        
        if not historical_data:
            logger.warning("无法获取持仓股票的历史数据")
            return None
            
        # 计算每只股票的权重
        total_value = sum(p['market_value'] for p in positions)
        weights = {p['symbol']: p['market_value'] / total_value for p in positions}
        
        # 计算每只股票的日收益率
        returns = {}
        for symbol, data in historical_data.items():
            returns[symbol] = data['close'].pct_change().dropna()
        
        # 找出共同的日期范围
        common_dates = None
        for symbol, ret in returns.items():
            if common_dates is None:
                common_dates = set(ret.index)
            else:
                common_dates = common_dates.intersection(set(ret.index))
        
        common_dates = sorted(list(common_dates))
        
        if len(common_dates) < 20:
            logger.warning("共同的历史数据不足，无法精确计算投资组合风险")
            return None
            
        # 创建收益率矩阵
        returns_matrix = pd.DataFrame(index=common_dates)
        for symbol, ret in returns.items():
            returns_matrix[symbol] = ret.loc[common_dates]
        
        # 计算投资组合日收益率
        weighted_returns = pd.Series(0, index=common_dates)
        for symbol in symbols:
            if symbol in weights and symbol in returns_matrix.columns:
                weighted_returns += returns_matrix[symbol] * weights.get(symbol, 0)
        
        # 计算风险指标
        portfolio_volatility = weighted_returns.std() * np.sqrt(252)
        portfolio_return = weighted_returns.mean() * 252
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility != 0 else 0
        
        # 计算最大回撤
        cum_returns = (1 + weighted_returns).cumprod()
        max_return = cum_returns.cummax()
        drawdown = (cum_returns - max_return) / max_return
        max_drawdown = drawdown.min()
        
        risk_metrics = {
            'volatility': portfolio_volatility,
            'annual_return': portfolio_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
        logger.info(f"投资组合风险分析: 波动率={portfolio_volatility:.4f}, 年化收益={portfolio_return:.4f}, 夏普比率={sharpe_ratio:.4f}")
        return risk_metrics
        
    except Exception as e:
        logger.error(f"计算投资组合风险时出错: {str(e)}")
        return None

def check_portfolio_risk_limits(positions, max_drawdown_limit=-0.15, max_var_limit=0.1):
    """
    检查投资组合风险是否超过限制
    
    参数:
        positions (list): 持仓列表
        max_drawdown_limit (float): 最大回撤限制
        max_var_limit (float): 最大风险价值限制
        
    返回:
        dict: 风险评估结果
    """
    try:
        # 计算投资组合风险
        risk_metrics = calculate_portfolio_risk(positions)
        
        if risk_metrics is None:
            return {
                'status': 'unknown',
                'message': '无法计算投资组合风险'
            }
            
        # 检查最大回撤是否超过限制
        if risk_metrics['max_drawdown'] < max_drawdown_limit:
            logger.warning(f"投资组合最大回撤 ({risk_metrics['max_drawdown']:.2%}) 超过限制 ({max_drawdown_limit:.2%})")
            return {
                'status': 'exceeded',
                'metric': 'max_drawdown',
                'value': risk_metrics['max_drawdown'],
                'limit': max_drawdown_limit,
                'message': f"最大回撤超过限制: {risk_metrics['max_drawdown']:.2%} > {max_drawdown_limit:.2%}"
            }
            
        # 计算简化的风险价值(VaR)，假设正态分布，95%置信度
        var_95 = 1.65 * risk_metrics['volatility'] / np.sqrt(252)
        
        if var_95 > max_var_limit:
            logger.warning(f"投资组合日风险价值 ({var_95:.2%}) 超过限制 ({max_var_limit:.2%})")
            return {
                'status': 'exceeded',
                'metric': 'var',
                'value': var_95,
                'limit': max_var_limit,
                'message': f"日风险价值超过限制: {var_95:.2%} > {max_var_limit:.2%}"
            }
            
        logger.info("投资组合风险在可接受范围内")
        return {
            'status': 'acceptable',
            'risk_metrics': risk_metrics,
            'message': '投资组合风险在可接受范围内'
        }
        
    except Exception as e:
        logger.error(f"检查投资组合风险限制时出错: {str(e)}")
        return {
            'status': 'error',
            'message': f'检查风险限制时出错: {str(e)}'
        }