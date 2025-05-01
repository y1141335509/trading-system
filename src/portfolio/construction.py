# src/portfolio/construction.py

import os
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime
from alpaca_trade_api import REST

from ..data.fetcher import get_stock_data, get_latest_quotes

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

def get_potential_stocks(min_price=5, max_price=500, limit=50):
    """
    获取潜在投资股票列表
    
    参数:
        min_price (float): 最低股价筛选
        max_price (float): 最高股价筛选
        limit (int): 返回的最大股票数量
        
    返回:
        list: 股票信息列表
    """
    try:
        # 获取API客户端
        api = get_api_client()
        
        # 获取活跃股票列表
        active_assets = api.list_assets(status='active', asset_class='us_equity')
        
        # 过滤标准
        tradable_assets = [asset for asset in active_assets 
                          if asset.tradable and 
                          asset.fractionable and  # 支持零股交易
                          not asset.symbol.startswith(('GOOG', '$')) and  # 排除特定前缀
                          '.' not in asset.symbol]  # 排除特殊符号
        
        logger.info(f"找到 {len(tradable_assets)} 只可交易资产")
        
        # 对潜在股票获取当前市场数据
        potential_stocks = []
        
        # 限制处理的数量以避免API限制
        for asset in tradable_assets[:min(200, len(tradable_assets))]:  # 处理前200个活跃资产
            try:
                # 获取最新报价
                last_quote = api.get_latest_quote(asset.symbol)
                last_price = last_quote.ap if hasattr(last_quote, 'ap') else None
                
                # 价格筛选
                if last_price and min_price <= last_price <= max_price:
                    potential_stocks.append({
                        'symbol': asset.symbol,
                        'name': asset.name,
                        'price': last_price
                    })
                    
                # 一旦收集到足够多的股票就停止
                if len(potential_stocks) >= limit:
                    break
                    
                # 防止API速率限制
                time.sleep(0.1)
                
            except Exception as e:
                logger.debug(f"获取 {asset.symbol} 报价时出错: {str(e)}")
                # 忽略单个股票的错误，继续处理其他股票
                continue
        
        logger.info(f"筛选出 {len(potential_stocks)} 只潜在股票")
        return potential_stocks
        
    except Exception as e:
        logger.error(f"获取潜在股票时出错: {str(e)}")
        return []

def build_portfolio(capital=1000, risk_allocation={'low': 0.5, 'medium': 0.3, 'high': 0.2}, max_stocks=15):
    """
    根据风险偏好构建投资组合
    
    参数:
        capital (float): 投资资金
        risk_allocation (dict): 不同风险类别的资金分配比例
        max_stocks (int): 最大股票数量
        
    返回:
        dict: 投资组合配置
    """
    from .risk import assess_stock_risk
    
    logger.info(f"开始构建投资组合，资金: ${capital}，最大股票数: {max_stocks}")
    
    # 获取潜在股票
    potential_stocks = get_potential_stocks(limit=100)
    
    if not potential_stocks:
        logger.warning("未找到符合条件的股票")
        return {}
    
    # 评估每只股票的风险
    categorized_stocks = {'low': [], 'medium': [], 'high': [], 'unknown': []}
    
    for stock in potential_stocks:
        symbol = stock['symbol']
        risk_category, risk_metrics = assess_stock_risk(symbol)
        
        if risk_metrics:
            stock['risk_category'] = risk_category
            stock['risk_metrics'] = risk_metrics
            categorized_stocks[risk_category].append(stock)
    
    # 打印每个风险类别的股票数量
    for category, stocks in categorized_stocks.items():
        if category != 'unknown':
            logger.info(f"{category.capitalize()}风险股票: {len(stocks)}只")
    
    # 确定每个风险类别的资金分配
    allocation = {}
    for category, ratio in risk_allocation.items():
        stocks_in_category = categorized_stocks[category]
        if not stocks_in_category:
            continue
            
        # 该类别的总资金
        category_capital = capital * ratio
        
        # 确定该类别要选择的股票数量
        num_stocks = min(len(stocks_in_category), max(1, int(max_stocks * ratio)))
        
        # 根据风险指标选择股票
        # 低风险类别：优先选择波动性低的
        if category == 'low':
            selected_stocks = sorted(stocks_in_category, key=lambda x: x['risk_metrics']['volatility'])[:num_stocks]
        # 中等风险类别：平衡选择
        elif category == 'medium':
            # 按beta排序，接近1的更优先
            selected_stocks = sorted(stocks_in_category, key=lambda x: abs(x['risk_metrics']['beta'] - 1))[:num_stocks]
        # 高风险类别：可以承担更高波动性，但避免极端回撤
        else:
            # 按最大回撤排序
            selected_stocks = sorted(stocks_in_category, key=lambda x: x['risk_metrics']['max_drawdown'])[:num_stocks]
        
        # 计算每只股票的投资金额
        stock_capital = category_capital / len(selected_stocks)
        
        for stock in selected_stocks:
            symbol = stock['symbol']
            price = stock['price']
            
            # 计算可购买的股数（考虑小数股）
            shares = stock_capital / price
            
            allocation[symbol] = {
                'risk_category': category,
                'capital': stock_capital,
                'price': price,
                'shares': shares,
                'risk_metrics': stock['risk_metrics']
            }
    
    logger.info(f"投资组合已构建完成，包含 {len(allocation)} 只股票")
    return allocation

def execute_portfolio(portfolio, dry_run=True):
    """
    执行投资组合交易（购买分配的股票）
    
    参数:
        portfolio (dict): 投资组合配置
        dry_run (bool): 是否为模拟模式
        
    返回:
        bool: 是否执行成功
    """
    from .risk import set_stop_loss
    
    logger.info(f"开始执行投资组合交易..." + (" (模拟模式)" if dry_run else ""))
    
    if not portfolio:
        logger.warning("投资组合为空，无法执行")
        return False
    
    # 获取API客户端
    api = get_api_client()
    
    # 投资组合摘要
    total_investment = sum(info['capital'] for info in portfolio.values())
    logger.info(f"总投资金额: ${total_investment:.2f}")
    
    # 按风险类别分组
    by_risk = {'low': [], 'medium': [], 'high': []}
    for symbol, info in portfolio.items():
        by_risk[info['risk_category']].append(symbol)
    
    for risk, symbols in by_risk.items():
        if symbols:
            logger.info(f"{risk.capitalize()}风险组合: {', '.join(symbols)}")
    
    if dry_run:
        logger.info("模拟模式，不执行实际交易")
        return True
    
    # 执行实际交易
    executed_orders = []
    failed_orders = []
    
    for symbol, info in portfolio.items():
        shares = info['shares']
        
        if shares <= 0:
            continue
            
        try:
            # 检查是否已有持仓
            try:
                position = api.get_position(symbol)
                logger.info(f"已持有 {position.qty} 股 {symbol}，跳过")
                continue
            except:
                # 没有持仓，继续购买
                pass
                
            # 提交市场买单
            order = api.submit_order(
                symbol=symbol,
                qty=shares,
                side='buy',
                type='market',
                time_in_force='day'
            )
            
            logger.info(f"已提交买入订单: {shares} 股 {symbol}, 订单ID: {order.id}")
            executed_orders.append(order)
            
            # 为新买入的股票设置止损
            time.sleep(2)  # 等待订单处理
            order_status = api.get_order(order.id)
            
            if order_status.status == 'filled':
                set_stop_loss(symbol, stop_percent=0.05)
                
        except Exception as e:
            logger.error(f"买入 {symbol} 失败: {str(e)}")
            failed_orders.append(symbol)
    
    logger.info(f"成功执行 {len(executed_orders)} 个订单，失败 {len(failed_orders)} 个")
    return len(failed_orders) == 0

def log_portfolio_transaction(symbol, action, price, qty, reason):
    """
    记录投资组合交易
    
    参数:
        symbol (str): 股票代码
        action (str): 交易类型（买入/卖出）
        price (float): 交易价格
        qty (float): 交易数量
        reason (str): 交易原因
        
    返回:
        bool: 是否记录成功
    """
    try:
        # 创建日志目录
        data_dir = os.getenv('DATA_DIR', 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # 日志文件路径
        log_file = os.path.join(data_dir, 'portfolio_transactions.csv')
        
        # 判断是否需要创建文件头
        file_exists = os.path.isfile(log_file)
        
        # 记录交易
        with open(log_file, 'a') as f:
            if not file_exists:
                f.write("timestamp,symbol,action,price,quantity,reason\n")
                
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{timestamp},{symbol},{action},{price},{qty},\"{reason}\"\n")
        
        logger.info(f"交易记录已保存: {symbol} {action} {qty}股")
        return True
        
    except Exception as e:
        logger.error(f"记录交易失败: {str(e)}")
        return False