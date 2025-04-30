# src/portfolio/rebalance.py

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from ..data.fetcher import get_api_client, get_stock_data
from .risk import assess_stock_risk

# 设置日志
logger = logging.getLogger(__name__)

def analyze_portfolio_allocation(positions=None):
    """
    分析当前投资组合分配
    
    参数:
        positions (list): 持仓列表，如果为None则从API获取
        
    返回:
        dict: 投资组合分配分析
    """
    try:
        # 如果没有提供持仓，从API获取
        if positions is None:
            api = get_api_client()
            positions_api = api.list_positions()
            
            positions = []
            for position in positions_api:
                positions.append({
                    'symbol': position.symbol,
                    'qty': float(position.qty),
                    'avg_entry_price': float(position.avg_entry_price),
                    'current_price': float(position.current_price),
                    'market_value': float(position.market_value),
                    'cost_basis': float(position.cost_basis),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc) * 100
                })
        
        if not positions:
            logger.info("当前无持仓")
            return {'status': 'empty', 'message': '当前无持仓'}
        
        # 计算总市值
        total_value = sum(position['market_value'] for position in positions)
        
        # 按风险类别分类当前持仓
        categorized_positions = {'low': [], 'medium': [], 'high': [], 'unknown': []}
        risk_allocation = {'low': 0, 'medium': 0, 'high': 0, 'unknown': 0}
        
        # 分析每个持仓
        for position in positions:
            symbol = position['symbol']
            market_value = position['market_value']
            weight = market_value / total_value if total_value > 0 else 0

            # 获取风险类别
            risk_category, _ = assess_stock_risk(symbol)
            
            position['risk_category'] = risk_category
            position['weight'] = weight
            
            # 添加到对应风险类别
            categorized_positions[risk_category].append(position)
            risk_allocation[risk_category] += weight
        
        # 分析每个风险类别
        analysis = {
            'total_value': total_value,
            'position_count': len(positions),
            'risk_allocation': risk_allocation,
            'categorized_positions': categorized_positions,
            'status': 'success'
        }
        
        logger.info(f"投资组合分析完成: 总市值=${total_value:.2f}, 风险分配: 低={risk_allocation['low']:.2%}, 中={risk_allocation['medium']:.2%}, 高={risk_allocation['high']:.2%}")
        return analysis
        
    except Exception as e:
        logger.error(f"分析投资组合分配时出错: {str(e)}")
        return {'status': 'error', 'message': str(e)}

def rebalance_portfolio(risk_allocation={'low': 0.5, 'medium': 0.3, 'high': 0.2}, threshold=0.1):
    """
    检查并再平衡投资组合
    
    参数:
        risk_allocation (dict): 目标风险分配比例
        threshold (float): 触发再平衡的阈值
        
    返回:
        dict: 再平衡结果
    """
    try:
        logger.info("开始检查投资组合平衡...")
        
        # 获取当前持仓分析
        current_analysis = analyze_portfolio_allocation()
        
        if current_analysis['status'] == 'empty':
            logger.info("当前无持仓，无需再平衡")
            return {'status': 'empty', 'message': '当前无持仓，无需再平衡'}
        
        if current_analysis['status'] == 'error':
            logger.error(f"获取当前持仓分析失败: {current_analysis['message']}")
            return current_analysis
        
        # 获取当前风险分配
        current_allocation = current_analysis['risk_allocation']
        total_value = current_analysis['total_value']
        
        # 检查是否需要再平衡
        needs_rebalance = False
        adjustments = {}
        
        for category, target_ratio in risk_allocation.items():
            current = current_allocation.get(category, 0)
            if abs(current - target_ratio) > threshold:
                needs_rebalance = True
                adjustments[category] = {
                    'current': current,
                    'target': target_ratio,
                    'diff': target_ratio - current,
                    'amount': (target_ratio - current) * total_value
                }
                logger.info(f"{category.capitalize()}风险资产偏离目标: 当前 {current:.2%} vs 目标 {target_ratio:.2%}")
        
        if not needs_rebalance:
            logger.info("投资组合平衡，无需调整")
            return {
                'status': 'balanced',
                'message': '投资组合平衡，无需调整',
                'current_allocation': current_allocation
            }
        
        # 生成再平衡建议
        rebalance_actions = []
        
        # 处理需要减少的类别
        for category, adjustment in adjustments.items():
            if adjustment['diff'] < -threshold:  # 需要减少持仓
                positions = current_analysis['categorized_positions'][category]
                
                # 按照盈利程度排序，先卖出盈利最多的
                positions_sorted = sorted(positions, key=lambda x: x.get('unrealized_plpc', 0), reverse=True)
                
                amount_to_reduce = -adjustment['amount']  # 转为正数
                remaining_amount = amount_to_reduce
                
                for position in positions_sorted:
                    if remaining_amount <= 0:
                        break
                        
                    symbol = position['symbol']
                    market_value = position['market_value']
                    
                    # 确定卖出的金额和比例
                    sell_amount = min(market_value * 0.8, remaining_amount)  # 最多卖出80%
                    sell_percent = sell_amount / market_value
                    
                    rebalance_actions.append({
                        'action': 'sell',
                        'symbol': symbol,
                        'risk_category': category,
                        'amount': sell_amount,
                        'percent': sell_percent,
                        'reason': f"减少{category}风险资产"
                    })
                    
                    remaining_amount -= sell_amount
        
        # 处理需要增加的类别
        for category, adjustment in adjustments.items():
            if adjustment['diff'] > threshold:  # 需要增加持仓
                # 获取新的可投资股票
                potential_stocks = get_potential_stocks(limit=50)
                
                # 筛选对应风险类别的股票
                risk_filtered_stocks = []
                for stock in potential_stocks:
                    symbol = stock['symbol']
                    stock_category, _ = assess_stock_risk(symbol)
                    
                    if stock_category == category:
                        risk_filtered_stocks.append(stock)
                
                # 选择前N只股票增加持仓
                amount_to_add = adjustment['amount']
                num_stocks_to_add = min(3, len(risk_filtered_stocks))
                
                if num_stocks_to_add > 0:
                    amount_per_stock = amount_to_add / num_stocks_to_add
                    
                    for i in range(num_stocks_to_add):
                        stock = risk_filtered_stocks[i]
                        symbol = stock['symbol']
                        price = stock['price']
                        
                        rebalance_actions.append({
                            'action': 'buy',
                            'symbol': symbol,
                            'risk_category': category,
                            'amount': amount_per_stock,
                            'shares': amount_per_stock / price,
                            'reason': f"增加{category}风险资产"
                        })
        
        result = {
            'status': 'needs_rebalance',
            'message': '投资组合需要再平衡',
            'current_allocation': current_allocation,
            'target_allocation': risk_allocation,
            'adjustments': adjustments,
            'actions': rebalance_actions
        }
        
        # 记录再平衡建议
        log_rebalance_recommendation(result)
        
        logger.info(f"生成了{len(rebalance_actions)}个再平衡操作建议")
        return result
        
    except Exception as e:
        logger.error(f"检查投资组合平衡时出错: {str(e)}")
        return {'status': 'error', 'message': str(e)}

def execute_rebalance(rebalance_plan, dry_run=True):
    """
    执行投资组合再平衡
    
    参数:
        rebalance_plan (dict): 再平衡计划
        dry_run (bool): 是否为模拟模式
        
    返回:
        dict: 执行结果
    """
    try:
        if rebalance_plan['status'] not in ['needs_rebalance']:
            logger.info(f"无需执行再平衡: {rebalance_plan['message']}")
            return rebalance_plan
            
        actions = rebalance_plan.get('actions', [])
        if not actions:
            logger.info("再平衡计划中没有具体操作")
            return {'status': 'no_actions', 'message': '再平衡计划中没有具体操作'}
            
        logger.info(f"开始执行{len(actions)}个再平衡操作" + (" (模拟模式)" if dry_run else ""))
        
        if dry_run:
            return {
                'status': 'simulated',
                'message': '模拟执行再平衡',
                'actions': actions
            }
            
        # 获取API客户端
        api = get_api_client()
        
        # 执行每个操作
        results = []
        
        for action in actions:
            try:
                symbol = action['symbol']
                
                if action['action'] == 'sell':
                    # 获取当前持仓数量
                    position = api.get_position(symbol)
                    qty = float(position.qty)
                    
                    # 计算要卖出的数量
                    sell_percent = action['percent']
                    sell_qty = qty * sell_percent
                    
                    # 提交市场卖单
                    order = api.submit_order(
                        symbol=symbol,
                        qty=sell_qty,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                    
                    logger.info(f"已提交卖出订单: {sell_qty} 股 {symbol}, 订单ID: {order.id}")
                    
                    # 记录结果
                    results.append({
                        'action': 'sell',
                        'symbol': symbol,
                        'qty': sell_qty,
                        'order_id': order.id,
                        'status': 'submitted'
                    })
                    
                elif action['action'] == 'buy':
                    # 计算要买入的数量
                    shares = action['shares']
                    
                    # 提交市场买单
                    order = api.submit_order(
                        symbol=symbol,
                        qty=shares,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )
                    
                    logger.info(f"已提交买入订单: {shares} 股 {symbol}, 订单ID: {order.id}")
                    
                    # 记录结果
                    results.append({
                        'action': 'buy',
                        'symbol': symbol,
                        'qty': shares,
                        'order_id': order.id,
                        'status': 'submitted'
                    })
                    
                # 记录交易
                from .construction import log_portfolio_transaction
                log_portfolio_transaction(
                    symbol,
                    action['action'],
                    0,  # 价格未知，由市场决定
                    action.get('shares', action.get('percent', 0)),
                    action['reason']
                )
                    
            except Exception as e:
                logger.error(f"执行{action['action']} {symbol}失败: {str(e)}")
                results.append({
                    'action': action['action'],
                    'symbol': symbol,
                    'status': 'failed',
                    'error': str(e)
                })
        
        return {
            'status': 'executed',
            'message': f"已执行{len(results)}个再平衡操作",
            'results': results
        }
            
    except Exception as e:
        logger.error(f"执行再平衡失败: {str(e)}")
        return {'status': 'error', 'message': str(e)}

def log_rebalance_recommendation(rebalance_plan):
    """
    记录再平衡建议到日志文件
    
    参数:
        rebalance_plan (dict): 再平衡计划
        
    返回:
        bool: 是否记录成功
    """
    try:
        # 创建日志目录
        data_dir = os.getenv('DATA_DIR', 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # 日志文件路径
        log_file = os.path.join(data_dir, 'rebalance_history.csv')
        
        # 判断是否需要创建文件头
        file_exists = os.path.isfile(log_file)
        
        # 获取当前时间
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 创建要记录的行
        actions = rebalance_plan.get('actions', [])
        
        with open(log_file, 'a') as f:
            if not file_exists:
                f.write("timestamp,status,action_count,low_current,medium_current,high_current,low_target,medium_target,high_target\n")
                
            current = rebalance_plan.get('current_allocation', {})
            target = rebalance_plan.get('target_allocation', {})
            
            f.write(f"{timestamp},{rebalance_plan['status']},{len(actions)},")
            f.write(f"{current.get('low', 0):.4f},{current.get('medium', 0):.4f},{current.get('high', 0):.4f},")
            f.write(f"{target.get('low', 0):.4f},{target.get('medium', 0):.4f},{target.get('high', 0):.4f}\n")
        
        # 如果有具体操作，也记录到详细日志
        if actions:
            detail_log_file = os.path.join(data_dir, 'rebalance_actions.csv')
            detail_exists = os.path.isfile(detail_log_file)
            
            with open(detail_log_file, 'a') as f:
                if not detail_exists:
                    f.write("timestamp,action,symbol,risk_category,amount,shares_percent,reason\n")
                    
                for action in actions:
                    symbol = action['symbol']
                    act = action['action']
                    category = action['risk_category']
                    reason = action['reason']
                    
                    # 获取金额和数量/比例
                    amount = action.get('amount', 0)
                    shares_percent = action.get('shares', action.get('percent', 0))
                    
                    f.write(f"{timestamp},{act},{symbol},{category},{amount:.2f},{shares_percent:.4f},\"{reason}\"\n")
        
        logger.info(f"再平衡建议已记录到日志")
        return True
        
    except Exception as e:
        logger.error(f"记录再平衡建议失败: {str(e)}")
        return False