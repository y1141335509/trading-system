# src/dashboard/api.py

import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import logging

# 导入系统组件
from ..data.fetcher import get_api_client
from ..reporting.performance import get_pnl_history, calculate_performance_metrics
from ..utils.config import get_data_paths

# 设置日志
logger = logging.getLogger(__name__)

def get_performance_data(days=30):
    """
    获取性能数据
    
    参数:
        days (int): 要查询的天数
        
    返回:
        dict: 性能数据字典
    """
    try:
        # 获取PnL历史
        pnl_data = get_pnl_history(days=days)
        
        if pnl_data.empty:
            logger.warning("没有找到PnL历史数据")
            return {
                'status': 'error',
                'message': '没有找到PnL历史数据'
            }
        
        # 计算性能指标
        metrics = calculate_performance_metrics(pnl_data)
        
        # 转换为JSON友好格式
        result = {
            'status': 'success',
            'days': days,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': metrics,
            'daily_data': []
        }
        
        # 添加日数据
        for _, row in pnl_data.iterrows():
            day_data = {
                'date': row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], datetime) else row['date'],
                'equity': float(row['equity']) if 'equity' in row else None,
                'daily_pnl': float(row['daily_pnl']) if 'daily_pnl' in row else None,
                'daily_return_pct': float(row['daily_return_pct']) if 'daily_return_pct' in row else None,
                'spy_return_pct': float(row['spy_return_pct']) if 'spy_return_pct' in row else None
            }
            
            # 添加累积回报
            if 'cumulative_return' not in row and len(result['daily_data']) > 0:
                prev_return = 0
                if len(result['daily_data']) > 0:
                    prev_return = result['daily_data'][-1].get('cumulative_return', 0)
                day_data['cumulative_return'] = prev_return + day_data['daily_return_pct']/100
            else:
                day_data['cumulative_return'] = float(row['cumulative_return']) if 'cumulative_return' in row else None
                
            result['daily_data'].append(day_data)
        
        return result
        
    except Exception as e:
        logger.error(f"获取性能数据失败: {str(e)}")
        return {
            'status': 'error',
            'message': str(e)
        }

def get_positions_data():
    """
    获取当前持仓数据
    
    返回:
        dict: 持仓数据字典
    """
    try:
        # 获取API客户端
        api = get_api_client()
        
        # 获取持仓信息
        positions = api.list_positions()
        
        # 转换为JSON友好格式
        result = {
            'status': 'success',
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'positions': []
        }
        
        # 计算总市值
        total_value = sum(float(position.market_value) for position in positions)
        result['total_value'] = total_value
        
        # 获取风险类别
        from ..portfolio.risk import assess_stock_risk
        
        for position in positions:
            # 获取风险类别
            risk_category, _ = assess_stock_risk(position.symbol)
            
            pos_data = {
                'symbol': position.symbol,
                'quantity': float(position.qty),
                'entry_price': float(position.avg_entry_price),
                'current_price': float(position.current_price),
                'market_value': float(position.market_value),
                'cost_basis': float(position.cost_basis),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc) * 100,
                'weight': float(position.market_value) / total_value if total_value > 0 else 0,
                'risk_category': risk_category
            }
            
            result['positions'].append(pos_data)
        
        return result
        
    except Exception as e:
        logger.error(f"获取持仓数据失败: {str(e)}")
        return {
            'status': 'error',
            'message': str(e)
        }

def get_trades_data(days=7):
    """
    获取最近交易数据
    
    参数:
        days (int): 要查询的天数
        
    返回:
        dict: 交易数据字典
    """
    try:
        # 获取API客户端
        api = get_api_client()
        
        # 计算起始日期
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # 获取订单历史
        start_date_str = start_date.strftime('%Y-%m-%d')
        orders = api.list_orders(
            status='all',
            after=start_date_str,
            limit=100,
            direction='desc'
        )
        
        # 转换为JSON友好格式
        result = {
            'status': 'success',
            'days': days,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'trades': []
        }
        
        # 过滤已成交订单
        filled_orders = [order for order in orders if order.status == 'filled']
        
        for order in filled_orders:
            # 计算交易金额
            quantity = float(order.qty) if order.qty else 0
            price = float(order.filled_avg_price) if order.filled_avg_price else 0
            amount = quantity * price
            
            # 转换日期格式
            created_at = order.created_at
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            
            filled_at = order.filled_at
            if isinstance(filled_at, str) and filled_at:
                filled_at = datetime.fromisoformat(filled_at.replace('Z', '+00:00'))
            
            trade_data = {
                'symbol': order.symbol,
                'side': order.side,
                'quantity': quantity,
                'price': price,
                'amount': amount,
                'created_at': created_at.strftime('%Y-%m-%d %H:%M:%S') if isinstance(created_at, datetime) else created_at,
                'filled_at': filled_at.strftime('%Y-%m-%d %H:%M:%S') if isinstance(filled_at, datetime) else filled_at,
                'order_type': order.type,
                'order_id': order.id
            }
            
            result['trades'].append(trade_data)
        
        return result
        
    except Exception as e:
        logger.error(f"获取交易数据失败: {str(e)}")
        return {
            'status': 'error',
            'message': str(e)
        }

def get_reports_list():
    """
    获取已生成报告列表
    
    返回:
        dict: 报告列表
    """
    try:
        # 获取报告目录
        data_paths = get_data_paths()
        report_dir = os.path.join(data_paths['data_dir'], 'reports')
        
        if not os.path.exists(report_dir):
            return {
                'status': 'error',
                'message': '报告目录不存在'
            }
        
        # 获取报告文件
        report_files = []
        for file in os.listdir(report_dir):
            if file.startswith('daily_report_') and file.endswith('.html'):
                file_path = os.path.join(report_dir, file)
                file_date = file.replace('daily_report_', '').replace('.html', '')
                
                # 获取文件大小和修改时间
                stats = os.stat(file_path)
                size_kb = stats.st_size / 1024
                modified = datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                
                report_files.append({
                    'filename': file,
                    'date': file_date,
                    'path': file_path,
                    'size_kb': size_kb,
                    'modified': modified
                })
        
        # 按日期排序
        report_files.sort(key=lambda x: x['date'], reverse=True)
        
        return {
            'status': 'success',
            'count': len(report_files),
            'reports': report_files
        }
        
    except Exception as e:
        logger.error(f"获取报告列表失败: {str(e)}")
        return {
            'status': 'error',
            'message': str(e)
        }

def get_system_status():
    """
    获取系统状态
    
    返回:
        dict: 系统状态
    """
    try:
        # 获取API客户端
        api = get_api_client()
        
        # 获取账户信息
        account = api.get_account()
        
        # 获取市场状态
        clock = api.get_clock()
        
        # 检查健康文件
        data_paths = get_data_paths()
        health_file = os.path.join(data_paths['data_dir'], 'health.txt')
        last_check = None
        
        if os.path.exists(health_file):
            with open(health_file, 'r') as f:
                last_check = f.read().strip()
        
        # 检查上次重训时间
        retrain_file = os.path.join(data_paths['data_dir'], 'last_retrain.txt')
        last_retrain = None
        
        if os.path.exists(retrain_file):
            with open(retrain_file, 'r') as f:
                last_retrain = f.read().strip()
        
        return {
            'status': 'success',
            'account': {
                'id': account.id,
                'status': account.status,
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'daytrade_count': int(account.daytrade_count)
            },
            'market': {
                'is_open': clock.is_open,
                'next_open': clock.next_open.strftime('%Y-%m-%d %H:%M:%S') if clock.next_open else None,
                'next_close': clock.next_close.strftime('%Y-%m-%d %H:%M:%S') if clock.next_close else None,
                'timestamp': clock.timestamp.strftime('%Y-%m-%d %H:%M:%S') if clock.timestamp else None
            },
            'system': {
                'last_health_check': last_check,
                'last_model_retrain': last_retrain,
                'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {str(e)}")
        return {
            'status': 'error',
            'message': str(e),
            'system': {
                'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }