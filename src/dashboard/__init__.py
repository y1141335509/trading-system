# src/dashboard/__init__.py

"""
交易系统仪表盘模块

为交易系统提供可视化和数据访问功能的Web应用
"""

from .app import create_app, run_dashboard
from .api import get_performance_data, get_positions_data, get_trades_data

__all__ = [
    'create_app',
    'run_dashboard',
    'get_performance_data',
    'get_positions_data',
    'get_trades_data'
]