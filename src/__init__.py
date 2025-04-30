# src/__init__.py

"""
Trading System Package

A comprehensive algorithmic trading system built with Python and Alpaca API,
supporting both paper trading and live trading environments.
"""

# Version information
__version__ = '0.1.0'
__author__ = 'Trading System Developer'

# Import sub-packages
from . import data
from . import indicators
from . import ml
from . import portfolio
from . import reporting
from . import rl
from . import utils

# Import main classes
from .trading_system import (
    run_intelligent_trading_system,
    monitor_market,
    setup_scheduled_monitoring
)

from .analyzer import (
    analyze_performance,
    show_historical_pnl,
    evaluate_all_models
)

__all__ = [
    'data',
    'indicators',
    'ml',
    'portfolio',
    'reporting',
    'rl',
    'utils',
    'run_intelligent_trading_system',
    'monitor_market',
    'setup_scheduled_monitoring',
    'analyze_performance',
    'show_historical_pnl',
    'evaluate_all_models'
]