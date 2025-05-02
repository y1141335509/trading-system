# run_trading_cycle.py
from src.trading_system import run_intelligent_trading_system
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义要交易的股票
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

# 执行一次交易循环
logger.info("手动执行一次交易循环...")
run_intelligent_trading_system(symbols=symbols, schedule_retrain_enabled=False)