# run_trading_cycle.py
from src.trading_system import run_intelligent_trading_system
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义要交易的股票
symbols = [
    # 科技股
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'CRM', 'ADBE', 'CSCO', 'ORCL', 'IBM', 'QCOM', 'NFLX',
    
    # 金融股
    'JPM', 'BAC', 'GS', 'MS', 'V', 'MA', 'WFC', 'C', 'AXP', 'BLK', 'COF', 'USB', 'PNC', 'SCHW',
    
    # 医疗健康
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'LLY', 'BMY', 'TMO', 'DHR', 'ABT', 'AMGN', 'CVS', 'GILD', 'ISRG', 'MDT',
    
    # 消费品
    'WMT', 'PG', 'KO', 'PEP', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'EL', 'CL', 'YUM', 'DIS',
    
    # 能源
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'PSX', 'VLO', 'MPC',
    
    # 工业股
    'HON', 'UNP', 'UPS', 'CAT', 'DE', 'RTX', 'LMT', 'GE', 'BA', 'MMM',
    
    # 电信
    'T', 'VZ', 'TMUS',
    
    # 房地产
    'AMT', 'EQIX', 'PLD', 'SPG', 'O', 'WELL',
    
    # ETFs
    'SPY', 'QQQ', 'IWM', 'VTI', 'XLK', 'XLF', 'XLV', 'XLP', 'XLE', 'ARKK', 'VGT', 'VOO', 'VUG', 'VYM', 'SOXX'
]

# 执行一次交易循环
logger.info("手动执行一次交易循环...")
run_intelligent_trading_system(symbols=symbols, schedule_retrain_enabled=False)