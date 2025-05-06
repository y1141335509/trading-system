# test_trading_decision.py
import logging
from src.trading_system import hybrid_trading_decision, get_market_context
from src.data.fetcher import get_stock_data
from src.data.processor import calculate_technical_indicators
from src.ml.train import load_model
from src.utils.config import get_data_paths
import os

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 测试股票列表
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

# 获取市场上下文
logger.info("获取市场上下文...")
context = get_market_context()
logger.info(f"市场状态: {context['market_regime']}, VIX水平: {context['vix_level']}")

# 获取模型目录
model_dir = get_data_paths()['model_dir']

# 针对每只股票进行决策测试
for symbol in symbols:
    logger.info(f"\n分析 {symbol}...")
    
    # 获取股票数据
    data = get_stock_data(symbol)
    
    if data is None or len(data) < 20:
        logger.warning(f"无足够的{symbol}数据进行决策")
        continue
    
    # 计算技术指标
    data = calculate_technical_indicators(data)
    
    # 加载机器学习模型
    ml_model, ml_scaler = load_model(symbol, model_dir)
    
    # 强化学习模型路径
    rl_model_path = os.path.join(model_dir, f"{symbol}_rl_model.h5")
    if not os.path.exists(rl_model_path):
        logger.info(f"{symbol}的强化学习模型不存在")
        rl_model_path = None
    
    # 执行决策逻辑
    decision = hybrid_trading_decision(
        symbol, 
        context,
        data=data,
        ml_model=ml_model, 
        ml_scaler=ml_scaler, 
        rl_model_path=rl_model_path
    )
    
    logger.info(f"{symbol}的最终决策: {decision}")