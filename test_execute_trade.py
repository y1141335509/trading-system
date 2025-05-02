# test_execute_trade.py
from trading_system import execute_trade
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 测试执行交易
logger.info("测试执行交易功能...")

# 注意：此处使用极小的数量进行测试
result = execute_trade("AAPL", "买入", qty=0.01)  # 买入很小数量的苹果股票

logger.info(f"交易结果: {result}")