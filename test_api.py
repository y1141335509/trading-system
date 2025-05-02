# test_api.py
import os
from alpaca_trade_api import REST

is_paper = os.getenv('ALPACA_PAPER', 'true').lower() == 'true'

if is_paper:
    API_KEY = os.getenv('ALPACA_PAPER_API_KEY')
    API_SECRET = os.getenv('ALPACA_PAPER_API_SECRET')
    BASE_URL = os.getenv('ALPACA_PAPER_URL', 'https://paper-api.alpaca.markets')
else:
    API_KEY = os.getenv('ALPACA_LIVE_API_KEY')
    API_SECRET = os.getenv('ALPACA_LIVE_API_SECRET')
    BASE_URL = os.getenv('ALPACA_LIVE_URL', 'https://api.alpaca.markets')

print(f"使用模式: {'Paper' if is_paper else 'Live'}")
print(f"API KEY: {API_KEY[:5]}...")
print(f"BASE URL: {BASE_URL}")

api = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

try:
    account = api.get_account()
    print(f"连接成功! 账户价值: ${account.portfolio_value}")
    
    # 测试获取市场数据
    aapl = api.get_latest_trade('AAPL')
    print(f"AAPL 最新价格: ${aapl.price}")
    
    # 测试下单功能（不实际执行）
    print("尝试检查订单功能...")
    orders = api.list_orders(limit=5)
    print(f"最近订单数: {len(orders)}")
    
except Exception as e:
    print(f"API连接或操作失败: {str(e)}")