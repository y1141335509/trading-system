# src/reporting/performance.py

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import json
import csv

# 设置日志
logger = logging.getLogger(__name__)

def get_api_client():
    """获取Alpaca API客户端"""
    is_paper = os.getenv('ALPACA_PAPER', 'true').lower() == 'true'
    
    if is_paper:
        API_KEY = os.getenv('ALPACA_PAPER_API_KEY')
        API_SECRET = os.getenv('ALPACA_PAPER_API_SECRET')
        BASE_URL = 'https://paper-api.alpaca.markets'
    else:
        API_KEY = os.getenv('ALPACA_LIVE_API_KEY')
        API_SECRET = os.getenv('ALPACA_LIVE_API_SECRET')
        BASE_URL = 'https://api.alpaca.markets'
    
    # Add this debug line
    logger.info(f"Using API with KEY: {API_KEY[:5]}..., BASE_URL: {BASE_URL}, PAPER: {is_paper}")
    
    return tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def track_daily_pnl(target_date=None):
    """
    记录每日盈亏到CSV文件
    
    参数:
        target_date (datetime, optional): 目标日期，默认为当天
        
    返回:
        tuple: (daily_pnl, daily_return_pct) 每日盈亏和收益率
    """
    try:
        if target_date is None:
            today = datetime.now()
        else:
            today = target_date
        
        date_str = today.strftime('%Y-%m-%d')
        
        # 获取API客户端
        api = get_api_client()
        
        # 获取账户信息
        account = api.get_account()
        portfolio_value = float(account.portfolio_value)
        equity = float(account.equity)
        cash = float(account.cash)
        
        # 计算日盈亏
        try:
            yesterday_equity = float(account.last_equity)
            daily_pnl = equity - yesterday_equity
            daily_return = (daily_pnl / yesterday_equity) * 100
        except:
            daily_pnl = 0
            daily_return = 0
        
        # 获取市场基准数据
        spy_return = 0
        try:
            spy_bars = api.get_bars('SPY', '1D', limit=2).df
            if len(spy_bars) >= 2:
                yesterday_close = spy_bars['close'].iloc[-2]
                today_close = spy_bars['close'].iloc[-1]
                spy_return = (today_close - yesterday_close) / yesterday_close * 100
        except Exception as e:
            logger.error(f"获取SPY数据失败: {str(e)}")
        
        # 计算相对表现
        relative_performance = daily_return - spy_return
        
        # 存储每日数据
        data_dir = os.getenv('DATA_DIR', 'data')
        os.makedirs(data_dir, exist_ok=True)
        pnl_file = os.path.join(data_dir, 'pnl_records.csv')
        file_exists = os.path.isfile(pnl_file)
        
        with open(pnl_file, 'a') as f:
            if not file_exists:
                f.write("date,portfolio_value,cash,equity,daily_pnl,daily_return_pct,spy_return_pct,relative_performance\n")
            
            f.write(f"{date_str},{portfolio_value},{cash},{equity},{daily_pnl},{daily_return},{spy_return},{relative_performance}\n")
        
        logger.info(f"每日盈亏已记录: ${daily_pnl:.2f} ({daily_return:.2f}%)")
        return daily_pnl, daily_return
        
    except Exception as e:
        logger.error(f"记录每日盈亏时出错: {str(e)}")
        return 0, 0

def calculate_performance_metrics(pnl_data):
    """
    计算进阶性能指标
    
    参数:
        pnl_data (DataFrame): 包含盈亏数据的DataFrame
        
    返回:
        dict: 性能指标
    """
    try:
        if pnl_data is None or pnl_data.empty:
            logger.warning("无盈亏数据，无法计算性能指标")
            return {}
        
        # 确保数据按日期排序
        df = pnl_data.copy()
        
        # 检查并转换日期列
        if 'date' in df.columns and not isinstance(df['date'].iloc[0], datetime):
            df['date'] = pd.to_datetime(df['date'])
        
        df = df.sort_values('date')
        
        # 计算基本指标
        total_days = len(df)
        win_days = len(df[df['daily_pnl'] > 0])
        loss_days = len(df[df['daily_pnl'] < 0])
        win_rate = win_days / total_days if total_days > 0 else 0
        
        # 计算累计回报
        cumulative_return = (1 + df['daily_return_pct']/100).prod() - 1
        
        # 计算年化回报
        days_count = (df['date'].max() - df['date'].min()).days
        if days_count > 0:
            annualized_return = (1 + cumulative_return) ** (365.0 / days_count) - 1
        else:
            annualized_return = 0
        
        # 计算波动率
        volatility = df['daily_return_pct'].std() * np.sqrt(252)  # 年化波动率
        
        # 计算夏普比率 (假设无风险利率为2%)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # 计算最大回撤
        df['cumulative_return'] = (1 + df['daily_return_pct']/100).cumprod()
        df['cumulative_max'] = df['cumulative_return'].cummax()
        df['drawdown'] = (df['cumulative_return'] / df['cumulative_max']) - 1
        max_drawdown = df['drawdown'].min()
        
        # 计算卡玛比率 (年化收益/最大回撤)
        calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0
        
        # 相对于市场的表现
        market_correlation = df['daily_return_pct'].corr(df['spy_return_pct'])
        avg_relative_performance = df['relative_performance'].mean()
        
        # 计算平均收益与亏损
        if win_days > 0:
            avg_win = df[df['daily_pnl'] > 0]['daily_pnl'].mean()
        else:
            avg_win = 0
            
        if loss_days > 0:
            avg_loss = df[df['daily_pnl'] < 0]['daily_pnl'].mean()
        else:
            avg_loss = 0
            
        # 计算盈亏比
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # 计算期望收益
        expectancy = (win_rate * avg_win + (1 - win_rate) * avg_loss)
        
        # 汇总指标
        metrics = {
            'total_days': total_days,
            'win_days': win_days,
            'loss_days': loss_days,
            'win_rate': win_rate,
            'total_pnl': df['daily_pnl'].sum(),
            'avg_daily_pnl': df['daily_pnl'].mean(),
            'max_daily_gain': df['daily_pnl'].max(),
            'max_daily_loss': df['daily_pnl'].min(),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_loss_ratio': profit_loss_ratio,
            'expectancy': expectancy,
            'cumulative_return': cumulative_return,
            'annualized_return': annualized_return,
            'volatility': volatility, 
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'market_correlation': market_correlation,
            'avg_relative_performance': avg_relative_performance
        }
        
        logger.info(f"性能指标计算完成: 总盈亏=${metrics['total_pnl']:.2f}, 胜率={win_rate:.2%}, 年化收益={annualized_return:.2%}")
        return metrics
        
    except Exception as e:
        logger.error(f"计算性能指标时出错: {str(e)}")
        return {}

def get_trade_history(days=30):
    """
    获取交易历史
    
    参数:
        days (int): 获取过去多少天的交易历史
        
    返回:
        DataFrame: 交易历史
    """
    try:
        # 获取API客户端
        api = get_api_client()
        
        # 计算开始日期
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # 获取活动订单历史
        orders = api.list_orders(
            status='all',
            limit=500,
            after=start_date.strftime('%Y-%m-%d'),
            until=end_date.strftime('%Y-%m-%d'),
            direction='desc'
        )
        
        # 转换为DataFrame
        if not orders:
            logger.info(f"过去{days}天没有交易记录")
            return pd.DataFrame()
            
        trades = []
        for order in orders:
            # 只考虑已成交的订单
            if order.status == 'filled':
                trade = {
                    'symbol': order.symbol,
                    'side': order.side,
                    'qty': float(order.qty),
                    'filled_qty': float(order.filled_qty),
                    'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else 0,
                    'order_type': order.type,
                    'created_at': pd.to_datetime(order.created_at),
                    'filled_at': pd.to_datetime(order.filled_at) if order.filled_at else None,
                    'order_id': order.id
                }
                
                # 计算交易金额
                trade['amount'] = trade['filled_qty'] * trade['filled_avg_price']
                
                trades.append(trade)
        
        # 创建DataFrame
        trades_df = pd.DataFrame(trades)
        
        if not trades_df.empty:
            # 按时间排序
            trades_df = trades_df.sort_values('filled_at', ascending=False)
            
            logger.info(f"获取到{len(trades_df)}笔交易记录")
            return trades_df
        else:
            logger.info(f"过去{days}天没有已成交的交易记录")
            return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"获取交易历史时出错: {str(e)}")
        return pd.DataFrame()

def get_pnl_history(days=30):
    """
    获取盈亏历史
    
    参数:
        days (int): 获取过去多少天的盈亏历史
        
    返回:
        DataFrame: 盈亏历史
    """
    try:
        # 读取盈亏记录文件
        data_dir = os.getenv('DATA_DIR', 'data')
        pnl_file = os.path.join(data_dir, 'pnl_records.csv')
        
        if not os.path.exists(pnl_file):
            logger.warning("盈亏记录文件不存在")
            return pd.DataFrame()
            
        # 读取CSV文件
        df = pd.read_csv(pnl_file)
        
        # 转换日期列
        df['date'] = pd.to_datetime(df['date'])
        
        # 筛选最近N天的数据
        if days > 0:
            start_date = datetime.now() - timedelta(days=days)
            df = df[df['date'] >= start_date]
            
        # 按日期排序
        df = df.sort_values('date')
        
        logger.info(f"读取盈亏历史记录，共{len(df)}天")
        return df
        
    except Exception as e:
        logger.error(f"获取盈亏历史时出错: {str(e)}")
        return pd.DataFrame()

def generate_daily_report(target_date=None, format='html'):
    """
    生成每日交易报告
    
    参数:
        target_date (datetime, optional): 目标日期，默认为当天
        format (str): 报告格式，'html'或'text'
        
    返回:
        str: 报告文件路径
    """
    try:
        # 使用指定日期或当前日期
        if target_date is None:
            report_date = datetime.now()
        elif isinstance(target_date, str):
            report_date = datetime.strptime(target_date, '%Y-%m-%d')
        else:
            report_date = target_date

        # 格式化日期字符串
        date_str = report_date.strftime('%Y-%m-%d')
        
        # 记录指定日期的盈亏
        daily_pnl, daily_return = track_daily_pnl(target_date=report_date)
        
        # 创建报告文件夹
        data_dir = os.getenv('DATA_DIR', 'data')
        report_dir = os.path.join(data_dir, 'reports')
        os.makedirs(report_dir, exist_ok=True)
        
        # 获取API客户端
        api = get_api_client()
        
        # 1. 获取账户信息
        account = api.get_account()
        portfolio_value = float(account.portfolio_value)
        cash = float(account.cash)
        equity = float(account.equity)
        
        # 2. 获取持仓信息
        positions = api.list_positions()
        positions_data = []
        
        for position in positions:
            symbol = position.symbol
            entry_price = float(position.avg_entry_price)
            current_price = float(position.current_price)
            qty = float(position.qty)
            market_value = float(position.market_value)
            cost_basis = entry_price * qty
            unrealized_pl = float(position.unrealized_pl)
            unrealized_plpc = float(position.unrealized_plpc) * 100
            
            positions_data.append({
                'symbol': symbol,
                'quantity': qty,
                'entry_price': entry_price,
                'current_price': current_price,
                'cost_basis': cost_basis,
                'market_value': market_value,
                'unrealized_pl': unrealized_pl,
                'unrealized_plpc': unrealized_plpc
            })
        
        # 3. 获取今日交易
        day_start = datetime.combine(report_date.date(), datetime.min.time())
        day_end = datetime.combine(report_date.date(), datetime.max.time())
        
        # 按照Alpaca API要求的ISO格式转换
        day_start_iso = day_start.isoformat() + 'Z'
        day_end_iso = day_end.isoformat() + 'Z'

        try:
            # 获取指定日期范围内的订单
            orders = api.list_orders(
                status='all',
                after=day_start_iso,
                until=day_end_iso,
                limit=100
            )
            
            orders_data = []
            for order in orders:
                if order.status == 'filled':
                    symbol = order.symbol
                    side = order.side
                    qty = float(order.qty)
                    filled_price = float(order.filled_avg_price) if order.filled_avg_price else 0
                    filled_at = order.filled_at
                    
                    orders_data.append({
                        'symbol': symbol,
                        'side': side,
                        'quantity': qty,
                        'filled_price': filled_price,
                        'filled_at': filled_at
                    })
        except Exception as e:
            logger.error(f"获取{date_str}订单数据出错: {str(e)}")
            orders_data = []
            
        # 4. 获取市场数据
        market_data = {}
        market_symbols = ['SPY', 'QQQ', 'DIA']  # S&P 500, NASDAQ, Dow Jones
        
        for symbol in market_symbols:
            try:
                bars = api.get_bars(symbol, '1D', limit=2).df
                if len(bars) >= 2:
                    yesterday_close = bars['close'].iloc[-2]
                    today_close = bars['close'].iloc[-1]
                    market_return = (today_close - yesterday_close) / yesterday_close * 100
                    
                    market_data[symbol] = {
                        'price': today_close,
                        'change_percent': market_return
                    }
            except Exception as e:
                logger.error(f"获取{symbol}市场数据出错: {str(e)}")
        
        # 获取SPY回报率用于比较
        spy_return = market_data.get('SPY', {}).get('change_percent', 0)
        relative_performance = daily_return - spy_return
        
        # 5. 获取历史盈亏统计
        pnl_data = get_pnl_history(days=30)
        
        # 计算性能指标
        performance_metrics = calculate_performance_metrics(pnl_data) if not pnl_data.empty else {}
        
        # 6. 根据格式生成报告
        if format.lower() == 'html':
            return generate_html_report(
                date_str, 
                portfolio_value, 
                cash, 
                equity, 
                daily_pnl, 
                daily_return, 
                positions_data, 
                orders_data, 
                market_data, 
                performance_metrics,
                report_dir
            )
        else:
            return generate_text_report(
                date_str, 
                portfolio_value, 
                cash, 
                equity, 
                daily_pnl, 
                daily_return, 
                positions_data, 
                orders_data, 
                market_data, 
                report_dir
            )
    
    except Exception as e:
        logger.error(f"生成每日报告时出错: {str(e)}")
        return None

def generate_html_report(date_str, portfolio_value, cash, equity, daily_pnl, daily_return, 
                         positions_data, orders_data, market_data, performance_metrics, report_dir):
    """
    生成HTML格式的报告
    
    参数:
        各种报告所需数据
        
    返回:
        str: 报告文件路径
    """
    try:
        # 报告文件路径
        report_file = os.path.join(report_dir, f'daily_report_{date_str}.html')
        
        # 获取SPY回报率
        spy_return = market_data.get('SPY', {}).get('change_percent', 0)
        relative_performance = daily_return - spy_return
        
        # 计算总体持仓指标
        total_cost_basis = sum(p['cost_basis'] for p in positions_data) if positions_data else 0
        total_market_value = sum(p['market_value'] for p in positions_data) if positions_data else 0
        total_unrealized_pl = sum(p['unrealized_pl'] for p in positions_data) if positions_data else 0
        
        if total_cost_basis > 0:
            total_unrealized_plpc = (total_unrealized_pl / total_cost_basis) * 100
        else:
            total_unrealized_plpc = 0
        
        # 生成HTML报告
        with open(report_file, 'w') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>交易系统每日报告 - {date_str}</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; color: #333; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; color: #2c3e50; }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .metrics {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
        .metric-box {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 15px; width: 30%; }}
    </style>
</head>
<body>
    <h1>交易系统每日报告</h1>
    <p>生成日期: {date_str}</p>
    
    <div class="summary">
        <h2>账户摘要</h2>
        <table>
            <tr><th>指标</th><th>数值</th></tr>
            <tr><td>投资组合总价值</td><td>${portfolio_value:.2f}</td></tr>
            <tr><td>可用现金</td><td>${cash:.2f}</td></tr>
            <tr><td>净值</td><td>${equity:.2f}</td></tr>
            <tr><td>今日收益</td><td class="{('positive' if daily_return > 0 else 'negative') if daily_return != 0 else ''}">${daily_pnl:.2f} ({daily_return:.2f}%)</td></tr>
        </table>
    </div>
    
    <h2>市场概览</h2>
    <table>
        <tr><th>指数</th><th>价格</th><th>今日涨跌</th></tr>
    """)
            
            # 添加市场数据
            for symbol, data in market_data.items():
                market_return = data['change_percent']
                color_class = 'positive' if market_return > 0 else 'negative'
                f.write(f'<tr><td>{symbol}</td><td>${data["price"]:.2f}</td><td class="{color_class}">{market_return:.2f}%</td></tr>\n')
            
            f.write("""</table>
    
    <h2>当前持仓</h2>
    """)
            
            if positions_data:
                f.write("""<table>
        <tr>
            <th>股票</th>
            <th>数量</th>
            <th>买入价</th>
            <th>当前价</th>
            <th>成本</th>
            <th>市值</th>
            <th>盈亏</th>
            <th>盈亏%</th>
        </tr>
    """)
                
                # 添加持仓数据
                for pos in positions_data:
                    pl_class = 'positive' if pos['unrealized_pl'] > 0 else 'negative'
                    f.write(f"""<tr>
            <td>{pos['symbol']}</td>
            <td>{pos['quantity']:.2f}</td>
            <td>${pos['entry_price']:.2f}</td>
            <td>${pos['current_price']:.2f}</td>
            <td>${pos['cost_basis']:.2f}</td>
            <td>${pos['market_value']:.2f}</td>
            <td class="{pl_class}">${pos['unrealized_pl']:.2f}</td>
            <td class="{pl_class}">{pos['unrealized_plpc']:.2f}%</td>
        </tr>
    """)
                
                # 添加合计行
                total_pl_class = 'positive' if total_unrealized_pl > 0 else 'negative'
                f.write(f"""<tr style="font-weight: bold;">
            <td colspan="4">总计</td>
            <td>${total_cost_basis:.2f}</td>
            <td>${total_market_value:.2f}</td>
            <td class="{total_pl_class}">${total_unrealized_pl:.2f}</td>
            <td class="{total_pl_class}">{total_unrealized_plpc:.2f}%</td>
        </tr>
    </table>""")
            else:
                f.write("<p>当前无持仓</p>")
            
            f.write("""
    <h2>今日交易</h2>
    """)
            
            if orders_data:
                f.write("""<table>
        <tr>
            <th>股票</th>
            <th>操作</th>
            <th>数量</th>
            <th>成交价</th>
            <th>成交时间</th>
        </tr>
    """)
                
                # 添加订单数据
                for order in orders_data:
                    side_display = "买入" if order['side'] == 'buy' else "卖出"
                    side_class = "positive" if order['side'] == 'buy' else "negative"
                    
                    f.write(f"""<tr>
            <td>{order['symbol']}</td>
            <td class="{side_class}">{side_display}</td>
            <td>{order['quantity']:.2f}</td>
            <td>${order['filled_price']:.2f}</td>
            <td>{order['filled_at']}</td>
        </tr>
    """)
                
                f.write("</table>")
            else:
                f.write("<p>今日无交易</p>")
            
            # 添加性能指标
            if performance_metrics:
                f.write("""
    <h2>绩效指标</h2>
    <div class="metrics">
    """)
                
                metrics_to_display = [
                    {'name': '胜率', 'value': f"{performance_metrics.get('win_rate', 0)*100:.2f}%", 
                     'class': 'positive' if performance_metrics.get('win_rate', 0) > 0.5 else ''},
                    {'name': '累计收益', 'value': f"{performance_metrics.get('cumulative_return', 0)*100:.2f}%", 
                     'class': 'positive' if performance_metrics.get('cumulative_return', 0) > 0 else 'negative'},
                    {'name': '年化收益', 'value': f"{performance_metrics.get('annualized_return', 0)*100:.2f}%", 
                     'class': 'positive' if performance_metrics.get('annualized_return', 0) > 0 else 'negative'},
                    {'name': '夏普比率', 'value': f"{performance_metrics.get('sharpe_ratio', 0):.2f}", 
                     'class': 'positive' if performance_metrics.get('sharpe_ratio', 0) > 1 else ''},
                    {'name': '最大回撤', 'value': f"{performance_metrics.get('max_drawdown', 0)*100:.2f}%", 
                     'class': 'negative'},
                    {'name': '卡玛比率', 'value': f"{performance_metrics.get('calmar_ratio', 0):.2f}", 
                     'class': 'positive' if performance_metrics.get('calmar_ratio', 0) > 1 else ''}
                ]
                
                for metric in metrics_to_display:
                    f.write(f"""
        <div class="metric-box">
            <h3>{metric['name']}</h3>
            <p class="{metric['class']}">{metric['value']}</p>
        </div>
    """)
                
                f.write("""
    </div>
    """)
            
            # 添加市场表现对比
            performance_diff = daily_return - spy_return
            perf_class = 'positive' if performance_diff > 0 else 'negative'
            
            f.write(f"""
    <div class="summary">
        <h2>表现对比</h2>
        <p>相比市场基准(SPY)表现：</p>
        <table>
            <tr><th>指标</th><th>数值</th></tr>
            <tr><td>账户今日收益</td><td class="{('positive' if daily_return > 0 else 'negative') if daily_return != 0 else ''}">{daily_return:.2f}%</td></tr>
            <tr><td>SPY今日收益</td><td class="{('positive' if spy_return > 0 else 'negative') if spy_return != 0 else ''}">{spy_return:.2f}%</td></tr>
            <tr><td>相对表现</td><td class="{perf_class}">{performance_diff:.2f}%</td></tr>
        </table>
        </div>
        
        <p><i>该报告由自动交易系统生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i></p>
    </body>
    </html>""")
        
        logger.info(f"HTML报告已生成: {report_file}")
        return report_file
        
    except Exception as e:
        logger.error(f"生成HTML报告时出错: {str(e)}")
        return None

def generate_text_report(date_str, portfolio_value, cash, equity, daily_pnl, daily_return, 
                        positions_data, orders_data, market_data, report_dir):
    """
    生成文本格式的报告
    
    参数:
        各种报告所需数据
        
    返回:
        str: 报告文件路径
    """
    try:
        # 报告文件路径
        report_file = os.path.join(report_dir, f'daily_report_{date_str}.txt')
        
        # 获取SPY回报率
        spy_return = market_data.get('SPY', {}).get('change_percent', 0)
        relative_performance = daily_return - spy_return
        
        # 生成报告内容
        report = f"===== 每日报告 {date_str} =====\n"
        report += f"账户价值: ${portfolio_value:.2f}\n"
        report += f"现金: ${cash:.2f}\n"
        report += f"净值: ${equity:.2f}\n"
        report += f"日收益: ${daily_pnl:.2f} ({daily_return:.2f}%)\n\n"
        
        # 添加市场数据
        report += "市场概览:\n"
        for symbol, data in market_data.items():
            report += f"- {symbol}: {data['change_percent']:.2f}%\n"
        report += "\n"
        
        # 添加持仓信息
        if positions_data:
            report += "持仓情况:\n"
            for pos in positions_data:
                report += f"- {pos['symbol']}: {pos['quantity']:.2f}股, 成本:${pos['cost_basis']:.2f}, 市值:${pos['market_value']:.2f}, 盈亏:${pos['unrealized_pl']:.2f} ({pos['unrealized_plpc']:.2f}%)\n"
        else:
            report += "当前无持仓\n"
        report += "\n"
        
        # 添加今日交易
        if orders_data:
            report += "今日交易:\n"
            for order in orders_data:
                side = "买入" if order['side'] == 'buy' else "卖出"
                report += f"- {side} {order['quantity']:.2f}股 {order['symbol']} @ ${order['filled_price']:.2f}\n"
        else:
            report += "今日无交易\n"
        report += "\n"
        
        # 添加表现对比
        report += "表现对比:\n"
        report += f"- 账户今日收益: {daily_return:.2f}%\n"
        report += f"- SPY今日收益: {spy_return:.2f}%\n"
        report += f"- 相对表现: {relative_performance:.2f}%\n\n"
        
        report += f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # 写入文件
        with open(report_file, 'w') as f:
            f.write(report)
            
        logger.info(f"文本报告已生成: {report_file}")
        return report_file
        
    except Exception as e:
        logger.error(f"生成文本报告时出错: {str(e)}")
        return None

def export_to_csv(daily_summary=None, positions_data=None, orders_data=None, market_data=None):
    """
    将交易数据导出为CSV文件
    
    参数:
        daily_summary (dict): 每日摘要数据
        positions_data (list): 持仓数据
        orders_data (list): 订单数据
        market_data (dict): 市场数据
        
    返回:
        list: 导出的CSV文件路径列表
    """
    try:
        # 创建报告目录
        data_dir = os.getenv('DATA_DIR', 'data')
        report_dir = os.path.join(data_dir, 'reports')
        os.makedirs(report_dir, exist_ok=True)
        
        today = datetime.now().strftime('%Y-%m-%d')
        csv_files = []
        
        # 导出每日摘要
        if daily_summary:
            summary_file = os.path.join(report_dir, f'summary_{today}.csv')
            with open(summary_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Date', 'Portfolio Value', 'Cash', 'Equity', 'Daily PnL', 'Daily Return %', 'SPY Return %', 'Relative Performance'])
                writer.writerow([
                    today,
                    daily_summary['portfolio_value'],
                    daily_summary['cash'],
                    daily_summary['equity'],
                    daily_summary['daily_pnl'],
                    daily_summary['daily_return'],
                    daily_summary['spy_return'],
                    daily_summary['relative_performance']
                ])
            csv_files.append(summary_file)
        
        # 导出持仓数据
        if positions_data:
            positions_file = os.path.join(report_dir, f'positions_{today}.csv')
            with open(positions_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'symbol', 'quantity', 'entry_price', 'current_price', 'cost_basis', 
                    'market_value', 'unrealized_pl', 'unrealized_plpc'
                ])
                writer.writeheader()
                writer.writerows(positions_data)
            csv_files.append(positions_file)
        
        # 导出订单数据
        if orders_data:
            orders_file = os.path.join(report_dir, f'orders_{today}.csv')
            with open(orders_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'symbol', 'side', 'quantity', 'filled_price', 'filled_at'
                ])
                writer.writeheader()
                writer.writerows(orders_data)
            csv_files.append(orders_file)
        
        # 导出市场数据
        if market_data:
            market_file = os.path.join(report_dir, f'market_{today}.csv')
            with open(market_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Symbol', 'Price', 'Change %'])
                for symbol, data in market_data.items():
                    writer.writerow([symbol, data['price'], data['change_percent']])
            csv_files.append(market_file)
        
        logger.info(f"CSV数据文件已导出到{report_dir}目录")
        return csv_files
        
    except Exception as e:
        logger.error(f"导出CSV文件时出错: {str(e)}")
        return []