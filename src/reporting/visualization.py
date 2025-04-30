# src/reporting/visualization.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import logging
from datetime import datetime, timedelta
import io
import base64

# 设置日志
logger = logging.getLogger(__name__)

# 设置Seaborn风格
sns.set_style('whitegrid')

def plot_portfolio_performance(pnl_data, days=30, save_path=None, show=False):
    """
    绘制投资组合绩效图表
    
    参数:
        pnl_data (DataFrame): 盈亏数据
        days (int): 显示天数
        save_path (str): 保存路径
        show (bool): 是否显示图表
        
    返回:
        str: 保存的图表路径或None
    """
    try:
        if pnl_data is None or pnl_data.empty:
            logger.warning("无盈亏数据，无法绘制绩效图表")
            return None
        
        # 确保日期列是datetime类型
        if 'date' in pnl_data.columns:
            pnl_data['date'] = pd.to_datetime(pnl_data['date'])
        
        # 排序并限制天数
        df = pnl_data.sort_values('date')
        if days > 0 and days < len(df):
            df = df.tail(days)
        
        # 创建图表
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # 1. 每日盈亏条形图
        axes[0].bar(df['date'], df['daily_pnl'], 
                color=df['daily_pnl'].apply(lambda x: 'green' if x > 0 else 'red'))
        axes[0].set_title('每日盈亏 ($)', fontsize=14)
        axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0].grid(True, alpha=0.3)
        
        # 格式化y轴
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.0f}'))
        
        # 2. 账户价值曲线图
        axes[1].plot(df['date'], df['equity'], label='账户价值', color='blue')
        axes[1].set_title('账户价值变化 ($)', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        # 格式化y轴
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.0f}'))
        
        # 3. 累计回报对比
        df['strategy_cum_return'] = (1 + df['daily_return_pct']/100).cumprod() - 1
        df['spy_cum_return'] = (1 + df['spy_return_pct']/100).cumprod() - 1
        
        axes[2].plot(df['date'], df['strategy_cum_return']*100, 
                    label='策略', color='blue', linewidth=2)
        axes[2].plot(df['date'], df['spy_cum_return']*100, 
                    label='SPY', color='gray', linewidth=2, linestyle='--')
        axes[2].set_title('累计收益率对比 (%)', fontsize=14)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 格式化x轴日期
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # 格式化y轴
        axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
        
        # 设置x轴标签
        axes[2].set_xlabel('日期', fontsize=12)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path)
            logger.info(f"绩效图表已保存到: {save_path}")
        elif not show:
            # 如果没有指定保存路径且不显示，则保存到默认位置
            data_dir = os.getenv('DATA_DIR', 'data')
            report_dir = os.path.join(data_dir, 'reports')
            os.makedirs(report_dir, exist_ok=True)
            
            default_path = os.path.join(report_dir, f"pnl_chart_{datetime.now().strftime('%Y%m%d')}.png")
            plt.savefig(default_path)
            logger.info(f"绩效图表已保存到: {default_path}")
            save_path = default_path
            
        # 显示图表
        if show:
            plt.show()
        else:
            plt.close()
            
        return save_path
        
    except Exception as e:
        logger.error(f"绘制投资组合绩效图表时出错: {str(e)}")
        return None

def plot_trade_history(trades_data, days=30, save_path=None, show=False):
    """
    绘制交易历史图表
    
    参数:
        trades_data (DataFrame): 交易数据
        days (int): 显示天数
        save_path (str): 保存路径
        show (bool): 是否显示图表
        
    返回:
        str: 保存的图表路径或None
    """
    try:
        if trades_data is None or trades_data.empty:
            logger.warning("无交易数据，无法绘制交易历史图表")
            return None
            
        # 确保日期列是datetime类型
        if 'filled_at' in trades_data.columns:
            trades_data['filled_at'] = pd.to_datetime(trades_data['filled_at'])
            
            # 添加日期列用于聚合
            trades_data['date'] = trades_data['filled_at'].dt.date
            
        # 限制天数
        if days > 0:
            start_date = datetime.now() - timedelta(days=days)
            trades_data = trades_data[trades_data['filled_at'] >= start_date]
            
        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # 1. 交易类型分布
        trade_counts = trades_data.groupby('side').size()
        axes[0].bar(trade_counts.index, trade_counts.values, color=['red', 'green'])
        axes[0].set_title('交易类型分布', fontsize=14)
        axes[0].set_ylabel('交易次数')
        
        # 2. 每日交易金额
        trades_data['amount'] = trades_data['filled_qty'] * trades_data['filled_avg_price']
        
        # 按日期和类型聚合
        daily_amounts = trades_data.groupby(['date', 'side'])['amount'].sum().unstack(fill_value=0)
        
        if not daily_amounts.empty:
            # 如果有 'buy' 列
            if 'buy' in daily_amounts.columns:
                axes[1].bar(daily_amounts.index, daily_amounts['buy'], color='green', label='买入')
            
            # 如果有 'sell' 列    
            if 'sell' in daily_amounts.columns:
                axes[1].bar(daily_amounts.index, -daily_amounts['sell'], color='red', label='卖出')
                
            axes[1].set_title('每日交易金额', fontsize=14)
            axes[1].set_ylabel('交易金额 ($)')
            axes[1].legend()
            
            # 格式化y轴
            axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${abs(x):.0f}'))
            
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path)
            logger.info(f"交易历史图表已保存到: {save_path}")
        elif not show:
            # 如果没有指定保存路径且不显示，则保存到默认位置
            data_dir = os.getenv('DATA_DIR', 'data')
            report_dir = os.path.join(data_dir, 'reports')
            os.makedirs(report_dir, exist_ok=True)
            
            default_path = os.path.join(report_dir, f"trades_chart_{datetime.now().strftime('%Y%m%d')}.png")
            plt.savefig(default_path)
            logger.info(f"交易历史图表已保存到: {default_path}")
            save_path = default_path
            
        # 显示图表
        if show:
            plt.show()
        else:
            plt.close()
            
        return save_path
        
    except Exception as e:
        logger.error(f"绘制交易历史图表时出错: {str(e)}")
        return None

def create_performance_dashboard(pnl_data, trades_data=None, days=30, save_path=None, show=False):
    """
    创建综合绩效仪表板
    
    参数:
        pnl_data (DataFrame): 盈亏数据
        trades_data (DataFrame): 交易数据
        days (int): 显示天数
        save_path (str): 保存路径
        show (bool): 是否显示图表
        
    返回:
        str: 保存的图表路径或None
    """
    try:
        if pnl_data is None or pnl_data.empty:
            logger.warning("无盈亏数据，无法创建绩效仪表板")
            return None
            
        # 确保日期列是datetime类型
        if 'date' in pnl_data.columns:
            pnl_data['date'] = pd.to_datetime(pnl_data['date'])
            
        # 排序并限制天数
        df = pnl_data.sort_values('date')
        if days > 0 and days < len(df):
            df = df.tail(days)
            
        # 计算性能指标
        total_days = len(df)
        win_days = len(df[df['daily_pnl'] > 0])
        loss_days = len(df[df['daily_pnl'] < 0])
        win_rate = win_days / total_days if total_days > 0 else 0
        
        cumulative_return = (1 + df['daily_return_pct']/100).prod() - 1
        benchmark_return = (1 + df['spy_return_pct']/100).prod() - 1
        
        # 创建图表
        fig = plt.figure(figsize=(15, 15))
        
        # 定义网格
        gs = plt.GridSpec(3, 2, figure=fig)
        
        # 1. 账户价值曲线
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df['date'], df['equity'], color='blue', linewidth=2)
        ax1.set_title('账户价值变化', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.0f}'))
        
        # 2. 累计收益对比
        ax2 = fig.add_subplot(gs[1, :])
        df['strategy_cum_return'] = (1 + df['daily_return_pct']/100).cumprod() - 1
        df['spy_cum_return'] = (1 + df['spy_return_pct']/100).cumprod() - 1
        
        ax2.plot(df['date'], df['strategy_cum_return']*100, 
                label='策略', color='blue', linewidth=2)
        ax2.plot(df['date'], df['spy_cum_return']*100, 
                label='SPY', color='gray', linewidth=2, linestyle='--')
        ax2.set_title('累计收益率对比', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
        
        # 3. 每日盈亏分布
        ax3 = fig.add_subplot(gs[2, 0])
        sns.histplot(df['daily_return_pct'], bins=20, kde=True, ax=ax3)
        ax3.set_title('每日收益分布', fontsize=14)
        ax3.set_xlabel('收益率 (%)')
        ax3.set_ylabel('频率')
        
        # 4. 胜率饼图
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.pie([win_days, loss_days], labels=['盈利', '亏损'], 
                autopct='%1.1f%%', colors=['green', 'red'])
        ax4.set_title('盈亏天数比例', fontsize=14)
        
        # 添加性能指标文本
        plt.figtext(0.5, 0.01, f"总交易天数: {total_days}    胜率: {win_rate:.2%}    总收益: {cumulative_return:.2%}    基准收益: {benchmark_return:.2%}", 
                   ha='center', fontsize=12, bbox=dict(facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # 保存图表
        if save_path:
            plt.savefig(save_path)
            logger.info(f"绩效仪表板已保存到: {save_path}")
        elif not show:
            # 如果没有指定保存路径且不显示，则保存到默认位置
            data_dir = os.getenv('DATA_DIR', 'data')
            report_dir = os.path.join(data_dir, 'reports')
            os.makedirs(report_dir, exist_ok=True)
            
            default_path = os.path.join(report_dir, f"dashboard_{datetime.now().strftime('%Y%m%d')}.png")
            plt.savefig(default_path)
            logger.info(f"绩效仪表板已保存到: {default_path}")
            save_path = default_path
            
        # 显示图表
        if show:
            plt.show()
        else:
            plt.close()
            
        return save_path
        
    except Exception as e:
        logger.error(f"创建绩效仪表板时出错: {str(e)}")
        return None

def generate_html_report_with_charts(pnl_data, positions_data=None, trades_data=None, save_path=None):
    """
    生成包含图表的HTML报告
    
    参数:
        pnl_data (DataFrame): 盈亏数据
        positions_data (list): 持仓数据
        trades_data (DataFrame): 交易数据
        save_path (str): 保存路径
        
    返回:
        str: 保存的报告路径或None
    """
    try:
        if pnl_data is None or pnl_data.empty:
            logger.warning("无盈亏数据，无法生成报告")
            return None
            
        # 确保日期列是datetime类型
        if 'date' in pnl_data.columns:
            pnl_data['date'] = pd.to_datetime(pnl_data['date'])
            
        # 排序并获取最近一天的数据
        df = pnl_data.sort_values('date')
        latest_data = df.iloc[-1]
        
        # 计算性能指标
        performance_metrics = calculate_performance_metrics(df)
        
        # 生成图表（嵌入到HTML中）
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # 账户价值曲线
        axes[0].plot(df['date'], df['equity'], color='blue', linewidth=2)
        axes[0].set_title('账户价值变化', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # 累计收益对比
        df['strategy_cum_return'] = (1 + df['daily_return_pct']/100).cumprod() - 1
        df['spy_cum_return'] = (1 + df['spy_return_pct']/100).cumprod() - 1
        
        axes[1].plot(df['date'], df['strategy_cum_return']*100, 
                    label='策略', color='blue', linewidth=2)
        axes[1].plot(df['date'], df['spy_cum_return']*100, 
                    label='SPY', color='gray', linewidth=2, linestyle='--')
        axes[1].set_title('累计收益率对比', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 将图表转换为base64编码，嵌入到HTML中
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        # 如果没有指定保存路径，则保存到默认位置
        if not save_path:
            data_dir = os.getenv('DATA_DIR', 'data')
            report_dir = os.path.join(data_dir, 'reports')
            os.makedirs(report_dir, exist_ok=True)
            
            save_path = os.path.join(report_dir, f"performance_report_{datetime.now().strftime('%Y%m%d')}.html")
        
        # 生成HTML报告
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>投资组合绩效报告</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; color: #333; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .chart {{ max-width: 100%; margin: 20px 0; }}
        .metrics {{ display: flex; flex-wrap: wrap; }}
        .metric-box {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 10px; width: 200px; }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <h1>投资组合绩效报告</h1>
    <p>生成日期: {datetime.now().strftime('%Y-%m-%d')}</p>
    
    <div>
        <h2>账户概览</h2>
        <p>账户价值: ${latest_data['equity']:.2f}</p>
        <p>最新日收益: <span class="{'positive' if latest_data['daily_pnl'] > 0 else 'negative'}">${latest_data['daily_pnl']:.2f} ({latest_data['daily_return_pct']:.2f}%)</span></p>
    </div>
    
    <div>
        <h2>绩效图表</h2>
        <img src="data:image/png;base64,{chart_base64}" class="chart">
    </div>
    
    <div>
        <h2>绩效指标</h2>
        <div class="metrics">
            <div class="metric-box">
                <h3>胜率</h3>
                <p>{performance_metrics.get('win_rate', 0)*100:.2f}%</p>
            </div>
            <div class="metric-box">
                <h3>累计收益</h3>
                <p class="{'positive' if performance_metrics.get('cumulative_return', 0) > 0 else 'negative'}">{performance_metrics.get('cumulative_return', 0)*100:.2f}%</p>
            </div>
            <div class="metric-box">
                <h3>年化收益</h3>
                <p class="{'positive' if performance_metrics.get('annualized_return', 0) > 0 else 'negative'}">{performance_metrics.get('annualized_return', 0)*100:.2f}%</p>
            </div>
            <div class="metric-box">
                <h3>夏普比率</h3>
                <p>{performance_metrics.get('sharpe_ratio', 0):.2f}</p>
            </div>
            <div class="metric-box">
                <h3>最大回撤</h3>
                <p class="negative">{performance_metrics.get('max_drawdown', 0)*100:.2f}%</p>
            </div>
            <div class="metric-box">
                <h3>相对表现</h3>
                <p class="{'positive' if performance_metrics.get('avg_relative_performance', 0) > 0 else 'negative'}">{performance_metrics.get('avg_relative_performance', 0):.2f}%</p>
            </div>
        </div>
    </div>
"""
        
        # 添加持仓信息
        if positions_data:
            html_content += """
    <div>
        <h2>当前持仓</h2>
        <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%;">
            <tr>
                <th>股票</th>
                <th>数量</th>
                <th>买入价</th>
                <th>当前价</th>
                <th>市值</th>
                <th>盈亏</th>
                <th>盈亏%</th>
            </tr>
"""
            for pos in positions_data:
                pl_class = 'positive' if pos['unrealized_pl'] > 0 else 'negative'
                html_content += f"""
            <tr>
                <td>{pos['symbol']}</td>
                <td>{pos['quantity']:.2f}</td>
                <td>${pos['entry_price']:.2f}</td>
                <td>${pos['current_price']:.2f}</td>
                <td>${pos['market_value']:.2f}</td>
                <td class="{pl_class}">${pos['unrealized_pl']:.2f}</td>
                <td class="{pl_class}">{pos['unrealized_plpc']:.2f}%</td>
            </tr>
"""
            html_content += """
        </table>
    </div>
"""
        
        # 添加最近交易
        if trades_data is not None and not trades_data.empty:
            # 获取最近10笔交易
            recent_trades = trades_data.head(10)
            
            html_content += """
    <div>
        <h2>最近交易</h2>
        <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%;">
            <tr>
                <th>日期</th>
                <th>股票</th>
                <th>类型</th>
                <th>数量</th>
                <th>价格</th>
                <th>金额</th>
            </tr>
"""
            for _, trade in recent_trades.iterrows():
                side_class = 'positive' if trade['side'] == 'buy' else 'negative'
                side_text = '买入' if trade['side'] == 'buy' else '卖出'
                trade_date = trade['filled_at'].strftime('%Y-%m-%d') if isinstance(trade['filled_at'], (datetime, pd.Timestamp)) else trade['filled_at']
                
                html_content += f"""
            <tr>
                <td>{trade_date}</td>
                <td>{trade['symbol']}</td>
                <td class="{side_class}">{side_text}</td>
                <td>{trade['filled_qty']:.2f}</td>
                <td>${trade['filled_avg_price']:.2f}</td>
                <td>${trade['filled_qty'] * trade['filled_avg_price']:.2f}</td>
            </tr>
"""
            html_content += """
        </table>
    </div>
"""
        
        # 结束HTML
        html_content += """
    <p><i>该报告由自动交易系统生成</i></p>
</body>
</html>
"""
        
        # 写入文件
        with open(save_path, 'w') as f:
            f.write(html_content)
            
        logger.info(f"带图表的HTML绩效报告已生成: {save_path}")
        return save_path
        
    except Exception as e:
        logger.error(f"生成带图表的HTML报告时出错: {str(e)}")
        return None

def calculate_performance_metrics(pnl_data):
    """导入性能指标计算函数"""
    from .performance import calculate_performance_metrics as calc_metrics
    return calc_metrics(pnl_data)