# src/analyzer.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta

from .data.fetcher import get_stock_data, get_api_client
from .data.processor import calculate_technical_indicators
from .ml.train import load_model
from .ml.evaluate import evaluate_model
from .reporting.performance import get_pnl_history
from .reporting.visualization import plot_portfolio_performance
from .utils.logger import setup_logger
from .utils.config import get_data_paths
from .utils.database import save_model_evaluation

# 设置日志
logger = logging.getLogger(__name__)

def analyze_performance():
    """
    分析交易系统表现
    
    返回:
        dict: 性能分析结果
    """
    try:
        logger.info("开始分析交易系统表现...")
        
        # 获取账户信息
        api = get_api_client()
        account = api.get_account()
        
        # 获取盈亏历史
        pnl_data = get_pnl_history(days=30)
        
        # 分析数据
        try:
            positions = api.list_positions()
            buy_trades = []
            sell_trades = []
            
            # 获取最近交易
            orders = api.list_orders(status='closed', limit=100)
            for order in orders:
                if order.side == 'buy':
                    buy_trades.append(order)
                else:
                    sell_trades.append(order)
        except Exception as e:
            logger.error(f"获取交易历史失败: {str(e)}")
            positions = []
            buy_trades = []
            sell_trades = []
        
        # 输出分析
        print("\n===== 交易系统表现分析 =====")
        print(f"账户价值: ${account.portfolio_value}")
        print(f"持有现金: ${account.cash}")
        
        # 计算基本指标
        if not pnl_data.empty:
            total_days = len(pnl_data)
            win_days = len(pnl_data[pnl_data['daily_pnl'] > 0])
            loss_days = len(pnl_data[pnl_data['daily_pnl'] < 0])
            win_rate = win_days / total_days if total_days > 0 else 0
            
            print(f"交易天数: {total_days}")
            print(f"盈利天数: {win_days} ({win_rate:.2%})")
            print(f"亏损天数: {loss_days} ({1-win_rate:.2%})")
            
            # 计算累计回报
            if 'daily_return_pct' in pnl_data.columns:
                cumulative_return = (1 + pnl_data['daily_return_pct']/100).prod() - 1
                print(f"累计收益率: {cumulative_return:.2%}")
            
            # 相对表现
            if 'spy_return_pct' in pnl_data.columns and 'relative_performance' in pnl_data.columns:
                spy_cumulative_return = (1 + pnl_data['spy_return_pct']/100).prod() - 1
                avg_relative_performance = pnl_data['relative_performance'].mean()
                
                print(f"SPY累计收益率: {spy_cumulative_return:.2%}")
                print(f"相对收益率: {cumulative_return - spy_cumulative_return:.2%}")
                print(f"平均每日相对表现: {avg_relative_performance:.2%}")
        
        # 输出持仓信息
        if positions:
            print("\n当前持仓:")
            for position in positions:
                entry = float(position.avg_entry_price)
                current = float(position.current_price)
                profit_pct = (current - entry) / entry * 100
                print(f"{position.symbol}: {position.qty} 股, 均价: ${entry:.2f}, 现价: ${current:.2f}, 盈亏: {profit_pct:.2f}%")
        
        print("================================")
        
        # 绘制盈亏图表
        if not pnl_data.empty:
            plot_portfolio_performance(pnl_data, show=True)
        
        # 返回分析结果
        return {
            'account_value': float(account.portfolio_value),
            'cash': float(account.cash),
            'positions': len(positions),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'win_rate': win_rate if 'win_rate' in locals() else None,
            'cumulative_return': cumulative_return if 'cumulative_return' in locals() else None
        }
        
    except Exception as e:
        logger.error(f"分析性能时出错: {str(e)}")
        return None

def show_historical_pnl(days=30, plot=True, save_path=None):
    """
    展示历史盈亏数据
    
    参数:
        days (int): 显示天数
        plot (bool): 是否绘制图表
        save_path (str, optional): 保存路径
        
    返回:
        DataFrame: 盈亏数据
    """
    try:
        logger.info(f"获取过去{days}天的盈亏记录...")
        
        # 获取盈亏历史
        pnl_data = get_pnl_history(days=days)
        
        if pnl_data.empty:
            logger.warning("未找到盈亏记录")
            return None
            
        # 计算性能指标
        total_days = len(pnl_data)
        win_days = len(pnl_data[pnl_data['daily_pnl'] > 0])
        loss_days = len(pnl_data[pnl_data['daily_pnl'] < 0])
        win_rate = win_days / total_days if total_days > 0 else 0
        
        # 计算累计回报
        pnl_data['cumulative_return'] = (1 + pnl_data['daily_return_pct']/100).cumprod() - 1
        pnl_data['spy_cumulative_return'] = (1 + pnl_data['spy_return_pct']/100).cumprod() - 1
        
        cumulative_return = pnl_data['cumulative_return'].iloc[-1]
        spy_cumulative_return = pnl_data['spy_cumulative_return'].iloc[-1]
        
        # 计算最大回撤
        pnl_data['cumulative_max'] = pnl_data['cumulative_return'].cummax()
        pnl_data['drawdown'] = (pnl_data['cumulative_return'] - pnl_data['cumulative_max']) / (1 + pnl_data['cumulative_max'])
        max_drawdown = pnl_data['drawdown'].min()
        
        # 打印报告
        print(f"\n==== {'所有' if days <= 0 or days >= len(pnl_data) else f'最近{days}天的'}盈亏记录 ====")
        print(f"交易日数: {total_days}")
        print(f"盈利天数: {win_days} ({win_rate*100:.2f}%)")
        print(f"亏损天数: {loss_days} ({(1-win_rate)*100:.2f}%)")
        print(f"总盈亏: ${pnl_data['daily_pnl'].sum():.2f}")
        print(f"平均日盈亏: ${pnl_data['daily_pnl'].mean():.2f}")
        print(f"最大单日盈利: ${pnl_data['daily_pnl'].max():.2f}")
        print(f"最大单日亏损: ${pnl_data['daily_pnl'].min():.2f}")
        print(f"累计收益率: {cumulative_return*100:.2f}%")
        print(f"SPY累计收益率: {spy_cumulative_return*100:.2f}%")
        print(f"相对收益率: {(cumulative_return-spy_cumulative_return)*100:.2f}%")
        print(f"最大回撤: {max_drawdown*100:.2f}%")
        
        # 绘制盈亏图表
        if plot:
            plot_portfolio_performance(pnl_data, days=days, save_path=save_path, show=True)
        
        return pnl_data
        
    except Exception as e:
        logger.error(f"展示历史盈亏时出错: {str(e)}")
        return None

def evaluate_all_models(days=120, generate_report=True):
    """
    评估所有已训练的股票模型并生成报告
    
    参数:
        days (int): 评估周期
        generate_report (bool): 是否生成报告
        
    返回:
        dict: 评估结果
    """
    try:
        logger.info(f"开始评估所有股票模型（过去{days}天）...")
        
        # 获取模型目录
        model_dir = get_data_paths()['model_dir']
        
        # 获取所有模型文件
        from glob import glob
        model_files = glob(os.path.join(model_dir, "*_ml_model.joblib"))
        
        if not model_files:
            logger.warning("未找到任何模型文件")
            return {}
            
        # 提取股票代码
        symbols = [os.path.basename(f).split('_ml_model.joblib')[0] for f in model_files]
        logger.info(f"找到{len(symbols)}个模型: {', '.join(symbols)}")
        
        # 评估结果
        results = {}
        
        for symbol in symbols:
            logger.info(f"\n评估 {symbol} 模型...")
            
            # 获取股票数据
            data = get_stock_data(symbol, days=days)
            
            if data is None or len(data) < 30:
                logger.warning(f"获取{symbol}数据失败或数据不足")
                continue
                
            # 计算技术指标
            data = calculate_technical_indicators(data)
            
            # 准备特征和目标
            from .data.processor import prepare_features
            X, y = prepare_features(data)
            
            if X is None or y is None:
                logger.warning(f"{symbol}的特征准备失败")
                continue
                
            # 加载模型
            model, scaler = load_model(symbol, model_dir)
            
            if model is None:
                logger.warning(f"加载{symbol}模型失败")
                continue
                
            # 评估模型
            metrics = evaluate_model(model, X, y, scaler)
            
            if metrics is None:
                logger.warning(f"评估{symbol}模型失败")
                continue
                
            # 添加回测性能
            from .ml.evaluate import backtest_strategy
            
            # 使用模型预测
            predictions = predict(X, model, scaler)
            
            # 回测
            backtest_results, performance = backtest_strategy(predictions, data[['close']])
            
            if backtest_results is not None and performance is not None:
                # 添加回测指标
                metrics['strategy_return'] = performance['total_return']
                metrics['max_drawdown'] = backtest_results['drawdown'].min()
                
                # 计算市场回报
                market_return = (data['close'].iloc[-1] / data['close'].iloc[0]) - 1
                metrics['market_return'] = market_return
                
                # 相对表现
                metrics['relative_performance'] = metrics['strategy_return'] - market_return
            
            # 记录结果
            results[symbol] = metrics
            
            # 保存到数据库
            save_model_evaluation(symbol, 'ml', metrics)
            
            # 输出评估结果
            print(f"\n{symbol} 模型评估结果:")
            print(f"准确率: {metrics['accuracy']:.4f}")
            if 'precision' in metrics:
                print(f"精确率: {metrics['precision']:.4f}")
            if 'recall' in metrics:
                print(f"召回率: {metrics['recall']:.4f}")
            if 'f1' in metrics:
                print(f"F1分数: {metrics['f1']:.4f}")
            if 'auc' in metrics:
                print(f"AUC: {metrics['auc']:.4f}")
            if 'strategy_return' in metrics:
                print(f"策略回报: {metrics['strategy_return']:.2%}")
            if 'market_return' in metrics:
                print(f"市场回报: {metrics['market_return']:.2%}")
            if 'relative_performance' in metrics:
                print(f"相对表现: {metrics['relative_performance']:.2%}")
        
        # 如果需要生成报告
        if generate_report and results:
            generate_model_evaluation_report(results, days)
            
        return results
        
    except Exception as e:
        logger.error(f"评估所有模型时出错: {str(e)}")
        return {}

def generate_model_evaluation_report(results, days):
    """
    生成模型评估报告
    
    参数:
        results (dict): 评估结果
        days (int): 评估周期
        
    返回:
        str: 报告文件路径
    """
    try:
        # 创建报告目录
        data_paths = get_data_paths()
        report_dir = os.path.join(data_paths['data_dir'], 'reports')
        os.makedirs(report_dir, exist_ok=True)
        
        # 报告文件名
        date_str = datetime.now().strftime('%Y%m%d')
        report_file = os.path.join(report_dir, f"model_evaluation_{date_str}.html")
        
        # 转换结果为DataFrame
        results_df = pd.DataFrame.from_dict(results, orient='index')
        
        # 四舍五入到4位小数
        results_df = results_df.round(4)
        
        # 保存为CSV
        csv_file = os.path.join(report_dir, f"model_evaluation_{date_str}.csv")
        results_df.to_csv(csv_file)
        
        # 计算平均指标
        mean_metrics = results_df.mean()
        median_metrics = results_df.median()
        
        # 生成HTML报告
        with open(report_file, 'w') as f:
            f.write(f'''<!DOCTYPE html>
<html>
<head>
    <title>模型评估报告 - {date_str}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; color: #333; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; color: #2c3e50; }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>股票预测模型评估报告</h1>
    <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>评估周期: 过去{days}天</p>
    
    <div class="summary">
        <h2>摘要</h2>
        <p>共评估了 {len(results)} 个模型</p>
    </div>
    
    <h2>模型表现</h2>
    <table>
        <tr>
            <th>股票</th>
            <th>准确率</th>
            <th>精确率</th>
            <th>召回率</th>
            <th>F1分数</th>
            <th>AUC</th>
            <th>策略回报</th>
            <th>市场回报</th>
            <th>相对表现</th>
        </tr>
''')
            
            # 添加每个模型的结果
            for symbol, metrics in results.items():
                # 处理可能缺失的指标
                precision = metrics.get('precision', float('nan'))
                recall = metrics.get('recall', float('nan'))
                f1 = metrics.get('f1', float('nan'))
                auc = metrics.get('auc', float('nan'))
                
                strategy_return = metrics.get('strategy_return', float('nan'))
                market_return = metrics.get('market_return', float('nan'))
                relative_performance = metrics.get('relative_performance', float('nan'))
                
                # 添加样式
                strategy_class = 'positive' if not np.isnan(strategy_return) and strategy_return > 0 else 'negative'
                market_class = 'positive' if not np.isnan(market_return) and market_return > 0 else 'negative'
                relative_class = 'positive' if not np.isnan(relative_performance) and relative_performance > 0 else 'negative'
                
                f.write(f'''
        <tr>
            <td>{symbol}</td>
            <td>{metrics['accuracy']:.4f}</td>
            <td>{precision:.4f}</td>
            <td>{recall:.4f}</td>
            <td>{f1:.4f}</td>
            <td>{auc:.4f}</td>
            <td class="{strategy_class}">{strategy_return:.2%}</td>
            <td class="{market_class}">{market_return:.2%}</td>
            <td class="{relative_class}">{relative_performance:.2%}</td>
        </tr>''')
            
            # 添加平均值
            f.write(f'''
        <tr style="font-weight: bold;">
            <td>平均值</td>
            <td>{mean_metrics.get('accuracy', 0):.4f}</td>
            <td>{mean_metrics.get('precision', 0):.4f}</td>
            <td>{mean_metrics.get('recall', 0):.4f}</td>
            <td>{mean_metrics.get('f1', 0):.4f}</td>
            <td>{mean_metrics.get('auc', 0):.4f}</td>
            <td class="{'positive' if mean_metrics.get('strategy_return', 0) > 0 else 'negative'}">{mean_metrics.get('strategy_return', 0):.2%}</td>
            <td class="{'positive' if mean_metrics.get('market_return', 0) > 0 else 'negative'}">{mean_metrics.get('market_return', 0):.2%}</td>
            <td class="{'positive' if mean_metrics.get('relative_performance', 0) > 0 else 'negative'}">{mean_metrics.get('relative_performance', 0):.2%}</td>
        </tr>
    </table>
    
    <p><i>该报告由自动交易系统生成</i></p>
</body>
</html>''')
        
        logger.info(f"模型评估报告已生成: {report_file}")
        return report_file
        
    except Exception as e:
        logger.error(f"生成模型评估报告时出错: {str(e)}")
        return None