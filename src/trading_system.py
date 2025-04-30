# src/trading_system.py

import os
import time
import schedule
import logging
from datetime import datetime, timedelta

# Import core modules
from .data.fetcher import get_stock_data, get_api_client
from .data.processor import calculate_technical_indicators
from .indicators.trend import calculate_macd
from .indicators.oscillators import calculate_rsi
from .indicators.volatility import calculate_bollinger_bands, calculate_atr
from .ml.train import train_model, load_model
from .ml.predict import predict
from .ml.evaluate import evaluate_model
from .portfolio.construction import build_portfolio, execute_portfolio, get_potential_stocks
from .portfolio.risk import assess_stock_risk, set_stop_loss, update_stop_loss
from .portfolio.rebalance import rebalance_portfolio
from .reporting.performance import track_daily_pnl, generate_daily_report
from .reporting.visualization import plot_portfolio_performance
from .reporting.notifications import send_notification
from .rl.agent import train_rl_agent_for_symbol, rl_decision
from .utils.logger import setup_logger
from .utils.config import get_env_variable, get_trading_params, get_data_paths
from .utils.database import create_tables_if_not_exist, save_to_mysql

# 设置日志
logger = logging.getLogger(__name__)

# Define a larger watchlist of stocks
WATCHLIST = [
    # Large cap tech (stable growth)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 
    
    # Financial sector
    'JPM', 'BAC', 'GS', 'MS', 'V', 'MA',
    
    # Healthcare
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',
    
    # Consumer
    'WMT', 'PG', 'KO', 'PEP', 'COST', 'HD',
    
    # ETFs for stability
    'SPY', 'QQQ', 'IWM', 'VTI', 'XLK'
]

def hybrid_trading_decision(symbol, context, data=None, ml_model=None, ml_scaler=None, rl_model_path=None):
    """
    综合使用规则、机器学习和强化学习的交易决策
    
    参数:
        symbol (str): 股票代码
        context (dict): 市场上下文
        data (DataFrame, optional): 股票数据
        ml_model: 机器学习模型
        ml_scaler: 特征缩放器
        rl_model_path (str, optional): 强化学习模型路径
        
    返回:
        str: 决策 ("买入", "卖出", "持有")
    """
    if data is None:
        data = get_stock_data(symbol)
    
    if data is None or len(data) < 20:
        logger.warning(f"{symbol} 数据不足，无法做出决策")
        return "持有"
    
    # 获取动态阈值
    trading_params = get_trading_params()
    rsi_buy = 30  # 默认买入阈值
    rsi_sell = 70  # 默认卖出阈值
    
    # 根据市场状态调整阈值
    if context.get('market_regime') == 'bull':
        rsi_buy += 10  # 牛市提高买入阈值
        rsi_sell += 10  # 牛市提高卖出阈值
    elif context.get('market_regime') == 'bear':
        rsi_buy -= 10  # 熊市降低买入阈值
        rsi_sell -= 10  # 熊市降低卖出阈值
    
    # 根据VIX调整阈值
    if context.get('vix_level') == 'high':
        rsi_buy = max(15, rsi_buy - 10)  # 高波动降低买入阈值
    elif context.get('vix_level') == 'low':
        rsi_sell = min(85, rsi_sell + 5)  # 低波动提高卖出阈值
    
    # 计算技术指标
    if 'rsi' not in data.columns:
        data = calculate_technical_indicators(data)
    
    # 获取最新指标
    latest = data.iloc[-1]
    current_rsi = latest['rsi']
    
    logger.info(f"{symbol} 最新RSI: {current_rsi:.2f}")
    logger.info(f"动态RSI阈值 - 买入: {rsi_buy}, 卖出: {rsi_sell}")
    
    # 检查是否已持有该股票
    api = get_api_client()
    has_position = False
    position_qty = 0
    
    try:
        position = api.get_position(symbol)
        has_position = True
        position_qty = float(position.qty)
        logger.info(f"当前持有 {position_qty} 股 {symbol}")
    except Exception:
        logger.info(f"当前未持有 {symbol}")
    
    # 1. 规则信号
    rule_signal = None
    if current_rsi <= rsi_buy:
        rule_signal = "买入"
    elif current_rsi >= rsi_sell:
        rule_signal = "卖出"
    else:
        rule_signal = "持有"
    
    # 2. 机器学习信号
    ml_signal = None
    if ml_model is not None and ml_scaler is not None:
        try:
            from .ml.predict import predict_proba
            
            # 准备特征
            features = data.iloc[-1:].copy()
            
            # 预测上涨概率
            prob_up = predict_proba(features, ml_model, ml_scaler)
            
            if isinstance(prob_up, (list, np.ndarray)) and len(prob_up) > 0:
                prob_up = prob_up[0]
            
            logger.info(f"机器学习预测上涨概率: {prob_up:.2f}")
            
            if prob_up > 0.65:
                ml_signal = "买入"
            elif prob_up < 0.35:
                ml_signal = "卖出"
            else:
                ml_signal = "持有"
                
        except Exception as e:
            logger.error(f"机器学习预测失败: {str(e)}")
            ml_signal = "持有"  # 失败时默认持有
    
    # 3. 强化学习信号
    rl_signal = None
    if rl_model_path is not None:
        try:
            # 使用强化学习做出决策
            model_dir = get_data_paths()['model_dir']
            full_model_path = os.path.join(model_dir, f"{symbol}_rl_model.h5")
            
            # 如果强化学习模型存在，使用它做出决策
            if os.path.exists(full_model_path):
                action = rl_decision(symbol, data, full_model_path)
                
                if action == 2:  # 买入
                    rl_signal = "买入"
                elif action == 0:  # 卖出
                    rl_signal = "卖出"
                else:  # 持有
                    rl_signal = "持有"
                
                logger.info(f"强化学习建议: {rl_signal}")
            else:
                logger.warning(f"{symbol}的强化学习模型不存在")
                
        except Exception as e:
            logger.error(f"强化学习预测失败: {str(e)}")
            rl_signal = "持有"  # 失败时默认持有
    
    # 综合决策
    signals = [rule_signal]
    if ml_signal:
        signals.append(ml_signal)
    if rl_signal:
        signals.append(rl_signal)
    
    # 简单多数投票
    buy_count = signals.count("买入")
    sell_count = signals.count("卖出")
    hold_count = signals.count("持有")
    
    # 输出信号汇总
    signal_summary = f"规则信号: {rule_signal}"
    if ml_signal:
        signal_summary += f", 机器学习信号: {ml_signal}"
    if rl_signal:
        signal_summary += f", 强化学习信号: {rl_signal}"
    logger.info(signal_summary)
    
    # 确定最终决策
    if buy_count > sell_count and buy_count > hold_count and not has_position:
        final_decision = "买入"
    elif sell_count > buy_count and sell_count > hold_count and has_position:
        final_decision = "卖出"
    else:
        final_decision = "持有"
    
    logger.info(f"最终决策: {final_decision}")
    return final_decision

def get_market_context():
    """
    获取市场上下文信息
    
    返回:
        dict: 市场上下文
    """
    try:
        from .data.fetcher import get_stock_data
        
        # 1. 获取SPY ETF数据作为大盘指标
        spy_data = get_stock_data('SPY', days=60)
        
        # 2. 判断市场状态
        market_regime = "neutral"  # 默认中性
        if spy_data is not None and len(spy_data) > 20:
            # 计算短期和长期趋势
            spy_data['short_ma'] = spy_data['close'].rolling(window=20).mean()
            spy_data['long_ma'] = spy_data['close'].rolling(window=50).mean()
            
            # 获取最新数据
            latest = spy_data.iloc[-1]
            
            # 判断市场状态
            if latest['short_ma'] > latest['long_ma'] * 1.05:
                market_regime = "bull"  # 牛市
            elif latest['short_ma'] < latest['long_ma'] * 0.95:
                market_regime = "bear"  # 熊市
        
        # 3. 获取VIX指数数据评估波动性
        vix_data = get_stock_data('VIX', days=30)
        vix_level = "medium"  # 默认中等
        
        if vix_data is not None and len(vix_data) > 0:
            latest_vix = vix_data['close'].iloc[-1]
            
            if latest_vix > 30:
                vix_level = "high"  # 高恐慌
            elif latest_vix < 15:
                vix_level = "low"  # 低恐慌
        
        # 组合上下文
        context = {
            'market_regime': market_regime,
            'vix_level': vix_level,
            'spy_data': spy_data,
            'vix_data': vix_data,
            'timestamp': datetime.now()
        }
        
        logger.info(f"市场状态: {market_regime}, VIX水平: {vix_level}")
        return context
        
    except Exception as e:
        logger.error(f"获取市场上下文失败: {str(e)}")
        return {
            'market_regime': 'neutral',
            'vix_level': 'medium',
            'spy_data': None,
            'vix_data': None,
            'timestamp': datetime.now()
        }

def execute_trade(symbol, action, qty=None):
    """
    执行交易操作
    
    参数:
        symbol (str): 股票代码
        action (str): 操作类型 ("买入", "卖出", "持有")
        qty (float, optional): 交易数量，如果为None则自动计算
        
    返回:
        dict or None: 订单信息或None
    """
    if action == "持有":
        logger.info(f"{symbol}: 保持当前仓位")
        return None
    
    api = get_api_client()
    
    try:
        # 转换操作类型
        side = "buy" if action == "买入" else "sell"
        
        # 获取交易参数
        trading_params = get_trading_params()
        risk_percent = trading_params['risk_percent']
        
        # 如果没有指定数量，计算适合的数量
        if qty is None:
            account = api.get_account()
            account_value = float(account.portfolio_value)
            
            if side == "buy":
                # 获取最新价格
                last_quote = api.get_latest_quote(symbol)
                price = last_quote.askprice if hasattr(last_quote, 'askprice') else last_quote.latestprice
                
                # 计算基于风险的数量
                risk_amount = account_value * risk_percent
                qty = risk_amount / price
            else:
                # 如果是卖出，获取全部持仓
                position = api.get_position(symbol)
                qty = float(position.qty)
        
        # 执行交易
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='day'
        )
        
        logger.info(f"{action} {qty} 股 {symbol}, 订单ID: {order.id}")
        
        # 如果是买入，设置止损
        if side == "buy":
            # 等待订单成交
            time.sleep(2)
            order_status = api.get_order(order.id)
            
            if order_status.status == 'filled':
                stop_loss_percent = trading_params['stop_loss_percent']
                set_stop_loss(symbol, stop_percent=stop_loss_percent)
        
        # 发送通知
        send_notification(
            f"{action} {qty} 股 {symbol} @ {datetime.now().strftime('%H:%M:%S')}",
            title="交易执行"
        )
        
        return {
            'symbol': symbol,
            'action': action,
            'qty': qty,
            'order_id': order.id,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"执行{symbol}的{action}交易失败: {str(e)}")
        
        # 发送错误通知
        send_notification(
            f"交易失败: {symbol} {action} - {str(e)}",
            title="交易错误"
        )
        
        return None

def run_intelligent_trading_system(symbols=None, schedule_retrain_enabled=True, use_short_term=True):
    """
    运行智能交易系统
    
    参数:
        symbols (list, optional): 要交易的股票列表
        schedule_retrain_enabled (bool): 是否启用定期重新训练
        use_short_term (bool): 是否使用短期交易策略
        
    返回:
        bool: 是否成功运行
    """
    logger.info("开始运行智能交易系统...")
    
    # 如果没有提供股票列表，使用扩展的观察列表
    if symbols is None:
        symbols = [
            # Large cap tech (stable growth)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 
            
            # Financial sector
            'JPM', 'BAC', 'GS', 'MS', 'V', 'MA',
            
            # Healthcare
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',
            
            # Consumer
            'WMT', 'PG', 'KO', 'PEP', 'COST', 'HD',
            
            # ETFs for stability
            'SPY', 'QQQ', 'IWM', 'VTI', 'XLK'
        ]
    
    # 检查账户状态
    api = get_api_client()
    account = api.get_account()
    logger.info(f"账户价值: ${account.portfolio_value}")
    logger.info(f"现金余额: ${account.cash}")
    
    # 检查市场状态
    clock = api.get_clock()
    logger.info(f"市场状态: {'开放' if clock.is_open else '关闭'}")
    
    # 获取市场上下文
    context = get_market_context()
    
    # 获取交易参数
    trading_params = get_trading_params()
    max_positions = int(trading_params.get('MAX_POSITIONS', 3))
    
    # 获取当前应交易的股票子集
    active_symbols = get_trading_symbols(context, max_positions)
    logger.info(f"当前市场状态: {context['market_regime']}, 活跃交易股票: {', '.join(active_symbols)}")
    
    # 获取当前持仓
    current_positions = []
    try:
        positions = api.list_positions()
        current_positions = [position.symbol for position in positions]
        logger.info(f"当前持仓: {', '.join(current_positions)}")
    except Exception as e:
        logger.warning(f"获取当前持仓失败: {str(e)}")
    
    # 针对每只股票进行决策和交易
    results = []
    for symbol in symbols:
        logger.info(f"\n分析 {symbol}...")
        
        # 获取股票数据
        data = get_stock_data(symbol)
        
        if data is None or len(data) < 20:
            logger.warning(f"无足够的{symbol}数据进行决策")
            continue
        
        # 计算技术指标，使用短期指标参数
        data = calculate_technical_indicators(data, use_short_term=use_short_term)
        
        # 加载机器学习模型
        model_dir = get_data_paths()['model_dir']
        ml_model, ml_scaler = load_model(symbol, model_dir)
        
        # 强化学习模型路径
        rl_model_path = os.path.join(model_dir, f"{symbol}_rl_model.h5")
        if not os.path.exists(rl_model_path):
            rl_model_path = None
        
        # 综合决策，使用短期策略
        decision = hybrid_trading_decision(
            symbol, 
            context,
            data=data,
            ml_model=ml_model, 
            ml_scaler=ml_scaler, 
            rl_model_path=rl_model_path,
            use_short_term=use_short_term
        )
        
        # 确定是否执行交易
        # 1. 如果是买入信号，只有在活跃列表中且当前持仓数量小于最大持仓数量时才买入
        if decision == "买入":
            if symbol in active_symbols and len(current_positions) < max_positions:
                logger.info(f"{symbol} 符合买入条件，执行交易")
                result = execute_trade(symbol, decision)
                if result:
                    results.append(result)
                    current_positions.append(symbol)  # 更新当前持仓列表
            else:
                reason = ""
                if symbol not in active_symbols:
                    reason = "不在活跃交易列表中"
                else:
                    reason = f"当前持仓数量({len(current_positions)})已达最大限制({max_positions})"
                logger.info(f"{symbol} 有买入信号但{reason}，跳过交易")
                
        # 2. 如果是卖出信号，只对当前持有的股票执行
        elif decision == "卖出":
            if symbol in current_positions:
                logger.info(f"{symbol} 符合卖出条件，执行交易")
                result = execute_trade(symbol, decision)
                if result:
                    results.append(result)
                    current_positions.remove(symbol)  # 更新当前持仓列表
            else:
                logger.info(f"{symbol} 有卖出信号但当前未持有，跳过交易")
        
        # 3. 持有信号不执行交易
        else:
            logger.info(f"{symbol} 持有信号，无需交易")
        
        # 更新现有持仓的止损
        if symbol in current_positions:
            try:
                # 使用更严格的止损设置
                stop_percent = float(trading_params.get('STOP_LOSS_PERCENT', 0.03))
                trail_percent = float(trading_params.get('TRAILING_STOP_PERCENT', 0.02))
                
                # 更新止损
                update_stop_loss(symbol, trail_percent=trail_percent)
            except Exception as e:
                logger.error(f"更新{symbol}止损时出错: {str(e)}")
    
    # 查看最终持仓
    logger.info("\n当前持仓:")
    try:
        positions = api.list_positions()
        for position in positions:
            print(f"{position.symbol}: {position.qty} 股, 均价: ${position.avg_entry_price}, 市值: ${position.market_value}")
    except Exception as e:
        logger.error(f"获取持仓失败: {str(e)}")
    
    # 生成每日报告
    if datetime.now().hour >= 16:  # 如果当前时间在下午4点以后
        generate_daily_report()
    
    # 设置定期任务
    if schedule_retrain_enabled:
        # 每周一凌晨2点重新训练模型
        schedule.every().monday.at("02:00").do(schedule_retrain, symbols=symbols)
        logger.info("已设置每周一凌晨2点重新训练模型")
        
        # 设置每日健康检查
        schedule.every(60).minutes.do(update_health_check)
        logger.info("已设置每小时健康检查")
        
        # 设置每日报告任务
        schedule.every().day.at("16:30").do(generate_daily_report)
        logger.info("已设置每日16:30生成报告")
        
        # 根据是否使用短期策略设置不同的监控频率
        if use_short_term:
            # 短期策略需要更频繁监控
            schedule.every(5).minutes.do(monitor_market, symbols=symbols, interval_minutes=5)
            logger.info("已设置每5分钟监控市场（短期策略）")
        else:
            # 标准策略监控频率
            schedule.every(15).minutes.do(monitor_market, symbols=symbols, interval_minutes=15)
            logger.info("已设置每15分钟监控市场（标准策略）")
    
    logger.info("交易系统运行完成")
    return True

def get_trading_symbols(context, max_positions=3):
    """
    根据市场状态选择要交易的股票子集
    
    参数:
        context (dict): 市场上下文
        max_positions (int): 最大持仓数量
        
    返回:
        list: 要交易的股票代码列表
    """
    # 根据市场状态选择合适的股票
    if context['market_regime'] == 'bull':
        # 牛市偏好成长股和科技股
        candidates = ['NVDA', 'TSLA', 'AMZN', 'META', 'QQQ', 'SPY']
    elif context['market_regime'] == 'bear':
        # 熊市偏好防御性股票和ETF
        candidates = ['WMT', 'PG', 'JNJ', 'KO', 'SPY', 'VTI']
    else:
        # 中性市场平衡选择
        candidates = ['AAPL', 'MSFT', 'JPM', 'V', 'SPY', 'QQQ']
    
    # 高波动率市场调整
    if context.get('vix_level') == 'high':
        # 高波动率偏好低Beta股票和ETF
        candidates = ['PG', 'JNJ', 'KO', 'WMT', 'VTI', 'SPY']
    
    # 确保不超过最大持仓数量
    return candidates[:max_positions]

def schedule_retrain(symbols=None):
    """
    定期重新训练所有模型
    
    参数:
        symbols (list, optional): 要训练的股票列表
        
    返回:
        bool: 是否成功训练
    """
    logger.info("开始定期重新训练模型...")
    
    # 如果没有提供股票列表，使用默认列表
    if symbols is None:
        symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
            'META', 'TSLA', 'JPM', 'V', 'PG'
        ]
    
    for symbol in symbols:
        logger.info(f"\n重新训练 {symbol} 的模型...")
        
        try:
            # 获取股票数据
            data = get_stock_data(symbol, days=365)
            
            if data is None or len(data) < 100:
                logger.warning(f"无足够的{symbol}数据进行训练")
                continue
            
            # 计算技术指标
            data = calculate_technical_indicators(data)
            
            # 1. 重新训练机器学习模型
            from .data.processor import prepare_features
            from .ml.train import train_model, save_model
            
            # 准备特征和目标
            X, y = prepare_features(data)
            
            if X is not None and y is not None:
                # 训练模型
                model, scaler = train_model(X, y)
                
                # 保存模型
                model_dir = get_data_paths()['model_dir']
                os.makedirs(model_dir, exist_ok=True)
                
                save_model(model, scaler, symbol, model_dir)
                logger.info(f"{symbol}的机器学习模型训练完成")
            
            # 2. 训练强化学习模型
            train_rl_agent_for_symbol(symbol, data, episodes=30)
            
        except Exception as e:
            logger.error(f"训练{symbol}模型失败: {str(e)}")
    
    logger.info("所有模型重新训练完成")
    
    # 保存重训时间记录
    data_paths = get_data_paths()
    with open(os.path.join(data_paths['data_dir'], 'last_retrain.txt'), "w") as f:
        f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    return True

def monitor_market(symbols=None, interval_minutes=15):
    """
    实时监控市场并在必要时发送通知
    
    参数:
        symbols (list, optional): 要监控的股票列表
        interval_minutes (int): 监控间隔（分钟）
        
    返回:
        dict: 监控结果
    """
    logger.info(f"开始市场监控，间隔: {interval_minutes}分钟")
    
    # 如果没有提供股票列表，使用默认列表
    if symbols is None:
        symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
            'META', 'TSLA', 'JPM', 'V', 'PG'
        ]
    
    # 获取市场上下文
    context = get_market_context()
    
    # 初始化结果
    monitor_results = {
        'timestamp': datetime.now().isoformat(),
        'market_status': context['market_regime'],
        'vix_level': context['vix_level'],
        'signals': []
    }
    
    # 检查每只股票
    for symbol in symbols:
        try:
            # 获取股票数据
            data = get_stock_data(symbol)
            
            if data is None or len(data) < 20:
                logger.warning(f"无足够的{symbol}数据进行监控")
                continue
            
            # 计算技术指标
            data = calculate_technical_indicators(data)
            
            # 加载机器学习模型
            model_dir = get_data_paths()['model_dir']
            ml_model, ml_scaler = load_model(symbol, model_dir)
            
            # 强化学习模型路径
            rl_model_path = os.path.join(model_dir, f"{symbol}_rl_model.h5")
            if not os.path.exists(rl_model_path):
                rl_model_path = None
            
            # 获取决策
            decision = hybrid_trading_decision(
                symbol, 
                context,
                data=data,
                ml_model=ml_model, 
                ml_scaler=ml_scaler, 
                rl_model_path=rl_model_path
            )
            
            # 记录信号
            signal_info = {
                'symbol': symbol,
                'decision': decision,
                'price': data.iloc[-1]['close'],
                'rsi': data.iloc[-1]['rsi'] if 'rsi' in data.columns else None
            }
            
            monitor_results['signals'].append(signal_info)
            
            # 如果有买入或卖出信号，发送通知
            if decision in ["买入", "卖出"]:
                message = f"{symbol} 产生{decision}信号! 当前价格: ${data.iloc[-1]['close']:.2f}"
                send_notification(message, title="交易信号")
                
        except Exception as e:
            logger.error(f"监控{symbol}时出错: {str(e)}")
    
    # 生成监控摘要
    buy_signals = [s['symbol'] for s in monitor_results['signals'] if s['decision'] == "买入"]
    sell_signals = [s['symbol'] for s in monitor_results['signals'] if s['decision'] == "卖出"]
    
    if buy_signals:
        logger.info(f"买入信号: {', '.join(buy_signals)}")
    
    if sell_signals:
        logger.info(f"卖出信号: {', '.join(sell_signals)}")
    
    logger.info("市场监控完成")
    return monitor_results

def setup_scheduled_monitoring(symbols=None, interval_minutes=15):
    """
    设置定时监控任务
    
    参数:
        symbols (list, optional): 要监控的股票列表
        interval_minutes (int): 监控间隔（分钟）
    """
    # 如果没有提供股票列表，使用默认列表
    if symbols is None:
        symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
            'META', 'TSLA', 'JPM', 'V', 'PG'
        ]
    
    logger.info(f"设置定时监控（每{interval_minutes}分钟）: {', '.join(symbols)}")
    
    # 立即执行一次监控
    monitor_market(symbols, interval_minutes)
    
    # 设置定期监控
    schedule.every(interval_minutes).minutes.do(
        monitor_market, symbols=symbols, interval_minutes=interval_minutes
    )
    
    # 设置定期健康检查
    schedule.every(60).minutes.do(
        update_health_check
    )
    
    # 设置每日盈亏记录
    schedule.every().day.at("16:00").do(
        track_daily_pnl
    )
    
    # 设置每日报告生成
    schedule.every().day.at("16:30").do(
        generate_daily_report
    )
    
    try:
        logger.info("开始定时监控循环...")
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次
    except KeyboardInterrupt:
        logger.info("监控已停止")
    except Exception as e:
        logger.error(f"监控过程中出错: {str(e)}")

def check_market_hours():
    """
    检查当前是否是市场交易时间
    
    返回:
        bool: 市场是否开放
    """
    try:
        api = get_api_client()
        clock = api.get_clock()
        
        is_open = clock.is_open
        next_open = clock.next_open
        next_close = clock.next_close
        
        if is_open:
            logger.info(f"市场开盘中 - 下次收盘时间: {next_close}")
        else:
            logger.info(f"市场已收盘 - 下次开盘时间: {next_open}")
            
        return is_open
    
    except Exception as e:
        logger.error(f"检查市场状态出错: {str(e)}")
        return False

def update_health_check():
    """
    更新健康检查文件
    
    返回:
        bool: 是否成功更新
    """
    try:
        data_paths = get_data_paths()
        health_file = os.path.join(data_paths['data_dir'], 'health.txt')
        
        with open(health_file, "w") as f:
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
        logger.debug("健康检查文件已更新")
        return True
        
    except Exception as e:
        logger.error(f"更新健康检查文件出错: {str(e)}")
        return False

def main():
    """主函数，程序入口"""
    # 设置日志
    setup_logger()
    
    # 确保数据库表存在
    create_tables_if_not_exist()
    
    # 检测是否在Docker环境中运行
    in_docker = os.getenv('DOCKER_ENVIRONMENT', 'false').lower() == 'true'
    
    if in_docker:
        logger.info("在Docker环境中运行，启动持续监控模式...")
        
        # 设置要监控的股票
        symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
            'META', 'TSLA', 'JPM', 'V', 'PG',
            'HD', 'UNH', 'BAC', 'XOM', 'JNJ',
            'PG', 'MA', 'AVGO', 'WMT', 'CVX'
        ]
        
        # 创建健康检查文件
        update_health_check()
        
        # 执行每日维护任务
        schedule.every().day.at("06:00").do(
            lambda: schedule_retrain(symbols)
        )
        
        # 设置市场开盘时的监控
        schedule.every().day.at("09:30").do(
            lambda: send_notification("市场开盘，开始监控", "市场状态")
        )
        
        # 设置收盘后任务
        schedule.every().day.at("16:00").do(
            track_daily_pnl
        )
        
        schedule.every().day.at("16:15").do(
            generate_daily_report
        )
        
        # 启动自适应频率监控
        logger.info("启动自适应频率监控...")
        
        try:
            while True:
                is_market_open = check_market_hours()
                
                if is_market_open:
                    # 市场开盘时更频繁检查（每5分钟）
                    monitor_market(symbols, interval_minutes=5)
                else:
                    # 市场收盘时减少检查频率（每小时）
                    logger.info("市场已收盘，减少监控频率")
                    monitor_market(symbols, interval_minutes=60)
                
                # 运行所有待处理的定时任务
                schedule.run_pending()
                
                # 休眠
                time.sleep(300)  # 5分钟后再检查
                
        except KeyboardInterrupt:
            logger.info("系统收到终止信号，正在关闭...")
        except Exception as e:
            logger.error(f"系统运行出错: {str(e)}")
            # 发送错误通知
            send_notification(f"交易系统发生错误: {str(e)}", "系统错误")
    
    else:
        # 正常的交互式模式
        mode = input("选择运行模式 (1: 训练模型, 2: 单次交易分析, 3: 持续监控, 4: 性能分析, 5: 投资组合管理, 6: 卖出评估, 7: 模型评估, 8: 盈亏分析): ")
        
        symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
            'META', 'TSLA', 'JPM', 'V', 'PG'
        ]
        
        if mode == '1':
            # 训练模型
            schedule_retrain(symbols)
        elif mode == '2':
            # 单次交易分析
            run_intelligent_trading_system(symbols=symbols)
        elif mode == '3':
            # 持续监控
            setup_scheduled_monitoring(symbols=symbols, interval_minutes=15)
        elif mode == '4':
            # 性能分析
            from .analyzer import analyze_performance
            analyze_performance()
        elif mode == '5':
            # 投资组合管理子菜单
            portfolio_mode = input("投资组合操作 (1: 构建新组合, 2: 检查再平衡, 3: 风险评估): ")
            
            if portfolio_mode == '1':
                # 获取资金和风险配置
                try:
                    capital = float(input("输入投资金额 (默认: 10000): ") or "10000")
                    low_risk = float(input("低风险资产比例 (默认: 0.5): ") or "0.5")
                    med_risk = float(input("中等风险资产比例 (默认: 0.3): ") or "0.3")
                    high_risk = float(input("高风险资产比例 (默认: 0.2): ") or "0.2")
                    
                    # 确保比例总和为1
                    total = low_risk + med_risk + high_risk
                    if abs(total - 1.0) > 0.01:
                        logger.warning(f"风险配置总和 ({total:.2f}) 不等于1，将进行调整")
                        low_risk /= total
                        med_risk /= total
                        high_risk /= total
                    
                    risk_allocation = {
                        'low': low_risk,
                        'medium': med_risk,
                        'high': high_risk
                    }
                    
                    # 构建投资组合
                    portfolio = build_portfolio(capital, risk_allocation)
                    
                    # 询问是否执行投资组合
                    if portfolio and input("是否执行投资组合? (y/n): ").lower() == 'y':
                        execute_portfolio(portfolio, dry_run=input("是否模拟执行? (y/n): ").lower() == 'y')
                except ValueError:
                    logger.error("请输入有效的数字")
            
            elif portfolio_mode == '2':
                # 检查再平衡
                rebalance_portfolio()
                
            elif portfolio_mode == '3':
                # 评估特定股票风险
                symbols_input = input("输入股票代码(多个用逗号分隔): ")
                symbols_to_check = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
                
                for symbol in symbols_to_check:
                    category, metrics = assess_stock_risk(symbol)
                    print(f"{symbol} 风险类别: {category.capitalize()}")
                    if metrics:
                        for name, value in metrics.items():
                            print(f"  {name}: {value:.4f}")
        
        elif mode == '7':
            # 模型评估
            from .analyzer import evaluate_all_models
            eval_choice = input("选择评估方式 (1: 单个股票, 2: 所有股票): ")
            
            if eval_choice == '1':
                symbol = input("输入要评估的股票代码: ")
                from .ml.evaluate import evaluate_model
                data = get_stock_data(symbol, days=120)
                if data is not None:
                    data = calculate_technical_indicators(data)
                    from .data.processor import prepare_features
                    X, y = prepare_features(data)
                    model_dir = get_data_paths()['model_dir']
                    model, scaler = load_model(symbol, model_dir)
                    if model is not None and X is not None and y is not None:
                        metrics = evaluate_model(model, X, y, scaler)
                        print(f"模型评估结果: {metrics}")
                    else:
                        print("模型或数据无效，无法评估")
                else:
                    print(f"无法获取{symbol}的数据")
            
            elif eval_choice == '2':
                days = int(input("输入评估周期(默认120天): ") or "120")
                evaluate_all_models(days=days)
            
            else:
                print("无效的选择")
                
        elif mode == '8':
            # 盈亏分析
            from .analyzer import show_historical_pnl
            days = int(input("分析最近几天的盈亏 (0表示全部): ") or "30")
            show_historical_pnl(days=days, plot=True)
            
        else:
            print("无效的选择，退出")

# 如果作为主程序运行
if __name__ == "__main__":
    main()