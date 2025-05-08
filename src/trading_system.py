# src/trading_system.py

import os
import time
import schedule
import logging
import numpy as np
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

# 设置目标池
SYMBOLS = [
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

# 修改4: 调整calculate_multi_factor_signal函数中的各项指标阈值
def calculate_multi_factor_signal(data, context):
    """Generate trading signals based on multiple technical factors with adjusted thresholds"""
    try:
        # Get latest data point
        latest = data.iloc[-1]
        
        # 1. RSI Signal - 更敏感的卖出阈值
        rsi_signal = "持有"
        if 'rsi' in latest:
            if latest['rsi'] <= 33:
                rsi_signal = "买入"
            elif latest['rsi'] >= 62:  # 降低RSI卖出阈值，更早发出卖出信号
                rsi_signal = "卖出"
        
        # 2. MACD Signal - 更敏感的卖出条件
        macd_signal = "持有"
        if 'macd_hist' in latest and 'macd_hist' in data.columns:
            # 如果有前一天数据，检查柱状图是否开始下降
            if len(data) > 1:
                prev_hist = data['macd_hist'].iloc[-2]
                curr_hist = latest['macd_hist']
                
                if curr_hist > 0 and curr_hist < prev_hist * 0.8:  # 柱状图明显下降
                    macd_signal = "卖出"  
                elif curr_hist > 0 and curr_hist > prev_hist * 1.2:  # 柱状图明显上升
                    macd_signal = "买入"
                elif curr_hist < -0.05:  # 柱状图为负且绝对值变大
                    macd_signal = "卖出"
        
        # 3. Bollinger Band Signal - 更敏感的卖出条件
        bb_signal = "持有"
        if all(k in latest for k in ['close', 'bb_lower', 'bb_middle', 'bb_upper']):
            if latest['close'] < latest['bb_lower']:
                bb_signal = "买入"  # 价格低于布林带下轨
            # 调整为更敏感的卖出条件：价格超过布林带中轨以上60%处
            elif latest['close'] > (latest['bb_middle'] + (latest['bb_upper'] - latest['bb_middle']) * 0.6):
                bb_signal = "卖出"
        
        # 4. 移动平均信号 - 更敏感的卖出条件
        ma_signal = "持有"
        if all(k in latest for k in ['close', 'ma10', 'ma50']):
            if latest['close'] > latest['ma50'] and latest['ma10'] > latest['ma50']:
                ma_signal = "买入"  # 价格和短期MA高于长期MA
            elif latest['close'] < latest['ma10'] * 0.98:  # 已跌破短期均线且下跌明显
                ma_signal = "卖出"
            
        # 5. 价格趋势信号 - 更敏感的卖出条件
        trend_signal = "持有"
        if 'trend_5d' in latest and 'trend_10d' in latest:
            if latest['trend_5d'] > 0 and latest['trend_10d'] > 0:
                trend_signal = "买入"  # 短期和中期趋势向上
            elif latest['trend_5d'] < -0.01:  # 短期趋势加速下跌
                trend_signal = "卖出"
        
        # 计算买入和卖出信号数量
        signals = [rsi_signal, macd_signal, bb_signal, ma_signal, trend_signal]
        buy_count = signals.count("买入")
        sell_count = signals.count("卖出")
        
        # 输出各信号的详情便于调试
        logger.debug(f"信号详情 - RSI: {rsi_signal}, MACD: {macd_signal}, BB: {bb_signal}, MA: {ma_signal}, 趋势: {trend_signal}")
        
        # 基于计数生成最终信号 - 对卖出更敏感
        if buy_count >= 2 and buy_count > sell_count:
            final_signal = "买入"
        elif sell_count >= 1:  # 只要有一个卖出信号就考虑卖出，更敏感
            final_signal = "卖出"
        else:
            final_signal = "持有"
            
        logger.info(f"多因子信号: RSI={rsi_signal}, MACD={macd_signal}, BB={bb_signal}, MA={ma_signal}, 趋势={trend_signal} => {final_signal}")
        return final_signal
        
    except Exception as e:
        logger.error(f"多因子信号计算失败: {str(e)}")
        return "持有"

# 修改1: 增加一个分层卖出的函数
def calculate_partial_sell_decision(data, symbol, position_info=None):
    """
    计算是否应该部分卖出的决策
    
    参数:
        data (DataFrame): 股票数据
        symbol (str): 股票代码
        position_info (dict, optional): 持仓信息
        
    返回:
        tuple: (是否部分卖出, 建议卖出比例, 原因)
    """
    try:
        # 获取最新数据点
        latest = data.iloc[-1]
        
        # 初始化卖出理由
        sell_reasons = []
        
        # 检查各个指标是否触发初步卖出信号
        
        # 1. RSI超过初级警戒线
        if 'rsi' in latest and latest['rsi'] >= 65:
            sell_reasons.append(f"RSI达到{latest['rsi']:.1f}(>65)")
        
        # 2. 价格接近布林带上轨
        if 'bb_upper' in latest and 'close' in latest:
            upper_ratio = (latest['close'] / latest['bb_upper'])
            if upper_ratio > 0.95:
                sell_reasons.append(f"价格接近布林带上轨({upper_ratio:.2f})")
        
        # 3. MACD柱状图开始下降
        if 'macd_hist' in latest and len(data) > 1:
            prev_hist = data['macd_hist'].iloc[-2]
            curr_hist = latest['macd_hist']
            
            if curr_hist > 0 and curr_hist < prev_hist:
                sell_reasons.append("MACD柱状图高位回落")
        
        # 4. 检查盈利情况
        if position_info is not None:
            current_price = latest['close'] if 'close' in latest else None
            entry_price = position_info.get('entry_price')
            
            if current_price and entry_price:
                profit_percent = (current_price - entry_price) / entry_price * 100
                
                # 盈利超过10%考虑部分获利了结
                if profit_percent >= 10:
                    sell_reasons.append(f"已盈利{profit_percent:.1f}%(>10%)")
        
        # 根据信号数量决定是否部分卖出及比例
        if len(sell_reasons) >= 1:
            # 根据信号数量确定卖出比例
            sell_ratio = min(0.3 * len(sell_reasons), 0.5)  # 最多卖出50%
            reason_text = "、".join(sell_reasons)
            
            return True, sell_ratio, reason_text
        
        return False, 0, ""
        
    except Exception as e:
        logger.error(f"部分卖出决策计算失败: {str(e)}")
        return False, 0, ""
    
# 修改2: 改进hybrid_trading_decision函数，加入分层卖出逻辑
def hybrid_trading_decision(symbol, context, data=None, ml_model=None, ml_scaler=None, rl_model_path=None):
    """
    修改后的混合交易决策函数，支持分层卖出策略
    """
    try:
        if data is None:
            data = get_stock_data(symbol)
        
        if data is None or len(data) < 20:
            logger.warning(f"{symbol} 数据不足，无法做出决策")
            return "持有"
        
        # 计算技术指标
        data = calculate_technical_indicators(data)
        if data is None:
            logger.error(f"{symbol} 技术指标计算失败")
            return "持有"
            
        # 1. 多因子规则信号
        rule_signal = calculate_multi_factor_signal(data, context)
        logger.info(f"{symbol} 多因子规则信号: {rule_signal}")
        
        # 2. 机器学习信号
        ml_signal = "持有"
        if ml_model is not None and ml_scaler is not None:
            try:
                from .ml.predict import predict_proba
                prob_up = predict_proba(data.iloc[-1:], ml_model, ml_scaler)
                logger.info(f"{symbol} ML预测上涨概率: {prob_up:.3f}")
                
                # 调整ML信号阈值: 更敏感的卖出信号但更严格的买入信号
                if prob_up > 0.67:  # 提高买入阈值
                    ml_signal = "买入"
                elif prob_up < 0.45:  # 略微提高卖出阈值，使其不那么敏感
                    ml_signal = "卖出"
            except Exception as e:
                logger.error(f"ML预测失败: {str(e)}")
        
        # 3. 强化学习信号
        rl_signal = "持有"
        if rl_model_path and os.path.exists(rl_model_path):
            try:
                from .rl.agent import rl_decision
                action = rl_decision(symbol, data, rl_model_path)
                if action == 2:
                    rl_signal = "买入"
                elif action == 0:
                    rl_signal = "卖出"
                logger.info(f"{symbol} RL决策信号: {rl_signal}")
            except Exception as e:
                logger.error(f"RL决策失败: {str(e)}")
        
        # 4. 检查当前持仓
        api = get_api_client()
        has_position = False
        position_value = 0
        position_info = None
        
        try:
            position = api.get_position(symbol)
            has_position = True
            position_value = float(position.market_value)
            
            # 保存持仓信息用于分层卖出决策
            position_info = {
                'entry_price': float(position.avg_entry_price),
                'current_price': float(position.current_price),
                'qty': float(position.qty),
                'market_value': position_value
            }
            
            logger.info(f"当前持有 {position.qty} 股 {symbol}，市值 ${position_value:.2f}")
        except Exception:
            logger.info(f"当前未持有 {symbol}")
        
        # 5. 综合决策
        signals = [rule_signal, ml_signal, rl_signal]
        buy_count = signals.count("买入")
        sell_count = signals.count("卖出")
        
        # 输出信号汇总
        logger.info(f"{symbol} 信号汇总 - 规则: {rule_signal}, ML: {ml_signal}, RL: {rl_signal}")
        
        # 市场条件筛选
        extreme_market = context['market_regime'] == 'bear' and context['vix_level'] == 'high'
        
        # --- 修改的决策逻辑开始 ---
        
        # 全部卖出条件: 有持仓 + 至少2个卖出信号
        if (sell_count >= 2) and has_position:
            logger.info(f"{symbol} 产生完全卖出信号，满足至少2个卖出信号条件")
            return "卖出"
            
        # 部分卖出条件: 有持仓 + 满足部分卖出的技术条件
        elif has_position:
            should_partial_sell, sell_ratio, reason = calculate_partial_sell_decision(data, symbol, position_info)
            
            if should_partial_sell:
                # 如果建议部分卖出比例超过80%，可以直接全部卖出
                if sell_ratio > 0.8:
                    logger.info(f"{symbol} 部分卖出比例 {sell_ratio:.1%} 过高，转为完全卖出")
                    return "卖出"
                    
                # 否则返回部分卖出的决策，包含比例信息
                logger.info(f"{symbol} 产生部分卖出信号 {sell_ratio:.1%}，原因: {reason}")
                return f"部分卖出:{sell_ratio:.2f}"
        
        # 买入条件: 无持仓 + 至少1个买入信号 + 非极端市场
        if (buy_count >= 1) and not has_position and not extreme_market:
            logger.info(f"{symbol} 产生买入信号")
            return "买入"
            
        # --- 修改的决策逻辑结束 ---
        
        return "持有"
            
    except Exception as e:
        logger.error(f"交易决策失败: {str(e)}")
        return "持有"

def calculate_rule_signal(latest, context):
    """计算基于规则的交易信号"""
    try:
        # 动态阈值设置
        rsi_buy = 33
        rsi_sell = 65
        
        # 根据市场状态调整阈值
        if context['market_regime'] == 'bull':
            rsi_buy += 5
            rsi_sell += 5
        elif context['market_regime'] == 'bear':
            rsi_buy -= 5
            rsi_sell -= 5
        
        # 波动率调整
        if context['vix_level'] == 'high':
            rsi_buy = max(25, rsi_buy - 10)
        elif context['vix_level'] == 'low':
            rsi_sell = min(80, rsi_sell + 5)
        
        # 生成信号
        if latest['rsi'] <= rsi_buy:
            return "买入"
        elif latest['rsi'] >= rsi_sell:
            return "卖出"
        else:
            return "持有"
            
    except Exception as e:
        logger.error(f"规则信号计算失败: {str(e)}")
        return "持有"

# 修改5: 增强check_profit_taking函数，更敏感地锁定利润
def check_profit_taking(symbol, data=None):
    """
    检查是否应该锁定利润的增强版函数
    """
    try:
        api = get_api_client()
        
        # 获取当前持仓
        position = api.get_position(symbol)
        current_price = float(position.current_price)
        entry_price = float(position.avg_entry_price)
        
        # 计算当前盈利百分比
        profit_percent = (current_price - entry_price) / entry_price * 100
        
        # 盈利锁定标准 (基于盈利百分比的分级锁定) - 降低阈值，更积极锁定利润
        if profit_percent >= 15:  # 从20%降低到15%
            # 超过15%盈利，考虑锁定大部分利润
            logger.info(f"{symbol} 盈利已达 {profit_percent:.2f}% (>= 15%)，建议锁定利润")
            return "卖出"
        elif profit_percent >= 8:  # 从10%降低到8%
            # 8-15%盈利，可以考虑锁定部分利润
            
            # 如果有其他卖出信号，更倾向于卖出
            if data is not None:
                latest = data.iloc[-1]
                if ('rsi' in latest and latest['rsi'] > 58) or ('close' in latest and 'bb_upper' in latest and latest['close'] > latest['bb_upper'] * 0.95):
                    logger.info(f"{symbol} 盈利已达 {profit_percent:.2f}% (>= 8%) 且技术指标显示超买，建议锁定利润")
                    return "卖出"
            
            logger.info(f"{symbol} 盈利已达 {profit_percent:.2f}% (>= 8%)，考虑部分锁定利润")
            return "部分卖出:0.40"  # 卖出40%仓位
        elif profit_percent >= 5:  # 增加一个5%的部分止盈档位
            if data is not None:
                latest = data.iloc[-1]
                # 如果同时有下跌的技术信号，锁定小部分利润
                if ('rsi' in latest and latest['rsi'] > 65) or ('macd_hist' in latest and latest['macd_hist'] < 0):
                    logger.info(f"{symbol} 盈利已达 {profit_percent:.2f}% (>= 5%) 且有下跌技术信号，建议部分锁定利润")
                    return "部分卖出:0.20"  # 卖出20%仓位
        
        # 检查是否接近突破支撑位
        if data is not None and len(data) > 20:
            latest = data.iloc[-1]
            if 'close' in latest and 'ma50' in latest:
                # 如果有任何盈利，且价格接近跌破50日均线，考虑保护利润
                if profit_percent > 0 and latest['close'] < latest['ma50'] * 1.02:
                    logger.info(f"{symbol} 有盈利且价格接近跌破50日均线，建议保护利润")
                    return "部分卖出:0.30"  # 卖出30%仓位
        
        return "持有"
            
    except Exception as e:
        logger.error(f"检查利润锁定失败: {str(e)}")
        return "持有"

# 修改6: 在run_intelligent_trading_system函数中更新检查利润锁定的逻辑
def update_profit_taking_check(decision, symbol, data, current_positions):
    """
    增强版的利润锁定检查，与原有交易决策结合
    
    参数:
        decision (str): 原始交易决策
        symbol (str): 股票代码
        data (DataFrame): 股票数据
        current_positions (list): 当前持仓列表
        
    返回:
        str: 更新后的决策
    """
    # 如果原决策不是持有，或者当前没有持仓，则保持原决策
    if decision != "持有" or symbol not in current_positions:
        return decision
        
    # 检查是否应该锁定利润
    try:
        profit_decision = check_profit_taking(symbol, data)
        if profit_decision != "持有":
            logger.info(f"{symbol} 原决定为持有，但利润锁定检查后改为: {profit_decision}")
            return profit_decision
    except Exception as e:
        logger.error(f"检查{symbol}利润锁定时出错: {str(e)}")
        
    return decision

def get_market_context():
    """
    修改市场上下文函数以处理 VIX 数据缺失的情况
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
        
        # 3. 使用 SPY 的波动率代替 VIX
        vix_level = "medium"  # 默认中等
        
        if spy_data is not None and len(spy_data) > 20:
            # 计算 SPY 的波动率
            returns = spy_data['close'].pct_change()
            volatility = returns.std() * np.sqrt(252)  # 年化波动率
            
            if volatility > 0.3:  # 25% 年化波动率
                vix_level = "high"
            elif volatility < 0.15:  # 15% 年化波动率
                vix_level = "low"
        
        # 组合上下文
        context = {
            'market_regime': market_regime,
            'vix_level': vix_level,
            'spy_data': spy_data,
            'timestamp': datetime.now()
        }
        
        logger.info(f"市场状态: {market_regime}, 波动水平: {vix_level}")
        return context
        
    except Exception as e:
        logger.error(f"获取市场上下文失败: {str(e)}")
        return {
            'market_regime': 'neutral',
            'vix_level': 'medium',
            'spy_data': None,
            'timestamp': datetime.now()
        }

# 修改3: 修改execute_trade函数，处理部分卖出的情况
def execute_trade(symbol, action, qty=None, force_trade=False):
    """
    执行交易，支持部分卖出功能
    """
    # 检查是否是部分卖出指令
    is_partial_sell = False
    sell_ratio = 1.0  # 默认卖出全部
    
    if action.startswith("部分卖出:"):
        is_partial_sell = True
        # 从指令中解析卖出比例
        try:
            sell_ratio = float(action.split(":")[-1])
            action = "卖出"  # 将动作重设为卖出
        except:
            logger.error(f"无法解析部分卖出比例: {action}")
            sell_ratio = 0.3  # 默认卖出30%
            action = "卖出"
    
    logger.info(f"尝试执行交易: {symbol} {action} " + (f"{sell_ratio:.1%}" if is_partial_sell else "") + (f" {qty}股" if qty else ""))
    
    if action == "持有":
        logger.info(f"{symbol}: 保持当前仓位")
        return None
    
    try:
        # Get API client
        api = get_api_client()
        
        # Check if market is open
        clock = api.get_clock()
        if not clock.is_open:
            logger.warning("市场已关闭，无法交易")
            return None
        
        # Get account information
        account = api.get_account()
        buying_power = float(account.buying_power)
        logger.info(f"当前账户状态: {account.status}")
        logger.info(f"可用资金: ${buying_power:.2f}")
        
        # Calculate quantity if not provided
        if qty is None or qty <= 0:
            if action == "买入":
                # Get latest price
                latest_quote = api.get_latest_quote(symbol)
                price = float(latest_quote.ap) if hasattr(latest_quote, 'ap') else None
                
                if price is None or price <= 0:
                    logger.error(f"无法获取{symbol}的有效价格")
                    return None
                
                # Use 5% of buying power for each position
                position_value = buying_power * 0.05
                qty = position_value / price
                
                # Ensure minimum quantity
                if qty < 0.01:  # For fractional shares
                    logger.warning(f"计算的交易数量过小: {qty}, 使用最小数量0.01")
                    qty = 0.01
                    
                logger.info(f"自动计算买入数量: {qty} 股 {symbol}")
            else:
                # For selling, get current position
                try:
                    position = api.get_position(symbol)
                    position_qty = float(position.qty)
                    
                    # 如果是部分卖出，按比例计算
                    if is_partial_sell:
                        qty = position_qty * sell_ratio
                        logger.info(f"部分卖出{sell_ratio:.1%}仓位: {qty:.6f} 股 {symbol}")
                    else:
                        qty = position_qty
                        logger.info(f"卖出全部持仓: {qty} 股 {symbol}")
                except Exception as e:
                    logger.error(f"获取{symbol}持仓失败: {str(e)}")
                    return None
        
        # Execute order
        side = "buy" if action == "买入" else "sell"
        try:
            order = api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )
            logger.info(f"订单提交成功: {order.id}")
            return order
            
        except Exception as e:
            logger.error(f"订单提交失败: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"交易执行失败: {str(e)}")
        return None

def run_intelligent_trading_system(symbols=None, schedule_retrain_enabled=True, use_short_term=True):
    """
    运行智能交易系统 - 修改版本，整合了update_profit_taking_check函数
    
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
        symbols = SYMBOLS
    
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
        
        # 获取交易决策
        decision = hybrid_trading_decision(
            symbol, 
            context,
            data=data,
            ml_model=ml_model, 
            ml_scaler=ml_scaler, 
            rl_model_path=rl_model_path
        )

        # 使用增强版的利润锁定检查，与原有交易决策结合
        decision = update_profit_taking_check(decision, symbol, data, current_positions)
        
        # 额外检查：对于已持有的股票，检查是否应该锁定利润
        if decision == "持有" and symbol in current_positions:
            try:
                # 如果持有，检查是否应该锁定利润
                profit_decision = check_profit_taking(symbol, data)
                if profit_decision == "卖出":
                    decision = profit_decision
                    logger.info(f"{symbol} 原决定为持有，但利润锁定检查后改为: {decision}")
            except Exception as e:
                logger.error(f"检查{symbol}利润锁定时出错: {str(e)}")
        
        # 确定是否执行交易
        # 1. 如果是买入信号
        if decision == "买入":
            logger.info(f"买入信号: {symbol}")
            
            # 强制执行交易进行测试
            force_trade = True  # 在测试时设置为True，正式环境改为False
            
            # 更宽松的买入条件：活跃股票列表更大，不严格限制持仓数量
            # 在测试模式下，最大持仓数量适当放宽(max_positions * 1.5)
            effective_max_positions = int(max_positions * 1.5) if force_trade else max_positions
            
            if force_trade or (len(current_positions) < effective_max_positions):
                logger.info(f"{symbol} 符合买入条件，执行交易")
                
                # 如果是新仓位，使用增强版的execute_trade函数自动计算数量
                qty = None  # 让系统自动计算购买数量
                
                result = execute_trade(symbol, decision, qty=qty, force_trade=force_trade)
                if result:
                    results.append(result)
                    if symbol not in current_positions:
                        current_positions.append(symbol)
                    
                    # 立即设置止损
                    try:
                        # 获取更保守的止损百分比
                        stop_percent = float(trading_params.get('STOP_LOSS_PERCENT', 0.08))
                        set_stop_loss(symbol, stop_percent=stop_percent)
                    except Exception as e:
                        logger.error(f"为{symbol}设置初始止损时出错: {str(e)}")
            else:
                reason = f"当前持仓数量({len(current_positions)})已达最大限制({effective_max_positions})"
                logger.info(f"{symbol} 有买入信号但{reason}，跳过交易")
                
        # 2. 如果是卖出信号或部分卖出信号
        elif decision == "卖出" or decision.startswith("部分卖出:"):
            if symbol in current_positions:
                logger.info(f"{symbol} 符合{'部分' if decision.startswith('部分卖出:') else ''}卖出条件，执行交易")
                result = execute_trade(symbol, decision)  # 修改后的execute_trade支持部分卖出
                if result:
                    results.append(result)
                    
                    # 如果是完全卖出，从持仓列表中移除
                    if decision == "卖出":
                        current_positions.remove(symbol)
            else:
                logger.info(f"{symbol} 有卖出信号但当前未持有，跳过交易")
        
        # 3. 持有信号不执行交易，但更新止损
        else:
            logger.info(f"{symbol} 持有信号，无需交易")
        
        # 更新现有持仓的止损
        if symbol in current_positions:
            try:
                # 使用较为激进的移动止损参数
                trail_percent = float(trading_params.get('TRAILING_STOP_PERCENT', 0.05))
                
                # 更新止损
                update_stop_loss(symbol, trail_percent=trail_percent)
            except Exception as e:
                logger.error(f"更新{symbol}止损时出错: {str(e)}")
    
    # 查看最终持仓
    logger.info("\n当前持仓:")
    try:
        positions = api.list_positions()
        for position in positions:
            # 计算当前盈亏百分比
            entry_price = float(position.avg_entry_price)
            current_price = float(position.current_price)
            profit_pct = ((current_price / entry_price) - 1) * 100
            
            print(f"{position.symbol}: {position.qty} 股, 均价: ${entry_price:.2f}, "
                  f"现价: ${current_price:.2f}, 盈亏: {profit_pct:.2f}%, 市值: ${position.market_value:.2f}")
    except Exception as e:
        logger.error(f"获取持仓失败: {str(e)}")
    
    # 生成每日报告
    if datetime.now().hour >= 16:  # 如果当前时间在下午4点以后
        generate_daily_report()
    
    # 设置定期任务
    if schedule_retrain_enabled:
        # 每周一凌晨2点重新训练模型
        schedule.every().monday.at("02:00").do(schedule_retrain, symbols=active_symbols)
        logger.info("已设置每周一凌晨2点重新训练模型")
        
        # 设置每日健康检查
        schedule.every(60).minutes.do(update_health_check)
        logger.info("已设置每小时健康检查")
        
        # 设置每日报告任务
        schedule.every().day.at("16:30").do(generate_daily_report)
        logger.info("已设置每日16:30生成报告")
        
        # 根据是否使用短期策略设置不同的监控频率 - 更频繁的监控
        if use_short_term:
            # 短期策略需要更频繁监控 - 从5分钟减少到3分钟
            schedule.every(3).minutes.do(monitor_market, symbols=symbols, interval_minutes=3)
            logger.info("已设置每3分钟监控市场（短期策略）")
        else:
            # 标准策略监控频率 - 从15分钟减少到10分钟
            schedule.every(10).minutes.do(monitor_market, symbols=symbols, interval_minutes=10)
            logger.info("已设置每10分钟监控市场（标准策略）")
    
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
        symbols = SYMBOLS
        
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
        symbols = SYMBOLS
    
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
        symbols = SYMBOLS

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
    """检查当前是否是市场交易时间"""
    try:
        api = get_api_client()
        
        # 直接尝试获取账户信息以测试认证，不依赖市场状态API
        account = api.get_account()
        logger.info(f"API连接成功: 账户状态 {account.status}")
        
        # 然后再尝试获取市场状态
        try:
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
            logger.warning(f"获取市场状态时出错，假设市场开盘: {str(e)}")
            # 认为市场总是开盘 - 确保交易得以执行
            return True
            
    except Exception as e:
        logger.error(f"API连接失败: {str(e)}")
        # 当无法连接时，默认假设市场关闭
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
        symbols = SYMBOLS
        
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
        
        symbols = SYMBOLS
        
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