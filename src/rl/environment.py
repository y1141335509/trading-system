# src/rl/environment.py

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import logging

# 设置日志
logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """
    交易环境，用于强化学习
    
    这个环境模拟股票交易，代理可以执行买入、持有或卖出操作。
    奖励基于交易结果和价格变动。
    """
    
    def __init__(self, data, initial_balance=10000.0, transaction_cost=0.001, window_size=10):
        """
        初始化交易环境
        
        参数:
            data (DataFrame): 包含股票数据的DataFrame，需要包含"close"等价格信息
            initial_balance (float): 初始账户余额
            transaction_cost (float): 交易成本比例
            window_size (int): 观察窗口大小
        """
        super(TradingEnvironment, self).__init__()
        
        # 保存传入的参数
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        
        # 确保数据有足够的特征
        self._validate_data()
        
        # 环境状态
        self.balance = initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.current_price = 0
        self.last_trade_tick = 0
        self.total_steps = len(data) - 1
        
        # 定义动作空间：0=卖出, 1=持有, 2=买入
        self.action_space = spaces.Discrete(3)
        
        # 定义观察空间
        # [持仓量, 账户余额, 归一化技术指标...]
        obs_dim = 2 + self._get_observation_dimension()  # 持仓量 + 账户余额 + 特征维度
        
        # 正规化后的值应该在一个合理范围内
        self.observation_space = spaces.Box(
            low=-np.ones(obs_dim) * 10,  # 合理的负范围
            high=np.ones(obs_dim) * 10,  # 合理的正范围
            dtype=np.float32
        )
        
        # 跟踪交易历史
        self.trades = []
        self.portfolio_values = []
    
    def _validate_data(self):
        """验证数据是否包含必要的列和足够的行"""
        required_columns = ['close']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"数据缺少必要的列: {col}")
        
        if len(self.data) <= self.window_size:
            raise ValueError(f"数据行数 ({len(self.data)}) 小于窗口大小 ({self.window_size})")
    
    def _get_observation_dimension(self):
        """获取观察空间的特征维度"""
        # 排除不作为特征的列
        excluded_columns = ['date', 'volume', 'open', 'high', 'low']
        feature_columns = [col for col in self.data.columns if col not in excluded_columns]
        return len(feature_columns)
    
    def reset(self, seed=None, options=None):
        """
        重置环境到初始状态
        
        参数:
            seed (int, optional): 随机种子
            options (dict, optional): 重置选项
            
        返回:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        
        # 重置环境状态
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.last_trade_tick = 0
        self.trades = []
        self.portfolio_values = []
        
        # 获取当前价格
        self.current_price = self.data.iloc[self.current_step]['close']
        
        # 返回初始观察和空的info字典
        return self._get_observation(), {}
    
    def step(self, action):
        """
        执行一个交易动作并更新环境
        
        参数:
            action (int): 交易动作 (0=卖出, 1=持有, 2=买入)
            
        返回:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # 保存执行动作前的信息，用于计算奖励
        prev_portfolio_value = self.balance + self.shares_held * self.current_price
        
        # 获取当前价格
        self.current_price = self.data.iloc[self.current_step]['close']
        
        # 执行交易动作
        self._execute_trade_action(action)
        
        # 计算当前投资组合价值
        current_portfolio_value = self.balance + self.shares_held * self.current_price
        
        # 添加到投资组合历史
        self.portfolio_values.append(current_portfolio_value)
        
        # 移动到下一步
        self.current_step += 1
        
        # 检查是否结束
        terminated = self.current_step >= self.total_steps
        truncated = False  # 暂不实现提前终止
        
        # 计算奖励
        reward = self._calculate_reward(action, prev_portfolio_value, current_portfolio_value)
        
        # 获取新的观察
        observation = self._get_observation()
        
        # 构建额外信息字典
        info = {
            'step': self.current_step,
            'portfolio_value': current_portfolio_value,
            'price': self.current_price,
            'shares_held': self.shares_held,
            'balance': self.balance
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        """
        构建当前状态的观察向量
        
        返回:
            ndarray: 状态观察向量
        """
        # 获取当前数据点
        features = self.data.iloc[self.current_step]
        
        # 构建基本观察（持仓和账户余额）
        # 将余额正规化为合理范围
        normalized_balance = self.balance / self.initial_balance
        
        # 起始观察
        observation = [self.shares_held, normalized_balance]
        
        # 添加其他特征（排除不需要的列）
        excluded_columns = ['date', 'volume', 'open', 'high', 'low']
        for col in features.index:
            if col not in excluded_columns and col != 'close':
                # 添加特征值（技术指标通常已经正规化或可以按原样使用）
                observation.append(features[col])
        
        # 转换为NumPy数组并确保为float32类型
        return np.array(observation, dtype=np.float32)
    
    def _execute_trade_action(self, action):
        """
        执行交易动作
        
        参数:
            action (int): 交易动作 (0=卖出, 1=持有, 2=买入)
        """
        if action == 0:  # 卖出
            if self.shares_held > 0:
                # 计算卖出收益（减去交易成本）
                sell_amount = self.shares_held * self.current_price
                sell_cost = sell_amount * self.transaction_cost
                self.balance += (sell_amount - sell_cost)
                
                # 记录交易
                self.trades.append({
                    'step': self.current_step,
                    'price': self.current_price,
                    'type': 'sell',
                    'shares': self.shares_held,
                    'cost': sell_cost,
                    'balance_after': self.balance
                })
                
                # 更新持仓
                self.shares_held = 0
                self.last_trade_tick = self.current_step
                
        elif action == 2:  # 买入
            if self.balance > 0:
                # 计算可买入的最大股数
                max_shares = self.balance / (self.current_price * (1 + self.transaction_cost))
                shares_to_buy = max_shares  # 买入最大可能股数
                
                # 计算买入成本
                buy_amount = shares_to_buy * self.current_price
                buy_cost = buy_amount * self.transaction_cost
                total_cost = buy_amount + buy_cost
                
                # 确保不超过账户余额
                if total_cost <= self.balance:
                    self.balance -= total_cost
                    self.shares_held += shares_to_buy
                    
                    # 记录交易
                    self.trades.append({
                        'step': self.current_step,
                        'price': self.current_price,
                        'type': 'buy',
                        'shares': shares_to_buy,
                        'cost': buy_cost,
                        'balance_after': self.balance
                    })
                    
                    self.last_trade_tick = self.current_step
    
    def _calculate_reward(self, action, prev_portfolio_value, current_portfolio_value):
        """
        计算奖励函数
        
        参数:
            action (int): 执行的动作
            prev_portfolio_value (float): 动作前的投资组合价值
            current_portfolio_value (float): 动作后的投资组合价值
            
        返回:
            float: 计算的奖励
        """
        # 计算投资组合价值变化
        portfolio_return = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value > 0 else 0
        
        # 基础奖励是投资组合回报
        reward = portfolio_return * 100  # 放大奖励
        
        # 根据动作调整奖励
        if action == 0:  # 卖出
            # 如果市场下跌且卖出，奖励加倍
            if portfolio_return < 0:
                reward = abs(reward) * 1.5
            # 如果市场上涨但卖出，轻微惩罚
            elif portfolio_return > 0 and self.shares_held > 0:
                reward *= 0.5
        
        elif action == 2:  # 买入
            # 如果市场上涨且买入，奖励加倍
            if portfolio_return > 0:
                reward *= 1.5
            # 如果市场下跌但买入，轻微惩罚
            elif portfolio_return < 0:
                reward *= 0.5
        
        # 对频繁交易施加小惩罚（抑制过度交易）
        if action != 1 and (self.current_step - self.last_trade_tick) < 5:
            reward -= 0.1
        
        # 鼓励更均衡的投资组合
        if self.balance < self.initial_balance * 0.1 or self.balance > self.initial_balance * 0.9:
            reward -= 0.1
        
        return reward
    
    def render(self, mode='human'):
        """
        渲染环境状态
        
        参数:
            mode (str): 渲染模式
        """
        if self.current_step > 0:
            portfolio_value = self.balance + self.shares_held * self.current_price
            print(f"Step: {self.current_step}, Price: ${self.current_price:.2f}, "
                  f"Balance: ${self.balance:.2f}, Shares: {self.shares_held:.6f}, "
                  f"Portfolio Value: ${portfolio_value:.2f}")
    
    def get_portfolio_value(self):
        """
        获取当前投资组合价值
        
        返回:
            float: 当前投资组合总价值
        """
        return self.balance + self.shares_held * self.current_price
    
    def get_performance_summary(self):
        """
        获取交易绩效摘要
        
        返回:
            dict: 绩效指标
        """
        # 确保有交易历史
        if not self.portfolio_values:
            return {
                'initial_value': self.initial_balance,
                'final_value': self.balance,
                'return': 0,
                'trades': 0
            }
        
        # 计算关键指标
        initial_value = self.initial_balance
        final_value = self.portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # 计算每日回报
        returns = []
        for i in range(1, len(self.portfolio_values)):
            daily_return = (self.portfolio_values[i] - self.portfolio_values[i-1]) / self.portfolio_values[i-1]
            returns.append(daily_return)
        
        # 计算夏普比率（假设无风险收益率为0）
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if returns and np.std(returns) > 0 else 0
        
        # 统计持仓状态
        buy_and_hold_return = (self.data.iloc[-1]['close'] - self.data.iloc[0]['close']) / self.data.iloc[0]['close']
        
        return {
            'initial_value': initial_value,
            'final_value': final_value,
            'return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'trades': len(self.trades),
            'buy_hold_return': buy_and_hold_return
        }