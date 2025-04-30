# src/rl/agent.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import logging
import random
from collections import deque
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

from .environment import TradingEnvironment

# 设置日志
logger = logging.getLogger(__name__)

class DQNAgent:
    """
    深度Q网络代理
    
    使用DQN算法进行强化学习的代理类。
    """
    
    def __init__(self, state_size, action_size, model=None, lr=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01, batch_size=32, memory_size=5000):
        """
        初始化DQN代理
        
        参数:
            state_size (int): 状态空间大小
            action_size (int): 动作空间大小
            model (Keras Model, optional): 预训练模型
            lr (float): 学习率
            gamma (float): 折扣因子
            epsilon (float): 初始探索率
            epsilon_decay (float): 探索率衰减
            epsilon_min (float): 最小探索率
            batch_size (int): 批次大小
            memory_size (int): 经验回放内存大小
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = lr
        self.batch_size = batch_size
        
        # 构建或加载模型
        if model is not None:
            self.model = model
        else:
            self.model = self._build_model()
        
        # 目标网络
        self.target_model = self._build_model()
        self.update_target_model()
        
        # 跟踪训练步骤
        self.train_step = 0
    
    def _build_model(self):
        """
        构建DQN模型
        
        返回:
            Keras Model: DQN神经网络模型
        """
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """更新目标网络权重"""
        self.target_model.set_weights(self.model.get_weights())
    
    def save_memory(self, state, action, reward, next_state, done):
        """
        将经验保存到回放内存
        
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, explore=True):
        """
        根据当前状态选择动作
        
        参数:
            state: 当前状态
            explore (bool): 是否进行探索
            
        返回:
            int: 选择的动作
        """
        if explore and np.random.rand() <= self.epsilon:
            # 探索 - 随机选择动作
            return random.randrange(self.action_size)
            
        # 开发 - 选择最佳动作
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size=None):
        """
        从经验回放中学习
        
        参数:
            batch_size (int, optional): 批次大小，默认使用初始化时设置的值
            
        返回:
            float: 训练损失
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # 确保内存中有足够的数据
        if len(self.memory) < batch_size:
            return 0.0
        
        # 从内存中随机抽样
        minibatch = random.sample(self.memory, batch_size)
        
        # 准备训练数据
        states = []
        targets = []
        
        for state, action, reward, next_state, done in minibatch:
            # 目标Q值
            target = reward
            if not done:
                # Q-target = r + γ * max(Q')
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state.reshape(1, -1), verbose=0)[0])
            
            # 目标函数
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action] = target
            
            # 添加到训练数据
            states.append(state.reshape(1, -1)[0])
            targets.append(target_f[0])
        
        # 训练模型
        history = self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        
        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 更新训练步骤
        self.train_step += 1
        
        # 定期更新目标网络
        if self.train_step % 10 == 0:
            self.update_target_model()
        
        return history.history['loss'][0] if 'loss' in history.history else 0.0
    
    def save(self, filepath):
        """
        保存模型
        
        参数:
            filepath (str): 保存路径
        """
        self.model.save(filepath)
        
        # 保存agent参数
        params = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'gamma': self.gamma,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }
        
        # 保存参数到JSON文件
        params_file = filepath.replace('.h5', '_params.joblib')
        joblib.dump(params, params_file)
        
        logger.info(f"模型已保存到 {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        加载模型
        
        参数:
            filepath (str): 模型路径
            
        返回:
            DQNAgent: 加载的代理
        """
        # 加载参数
        params_file = filepath.replace('.h5', '_params.joblib')
        
        if os.path.exists(params_file):
            params = joblib.load(params_file)
        else:
            # 如果没有参数文件，使用默认值
            params = {
                'state_size': 10,  # 默认状态大小
                'action_size': 3,  # 默认动作大小
                'epsilon': 0.1,    # 较低的探索率用于评估
                'epsilon_min': 0.01,
                'epsilon_decay': 0.995,
                'gamma': 0.95,
                'learning_rate': 0.001,
                'batch_size': 32
            }
        
        # 加载模型
        try:
            model = load_model(filepath)
            logger.info(f"模型已从 {filepath} 加载")
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            return None
        
        # 创建代理
        agent = cls(
            state_size=params['state_size'],
            action_size=params['action_size'],
            model=model,
            lr=params['learning_rate'],
            gamma=params['gamma'],
            epsilon=params['epsilon'],
            epsilon_decay=params['epsilon_decay'],
            epsilon_min=params['epsilon_min'],
            batch_size=params['batch_size']
        )
        
        return agent

def train_agent(data, model_path=None, episodes=50, window_size=10, initial_balance=10000, 
               batch_size=32, gamma=0.95, epsilon_decay=0.995, model_dir='models',
               checkpoint_interval=10):
    """
    训练DQN代理
    
    参数:
        data (DataFrame): 训练数据
        model_path (str, optional): 预训练模型路径
        episodes (int): 训练轮数
        window_size (int): 观察窗口大小
        initial_balance (float): 初始账户余额
        batch_size (int): 批次大小
        gamma (float): 折扣因子
        epsilon_decay (float): 探索率衰减
        model_dir (str): 模型保存目录
        checkpoint_interval (int): 保存检查点的间隔
        
    返回:
        tuple: (代理, 训练历史)
    """
    # 创建环境
    env = TradingEnvironment(data, initial_balance=initial_balance, window_size=window_size)
    
    # 获取状态大小
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # 创建或加载代理
    if model_path and os.path.exists(model_path):
        agent = DQNAgent.load(model_path)
        if agent is None:
            logger.warning(f"加载模型失败，将创建新模型")
            agent = DQNAgent(state_size=state_size, action_size=action_size, 
                            gamma=gamma, epsilon_decay=epsilon_decay, batch_size=batch_size)
    else:
        agent = DQNAgent(state_size=state_size, action_size=action_size, 
                        gamma=gamma, epsilon_decay=epsilon_decay, batch_size=batch_size)
    
    # 训练历史
    history = {
        'episode_rewards': [],
        'portfolio_values': [],
        'losses': []
    }
    
    # 创建模型目录
    os.makedirs(model_dir, exist_ok=True)
    
    # 训练循环
    for episode in range(episodes):
        # 重置环境
        observation, _ = env.reset()
        
        # 记录当前轮次信息
        episode_reward = 0
        episode_loss = []
        
        # 训练步骤
        while True:
            # 选择动作
            action = agent.act(observation)
            
            # 执行动作
            next_observation, reward, done, truncated, info = env.step(action)
            
            # 保存经验
            agent.save_memory(observation, action, reward, next_observation, done)
            
            # 训练代理
            loss = agent.replay(batch_size)
            episode_loss.append(loss)
            
            # 更新状态
            observation = next_observation
            episode_reward += reward
            
            # 如果结束，退出循环
            if done or truncated:
                break
        
        # 记录历史
        history['episode_rewards'].append(episode_reward)
        history['portfolio_values'].append(env.get_portfolio_value())
        history['losses'].append(np.mean(episode_loss) if episode_loss else 0)
        
        # 输出进度
        performance = env.get_performance_summary()
        logger.info(f"轮次: {episode+1}/{episodes}, 奖励: {episode_reward:.2f}, 最终价值: ${performance['final_value']:.2f}, 回报: {performance['return']:.2%}")
        
        # 保存检查点
        if (episode + 1) % checkpoint_interval == 0 or episode == episodes - 1:
            checkpoint_path = os.path.join(model_dir, f"dqn_model_ep{episode+1}.h5")
            agent.save(checkpoint_path)
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 保存最终模型
    final_model_path = os.path.join(model_dir, "dqn_model_final.h5")
    agent.save(final_model_path)
    
    return agent, history

def predict_action(model_path, state, explore=False):
    """
    使用训练好的模型预测动作
    
    参数:
        model_path (str): 模型路径
        state (array): 当前状态
        explore (bool): 是否探索
        
    返回:
        int: 预测的动作
    """
    try:
        # 加载代理
        agent = DQNAgent.load(model_path)
        
        if agent is None:
            logger.error("加载模型失败，无法预测")
            return 1  # 默认持有
        
        # 预测动作
        action = agent.act(state, explore=explore)
        return action
        
    except Exception as e:
        logger.error(f"预测动作时出错: {str(e)}")
        return 1  # 默认持有

def plot_training_history(history):
    """
    绘制训练历史
    
    参数:
        history (dict): 训练历史
    """
    try:
        # 创建图表
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        # 1. 轮次奖励
        axes[0].plot(history['episode_rewards'])
        axes[0].set_title('Episodic Reward')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].grid(True)
        
        # 2. 投资组合价值
        axes[1].plot(history['portfolio_values'])
        axes[1].set_title('Portfolio Value')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Value ($)')
        axes[1].grid(True)
        
        # 3. 训练损失
        axes[2].plot(history['losses'])
        axes[2].set_title('Training Loss')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Loss')
        axes[2].grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        save_dir = os.getenv('DATA_DIR', 'data')
        os.makedirs(save_dir, exist_ok=True)
        
        plot_path = os.path.join(save_dir, f"rl_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_path)
        logger.info(f"训练历史图表已保存至: {plot_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"绘制训练历史时出错: {str(e)}")

def evaluate_agent(data, model_path, initial_balance=10000, render=False):
    """
    评估代理性能
    
    参数:
        data (DataFrame):# src/rl/agent.py (continued)
        data (DataFrame): 评估数据
        model_path (str): 模型路径
        initial_balance (float): 初始账户余额
        render (bool): 是否渲染环境
        
    返回:
        dict: 评估结果
    """
    try:
        # 创建环境
        env = TradingEnvironment(data, initial_balance=initial_balance)
        
        # 加载代理
        agent = DQNAgent.load(model_path)
        
        if agent is None:
            logger.error("加载模型失败，无法评估")
            return {
                'success': False,
                'error': 'Failed to load model'
            }
        
        # 重置环境
        observation, _ = env.reset()
        
        # 评估步骤
        episode_reward = 0
        actions_taken = []
        
        while True:
            # 选择动作（不进行探索）
            action = agent.act(observation, explore=False)
            actions_taken.append(action)
            
            # 执行动作
            next_observation, reward, done, truncated, info = env.step(action)
            
            # 渲染环境
            if render:
                env.render()
            
            # 更新状态
            observation = next_observation
            episode_reward += reward
            
            # 如果结束，退出循环
            if done or truncated:
                break
        
        # 获取性能摘要
        performance = env.get_performance_summary()
        
        # 计算动作统计
        action_counts = {
            'sell': actions_taken.count(0),
            'hold': actions_taken.count(1),
            'buy': actions_taken.count(2)
        }
        
        # 合并结果
        result = {
            'success': True,
            'total_reward': episode_reward,
            'final_balance': env.balance,
            'final_shares': env.shares_held,
            'final_portfolio_value': performance['final_value'],
            'total_return': performance['return'],
            'sharpe_ratio': performance.get('sharpe_ratio', 0),
            'trades': performance['trades'],
            'action_counts': action_counts,
            'buy_hold_return': performance.get('buy_hold_return', 0)
        }
        
        logger.info(f"评估完成 - 总奖励: {episode_reward:.2f}, 最终价值: ${performance['final_value']:.2f}, 回报: {performance['return']:.2%}")
        return result
        
    except Exception as e:
        logger.error(f"评估代理时出错: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def train_rl_agent_for_symbol(symbol, data=None, episodes=50, model_dir='models'):
    """
    为特定股票训练强化学习代理
    
    参数:
        symbol (str): 股票代码
        data (DataFrame, optional): 预处理的股票数据
        episodes (int): 训练轮数
        model_dir (str): 模型保存目录
        
    返回:
        str: 模型路径或None
    """
    try:
        # 如果没有提供数据，获取数据
        if data is None:
            from ..data.fetcher import get_stock_data
            from ..data.processor import calculate_technical_indicators
            
            # 获取股票数据
            raw_data = get_stock_data(symbol, days=365)
            
            if raw_data is None or len(raw_data) < 100:
                logger.error(f"获取{symbol}数据失败或数据不足")
                return None
                
            # 计算技术指标
            data = calculate_technical_indicators(raw_data)
            
            # 删除缺失值
            data = data.dropna()
            
            if len(data) < 100:
                logger.error(f"处理后的{symbol}数据不足")
                return None
        
        # 确保目录存在
        os.makedirs(model_dir, exist_ok=True)
        
        # 训练代理
        agent, history = train_agent(
            data=data,
            episodes=episodes,
            model_dir=model_dir,
            initial_balance=10000,
            window_size=10
        )
        
        # 最终模型路径
        final_model_path = os.path.join(model_dir, f"{symbol}_rl_model.h5")
        
        # 保存最终模型
        agent.save(final_model_path)
        
        logger.info(f"{symbol}的强化学习模型已保存到 {final_model_path}")
        return final_model_path
        
    except Exception as e:
        logger.error(f"训练{symbol}的强化学习模型时出错: {str(e)}")
        return None

def rl_decision(symbol, data, rl_model_path=None):
    """
    使用强化学习模型做出交易决策
    
    参数:
        symbol (str): 股票代码
        data (DataFrame): 股票数据
        rl_model_path (str, optional): 模型路径
        
    返回:
        int: 决策（0=卖出, 1=持有, 2=买入）
    """
    try:
        # 如果没有提供模型路径，构建默认路径
        if rl_model_path is None:
            model_dir = os.getenv('MODEL_DIR', 'models')
            rl_model_path = os.path.join(model_dir, f"{symbol}_rl_model.h5")
        
        # 检查模型是否存在
        if not os.path.exists(rl_model_path):
            logger.warning(f"{symbol}的强化学习模型不存在: {rl_model_path}")
            return 1  # 默认持有
        
        # 确保数据有足够的行
        if data is None or len(data) < 20:
            logger.warning(f"{symbol}的数据不足")
            return 1  # 默认持有
        
        # 创建临时环境用于决策
        env = TradingEnvironment(data)
        
        # 获取当前状态
        state, _ = env.reset()
        env.current_step = len(data) - 1  # 设置为最新数据点
        state = env._get_observation()
        
        # 预测动作
        action = predict_action(rl_model_path, state)
        
        # 记录决策
        action_map = {0: "卖出", 1: "持有", 2: "买入"}
        logger.info(f"强化学习模型为{symbol}做出决策: {action_map.get(action, str(action))}")
        
        return action
        
    except Exception as e:
        logger.error(f"为{symbol}做出强化学习决策时出错: {str(e)}")
        return 1  # 默认持有