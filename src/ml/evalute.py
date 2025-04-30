# src/ml/evaluate.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import logging

# 设置日志
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test, scaler=None):
    """
    评估模型性能
    
    参数:
        model: 训练好的模型
        X_test (DataFrame): 测试集特征
        y_test (Series): 测试集标签
        scaler: 特征缩放器（可选）
        
    返回:
        dict: 性能指标
    """
    try:
        if model is None or X_test is None or y_test is None:
            logger.error("模型或测试数据无效，无法评估")
            return None
            
        # 准备测试数据
        X = X_test.copy()
        if scaler is not None:
            X = scaler.transform(X)
            
        # 预测
        y_pred = model.predict(X)
        
        # 计算性能指标
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        
        # 处理只有一个类别的情况
        if len(np.unique(y_test)) < 2:
            logger.warning("测试集只有一个类别，某些指标无法计算")
            metrics['precision'] = metrics['recall'] = metrics['f1'] = np.nan
        else:
            metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
            metrics['f1'] = f1_score(y_test, y_pred, zero_division=0)
        
        # 如果模型支持概率预测，计算AUC
        if hasattr(model, 'predict_proba'):
            try:
                y_prob = model.predict_proba(X)[:, 1]
                metrics['auc'] = roc_auc_score(y_test, y_prob)
            except:
                metrics['auc'] = np.nan
                
        # 计算混淆矩阵
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        
        logger.info(f"模型评估完成: 准确率={metrics['accuracy']:.4f}")
        return metrics
        
    except Exception as e:
        logger.error(f"评估模型失败: {str(e)}")
        return None

def plot_feature_importance(model, feature_names=None, top_n=10, figsize=(10, 6)):
    """
    绘制特征重要性图
    
    参数:
        model: 训练好的模型
        feature_names (list): 特征名列表
        top_n (int): 显示前N个重要特征
        figsize (tuple): 图形大小
        
    返回:
        tuple: (fig, ax) matplotlib图形对象
    """
    try:
        if model is None or not hasattr(model, 'feature_importances_'):
            logger.error("模型无效或不支持特征重要性")
            return None, None
            
        # 获取特征重要性
        importances = model.feature_importances_
        
        # 如果没有提供特征名，使用索引
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importances))]
            
        # 创建数据框并排序
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # 选择前N个特征
        if top_n > 0 and top_n < len(feature_importance):
            feature_importance = feature_importance.head(top_n)
            
        # 绘制图形
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
        ax.set_title('Feature Importance')
        
        return fig, ax
        
    except Exception as e:
        logger.error(f"绘制特征重要性失败: {str(e)}")
        return None, None

def backtest_strategy(predictions, price_data, transaction_cost=0.001, initial_capital=10000):
    """
    回测交易策略
    
    参数:
        predictions (Series): 模型预测结果
        price_data (DataFrame): 价格数据
        transaction_cost (float): 交易成本（占交易金额的比例）
        initial_capital (float): 初始资金
        
    返回:
        DataFrame: 回测结果
    """
    try:
        if predictions is None or price_data is None:
            logger.error("预测结果或价格数据无效，无法回测")
            return None
            
        # 确保预测和价格索引匹配
        if not isinstance(predictions, pd.Series):
            predictions = pd.Series(predictions, index=price_data.index)
        
        # 将预测与价格数据对齐
        aligned_data = pd.concat([price_data['close'], predictions], axis=1)
        aligned_data.columns = ['price', 'position']
        
        # 计算收益率
        aligned_data['returns'] = aligned_data['price'].pct_change()
        
        # 计算策略收益率
        aligned_data['strategy_returns'] = aligned_data['position'].shift(1) * aligned_data['returns']
        
        # 计算交易信号变化
        aligned_data['trade'] = aligned_data['position'].diff().fillna(0) != 0
        
        # 计算交易成本
        aligned_data['cost'] = np.where(aligned_data['trade'], transaction_cost, 0)
        
        # 计算净策略收益
        aligned_data['net_strategy_returns'] = aligned_data['strategy_returns'] - aligned_data['cost']
        
        # 计算累积收益
        aligned_data['cumulative_returns'] = (1 + aligned_data['returns']).cumprod()
        aligned_data['cumulative_strategy_returns'] = (1 + aligned_data['net_strategy_returns']).cumprod()
        
        # 计算回撤
        aligned_data['peak'] = aligned_data['cumulative_strategy_returns'].cummax()
        aligned_data['drawdown'] = (aligned_data['cumulative_strategy_returns'] - aligned_data['peak']) / aligned_data['peak']
        
        # 计算性能指标
        total_return = aligned_data['cumulative_strategy_returns'].iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(aligned_data)) - 1
        max_drawdown = aligned_data['drawdown'].min()
        sharpe_ratio = np.sqrt(252) * aligned_data['net_strategy_returns'].mean() / aligned_data['net_strategy_returns'].std()
        
        # 创建性能报告
        performance = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': aligned_data['trade'].sum()
        }
        
        logger.info(f"回测完成: 总回报={total_return:.4f}, 年化回报={annualized_return:.4f}")
        
        return aligned_data, performance
        
    except Exception as e:
        logger.error(f"回测失败: {str(e)}")
        return None, None