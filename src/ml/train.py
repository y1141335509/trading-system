# src/ml/train.py

import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 设置日志
logger = logging.getLogger(__name__)

def train_model(X, y, model_type='random_forest', params=None, scale=True):
    """
    训练机器学习模型
    
    参数:
        X (DataFrame): 特征数据
        y (Series): 目标变量
        model_type (str): 模型类型，目前支持 'random_forest'
        params (dict): 模型参数
        scale (bool): 是否标准化特征
        
    返回:
        tuple: (模型, 缩放器)
    """
    try:
        if X is None or y is None or len(X) < 10:
            logger.error("训练数据不足或无效")
            return None, None
            
        # 默认参数
        if params is None:
            params = {}
            
        # 标准化特征（如果需要）
        scaler = None
        X_train = X.copy()
        
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            logger.info("特征已标准化")
        
        # 根据模型类型创建模型
        if model_type == 'random_forest':
            # 默认随机森林参数
            rf_params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
            
            # 更新参数
            rf_params.update(params)
            
            # 创建和训练模型
            model = RandomForestClassifier(**rf_params)
            model.fit(X_train, y)
            
            logger.info(f"随机森林模型训练完成，树的数量: {rf_params['n_estimators']}")
            
        else:
            logger.error(f"不支持的模型类型: {model_type}")
            return None, None
            
        return model, scaler
        
    except Exception as e:
        logger.error(f"模型训练失败: {str(e)}")
        return None, None

def save_model(model, scaler=None, symbol='general', model_dir='models'):
    """
    保存训练好的模型和缩放器
    
    参数:
        model: 训练好的模型
        scaler: 特征缩放器（可选）
        symbol (str): 股票代码或模型标识符
        model_dir (str): 模型保存目录
        
    返回:
        bool: 是否保存成功
    """
    try:
        if model is None:
            logger.error("无效的模型，无法保存")
            return False
            
        # 创建模型目录（如果不存在）
        os.makedirs(model_dir, exist_ok=True)
        
        # 生成文件名
        model_path = os.path.join(model_dir, f"{symbol}_ml_model.joblib")
        
        # 保存模型
        joblib.dump(model, model_path)
        logger.info(f"模型已保存到: {model_path}")
        
        # 如果有缩放器，也保存它
        if scaler is not None:
            scaler_path = os.path.join(model_dir, f"{symbol}_scaler.joblib")
            joblib.dump(scaler, scaler_path)
            logger.info(f"缩放器已保存到: {scaler_path}")
            
        return True
        
    except Exception as e:
        logger.error(f"保存模型失败: {str(e)}")
        return False

def load_model(symbol='general', model_dir='models', force_retrain=False):
    """
    加载已保存的模型和缩放器
    
    参数:
        symbol (str): 股票代码或模型标识符
        model_dir (str): 模型保存目录
        force_retrain (bool): 是否强制重新训练
        
    返回:
        tuple: (模型, 缩放器)
    """
    if force_retrain:
        logger.info(f"强制重新训练模式，跳过加载{symbol}的模型")
        return None, None
        
    try:
        # 生成文件路径
        model_path = os.path.join(model_dir, f"{symbol}_ml_model.joblib")
        scaler_path = os.path.join(model_dir, f"{symbol}_scaler.joblib")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            logger.warning(f"未找到{symbol}的模型文件，需要训练新模型")
            return None, None
            
        # 加载模型
        model = joblib.load(model_path)
        logger.info(f"已加载{symbol}的模型")
        
        # 尝试加载缩放器（如果存在）
        scaler = None
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info(f"已加载{symbol}的缩放器")
            
        return model, scaler
        
    except Exception as e:
        logger.error(f"加载{symbol}的模型失败: {str(e)}")
        return None, None

def train_test_split_time(X, y, test_size=0.2):
    """
    按时间顺序分割训练集和测试集
    
    参数:
        X (DataFrame): 特征数据
        y (Series): 目标变量
        test_size (float): 测试集比例
        
    返回:
        tuple: (X_train, X_test, y_train, y_test)
    """
    try:
        if X is None or y is None:
            return None, None, None, None
            
        # 确保索引匹配
        y = y.loc[X.index]
        
        # 按时间顺序分割（假设数据已按时间排序）
        n = len(X)
        train_size = int(n * (1 - test_size))
        
        X_train = X.iloc[:train_size]
        X_test = X.iloc[train_size:]
        y_train = y.iloc[:train_size]
        y_test = y.iloc[train_size:]
        
        logger.info(f"数据已按时间分割: 训练集 {len(X_train)} 样本, 测试集 {len(X_test)} 样本")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"分割训练集和测试集失败: {str(e)}")
        return None, None, None, None