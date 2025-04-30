# src/ml/predict.py

import numpy as np
import pandas as pd
import logging
from .train import load_model

# 设置日志
logger = logging.getLogger(__name__)

def predict(data, model, scaler=None, threshold=0.5, feature_names=None):
    """
    使用模型进行预测
    
    参数:
        data (DataFrame): 要预测的数据
        model: 训练好的模型
        scaler: 特征缩放器（可选）
        threshold (float): 决策阈值
        feature_names (list): 特征名列表，如果为None则使用data的所有列
        
    返回:
        array: 预测结果
    """
    try:
        if model is None:
            logger.error("模型无效，无法预测")
            return None
            
        # 准备特征
        X = prepare_prediction_data(data, scaler, feature_names)
        if X is None:
            return None
            
        # 进行预测
        if hasattr(model, 'predict_proba'):
            # 如果模型支持概率预测，使用阈值
            probas = model.predict_proba(X)
            if probas.shape[1] >= 2:
                preds = (probas[:, 1] >= threshold).astype(int)
            else:
                preds = model.predict(X)
        else:
            preds = model.predict(X)
            
        return preds
        
    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        return None

def predict_proba(data, model, scaler=None, feature_names=None):
    """
    使用模型进行概率预测
    
    参数:
        data (DataFrame): 要预测的数据
        model: 训练好的模型
        scaler: 特征缩放器（可选）
        feature_names (list): 特征名列表，如果为None则使用data的所有列
        
    返回:
        array: 预测的概率
    """
    try:
        if model is None:
            logger.error("模型无效，无法预测")
            return None
            
        # 确保模型支持概率预测
        if not hasattr(model, 'predict_proba'):
            logger.error("模型不支持概率预测")
            return None
            
        # 准备特征
        X = prepare_prediction_data(data, scaler, feature_names)
        if X is None:
            return None
            
        # 进行概率预测
        probas = model.predict_proba(X)
        
        # 返回正类的概率
        if probas.shape[1] >= 2:
            return probas[:, 1]
        else:
            return probas[:, 0]
            
    except Exception as e:
        logger.error(f"概率预测失败: {str(e)}")
        return None

def prepare_prediction_data(data, scaler=None, feature_names=None):
    """
    准备预测用的数据
    
    参数:
        data (DataFrame): 原始数据
        scaler: 特征缩放器
        feature_names (list): 特征名列表
        
    返回:
        array: 准备好的特征数据
    """
    try:
        if data is None or len(data) == 0:
            logger.error("预测数据为空")
            return None
            
        # 如果提供了特征名列表，只使用这些特征
        if feature_names is not None:
            # 检查所有特征是否存在
            missing_features = [f for f in feature_names if f not in data.columns]
            if missing_features:
                logger.warning(f"缺少以下特征: {missing_features}")
                
            # 只使用可用的特征
            available_features = [f for f in feature_names if f in data.columns]
            X = data[available_features].copy()
        else:
            X = data.copy()
            
        # 应用缩放（如果提供了缩放器）
        if scaler is not None:
            X = scaler.transform(X)
            
        return X
        
    except Exception as e:
        logger.error(f"准备预测数据失败: {str(e)}")
        return None

def predict_with_symbol(symbol, data, model_dir='models', feature_names=None):
    """
    使用为特定股票训练的模型进行预测
    
    参数:
        symbol (str): 股票代码
        data (DataFrame): 要预测的数据
        model_dir (str): 模型保存目录
        feature_names (list): 特征名列表
        
    返回:
        dict: 包含预测结果和概率的字典
    """
    try:
        # 加载模型和缩放器
        model, scaler = load_model(symbol, model_dir)
        
        if model is None:
            logger.error(f"未找到{symbol}的模型，无法预测")
            return None
            
        # 进行预测
        predictions = predict(data, model, scaler, feature_names=feature_names)
        
        # 如果模型支持概率预测，也获取概率
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = predict_proba(data, model, scaler, feature_names=feature_names)
            
        return {
            'predictions': predictions,
            'probabilities': probabilities
        }
        
    except Exception as e:
        logger.error(f"使用{symbol}的模型预测失败: {str(e)}")
        return None