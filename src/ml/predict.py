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
    """
    try:
        if model is None:
            logger.error("模型无效，无法预测")
            return None
            
        # 首先计算所有必要的技术指标
        data = calculate_missing_indicators(data.copy())
        
        # 准备特征
        X = prepare_prediction_data(data, scaler, feature_names)
        if X is None:
            return None
            
        # 进行预测
        if hasattr(model, 'predict_proba'):
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

def predict_proba(features, model, scaler):
    """
    预测价格变动概率，自动计算缺失的技术指标
    """
    try:
        # 确保数据为DataFrame类型
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)
            
        # 计算缺失的技术指标
        features = calculate_missing_indicators(features)
        
        # 定义预期的特征列
        expected_features = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'atr',
            'ma10', 'ma50', 'trend_5d', 'trend_10d', 'trend_20d',
            'ma10_ratio', 'ma50_ratio', 'volatility'
        ]
        
        # 确保所有特征都存在
        missing_features = [f for f in expected_features if f not in features.columns]
        if missing_features:
            logger.error(f"缺少特征: {missing_features}")
            return 0.5  # 返回中性预测
        
        # 选择特征并处理缺失值
        X = features[expected_features].copy()
        X = X.fillna(method='ffill').fillna(method='bfill')
        
        # 标准化特征
        if scaler is not None:
            X = scaler.transform(X)
        
        # 预测概率
        try:
            proba = model.predict_proba(X)
            logger.info(f"预测概率: {proba[0][1]:.3f}")
            return proba[0][1]  # 返回上涨概率
        except Exception as e:
            logger.error(f"模型预测失败: {str(e)}")
            return 0.5
            
    except Exception as e:
        logger.error(f"预测过程出错: {str(e)}")
        return 0.5

def prepare_prediction_data(data, scaler=None, feature_names=None):
    """
    准备预测用的数据
    """
    try:
        if data is None or len(data) == 0:
            logger.error("预测数据为空")
            return None
        
        # 确保所有必要的特征都存在
        required_features = [
            'open', 'high', 'low', 'close', 'volume',
            'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower',
            'rsi'
        ]
        
        # 如果没有提供特征名列表，使用所有必要特征
        if feature_names is None:
            feature_names = required_features
        
        # 检查是否所有必要特征都存在
        missing_features = [f for f in feature_names if f not in data.columns]
        if missing_features:
            raise ValueError(f"缺少必要特征: {missing_features}")
        
        # 选择特征
        X = data[feature_names].copy()
        
        # 应用缩放
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

def calculate_missing_indicators(df):
    """
    计算缺失的技术指标
    """
    try:
        # 确保DataFrame包含必要的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"缺少基础数据列: {missing}")
        
        # 计算MACD
        if not all(col in df.columns for col in ['macd', 'macd_signal', 'macd_hist']):
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 计算布林带
        if not all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            rolling_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (rolling_std * 2)
            df['bb_lower'] = df['bb_middle'] - (rolling_std * 2)
        
        # 计算RSI（如果需要）
        if 'rsi' not in df.columns:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
        
        # 填充NaN值
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
        
    except Exception as e:
        logger.error(f"计算技术指标时出错: {str(e)}")
        raise

