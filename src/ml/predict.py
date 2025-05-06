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

# 最终版本的predict_proba函数 - 完全解决特征名称警告

def predict_proba(features, model, scaler):
    """
    使用机器学习模型预测上涨概率 - 最终版
    
    参数:
        features (DataFrame): 包含特征的数据框
        model: 训练好的模型
        scaler: 特征缩放器
        
    返回:
        float: 上涨的概率（0-1之间）
    """
    try:
        if model is None:
            logger.error("模型无效，无法预测")
            return 0.5
        
        # 获取模型期望的特征数量
        expected_feature_count = 12  # 默认值
        
        if hasattr(model, 'n_features_in_'):
            expected_feature_count = model.n_features_in_
        elif hasattr(model, 'feature_importances_') and isinstance(model.feature_importances_, np.ndarray):
            expected_feature_count = len(model.feature_importances_)
            
        logger.info(f"模型期望的特征数量: {expected_feature_count}")
        
        # 尝试获取模型训练时使用的特征名称
        original_feature_names = None
        if hasattr(model, 'feature_names_in_'):
            original_feature_names = model.feature_names_in_
        
        # 1. 确定当前可用的特征
        available_features = features.columns.tolist()
        logger.debug(f"当前可用特征 ({len(available_features)}): {available_features}")
        
        # 2. 创建通用特征列表 - 这个顺序很重要
        standard_features = [
            'open', 'high', 'low', 'close',     # 基本价格数据
            'rsi', 'macd', 'signal', 'hist',    # 技术指标组1
            'ma20', 'atr', 'trend_5d', 'trend_10d',  # 技术指标组2
            'ma10', 'ma50', 'ma10_ratio', 'ma50_ratio',  # 均线组
            'volatility', 'trend_20d', 'upper_band', 'lower_band'  # 额外指标
        ]
        
        # 3. 构建特征映射
        feature_mapping = {
            'macd_signal': 'signal',
            'macd_hist': 'hist',
            'bb_middle': 'ma20',
            'bb_upper': 'upper_band',
            'bb_lower': 'lower_band'
        }
        
        # 4. 根据模型期望的特征数量，创建特征输入
        X_input = pd.DataFrame(index=features.index)
        
        # 如果我们有原始特征名，使用它们；否则使用标准名称
        if original_feature_names is not None and len(original_feature_names) == expected_feature_count:
            target_features = original_feature_names
        else:
            # 从标准特征列表中选择前N个(N=expected_feature_count)
            target_features = standard_features[:expected_feature_count]
        
        # 为每个目标特征找到或创建对应的数据
        for i, feature_name in enumerate(target_features):
            column_name = str(feature_name) if original_feature_names is not None else f"feature_{i}"
            
            # 检查特征是否直接可用
            if feature_name in features.columns:
                X_input[column_name] = features[feature_name]
                logger.debug(f"使用原始特征: {feature_name}")
                continue
                
            # 检查是否有别名可用
            alias_found = False
            for alias, original in feature_mapping.items():
                if original == feature_name and alias in features.columns:
                    X_input[column_name] = features[alias]
                    logger.debug(f"使用别名特征: {alias} 替代 {feature_name}")
                    alias_found = True
                    break
                    
            if alias_found:
                continue
                
            # 特殊合成特征
            if feature_name == 'ma20' and 'close' in features.columns:
                X_input[column_name] = features['close']
                logger.debug(f"使用close值替代 {feature_name}")
            elif feature_name == 'signal' and 'macd' in features.columns:
                X_input[column_name] = features['macd']
                logger.debug(f"使用macd值替代 {feature_name}")
            elif feature_name == 'hist' and 'macd' in features.columns:
                X_input[column_name] = features['macd'] * 0.2  # 简单估计
                logger.debug(f"使用macd值*0.2估计 {feature_name}")
            elif feature_name in ['ma10', 'ma50'] and 'close' in features.columns:
                X_input[column_name] = features['close']
                logger.debug(f"使用close值替代 {feature_name}")
            elif feature_name in ['ma10_ratio', 'ma50_ratio'] and 'close' in features.columns:
                X_input[column_name] = 1.0  # 默认比率为1
                logger.debug(f"使用1.0替代比率 {feature_name}")
            elif feature_name in ['trend_5d', 'trend_10d', 'trend_20d'] and 'close' in features.columns:
                X_input[column_name] = 0.0  # 假设趋势平稳
                logger.debug(f"使用0.0替代趋势 {feature_name}")
            elif feature_name == 'volatility' and 'atr' in features.columns:
                X_input[column_name] = features['atr'] / features['close']
                logger.debug(f"使用atr/close估算 {feature_name}")
            elif feature_name in ['upper_band', 'lower_band'] and 'close' in features.columns and 'atr' in features.columns:
                factor = 2.0 if feature_name == 'upper_band' else -2.0
                X_input[column_name] = features['close'] + factor * features['atr']
                logger.debug(f"使用close+factor*atr估算 {feature_name}")
            else:
                # 使用0填充缺失特征
                X_input[column_name] = 0.0
                logger.debug(f"缺失特征 {feature_name}，使用0填充")
        
        # 确保特征数量正确
        if len(X_input.columns) != expected_feature_count:
            logger.warning(f"特征数量不匹配: 需要{expected_feature_count}，实际{len(X_input.columns)}")
            # 填充或截断特征
            if len(X_input.columns) < expected_feature_count:
                for i in range(len(X_input.columns), expected_feature_count):
                    X_input[f"extra_feature_{i}"] = 0.0
            
            if len(X_input.columns) > expected_feature_count:
                X_input = X_input.iloc[:, :expected_feature_count]
        
        # 调试信息
        logger.debug(f"最终特征数量: {len(X_input.columns)}")
        
        # 标准化特征 - 处理警告
        try:
            if scaler is not None:
                # 直接操作numpy数组，避免特征名称警告
                X_values = X_input.values
                X_scaled = scaler.transform(X_values)
            else:
                X_scaled = X_input.values
        except Exception as e:
            logger.error(f"特征缩放失败: {str(e)}")
            # 回退到不缩放
            X_scaled = X_input.values
        
        # 进行预测
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_scaled)[0][1]
            else:
                pred = model.predict(X_scaled)[0]
                proba = float(pred)
                
            logger.info(f"预测上涨概率: {proba:.3f}")
            return proba
            
        except Exception as e:
            logger.error(f"模型预测失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return 0.5
        
    except Exception as e:
        logger.error(f"预测过程出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 0.5
    
def prepare_prediction_data(data, scaler=None, feature_names=None):
    """
    准备预测用的数据 - 修复版
    
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
        
        # 如果未提供特征名列表，使用所有数值列
        if feature_names is None:
            # 排除日期和非数值列
            excluded_cols = ['date', 'timestamp', 'time', 'volume']
            feature_names = [col for col in data.columns if col not in excluded_cols and pd.api.types.is_numeric_dtype(data[col])]
        
        # 检查所有特征是否存在，仅使用可用特征
        available_features = [f for f in feature_names if f in data.columns]
        missing_features = [f for f in feature_names if f not in data.columns]
        
        if missing_features:
            logger.warning(f"缺少以下特征: {missing_features}")
        
        if not available_features:
            logger.error("无可用特征进行预测")
            return None
        
        # 仅使用可用特征
        X = data[available_features].copy()
        
        # 处理缺失值
        X = X.fillna(0)
        
        # 应用缩放（如果提供了缩放器）
        if scaler is not None:
            X = scaler.transform(X)
        
        return X
        
    except Exception as e:
        logger.error(f"准备预测数据失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
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

