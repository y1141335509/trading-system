# src/utils/config.py

import os
import json
import yaml
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 获取日志实例
logger = logging.getLogger(__name__)

def get_env_variable(name, default=None):
    """
    获取环境变量
    
    参数:
        name (str): 环境变量名
        default: 默认值，如果环境变量不存在
        
    返回:
        环境变量值
    """
    value = os.getenv(name, default)
    
    # 转换布尔值字符串
    if isinstance(value, str):
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
            
    return value

def load_config(config_path=None):
    """
    加载配置文件
    
    参数:
        config_path (str, optional): 配置文件路径，如果为None则使用默认路径
        
    返回:
        dict: 配置字典
    """
    # 如果没有指定配置文件路径，使用默认路径
    if config_path is None:
        config_path = get_env_variable('CONFIG_PATH', 'config.yaml')
    
    # 判断文件是否存在
    if not os.path.exists(config_path):
        logger.warning(f"配置文件不存在: {config_path}")
        return {}
    
    try:
        # 根据文件扩展名选择解析方式
        _, ext = os.path.splitext(config_path)
        
        with open(config_path, 'r') as file:
            if ext.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(file)
            elif ext.lower() == '.json':
                config = json.load(file)
            else:
                logger.error(f"不支持的配置文件格式: {ext}")
                return {}
        
        logger.info(f"已加载配置文件: {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"加载配置文件时出错: {str(e)}")
        return {}

def save_config(config, config_path=None):
    """
    保存配置到文件
    
    参数:
        config (dict): 配置字典
        config_path (str, optional): 配置文件路径，如果为None则使用默认路径
        
    返回:
        bool: 是否保存成功
    """
    # 如果没有指定配置文件路径，使用默认路径
    if config_path is None:
        config_path = get_env_variable('CONFIG_PATH', 'config.yaml')
    
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        # 根据文件扩展名选择序列化方式
        _, ext = os.path.splitext(config_path)
        
        with open(config_path, 'w') as file:
            if ext.lower() in ['.yaml', '.yml']:
                yaml.dump(config, file, default_flow_style=False)
            elif ext.lower() == '.json':
                json.dump(config, file, indent=4)
            else:
                logger.error(f"不支持的配置文件格式: {ext}")
                return False
        
        logger.info(f"已保存配置到文件: {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"保存配置文件时出错: {str(e)}")
        return False

def get_alpaca_credentials():
    """
    获取Alpaca API凭证
    
    返回:
        tuple: (API_KEY, API_SECRET, BASE_URL)
    """
    API_KEY = get_env_variable('ALPACA_PPAPER_API_KEY')
    API_SECRET = get_env_variable('ALPACA_PAPER_API_SECRET')
    
    # 确定是生产环境还是测试环境
    is_paper = get_env_variable('ALPACA_PAPER', 'true').lower() == 'true'
    
    if is_paper:
        BASE_URL = get_env_variable('ALPACA_PAPER_URL', 'https://paper-api.alpaca.markets')
    else:
        BASE_URL = get_env_variable('ALPACA_LIVE_URL', 'https://api.alpaca.markets')
    
    return API_KEY, API_SECRET, BASE_URL

def get_trading_params():
    """
    获取交易参数
    
    返回:
        dict: 交易参数字典
    """
    return {
        'risk_percent': float(get_env_variable('RISK_PERCENT', '0.02')),
        'max_positions': int(get_env_variable('MAX_POSITIONS', '5')),
        'stop_loss_percent': float(get_env_variable('STOP_LOSS_PERCENT', '0.05')),
        'trailing_stop_percent': float(get_env_variable('TRAILING_STOP_PERCENT', '0.03')),
        'max_drawdown_limit': float(get_env_variable('MAX_DRAWDOWN_LIMIT', '-0.15')),
        'risk_allocation': {
            'low': float(get_env_variable('RISK_ALLOCATION_LOW', '0.5')),
            'medium': float(get_env_variable('RISK_ALLOCATION_MEDIUM', '0.3')),
            'high': float(get_env_variable('RISK_ALLOCATION_HIGH', '0.2'))
        }
    }

def get_data_paths():
    """
    获取数据相关路径
    
    返回:
        dict: 路径字典
    """
    # 基础目录
    data_dir = get_env_variable('DATA_DIR', 'data')
    model_dir = get_env_variable('MODEL_DIR', 'models')
    
    return {
        'data_dir': data_dir,
        'model_dir': model_dir,
        'logs_dir': os.path.join(data_dir, 'logs'),
        'reports_dir': os.path.join(data_dir, 'reports'),
        'backtest_dir': os.path.join(data_dir, 'backtest')
    }

def get_notification_settings():
    """
    获取通知设置
    
    返回:
        dict: 通知设置字典
    """
    return {
        'enabled': get_env_variable('ENABLE_NOTIFICATIONS', 'true').lower() == 'true',
        'method': get_env_variable('NOTIFICATION_METHOD', 'print'),
        'webhook_url': get_env_variable('NOTIFICATION_WEBHOOK_URL', ''),
        'email': {
            'enabled': get_env_variable('EMAIL_NOTIFICATIONS', 'false').lower() == 'true',
            'smtp_server': get_env_variable('SMTP_SERVER', ''),
            'smtp_port': int(get_env_variable('SMTP_PORT', '587')),
            'username': get_env_variable('SMTP_USERNAME', ''),
            'password': get_env_variable('SMTP_PASSWORD', ''),
            'recipients': get_env_variable('EMAIL_RECIPIENTS', '').split(',')
        }
    }