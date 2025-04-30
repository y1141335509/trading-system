# src/utils/logger.py

import os
import logging
import logging.handlers
from datetime import datetime
import sys

def setup_logger(name=None, log_level=None, log_file=None, console=True):
    """
    设置日志记录器
    
    参数:
        name (str, optional): 日志记录器名称，如果为None则为根记录器
        log_level (str, optional): 日志级别，如果为None则使用环境变量或默认值
        log_file (str, optional): 日志文件路径，如果为None则使用环境变量或默认值
        console (bool): 是否输出到控制台
        
    返回:
        logging.Logger: 配置好的日志记录器
    """
    # 确定日志级别
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    level = getattr(logging, log_level)
    
    # 获取或创建记录器
    logger = logging.getLogger(name)
    
    # 如果记录器已经配置过，直接返回
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 添加控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 添加文件处理器（如果指定了日志文件）
    if log_file is None:
        # 默认日志文件路径
        logs_dir = os.getenv('LOGS_DIR', os.path.join(os.getenv('DATA_DIR', 'data'), 'logs'))
        os.makedirs(logs_dir, exist_ok=True)
        
        # 使用日期作为文件名
        date_str = datetime.now().strftime('%Y%m%d')
        log_file = os.path.join(logs_dir, f"{date_str}.log")
    
    # 确保日志目录存在
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    
    # 创建按大小轮转的文件处理器
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def get_logger(name=None):
    """
    获取已配置的记录器，如果不存在则创建新的
    
    参数:
        name (str, optional): 记录器名称
        
    返回:
        logging.Logger: 日志记录器
    """
    logger = logging.getLogger(name)
    
    # 如果记录器没有处理器，则设置
    if not logger.handlers:
        return setup_logger(name)
    
    return logger

def log_to_file(message, log_level='INFO', file_path=None):
    """
    将消息直接记录到特定文件
    
    参数:
        message (str): 要记录的消息
        log_level (str): 日志级别
        file_path (str, optional): 日志文件路径，如果为None则使用默认路径
        
    返回:
        bool: 是否成功记录
    """
    try:
        # 设置日志文件路径
        if file_path is None:
            logs_dir = os.getenv('LOGS_DIR', os.path.join(os.getenv('DATA_DIR', 'data'), 'logs'))
            os.makedirs(logs_dir, exist_ok=True)
            
            file_path = os.path.join(logs_dir, 'custom.log')
        
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # 创建特定文件的记录器
        file_logger = logging.getLogger(f"file_logger_{os.path.basename(file_path)}")
        file_logger.setLevel(logging.INFO)
        
        # 如果已有处理器，则移除
        if file_logger.handlers:
            file_logger.handlers = []
        
        # 添加文件处理器
        handler = logging.FileHandler(file_path, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        file_logger.addHandler(handler)
        
        # 记录消息
        level = getattr(logging, log_level.upper())
        file_logger.log(level, message)
        
        # 关闭处理器
        handler.close()
        file_logger.removeHandler(handler)
        
        return True
        
    except Exception as e:
        print(f"记录到文件失败: {str(e)}")
        return False

def log_exception(e, context=''):
    """
    记录异常信息
    
    参数:
        e (Exception): 异常对象
        context (str): 上下文信息
        
    返回:
        None
    """
    logger = get_logger()
    
    if context:
        logger.error(f"{context} - 异常: {str(e)}")
    else:
        logger.error(f"异常: {str(e)}")
        
    # 记录详细的异常信息
    import traceback
    logger.debug(f"异常详细信息:\n{traceback.format_exc()}")