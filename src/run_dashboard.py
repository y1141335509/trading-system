#!/usr/bin/env python3
# run_dashboard.py - 启动交易系统仪表盘

import os
import sys
import logging
import argparse
from datetime import datetime

# 确保src包在路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dashboard.app import run_dashboard
from src.utils.logger import setup_logger
from src.utils.config import get_data_paths

def main():
    """主函数，解析命令行参数并启动仪表盘"""
    parser = argparse.ArgumentParser(description='启动交易系统仪表盘')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='主机地址(默认: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='端口号(默认: 5000)')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logger()
    logger = logging.getLogger('dashboard')
    
    # 创建数据目录
    data_paths = get_data_paths()
    os.makedirs(data_paths['data_dir'], exist_ok=True)
    os.makedirs(data_paths['reports_dir'], exist_ok=True)
    os.makedirs(data_paths['logs_dir'], exist_ok=True)
    
    # 打印启动信息
    logger.info("="*50)
    logger.info(f"启动交易系统仪表盘 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"主机: {args.host}, 端口: {args.port}, 调试模式: {'开启' if args.debug else '关闭'}")
    logger.info("="*50)
    
    # 启动仪表盘
    run_dashboard(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()