# src/dashboard/app.py

import os
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
from datetime import datetime
import logging

from .api import (
    get_performance_data,
    get_positions_data, 
    get_trades_data,
    get_reports_list,
    get_system_status
)

from ..utils.config import get_data_paths

# 设置日志
logger = logging.getLogger(__name__)

def create_app():
    """创建Flask应用"""
    # 确定模板和静态文件目录
    template_folder = os.path.join(os.path.dirname(__file__), 'templates')
    static_folder = os.path.join(os.path.dirname(__file__), 'static')
    
    # 确保目录存在
    os.makedirs(template_folder, exist_ok=True)
    os.makedirs(static_folder, exist_ok=True)
    
    # 创建Flask应用
    app = Flask(__name__, 
                template_folder=template_folder,
                static_folder=static_folder)
    
    # 配置
    app.config['JSON_SORT_KEYS'] = False
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
    
    # 注册路由
    @app.route('/')
    def home():
        """仪表盘首页"""
        return render_template('index.html')
    
    @app.route('/api/performance')
    def api_performance():
        """性能数据API"""
        days = request.args.get('days', default=30, type=int)
        data = get_performance_data(days=days)
        return jsonify(data)
    
    @app.route('/api/positions')
    def api_positions():
        """持仓数据API"""
        data = get_positions_data()
        return jsonify(data)
    
    @app.route('/api/trades')
    def api_trades():
        """交易数据API"""
        days = request.args.get('days', default=7, type=int)
        data = get_trades_data(days=days)
        return jsonify(data)
    
    @app.route('/api/reports')
    def api_reports():
        """报告列表API"""
        data = get_reports_list()
        return jsonify(data)
    
    @app.route('/api/status')
    def api_status():
        """系统状态API"""
        data = get_system_status()
        return jsonify(data)
    
    @app.route('/report/<path:filename>')
    def get_report(filename):
        """获取报告文件"""
        data_paths = get_data_paths()
        report_dir = os.path.join(data_paths['data_dir'], 'reports')
        return send_from_directory(report_dir, filename)
    
    @app.route('/charts')
    def charts():
        """图表页面"""
        return render_template('charts.html')
    
    @app.route('/reports')
    def reports():
        """报告页面"""
        return render_template('reports.html')
    
    @app.route('/settings')
    def settings():
        """设置页面"""
        return render_template('settings.html')
    
    @app.route('/about')
    def about():
        """关于页面"""
        return render_template('about.html')
    
    # 记录路由
    logger.info(f"已注册路由: {[rule.rule for rule in app.url_map.iter_rules()]}")
    
    return app

def run_dashboard(host='0.0.0.0', port=5000, debug=False):
    """运行仪表盘应用"""
    app = create_app()
    logger.info(f"启动仪表盘应用，地址: http://{host}:{port}/")
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    # 运行应用
    run_dashboard(debug=True)