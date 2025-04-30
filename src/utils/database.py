# src/utils/database.py

import os
import mysql.connector
from mysql.connector import Error
import pandas as pd
import logging
from datetime import datetime

# 获取日志实例
logger = logging.getLogger(__name__)

def connect_to_mysql():
    """
    连接到MySQL数据库
    
    返回:
        connection: 数据库连接，失败则返回None
    """
    try:
        connection = mysql.connector.connect(
            host=os.getenv('MYSQL_HOST', 'localhost'),
            user=os.getenv('MYSQL_USER', 'root'),
            password=os.getenv('MYSQL_PASSWORD', ''),
            database=os.getenv('MYSQL_QUANT_DB', 'Quant')
        )
        
        if connection.is_connected():
            logger.info(f"已连接到MySQL数据库: {connection.database}")
            return connection
        
    except Error as e:
        logger.error(f"连接MySQL数据库时出错: {e}")
        
        # 尝试创建数据库
        try:
            connection = mysql.connector.connect(
                host=os.getenv('MYSQL_HOST', 'localhost'),
                user=os.getenv('MYSQL_USER', 'root'),
                password=os.getenv('MYSQL_PASSWORD', '')
            )
            
            if connection.is_connected():
                cursor = connection.cursor()
                
                # 创建数据库
                db_name = os.getenv('MYSQL_QUANT_DB', 'Quant')
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
                
                # 切换到新创建的数据库
                cursor.execute(f"USE {db_name}")
                
                logger.info(f"已创建并连接到MySQL数据库: {db_name}")
                
                # 关闭游标
                cursor.close()
                
                return connection
                
        except Error as e2:
            logger.error(f"创建MySQL数据库时出错: {e2}")
    
    return None

def create_tables_if_not_exist():
    """
    创建必要的数据库表（如果不存在）
    
    返回:
        bool: 是否成功创建
    """
    connection = connect_to_mysql()
    if connection is None:
        return False
    
    cursor = connection.cursor()
    
    try:
        # 创建每日账户表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_account_summary (
            date DATE PRIMARY KEY,
            portfolio_value DECIMAL(15,2),
            cash DECIMAL(15,2),
            equity DECIMAL(15,2),
            daily_pnl DECIMAL(15,2),
            daily_return_pct DECIMAL(8,4),
            spy_return_pct DECIMAL(8,4),
            relative_performance DECIMAL(8,4)
        )
        ''')
        
        # 创建持仓表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            date DATE,
            symbol VARCHAR(10),
            quantity DECIMAL(15,6),
            entry_price DECIMAL(15,2),
            current_price DECIMAL(15,2),
            cost_basis DECIMAL(15,2),
            market_value DECIMAL(15,2),
            unrealized_pl DECIMAL(15,2),
            unrealized_plpc DECIMAL(8,4),
            UNIQUE KEY unique_position (date, symbol)
        )
        ''')
        
        # 创建交易表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INT AUTO_INCREMENT PRIMARY KEY,
            date DATE,
            trade_time DATETIME,
            symbol VARCHAR(10),
            side VARCHAR(10),
            quantity DECIMAL(15,6),
            price DECIMAL(15,2),
            order_id VARCHAR(50)
        )
        ''')
        
        # 创建市场数据表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            date DATE,
            symbol VARCHAR(10),
            price DECIMAL(15,2),
            change_percent DECIMAL(8,4),
            UNIQUE KEY unique_market_data (date, symbol)
        )
        ''')
        
        # 创建模型评估表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_evaluations (
            id INT AUTO_INCREMENT PRIMARY KEY,
            date DATE,
            symbol VARCHAR(10),
            model_type VARCHAR(20),
            accuracy DECIMAL(8,4),
            precision_score DECIMAL(8,4),
            recall_score DECIMAL(8,4),
            f1_score DECIMAL(8,4),
            auc_score DECIMAL(8,4),
            strategy_return DECIMAL(8,4),
            market_return DECIMAL(8,4)
        )
        ''')
        
        connection.commit()
        logger.info("数据库表已创建")
        return True
        
    except Error as e:
        logger.error(f"创建数据库表时出错: {e}")
        return False
        
    finally:
        cursor.close()
        connection.close()

def save_to_mysql(daily_summary=None, positions_data=None, orders_data=None, market_data=None):
    """
    将数据保存到MySQL数据库
    
    参数:
        daily_summary (dict): 每日摘要数据
        positions_data (list): 持仓数据
        orders_data (list): 订单数据
        market_data (dict): 市场数据
        
    返回:
        bool: 是否保存成功
    """
    connection = connect_to_mysql()
    if connection is None:
        return False
    
    cursor = connection.cursor()
    today = datetime.now().strftime('%Y-%m-%d')  # 使用字符串格式的日期
    
    try:
        # 保存每日账户摘要
        if daily_summary:
            cursor.execute('''
            INSERT INTO daily_account_summary 
            (date, portfolio_value, cash, equity, daily_pnl, daily_return_pct, spy_return_pct, relative_performance)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            portfolio_value = VALUES(portfolio_value),
            cash = VALUES(cash),
            equity = VALUES(equity),
            daily_pnl = VALUES(daily_pnl),
            daily_return_pct = VALUES(daily_return_pct),
            spy_return_pct = VALUES(spy_return_pct),
            relative_performance = VALUES(relative_performance)
            ''', (
                today,
                daily_summary['portfolio_value'],
                daily_summary['cash'],
                daily_summary['equity'],
                daily_summary['daily_pnl'],
                daily_summary['daily_return'],
                daily_summary['spy_return'],
                daily_summary['relative_performance']
            ))
            
            logger.debug("已保存每日账户摘要")
        
        # 保存持仓数据
        if positions_data:
            for position in positions_data:
                cursor.execute('''
                INSERT INTO positions
                (date, symbol, quantity, entry_price, current_price, cost_basis, market_value, unrealized_pl, unrealized_plpc)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                quantity = VALUES(quantity),
                entry_price = VALUES(entry_price),
                current_price = VALUES(current_price),
                cost_basis = VALUES(cost_basis),
                market_value = VALUES(market_value),
                unrealized_pl = VALUES(unrealized_pl),
                unrealized_plpc = VALUES(unrealized_plpc)
                ''', (
                    today,
                    position['symbol'],
                    position['quantity'],
                    position['entry_price'],
                    position['current_price'],
                    position['cost_basis'],
                    position['market_value'],
                    position['unrealized_pl'],
                    position['unrealized_plpc']
                ))
            
            logger.debug(f"已保存 {len(positions_data)} 条持仓数据")
        
        # 保存订单数据
        if orders_data:
            for order in orders_data:
                # 确保时间是字符串格式
                filled_at = order['filled_at']
                if not isinstance(filled_at, str):
                    filled_at = str(filled_at)
                    
                cursor.execute('''
                INSERT INTO trades
                (date, trade_time, symbol, side, quantity, price, order_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ''', (
                    today,
                    filled_at,
                    order['symbol'],
                    order['side'],
                    order['quantity'],
                    order['filled_price'],
                    order.get('order_id', 'unknown')
                ))
            
            logger.debug(f"已保存 {len(orders_data)} 条订单数据")
        
        # 保存市场数据
        if market_data:
            for symbol, data in market_data.items():
                cursor.execute('''
                INSERT INTO market_data
                (date, symbol, price, change_percent)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                price = VALUES(price),
                change_percent = VALUES(change_percent)
                ''', (
                    today,
                    symbol,
                    data['price'],
                    data['change_percent']
                ))
            
            logger.debug(f"已保存 {len(market_data)} 条市场数据")
        
        connection.commit()
        logger.info("数据已成功保存到MySQL数据库")
        return True
        
    except Error as e:
        logger.error(f"保存数据到MySQL出错: {e}")
        connection.rollback()
        return False
        
    finally:
        cursor.close()
        connection.close()

def save_model_evaluation(symbol, model_type, metrics):
    """
    保存模型评估结果到数据库
    
    参数:
        symbol (str): 股票代码
        model_type (str): 模型类型
        metrics (dict): 评估指标
        
    返回:
        bool: 是否保存成功
    """
    connection = connect_to_mysql()
    if connection is None:
        return False
    
    cursor = connection.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    
    try:
        cursor.execute('''
        INSERT INTO model_evaluations
        (date, symbol, model_type, accuracy, precision_score, recall_score, f1_score, auc_score, strategy_return, market_return)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (
            today,
            symbol,
            model_type,
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1', 0),
            metrics.get('auc', 0),
            metrics.get('strategy_return', 0),
            metrics.get('market_return', 0)
        ))
        
        connection.commit()
        logger.info(f"已保存 {symbol} 的 {model_type} 模型评估结果")
        return True
        
    except Error as e:
        logger.error(f"保存模型评估结果时出错: {e}")
        connection.rollback()
        return False
        
    finally:
        cursor.close()
        connection.close()

def query_mysql(query, params=None):
    """
    执行SQL查询并返回结果
    
    参数:
        query (str): SQL查询语句
        params (tuple, optional): 查询参数
        
    返回:
        DataFrame: 查询结果，失败则返回None
    """
    connection = connect_to_mysql()
    if connection is None:
        return None
    
    try:
        # 执行查询
        if params:
            df = pd.read_sql(query, connection, params=params)
        else:
            df = pd.read_sql(query, connection)
            
        logger.debug(f"查询返回 {len(df)} 行数据")
        return df
        
    except Error as e:
        logger.error(f"执行MySQL查询时出错: {e}")
        return None
        
    finally:
        connection.close()

def execute_mysql(query, params=None):
    """
    执行MySQL命令（非查询）
    
    参数:
        query (str): SQL命令
        params (tuple, optional): 命令参数
        
    返回:
        bool: 是否执行成功
    """
    connection = connect_to_mysql()
    if connection is None:
        return False
    
    cursor = connection.cursor()
    
    try:
        # 执行命令
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
            
        # 提交更改
        connection.commit()
        
        # 返回影响的行数
        affected_rows = cursor.rowcount
        logger.debug(f"SQL命令已执行，影响了 {affected_rows} 行")
        
        return True
        
    except Error as e:
        logger.error(f"执行MySQL命令时出错: {e}")
        connection.rollback()
        return False
        
    finally:
        cursor.close()
        connection.close()