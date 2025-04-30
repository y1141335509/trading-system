# src/reporting/notifications.py

import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from datetime import datetime
import subprocess
import requests

# 设置日志
logger = logging.getLogger(__name__)

def send_notification(message, title="交易提醒", method="print"):
    """
    发送通知
    
    参数:
        message (str): 通知内容
        title (str): 通知标题
        method (str): 通知方式，可选 "print"、"system"、"webhook"
        
    返回:
        bool: 是否发送成功
    """
    try:
        # 添加时间戳
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        full_message = f"[{timestamp}] {title}: {message}"
        
        # 根据不同方式发送通知
        if method == "print":
            # 简单打印到控制台
            print(f"[通知] {full_message}")
            success = True
            
        elif method == "system":
            # 系统通知（适用于macOS/Linux/Windows）
            try:
                # 检测操作系统
                if os.name == 'posix':  # macOS 或 Linux
                    if 'darwin' in os.sys.platform:  # macOS
                        # 使用 osascript 发送通知
                        cmd = f"""osascript -e 'display notification "{message}" with title "{title}"'"""
                        subprocess.run(cmd, shell=True)
                    else:  # Linux
                        # 使用 notify-send 发送通知
                        cmd = f"""notify-send "{title}" "{message}" """
                        subprocess.run(cmd, shell=True)
                else:  # Windows
                    # 使用 Windows 通知（需要安装 win10toast 包）
                    try:
                        from win10toast import ToastNotifier
                        toaster = ToastNotifier()
                        toaster.show_toast(title, message, duration=10)
                    except ImportError:
                        logger.warning("Windows通知需要安装win10toast包")
                        print(f"[通知] {full_message}")
                
                success = True
                
            except Exception as e:
                logger.error(f"发送系统通知失败: {str(e)}")
                # 回退到打印
                print(f"[通知] {full_message}")
                success = False
                
        elif method == "webhook":
            # 使用Webhook发送通知（例如发送到Slack、Discord等）
            webhook_url = os.getenv('NOTIFICATION_WEBHOOK_URL')
            
            if not webhook_url:
                logger.warning("未设置NOTIFICATION_WEBHOOK_URL环境变量")
                print(f"[通知] {full_message}")
                return False
                
            # 构建payload
            payload = {
                "text": full_message
            }
            
            # 发送请求
            response = requests.post(webhook_url, json=payload)
            
            if response.status_code == 200:
                success = True
            else:
                logger.error(f"发送Webhook通知失败: {response.text}")
                print(f"[通知] {full_message}")
                success = False
                
        else:
            logger.warning(f"不支持的通知方式: {method}")
            print(f"[通知] {full_message}")
            success = True
            
        # 记录通知到日志文件
        log_notification(title, message)
        
        return success
        
    except Exception as e:
        logger.error(f"发送通知失败: {str(e)}")
        print(f"[通知失败] {title}: {message} - 错误: {str(e)}")
        return False

def log_notification(title, message):
    """
    记录通知到日志文件
    
    参数:
        title (str): 通知标题
        message (str): 通知内容
        
    返回:
        bool: 是否记录成功
    """
    try:
        # 创建日志目录
        data_dir = os.getenv('DATA_DIR', 'data')
        log_dir = os.path.join(data_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 日志文件路径
        log_file = os.path.join(log_dir, 'notifications.log')
        
        # 添加时间戳
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 写入日志
        with open(log_file, 'a') as f:
            f.write(f"{timestamp} - {title}: {message}\n")
            
        return True
        
    except Exception as e:
        logger.error(f"记录通知到日志失败: {str(e)}")
        return False

def send_email_report(recipient, subject, body, attachments=None, html=False):
    """
    发送邮件报告
    
    参数:
        recipient (str): 收件人邮箱
        subject (str): 邮件主题
        body (str): 邮件正文
        attachments (list): 附件路径列表
        html (bool): 是否为HTML格式
        
    返回:
        bool: 是否发送成功
    """
    try:
        # 获取SMTP配置
        smtp_server = os.getenv('SMTP_SERVER')
        smtp_port = int(os.getenv('SMTP_PORT', '587'))
        smtp_username = os.getenv('SMTP_USERNAME')
        smtp_password = os.getenv('SMTP_PASSWORD')
        sender_email = os.getenv('SENDER_EMAIL', smtp_username)
        
        if not all([smtp_server, smtp_username, smtp_password]):
            logger.error("SMTP配置不完整，无法发送邮件")
            return False
            
        # 创建邮件
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient
        msg['Subject'] = subject
        
        # 添加正文
        if html:
            msg.attach(MIMEText(body, 'html'))
        else:
            msg.attach(MIMEText(body, 'plain'))
            
        # 添加附件
        if attachments:
            for file_path in attachments:
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as file:
                        part = MIMEApplication(file.read(), Name=os.path.basename(file_path))
                        part['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
                        msg.attach(part)
                else:
                    logger.warning(f"附件不存在: {file_path}")
        
        # 连接到SMTP服务器并发送
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            # 使用TLS加密
            server.starttls()
            
            # 登录
            server.login(smtp_username, smtp_password)
            
            # 发送邮件
            server.send_message(msg)
            
        logger.info(f"已成功发送邮件到: {recipient}")
        return True
        
    except Exception as e:
        logger.error(f"发送邮件失败: {str(e)}")
        return False

def send_daily_report_email(recipient, report_path=None, include_charts=True):
    """
    发送每日报告邮件
    
    参数:
        recipient (str): 收件人邮箱
        report_path (str): 报告路径，如果为None则生成新报告
        include_charts (bool): 是否包含图表
        
    返回:
        bool: 是否发送成功
    """
    try:
        # 如果没有提供报告路径，生成新报告
        if not report_path:
            from .performance import generate_daily_report
            report_path = generate_daily_report(format='html')
            
            if not report_path:
                logger.error("生成每日报告失败")
                return False
        
        # 读取报告内容
        with open(report_path, 'r') as f:
            report_content = f.read()
            
        # 准备邮件附件
        attachments = []
        
        # 如果包含图表，生成并添加图表
        if include_charts:
            from .performance import get_pnl_history
            from .visualization import plot_portfolio_performance
            
            # 获取盈亏历史
            pnl_data = get_pnl_history(days=30)
            
            if not pnl_data.empty:
                # 生成绩效图表
                data_dir = os.getenv('DATA_DIR', 'data')
                report_dir = os.path.join(data_dir, 'reports')
                chart_path = os.path.join(report_dir, f"pnl_chart_{datetime.now().strftime('%Y%m%d')}.png")
                
                plot_portfolio_performance(pnl_data, save_path=chart_path)
                
                if os.path.exists(chart_path):
                    attachments.append(chart_path)
        
        # 生成邮件标题
        today = datetime.now().strftime('%Y-%m-%d')
        subject = f"交易系统每日报告 - {today}"
        
        # 发送邮件
        success = send_email_report(
            recipient=recipient,
            subject=subject,
            body=report_content,
            attachments=attachments,
            html=True
        )
        
        return success
        
    except Exception as e:
        logger.error(f"发送每日报告邮件失败: {str(e)}")
        return False