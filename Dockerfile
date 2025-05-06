# 使用Python 3.10作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置时区为美国东部时间 (纽约，美国市场时区)
RUN apt-get update && apt-get install -y \
    gcc \
    tzdata \
    default-mysql-client \
    netcat-traditional \
    cron \
    && rm -rf /var/lib/apt/lists/* \
    && ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制程序代码和其他文件
COPY . .

# 创建模型和数据目录（如果不存在）
RUN mkdir -p models data logs

# 设置环境变量，指示在Docker环境中运行
ENV DOCKER_ENVIRONMENT=true

# 创建调试脚本用于检查环境变量
RUN echo '#!/bin/bash\n\
echo "==== 环境变量检查 ===="\n\
echo "DOCKER_ENVIRONMENT: $DOCKER_ENVIRONMENT"\n\
echo "ALPACA_PAPER: $ALPACA_PAPER"\n\
echo "ALPACA_PAPER_API_KEY: ${ALPACA_PAPER_API_KEY:0:5}..."\n\
echo "ALPACA_PAPER_API_SECRET: ${ALPACA_PAPER_API_SECRET:0:5}..."\n\
echo "ALPACA_PAPER_URL: $ALPACA_PAPER_URL"\n\
echo "ALPACA_LIVE_API_KEY: ${ALPACA_LIVE_API_KEY:0:5}..."\n\
echo "ALPACA_LIVE_API_SECRET: ${ALPACA_LIVE_API_SECRET:0:5}..."\n\
echo "ALPACA_LIVE_URL: $ALPACA_LIVE_URL"\n\
echo "MYSQL_HOST: $MYSQL_HOST"\n\
echo "MYSQL_PORT: $MYSQL_PORT"\n\
echo "MYSQL_USER: $MYSQL_USER"\n\
echo "MYSQL_PASSWORD: ${MYSQL_PASSWORD:0:3}..."\n\
echo "MYSQL_QUANT_DB: $MYSQL_QUANT_DB"\n\
echo "==== 环境变量检查结束 ===="\n\
' > /app/check_env.sh
RUN chmod +x /app/check_env.sh

# 创建启动脚本
RUN echo '#!/bin/bash\n\
echo "启动交易系统..."\n\
# 创建日志目录\n\
mkdir -p /app/logs\n\
\n\
# 输出日期时间和环境变量\n\
echo "当前时间: $(date)"\n\
/app/check_env.sh\n\
\n\
# 尝试连接到API\n\
echo "测试API连接..."\n\
python test_api.py\n\
\n\
# 立即执行一次交易循环\n\
echo "执行初始交易循环..."\n\
python run_trading_cycle.py\n\
\n\
# 启动主交易系统\n\
echo "启动持续监控..."\n\
python -c "from src.trading_system import main; main()" > /app/logs/trading_system.log 2>&1 &\n\
\n\
# 保持容器运行\n\
echo "系统已启动，查看日志..."\n\
tail -f /app/logs/trading_system.log\n\
' > /app/start.sh

# Create wait-for-mysql script with improved connection checking
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
host="$MYSQL_HOST"\n\
port="$MYSQL_PORT"\n\
user="$MYSQL_USER"\n\
password="$MYSQL_PASSWORD"\n\
\n\
echo "Waiting for MySQL to be ready..."\n\
until nc -z -v -w30 "$host" "$port" && \
      mysql -h"$host" -u"$user" -p"$password" -e "SELECT 1;" > /dev/null 2>&1; do\n\
  echo "MySQL is unavailable - sleeping"\n\
  sleep 2\n\
done\n\
\n\
echo "MySQL is up and running!"\n\
exec "$@"' > /wait-for-mysql.sh \
    && chmod +x /wait-for-mysql.sh

# 赋予启动脚本执行权限
RUN chmod +x /app/start.sh

# 设置容器启动命令
CMD ["/wait-for-mysql.sh", "/app/start.sh"]