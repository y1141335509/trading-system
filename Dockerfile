# 使用Python 3.10作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置时区为美国太平洋时间
RUN apt-get update && apt-get install -y \
    gcc \
    tzdata \
    default-mysql-client \
    netcat-traditional \
    cron \
    && rm -rf /var/lib/apt/lists/* \
    && ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime \
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

# 创建启动脚本
RUN echo '#!/bin/bash\n\
echo "启动交易系统..."\n\
# 创建日志目录\n\
mkdir -p /app/logs\n\
\n\
# 输出日期时间
echo "当前时间: $(date)"\n\
\n\
# 立即执行一次交易循环
echo "执行初始交易循环..."\n\
python -c "from src.trading_system import run_intelligent_trading_system; run_intelligent_trading_system()"\n\
\n\
# 启动主交易系统
echo "启动持续监控..."\n\
python -c "from src.trading_system import main; main()" > /app/logs/trading_system.log 2>&1 &\n\
\n\
# 保持容器运行
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