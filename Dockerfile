# 使用Python 3.10作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置时区为美国太平洋时间
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    tzdata \
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
# 启动后台监控进程\n\
python trading_system.py > /app/logs/trading_system.log 2>&1 &\n\
\n\
# 保持容器运行\n\
tail -f /app/logs/trading_system.log\n\
' > /app/start.sh

# 赋予启动脚本执行权限
RUN chmod +x /app/start.sh

# 设置容器启动命令
CMD ["/app/start.sh"]