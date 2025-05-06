FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    tzdata \
    default-mysql-client \
    netcat-traditional \
    cron \
    && rm -rf /var/lib/apt/lists/* \
    && ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码和创建目录
COPY . .
RUN mkdir -p models data logs

# 设置环境变量
ENV DOCKER_ENVIRONMENT=true

# 创建并设置启动脚本
RUN echo '#!/bin/bash\n\
echo "启动交易系统..."\n\
mkdir -p /app/logs\n\
python -m src.trading_system > /app/logs/trading_system.log 2>&1 &\n\
tail -f /app/logs/trading_system.log\n\
' > /app/start.sh \
    && chmod +x /app/start.sh

CMD ["/app/start.sh"]