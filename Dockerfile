# 使用Python 3.10作为基础镜像
FROM python:3.10-slim

# 其余部分不变
WORKDIR /app

# 安装基本系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制程序代码和其他文件
COPY . .

# 创建模型和数据目录（如果不存在）
RUN mkdir -p models data

# 设置容器启动命令
CMD ["python", "trading_system.py"]