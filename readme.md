


---
## Using Docker
```bash
############### 每日使用 ###############
# 停止当前容器
docker-compose down
# 重新构建并启动
docker-compose build
docker-compose up -d
# 进行监控：
docker logs -f trading-bot
############### 每日使用 ###############
```

```bash
############### 删除当前所有docker环境，并重新运行 ###############
# 停止当前容器
docker-compose down
docker stop trading-bot
docker rm trading-bot

# 确认想要删除的volume：
docker volume ls

# 删除指定volume（注意要选对想要删除的）：
docker volume rm trading_system_trading-models
docker volume rm trading_system_trading-data

# 然后列出images
docker images
# 删除指定images：
docker rmi trading_system-trading-system

# 重新构建并启动
docker-compose build --no-cache
docker-compose up -d
# 进行监控：
docker logs -f trading-bot
############### 删除当前所有docker环境，并重新运行 ###############
```

```bash
# 停止当前容器
docker stop trading-bot

# 重新构建镜像
docker compose build

# 启动
docker-compose up -d

# 停止
docker-compose down

# 查看日志
docker-compose logs -f

```