


---
<details>

<summary>Using Docker来24/7运行代码</summary>


```bash
############### 每日实际交易使用 ###############
# Stop containers
docker-compose -f docker-compose.live.yml down
# Clean up
docker system prune -f
# Rebuild and restart
docker-compose -f docker-compose.live.yml build --no-cache
docker-compose -f docker-compose.live.yml up -d
# Watch logs
docker-compose -f docker-compose.live.yml logs -f
############### 每日实际交易使用 ###############
```

```bash
############### 每日运行测试使用 ###############
# 停止当前容器
docker-compose down
# 重新构建并启动
docker-compose build
docker-compose -f docker-compose.paper.yml up -d
# 进行监控：
docker logs -f paper-trading  
############### 每日运行测试使用 ###############
```



```bash
############### 删除当前所有docker环境，并重新运行 ###############
# 停止当前容器
docker-compose down

# 确认想要删除的volume
docker volume ls

# 删除指定volume（如果需要清理）
# docker volume rm trading_system_mysql-data

# 列出images
docker images

# 删除指定images（如果需要完全重建）
docker rmi trading_system-trading-system

# 重新构建并启动
docker-compose build --no-cache
docker-compose up -d

# 进行监控：使用正确的容器名称
docker logs -f trading-system
############### 删除当前所有docker环境，并重新运行 ###############
```

```bash
############### 常见Docker指令 ###############
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
############### 常见Docker指令 ###############
```
</details>

---
<details>

<summary>关于项目</summary>

```
Algorithmic Trading System

An automated trading system built with Python and Alpaca API, supporting both paper trading and live trading environments.

Project Structure

trading_system/
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore file
├── docker-compose.yml          # Docker Compose for paper trading
├── docker-compose.live.yml     # Docker Compose for live trading
├── Dockerfile                  # Docker configuration
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── CHANGELOG.md                # Version history
├── LICENSE                     # License information
├── src/                        # Source code
│   ├── init.py
│   ├── trading_system.py       # Main system file
│   ├── indicators/             # Technical indicators
│   │   ├── init.py
│   │   ├── trend.py            # Trend indicators (MA, MACD)
│   │   ├── oscillators.py      # Oscillator indicators (RSI, etc)
│   │   └── volatility.py       # Volatility indicators (BB, ATR)
│   ├── ml/                     # Machine learning models
│   │   ├── init.py
│   │   ├── train.py            # Model training functions
│   │   ├── predict.py          # Prediction functions
│   │   └── evaluate.py         # Model evaluation
│   ├── rl/                     # Reinforcement learning components
│   │   ├── init.py
│   │   ├── environment.py      # Trading environment
│   │   └── agent.py            # RL agent implementation
│   ├── portfolio/              # Portfolio management
│   │   ├── init.py
│   │   ├── construction.py     # Portfolio construction
│   │   ├── risk.py             # Risk management
│   │   └── rebalance.py        # Portfolio rebalancing
│   ├── data/                   # Data handling
│   │   ├── init.py
│   │   ├── fetcher.py          # Data fetching from API
│   │   └── processor.py        # Data preprocessing
│   ├── execution/              # Trade execution
│   │   ├── init.py
│   │   ├── orders.py           # Order management
│   │   └── broker.py           # Broker interface
│   ├── reporting/              # Reporting and analysis
│   │   ├── init.py
│   │   ├── performance.py      # Performance metrics
│   │   ├── visualization.py    # Data visualization
│   │   └── notifications.py    # Alerts and notifications
│   └── utils/                  # Utilities
│       ├── init.py
│       ├── config.py           # Configuration handling
│       ├── logger.py           # Logging
│       └── database.py         # Database operations
├── tests/                      # Unit and integration tests
│   ├── init.py
│   ├── test_indicators.py
│   ├── test_ml_models.py
│   └── test_portfolio.py
├── models/                     # Saved ML/RL models
│   ├── .gitkeep
│   └── README.md               # Models readme
├── data/                       # Data storage
│   ├── .gitkeep
│   └── README.md               # Data readme
├── logs/                       # Log files
│   ├── .gitkeep
│   └── README.md               # Logs readme
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── backtest.ipynb
│   ├── strategy_development.ipynb
│   └── model_exploration.ipynb
└── scripts/                    # Utility scripts
├── setup.sh                # Setup script
├── start_trading.sh        # Start trading script
└── backup.sh               # Backup script

```
</details>