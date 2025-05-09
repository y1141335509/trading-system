# 交易系统仪表盘

交易系统仪表盘是一个基于Web的可视化和管理工具，提供实时监控、详细分析和系统配置功能。

## 功能特性

- **实时监控**: 账户状态、持仓和交易的实时更新
- **详细分析**: 性能图表、回报分析和风险评估
- **报告管理**: 查看和生成交易报告
- **系统配置**: 完整的交易系统设置界面

## 快速开始

### 直接运行

```bash
# 从项目根目录
python run_dashboard.py

# 指定端口
python run_dashboard.py --port 8080

# 启用调试模式
python run_dashboard.py --debug
```

### Docker中运行

如果您使用Docker Compose运行交易系统，仪表盘会自动启动。
访问 `http://localhost:5000` 查看仪表盘。

## 仪表盘结构

```
src/dashboard/
├── __init__.py          # 初始化模块
├── app.py               # Flask应用
├── api.py               # API数据获取
├── templates/           # HTML模板
│   ├── index.html       # 主页
│   ├── charts.html      # 图表页
│   ├── reports.html     # 报告页
│   ├── settings.html    # 设置页
│   └── about.html       # 关于页
└── static/              # 静态资源(CSS/JS)
```

## API端点

仪表盘提供以下API端点:

- `/api/performance`: 获取性能数据
- `/api/positions`: 获取当前持仓数据
- `/api/trades`: 获取交易历史
- `/api/reports`: 获取报告列表
- `/api/status`: 获取系统状态
- `/api/settings`: 获取/保存设置
- `/api/maintenance`: 系统维护操作
- `/api/system_info`: 获取系统信息

## 自定义和扩展

仪表盘设计为可扩展的，您可以通过以下方式进行自定义:

1. 添加新页面: 在`templates/`目录创建新HTML模板
2. 添加新API: 在`api.py`中添加新的数据获取函数
3. 注册新路由: 在`app.py`的`create_app()`函数中注册新路由

## 安全注意事项

- 默认情况下，仪表盘不启用认证
- 在生产环境中，强烈建议启用认证功能
- 可以通过设置页面的"仪表盘设置"部分启用认证

## 故障排除