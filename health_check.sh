#!/bin/bash

# 检查健康文件时间戳
LIVE_HEALTH=$(cat data/health.txt)
PAPER_HEALTH=$(cat paper_data/health.txt)

# 当前时间
NOW=$(date +%s)

# 转换健康文件时间戳
LIVE_TIME=$(date -d "$LIVE_HEALTH" +%s)
PAPER_TIME=$(date -d "$PAPER_HEALTH" +%s)

# 检查间隔（超过15分钟视为异常）
if [ $((NOW - LIVE_TIME)) -gt 900 ]; then
  echo "实盘交易系统可能异常，请检查"
fi

if [ $((NOW - PAPER_TIME)) -gt 900 ]; then
  echo "模拟交易系统可能异常，请检查"
fi