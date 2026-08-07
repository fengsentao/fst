#!/bin/bash
# FST-Quant Web 启动脚本
# 用法: bash start.sh [port]

PORT=${1:-8000}
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "🚀 启动 FST-Quant Web 服务..."
echo "📍 地址: http://localhost:$PORT"
echo "📍 API文档: http://localhost:$PORT/docs"
echo ""

# 安装依赖
pip install -q fastapi uvicorn pydantic python-multipart 2>/dev/null

# 启动服务
cd "$DIR/.."
python -m uvicorn web.backend.app:app --host 0.0.0.0 --port $PORT --reload
