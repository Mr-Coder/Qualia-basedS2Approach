#!/bin/bash
# COT-DIR 本地部署启动脚本
# 快速启动Web UI界面

echo "🚀 COT-DIR 本地部署启动脚本"
echo "================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装，请先安装Python3"
    exit 1
fi

# 检查是否在项目根目录
if [ ! -f "CLAUDE.md" ]; then
    echo "❌ 请在项目根目录运行此脚本"
    exit 1
fi

# 安装依赖
echo "📦 检查并安装依赖..."
pip install flask flask-cors psutil >/dev/null 2>&1

# 检查端口占用
if lsof -i :8080 >/dev/null 2>&1; then
    echo "⚠️  端口8080被占用，尝试使用端口8081..."
    PORT=8081
else
    PORT=8080
fi

# 启动Web UI
echo "🌐 启动Web UI (端口: $PORT)..."
cd ui
python3 -c "
import sys
sys.path.append('..')
from app import app
app.run(host='0.0.0.0', port=$PORT, debug=False)
" &

WEB_PID=$!
echo "✅ Web UI已启动 (PID: $WEB_PID)"
echo "🌐 访问地址: http://localhost:$PORT"
echo ""
echo "📋 可用页面:"
echo "  - 概览页面: http://localhost:$PORT/"
echo "  - 推理策略: http://localhost:$PORT/strategies" 
echo "  - 文档查看: http://localhost:$PORT/docs"
echo "  - 系统状态: http://localhost:$PORT/system"
echo "  - 测试控制台: http://localhost:$PORT/test"
echo ""
echo "🛑 要停止服务，请按 Ctrl+C 或运行: kill $WEB_PID"

# 等待用户中断
trap "echo ''; echo '🛑 正在停止服务...'; kill $WEB_PID; exit 0" INT

echo "⏳ 服务运行中，按 Ctrl+C 停止..."
wait $WEB_PID