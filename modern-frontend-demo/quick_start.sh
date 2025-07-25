#!/bin/bash

echo "🚀 快速启动 COT-DIR 物性推理系统"
echo "================================"

# 清理缓存
echo "🧹 清理缓存..."
rm -rf node_modules/.vite
rm -rf dist

# 启动 API 服务器
echo "📡 启动 API 服务器 (端口 5000)..."
python demo_frontend.py &
API_PID=$!
echo "API 服务器 PID: $API_PID"

# 等待 API 服务器启动
sleep 3

# 启动前端开发服务器
echo "🖥️  启动前端开发服务器 (端口 3000)..."
npm run dev &
FRONTEND_PID=$!
echo "前端服务器 PID: $FRONTEND_PID"

echo ""
echo "✅ 服务器已启动!"
echo "================================"
echo "📌 访问地址: http://localhost:3000"
echo "📌 API 地址: http://localhost:5000"
echo ""
echo "🧩 物性推理功能："
echo "1. 访问 http://localhost:3000"
echo "2. 点击 '🧩 物性推理' 标签页"
echo "3. 查看算法讲解和分步演示"
echo ""
echo "⚠️  停止服务器: 按 Ctrl+C"
echo ""

# 等待用户中断
trap "kill $API_PID $FRONTEND_PID; exit" INT
wait