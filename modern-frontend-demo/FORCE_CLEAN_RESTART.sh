#!/bin/bash

echo "🧹 强制清理所有缓存和重启服务"
echo "=================================="

# 停止所有相关进程
echo "🛑 停止所有Node.js和Python进程..."
pkill -f "node.*vite"
pkill -f "python.*demo_frontend"
sleep 2

# 清理所有缓存
echo "🗑️  清理所有缓存文件..."
rm -rf node_modules/.vite
rm -rf dist
rm -rf .parcel-cache
rm -rf .cache

# 清理浏览器缓存相关
echo "🌐 清理本地存储..."
if command -v osascript &> /dev/null; then
    # macOS 清理 Chrome 缓存
    osascript -e 'tell application "Google Chrome" to execute front tab of front window javascript "localStorage.clear(); sessionStorage.clear(); location.reload(true);"' 2>/dev/null || true
fi

# 检查端口占用
echo "🔍 检查端口占用..."
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null; then
    echo "⚠️  端口 5000 被占用，尝试释放..."
    lsof -ti:5000 | xargs kill -9 2>/dev/null || true
fi

if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null; then
    echo "⚠️  端口 3000 被占用，尝试释放..."
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true
fi

# 等待端口释放
sleep 3

echo ""
echo "✅ 清理完成！现在启动服务..."
echo "=================================="

# 启动API服务器
echo "🚀 启动API服务器 (端口 5000)..."
python demo_frontend.py &
API_PID=$!
echo "API服务器PID: $API_PID"

# 等待API服务器启动
sleep 4

# 检查API服务器是否启动成功
if curl -s http://localhost:5000/api/test > /dev/null; then
    echo "✅ API服务器启动成功"
else
    echo "❌ API服务器启动失败"
fi

# 启动前端服务器
echo "🖥️  启动前端服务器 (端口 3000)..."
npm run dev &
FRONTEND_PID=$!
echo "前端服务器PID: $FRONTEND_PID"

echo ""
echo "🎉 服务器启动完成！"
echo "=================================="
echo "📱 访问地址: http://localhost:3000"
echo "🔗 API地址: http://localhost:5000"
echo ""
echo "🧩 使用步骤:"
echo "1. 访问 http://localhost:3000"
echo "2. 点击 '🧩 物性推理' 标签页"
echo "3. 查看合并后的算法讲解和分步演示"
echo ""
echo "⚠️  如果仍有问题，请使用无痕模式访问"
echo "⚠️  停止服务器: 按 Ctrl+C 或运行 ./stop_services.sh"
echo ""

# 创建停止脚本
cat > stop_services.sh << 'EOF'
#!/bin/bash
echo "🛑 停止所有服务..."
kill $API_PID $FRONTEND_PID 2>/dev/null
pkill -f "node.*vite"  
pkill -f "python.*demo_frontend"
echo "✅ 所有服务已停止"
EOF

chmod +x stop_services.sh

# 等待用户中断
trap "kill $API_PID $FRONTEND_PID 2>/dev/null; pkill -f 'node.*vite'; pkill -f 'python.*demo_frontend'; exit" INT TERM
wait