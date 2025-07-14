#!/bin/bash

# COT-DIR 项目启动脚本
echo "🚀 启动COT-DIR本地Web UI"
echo "================================="

# 检查是否有现有进程
if lsof -i :8082 > /dev/null 2>&1; then
    echo "⚠️  端口8082已被占用，正在停止现有进程..."
    lsof -ti :8082 | xargs kill -9
    sleep 2
fi

# 启动UI服务器
echo "🌐 启动Web UI (端口: 8082)..."
python ui/app.py > ui_log.txt 2>&1 &
UI_PID=$!

# 等待服务器启动
sleep 3

# 检查进程是否成功启动
if ps -p $UI_PID > /dev/null; then
    echo "✅ Web UI已启动 (PID: $UI_PID)"
    echo "🌐 访问地址: http://localhost:8082"
    echo ""
    echo "📋 可用页面:"
    echo "  - 概览页面: http://localhost:8082/"
    echo "  - 推理策略: http://localhost:8082/strategies"
    echo "  - 文档查看: http://localhost:8082/docs"
    echo "  - 系统状态: http://localhost:8082/system"
    echo "  - 测试控制台: http://localhost:8082/test"
    echo ""
    echo "🛑 要停止服务，请按 Ctrl+C 或运行: kill $UI_PID"
    echo "📋 查看日志: tail -f ui_log.txt"
    echo ""
    echo "⏳ 服务运行中，按 Ctrl+C 停止..."
    
    # 等待用户中断
    trap "echo ''; echo '🛑 正在停止服务...'; kill $UI_PID; exit 0" INT
    
    # 持续监控进程
    while ps -p $UI_PID > /dev/null; do
        sleep 1
    done
    
    echo "❌ 服务意外停止"
else
    echo "❌ 服务启动失败"
    exit 1
fi