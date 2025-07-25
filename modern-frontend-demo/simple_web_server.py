#!/usr/bin/env python3
"""
简单的HTTP服务器，用于提供前端文件
避免file://协议导致的CORS问题
"""
import os
import http.server
import socketserver
from pathlib import Path

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=Path(__file__).parent, **kwargs)

    def end_headers(self):
        # 添加CORS头
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        # 简化日志输出
        print(f"[WEB] {format % args}")

if __name__ == "__main__":
    PORT = 8080
    
    print(f"🌐 启动前端文件服务器...")
    print(f"📁 服务目录: {Path(__file__).parent}")
    print(f"🚀 访问地址: http://localhost:{PORT}")
    print(f"🔧 解题调试页面: http://localhost:{PORT}/debug-solver.html")
    print(f"📊 服务监控面板: http://localhost:{PORT}/service-dashboard.html")
    print(f"📁 分块上传界面: http://localhost:{PORT}/chunked-upload-demo.html")
    print(f"🎯 完整解题界面: http://localhost:{PORT}/integrated-demo.html")
    
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f"\n✅ 服务器运行在端口 {PORT}")
        print("按 Ctrl+C 停止服务器")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 服务器已停止")