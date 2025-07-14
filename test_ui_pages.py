#!/usr/bin/env python3
"""
测试UI页面是否正常
"""
import requests
import time

def test_ui_pages():
    base_url = "http://localhost:8082"
    
    print("🧪 测试COT-DIR UI页面...")
    time.sleep(2)  # 等待服务器启动
    
    test_urls = [
        ("/", "主页"),
        ("/strategies", "推理策略页面"),
        ("/docs", "文档页面"),
        ("/system", "系统状态页面"),
        ("/test", "测试页面"),
        ("/api/stats", "统计API"),
        ("/api/strategies", "策略API"),
        ("/api/system", "系统API"),
    ]
    
    for url, name in test_urls:
        try:
            response = requests.get(f"{base_url}{url}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {name}: 正常 (状态码: {response.status_code})")
            else:
                print(f"❌ {name}: 异常 (状态码: {response.status_code})")
        except Exception as e:
            print(f"❌ {name}: 连接失败 - {e}")
    
    print(f"\n🌐 访问地址: {base_url}")
    print("📋 可用页面:")
    print("  - 概览页面: http://localhost:8082/")
    print("  - 推理策略: http://localhost:8082/strategies")
    print("  - 文档查看: http://localhost:8082/docs")
    print("  - 系统状态: http://localhost:8082/system")
    print("  - 测试控制台: http://localhost:8082/test")

if __name__ == "__main__":
    test_ui_pages()