#!/usr/bin/env python3
"""
演示前端系统的运行效果
"""

import requests
import json
import time
import webbrowser
from pathlib import Path

def test_frontend_system():
    """测试前端系统"""
    print("🚀 COT-DIR 现代化前端系统演示")
    print("=" * 50)
    
    # 1. 测试后端API
    print("\n📡 测试后端API:")
    print("-" * 30)
    
    try:
        # 健康检查
        response = requests.get("http://localhost:3002/api/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ 后端服务器状态: {health_data['status']}")
            print(f"📊 推理系统: {health_data['reasoning_system']}")
            print(f"🔗 版本: {health_data['version']}")
        else:
            print(f"❌ 后端服务器无响应: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到后端服务器: {e}")
        return False
    
    # 2. 测试解题功能
    print("\n🧮 测试解题功能:")
    print("-" * 30)
    
    test_problems = [
        "小明有10个苹果，给了小红3个，还剩多少个？",
        "一辆汽车以60公里/小时的速度行驶2小时，行驶了多少公里？",
        "班级有40个学生，其中60%是男生，男生有多少人？"
    ]
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\n问题 {i}: {problem}")
        
        try:
            response = requests.post(
                "http://localhost:3002/api/solve",
                json={"problem": problem},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 答案: {result.get('answer', '未知')}")
                print(f"🎯 置信度: {result.get('confidence', 0):.2f}")
                print(f"⏱️  处理时间: {result.get('processing_time', 0):.3f}s")
                
                # 显示推理步骤
                steps = result.get('reasoning_steps', [])
                if steps:
                    print(f"📋 推理步骤 ({len(steps)}步):")
                    for j, step in enumerate(steps[:3], 1):
                        print(f"  {j}. {step.get('description', '未知步骤')}")
                
                # 显示发现的关系
                relations = result.get('relations_found', [])
                if relations:
                    print(f"🔍 发现关系 ({len(relations)}个):")
                    for j, rel in enumerate(relations[:2], 1):
                        print(f"  {j}. {rel.get('description', '未知关系')}")
                
            else:
                print(f"❌ 解题失败: {response.status_code}")
                
        except Exception as e:
            print(f"❌ 解题请求失败: {e}")
    
    # 3. 显示前端访问信息
    print("\n🌐 前端访问信息:")
    print("-" * 30)
    
    print("可用的前端界面:")
    print("1. 完整系统界面: http://localhost:8080/integrated-demo.html")
    print("2. 简化演示界面: http://localhost:8080/simple-demo.html")
    print("3. 完整演示界面: http://localhost:8080/complete-demo.html")
    print("4. 基础界面: http://localhost:8080/demo.html")
    print("5. 首页: http://localhost:8080/index.html")
    
    # 4. 显示后端接口
    print("\n📋 后端API接口:")
    print("-" * 30)
    
    print("- 健康检查: GET http://localhost:3002/api/health")
    print("- 解题接口: POST http://localhost:3002/api/solve")
    print("- 批量解题: POST http://localhost:3002/api/batch-solve")
    print("- 获取历史: GET http://localhost:3002/api/history")
    print("- 获取统计: GET http://localhost:3002/api/stats")
    
    # 5. 系统功能特性
    print("\n⭐ 系统功能特性:")
    print("-" * 30)
    
    print("✅ 智能数学问题解答")
    print("✅ 实时推理步骤展示")
    print("✅ 实体关系图可视化")
    print("✅ 多种推理策略")
    print("✅ 历史记录管理")
    print("✅ 学习指导系统")
    print("✅ 错误分析功能")
    print("✅ 知识图谱展示")
    print("✅ 响应式设计")
    print("✅ 现代化UI界面")
    
    # 6. 技术栈信息
    print("\n🔧 技术栈:")
    print("-" * 30)
    
    print("前端:")
    print("  • HTML5 + CSS3 + JavaScript")
    print("  • Tailwind CSS 样式框架")
    print("  • 响应式设计")
    print("  • 现代化UI组件")
    
    print("后端:")
    print("  • Flask Web框架")
    print("  • COT-DIR算法引擎")
    print("  • 增强IRD引擎 v2.0")
    print("  • RESTful API")
    print("  • JSON数据交换")
    
    print("\n🎉 系统已成功启动!")
    print("=" * 50)
    
    print("💡 使用说明:")
    print("1. 在浏览器中打开上述任一前端界面")
    print("2. 输入数学问题进行求解")
    print("3. 查看详细的推理过程")
    print("4. 探索各种功能特性")
    
    print("\n🚨 注意事项:")
    print("- 当前使用模拟推理系统")
    print("- 实际部署时会连接真实的增强引擎")
    print("- 前端界面完全响应式，支持移动设备")
    
    return True

if __name__ == "__main__":
    success = test_frontend_system()
    if success:
        print("\n✅ 前端系统演示完成!")
        print("🌐 请在浏览器中访问前端界面体验完整功能")
    else:
        print("\n❌ 前端系统演示失败!")