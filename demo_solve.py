#!/usr/bin/env python3
"""
COT-DIR 解题系统演示脚本
"""

import requests
import json
import time

def test_solve_problem(problem, strategy='auto'):
    """测试解题功能"""
    url = 'http://localhost:8082/api/solve'
    data = {
        'problem': problem,
        'strategy': strategy
    }
    
    print(f"🤖 问题: {problem}")
    print(f"📝 策略: {strategy}")
    print("⏳ 正在思考...")
    
    try:
        response = requests.post(url, json=data)
        result = response.json()
        
        if result.get('success'):
            print(f"✅ 答案: {result['answer']}")
            print(f"🎯 置信度: {result['confidence']*100:.1f}%")
            print(f"⚡ 执行时间: {result['execution_time']}秒")
            print(f"🧠 使用策略: {result['strategy_used'].upper()}")
            
            print("\n🔍 推理过程:")
            for step in result['reasoning_steps']:
                print(f"  {step['step']}. {step['description']}")
        else:
            print(f"❌ 解题失败: {result.get('error', '未知错误')}")
            
    except Exception as e:
        print(f"❌ 请求失败: {e}")
    
    print("-" * 50)

def main():
    """主函数"""
    print("🧠 COT-DIR 智能数学解题系统演示")
    print("=" * 50)
    
    # 测试问题集
    test_problems = [
        ("小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？", "cot"),
        ("一个长方形的长是8米，宽是5米，求它的面积和周长。", "auto"),
        ("小张买了3支笔，每支笔5元，给了店主20元，应该找回多少钱？", "got"),
        ("一个班级有40名学生，其中男生占60%，女生有多少人？", "tot"),
        ("如果今天是星期一，那么100天后是星期几？", "auto")
    ]
    
    for problem, strategy in test_problems:
        test_solve_problem(problem, strategy)
        time.sleep(1)  # 避免请求过快
    
    print("\n✨ 演示完成！")
    print("🌐 打开浏览器访问: http://localhost:8082")
    print("💡 在网页界面中可以输入自定义问题进行测试")

if __name__ == "__main__":
    main()