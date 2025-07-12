"""
快速测试脚本 - COT-DIR + MLR 演示
包含多个预设问题，可以快速测试系统的不同功能

使用方法：
python quick_test.py
"""

import subprocess
import sys


def run_demo(question: str, demo_name: str):
    """运行演示并显示结果"""
    print(f"\n🎯 演示：{demo_name}")
    print("="*60)
    print(f"📝 问题：{question}")
    print("="*60)
    
    try:
        result = subprocess.run([
            sys.executable, "interactive_demo.py", question
        ], capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"❌ 错误：{result.stderr}")
    except Exception as e:
        print(f"❌ 运行错误：{e}")
    
    print("\n" + "="*60)
    input("按回车键继续下一个演示...")

def main():
    """主程序"""
    print("🚀 COT-DIR + MLR 快速测试套件")
    print("="*80)
    print("📚 包含多种类型的数学问题演示")
    print("🔍 展示完整的推理过程：文字输入→实体识别→关系发现→多层推理→答案生成")
    print("="*80)
    
    # 测试用例
    test_cases = [
        {
            "question": "小明有3个苹果，小红有5个苹果，他们一共有多少个苹果？",
            "name": "基础加法问题"
        },
        {
            "question": "小华买了12本书，小李买了7本书，他们加起来有多少本书？",
            "name": "购买场景加法"
        },
        {
            "question": "教室里有25个学生，又来了8个学生，总共有多少个学生？",
            "name": "增加场景问题"
        },
        {
            "question": "小明有10个苹果，吃了3个，还剩多少个？",
            "name": "减法计算问题"
        },
        {
            "question": "每盒有6支笔，买了4盒，一共有多少支笔？",
            "name": "乘法计算问题"
        }
    ]
    
    print(f"\n📋 测试案例总览：")
    for i, case in enumerate(test_cases, 1):
        print(f"   {i}. {case['name']}：{case['question']}")
    
    print(f"\n🎮 选择运行模式：")
    print("   1. 逐个演示（推荐）")
    print("   2. 全部演示")
    print("   3. 选择特定演示")
    print("   4. 自定义问题")
    
    choice = input("\n请选择运行模式 (1-4): ").strip()
    
    if choice == "1":
        # 逐个演示
        for i, case in enumerate(test_cases, 1):
            print(f"\n🔄 正在运行第 {i}/{len(test_cases)} 个演示...")
            run_demo(case["question"], case["name"])
        
        print("\n🎉 所有演示完成！")
    
    elif choice == "2":
        # 全部演示（连续运行）
        for i, case in enumerate(test_cases, 1):
            print(f"\n🔄 运行演示 {i}/{len(test_cases)}：{case['name']}")
            try:
                subprocess.run([
                    sys.executable, "interactive_demo.py", case["question"]
                ], check=True)
            except Exception as e:
                print(f"❌ 演示 {i} 失败：{e}")
        
        print("\n🎉 批量演示完成！")
    
    elif choice == "3":
        # 选择特定演示
        print(f"\n请选择要运行的演示 (1-{len(test_cases)}):")
        for i, case in enumerate(test_cases, 1):
            print(f"   {i}. {case['name']}")
        
        try:
            index = int(input("\n请输入编号: ").strip()) - 1
            if 0 <= index < len(test_cases):
                case = test_cases[index]
                run_demo(case["question"], case["name"])
            else:
                print("❌ 无效的编号")
        except ValueError:
            print("❌ 请输入有效的数字")
    
    elif choice == "4":
        # 自定义问题
        print("\n📝 请输入您的数学问题：")
        print("💡 提示：使用中文，包含具体数字，表述清晰")
        print("📚 例如：小明有8个苹果，小红比小明多3个，他们一共有多少个苹果？")
        
        custom_question = input("\n问题: ").strip()
        if custom_question:
            run_demo(custom_question, "自定义问题")
        else:
            print("❌ 问题不能为空")
    
    else:
        print("❌ 无效的选择")

if __name__ == "__main__":
    main() 