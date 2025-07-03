#!/usr/bin/env python3
"""
Simple demonstration of Intelligent Math Tutor System
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.intelligent_tutor import (IntelligentTutor, ProblemContext,
                                          StudentState)


def create_demo_problem():
    """创建演示问题"""
    return ProblemContext(
        problem_text="小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？",
        problem_id="demo_001",
        difficulty_level=1,
        concept_tags=["addition", "counting"],
        expected_answer="8",
        solution_steps=[
            "1. 识别问题：这是一个加法问题",
            "2. 提取数字：小明有5个苹果，小红有3个苹果",
            "3. 计算：5 + 3 = 8",
            "4. 答案：他们一共有8个苹果"
        ],
        hints_available=["想想你有几个苹果，再拿来几个苹果，现在总共有多少个？"],
        similar_problems=["小华有4个橘子，小李有2个橘子，他们一共有多少个橘子？"]
    )


def demo_learning_process():
    """演示完整的学习过程"""
    print("🎓 智能数学辅导系统演示")
    print("=" * 50)
    
    # 创建辅导系统
    tutor = IntelligentTutor()
    problem = create_demo_problem()
    student_id = "demo_student"
    
    print(f"📚 问题：{problem.problem_text}")
    print(f"🎯 正确答案：{problem.expected_answer}")
    print()
    
    # 模拟学习过程
    scenarios = [
        {"name": "首次尝试（无答案）", "answer": "", "description": "展示初始辅导"},
        {"name": "提交错误答案", "answer": "6", "description": "展示错误处理"},
        {"name": "再次错误", "answer": "7", "description": "展示渐进式辅导"},
        {"name": "提交正确答案", "answer": "8", "description": "展示成功反馈"}
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"📝 场景 {i}: {scenario['name']}")
        print(f"   描述：{scenario['description']}")
        if scenario['answer']:
            print(f"   学生答案：{scenario['answer']}")
        
        # 获取系统响应
        response = tutor.solve_problem(student_id, problem, scenario['answer'])
        
        print(f"   响应类型：{response.response_type}")
        print(f"   置信度：{response.confidence_level:.2f}")
        print(f"   系统响应：")
        print(f"   {response.message}")
        print()
    
    # 显示学习进度
    print("📊 学习进度总结")
    progress = tutor.get_student_progress(student_id)
    for key, value in progress.items():
        if key != "student_id":
            print(f"   {key}: {value}")
    
    print("\n✅ 演示完成！")


if __name__ == "__main__":
    demo_learning_process() 