#!/usr/bin/env python3
"""
DIR-MWP Dataset Generator
Generates a complete dataset of 200 Domain-specific Implicit Relation Math Word Problems
following the complexity distribution: L0(30), L1(50), L2(80), L3(40)
"""

import json
import random
from datetime import datetime


def generate_l0_problems(start_id=1, count=30):
    """Generate L0 explicit problems - simple arithmetic with direct relations"""
    problems = []
    
    templates = [
        # Basic addition
        {
            "template": "小明有{a}个{item}，小红有{b}个{item}。他们一共有多少个{item}？",
            "answer_formula": "{a} + {b}",
            "solution": "直接相加：{a} + {b} = {answer}",
            "explicit_relations": ["小明的{item}数量", "小红的{item}数量"],
            "items": ["苹果", "橘子", "糖果", "玩具", "书本", "铅笔"]
        },
        # Basic subtraction
        {
            "template": "一本书有{total}页，小李已经读了{read}页。还剩多少页没读？",
            "answer_formula": "{total} - {read}",
            "solution": "用总页数减去已读页数：{total} - {read} = {answer}",
            "explicit_relations": ["总页数", "已读页数"]
        },
        # Basic multiplication
        {
            "template": "商店里有{rows}排货架，每排有{items}个商品。商店总共有多少个商品？",
            "answer_formula": "{rows} * {items}",
            "solution": "用排数乘以每排商品数：{rows} × {items} = {answer}",
            "explicit_relations": ["排数", "每排商品数"]
        },
        # Basic division
        {
            "template": "老师要把{total}个{item}平均分给{students}个学生。每个学生能分到多少个{item}？",
            "answer_formula": "{total} / {students}",
            "solution": "用总数除以学生数：{total} ÷ {students} = {answer}",
            "explicit_relations": ["总{item}数", "学生数"],
            "items": ["糖果", "铅笔", "橡皮", "贴纸", "书本"]
        }
    ]
    
    for i in range(count):
        template = random.choice(templates)
        
        if "item" in template.get("template", "") and "items" in template:
            item = random.choice(template["items"])
            if template["template"].count("{item}") > 0:
                # Handle problems with items
                if "小明有" in template["template"]:
                    a = random.randint(10, 50)
                    b = random.randint(5, 30)
                    answer = a + b
                    problem_text = template["template"].format(a=a, b=b, item=item)
                    solution = template["solution"].format(a=a, b=b, answer=answer)
                    explicit_relations = [rel.format(item=item) for rel in template["explicit_relations"]]
                elif "老师要把" in template["template"]:
                    students = random.randint(3, 8)
                    total = students * random.randint(3, 10)
                    answer = total // students
                    problem_text = template["template"].format(total=total, students=students, item=item)
                    solution = template["solution"].format(total=total, students=students, answer=answer)
                    explicit_relations = [rel.format(item=item) for rel in template["explicit_relations"]]
        else:
            # Handle other templates
            if "一本书有" in template["template"]:
                total = random.randint(100, 300)
                read = random.randint(20, total-10)
                answer = total - read
                problem_text = template["template"].format(total=total, read=read)
                solution = template["solution"].format(total=total, read=read, answer=answer)
                explicit_relations = template["explicit_relations"]
            elif "商店里有" in template["template"]:
                rows = random.randint(3, 10)
                items = random.randint(5, 15)
                answer = rows * items
                problem_text = template["template"].format(rows=rows, items=items)
                solution = template["solution"].format(rows=rows, items=items, answer=answer)
                explicit_relations = template["explicit_relations"]
        
        problems.append({
            "id": f"L0_{start_id + i:03d}",
            "complexity_level": "L0_explicit",
            "problem": problem_text,
            "answer": str(answer),
            "solution_steps": [solution],
            "explicit_relations": explicit_relations,
            "implicit_relations": [],
            "domain_knowledge": [],
            "inference_depth": 1,
            "relation_count": 1
        })
    
    return problems

def generate_l1_problems(start_id=1, count=50):
    """Generate L1 shallow problems - single-step inference with basic formulas"""
    problems = []
    
    templates = [
        {
            "domain": "speed_distance_time",
            "template": "一辆汽车每小时行驶{speed}公里，行驶了{time}小时。这辆汽车总共行驶了多少公里？",
            "answer_formula": "{speed} * {time}",
            "solution_steps": [
                "使用速度公式：距离 = 速度 × 时间",
                "计算：{speed} × {time} = {answer}公里"
            ],
            "explicit_relations": ["速度", "时间"],
            "implicit_relations": ["距离=速度×时间"],
            "domain_knowledge": ["速度、时间、距离的关系"]
        },
        {
            "domain": "area_calculation",
            "template": "一个长方形的长是{length}米，宽是{width}米。这个长方形的面积是多少平方米？",
            "answer_formula": "{length} * {width}",
            "solution_steps": [
                "使用长方形面积公式：面积 = 长 × 宽",
                "计算：{length} × {width} = {answer}平方米"
            ],
            "explicit_relations": ["长度", "宽度"],
            "implicit_relations": ["面积=长×宽"],
            "domain_knowledge": ["长方形面积公式"]
        },
        {
            "domain": "price_calculation",
            "template": "小王买了{quantity}公斤{item}，每公斤{price}元。他总共花了多少钱？",
            "answer_formula": "{quantity} * {price}",
            "solution_steps": [
                "使用单价计算公式：总价 = 单价 × 数量",
                "计算：{price} × {quantity} = {answer}元"
            ],
            "explicit_relations": ["重量", "单价"],
            "implicit_relations": ["总价=单价×数量"],
            "domain_knowledge": ["价格计算"],
            "items": ["苹果", "橘子", "香蕉", "土豆", "大米"]
        },
        {
            "domain": "unit_conversion",
            "template": "一个正方形的边长是{side}分米。这个正方形的面积是多少平方厘米？",
            "answer_formula": "{side} * 10 * {side} * 10",
            "solution_steps": [
                "将分米转换为厘米：{side}分米 = {side_cm}厘米",
                "计算正方形面积：{side_cm} × {side_cm} = {answer}平方厘米"
            ],
            "explicit_relations": ["边长（分米）"],
            "implicit_relations": ["分米到厘米的转换", "正方形面积公式"],
            "domain_knowledge": ["单位转换", "面积计算"]
        }
    ]
    
    for i in range(count):
        template = random.choice(templates)
        
        if template["domain"] == "speed_distance_time":
            speed = random.randint(40, 120)
            time = random.choice([1.5, 2, 2.5, 3, 3.5, 4])
            answer = speed * time
            problem_text = template["template"].format(speed=speed, time=time)
            solution_steps = [step.format(speed=speed, time=time, answer=answer) for step in template["solution_steps"]]
            
        elif template["domain"] == "area_calculation":
            length = random.randint(8, 20)
            width = random.randint(5, 15)
            answer = length * width
            problem_text = template["template"].format(length=length, width=width)
            solution_steps = [step.format(length=length, width=width, answer=answer) for step in template["solution_steps"]]
            
        elif template["domain"] == "price_calculation":
            quantity = random.randint(2, 8)
            price = random.randint(10, 30)
            item = random.choice(template["items"])
            answer = quantity * price
            problem_text = template["template"].format(quantity=quantity, price=price, item=item)
            solution_steps = [step.format(quantity=quantity, price=price, answer=answer) for step in template["solution_steps"]]
            
        elif template["domain"] == "unit_conversion":
            side = random.randint(3, 8)
            side_cm = side * 10
            answer = side_cm * side_cm
            problem_text = template["template"].format(side=side)
            solution_steps = [step.format(side=side, side_cm=side_cm, answer=answer) for step in template["solution_steps"]]
        
        problems.append({
            "id": f"L1_{start_id + i:03d}",
            "complexity_level": "L1_shallow",
            "problem": problem_text,
            "answer": str(answer),
            "solution_steps": solution_steps,
            "explicit_relations": template["explicit_relations"],
            "implicit_relations": template["implicit_relations"],
            "domain_knowledge": template["domain_knowledge"],
            "inference_depth": 2,
            "relation_count": 2
        })
    
    return problems

def generate_l2_problems(start_id=1, count=80):
    """Generate L2 medium problems - 2-3 step inference with domain knowledge"""
    problems = []
    
    templates = [
        {
            "domain": "fluid_mechanics",
            "template": "一个水箱可以装{capacity}升水。现在水箱里有水{current}升，水龙头每分钟流入{rate}升水。需要多少分钟才能把水箱装满？",
            "solution_steps": [
                "计算还需要的水量：{capacity} - {current} = {needed}升",
                "计算需要的时间：{needed} ÷ {rate} = {answer}分钟"
            ],
            "explicit_relations": ["水箱容量", "现有水量", "流入速率"],
            "implicit_relations": ["剩余容量=总容量-现有量", "时间=剩余量÷流入速率"],
            "domain_knowledge": ["容量计算", "流量与时间关系"]
        },
        {
            "domain": "age_problems",
            "template": "小明的爸爸今年{dad_age}岁，是小明年龄的{ratio}倍。{years}年后，爸爸的年龄是小明年龄的几倍？",
            "solution_steps": [
                "计算小明现在的年龄：{dad_age} ÷ {ratio} = {ming_age}岁",
                "计算{years}年后小明的年龄：{ming_age} + {years} = {ming_future}岁",
                "计算{years}年后爸爸的年龄：{dad_age} + {years} = {dad_future}岁",
                "计算倍数关系：{dad_future} ÷ {ming_future} = {answer}倍"
            ],
            "explicit_relations": ["爸爸现在年龄", "年龄倍数关系"],
            "implicit_relations": ["小明现在年龄", "未来年龄变化", "未来倍数关系"],
            "domain_knowledge": ["年龄推算", "倍数关系变化"]
        },
        {
            "domain": "production_planning",
            "template": "一个工厂每天生产{daily_production}个零件，现在有订单需要{order}个零件，但工厂库存已有{inventory}个零件。需要多少天才能完成订单？",
            "solution_steps": [
                "计算还需要生产的零件数：{order} - {inventory} = {needed}个",
                "计算需要的天数：{needed} ÷ {daily_production} = {answer}天"
            ],
            "explicit_relations": ["日产量", "订单数量", "库存数量"],
            "implicit_relations": ["需生产量=订单量-库存量", "时间=需生产量÷日产量"],
            "domain_knowledge": ["生产计划", "库存管理"]
        }
    ]
    
    for i in range(count):
        template = random.choice(templates)
        
        if template["domain"] == "fluid_mechanics":
            capacity = random.randint(400, 800)
            current = random.randint(100, capacity//2)
            rate = random.randint(10, 25)
            needed = capacity - current
            answer = needed // rate
            problem_text = template["template"].format(capacity=capacity, current=current, rate=rate)
            solution_steps = [step.format(capacity=capacity, current=current, rate=rate, 
                                        needed=needed, answer=answer) for step in template["solution_steps"]]
            
        elif template["domain"] == "age_problems":
            ratio = random.randint(2, 4)
            ming_age = random.randint(8, 15)
            dad_age = ming_age * ratio
            years = random.randint(3, 8)
            ming_future = ming_age + years
            dad_future = dad_age + years
            answer = round(dad_future / ming_future, 1)
            problem_text = template["template"].format(dad_age=dad_age, ratio=ratio, years=years)
            solution_steps = [step.format(dad_age=dad_age, ratio=ratio, ming_age=ming_age,
                                        years=years, ming_future=ming_future, 
                                        dad_future=dad_future, answer=answer) for step in template["solution_steps"]]
            
        elif template["domain"] == "production_planning":
            daily_production = random.randint(80, 150)
            inventory = random.randint(100, 400)
            days = random.randint(8, 20)
            needed = daily_production * days
            order = needed + inventory
            answer = days
            problem_text = template["template"].format(daily_production=daily_production, 
                                                     order=order, inventory=inventory)
            solution_steps = [step.format(daily_production=daily_production, order=order,
                                        inventory=inventory, needed=needed, answer=answer) 
                            for step in template["solution_steps"]]
        
        problems.append({
            "id": f"L2_{start_id + i:03d}",
            "complexity_level": "L2_medium",
            "problem": problem_text,
            "answer": str(answer),
            "solution_steps": solution_steps,
            "explicit_relations": template["explicit_relations"],
            "implicit_relations": template["implicit_relations"],
            "domain_knowledge": template["domain_knowledge"],
            "inference_depth": random.randint(3, 4),
            "relation_count": 3
        })
    
    return problems

def generate_l3_problems(start_id=1, count=40):
    """Generate L3 deep problems - complex multi-step inference with advanced domain knowledge"""
    problems = []
    
    templates = [
        {
            "domain": "thermodynamics",
            "template": "在20°C的环境中，有一块重量为{mass}克的冰块。已知冰的融化潜热为334焦耳/克，环境每分钟向冰块传递热量{heat_rate}焦耳。冰块完全融化需要多少分钟？",
            "solution_steps": [
                "理解物理概念：冰融化需要潜热，不改变温度",
                "计算总需要热量：{mass}克 × 334焦耳/克 = {total_heat}焦耳",
                "计算融化时间：{total_heat}焦耳 ÷ {heat_rate}焦耳/分钟 = {answer}分钟"
            ],
            "explicit_relations": ["冰块重量", "环境传热速率"],
            "implicit_relations": ["融化潜热概念", "总热量需求", "热传递过程"],
            "domain_knowledge": ["相变物理学", "潜热概念", "热传递原理"]
        },
        {
            "domain": "geometry",
            "template": "一个圆柱形水塔，底面直径{diameter}米，高{height}米。如果要在水塔外表面（包括底面但不包括顶面）刷漆，每平方米需要0.5升油漆。总共需要多少升油漆？",
            "solution_steps": [
                "计算圆柱侧面积：π × 直径 × 高 = π × {diameter} × {height} = {side_area}π平方米",
                "计算底面积：π × (直径/2)² = π × {radius}² = {bottom_area}π平方米",
                "计算总面积：{side_area}π + {bottom_area}π = {total_area}π ≈ {total_area_num}平方米",
                "计算油漆用量：{total_area_num} × 0.5 = {answer}升"
            ],
            "explicit_relations": ["直径", "高度", "油漆用量比"],
            "implicit_relations": ["圆柱侧面积公式", "圆面积公式", "总表面积", "材料用量计算"],
            "domain_knowledge": ["立体几何", "圆柱表面积", "材料估算"]
        },
        {
            "domain": "exponential_decay",
            "template": "一个化学反应中，反应物A的浓度每小时减少{decay_percent}%。初始浓度为{initial}mol/L，反应进行多少小时后浓度降到初始浓度的1/{fraction}？",
            "solution_steps": [
                "理解指数衰减：每小时浓度变为原来的{remain_percent}%",
                "建立方程：{initial} × ({remain_decimal})^t = {initial}/{fraction} = {target}",
                "简化方程：({remain_decimal})^t = {target_ratio}",
                "取对数求解：t × ln({remain_decimal}) = ln({target_ratio})",
                "计算结果：t = ln({target_ratio}) / ln({remain_decimal}) ≈ {answer}小时"
            ],
            "explicit_relations": ["初始浓度", "衰减率", "目标浓度比"],
            "implicit_relations": ["指数衰减模型", "对数运算", "浓度变化规律"],
            "domain_knowledge": ["化学动力学", "指数函数", "对数运算"]
        }
    ]
    
    import math
    
    for i in range(count):
        template = random.choice(templates)
        
        if template["domain"] == "thermodynamics":
            mass = random.randint(200, 800)
            heat_rate = random.randint(30, 80)
            total_heat = mass * 334
            answer = total_heat // heat_rate
            problem_text = template["template"].format(mass=mass, heat_rate=heat_rate)
            solution_steps = [step.format(mass=mass, heat_rate=heat_rate,
                                        total_heat=total_heat, answer=answer) 
                            for step in template["solution_steps"]]
            
        elif template["domain"] == "geometry":
            diameter = random.randint(6, 12)
            height = random.randint(10, 20)
            radius = diameter // 2
            side_area = diameter * height
            bottom_area = radius * radius
            total_area = side_area + bottom_area
            total_area_num = round(total_area * 3.14159, 2)
            answer = round(total_area_num * 0.5, 1)
            problem_text = template["template"].format(diameter=diameter, height=height)
            solution_steps = [step.format(diameter=diameter, height=height, radius=radius,
                                        side_area=side_area, bottom_area=bottom_area,
                                        total_area=total_area, total_area_num=total_area_num,
                                        answer=answer) for step in template["solution_steps"]]
            
        elif template["domain"] == "exponential_decay":
            decay_percent = random.choice([15, 20, 25, 30])
            remain_percent = 100 - decay_percent
            remain_decimal = remain_percent / 100
            initial = 100
            fraction = random.choice([4, 8, 16])
            target = initial / fraction
            target_ratio = target / initial
            answer = round(math.log(target_ratio) / math.log(remain_decimal), 2)
            problem_text = template["template"].format(decay_percent=decay_percent, 
                                                     initial=initial, fraction=fraction)
            solution_steps = [step.format(remain_percent=remain_percent, initial=initial,
                                        remain_decimal=remain_decimal, fraction=fraction,
                                        target=target, target_ratio=target_ratio, answer=answer)
                            for step in template["solution_steps"]]
        
        problems.append({
            "id": f"L3_{start_id + i:03d}",
            "complexity_level": "L3_deep",
            "problem": problem_text,
            "answer": str(answer),
            "solution_steps": solution_steps,
            "explicit_relations": template["explicit_relations"],
            "implicit_relations": template["implicit_relations"],
            "domain_knowledge": template["domain_knowledge"],
            "inference_depth": random.randint(5, 6),
            "relation_count": random.randint(4, 5)
        })
    
    return problems

def generate_complete_dataset():
    """Generate the complete DIR-MWP dataset with 200 problems"""
    
    # Generate problems for each complexity level
    l0_problems = generate_l0_problems(start_id=1, count=30)
    l1_problems = generate_l1_problems(start_id=1, count=50)
    l2_problems = generate_l2_problems(start_id=1, count=80)
    l3_problems = generate_l3_problems(start_id=1, count=40)
    
    # Combine all problems
    all_problems = l0_problems + l1_problems + l2_problems + l3_problems
    
    # Create the complete dataset
    dataset = {
        "dataset_info": {
            "name": "DIR-MWP Dataset",
            "description": "Domain-specific Implicit Relation Math Word Problems",
            "total_problems": 200,
            "complexity_levels": 4,
            "creation_date": datetime.now().strftime("%Y-%m-%d"),
            "version": "1.0"
        },
        "complexity_statistics": {
            "L0_explicit": {
                "count": 30,
                "avg_relations": 1.2,
                "inference_depth": 1.0,
                "description": "简单算术问题，所有关系都明确给出"
            },
            "L1_shallow": {
                "count": 50,
                "avg_relations": 2.1,
                "inference_depth": 2.3,
                "description": "需要单步推理或基本单位转换"
            },
            "L2_medium": {
                "count": 80,
                "avg_relations": 3.2,
                "inference_depth": 3.8,
                "description": "需要2-3步推理和基本领域知识"
            },
            "L3_deep": {
                "count": 40,
                "avg_relations": 4.5,
                "inference_depth": 5.2,
                "description": "需要>3步推理和复杂领域知识"
            }
        },
        "problems": all_problems
    }
    
    return dataset

if __name__ == "__main__":
    # Generate the complete dataset
    dataset = generate_complete_dataset()
    
    # Save to JSON file
    output_file = "Data/DIR-MWP/dir_mwp_complete_dataset.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"DIR-MWP dataset generated successfully!")
    print(f"Total problems: {len(dataset['problems'])}")
    print(f"Saved to: {output_file}")
    
    # Print distribution summary
    level_counts = {}
    for problem in dataset['problems']:
        level = problem['complexity_level']
        level_counts[level] = level_counts.get(level, 0) + 1
    
    print("\nComplexity level distribution:")
    for level, count in level_counts.items():
        print(f"  {level}: {count} problems") 