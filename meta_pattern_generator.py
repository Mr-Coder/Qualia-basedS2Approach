#!/usr/bin/env python3
"""
元模式生成器 - 从元模式定义生成具体的推理模式
"""

import json
import itertools
from typing import List, Dict, Any

def generate_concrete_patterns(meta_pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从元模式生成具体模式"""
    concrete_patterns = []
    
    # 获取模式模板
    template = meta_pattern.get('pattern_template', {})
    parameters = meta_pattern.get('parameters', [])
    
    # 如果没有参数，生成基础模式
    if not parameters:
        pattern = {
            'id': f"{meta_pattern['id']}_basic",
            'name': f"{meta_pattern['name']}_基础版",
            'description': meta_pattern['description'],
            'trigger': template.get('trigger', []),
            'steps': template.get('steps', []),
            'validation': template.get('validation', [])
        }
        concrete_patterns.append(pattern)
        return concrete_patterns
    
    # 生成参数组合
    param_keys = []
    param_values = []
    
    for param in parameters:
        for key, value in param.items():
            if key not in param_keys:
                param_keys.append(key)
                param_values.append([])
            
            idx = param_keys.index(key)
            if value not in param_values[idx]:
                param_values[idx].append(value)
    
    # 生成所有组合
    if param_values:
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_keys, combination))
            
            # 创建具体模式
            pattern_id = f"{meta_pattern['id']}_{'_'.join(str(v) for v in combination)}"
            pattern_name = f"{meta_pattern['name']}_{'-'.join(str(v) for v in combination)}"
            
            pattern = {
                'id': pattern_id.replace(' ', '_').replace('/', '_'),
                'name': pattern_name,
                'description': f"{meta_pattern['description']} - {param_dict}",
                'parameters': param_dict,
                'trigger': template.get('trigger', []),
                'steps': template.get('steps', []),
                'validation': template.get('validation', [])
            }
            concrete_patterns.append(pattern)
    
    return concrete_patterns

def main():
    """主函数"""
    print("🧩 Meta Pattern Generator")
    print("=" * 50)
    
    try:
        # 加载元模式
        with open('meta_patterns.json', 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
        
        print(f"📥 Loaded {len(meta_data['meta_patterns'])} meta patterns")
        
        # 生成具体模式
        all_concrete_patterns = []
        
        for meta in meta_data['meta_patterns']:
            print(f"\n🔄 Processing: {meta['name']}")
            concrete_patterns = generate_concrete_patterns(meta)
            all_concrete_patterns.extend(concrete_patterns)
            print(f"   Generated {len(concrete_patterns)} concrete patterns")
        
        # 保存结果
        output = {
            'generated_patterns': all_concrete_patterns,
            'total_count': len(all_concrete_patterns),
            'meta_source_count': len(meta_data['meta_patterns'])
        }
        
        with open('generated_concrete_patterns.json', 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Generated {len(all_concrete_patterns)} concrete patterns")
        print("📁 Saved to: generated_concrete_patterns.json")
        
        # 显示示例
        if all_concrete_patterns:
            print(f"\n📋 Sample pattern:")
            sample = all_concrete_patterns[0]
            print(f"   ID: {sample['id']}")
            print(f"   Name: {sample['name']}")
            print(f"   Description: {sample['description']}")
            if 'parameters' in sample:
                print(f"   Parameters: {sample['parameters']}")
        
    except FileNotFoundError:
        print("❌ Error: meta_patterns.json not found")
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
