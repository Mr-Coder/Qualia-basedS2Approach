#!/usr/bin/env python3
"""
Test script to identify UI issues
"""

import sys
import os
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")

# Test 1: Check if strategy files exist
print("\n=== Test 1: Strategy Files ===")
strategy_dir = project_root / "src" / "reasoning" / "strategy_manager"
print(f"Strategy directory: {strategy_dir}")
print(f"Directory exists: {strategy_dir.exists()}")

if strategy_dir.exists():
    strategy_files = list(strategy_dir.glob("*_strategy.py"))
    print(f"Strategy files found: {[f.name for f in strategy_files]}")
else:
    print("Strategy directory not found!")

# Test 2: Try importing strategy manager
print("\n=== Test 2: Import Strategy Manager ===")
try:
    from src.reasoning.strategy_manager.strategy_manager import StrategyManager
    print("✅ StrategyManager imported successfully")
except Exception as e:
    print(f"❌ StrategyManager import failed: {e}")

# Test 3: Try importing individual strategies
print("\n=== Test 3: Import Individual Strategies ===")
try:
    from src.reasoning.strategy_manager.cot_strategy import ChainOfThoughtStrategy
    print("✅ ChainOfThoughtStrategy imported successfully")
except Exception as e:
    print(f"❌ ChainOfThoughtStrategy import failed: {e}")

try:
    from src.reasoning.strategy_manager.got_strategy import GraphOfThoughtsStrategy
    print("✅ GraphOfThoughtsStrategy imported successfully")
except Exception as e:
    print(f"❌ GraphOfThoughtsStrategy import failed: {e}")

try:
    from src.reasoning.strategy_manager.tot_strategy import TreeOfThoughtsStrategy
    print("✅ TreeOfThoughtsStrategy imported successfully")
except Exception as e:
    print(f"❌ TreeOfThoughtsStrategy import failed: {e}")

# Test 4: Check if ui/app.py can load strategies
print("\n=== Test 4: UI App Strategy Loading ===")
try:
    # Simulate the UI app's strategy loading
    SRC_DIR = project_root / "src"
    strategy_dir = SRC_DIR / "reasoning" / "strategy_manager"
    
    if strategy_dir.exists():
        strategies = []
        for strategy_file in strategy_dir.glob("*_strategy.py"):
            try:
                with open(strategy_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Simple parsing like in UI
                name = strategy_file.stem.replace('_strategy', '').title()
                strategies.append({
                    "name": name,
                    "file": strategy_file.name,
                    "size": strategy_file.stat().st_size
                })
            except Exception as e:
                print(f"Error reading {strategy_file}: {e}")
        
        print(f"Strategies found by UI logic: {len(strategies)}")
        for strategy in strategies:
            print(f"  - {strategy['name']} ({strategy['file']})")
    else:
        print("Strategy directory not found by UI logic!")
        
except Exception as e:
    print(f"❌ UI strategy loading failed: {e}")

# Test 5: Check docs directory
print("\n=== Test 5: Docs Directory ===")
docs_dir = project_root / "docs" / "generated"
print(f"Docs directory: {docs_dir}")
print(f"Directory exists: {docs_dir.exists()}")

if docs_dir.exists():
    doc_files = list(docs_dir.glob("*.md"))
    print(f"Doc files found: {[f.name for f in doc_files]}")
else:
    print("Docs directory not found!")

# Test 6: Try running a simple strategy
print("\n=== Test 6: Strategy Execution Test ===")
try:
    from src.reasoning.strategy_manager.cot_strategy import ChainOfThoughtStrategy
    
    strategy = ChainOfThoughtStrategy()
    test_problem = "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？"
    
    can_handle = strategy.can_handle(test_problem)
    print(f"Strategy can handle test problem: {can_handle}")
    
    if can_handle:
        complexity = strategy.estimate_complexity(test_problem)
        print(f"Problem complexity: {complexity}")
    
except Exception as e:
    print(f"❌ Strategy execution test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test Complete ===")