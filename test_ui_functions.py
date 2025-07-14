#!/usr/bin/env python3
"""
Test the UI app with fixes
"""

import sys
import os
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the UI app functions
sys.path.insert(0, str(project_root / "src"))

# Import app module
ui_app_path = project_root / "ui" / "app.py"
spec = __import__('importlib.util').util.spec_from_file_location("ui_app", ui_app_path)
ui_app = __import__('importlib.util').util.module_from_spec(spec)
sys.modules["ui_app"] = ui_app
spec.loader.exec_module(ui_app)

def test_ui_functions():
    print("=== Testing UI Functions ===")
    
    # Test 1: Get project stats
    try:
        stats = ui_app.get_project_stats()
        print(f"✅ Project stats: {stats['python_files']} Python files, {stats['tests']} tests")
        print(f"  Modules: {[m['name'] for m in stats['modules'][:5]]}")
    except Exception as e:
        print(f"❌ Project stats failed: {e}")
    
    # Test 2: Get reasoning strategies
    try:
        strategies = ui_app.get_reasoning_strategies()
        print(f"✅ Found {len(strategies)} strategies:")
        for strategy in strategies:
            print(f"  - {strategy['name']}: {strategy['description'][:50]}...")
    except Exception as e:
        print(f"❌ Reasoning strategies failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ui_functions()