#!/usr/bin/env python3
"""
Test the UI API directly
"""

import requests
import json

def test_ui_api():
    base_url = "http://127.0.0.1:8081"
    
    print("=== Testing UI API ===")
    
    # Test 1: Main page
    try:
        response = requests.get(f"{base_url}/")
        print(f"✅ Main page status: {response.status_code}")
    except Exception as e:
        print(f"❌ Main page failed: {e}")
    
    # Test 2: Strategies API
    try:
        response = requests.get(f"{base_url}/api/strategies")
        if response.status_code == 200:
            strategies = response.json()
            print(f"✅ Strategies API returned {len(strategies)} strategies")
            for strategy in strategies:
                print(f"  - {strategy['name']}: {strategy['description'][:50]}...")
        else:
            print(f"❌ Strategies API failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ Strategies API failed: {e}")
    
    # Test 3: Docs API
    try:
        response = requests.get(f"{base_url}/docs")
        print(f"✅ Docs page status: {response.status_code}")
    except Exception as e:
        print(f"❌ Docs page failed: {e}")
    
    # Test 4: System API
    try:
        response = requests.get(f"{base_url}/api/system")
        if response.status_code == 200:
            system = response.json()
            print(f"✅ System API returned info with {len(system)} fields")
        else:
            print(f"❌ System API failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ System API failed: {e}")

if __name__ == "__main__":
    test_ui_api()