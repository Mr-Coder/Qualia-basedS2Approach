#!/usr/bin/env python3
"""
Import Path Fixer Script
Fix all import paths to use proper relative imports
"""

import os
import re
import sys
from pathlib import Path

def fix_imports_in_file(file_path):
    """Fix imports in a single file"""
    print(f"Processing: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix patterns - only if not already using relative imports
    patterns = [
        # Main source files
        (r'from models\.', 'from ..models.'),  # models.xxx -> ..models.xxx
        (r'from processors\.', 'from ..processors.'),  # processors.xxx -> ..processors.xxx
        
        # Test files - need different pattern
        (r'from models\.structures import ProcessedText', 'from src.models.structures import ProcessedText'),
        (r'from models\.relation import', 'from src.models.relation import'),
        (r'from processors\.', 'from src.processors.'),
    ]
    
    # Determine if this is a test file or source file
    is_test_file = 'test' in str(file_path)
    is_source_file = 'src/' in str(file_path)
    
    if is_test_file:
        # For test files, use absolute imports from src
        test_patterns = [
            (r'from models\.structures import ProcessedText', 'from src.models.structures import ProcessedText'),
            (r'from models\.structures import', 'from src.models.structures import'),
            (r'from models\.relation import', 'from src.models.relation import'),
            (r'from processors\.(\w+)', r'from src.processors.\1'),
        ]
        
        for old_pattern, new_pattern in test_patterns:
            content = re.sub(old_pattern, new_pattern, content)
            
    elif is_source_file:
        # For source files, use relative imports
        src_patterns = [
            (r'(?<!from \.)from models\.', 'from ..models.'),  # models.xxx -> ..models.xxx (not already relative)
            (r'(?<!from \.)from processors\.', 'from ..processors.'),  # processors.xxx -> ..processors.xxx
        ]
        
        for old_pattern, new_pattern in src_patterns:
            content = re.sub(old_pattern, new_pattern, content)
    
    # Write back if changed
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Fixed imports in {file_path}")
        return True
    else:
        print(f"ℹ️  No changes needed in {file_path}")
        return False

def find_python_files_with_imports(root_dir):
    """Find all Python files that contain problematic imports"""
    problematic_files = []
    
    for root, dirs, files in os.walk(root_dir):
        # Skip certain directories
        skip_dirs = ['.git', '__pycache__', '.pytest_cache', 'node_modules', 'archive', 'backups']
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for problematic import patterns
                    problematic_patterns = [
                        r'from models\.',
                        r'from processors\.',
                    ]
                    
                    for pattern in problematic_patterns:
                        if re.search(pattern, content):
                            problematic_files.append(file_path)
                            break
                            
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")
    
    return problematic_files

def main():
    """Main function"""
    print("🔧 Import Path Fixer")
    print("=" * 50)
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir
    
    print(f"Project root: {project_root}")
    
    # Find files with problematic imports
    print("\n📁 Finding files with problematic imports...")
    problematic_files = find_python_files_with_imports(project_root)
    
    if not problematic_files:
        print("✅ No files with problematic imports found!")
        return
    
    print(f"\n🔍 Found {len(problematic_files)} files with problematic imports:")
    for file_path in problematic_files:
        print(f"  - {file_path}")
    
    # Fix imports
    print(f"\n🛠️  Fixing imports...")
    fixed_count = 0
    
    for file_path in problematic_files:
        try:
            if fix_imports_in_file(file_path):
                fixed_count += 1
        except Exception as e:
            print(f"❌ Error fixing {file_path}: {e}")
    
    print(f"\n📊 Summary:")
    print(f"  Total files processed: {len(problematic_files)}")
    print(f"  Files fixed: {fixed_count}")
    print(f"  No changes needed: {len(problematic_files) - fixed_count}")
    
    if fixed_count > 0:
        print(f"\n✅ Import paths have been fixed!")
        print(f"💡 You may need to restart your Python interpreter/IDE")
    else:
        print(f"\nℹ️  All import paths were already correct")

if __name__ == "__main__":
    main()