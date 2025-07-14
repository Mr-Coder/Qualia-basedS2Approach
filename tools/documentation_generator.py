"""
COT-DIR API文档生成器

自动生成项目的API文档、架构图和使用指南。
"""

import ast
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

# 文档生成配置
@dataclass
class DocumentationConfig:
    """文档生成配置"""
    project_root: Path
    output_dir: Path
    include_private: bool = False
    generate_architecture_diagram: bool = True
    generate_api_docs: bool = True
    generate_user_guide: bool = True
    include_examples: bool = True
    format: str = "markdown"  # markdown, html, json


class DocType(Enum):
    """文档类型枚举"""
    API_REFERENCE = "api_reference"
    ARCHITECTURE = "architecture"
    USER_GUIDE = "user_guide"
    EXAMPLES = "examples"
    CHANGELOG = "changelog"


class APIDocumentationGenerator:
    """API文档生成器"""
    
    def __init__(self, config: DocumentationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.modules_info = {}
        self.classes_info = {}
        self.functions_info = {}
        
    def generate_full_documentation(self) -> Dict[str, str]:
        """生成完整文档套件"""
        self.logger.info("开始生成COT-DIR项目文档...")
        
        # 确保输出目录存在
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_docs = {}
        
        # 1. 扫描项目结构
        self._scan_project_structure()
        
        # 2. 生成API参考文档
        if self.config.generate_api_docs:
            api_docs = self._generate_api_reference()
            generated_docs["api_reference"] = api_docs
            self._save_document("API_Reference.md", api_docs)
        
        # 3. 生成架构文档
        if self.config.generate_architecture_diagram:
            arch_docs = self._generate_architecture_documentation()
            generated_docs["architecture"] = arch_docs
            self._save_document("Architecture.md", arch_docs)
        
        # 4. 生成用户指南
        if self.config.generate_user_guide:
            user_guide = self._generate_user_guide()
            generated_docs["user_guide"] = user_guide
            self._save_document("User_Guide.md", user_guide)
        
        # 5. 生成示例文档
        if self.config.include_examples:
            examples = self._generate_examples_documentation()
            generated_docs["examples"] = examples
            self._save_document("Examples.md", examples)
        
        # 6. 生成总索引
        index = self._generate_documentation_index(generated_docs)
        generated_docs["index"] = index
        self._save_document("README.md", index)
        
        self.logger.info(f"文档生成完成，输出目录: {self.config.output_dir}")
        return generated_docs
    
    def _scan_project_structure(self):
        """扫描项目结构"""
        self.logger.info("扫描项目结构...")
        
        src_dir = self.config.project_root / "src"
        if not src_dir.exists():
            self.logger.warning(f"源码目录不存在: {src_dir}")
            return
        
        for py_file in src_dir.rglob("*.py"):
            if py_file.name.startswith("__") and py_file.name.endswith("__.py"):
                continue
                
            try:
                module_info = self._analyze_python_file(py_file)
                if module_info:
                    relative_path = py_file.relative_to(src_dir)
                    module_name = str(relative_path).replace("/", ".").replace(".py", "")
                    self.modules_info[module_name] = module_info
            except Exception as e:
                self.logger.warning(f"分析文件失败 {py_file}: {e}")
        
        self.logger.info(f"扫描完成，发现 {len(self.modules_info)} 个模块")
    
    def _analyze_python_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """分析Python文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            module_info = {
                "file_path": str(file_path),
                "docstring": ast.get_docstring(tree),
                "classes": [],
                "functions": [],
                "imports": [],
                "constants": []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class(node)
                    module_info["classes"].append(class_info)
                    self.classes_info[node.name] = class_info
                
                elif isinstance(node, ast.FunctionDef):
                    if not node.name.startswith("_") or self.config.include_private:
                        func_info = self._analyze_function(node)
                        module_info["functions"].append(func_info)
                        self.functions_info[node.name] = func_info
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_info = self._analyze_import(node)
                    module_info["imports"].append(import_info)
                
                elif isinstance(node, ast.Assign):
                    const_info = self._analyze_constant(node)
                    if const_info:
                        module_info["constants"].append(const_info)
            
            return module_info
            
        except Exception as e:
            self.logger.error(f"解析文件错误 {file_path}: {e}")
            return None
    
    def _analyze_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """分析类定义"""
        class_info = {
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "bases": [self._get_name(base) for base in node.bases],
            "methods": [],
            "attributes": [],
            "line_number": node.lineno
        }
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if not item.name.startswith("_") or self.config.include_private:
                    method_info = self._analyze_function(item, is_method=True)
                    class_info["methods"].append(method_info)
            elif isinstance(item, ast.Assign):
                attr_info = self._analyze_attribute(item)
                if attr_info:
                    class_info["attributes"].append(attr_info)
        
        return class_info
    
    def _analyze_function(self, node: ast.FunctionDef, is_method: bool = False) -> Dict[str, Any]:
        """分析函数定义"""
        func_info = {
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "args": [],
            "returns": None,
            "decorators": [],
            "line_number": node.lineno,
            "is_method": is_method,
            "is_async": isinstance(node, ast.AsyncFunctionDef)
        }
        
        # 分析参数
        for arg in node.args.args:
            arg_info = {
                "name": arg.arg,
                "annotation": self._get_annotation(arg.annotation) if arg.annotation else None
            }
            func_info["args"].append(arg_info)
        
        # 分析返回值注解
        if node.returns:
            func_info["returns"] = self._get_annotation(node.returns)
        
        # 分析装饰器
        for decorator in node.decorator_list:
            func_info["decorators"].append(self._get_name(decorator))
        
        return func_info
    
    def _analyze_import(self, node) -> Dict[str, Any]:
        """分析导入语句"""
        if isinstance(node, ast.Import):
            return {
                "type": "import",
                "names": [alias.name for alias in node.names]
            }
        elif isinstance(node, ast.ImportFrom):
            return {
                "type": "from_import",
                "module": node.module,
                "names": [alias.name for alias in node.names]
            }
        return {}
    
    def _analyze_constant(self, node: ast.Assign) -> Optional[Dict[str, Any]]:
        """分析常量定义"""
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.isupper():  # 通常常量是大写
                return {
                    "name": name,
                    "value": self._get_value_string(node.value),
                    "line_number": node.lineno
                }
        return None
    
    def _analyze_attribute(self, node: ast.Assign) -> Optional[Dict[str, Any]]:
        """分析类属性"""
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            return {
                "name": node.targets[0].id,
                "value": self._get_value_string(node.value),
                "line_number": node.lineno
            }
        return None
    
    def _get_name(self, node) -> str:
        """获取节点名称"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        else:
            return "Unknown"
    
    def _get_annotation(self, node) -> str:
        """获取类型注解"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return f"{self._get_name(node.value)}[{self._get_name(node.slice)}]"
        else:
            return "Any"
    
    def _get_value_string(self, node) -> str:
        """获取值的字符串表示"""
        if isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Name):
            return node.id
        else:
            return "..."
    
    def _generate_api_reference(self) -> str:
        """生成API参考文档"""
        self.logger.info("生成API参考文档...")
        
        doc = []
        doc.append("# COT-DIR API Reference")
        doc.append("")
        doc.append("This document provides a comprehensive reference for the COT-DIR API.")
        doc.append("")
        doc.append("## Table of Contents")
        doc.append("")
        
        # 生成目录
        for module_name in sorted(self.modules_info.keys()):
            doc.append(f"- [{module_name}](#{module_name.replace('.', '-')})")
        doc.append("")
        
        # 生成每个模块的文档
        for module_name, module_info in sorted(self.modules_info.items()):
            doc.append(f"## {module_name}")
            doc.append("")
            
            if module_info["docstring"]:
                doc.append(module_info["docstring"])
                doc.append("")
            
            # 类文档
            if module_info["classes"]:
                doc.append("### Classes")
                doc.append("")
                
                for class_info in module_info["classes"]:
                    doc.append(f"#### {class_info['name']}")
                    doc.append("")
                    
                    if class_info["docstring"]:
                        doc.append(class_info["docstring"])
                        doc.append("")
                    
                    # 继承关系
                    if class_info["bases"]:
                        doc.append(f"**Inherits from:** {', '.join(class_info['bases'])}")
                        doc.append("")
                    
                    # 方法文档
                    if class_info["methods"]:
                        doc.append("**Methods:**")
                        doc.append("")
                        
                        for method in class_info["methods"]:
                            doc.append(f"##### {method['name']}")
                            
                            # 方法签名
                            args_str = ", ".join([
                                f"{arg['name']}: {arg['annotation'] or 'Any'}" 
                                for arg in method['args']
                            ])
                            return_str = f" -> {method['returns']}" if method['returns'] else ""
                            
                            async_prefix = "async " if method['is_async'] else ""
                            doc.append(f"```python")
                            doc.append(f"{async_prefix}def {method['name']}({args_str}){return_str}")
                            doc.append("```")
                            doc.append("")
                            
                            if method["docstring"]:
                                doc.append(method["docstring"])
                                doc.append("")
                    
                    doc.append("---")
                    doc.append("")
            
            # 函数文档
            if module_info["functions"]:
                doc.append("### Functions")
                doc.append("")
                
                for func_info in module_info["functions"]:
                    doc.append(f"#### {func_info['name']}")
                    doc.append("")
                    
                    # 函数签名
                    args_str = ", ".join([
                        f"{arg['name']}: {arg['annotation'] or 'Any'}" 
                        for arg in func_info['args']
                    ])
                    return_str = f" -> {func_info['returns']}" if func_info['returns'] else ""
                    
                    async_prefix = "async " if func_info['is_async'] else ""
                    doc.append(f"```python")
                    doc.append(f"{async_prefix}def {func_info['name']}({args_str}){return_str}")
                    doc.append("```")
                    doc.append("")
                    
                    if func_info["docstring"]:
                        doc.append(func_info["docstring"])
                        doc.append("")
                    
                    doc.append("---")
                    doc.append("")
            
            doc.append("")
        
        return "\\n".join(doc)
    
    def _generate_architecture_documentation(self) -> str:
        """生成架构文档"""
        self.logger.info("生成架构文档...")
        
        doc = []
        doc.append("# COT-DIR Architecture Documentation")
        doc.append("")
        doc.append("This document describes the architecture and design patterns used in the COT-DIR system.")
        doc.append("")
        
        # 系统概览
        doc.append("## System Overview")
        doc.append("")
        doc.append("COT-DIR (Chain-of-Thought with Directional Implicit Reasoning) is a mathematical reasoning system")
        doc.append("that implements advanced reasoning capabilities through a modular, extensible architecture.")
        doc.append("")
        
        # 架构原则
        doc.append("## Architecture Principles")
        doc.append("")
        doc.append("1. **Modularity**: Clear separation of concerns across different modules")
        doc.append("2. **Extensibility**: Plugin-based architecture for easy feature addition")
        doc.append("3. **Performance**: Async/await patterns for high-throughput processing")
        doc.append("4. **Security**: Centralized security services and safe evaluation")
        doc.append("5. **Maintainability**: Consistent patterns and comprehensive documentation")
        doc.append("")
        
        # 核心组件
        doc.append("## Core Components")
        doc.append("")
        
        # 分析模块结构
        core_modules = {}
        for module_name, module_info in self.modules_info.items():
            parts = module_name.split(".")
            if len(parts) > 0:
                root_module = parts[0]
                if root_module not in core_modules:
                    core_modules[root_module] = []
                core_modules[root_module].append(module_name)
        
        for root_module, modules in sorted(core_modules.items()):
            doc.append(f"### {root_module.title()} Module")
            doc.append("")
            doc.append(f"The {root_module} module contains {len(modules)} submodules:")
            doc.append("")
            
            for module in sorted(modules):
                if module in self.modules_info:
                    module_info = self.modules_info[module]
                    docstring = module_info.get('docstring') or 'No description available'
                    doc.append(f"- **{module}**: {docstring[:100]}...")
            doc.append("")
        
        # 设计模式
        doc.append("## Design Patterns")
        doc.append("")
        doc.append("### Strategy Pattern")
        doc.append("Used in the orchestration system to support different coordination strategies:")
        doc.append("- UnifiedStrategy: General-purpose coordination")
        doc.append("- ReasoningStrategy: Specialized for reasoning tasks")
        doc.append("- ProcessingStrategy: Optimized for data processing")
        doc.append("")
        
        doc.append("### Template Method Pattern")
        doc.append("Implemented in the problem solving interface to ensure consistent processing:")
        doc.append("1. Input standardization")
        doc.append("2. Preprocessing")
        doc.append("3. Core solving")
        doc.append("4. Postprocessing")
        doc.append("5. Validation")
        doc.append("")
        
        doc.append("### Singleton Pattern")
        doc.append("Used for shared services like security evaluators to ensure resource efficiency.")
        doc.append("")
        
        # 数据流
        doc.append("## Data Flow")
        doc.append("")
        doc.append("```")
        doc.append("Input Problem → Standardization → Preprocessing → Core Solving → Postprocessing → Output")
        doc.append("                     ↓")
        doc.append("               Security Validation")
        doc.append("                     ↓")
        doc.append("               Error Recovery")
        doc.append("```")
        doc.append("")
        
        return "\\n".join(doc)
    
    def _generate_user_guide(self) -> str:
        """生成用户指南"""
        self.logger.info("生成用户指南...")
        
        doc = []
        doc.append("# COT-DIR User Guide")
        doc.append("")
        doc.append("Welcome to COT-DIR! This guide will help you get started with the system.")
        doc.append("")
        
        # 快速开始
        doc.append("## Quick Start")
        doc.append("")
        doc.append("### Installation")
        doc.append("")
        doc.append("```bash")
        doc.append("# Clone the repository")
        doc.append("git clone <repository-url>")
        doc.append("cd cot-dir")
        doc.append("")
        doc.append("# Install dependencies")
        doc.append("pip install -r requirements.txt")
        doc.append("```")
        doc.append("")
        
        # 基本使用
        doc.append("### Basic Usage")
        doc.append("")
        doc.append("```python")
        doc.append("from core import solve_problem_unified")
        doc.append("")
        doc.append("# Solve a mathematical problem")
        doc.append("result = solve_problem_unified(")
        doc.append('    "Calculate 2 + 3 * 4",')
        doc.append('    strategy="chain_of_thought"')
        doc.append(")")
        doc.append("")
        doc.append('print(f"Answer: {result[\'final_answer\']}")')
        doc.append('print(f"Confidence: {result[\'confidence\']}")')
        doc.append("```")
        doc.append("")
        
        # 高级功能
        doc.append("## Advanced Features")
        doc.append("")
        doc.append("### Batch Processing")
        doc.append("")
        doc.append("```python")
        doc.append("from core import create_problem_solver")
        doc.append("")
        doc.append("solver = create_problem_solver('chain_of_thought')")
        doc.append("problems = [")
        doc.append('    "What is 5 + 7?",')
        doc.append('    "Solve x + 3 = 10",')
        doc.append('    "Calculate the area of a circle with radius 5"')
        doc.append("]")
        doc.append("")
        doc.append("results = solver.batch_solve(problems)")
        doc.append("for i, result in enumerate(results):")
        doc.append('    print(f"Problem {i+1}: {result.final_answer}")')
        doc.append("```")
        doc.append("")
        
        # 配置选项
        doc.append("### Configuration")
        doc.append("")
        doc.append("You can customize the system behavior through configuration:")
        doc.append("")
        doc.append("```python")
        doc.append("config = {")
        doc.append('    "max_workers": 8,')
        doc.append('    "timeout": 30,')
        doc.append('    "enable_caching": True,')
        doc.append('    "security_level": "high"')
        doc.append("}")
        doc.append("")
        doc.append("solver = create_problem_solver('direct_reasoning', config)")
        doc.append("```")
        doc.append("")
        
        return "\\n".join(doc)
    
    def _generate_examples_documentation(self) -> str:
        """生成示例文档"""
        self.logger.info("生成示例文档...")
        
        doc = []
        doc.append("# COT-DIR Examples")
        doc.append("")
        doc.append("This document contains practical examples of using the COT-DIR system.")
        doc.append("")
        
        # 基础示例
        doc.append("## Basic Examples")
        doc.append("")
        
        examples = [
            {
                "title": "Simple Arithmetic",
                "description": "Solving basic mathematical operations",
                "code": '''
from core import solve_problem_unified

# Simple addition
result = solve_problem_unified("What is 15 + 27?")
print(f"Answer: {result['final_answer']}")

# With chain of thought
result = solve_problem_unified(
    "Calculate 12 * 8 + 5",
    strategy="chain_of_thought"
)
print(f"Reasoning steps: {result['reasoning_steps']}")
'''
            },
            {
                "title": "Word Problems",
                "description": "Solving mathematical word problems",
                "code": '''
from core import create_problem_solver

solver = create_problem_solver("chain_of_thought")

problem = """
A store sells apples for $2 each and oranges for $3 each.
If someone buys 5 apples and 3 oranges, how much do they pay in total?
"""

result = solver.solve_problem(problem)
print(f"Solution: {result.final_answer}")
print(f"Confidence: {result.confidence:.2f}")
'''
            },
            {
                "title": "Batch Processing",
                "description": "Processing multiple problems efficiently",
                "code": '''
from core import create_problem_solver

solver = create_problem_solver("direct_reasoning")

math_problems = [
    "Find the square root of 144",
    "What is 25% of 80?",
    "Solve for x: 2x + 5 = 13",
    "Calculate the perimeter of a rectangle with length 8 and width 5"
]

results = solver.batch_solve(math_problems)

for i, result in enumerate(results):
    print(f"Problem {i+1}: {result.final_answer}")
    print(f"Success: {result.success}")
    print("---")
'''
            }
        ]
        
        for example in examples:
            doc.append(f"### {example['title']}")
            doc.append("")
            doc.append(example['description'])
            doc.append("")
            doc.append("```python")
            doc.append(example['code'].strip())
            doc.append("```")
            doc.append("")
        
        return "\\n".join(doc)
    
    def _generate_documentation_index(self, generated_docs: Dict[str, str]) -> str:
        """生成文档索引"""
        doc = []
        doc.append("# COT-DIR Documentation")
        doc.append("")
        doc.append("Complete documentation for the COT-DIR (Chain-of-Thought with Directional Implicit Reasoning) system.")
        doc.append("")
        
        # 项目概述
        doc.append("## Project Overview")
        doc.append("")
        doc.append("COT-DIR is an advanced mathematical reasoning system that implements")
        doc.append("sophisticated problem-solving capabilities through modular architecture,")
        doc.append("async processing, and comprehensive security measures.")
        doc.append("")
        
        # 关键特性
        doc.append("## Key Features")
        doc.append("")
        doc.append("- 🧠 **Advanced Reasoning**: Chain-of-thought and direct reasoning strategies")
        doc.append("- ⚡ **High Performance**: Async/await patterns for concurrent processing")
        doc.append("- 🔒 **Security First**: Centralized security services and safe evaluation")
        doc.append("- 🔧 **Modular Design**: Plugin-based architecture for easy extension")
        doc.append("- 📊 **Comprehensive Monitoring**: Built-in performance tracking and logging")
        doc.append("")
        
        # 文档导航
        doc.append("## Documentation")
        doc.append("")
        
        if "api_reference" in generated_docs:
            doc.append("- **[API Reference](API_Reference.md)**: Complete API documentation")
        if "architecture" in generated_docs:
            doc.append("- **[Architecture](Architecture.md)**: System design and patterns")
        if "user_guide" in generated_docs:
            doc.append("- **[User Guide](User_Guide.md)**: Getting started and usage examples")
        if "examples" in generated_docs:
            doc.append("- **[Examples](Examples.md)**: Practical code examples")
        
        doc.append("")
        
        # 快速开始
        doc.append("## Quick Start")
        doc.append("")
        doc.append("```python")
        doc.append("from core import solve_problem_unified")
        doc.append("")
        doc.append('result = solve_problem_unified("Calculate 2 + 3 * 4")')
        doc.append('print(result["final_answer"])')
        doc.append("```")
        doc.append("")
        
        # 项目统计
        doc.append("## Project Statistics")
        doc.append("")
        doc.append(f"- **Modules**: {len(self.modules_info)}")
        doc.append(f"- **Classes**: {len(self.classes_info)}")
        doc.append(f"- **Functions**: {len(self.functions_info)}")
        doc.append("")
        doc.append("---")
        doc.append("")
        doc.append("*Documentation generated automatically by COT-DIR Documentation Generator*")
        
        return "\\n".join(doc)
    
    def _save_document(self, filename: str, content: str):
        """保存文档到文件"""
        file_path = self.config.output_dir / filename
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.logger.info(f"文档已保存: {file_path}")
        except Exception as e:
            self.logger.error(f"保存文档失败 {file_path}: {e}")


def generate_project_documentation(
    project_root: str = None,
    output_dir: str = None,
    include_private: bool = False
) -> Dict[str, str]:
    """生成项目文档的便利函数"""
    
    if project_root is None:
        project_root = Path(__file__).parent.parent
    else:
        project_root = Path(project_root)
    
    if output_dir is None:
        output_dir = project_root / "docs" / "generated"
    else:
        output_dir = Path(output_dir)
    
    config = DocumentationConfig(
        project_root=project_root,
        output_dir=output_dir,
        include_private=include_private,
        generate_architecture_diagram=True,
        generate_api_docs=True,
        generate_user_guide=True,
        include_examples=True
    )
    
    generator = APIDocumentationGenerator(config)
    return generator.generate_full_documentation()


if __name__ == "__main__":
    # 示例用法
    docs = generate_project_documentation()
    print("文档生成完成！")
    for doc_type, content in docs.items():
        print(f"- {doc_type}: {len(content)} characters")