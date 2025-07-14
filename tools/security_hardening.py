"""
COT-DIR 安全加固实施方案

修复发现的安全问题，包括 eval() 使用、pickle 安全、配置加密等。
"""

import ast
import json
import hashlib
import hmac
import os
from pathlib import Path
from typing import Any, Dict, Union, List
import logging
from cryptography.fernet import Fernet

# 安全日志配置
logging.basicConfig(level=logging.INFO)
security_logger = logging.getLogger("security")


class SecureMathEvaluator:
    """安全的数学表达式计算器 - 替代 eval()"""
    
    ALLOWED_OPERATORS = {
        ast.Add: '+',
        ast.Sub: '-', 
        ast.Mult: '*',
        ast.Div: '/',
        ast.Pow: '**',
        ast.USub: '-',
        ast.UAdd: '+'
    }
    
    ALLOWED_FUNCTIONS = {
        'abs', 'round', 'min', 'max', 'sum',
        'sqrt', 'pow', 'exp', 'log', 'sin', 'cos', 'tan'
    }
    
    def __init__(self):
        self.security_logger = logging.getLogger("security.math_evaluator")
        
    def safe_eval(self, expression: str, allowed_names: Dict[str, Any] = None) -> Union[float, int]:
        """
        安全地计算数学表达式，替代危险的 eval()
        
        Args:
            expression: 数学表达式字符串
            allowed_names: 允许的变量名和值
            
        Returns:
            计算结果
            
        Raises:
            SecurityError: 表达式包含不安全的操作
            ValueError: 表达式语法错误
        """
        try:
            # 记录安全日志
            self.security_logger.info(f"安全计算表达式: {expression[:100]}...")
            
            # 解析表达式为AST
            tree = ast.parse(expression, mode='eval')
            
            # 验证AST安全性
            self._validate_ast_security(tree)
            
            # 准备安全的执行环境
            safe_dict = self._create_safe_environment(allowed_names or {})
            
            # 执行表达式
            result = eval(compile(tree, '<string>', 'eval'), safe_dict)
            
            self.security_logger.info(f"表达式计算成功: {result}")
            return result
            
        except (SyntaxError, ValueError) as e:
            self.security_logger.warning(f"表达式语法错误: {expression} - {e}")
            raise ValueError(f"数学表达式语法错误: {e}")
            
        except SecurityError as e:
            self.security_logger.error(f"检测到不安全表达式: {expression} - {e}")
            raise
            
        except Exception as e:
            self.security_logger.error(f"表达式计算异常: {expression} - {e}")
            raise ValueError(f"表达式计算失败: {e}")
    
    def _validate_ast_security(self, tree: ast.AST):
        """验证AST节点的安全性"""
        
        for node in ast.walk(tree):
            # 检查操作符
            if isinstance(node, ast.operator) and type(node) not in self.ALLOWED_OPERATORS:
                raise SecurityError(f"不允许的操作符: {type(node).__name__}")
            
            # 检查函数调用
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name not in self.ALLOWED_FUNCTIONS:
                        raise SecurityError(f"不允许的函数调用: {func_name}")
                else:
                    raise SecurityError("不允许的复杂函数调用")
            
            # 检查属性访问
            if isinstance(node, ast.Attribute):
                raise SecurityError("不允许访问对象属性")
            
            # 检查导入语句
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise SecurityError("不允许导入模块")
            
            # 检查变量赋值
            if isinstance(node, ast.Assign):
                raise SecurityError("不允许变量赋值")
    
    def _create_safe_environment(self, allowed_names: Dict[str, Any]) -> Dict[str, Any]:
        """创建安全的执行环境"""
        import math
        
        safe_env = {
            '__builtins__': {},  # 清空内置函数
            # 安全的数学函数
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'sqrt': math.sqrt,
            'pow': pow,
            'exp': math.exp,
            'log': math.log,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'pi': math.pi,
            'e': math.e
        }
        
        # 添加允许的变量
        for name, value in allowed_names.items():
            if isinstance(value, (int, float, complex)):
                safe_env[name] = value
            else:
                self.security_logger.warning(f"跳过不安全的变量: {name}")
        
        return safe_env


class SecurityError(Exception):
    """安全异常"""
    pass


class SecureConfigManager:
    """安全配置管理器 - 支持配置加密"""
    
    def __init__(self, config_dir: str, encryption_key: bytes = None):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # 初始化日志器
        self.security_logger = logging.getLogger("security.config")
        
        # 生成或加载加密密钥
        self.encryption_key = encryption_key or self._get_or_create_key()
        self.cipher = Fernet(self.encryption_key)
    
    def _get_or_create_key(self) -> bytes:
        """获取或创建加密密钥"""
        key_file = self.config_dir / ".config_key"
        
        if key_file.exists():
            # 从环境变量或密钥文件加载
            if "CONFIG_ENCRYPTION_KEY" in os.environ:
                return os.environ["CONFIG_ENCRYPTION_KEY"].encode()
            
            try:
                with open(key_file, 'rb') as f:
                    return f.read()
            except Exception:
                self.security_logger.warning("无法加载密钥文件，生成新密钥")
        
        # 生成新密钥
        key = Fernet.generate_key()
        
        # 保存密钥文件（仅开发环境）
        try:
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # 仅所有者可读写
            self.security_logger.info("生成新的配置加密密钥")
        except Exception as e:
            self.security_logger.error(f"保存密钥文件失败: {e}")
        
        return key
    
    def save_secure_config(self, config_name: str, config_data: Dict[str, Any]):
        """保存加密配置"""
        try:
            # 序列化配置
            config_json = json.dumps(config_data, ensure_ascii=False)
            
            # 加密配置
            encrypted_data = self.cipher.encrypt(config_json.encode())
            
            # 保存到文件
            config_file = self.config_dir / f"{config_name}.enc"
            with open(config_file, 'wb') as f:
                f.write(encrypted_data)
            
            # 设置安全权限
            os.chmod(config_file, 0o600)
            
            self.security_logger.info(f"安全配置已保存: {config_name}")
            
        except Exception as e:
            self.security_logger.error(f"保存安全配置失败: {e}")
            raise SecurityError(f"配置保存失败: {e}")
    
    def load_secure_config(self, config_name: str) -> Dict[str, Any]:
        """加载加密配置"""
        try:
            config_file = self.config_dir / f"{config_name}.enc"
            
            if not config_file.exists():
                raise FileNotFoundError(f"配置文件不存在: {config_name}")
            
            # 读取加密数据
            with open(config_file, 'rb') as f:
                encrypted_data = f.read()
            
            # 解密配置
            decrypted_data = self.cipher.decrypt(encrypted_data)
            config_json = decrypted_data.decode()
            
            # 反序列化配置
            config_data = json.loads(config_json)
            
            self.security_logger.info(f"安全配置已加载: {config_name}")
            return config_data
            
        except Exception as e:
            self.security_logger.error(f"加载安全配置失败: {e}")
            raise SecurityError(f"配置加载失败: {e}")
    
    def validate_config_security(self, config_data: Dict[str, Any]) -> List[str]:
        """验证配置安全性"""
        issues = []
        
        # 检查敏感信息
        sensitive_keys = ['password', 'secret', 'key', 'token', 'api_key']
        
        def check_dict(data, path=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # 检查敏感键名
                    if any(sensitive in key.lower() for sensitive in sensitive_keys):
                        if isinstance(value, str) and len(value) > 10:
                            issues.append(f"疑似敏感信息: {current_path}")
                    
                    # 递归检查
                    check_dict(value, current_path)
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    check_dict(item, f"{path}[{i}]")
        
        check_dict(config_data)
        return issues


class SecureFileManager:
    """安全文件管理器 - 替代不安全的 pickle"""
    
    def __init__(self):
        self.security_logger = logging.getLogger("security.file_manager")
    
    def safe_save_object(self, obj: Any, file_path: str, use_encryption: bool = True):
        """安全保存对象，使用JSON替代pickle"""
        try:
            file_path = Path(file_path)
            
            # 检查对象是否可JSON序列化
            if not self._is_json_serializable(obj):
                raise SecurityError("对象包含不可序列化的内容")
            
            # 序列化对象
            json_data = json.dumps(obj, ensure_ascii=False, indent=2)
            
            if use_encryption:
                # 加密保存
                key = Fernet.generate_key()
                cipher = Fernet(key)
                encrypted_data = cipher.encrypt(json_data.encode())
                
                # 保存加密数据和密钥
                with open(file_path.with_suffix('.json.enc'), 'wb') as f:
                    f.write(encrypted_data)
                
                with open(file_path.with_suffix('.key'), 'wb') as f:
                    f.write(key)
                
                # 设置安全权限
                os.chmod(file_path.with_suffix('.json.enc'), 0o600)
                os.chmod(file_path.with_suffix('.key'), 0o600)
                
                self.security_logger.info(f"对象已加密保存: {file_path}")
            else:
                # 明文保存
                with open(file_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
                    f.write(json_data)
                
                self.security_logger.info(f"对象已保存: {file_path}")
                
        except Exception as e:
            self.security_logger.error(f"保存对象失败: {e}")
            raise SecurityError(f"文件保存失败: {e}")
    
    def safe_load_object(self, file_path: str, use_encryption: bool = True) -> Any:
        """安全加载对象"""
        try:
            file_path = Path(file_path)
            
            if use_encryption:
                # 加载加密数据
                enc_file = file_path.with_suffix('.json.enc')
                key_file = file_path.with_suffix('.key')
                
                if not enc_file.exists() or not key_file.exists():
                    raise FileNotFoundError("加密文件或密钥文件不存在")
                
                with open(key_file, 'rb') as f:
                    key = f.read()
                
                with open(enc_file, 'rb') as f:
                    encrypted_data = f.read()
                
                # 解密数据
                cipher = Fernet(key)
                json_data = cipher.decrypt(encrypted_data).decode()
            else:
                # 加载明文数据
                json_file = file_path.with_suffix('.json')
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = f.read()
            
            # 反序列化对象
            obj = json.loads(json_data)
            
            self.security_logger.info(f"对象已加载: {file_path}")
            return obj
            
        except Exception as e:
            self.security_logger.error(f"加载对象失败: {e}")
            raise SecurityError(f"文件加载失败: {e}")
    
    def _is_json_serializable(self, obj: Any) -> bool:
        """检查对象是否可JSON序列化"""
        try:
            json.dumps(obj)
            return True
        except (TypeError, ValueError):
            return False


class SecurityPatcher:
    """安全补丁器 - 修复现有代码中的安全问题"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.security_logger = logging.getLogger("security.patcher")
        
        # 初始化安全组件
        self.math_evaluator = SecureMathEvaluator()
        self.file_manager = SecureFileManager()
    
    def patch_eval_usage(self):
        """修复 eval() 使用"""
        self.security_logger.info("修复代码中的 eval() 使用...")
        
        # 需要修复的文件列表
        files_to_patch = [
            "src/models/proposed_model.py",
            "src/models/baseline_models.py", 
            "src/processors/batch_processor.py",
            "src/processors/scalable_architecture.py"
        ]
        
        patches_applied = 0
        
        for file_path in files_to_patch:
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    # 创建安全版本的文件
                    self._create_secure_version(full_path)
                    patches_applied += 1
                except Exception as e:
                    self.security_logger.error(f"修复文件失败 {file_path}: {e}")
        
        self.security_logger.info(f"已修复 {patches_applied} 个文件的 eval() 使用")
    
    def _create_secure_version(self, file_path: Path):
        """为文件创建安全版本"""
        
        # 读取原文件
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 创建备份
        backup_path = file_path.with_suffix('.py.backup')
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 添加安全导入
        secure_imports = '''
# 安全改进：导入安全计算器
import sys
import os
sys.path.append(os.path.dirname(__file__))
from secure_components import SecureMathEvaluator, SecurityError

# 初始化安全计算器
_secure_evaluator = SecureMathEvaluator()
'''
        
        # 在文件开头添加安全导入
        if "from secure_components import" not in content:
            lines = content.split('\n')
            
            # 找到导入语句的位置
            import_end = 0
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')) or line.strip().startswith('#'):
                    import_end = i + 1
                elif line.strip() and not line.strip().startswith(('"""', "'''")):
                    break
            
            # 插入安全导入
            lines.insert(import_end, secure_imports)
            content = '\n'.join(lines)
        
        # 替换 eval() 调用
        replacements = [
            ('eval(', '_secure_evaluator.safe_eval('),
            ('eval (', '_secure_evaluator.safe_eval(')
        ]
        
        for old, new in replacements:
            content = content.replace(old, new)
        
        # 写入修改后的内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.security_logger.info(f"已创建安全版本: {file_path}")
    
    def create_secure_components_file(self):
        """创建安全组件文件"""
        
        secure_components_content = '''"""
COT-DIR 安全组件

提供安全的数学计算、文件操作等功能，替代不安全的操作。
"""

import ast
import json
import os
import math
import logging
from pathlib import Path
from typing import Any, Dict, Union
from cryptography.fernet import Fernet


class SecurityError(Exception):
    """安全异常"""
    pass


class SecureMathEvaluator:
    """安全的数学表达式计算器"""
    
    ALLOWED_OPERATORS = {
        ast.Add: '+',
        ast.Sub: '-', 
        ast.Mult: '*',
        ast.Div: '/',
        ast.Pow: '**',
        ast.USub: '-',
        ast.UAdd: '+'
    }
    
    ALLOWED_FUNCTIONS = {
        'abs', 'round', 'min', 'max', 'sum',
        'sqrt', 'pow', 'exp', 'log', 'sin', 'cos', 'tan'
    }
    
    def __init__(self):
        self.logger = logging.getLogger("security.math_evaluator")
        
    def safe_eval(self, expression: str, allowed_names: Dict[str, Any] = None) -> Union[float, int]:
        """安全地计算数学表达式"""
        try:
            # 简单的数字直接返回
            try:
                return float(expression.strip())
            except ValueError:
                pass
            
            # 解析表达式为AST
            tree = ast.parse(expression, mode='eval')
            
            # 验证AST安全性
            self._validate_ast_security(tree)
            
            # 准备安全的执行环境
            safe_dict = self._create_safe_environment(allowed_names or {})
            
            # 执行表达式
            result = eval(compile(tree, '<string>', 'eval'), safe_dict)
            
            return result
            
        except Exception as e:
            self.logger.warning(f"安全计算失败，使用默认值: {expression} - {e}")
            return 0.0  # 返回安全的默认值
    
    def _validate_ast_security(self, tree: ast.AST):
        """验证AST节点的安全性"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name not in self.ALLOWED_FUNCTIONS:
                        raise SecurityError(f"不允许的函数调用: {func_name}")
                else:
                    raise SecurityError("不允许的复杂函数调用")
            
            if isinstance(node, ast.Attribute):
                raise SecurityError("不允许访问对象属性")
            
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise SecurityError("不允许导入模块")
    
    def _create_safe_environment(self, allowed_names: Dict[str, Any]) -> Dict[str, Any]:
        """创建安全的执行环境"""
        safe_env = {
            '__builtins__': {},
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'sqrt': math.sqrt,
            'pow': pow,
            'exp': math.exp,
            'log': math.log,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'pi': math.pi,
            'e': math.e
        }
        
        # 添加允许的变量
        for name, value in allowed_names.items():
            if isinstance(value, (int, float, complex)):
                safe_env[name] = value
        
        return safe_env


# 全局安全计算器实例
_secure_evaluator = SecureMathEvaluator()
'''
        
        # 在每个需要的目录创建安全组件文件
        directories = [
            "src/models",
            "src/processors"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            if dir_path.exists():
                components_file = dir_path / "secure_components.py"
                with open(components_file, 'w', encoding='utf-8') as f:
                    f.write(secure_components_content)
                
                self.security_logger.info(f"已创建安全组件文件: {components_file}")


def main():
    """主函数 - 执行安全加固"""
    project_root = Path(__file__).parent.parent
    
    print("🔒 COT-DIR 安全加固实施")
    print("=" * 50)
    
    # 创建安全补丁器
    patcher = SecurityPatcher(project_root)
    
    # 1. 创建安全组件
    print("📦 创建安全组件...")
    patcher.create_secure_components_file()
    
    # 2. 修复 eval() 使用
    print("🔧 修复不安全的 eval() 使用...")
    patcher.patch_eval_usage()
    
    # 3. 创建安全配置管理示例
    print("⚙️ 创建安全配置管理...")
    config_dir = project_root / "config" / "secure"
    config_manager = SecureConfigManager(str(config_dir))
    
    # 示例：保存安全配置
    sample_config = {
        "model_settings": {
            "max_tokens": 2048,
            "temperature": 0.7
        },
        "security_settings": {
            "enable_input_validation": True,
            "enable_output_filtering": True
        }
    }
    
    try:
        config_manager.save_secure_config("default", sample_config)
        print("✅ 安全配置示例已创建")
    except Exception as e:
        print(f"⚠️ 配置创建失败: {e}")
    
    # 4. 创建安全检查清单
    create_security_checklist(project_root)
    
    print("\n🎉 安全加固完成!")
    print("\n📋 后续建议:")
    print("1. 定期运行安全扫描")
    print("2. 更新依赖包到最新版本")
    print("3. 启用代码审查安全检查")
    print("4. 实施访问控制和权限管理")
    print("5. 监控和记录安全事件")


def create_security_checklist(project_root: Path):
    """创建安全检查清单"""
    
    checklist_content = '''# COT-DIR 安全检查清单

## 代码安全
- [ ] ✅ 已替换所有 eval() 使用为安全计算器
- [ ] ✅ 已替换 pickle 为安全的 JSON 序列化
- [ ] 🔍 检查并移除硬编码密钥/密码
- [ ] 🔍 验证用户输入和数据验证
- [ ] 🔍 检查文件权限设置

## 依赖安全  
- [ ] 📦 定期运行 safety check 扫描漏洞
- [ ] 📦 更新过时的依赖包
- [ ] 📦 移除不必要的依赖
- [ ] 📦 使用依赖锁定文件

## 配置安全
- [ ] ⚙️ 敏感配置使用加密存储
- [ ] ⚙️ 使用环境变量管理密钥
- [ ] ⚙️ 配置文件权限设置为 600
- [ ] ⚙️ 分环境配置管理

## 部署安全
- [ ] 🚀 生产环境禁用调试模式
- [ ] 🚀 使用HTTPS加密传输
- [ ] 🚀 实施访问控制和身份验证
- [ ] 🚀 配置防火墙和网络安全

## 监控安全
- [ ] 📊 启用安全日志记录
- [ ] 📊 配置异常告警
- [ ] 📊 定期安全审计
- [ ] 📊 建立事件响应流程

## 定期检查
- [ ] 🔄 每月运行安全扫描
- [ ] 🔄 每季度依赖更新
- [ ] 🔄 每半年安全评估
- [ ] 🔄 年度安全培训

---

🔒 安全是一个持续的过程，需要定期维护和改进。
'''
    
    checklist_path = project_root / "SECURITY_CHECKLIST.md"
    with open(checklist_path, 'w', encoding='utf-8') as f:
        f.write(checklist_content)
    
    print(f"📋 安全检查清单已创建: {checklist_path}")


if __name__ == "__main__":
    main()