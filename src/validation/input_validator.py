"""
输入验证系统
提供全面的输入安全检查和数据验证
"""

import html
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..config.config_manager import get_config
from ..core.exceptions import InputValidationError, SecurityError

logger = logging.getLogger(__name__)

class InputValidator:
    """输入验证器"""
    
    def __init__(self):
        try:
            self.config = get_config()
        except Exception:
            # 配置不可用时使用默认值
            self.config = None
        
        # 危险模式列表
        self.dangerous_patterns = [
            r'<script.*?>.*?</script>',  # JavaScript
            r'javascript:',              # JavaScript协议
            r'eval\s*\(',               # eval函数
            r'exec\s*\(',               # exec函数
            r'import\s+os',             # 系统导入
            r'__import__',              # 动态导入
            r'subprocess',              # 子进程
            r'system\s*\(',             # 系统调用
            r'shell=True',              # Shell执行
            r'\.\./',                   # 路径遍历
            r'file://',                 # 文件协议
            r'ftp://',                  # FTP协议
        ]
        
        # 编译正则表达式
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE | re.DOTALL) 
                                for pattern in self.dangerous_patterns]
    
    def validate_math_problem(self, text: str) -> Dict[str, Any]:
        """验证数学问题输入"""
        result = {
            "valid": True,
            "sanitized_text": text,
            "warnings": [],
            "errors": []
        }
        
        try:
            # 基础检查
            if not text:
                result["valid"] = False
                result["errors"].append("输入不能为空")
                return result
            
            if not isinstance(text, str):
                result["valid"] = False
                result["errors"].append("输入必须为字符串")
                return result
            
            # 长度检查
            max_length = self._get_config_value("validation.max_input_length", 10000)
            if len(text) > max_length:
                result["valid"] = False
                result["errors"].append(f"输入长度超过限制 ({max_length} 字符)")
                return result
            
            # 安全检查
            security_result = self._check_security_threats(text)
            if not security_result["safe"]:
                result["valid"] = False
                result["errors"].extend(security_result["threats"])
                return result
            
            # 内容合理性检查
            content_result = self._check_content_validity(text)
            if not content_result["valid"]:
                result["warnings"].extend(content_result["warnings"])
            
            # HTML转义
            result["sanitized_text"] = html.escape(text)
            
            # 数学符号标准化
            result["sanitized_text"] = self._normalize_math_symbols(result["sanitized_text"])
            
        except Exception as e:
            logger.error(f"输入验证过程出错: {str(e)}")
            result["valid"] = False
            result["errors"].append("验证过程出现内部错误")
        
        return result
    
    def _get_config_value(self, key: str, default: Any) -> Any:
        """安全获取配置值"""
        if self.config:
            return self.config.get(key, default)
        return default
    
    def _check_security_threats(self, text: str) -> Dict[str, Any]:
        """检查安全威胁"""
        result = {
            "safe": True,
            "threats": []
        }
        
        # 检查危险模式
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                result["safe"] = False
                result["threats"].append(f"检测到潜在安全威胁: {pattern.pattern}")
        
        # 检查异常字符
        if self._contains_suspicious_chars(text):
            result["safe"] = False
            result["threats"].append("包含可疑字符")
        
        # 检查编码攻击
        if self._check_encoding_attacks(text):
            result["safe"] = False
            result["threats"].append("检测到编码攻击")
        
        return result
    
    def _contains_suspicious_chars(self, text: str) -> bool:
        """检查可疑字符"""
        suspicious_chars = [
            '\x00',  # NULL字符
            '\x08',  # 退格
            '\x0b',  # 垂直制表符
            '\x0c',  # 换页符
            '\x7f',  # DEL字符
        ]
        
        return any(char in text for char in suspicious_chars)
    
    def _check_encoding_attacks(self, text: str) -> bool:
        """检查编码攻击"""
        # 检查Unicode控制字符
        control_chars = [chr(i) for i in range(0x00, 0x20)] + [chr(0x7f)]
        allowed_control = ['\n', '\r', '\t']
        
        for char in text:
            if char in control_chars and char not in allowed_control:
                return True
        
        # 检查过长的Unicode序列
        if '\\u' in text and len(re.findall(r'\\u[0-9a-fA-F]{4}', text)) > 10:
            return True
        
        return False
    
    def _check_content_validity(self, text: str) -> Dict[str, Any]:
        """检查内容合理性"""
        result = {
            "valid": True,
            "warnings": []
        }
        
        # 检查是否包含数字
        if not re.search(r'\d', text):
            result["warnings"].append("输入中未检测到数字，可能不是数学问题")
        
        # 检查是否包含数学关键词
        math_keywords = [
            '加', '减', '乘', '除', '等于', '多少', '计算', '求',
            '面积', '周长', '体积', '长度', '宽度', '高度',
            '+', '-', '*', '/', '=', '×', '÷',
            'add', 'subtract', 'multiply', 'divide', 'calculate'
        ]
        
        if not any(keyword in text.lower() for keyword in math_keywords):
            result["warnings"].append("输入中未检测到数学关键词")
        
        # 检查重复字符
        if self._has_excessive_repetition(text):
            result["warnings"].append("检测到异常的字符重复")
        
        return result
    
    def _has_excessive_repetition(self, text: str) -> bool:
        """检查过度重复的字符"""
        # 检查连续重复字符
        if re.search(r'(.)\1{10,}', text):  # 同一字符连续出现10次以上
            return True
        
        # 检查重复模式
        if re.search(r'(.{2,})\1{5,}', text):  # 2字符以上的模式重复5次以上
            return True
        
        return False
    
    def _normalize_math_symbols(self, text: str) -> str:
        """标准化数学符号"""
        # 符号映射
        symbol_map = {
            '×': '*',
            '÷': '/',
            '－': '-',
            '＋': '+',
            '＝': '=',
            '（': '(',
            '）': ')',
            '。': '.',
            '，': ',',
        }
        
        normalized = text
        for old_symbol, new_symbol in symbol_map.items():
            normalized = normalized.replace(old_symbol, new_symbol)
        
        return normalized
    
    def validate_file_path(self, path: str) -> Dict[str, Any]:
        """验证文件路径安全性"""
        result = {
            "valid": True,
            "safe_path": None,
            "errors": []
        }
        
        try:
            if not path:
                result["valid"] = False
                result["errors"].append("路径不能为空")
                return result
            
            # 解析路径
            safe_path = Path(path).resolve()
            
            # 检查路径遍历攻击
            if '..' in path or path.startswith('/'):
                result["valid"] = False
                result["errors"].append("检测到路径遍历攻击")
                return result
            
            # 检查是否在项目目录内
            project_root = Path(__file__).parent.parent.parent.resolve()
            try:
                safe_path.relative_to(project_root)
                result["safe_path"] = str(safe_path)
            except ValueError:
                result["valid"] = False
                result["errors"].append("路径超出项目范围")
                return result
            
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"路径验证失败: {str(e)}")
        
        return result
    
    def validate_numeric_input(self, value: Any) -> Dict[str, Any]:
        """验证数值输入"""
        result = {
            "valid": True,
            "numeric_value": None,
            "errors": []
        }
        
        try:
            if value is None:
                result["valid"] = False
                result["errors"].append("数值不能为空")
                return result
            
            # 尝试转换为数字
            if isinstance(value, (int, float)):
                result["numeric_value"] = float(value)
            elif isinstance(value, str):
                # 清理字符串
                cleaned = re.sub(r'[^\d\.\-\+]', '', value)
                if not cleaned:
                    result["valid"] = False
                    result["errors"].append("无法从输入中提取数字")
                    return result
                
                try:
                    result["numeric_value"] = float(cleaned)
                except ValueError:
                    result["valid"] = False
                    result["errors"].append("输入不是有效的数字")
                    return result
            else:
                result["valid"] = False
                result["errors"].append("输入类型不支持数值转换")
                return result
            
            # 检查数值范围
            if abs(result["numeric_value"]) > 1e15:
                result["valid"] = False
                result["errors"].append("数值超出处理范围")
                return result
            
            # 检查是否为NaN或无穷大
            import math
            if math.isnan(result["numeric_value"]) or math.isinf(result["numeric_value"]):
                result["valid"] = False
                result["errors"].append("数值无效（NaN或无穷大）")
                return result
            
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"数值验证失败: {str(e)}")
        
        return result
    
    def batch_validate(self, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量验证输入"""
        results = {
            "all_valid": True,
            "results": [],
            "summary": {
                "total": len(inputs),
                "valid": 0,
                "invalid": 0,
                "warnings": 0
            }
        }
        
        for i, input_data in enumerate(inputs):
            input_type = input_data.get("type", "math_problem")
            input_value = input_data.get("value", "")
            
            if input_type == "math_problem":
                validation_result = self.validate_math_problem(input_value)
            elif input_type == "numeric":
                validation_result = self.validate_numeric_input(input_value)
            elif input_type == "file_path":
                validation_result = self.validate_file_path(input_value)
            else:
                validation_result = {
                    "valid": False,
                    "errors": [f"不支持的输入类型: {input_type}"]
                }
            
            validation_result["index"] = i
            validation_result["type"] = input_type
            results["results"].append(validation_result)
            
            # 更新摘要
            if validation_result["valid"]:
                results["summary"]["valid"] += 1
            else:
                results["summary"]["invalid"] += 1
                results["all_valid"] = False
            
            if validation_result.get("warnings"):
                results["summary"]["warnings"] += len(validation_result["warnings"])
        
        return results

# 全局验证器实例
_global_validator = None

def get_validator() -> InputValidator:
    """获取全局验证器实例"""
    global _global_validator
    if _global_validator is None:
        _global_validator = InputValidator()
    return _global_validator

def validate_input(text: str, input_type: str = "math_problem") -> Dict[str, Any]:
    """便捷的输入验证函数"""
    validator = get_validator()
    
    if input_type == "math_problem":
        return validator.validate_math_problem(text)
    elif input_type == "numeric":
        return validator.validate_numeric_input(text)
    elif input_type == "file_path":
        return validator.validate_file_path(text)
    else:
        return {
            "valid": False,
            "errors": [f"不支持的输入类型: {input_type}"]
        } 