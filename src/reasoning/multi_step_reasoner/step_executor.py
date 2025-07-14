"""
推理步骤执行器
负责执行具体的推理步骤和操作
"""

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from ...core.exceptions import ReasoningError, handle_exceptions
from ...core.interfaces import ReasoningContext
from ...core.interfaces import ReasoningStep as ReasoningStepEnum
from ...monitoring.performance_monitor import get_monitor, monitor_performance


class StepType(Enum):
    """推理步骤类型"""
    PARSE = "parse"              # 解析
    EXTRACT = "extract"          # 提取
    CALCULATE = "calculate"      # 计算
    TRANSFORM = "transform"      # 转换
    VALIDATE = "validate"        # 验证
    SYNTHESIZE = "synthesize"    # 综合
    REASON = "reason"            # 推理

class OperationType(Enum):
    """操作类型"""
    ARITHMETIC = "arithmetic"    # 算术运算
    LOGICAL = "logical"          # 逻辑运算
    COMPARISON = "comparison"    # 比较运算
    EXTRACTION = "extraction"    # 信息提取
    TRANSFORMATION = "transformation"  # 数据转换
    VALIDATION = "validation"    # 验证操作

@dataclass
class ExecutionResult:
    """执行结果"""
    success: bool
    result: Any
    confidence: float
    execution_time: float
    step_type: StepType
    operation_type: OperationType
    details: Dict[str, Any]
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "result": self.result,
            "confidence": self.confidence,
            "execution_time": self.execution_time,
            "step_type": self.step_type.value,
            "operation_type": self.operation_type.value,
            "details": self.details,
            "error_message": self.error_message
        }

class StepExecutor:
    """推理步骤执行器"""
    
    def __init__(self):
        """初始化步骤执行器"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.monitor = get_monitor()
        
        # 执行统计
        self.execution_stats = {
            "total_steps": 0,
            "successful_steps": 0,
            "failed_steps": 0,
            "average_execution_time": 0.0
        }
        
        # 注册执行器函数
        self.executors = {
            StepType.PARSE: self._execute_parse,
            StepType.EXTRACT: self._execute_extract,
            StepType.CALCULATE: self._execute_calculate,
            StepType.TRANSFORM: self._execute_transform,
            StepType.VALIDATE: self._execute_validate,
            StepType.SYNTHESIZE: self._execute_synthesize,
            StepType.REASON: self._execute_reason
        }
        
        # 数学运算函数映射
        self.math_operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else None,
            "power": lambda x, y: x ** y,
            "mod": lambda x, y: x % y if y != 0 else None
        }
        
        self.logger.info("步骤执行器初始化完成")
    
    @monitor_performance("step_execution")
    @handle_exceptions(reraise_as=ReasoningError)
    def execute_step(self, step_data: Dict[str, Any], context: Optional[ReasoningContext] = None) -> ExecutionResult:
        """
        执行推理步骤
        
        Args:
            step_data: 步骤数据
            context: 推理上下文
            
        Returns:
            ExecutionResult: 执行结果
        """
        start_time = time.time()
        
        try:
            # 解析步骤信息
            step_type = self._parse_step_type(step_data)
            operation_type = self._parse_operation_type(step_data)
            
            self.logger.debug(f"执行步骤: {step_type.value}, 操作: {operation_type.value}")
            
            # 验证步骤数据
            if not self._validate_step_data(step_data, step_type):
                return ExecutionResult(
                    success=False,
                    result=None,
                    confidence=0.0,
                    execution_time=time.time() - start_time,
                    step_type=step_type,
                    operation_type=operation_type,
                    details={},
                    error_message="步骤数据验证失败"
                )
            
            # 执行步骤
            executor = self.executors.get(step_type)
            if not executor:
                raise ReasoningError(f"未找到步骤类型 {step_type.value} 的执行器")
            
            result = executor(step_data, operation_type, context)
            
            # 更新统计
            self._update_stats(result)
            
            # 记录监控指标
            self.monitor.increment_counter("step_executions_total")
            if result.success:
                self.monitor.increment_counter("step_executions_success")
            
            return result
            
        except Exception as e:
            self.logger.error(f"步骤执行失败: {str(e)}")
            
            return ExecutionResult(
                success=False,
                result=None,
                confidence=0.0,
                execution_time=time.time() - start_time,
                step_type=StepType.REASON,  # 默认类型
                operation_type=OperationType.ARITHMETIC,  # 默认操作
                details={},
                error_message=str(e)
            )
    
    def _parse_step_type(self, step_data: Dict[str, Any]) -> StepType:
        """解析步骤类型"""
        step_type_str = step_data.get("action", step_data.get("type", "reason"))
        
        # 映射常见的动作名称
        action_mapping = {
            "number_extraction": StepType.EXTRACT,
            "information_extraction": StepType.EXTRACT,
            "problem_analysis": StepType.PARSE,
            "template_identification": StepType.PARSE,
            "addition": StepType.CALCULATE,
            "subtraction": StepType.CALCULATE,
            "multiplication": StepType.CALCULATE,
            "division": StepType.CALCULATE,
            "calculation": StepType.CALCULATE,
            "answer_validation": StepType.VALIDATE,
            "result_validation": StepType.VALIDATE,
            "reasoning": StepType.REASON,
            "synthesis": StepType.SYNTHESIZE
        }
        
        mapped_type = action_mapping.get(step_type_str.lower())
        if mapped_type:
            return mapped_type
        
        # 尝试直接匹配枚举值
        try:
            return StepType(step_type_str.lower())
        except ValueError:
            return StepType.REASON  # 默认为推理类型
    
    def _parse_operation_type(self, step_data: Dict[str, Any]) -> OperationType:
        """解析操作类型"""
        action = step_data.get("action", "").lower()
        
        if any(op in action for op in ["add", "subtract", "multiply", "divide", "calculation"]):
            return OperationType.ARITHMETIC
        elif any(op in action for op in ["extract", "identify", "parse"]):
            return OperationType.EXTRACTION
        elif any(op in action for op in ["validate", "verify", "check"]):
            return OperationType.VALIDATION
        elif any(op in action for op in ["transform", "convert", "normalize"]):
            return OperationType.TRANSFORMATION
        elif any(op in action for op in ["compare", "greater", "less", "equal"]):
            return OperationType.COMPARISON
        elif any(op in action for op in ["and", "or", "not", "if", "logic"]):
            return OperationType.LOGICAL
        else:
            return OperationType.ARITHMETIC  # 默认为算术操作
    
    def _validate_step_data(self, step_data: Dict[str, Any], step_type: StepType) -> bool:
        """验证步骤数据"""
        # 基本字段检查
        if not isinstance(step_data, dict):
            return False
        
        # 根据步骤类型进行特定验证
        if step_type == StepType.CALCULATE:
            # 计算步骤需要数字数据
            numbers = step_data.get("numbers", [])
            operands = step_data.get("operands", [])
            if not numbers and not operands:
                return False
        
        return True
    
    def _execute_parse(self, step_data: Dict[str, Any], operation_type: OperationType, 
                      context: Optional[ReasoningContext]) -> ExecutionResult:
        """执行解析步骤"""
        start_time = time.time()
        
        try:
            text = step_data.get("text", step_data.get("description", ""))
            
            # 解析文本中的关键信息
            parsed_info = {
                "numbers": self._extract_numbers(text),
                "keywords": self._extract_keywords(text),
                "operations": self._identify_operations(text),
                "units": self._extract_units(text)
            }
            
            confidence = 0.9 if parsed_info["numbers"] else 0.6
            
            return ExecutionResult(
                success=True,
                result=parsed_info,
                confidence=confidence,
                execution_time=time.time() - start_time,
                step_type=StepType.PARSE,
                operation_type=operation_type,
                details={"parsed_text": text}
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                result=None,
                confidence=0.0,
                execution_time=time.time() - start_time,
                step_type=StepType.PARSE,
                operation_type=operation_type,
                details={},
                error_message=str(e)
            )
    
    def _execute_extract(self, step_data: Dict[str, Any], operation_type: OperationType,
                        context: Optional[ReasoningContext]) -> ExecutionResult:
        """执行提取步骤"""
        start_time = time.time()
        
        try:
            text = step_data.get("text", step_data.get("description", ""))
            extract_type = step_data.get("extract_type", "numbers")
            
            if extract_type == "numbers":
                extracted = self._extract_numbers(text)
            elif extract_type == "keywords":
                extracted = self._extract_keywords(text)
            elif extract_type == "units":
                extracted = self._extract_units(text)
            else:
                extracted = self._extract_general_info(text)
            
            confidence = 0.95 if extracted else 0.3
            
            return ExecutionResult(
                success=len(extracted) > 0,
                result=extracted,
                confidence=confidence,
                execution_time=time.time() - start_time,
                step_type=StepType.EXTRACT,
                operation_type=operation_type,
                details={"extract_type": extract_type, "source_text": text}
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                result=[],
                confidence=0.0,
                execution_time=time.time() - start_time,
                step_type=StepType.EXTRACT,
                operation_type=operation_type,
                details={},
                error_message=str(e)
            )
    
    def _execute_calculate(self, step_data: Dict[str, Any], operation_type: OperationType,
                          context: Optional[ReasoningContext]) -> ExecutionResult:
        """执行计算步骤"""
        start_time = time.time()
        
        try:
            # 获取操作数
            numbers = step_data.get("numbers", step_data.get("operands", []))
            operation = step_data.get("operation", step_data.get("action", "add"))
            
            if not numbers:
                raise ValueError("没有找到可计算的数字")
            
            # 确保数字是数值类型
            numbers = [float(n) for n in numbers if isinstance(n, (int, float, str)) and str(n).replace('.', '').replace('-', '').isdigit()]
            
            if not numbers:
                raise ValueError("没有有效的数字")
            
            # 执行计算
            result = self._perform_calculation(numbers, operation)
            
            if result is None:
                raise ValueError(f"计算操作 {operation} 失败")
            
            # 计算置信度
            confidence = 0.95 if len(numbers) >= 2 else 0.8
            
            return ExecutionResult(
                success=True,
                result=result,
                confidence=confidence,
                execution_time=time.time() - start_time,
                step_type=StepType.CALCULATE,
                operation_type=operation_type,
                details={
                    "operation": operation,
                    "operands": numbers,
                    "calculation": f"{operation}({numbers}) = {result}"
                }
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                result=None,
                confidence=0.0,
                execution_time=time.time() - start_time,
                step_type=StepType.CALCULATE,
                operation_type=operation_type,
                details={},
                error_message=str(e)
            )
    
    def _execute_transform(self, step_data: Dict[str, Any], operation_type: OperationType,
                          context: Optional[ReasoningContext]) -> ExecutionResult:
        """执行转换步骤"""
        start_time = time.time()
        
        try:
            input_data = step_data.get("input", step_data.get("data"))
            transform_type = step_data.get("transform_type", "normalize")
            
            if transform_type == "normalize":
                result = self._normalize_data(input_data)
            elif transform_type == "convert_unit":
                result = self._convert_units(input_data, step_data.get("target_unit"))
            elif transform_type == "format":
                result = self._format_data(input_data, step_data.get("format_type"))
            else:
                result = input_data  # 默认不变
            
            return ExecutionResult(
                success=True,
                result=result,
                confidence=0.9,
                execution_time=time.time() - start_time,
                step_type=StepType.TRANSFORM,
                operation_type=operation_type,
                details={"transform_type": transform_type, "input": input_data}
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                result=None,
                confidence=0.0,
                execution_time=time.time() - start_time,
                step_type=StepType.TRANSFORM,
                operation_type=operation_type,
                details={},
                error_message=str(e)
            )
    
    def _execute_validate(self, step_data: Dict[str, Any], operation_type: OperationType,
                         context: Optional[ReasoningContext]) -> ExecutionResult:
        """执行验证步骤"""
        start_time = time.time()
        
        try:
            value = step_data.get("value", step_data.get("result"))
            validation_type = step_data.get("validation_type", "range")
            
            validation_result = {
                "valid": True,
                "confidence": 0.8,
                "checks": []
            }
            
            if validation_type == "range":
                min_val = step_data.get("min_value", float('-inf'))
                max_val = step_data.get("max_value", float('inf'))
                
                if isinstance(value, (int, float)):
                    if min_val <= value <= max_val:
                        validation_result["checks"].append("范围检查通过")
                    else:
                        validation_result["valid"] = False
                        validation_result["confidence"] = 0.2
                        validation_result["checks"].append(f"值 {value} 超出范围 [{min_val}, {max_val}]")
            
            elif validation_type == "type":
                expected_type = step_data.get("expected_type", "number")
                
                if expected_type == "number" and isinstance(value, (int, float)):
                    validation_result["checks"].append("类型检查通过")
                elif expected_type == "string" and isinstance(value, str):
                    validation_result["checks"].append("类型检查通过")
                else:
                    validation_result["valid"] = False
                    validation_result["confidence"] = 0.1
                    validation_result["checks"].append(f"类型不匹配，期望 {expected_type}")
            
            return ExecutionResult(
                success=validation_result["valid"],
                result=validation_result,
                confidence=validation_result["confidence"],
                execution_time=time.time() - start_time,
                step_type=StepType.VALIDATE,
                operation_type=operation_type,
                details={"validation_type": validation_type, "validated_value": value}
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                result={"valid": False, "error": str(e)},
                confidence=0.0,
                execution_time=time.time() - start_time,
                step_type=StepType.VALIDATE,
                operation_type=operation_type,
                details={},
                error_message=str(e)
            )
    
    def _execute_synthesize(self, step_data: Dict[str, Any], operation_type: OperationType,
                           context: Optional[ReasoningContext]) -> ExecutionResult:
        """执行综合步骤"""
        start_time = time.time()
        
        try:
            inputs = step_data.get("inputs", [])
            synthesis_type = step_data.get("synthesis_type", "combine")
            
            if synthesis_type == "combine":
                result = self._combine_results(inputs)
            elif synthesis_type == "aggregate":
                result = self._aggregate_results(inputs)
            elif synthesis_type == "select_best":
                result = self._select_best_result(inputs)
            else:
                result = inputs[0] if inputs else None
            
            confidence = 0.8 if result is not None else 0.3
            
            return ExecutionResult(
                success=result is not None,
                result=result,
                confidence=confidence,
                execution_time=time.time() - start_time,
                step_type=StepType.SYNTHESIZE,
                operation_type=operation_type,
                details={"synthesis_type": synthesis_type, "input_count": len(inputs)}
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                result=None,
                confidence=0.0,
                execution_time=time.time() - start_time,
                step_type=StepType.SYNTHESIZE,
                operation_type=operation_type,
                details={},
                error_message=str(e)
            )
    
    def _execute_reason(self, step_data: Dict[str, Any], operation_type: OperationType,
                       context: Optional[ReasoningContext]) -> ExecutionResult:
        """执行推理步骤"""
        start_time = time.time()
        
        try:
            premise = step_data.get("premise", step_data.get("description", ""))
            reasoning_type = step_data.get("reasoning_type", "logical")
            
            # 简单的推理逻辑
            reasoning_result = {
                "conclusion": f"基于前提 '{premise}' 的推理结论",
                "confidence": 0.7,
                "reasoning_chain": [premise]
            }
            
            return ExecutionResult(
                success=True,
                result=reasoning_result,
                confidence=0.7,
                execution_time=time.time() - start_time,
                step_type=StepType.REASON,
                operation_type=operation_type,
                details={"reasoning_type": reasoning_type, "premise": premise}
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                result=None,
                confidence=0.0,
                execution_time=time.time() - start_time,
                step_type=StepType.REASON,
                operation_type=operation_type,
                details={},
                error_message=str(e)
            )
    
    # 辅助方法
    def _extract_numbers(self, text: str) -> List[float]:
        """从文本中提取数字"""
        pattern = r'-?\d+(?:\.\d+)?'
        matches = re.findall(pattern, text)
        return [float(match) for match in matches]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        keywords = ["加", "减", "乘", "除", "等于", "总共", "一共", "剩余", "平均", 
                   "面积", "周长", "长度", "宽度", "高度", "速度", "时间"]
        found = [kw for kw in keywords if kw in text]
        return found
    
    def _extract_units(self, text: str) -> List[str]:
        """提取单位"""
        units = ["米", "厘米", "千米", "公里", "元", "角", "分", "个", "只", "本", 
                "小时", "分钟", "秒", "天", "平方米", "立方米"]
        found = [unit for unit in units if unit in text]
        return found
    
    def _identify_operations(self, text: str) -> List[str]:
        """识别运算操作"""
        operations = []
        if any(kw in text for kw in ["加", "+", "和", "总共"]):
            operations.append("addition")
        if any(kw in text for kw in ["减", "-", "少", "剩"]):
            operations.append("subtraction")
        if any(kw in text for kw in ["乘", "×", "*", "倍"]):
            operations.append("multiplication")
        if any(kw in text for kw in ["除", "÷", "/", "平均"]):
            operations.append("division")
        return operations
    
    def _extract_general_info(self, text: str) -> List[str]:
        """提取一般信息"""
        # 简单的关键信息提取
        words = text.split()
        important_words = [word for word in words if len(word) > 2]
        return important_words[:5]  # 返回前5个重要词汇
    
    def _perform_calculation(self, numbers: List[float], operation: str) -> Optional[float]:
        """执行数学计算"""
        if not numbers:
            return None
        
        try:
            if operation in ["add", "addition", "加"]:
                return sum(numbers)
            elif operation in ["subtract", "subtraction", "减"]:
                return numbers[0] - sum(numbers[1:]) if len(numbers) > 1 else numbers[0]
            elif operation in ["multiply", "multiplication", "乘"]:
                result = 1
                for num in numbers:
                    result *= num
                return result
            elif operation in ["divide", "division", "除"]:
                if len(numbers) >= 2 and numbers[1] != 0:
                    return numbers[0] / numbers[1]
                return None
            else:
                # 默认尝试加法
                return sum(numbers)
                
        except Exception:
            return None
    
    def _normalize_data(self, data: Any) -> Any:
        """标准化数据"""
        if isinstance(data, (int, float)):
            return float(data)
        elif isinstance(data, str):
            return data.strip().lower()
        elif isinstance(data, list):
            return [self._normalize_data(item) for item in data]
        else:
            return data
    
    def _convert_units(self, data: Any, target_unit: Optional[str]) -> Any:
        """单位转换"""
        # 简单的单位转换逻辑
        if target_unit and isinstance(data, (int, float)):
            # 这里可以添加具体的单位转换逻辑
            return data
        return data
    
    def _format_data(self, data: Any, format_type: Optional[str]) -> Any:
        """格式化数据"""
        if format_type == "integer" and isinstance(data, float):
            return int(data)
        elif format_type == "string":
            return str(data)
        return data
    
    def _combine_results(self, inputs: List[Any]) -> Any:
        """组合结果"""
        if not inputs:
            return None
        
        # 如果都是数字，返回列表
        if all(isinstance(x, (int, float)) for x in inputs):
            return inputs
        
        # 如果都是字符串，连接它们
        if all(isinstance(x, str) for x in inputs):
            return " ".join(inputs)
        
        return inputs
    
    def _aggregate_results(self, inputs: List[Any]) -> Any:
        """聚合结果"""
        if not inputs:
            return None
        
        # 如果都是数字，计算平均值
        if all(isinstance(x, (int, float)) for x in inputs):
            return sum(inputs) / len(inputs)
        
        return inputs[0] if inputs else None
    
    def _select_best_result(self, inputs: List[Any]) -> Any:
        """选择最佳结果"""
        if not inputs:
            return None
        
        # 简单地返回第一个非空结果
        for item in inputs:
            if item is not None:
                return item
        
        return None
    
    def _update_stats(self, result: ExecutionResult):
        """更新执行统计"""
        self.execution_stats["total_steps"] += 1
        
        if result.success:
            self.execution_stats["successful_steps"] += 1
        else:
            self.execution_stats["failed_steps"] += 1
        
        # 更新平均执行时间
        total_time = (self.execution_stats["average_execution_time"] * 
                     (self.execution_stats["total_steps"] - 1) + result.execution_time)
        self.execution_stats["average_execution_time"] = total_time / self.execution_stats["total_steps"]
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计信息"""
        return self.execution_stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.execution_stats = {
            "total_steps": 0,
            "successful_steps": 0,
            "failed_steps": 0,
            "average_execution_time": 0.0
        } 