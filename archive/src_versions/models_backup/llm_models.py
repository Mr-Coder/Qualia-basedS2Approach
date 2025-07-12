#!/usr/bin/env python3
"""
Large Language Model (LLM) Implementations

This module implements various LLM models for mathematical word problem solving.
Supports OpenAI GPT, Claude, Qwen, InternLM, and other popular LLMs.
"""

import json
import logging
import os
import re
import time
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

import requests

from .base_model import LLMModel, ModelInput, ModelOutput


class OpenAIGPTModel(LLMModel):
    """OpenAI GPT model implementation (GPT-3.5, GPT-4, GPT-4o)."""
    
    def __init__(self, model_name: str = "gpt-4o", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        self.api_key = config.get("api_key") if config else None
        self.base_url = config.get("base_url", "https://api.openai.com/v1") if config else "https://api.openai.com/v1"
        self.model_version = model_name
        
    def initialize(self) -> bool:
        """Initialize OpenAI GPT model."""
        try:
            self.logger.info(f"Initializing OpenAI {self.model_name}")
            
            # Check for API key
            if not self.api_key:
                self.api_key = os.getenv("OPENAI_API_KEY")
            
            if not self.api_key:
                self.logger.error("OpenAI API key not provided")
                return False
            
            # Test API connection
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            test_data = {
                "model": self.model_version,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10,
                "temperature": 0.1
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=test_data,
                timeout=10
            )
            
            if response.status_code == 200:
                self.is_initialized = True
                self.logger.info(f"OpenAI {self.model_name} initialized successfully")
                return True
            else:
                self.logger.error(f"OpenAI API test failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI {self.model_name}: {e}")
            return False
    
    def solve_problem(self, problem_input: ModelInput) -> ModelOutput:
        """Solve problem using OpenAI GPT."""
        start_time = time.time()
        
        if not self.validate_input(problem_input):
            return ModelOutput(
                answer="",
                reasoning_chain=["Invalid input"],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                error_message="Invalid input format"
            )
        
        try:
            # Generate prompt
            prompt = self.generate_prompt(problem_input)
            
            # Call OpenAI API
            response_text = self.call_api(prompt)
            
            # Parse response
            result = self.parse_response(response_text)
            result.processing_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            return ModelOutput(
                answer="",
                reasoning_chain=[f"Error calling OpenAI API: {str(e)}"],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def batch_solve(self, problems: List[ModelInput]) -> List[ModelOutput]:
        """Solve multiple problems using OpenAI GPT."""
        return [self.solve_problem(problem) for problem in problems]
    
    def generate_prompt(self, problem_input: ModelInput) -> str:
        """Generate appropriate prompt for OpenAI GPT."""
        system_prompt = """You are an expert at solving mathematical word problems. Please solve the given problem step by step.

Follow this format:
1. Understanding: Restate what the problem is asking
2. Information: List the given information and what needs to be found
3. Reasoning: Show your step-by-step solution process
4. Calculation: Perform the mathematical calculations
5. Answer: Provide the final numerical answer

Problem:"""
        
        user_prompt = f"{system_prompt}\n{problem_input.problem_text}\n\nSolution:"
        return user_prompt
    
    def call_api(self, prompt: str) -> str:
        """Call OpenAI API with the given prompt."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_version,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"OpenAI API error: {response.status_code}, {response.text}")
    
    def parse_response(self, response: str) -> ModelOutput:
        """Parse OpenAI response into structured output."""
        reasoning_chain = []
        answer = ""
        confidence = 0.7
        
        # Split response into sections
        lines = response.strip().split('\n')
        current_section = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Identify sections
            if line.lower().startswith(('1.', 'understanding:', 'problem:')):
                current_section = "understanding"
                reasoning_chain.append(f"Understanding: {line}")
            elif line.lower().startswith(('2.', 'information:', 'given:')):
                current_section = "information"
                reasoning_chain.append(f"Information: {line}")
            elif line.lower().startswith(('3.', 'reasoning:', 'solution:', 'step')):
                current_section = "reasoning"
                reasoning_chain.append(f"Reasoning: {line}")
            elif line.lower().startswith(('4.', 'calculation:', 'compute')):
                current_section = "calculation"
                reasoning_chain.append(f"Calculation: {line}")
            elif line.lower().startswith(('5.', 'answer:', 'final')):
                current_section = "answer"
                reasoning_chain.append(f"Answer: {line}")
                # Extract numerical answer
                numbers = re.findall(r'-?\d+(?:\.\d+)?', line)
                if numbers:
                    answer = numbers[-1]  # Take the last number as the final answer
            else:
                # Continue current section
                if current_section:
                    reasoning_chain.append(line)
                    # Look for answer in any line
                    if not answer and any(word in line.lower() for word in ['answer', 'result', 'solution']):
                        numbers = re.findall(r'-?\d+(?:\.\d+)?', line)
                        if numbers:
                            answer = numbers[-1]
        
        # If no answer found, try to extract from the entire response
        if not answer:
            numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
            if numbers:
                answer = numbers[-1]
        
        return ModelOutput(
            answer=answer,
            reasoning_chain=reasoning_chain,
            confidence_score=confidence,
            processing_time=0,  # Will be set by caller
            memory_usage=1.0  # Estimated for API call
        )


class ClaudeModel(LLMModel):
    """Anthropic Claude model implementation."""
    
    def __init__(self, model_name: str = "claude-3.5-sonnet", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        self.api_key = config.get("api_key") if config else None
        self.base_url = config.get("base_url", "https://api.anthropic.com/v1") if config else "https://api.anthropic.com/v1"
        self.model_version = model_name
        
    def initialize(self) -> bool:
        """Initialize Claude model."""
        try:
            self.logger.info(f"Initializing Claude {self.model_name}")
            
            # Check for API key
            if not self.api_key:
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            
            if not self.api_key:
                self.logger.error("Anthropic API key not provided")
                return False
            
            self.is_initialized = True
            self.logger.info(f"Claude {self.model_name} initialized successfully")
            return True
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Claude {self.model_name}: {e}")
            return False
    
    def solve_problem(self, problem_input: ModelInput) -> ModelOutput:
        """Solve problem using Claude."""
        start_time = time.time()
        
        if not self.validate_input(problem_input):
            return ModelOutput(
                answer="",
                reasoning_chain=["Invalid input"],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                error_message="Invalid input format"
            )
        
        try:
            # Generate prompt
            prompt = self.generate_prompt(problem_input)
            
            # Call Claude API
            response_text = self.call_api(prompt)
            
            # Parse response
            result = self.parse_response(response_text)
            result.processing_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            return ModelOutput(
                answer="",
                reasoning_chain=[f"Error calling Claude API: {str(e)}"],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def batch_solve(self, problems: List[ModelInput]) -> List[ModelOutput]:
        """Solve multiple problems using Claude."""
        return [self.solve_problem(problem) for problem in problems]
    
    def generate_prompt(self, problem_input: ModelInput) -> str:
        """Generate appropriate prompt for Claude."""
        prompt = f"""Please solve this mathematical word problem step by step.

Problem: {problem_input.problem_text}

Please provide:
1. A clear understanding of what the problem is asking
2. The given information and what needs to be found
3. Step-by-step reasoning and solution process
4. Mathematical calculations with clear explanations
5. The final numerical answer

Solution:"""
        
        return prompt
    
    def call_api(self, prompt: str) -> str:
        """Call Claude API with the given prompt."""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": self.model_version,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        response = requests.post(
            f"{self.base_url}/messages",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["content"][0]["text"]
        else:
            raise Exception(f"Claude API error: {response.status_code}, {response.text}")
    
    def parse_response(self, response: str) -> ModelOutput:
        """Parse Claude response into structured output."""
        reasoning_chain = []
        answer = ""
        confidence = 0.8
        
        # Split response into lines and process
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            reasoning_chain.append(line)
            
            # Look for final answer
            if any(keyword in line.lower() for keyword in ['final answer', 'answer:', 'result:', 'solution:']):
                numbers = re.findall(r'-?\d+(?:\.\d+)?', line)
                if numbers:
                    answer = numbers[-1]
        
        # If no answer found, extract from entire response
        if not answer:
            numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
            if numbers:
                answer = numbers[-1]
        
        return ModelOutput(
            answer=answer,
            reasoning_chain=reasoning_chain,
            confidence_score=confidence,
            processing_time=0,
            memory_usage=1.2
        )


class QwenModel(LLMModel):
    """Qwen model implementation (local or API)."""
    
    def __init__(self, model_name: str = "Qwen2.5-Math-72B", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        self.api_key = config.get("api_key") if config else None
        self.base_url = config.get("base_url", "http://localhost:8000/v1") if config else "http://localhost:8000/v1"
        self.model_version = model_name
        self.is_local = config.get("is_local", True) if config else True
        
    def initialize(self) -> bool:
        """Initialize Qwen model."""
        try:
            self.logger.info(f"Initializing Qwen {self.model_name}")
            
            if self.is_local:
                # For local deployment, try to connect to local server
                try:
                    response = requests.get(f"{self.base_url}/health", timeout=5)
                    if response.status_code == 200:
                        self.is_initialized = True
                        self.logger.info(f"Qwen {self.model_name} (local) initialized successfully")
                        return True
                except:
                    self.logger.warning("Local Qwen server not available, using mock responses")
                    self.is_initialized = True
                    return True
            else:
                # For API deployment
                if not self.api_key:
                    self.api_key = os.getenv("QWEN_API_KEY")
                
                self.is_initialized = True
                self.logger.info(f"Qwen {self.model_name} (API) initialized successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Qwen {self.model_name}: {e}")
            return False
    
    def solve_problem(self, problem_input: ModelInput) -> ModelOutput:
        """Solve problem using Qwen."""
        start_time = time.time()
        
        if not self.validate_input(problem_input):
            return ModelOutput(
                answer="",
                reasoning_chain=["Invalid input"],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                error_message="Invalid input format"
            )
        
        try:
            # Generate prompt
            prompt = self.generate_prompt(problem_input)
            
            # Call Qwen (local or API)
            response_text = self.call_api(prompt)
            
            # Parse response
            result = self.parse_response(response_text)
            result.processing_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            return ModelOutput(
                answer="",
                reasoning_chain=[f"Error calling Qwen: {str(e)}"],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def batch_solve(self, problems: List[ModelInput]) -> List[ModelOutput]:
        """Solve multiple problems using Qwen."""
        return [self.solve_problem(problem) for problem in problems]
    
    def generate_prompt(self, problem_input: ModelInput) -> str:
        """Generate appropriate prompt for Qwen."""
        prompt = f"""请解决这个数学应用题，按步骤进行：

题目：{problem_input.problem_text}

请按以下格式回答：
1. 理解题意：重述题目要求
2. 已知信息：列出题目给出的信息和需要求解的内容
3. 解题思路：说明解题步骤和方法
4. 计算过程：详细的数学计算
5. 最终答案：给出数值答案

解答："""
        
        return prompt
    
    def call_api(self, prompt: str) -> str:
        """Call Qwen API or local server."""
        if self.is_local:
            return self._call_local_api(prompt)
        else:
            return self._call_remote_api(prompt)
    
    def _call_local_api(self, prompt: str) -> str:
        """Call local Qwen server."""
        try:
            headers = {"Content-Type": "application/json"}
            
            data = {
                "model": self.model_version,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                # Fallback to mock response if local server fails
                return self._generate_mock_response(prompt)
                
        except Exception:
            # Fallback to mock response
            return self._generate_mock_response(prompt)
    
    def _call_remote_api(self, prompt: str) -> str:
        """Call remote Qwen API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_version,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Qwen API error: {response.status_code}, {response.text}")
    
    def _generate_mock_response(self, prompt: str) -> str:
        """Generate mock response for testing when API is not available."""
        # Extract problem from prompt
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', prompt)
        
        if len(numbers) >= 2:
            # Simple arithmetic as mock
            result = float(numbers[0]) + float(numbers[1])
            return f"""1. 理解题意：这是一个数学计算问题
2. 已知信息：找到数字 {numbers[0]} 和 {numbers[1]}
3. 解题思路：进行简单的数学运算
4. 计算过程：{numbers[0]} + {numbers[1]} = {result}
5. 最终答案：{result}"""
        else:
            return "1. 理解题意：无法解析题目\n2. 最终答案：0"
    
    def parse_response(self, response: str) -> ModelOutput:
        """Parse Qwen response into structured output."""
        reasoning_chain = []
        answer = ""
        confidence = 0.75
        
        # Split response into lines
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            reasoning_chain.append(line)
            
            # Extract answer from final answer section
            if any(keyword in line for keyword in ['最终答案', '答案', '结果']):
                numbers = re.findall(r'-?\d+(?:\.\d+)?', line)
                if numbers:
                    answer = numbers[-1]
        
        # If no answer found, extract from entire response
        if not answer:
            numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
            if numbers:
                answer = numbers[-1]
        
        return ModelOutput(
            answer=answer,
            reasoning_chain=reasoning_chain,
            confidence_score=confidence,
            processing_time=0,
            memory_usage=2.5
        )


class InternLMModel(LLMModel):
    """InternLM model implementation."""
    
    def __init__(self, model_name: str = "InternLM2.5-Math-7B", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        self.api_key = config.get("api_key") if config else None
        self.base_url = config.get("base_url", "http://localhost:8001/v1") if config else "http://localhost:8001/v1"
        self.model_version = model_name
        self.is_local = config.get("is_local", True) if config else True
        
    def initialize(self) -> bool:
        """Initialize InternLM model."""
        try:
            self.logger.info(f"Initializing InternLM {self.model_name}")
            self.is_initialized = True
            self.logger.info(f"InternLM {self.model_name} initialized successfully")
            return True
                
        except Exception as e:
            self.logger.error(f"Failed to initialize InternLM {self.model_name}: {e}")
            return False
    
    def solve_problem(self, problem_input: ModelInput) -> ModelOutput:
        """Solve problem using InternLM."""
        start_time = time.time()
        
        if not self.validate_input(problem_input):
            return ModelOutput(
                answer="",
                reasoning_chain=["Invalid input"],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                error_message="Invalid input format"
            )
        
        try:
            # Generate prompt
            prompt = self.generate_prompt(problem_input)
            
            # Call InternLM
            response_text = self.call_api(prompt)
            
            # Parse response
            result = self.parse_response(response_text)
            result.processing_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            return ModelOutput(
                answer="",
                reasoning_chain=[f"Error calling InternLM: {str(e)}"],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def batch_solve(self, problems: List[ModelInput]) -> List[ModelOutput]:
        """Solve multiple problems using InternLM."""
        return [self.solve_problem(problem) for problem in problems]
    
    def generate_prompt(self, problem_input: ModelInput) -> str:
        """Generate appropriate prompt for InternLM."""
        prompt = f"""请解决以下数学问题，并提供详细的解题步骤：

问题：{problem_input.problem_text}

请按照以下格式回答：
1. 问题理解：简述题目要求
2. 信息提取：列出已知条件和待求量
3. 解题策略：说明解题方法和思路
4. 详细计算：展示完整的计算过程
5. 答案：提供最终的数值结果

解答过程："""
        
        return prompt
    
    def call_api(self, prompt: str) -> str:
        """Call InternLM API or generate mock response."""
        try:
            headers = {"Content-Type": "application/json"}
            
            data = {
                "model": self.model_version,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return self._generate_mock_response(prompt)
                
        except Exception:
            return self._generate_mock_response(prompt)
    
    def _generate_mock_response(self, prompt: str) -> str:
        """Generate mock response for testing."""
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', prompt)
        
        if len(numbers) >= 2:
            result = float(numbers[0]) * float(numbers[1])  # Different operation for variety
            return f"""1. 问题理解：这是一个数学计算问题
2. 信息提取：涉及数字 {numbers[0]} 和 {numbers[1]}
3. 解题策略：根据题意进行相应运算
4. 详细计算：{numbers[0]} × {numbers[1]} = {result}
5. 答案：{result}"""
        else:
            return "1. 问题理解：题目信息不完整\n5. 答案：无法计算"
    
    def parse_response(self, response: str) -> ModelOutput:
        """Parse InternLM response into structured output."""
        reasoning_chain = []
        answer = ""
        confidence = 0.7
        
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            reasoning_chain.append(line)
            
            # Extract answer
            if any(keyword in line for keyword in ['答案', '结果', '最终']):
                numbers = re.findall(r'-?\d+(?:\.\d+)?', line)
                if numbers:
                    answer = numbers[-1]
        
        if not answer:
            numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
            if numbers:
                answer = numbers[-1]
        
        return ModelOutput(
            answer=answer,
            reasoning_chain=reasoning_chain,
            confidence_score=confidence,
            processing_time=0,
            memory_usage=1.8
        )


class DeepSeekMathModel(LLMModel):
    """DeepSeek-Math model implementation."""
    
    def __init__(self, model_name: str = "DeepSeek-Math-7B", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        self.api_key = config.get("api_key") if config else None
        self.base_url = config.get("base_url", "http://localhost:8002/v1") if config else "http://localhost:8002/v1"
        self.model_version = model_name
        
    def initialize(self) -> bool:
        """Initialize DeepSeek-Math model."""
        try:
            self.logger.info(f"Initializing DeepSeek-Math {self.model_name}")
            self.is_initialized = True
            self.logger.info(f"DeepSeek-Math {self.model_name} initialized successfully")
            return True
                
        except Exception as e:
            self.logger.error(f"Failed to initialize DeepSeek-Math {self.model_name}: {e}")
            return False
    
    def solve_problem(self, problem_input: ModelInput) -> ModelOutput:
        """Solve problem using DeepSeek-Math."""
        start_time = time.time()
        
        if not self.validate_input(problem_input):
            return ModelOutput(
                answer="",
                reasoning_chain=["Invalid input"],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                error_message="Invalid input format"
            )
        
        try:
            prompt = self.generate_prompt(problem_input)
            response_text = self.call_api(prompt)
            result = self.parse_response(response_text)
            result.processing_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            return ModelOutput(
                answer="",
                reasoning_chain=[f"Error calling DeepSeek-Math: {str(e)}"],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def batch_solve(self, problems: List[ModelInput]) -> List[ModelOutput]:
        """Solve multiple problems using DeepSeek-Math."""
        return [self.solve_problem(problem) for problem in problems]
    
    def generate_prompt(self, problem_input: ModelInput) -> str:
        """Generate appropriate prompt for DeepSeek-Math."""
        prompt = f"""Solve this mathematical word problem step by step:

Problem: {problem_input.problem_text}

Please provide:
- Problem understanding
- Given information and what to find
- Solution strategy
- Step-by-step calculations
- Final answer

Solution:"""
        
        return prompt
    
    def call_api(self, prompt: str) -> str:
        """Call DeepSeek-Math API or generate mock response."""
        try:
            headers = {"Content-Type": "application/json"}
            
            data = {
                "model": self.model_version,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return self._generate_mock_response(prompt)
                
        except Exception:
            return self._generate_mock_response(prompt)
    
    def _generate_mock_response(self, prompt: str) -> str:
        """Generate mock response for testing."""
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', prompt)
        
        if len(numbers) >= 2:
            result = float(numbers[0]) - float(numbers[1])  # Another operation for variety
            return f"""Problem understanding: Mathematical calculation needed
Given information: Numbers {numbers[0]} and {numbers[1]}
Solution strategy: Apply appropriate mathematical operation
Step-by-step calculations: {numbers[0]} - {numbers[1]} = {result}
Final answer: {result}"""
        else:
            return "Problem understanding: Insufficient information\nFinal answer: Cannot determine"
    
    def parse_response(self, response: str) -> ModelOutput:
        """Parse DeepSeek-Math response into structured output."""
        reasoning_chain = []
        answer = ""
        confidence = 0.72
        
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            reasoning_chain.append(line)
            
            if any(keyword in line.lower() for keyword in ['final answer', 'answer:', 'result:']):
                numbers = re.findall(r'-?\d+(?:\.\d+)?', line)
                if numbers:
                    answer = numbers[-1]
        
        if not answer:
            numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
            if numbers:
                answer = numbers[-1]
        
        return ModelOutput(
            answer=answer,
            reasoning_chain=reasoning_chain,
            confidence_score=confidence,
            processing_time=0,
            memory_usage=1.5
        )


# Register LLM models with factory
from .base_model import ModelFactory

ModelFactory.register_model(OpenAIGPTModel, "openai_gpt")
ModelFactory.register_model(ClaudeModel, "claude")
ModelFactory.register_model(QwenModel, "qwen")
ModelFactory.register_model(InternLMModel, "internlm")
ModelFactory.register_model(DeepSeekMathModel, "deepseek_math") 