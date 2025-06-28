#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced Configuration System

This module provides comprehensive configuration management for the mathematical
reasoning system, supporting different experimental setups, model parameters,
and evaluation metrics.

Features:
1. Hierarchical configuration structure
2. Validation and type checking
3. Environment-specific configurations
4. Dynamic parameter adjustment
5. Configuration versioning

Author: AI Research Team
Date: 2025-01-31
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass


class ExperimentMode(Enum):
    """Experiment execution modes."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    EVALUATION = "evaluation"
    PRODUCTION = "production"
    RESEARCH = "research"


class ModelType(Enum):
    """Mathematical reasoning model types."""
    COT_DIR = "cot_dir"
    BASELINE = "baseline"
    HYBRID = "hybrid"
    EXPERIMENTAL = "experimental"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    # Logging levels
    log_level: str = "INFO"
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    
    # Log files
    log_directory: str = "logs"
    log_filename: str = "math_reasoning.log"
    max_log_size: int = 10485760  # 10MB
    backup_count: int = 5
    
    # Log format
    detailed_format: bool = True
    include_timestamp: bool = True
    include_process_info: bool = True
    
    def validate(self) -> bool:
        """Validate logging configuration."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        
        if self.log_level not in valid_levels:
            raise ConfigurationError(f"Invalid log_level: {self.log_level}")
        
        if self.console_level not in valid_levels:
            raise ConfigurationError(f"Invalid console_level: {self.console_level}")
        
        if self.file_level not in valid_levels:
            raise ConfigurationError(f"Invalid file_level: {self.file_level}")
        
        return True


@dataclass
class NLPConfig:
    """NLP processing configuration."""
    # Entity extraction settings
    number_precision: int = 6
    unit_detection_window: int = 10
    entity_confidence_threshold: float = 0.7
    
    # Pattern matching settings
    enable_advanced_patterns: bool = True
    pattern_weight_syntax: float = 0.4
    pattern_weight_pos: float = 0.3
    pattern_weight_keyword: float = 0.3
    
    # Language model settings
    use_pretrained_models: bool = False
    model_cache_size: int = 1000
    
    def validate(self) -> bool:
        """Validate NLP configuration."""
        if not 0 <= self.entity_confidence_threshold <= 1:
            raise ConfigurationError("entity_confidence_threshold must be between 0 and 1")
        
        if not (self.pattern_weight_syntax + self.pattern_weight_pos + self.pattern_weight_keyword == 1.0):
            raise ConfigurationError("Pattern weights must sum to 1.0")
        
        return True


@dataclass
class RelationDiscoveryConfig:
    """Implicit relation discovery configuration."""
    # Discovery thresholds
    confidence_threshold: float = 0.6
    arithmetic_threshold: float = 0.8
    proportion_threshold: float = 0.7
    unit_conversion_threshold: float = 0.9
    temporal_threshold: float = 0.7
    constraint_threshold: float = 0.6
    
    # Discovery algorithms
    enable_pattern_matching: bool = True
    enable_semantic_analysis: bool = True
    enable_dependency_analysis: bool = True
    
    # Performance settings
    max_relations_per_problem: int = 20
    relation_deduplication: bool = True
    
    def validate(self) -> bool:
        """Validate relation discovery configuration."""
        thresholds = [
            self.confidence_threshold, self.arithmetic_threshold,
            self.proportion_threshold, self.unit_conversion_threshold,
            self.temporal_threshold, self.constraint_threshold
        ]
        
        for threshold in thresholds:
            if not 0 <= threshold <= 1:
                raise ConfigurationError(f"All thresholds must be between 0 and 1")
        
        return True


@dataclass
class ReasoningConfig:
    """Multi-level reasoning configuration."""
    # Reasoning parameters
    max_reasoning_steps: int = 10
    reasoning_depth_limit: int = 5
    confidence_propagation: bool = True
    
    # Step execution settings
    enable_parallel_execution: bool = False
    step_timeout_seconds: float = 30.0
    error_recovery_attempts: int = 3
    
    # Template settings
    use_custom_templates: bool = True
    template_adaptation: bool = True
    
    # Optimization settings
    enable_step_caching: bool = True
    cache_size: int = 500
    
    def validate(self) -> bool:
        """Validate reasoning configuration."""
        if self.max_reasoning_steps <= 0:
            raise ConfigurationError("max_reasoning_steps must be positive")
        
        if self.step_timeout_seconds <= 0:
            raise ConfigurationError("step_timeout_seconds must be positive")
        
        return True


@dataclass
class VerificationConfig:
    """Chain verification configuration."""
    # Verification settings
    enable_verification: bool = True
    strict_mode: bool = False
    mathematical_precision: float = 1e-6
    
    # Verification components
    enable_logical_check: bool = True
    enable_mathematical_check: bool = True
    enable_consistency_check: bool = True
    enable_completeness_check: bool = True
    
    # Error handling
    fail_on_verification_error: bool = False
    verification_timeout: float = 10.0
    
    def validate(self) -> bool:
        """Validate verification configuration."""
        if self.mathematical_precision <= 0:
            raise ConfigurationError("mathematical_precision must be positive")
        
        if self.verification_timeout <= 0:
            raise ConfigurationError("verification_timeout must be positive")
        
        return True


@dataclass
class EvaluationConfig:
    """Evaluation and testing configuration."""
    # Evaluation metrics
    enable_accuracy_metrics: bool = True
    enable_performance_metrics: bool = True
    enable_robustness_metrics: bool = True
    enable_efficiency_metrics: bool = True
    
    # Testing settings
    test_data_path: str = "data/test_problems.json"
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    
    # Benchmarking
    benchmark_iterations: int = 10
    warmup_iterations: int = 3
    enable_memory_profiling: bool = True
    
    # Output settings
    save_detailed_results: bool = True
    generate_visualizations: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["json", "csv", "txt"])
    
    def validate(self) -> bool:
        """Validate evaluation configuration."""
        if not 0 < self.validation_split < 1:
            raise ConfigurationError("validation_split must be between 0 and 1")
        
        if self.cross_validation_folds <= 1:
            raise ConfigurationError("cross_validation_folds must be greater than 1")
        
        valid_formats = {"json", "csv", "txt", "xlsx", "png", "pdf"}
        for fmt in self.export_formats:
            if fmt not in valid_formats:
                raise ConfigurationError(f"Invalid export format: {fmt}")
        
        return True


@dataclass
class ExperimentConfig:
    """Experiment-specific configuration."""
    # Experiment metadata
    experiment_name: str = "default_experiment"
    experiment_version: str = "1.0.0"
    description: str = "Mathematical reasoning experiment"
    
    # Execution settings
    mode: ExperimentMode = ExperimentMode.DEVELOPMENT
    model_type: ModelType = ModelType.COT_DIR
    random_seed: int = 42
    
    # Resource limits
    max_memory_mb: int = 4096
    max_execution_time: float = 300.0  # 5 minutes
    parallel_workers: int = 1
    
    # Data settings
    input_data_path: str = "data/problems"
    output_data_path: str = "results"
    cache_enabled: bool = True
    
    def validate(self) -> bool:
        """Validate experiment configuration."""
        if self.max_memory_mb <= 0:
            raise ConfigurationError("max_memory_mb must be positive")
        
        if self.max_execution_time <= 0:
            raise ConfigurationError("max_execution_time must be positive")
        
        if self.parallel_workers < 1:
            raise ConfigurationError("parallel_workers must be at least 1")
        
        return True


@dataclass
class AdvancedConfiguration:
    """Main configuration class containing all sub-configurations."""
    
    # Sub-configurations
    nlp: NLPConfig = field(default_factory=NLPConfig)
    relation_discovery: RelationDiscoveryConfig = field(default_factory=RelationDiscoveryConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    # Global settings
    debug_mode: bool = False
    verbose: bool = False
    config_version: str = "2.0.0"
    
    def validate(self) -> bool:
        """Validate entire configuration."""
        try:
            self.nlp.validate()
            self.relation_discovery.validate()
            self.reasoning.validate()
            self.verification.validate()
            self.evaluation.validate()
            self.logging.validate()
            self.experiment.validate()
            return True
        except ConfigurationError as e:
            print(f"Configuration validation error: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """Export configuration to JSON."""
        config_dict = self.to_dict()
        
        # Convert enums to strings
        config_dict['experiment']['mode'] = self.experiment.mode.value
        config_dict['experiment']['model_type'] = self.experiment.model_type.value
        
        json_str = json.dumps(config_dict, indent=2, default=str)
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
        
        return json_str
    
    def to_yaml(self, filepath: Optional[str] = None) -> str:
        """Export configuration to YAML."""
        config_dict = self.to_dict()
        
        # Convert enums to strings
        config_dict['experiment']['mode'] = self.experiment.mode.value
        config_dict['experiment']['model_type'] = self.experiment.model_type.value
        
        yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(yaml_str)
        
        return yaml_str
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AdvancedConfiguration':
        """Create configuration from dictionary."""
        config = cls()
        
        # Update each sub-configuration
        if 'nlp' in config_dict:
            config.nlp = NLPConfig(**config_dict['nlp'])
        
        if 'relation_discovery' in config_dict:
            config.relation_discovery = RelationDiscoveryConfig(**config_dict['relation_discovery'])
        
        if 'reasoning' in config_dict:
            config.reasoning = ReasoningConfig(**config_dict['reasoning'])
        
        if 'verification' in config_dict:
            config.verification = VerificationConfig(**config_dict['verification'])
        
        if 'evaluation' in config_dict:
            config.evaluation = EvaluationConfig(**config_dict['evaluation'])
        
        if 'logging' in config_dict:
            config.logging = LoggingConfig(**config_dict['logging'])
        
        if 'experiment' in config_dict:
            exp_config = config_dict['experiment'].copy()
            # Convert string enums back to enum objects
            if 'mode' in exp_config:
                exp_config['mode'] = ExperimentMode(exp_config['mode'])
            if 'model_type' in exp_config:
                exp_config['model_type'] = ModelType(exp_config['model_type'])
            
            config.experiment = ExperimentConfig(**exp_config)
        
        # Update global settings
        if 'debug_mode' in config_dict:
            config.debug_mode = config_dict['debug_mode']
        if 'verbose' in config_dict:
            config.verbose = config_dict['verbose']
        if 'config_version' in config_dict:
            config.config_version = config_dict['config_version']
        
        return config
    
    @classmethod
    def from_json(cls, filepath: str) -> 'AdvancedConfiguration':
        """Load configuration from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'AdvancedConfiguration':
        """Load configuration from YAML file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


class ConfigurationManager:
    """Manages configuration loading, validation, and environment-specific settings."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Default configurations for different environments
        self.default_configs = {
            "development": self._create_development_config(),
            "testing": self._create_testing_config(),
            "evaluation": self._create_evaluation_config(),
            "production": self._create_production_config(),
            "research": self._create_research_config()
        }
    
    def get_config(self, environment: str = "development") -> AdvancedConfiguration:
        """Get configuration for specific environment."""
        if environment in self.default_configs:
            return self.default_configs[environment]
        
        # Try to load from file
        config_file = self.config_dir / f"{environment}.yaml"
        if config_file.exists():
            return AdvancedConfiguration.from_yaml(str(config_file))
        
        # Fallback to development config
        print(f"Warning: Configuration for '{environment}' not found. Using development config.")
        return self.default_configs["development"]
    
    def save_config(self, config: AdvancedConfiguration, environment: str, format: str = "yaml") -> str:
        """Save configuration to file."""
        if format.lower() == "yaml":
            filepath = self.config_dir / f"{environment}.yaml"
            config.to_yaml(str(filepath))
        elif format.lower() == "json":
            filepath = self.config_dir / f"{environment}.json"
            config.to_json(str(filepath))
        else:
            raise ConfigurationError(f"Unsupported format: {format}")
        
        return str(filepath)
    
    def _create_development_config(self) -> AdvancedConfiguration:
        """Create development configuration."""
        config = AdvancedConfiguration()
        config.experiment.mode = ExperimentMode.DEVELOPMENT
        config.debug_mode = True
        config.verbose = True
        config.logging.log_level = "DEBUG"
        config.reasoning.max_reasoning_steps = 5
        config.evaluation.benchmark_iterations = 3
        return config
    
    def _create_testing_config(self) -> AdvancedConfiguration:
        """Create testing configuration."""
        config = AdvancedConfiguration()
        config.experiment.mode = ExperimentMode.TESTING
        config.debug_mode = True
        config.verbose = False
        config.logging.log_level = "INFO"
        config.reasoning.max_reasoning_steps = 8
        config.evaluation.benchmark_iterations = 5
        config.verification.strict_mode = True
        return config
    
    def _create_evaluation_config(self) -> AdvancedConfiguration:
        """Create evaluation configuration."""
        config = AdvancedConfiguration()
        config.experiment.mode = ExperimentMode.EVALUATION
        config.debug_mode = False
        config.verbose = False
        config.logging.log_level = "INFO"
        config.reasoning.max_reasoning_steps = 10
        config.evaluation.benchmark_iterations = 10
        config.evaluation.enable_memory_profiling = True
        config.evaluation.generate_visualizations = True
        return config
    
    def _create_production_config(self) -> AdvancedConfiguration:
        """Create production configuration."""
        config = AdvancedConfiguration()
        config.experiment.mode = ExperimentMode.PRODUCTION
        config.debug_mode = False
        config.verbose = False
        config.logging.log_level = "WARNING"
        config.reasoning.max_reasoning_steps = 12
        config.reasoning.enable_step_caching = True
        config.verification.enable_verification = True
        config.evaluation.save_detailed_results = False
        return config
    
    def _create_research_config(self) -> AdvancedConfiguration:
        """Create research configuration."""
        config = AdvancedConfiguration()
        config.experiment.mode = ExperimentMode.RESEARCH
        config.debug_mode = True
        config.verbose = True
        config.logging.log_level = "DEBUG"
        config.reasoning.max_reasoning_steps = 15
        config.relation_discovery.enable_semantic_analysis = True
        config.relation_discovery.enable_dependency_analysis = True
        config.evaluation.benchmark_iterations = 20
        config.evaluation.cross_validation_folds = 10
        config.evaluation.generate_visualizations = True
        return config
    
    def create_custom_config(self, base_environment: str = "development", **overrides) -> AdvancedConfiguration:
        """Create custom configuration with overrides."""
        base_config = self.get_config(base_environment)
        config_dict = base_config.to_dict()
        
        # Apply overrides
        def update_nested_dict(d, overrides):
            for key, value in overrides.items():
                if '.' in key:
                    # Handle nested keys like 'nlp.confidence_threshold'
                    parts = key.split('.')
                    current = d
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = value
                else:
                    d[key] = value
        
        update_nested_dict(config_dict, overrides)
        
        return AdvancedConfiguration.from_dict(config_dict)
    
    def validate_all_configs(self) -> Dict[str, bool]:
        """Validate all default configurations."""
        results = {}
        for env_name, config in self.default_configs.items():
            try:
                results[env_name] = config.validate()
            except Exception as e:
                print(f"Validation failed for {env_name}: {e}")
                results[env_name] = False
        
        return results


# Example usage and testing
if __name__ == "__main__":
    print("🔧 Advanced Configuration System Demo")
    print("=" * 50)
    
    # Create configuration manager
    config_manager = ConfigurationManager()
    
    # Test all configurations
    print("Testing all default configurations...")
    validation_results = config_manager.validate_all_configs()
    
    for env, is_valid in validation_results.items():
        status = "✅ VALID" if is_valid else "❌ INVALID"
        print(f"{env}: {status}")
    
    print("\n" + "=" * 50)
    
    # Demonstrate configuration usage
    print("Getting evaluation configuration...")
    eval_config = config_manager.get_config("evaluation")
    print(f"Max reasoning steps: {eval_config.reasoning.max_reasoning_steps}")
    print(f"Verification enabled: {eval_config.verification.enable_verification}")
    print(f"Benchmark iterations: {eval_config.evaluation.benchmark_iterations}")
    
    # Save configuration example
    print("\nSaving configuration to file...")
    saved_path = config_manager.save_config(eval_config, "evaluation_example")
    print(f"Configuration saved to: {saved_path}")
    
    # Create custom configuration
    print("\nCreating custom configuration...")
    custom_config = config_manager.create_custom_config(
        base_environment="development",
        **{
            "reasoning.max_reasoning_steps": 20,
            "nlp.entity_confidence_threshold": 0.9,
            "experiment.experiment_name": "custom_experiment"
        }
    )
    
    print(f"Custom config - Max steps: {custom_config.reasoning.max_reasoning_steps}")
    print(f"Custom config - Entity threshold: {custom_config.nlp.entity_confidence_threshold}")
    print(f"Custom config - Experiment name: {custom_config.experiment.experiment_name}")
    
    print("\n🎯 Configuration system demo completed!")
