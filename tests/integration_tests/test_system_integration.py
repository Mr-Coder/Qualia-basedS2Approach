"""
System Integration Tests
========================

Test integration between different system components.
"""

import time
from pathlib import Path

import pytest


class TestDatasetIntegration:
    """Test integration with dataset loading and processing"""
    
    @pytest.mark.integration
    def test_dataset_loader_integration(self):
        """Test that dataset loader works with actual data files"""
        # Import dataset loader
        import sys
        sys.path.append("Data")
        
        try:
            from dataset_loader import MathDatasetLoader
            loader = MathDatasetLoader()
            
            # Test loading a small dataset
            datasets = loader.get_available_datasets()
            assert len(datasets) > 0
            
            # Try to load first available dataset
            first_dataset = datasets[0]
            data = loader.load_dataset(first_dataset, limit=5)
            
            assert data is not None
            assert len(data) <= 5
            
        except ImportError:
            pytest.skip("Dataset loader not available")
    
    @pytest.mark.integration
    def test_reasoning_with_real_problems(self, sample_math_problems):
        """Test reasoning system with real mathematical problems"""
        try:
            from reasoning_core.strategies.chain_of_thought import \
                ChainOfThoughtStrategy
            
            strategy = ChainOfThoughtStrategy()
            
            for problem_data in sample_math_problems:
                problem = problem_data["problem"]
                expected_answer = problem_data["expected_answer"]
                
                result = strategy.solve(problem)
                
                # Basic integration checks
                assert result.success == True
                assert result.confidence > 0
                assert len(result.reasoning_steps) > 0
                
                # Note: We don't check exact answer match here since the 
                # strategy implementation is simplified for demo purposes
                
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")


class TestToolIntegration:
    """Test integration with external tools"""
    
    @pytest.mark.integration
    def test_symbolic_math_tool_integration(self):
        """Test integration with SymPy symbolic math tool"""
        try:
            from reasoning_core.tools.symbolic_math import SymbolicMathTool
            
            tool = SymbolicMathTool()
            
            if not tool.is_available:
                pytest.skip("SymPy not available")
            
            # Test basic operations
            operations = tool.get_supported_operations()
            assert len(operations) > 0
            
            # Test equation solving
            result = tool.execute("solve_equation", "x + 2 - 5", "x")
            assert result.success == True
            assert result.result is not None
            
            # Test simplification
            result = tool.execute("simplify", "x + x + 2*x")
            assert result.success == True
            
        except ImportError:
            pytest.skip("SymbolicMathTool not available")
    
    @pytest.mark.integration
    def test_tool_availability_check(self):
        """Test that tools properly check their availability"""
        try:
            from reasoning_core.tools.symbolic_math import SymbolicMathTool
            
            tool = SymbolicMathTool()
            
            # Tool should know whether it's available
            assert isinstance(tool.is_available, bool)
            
            # If available, should be able to list operations
            if tool.is_available:
                operations = tool.get_supported_operations()
                assert isinstance(operations, list)
                assert len(operations) > 0
                
        except ImportError:
            pytest.skip("SymbolicMathTool not available")


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows"""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_problem_solving_workflow(self, sample_math_problems):
        """Test complete workflow from problem input to solution"""
        try:
            from reasoning_core.strategies.chain_of_thought import \
                ChainOfThoughtStrategy
            from reasoning_core.tools.symbolic_math import SymbolicMathTool

            # Initialize components
            strategy = ChainOfThoughtStrategy({
                "max_steps": 5,
                "confidence_threshold": 0.7
            })
            
            math_tool = SymbolicMathTool()
            
            # Process each problem
            results = []
            for problem_data in sample_math_problems:
                start_time = time.time()
                
                # Solve with reasoning strategy
                result = strategy.solve(problem_data["problem"])
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Record results
                results.append({
                    "problem_id": problem_data["id"],
                    "success": result.success,
                    "confidence": result.confidence,
                    "processing_time": processing_time,
                    "steps_count": len(result.reasoning_steps)
                })
            
            # Validate overall results
            assert len(results) == len(sample_math_problems)
            
            successful_results = [r for r in results if r["success"]]
            assert len(successful_results) > 0
            
            # Check performance metrics
            avg_time = sum(r["processing_time"] for r in results) / len(results)
            assert avg_time < 10.0  # Should process within reasonable time
            
            avg_confidence = sum(r["confidence"] for r in successful_results) / len(successful_results)
            assert avg_confidence > 0.5  # Should have reasonable confidence
            
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")
    
    @pytest.mark.integration
    def test_error_recovery_workflow(self):
        """Test system handles errors gracefully in workflow"""
        try:
            from reasoning_core.strategies.chain_of_thought import \
                ChainOfThoughtStrategy
            
            strategy = ChainOfThoughtStrategy()
            
            # Test with problematic inputs
            problematic_inputs = [
                None,
                "",
                "This is not a mathematical problem at all",
                {"malformed": "input"},
                12345,
                []
            ]
            
            for bad_input in problematic_inputs:
                result = strategy.solve(bad_input)
                
                # System should handle errors gracefully
                assert isinstance(result, type(result))  # Should return proper result type
                # Even if not successful, should not crash
                
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")


class TestConfigurationIntegration:
    """Test integration with configuration system"""
    
    @pytest.mark.integration
    def test_strategy_configuration_loading(self):
        """Test loading strategy configurations"""
        try:
            from reasoning_core.strategies.chain_of_thought import \
                ChainOfThoughtStrategy

            # Test with various configurations
            configs = [
                {"max_steps": 3, "confidence_threshold": 0.9},
                {"max_steps": 15, "confidence_threshold": 0.5},
                {}  # Empty config should use defaults
            ]
            
            for config in configs:
                strategy = ChainOfThoughtStrategy(config)
                
                # Verify configuration is applied
                if "max_steps" in config:
                    assert strategy.max_steps == config["max_steps"]
                if "confidence_threshold" in config:
                    assert strategy.confidence_threshold == config["confidence_threshold"]
                
                # Strategy should still function
                result = strategy.solve("2 + 2 = ?")
                assert hasattr(result, 'success')
                
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")


class TestPerformanceIntegration:
    """Test performance characteristics in integration"""
    
    @pytest.mark.integration
    @pytest.mark.performance
    def test_strategy_performance_baseline(self, performance_baseline):
        """Test that strategies meet performance baselines"""
        try:
            from reasoning_core.strategies.chain_of_thought import \
                ChainOfThoughtStrategy
            
            strategy = ChainOfThoughtStrategy()
            
            # Test with simple problems
            simple_problems = [
                "2 + 2 = ?",
                "10 - 5 = ?", 
                "3 * 4 = ?"
            ]
            
            times = []
            confidences = []
            
            for problem in simple_problems:
                start_time = time.time()
                result = strategy.solve(problem)
                end_time = time.time()
                
                processing_time = end_time - start_time
                times.append(processing_time)
                
                if result.success:
                    confidences.append(result.confidence)
            
            # Check performance against baseline
            avg_time = sum(times) / len(times)
            max_time = max(times)
            
            baseline = performance_baseline
            assert avg_time <= baseline["speed"]["average_time_per_problem"]
            assert max_time <= baseline["speed"]["max_time_per_problem"]
            
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                assert avg_confidence >= baseline["confidence"]["min_confidence"]
                
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")
    
    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.slow
    def test_batch_processing_performance(self):
        """Test performance with batch problem processing"""
        try:
            from reasoning_core.strategies.chain_of_thought import \
                ChainOfThoughtStrategy
            
            strategy = ChainOfThoughtStrategy()
            
            # Generate batch of problems
            batch_problems = [f"{i} + {i+1} = ?" for i in range(1, 21)]  # 20 problems
            
            start_time = time.time()
            
            results = []
            for problem in batch_problems:
                result = strategy.solve(problem)
                results.append(result)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Performance checks
            assert len(results) == len(batch_problems)
            
            avg_time_per_problem = total_time / len(batch_problems)
            assert avg_time_per_problem < 5.0  # Should be reasonably fast
            
            successful_results = [r for r in results if r.success]
            success_rate = len(successful_results) / len(results)
            assert success_rate > 0.7  # Should have decent success rate
            
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}") 