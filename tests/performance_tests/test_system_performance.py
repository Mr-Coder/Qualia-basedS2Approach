"""
System Performance Tests
========================

Test system performance and benchmark against requirements.
"""

import statistics
import time
from typing import Any, Dict, List

import pytest


class TestReasoningPerformance:
    """Test performance of reasoning strategies"""
    
    @pytest.mark.performance
    def test_chain_of_thought_speed(self, sample_math_problems, performance_baseline):
        """Test Chain of Thought reasoning speed"""
        try:
            from reasoning_core.strategies.chain_of_thought import \
                ChainOfThoughtStrategy
            
            strategy = ChainOfThoughtStrategy()
            times = []
            
            # Run multiple iterations for better statistics
            for _ in range(3):
                for problem_data in sample_math_problems:
                    start_time = time.time()
                    result = strategy.solve(problem_data["problem"])
                    end_time = time.time()
                    
                    processing_time = end_time - start_time
                    times.append(processing_time)
            
            # Calculate statistics
            avg_time = statistics.mean(times)
            max_time = max(times)
            min_time = min(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            
            # Check against baseline
            baseline = performance_baseline["speed"]
            assert avg_time <= baseline["average_time_per_problem"], \
                f"Average time {avg_time:.3f}s exceeds baseline {baseline['average_time_per_problem']}s"
            assert max_time <= baseline["max_time_per_problem"], \
                f"Max time {max_time:.3f}s exceeds baseline {baseline['max_time_per_problem']}s"
            
            # Log performance metrics
            print(f"\nPerformance Metrics:")
            print(f"Average time: {avg_time:.3f}s")
            print(f"Min time: {min_time:.3f}s")
            print(f"Max time: {max_time:.3f}s")
            print(f"Standard deviation: {std_time:.3f}s")
            
        except ImportError:
            pytest.skip("ChainOfThoughtStrategy not available")
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_batch_processing_performance(self):
        """Test performance with large batch of problems"""
        try:
            from reasoning_core.strategies.chain_of_thought import \
                ChainOfThoughtStrategy
            
            strategy = ChainOfThoughtStrategy()
            
            # Generate large batch
            batch_size = 50
            problems = [f"Calculate {i} + {i+1}" for i in range(batch_size)]
            
            start_time = time.time()
            
            results = []
            for problem in problems:
                result = strategy.solve(problem)
                results.append(result)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Performance checks
            avg_time_per_problem = total_time / batch_size
            throughput = batch_size / total_time  # problems per second
            
            # Requirements
            assert avg_time_per_problem < 1.0, f"Average time per problem too high: {avg_time_per_problem:.3f}s"
            assert throughput > 1.0, f"Throughput too low: {throughput:.2f} problems/second"
            
            print(f"\nBatch Performance Metrics:")
            print(f"Total time: {total_time:.3f}s")
            print(f"Average time per problem: {avg_time_per_problem:.3f}s")
            print(f"Throughput: {throughput:.2f} problems/second")
            
        except ImportError:
            pytest.skip("ChainOfThoughtStrategy not available")
    
    @pytest.mark.performance
    def test_memory_usage_stability(self):
        """Test that memory usage remains stable during processing"""
        try:
            import os

            import psutil

            from reasoning_core.strategies.chain_of_thought import \
                ChainOfThoughtStrategy
            
            strategy = ChainOfThoughtStrategy()
            process = psutil.Process(os.getpid())
            
            # Measure initial memory
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process many problems
            problems = [f"What is {i} * {i+1}?" for i in range(100)]
            
            memory_measurements = []
            for i, problem in enumerate(problems):
                result = strategy.solve(problem)
                
                # Measure memory every 10 problems
                if i % 10 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_measurements.append(current_memory)
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Check memory growth
            memory_growth = final_memory - initial_memory
            max_memory = max(memory_measurements)
            
            # Memory should not grow excessively
            assert memory_growth < 100, f"Memory growth too high: {memory_growth:.2f}MB"
            assert max_memory < initial_memory + 200, f"Peak memory usage too high: {max_memory:.2f}MB"
            
            print(f"\nMemory Usage Metrics:")
            print(f"Initial memory: {initial_memory:.2f}MB")
            print(f"Final memory: {final_memory:.2f}MB")
            print(f"Memory growth: {memory_growth:.2f}MB")
            print(f"Peak memory: {max_memory:.2f}MB")
            
        except ImportError:
            pytest.skip("psutil or ChainOfThoughtStrategy not available")


class TestEvaluationPerformance:
    """Test performance of evaluation system"""
    
    @pytest.mark.performance
    def test_evaluation_speed(self):
        """Test evaluation system performance"""
        try:
            from evaluation.evaluator import ComprehensiveEvaluator
            
            evaluator = ComprehensiveEvaluator()
            
            # Generate test data
            predictions = [f"answer_{i}" for i in range(100)]
            ground_truth = [f"answer_{i}" for i in range(100)]
            
            start_time = time.time()
            result = evaluator.evaluate(predictions, ground_truth, "test_dataset", "test_model")
            end_time = time.time()
            
            evaluation_time = end_time - start_time
            
            # Should evaluate quickly
            assert evaluation_time < 5.0, f"Evaluation too slow: {evaluation_time:.3f}s"
            
            print(f"\nEvaluation Performance:")
            print(f"Evaluation time: {evaluation_time:.3f}s")
            print(f"Items per second: {len(predictions) / evaluation_time:.1f}")
            
        except ImportError:
            pytest.skip("ComprehensiveEvaluator not available")
    
    @pytest.mark.performance
    def test_batch_evaluation_performance(self):
        """Test batch evaluation performance"""
        try:
            from evaluation.evaluator import ComprehensiveEvaluator
            
            evaluator = ComprehensiveEvaluator()
            
            # Generate multiple datasets
            num_datasets = 5
            batch_predictions = []
            batch_ground_truth = []
            dataset_names = []
            
            for i in range(num_datasets):
                predictions = [f"answer_{j}" for j in range(20)]
                ground_truth = [f"answer_{j}" for j in range(20)]
                
                batch_predictions.append(predictions)
                batch_ground_truth.append(ground_truth)
                dataset_names.append(f"dataset_{i}")
            
            start_time = time.time()
            results = evaluator.evaluate_batch(
                batch_predictions, batch_ground_truth, 
                dataset_names, "test_model"
            )
            end_time = time.time()
            
            batch_evaluation_time = end_time - start_time
            total_items = sum(len(pred) for pred in batch_predictions)
            
            # Performance requirements
            assert batch_evaluation_time < 10.0, f"Batch evaluation too slow: {batch_evaluation_time:.3f}s"
            assert len(results) == num_datasets, "Should return result for each dataset"
            
            print(f"\nBatch Evaluation Performance:")
            print(f"Total time: {batch_evaluation_time:.3f}s")
            print(f"Datasets processed: {num_datasets}")
            print(f"Total items: {total_items}")
            print(f"Items per second: {total_items / batch_evaluation_time:.1f}")
            
        except ImportError:
            pytest.skip("ComprehensiveEvaluator not available")


class TestDatasetLoadingPerformance:
    """Test dataset loading performance"""
    
    @pytest.mark.performance
    def test_dataset_loading_speed(self):
        """Test dataset loading performance"""
        try:
            import sys
            sys.path.append("Data")
            from dataset_loader import MathDatasetLoader
            
            loader = MathDatasetLoader()
            
            # Test loading different dataset sizes
            test_limits = [10, 50, 100]
            
            for limit in test_limits:
                start_time = time.time()
                
                # Load first available dataset
                datasets = loader.get_available_datasets()
                if datasets:
                    data = loader.load_dataset(datasets[0], limit=limit)
                    
                    end_time = time.time()
                    loading_time = end_time - start_time
                    
                    # Performance requirements
                    items_per_second = limit / loading_time if loading_time > 0 else float('inf')
                    assert items_per_second > 10, f"Loading too slow: {items_per_second:.1f} items/sec"
                    
                    print(f"\nDataset Loading Performance (limit={limit}):")
                    print(f"Loading time: {loading_time:.3f}s")
                    print(f"Items per second: {items_per_second:.1f}")
            
        except ImportError:
            pytest.skip("MathDatasetLoader not available")


class TestStressTests:
    """Stress tests for system reliability"""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_extended_operation_stability(self):
        """Test system stability under extended operation"""
        try:
            from reasoning_core.strategies.chain_of_thought import \
                ChainOfThoughtStrategy
            
            strategy = ChainOfThoughtStrategy()
            
            # Run for extended period
            num_iterations = 200
            problems = [
                "Calculate 5 + 3",
                "What is 10 - 4?", 
                "Solve 2 * 6",
                "Find 15 / 3"
            ]
            
            success_count = 0
            error_count = 0
            times = []
            
            for i in range(num_iterations):
                problem = problems[i % len(problems)]
                
                try:
                    start_time = time.time()
                    result = strategy.solve(problem)
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
                    
                    if result.success:
                        success_count += 1
                    else:
                        error_count += 1
                        
                except Exception:
                    error_count += 1
            
            # Calculate stability metrics
            success_rate = success_count / num_iterations
            avg_time = statistics.mean(times) if times else 0
            time_stability = statistics.stdev(times) if len(times) > 1 else 0
            
            # Stability requirements
            assert success_rate > 0.95, f"Success rate too low: {success_rate:.3f}"
            assert time_stability < avg_time, f"Time variance too high: {time_stability:.3f}"
            
            print(f"\nStability Test Results:")
            print(f"Iterations: {num_iterations}")
            print(f"Success rate: {success_rate:.3f}")
            print(f"Average time: {avg_time:.3f}s")
            print(f"Time stability (std): {time_stability:.3f}s")
            
        except ImportError:
            pytest.skip("ChainOfThoughtStrategy not available")
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_concurrent_processing_performance(self):
        """Test performance under concurrent processing"""
        try:
            import queue
            import threading

            from reasoning_core.strategies.chain_of_thought import \
                ChainOfThoughtStrategy
            
            strategy = ChainOfThoughtStrategy()
            
            # Setup for concurrent processing
            num_threads = 4
            problems_per_thread = 10
            results_queue = queue.Queue()
            
            def worker(thread_id: int):
                """Worker function for each thread"""
                thread_results = []
                
                for i in range(problems_per_thread):
                    problem = f"Thread {thread_id}: Calculate {i} + {i+1}"
                    
                    start_time = time.time()
                    result = strategy.solve(problem)
                    end_time = time.time()
                    
                    thread_results.append({
                        'thread_id': thread_id,
                        'problem_id': i,
                        'success': result.success,
                        'time': end_time - start_time
                    })
                
                results_queue.put(thread_results)
            
            # Start concurrent processing
            start_time = time.time()
            threads = []
            
            for thread_id in range(num_threads):
                thread = threading.Thread(target=worker, args=(thread_id,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Collect results
            all_results = []
            while not results_queue.empty():
                thread_results = results_queue.get()
                all_results.extend(thread_results)
            
            # Analyze performance
            total_problems = num_threads * problems_per_thread
            throughput = total_problems / total_time
            success_rate = sum(1 for r in all_results if r['success']) / len(all_results)
            
            # Performance requirements
            assert throughput > 2.0, f"Concurrent throughput too low: {throughput:.2f} problems/sec"
            assert success_rate > 0.95, f"Concurrent success rate too low: {success_rate:.3f}"
            
            print(f"\nConcurrent Processing Performance:")
            print(f"Threads: {num_threads}")
            print(f"Total problems: {total_problems}")
            print(f"Total time: {total_time:.3f}s")
            print(f"Throughput: {throughput:.2f} problems/sec")
            print(f"Success rate: {success_rate:.3f}")
            
        except ImportError:
            pytest.skip("ChainOfThoughtStrategy not available") 