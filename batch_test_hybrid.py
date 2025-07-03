#!/usr/bin/env python3
"""
Batch test script for comparing HybridModel vs SimplePatternModel performance.
"""

import json
import re
import time
from typing import Any, Dict, List

from tqdm import tqdm

from Data.dataset_loader import MathDatasetLoader
from src.models.base_model import ModelInput
from src.models.model_manager import ModelManager


def extract_numerical_answer(answer_str: str) -> float:
    """Extract numerical answer from string."""
    try:
        # Try to find the first number in the string
        match = re.search(r'[-+]?\d*\.\d+|\d+', str(answer_str))
        if match:
            return float(match.group())
        return None
    except (ValueError, TypeError):
        return None


def run_batch_test(model_name: str, dataset_name: str = "SVAMP", max_samples: int = 50) -> Dict[str, Any]:
    """
    Run batch test on a specific model.
    
    Args:
        model_name: Name of the model to test
        dataset_name: Dataset to test on
        max_samples: Maximum number of samples to test
        
    Returns:
        Dictionary with test results
    """
    print(f"Starting batch test on '{dataset_name}' dataset with '{model_name}' (max {max_samples} samples).")
    
    # Initialize model manager
    manager = ModelManager()
    
    # Initialize the model
    success = manager.initialize_model(model_name)
    if not success:
        print(f"FATAL: Could not initialize model '{model_name}'. Aborting.")
        return None
    
    # Load dataset
    try:
        loader = MathDatasetLoader()
        if dataset_name == "SVAMP":
            raw_dataset = loader.load_dataset(dataset_name)
            if max_samples and len(raw_dataset) > max_samples:
                raw_dataset = raw_dataset[:max_samples]
            
            # Convert SVAMP format manually
            dataset = []
            for item in raw_dataset:
                problem_text = f"{item.get('Body', '')} {item.get('Question', '')}".strip()
                answer_value = item.get('Answer', 0)
                dataset.append({
                    "id": item.get('ID', f"svamp_{len(dataset)}"),
                    "problem": problem_text,
                    "answer": str(answer_value)
                })
        else:
            dataset = loader.create_unified_format(dataset_name)
            if max_samples and len(dataset) > max_samples:
                dataset = dataset[:max_samples]
    except Exception as e:
        print(f"FATAL: Failed to load dataset '{dataset_name}': {e}")
        return None
    
    # Run the test
    correct_count = 0
    failed_samples = []
    pattern_success_count = 0
    llm_fallback_count = 0
    
    print(f"Running {len(dataset)} problems through the solver...")
    for item in tqdm(dataset, desc=f"Processing with {model_name}"):
        problem_text = item.get("problem", "")
        expected_answer_str = item.get("answer", "")
        
        # Clean up expected answer
        try:
            match = re.search(r'[-+]?\d*\.\d+|\d+', str(expected_answer_str))
            if match:
                expected_answer = float(match.group())
            else:
                continue
        except (ValueError, TypeError):
            continue
        
        problem_input = ModelInput(
            problem_text=problem_text, 
            problem_id=item.get("id"), 
            expected_answer=str(expected_answer)
        )
        
        # Solve the problem
        result = manager.solve_problem(model_name, problem_input)
        
        if result and result.answer:
            # Extract numerical answer from result
            predicted_answer = extract_numerical_answer(result.answer)
            
            if predicted_answer is not None:
                # Check if answer is correct (within small tolerance for floating point)
                if abs(predicted_answer - expected_answer) < 0.01:
                    correct_count += 1
                else:
                    failed_samples.append({
                        "id": item.get("id"),
                        "problem": problem_text,
                        "expected_answer": str(expected_answer),
                        "predicted_answer": result.answer,
                        "confidence": result.confidence_score,
                        "processing_time": result.processing_time,
                        "metadata": result.metadata
                    })
            else:
                failed_samples.append({
                    "id": item.get("id"),
                    "problem": problem_text,
                    "expected_answer": str(expected_answer),
                    "predicted_answer": result.answer,
                    "confidence": result.confidence_score,
                    "processing_time": result.processing_time,
                    "metadata": result.metadata
                })
        else:
            failed_samples.append({
                "id": item.get("id"),
                "problem": problem_text,
                "expected_answer": str(expected_answer),
                "predicted_answer": "",
                "confidence": 0.0,
                "processing_time": 0.0,
                "metadata": {}
            })
        
        # Track solver statistics for hybrid model
        if model_name == "equation_baseline" and result and result.metadata:
            if result.metadata.get("llm_fallback_used", False):
                llm_fallback_count += 1
            else:
                pattern_success_count += 1
    
    # Calculate accuracy
    accuracy = (correct_count / len(dataset)) * 100 if dataset else 0
    
    # Save failed samples
    failure_file = f'failure_analysis_{model_name}.json'
    with open(failure_file, 'w', encoding='utf-8') as f:
        json.dump(failed_samples, f, indent=2, ensure_ascii=False)
    
    # Print results
    print("\n" + "="*50)
    print(f"                  BATCH TEST RESULTS")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Problems Tested: {len(dataset)}")
    print(f"Correct Answers: {correct_count}")
    print(f"Incorrect Answers: {len(failed_samples)}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    if model_name == "equation_baseline":
        print(f"Pattern Solver Success: {pattern_success_count}")
        print(f"LLM Fallback Used: {llm_fallback_count}")
        print(f"Pattern Success Rate: {(pattern_success_count/len(dataset)*100):.2f}%")
        print(f"LLM Fallback Rate: {(llm_fallback_count/len(dataset)*100):.2f}%")
    
    print(f"\nSaved {len(failed_samples)} failed samples to '{failure_file}' for analysis.")
    print("="*50)
    
    return {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "total_problems": len(dataset),
        "correct_count": correct_count,
        "accuracy": accuracy,
        "pattern_success_count": pattern_success_count,
        "llm_fallback_count": llm_fallback_count,
        "failure_file": failure_file
    }


def compare_models():
    """Compare SimplePatternModel vs HybridModel performance."""
    
    print("🔬 Model Comparison: SimplePatternModel vs HybridModel")
    print("=" * 60)
    
    # Test both models
    results = {}
    
    # Test SimplePatternModel
    print("\n📊 Testing SimplePatternModel...")
    results["simple_pattern"] = run_batch_test("template_baseline", max_samples=50)
    
    # Test HybridModel
    print("\n📊 Testing HybridModel...")
    results["hybrid"] = run_batch_test("equation_baseline", max_samples=50)
    
    # Comparison summary
    print("\n" + "="*60)
    print("                    COMPARISON SUMMARY")
    print("="*60)
    
    if results["simple_pattern"] and results["hybrid"]:
        simple_acc = results["simple_pattern"]["accuracy"]
        hybrid_acc = results["hybrid"]["accuracy"]
        improvement = hybrid_acc - simple_acc
        
        print(f"SimplePatternModel Accuracy: {simple_acc:.2f}%")
        print(f"HybridModel Accuracy: {hybrid_acc:.2f}%")
        print(f"Improvement: {improvement:+.2f}%")
        
        if results["hybrid"]["llm_fallback_count"] > 0:
            llm_rate = (results["hybrid"]["llm_fallback_count"] / results["hybrid"]["total_problems"]) * 100
            print(f"LLM Fallback Rate: {llm_rate:.2f}%")
            print(f"Pattern Solver Success Rate: {100-llm_rate:.2f}%")
        
        print("\n💡 Analysis:")
        if improvement > 0:
            print(f"   ✅ HybridModel shows {improvement:.2f}% improvement")
            print(f"   🎯 LLM fallback successfully handled additional cases")
        else:
            print(f"   ⚠️  No improvement observed (Mock LLM returns fixed answer)")
            print(f"   🔧 Consider implementing real LLM integration")
    
    print("="*60)


if __name__ == "__main__":
    compare_models() 