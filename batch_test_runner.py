import json
import logging
import re
import sys
from pathlib import Path

from tqdm import tqdm

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from Data.dataset_loader import MathDatasetLoader
from src.models.base_model import ModelInput
from src.models.model_manager import ModelManager


def setup_logging():
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

def run_batch_test(dataset_name: str, max_samples: int = 100):
    """
    Runs a batch test on a specified dataset and collects failed samples.
    """
    print(f"Starting batch test on '{dataset_name}' dataset (max {max_samples} samples)...")
    
    # 1. Initialize ModelManager and the solver
    manager = ModelManager()
    model_name = "simple_pattern_solver"
    if not manager.initialize_model(model_name):
        print(f"FATAL: Could not initialize model '{model_name}'. Aborting.")
        return

    # 2. Load the dataset
    try:
        loader = MathDatasetLoader()
        # For SVAMP dataset, we'll load it directly instead of using create_unified_format
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
            # Use the unified format for other datasets
            dataset = loader.create_unified_format(dataset_name)
            if max_samples and len(dataset) > max_samples:
                dataset = dataset[:max_samples]
    except Exception as e:
        print(f"FATAL: Failed to load dataset '{dataset_name}': {e}")
        return

    # 3. Run the test
    correct_count = 0
    failed_samples = []

    print(f"Running {len(dataset)} problems through the solver...")
    for item in tqdm(dataset, desc="Processing problems"):
        problem_text = item.get("problem", "")
        expected_answer_str = item.get("answer", "")
        
        # Clean up expected answer
        try:
            # Extract the first numerical value from the string
            match = re.search(r'[-+]?\d*\.\d+|\d+', str(expected_answer_str))
            if match:
                expected_answer = float(match.group())
            else:
                # If no number is found, we can't test it.
                print(f"DEBUG: No number found in answer: '{expected_answer_str}'")
                continue
        except (ValueError, TypeError):
            # This will help us debug if there are unexpected formats
            print(f"DEBUG: Could not parse expected answer: '{expected_answer_str}'")
            continue

        problem_input = ModelInput(problem_text=problem_text, problem_id=item.get("id"), expected_answer=str(expected_answer))
        
        result = manager.solve_problem(model_name, problem_input)
        
        # 4. Compare results
        is_correct = False
        try:
            # Compare as floats
            predicted_answer = float(result.answer)
            if abs(predicted_answer - expected_answer) < 1e-4: # Tolerance for float comparison
                is_correct = True
        except (ValueError, TypeError):
            is_correct = False

        if is_correct:
            correct_count += 1
        else:
            failed_samples.append({
                "id": item.get("id"),
                "problem": problem_text,
                "expected_answer": expected_answer_str,
                "predicted_answer": result.answer,
                "reasoning_steps": result.reasoning_chain
            })
            
    # 5. Report results
    accuracy = (correct_count / len(dataset)) * 100 if dataset else 0
    print("\n" + "="*50)
    print(" " * 18 + "BATCH TEST RESULTS")
    print("="*50)
    print(f"Dataset: {dataset_name}")
    print(f"Problems Tested: {len(dataset)}")
    print(f"Correct Answers: {correct_count}")
    print(f"Incorrect Answers: {len(failed_samples)}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # 6. Save failed samples for analysis
    if failed_samples:
        failure_file = "failure_analysis.json"
        with open(failure_file, 'w', encoding='utf-8') as f:
            json.dump(failed_samples, f, indent=2)
        print(f"\nSaved {len(failed_samples)} failed samples to '{failure_file}' for analysis.")
    
    print("="*50)

if __name__ == "__main__":
    setup_logging()
    # We use 'SVAMP' as it's a good benchmark for simple word problems
    run_batch_test(dataset_name="SVAMP", max_samples=200) 