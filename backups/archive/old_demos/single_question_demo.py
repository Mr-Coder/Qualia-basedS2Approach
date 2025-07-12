import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.models.base_model import ModelInput
from src.models.model_manager import ModelManager


class DemoLogger:
    def __init__(self, log_file='demo_output.txt'):
        self.terminal = sys.stdout
        self.log = open(log_file, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def run_demo():
    """
    Runs a demonstration of the math problem-solving system using the ModelManager.
    """
    # Redirect stdout to our custom logger
    sys.stdout = DemoLogger()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Starting single question demonstration using ModelManager.")

    # 1. Initialize ModelManager
    logger.info("Initializing ModelManager...")
    try:
        manager = ModelManager()
    except Exception as e:
        logger.error(f"Failed to initialize ModelManager: {e}", exc_info=True)
        return

    # 2. Initialize the simple_pattern_solver
    model_name = "simple_pattern_solver"
    logger.info(f"Initializing model: {model_name}...")
    if not manager.initialize_model(model_name):
        logger.error(f"Could not initialize {model_name}. Aborting.")
        return

    # 3. Prepare the problem
    problem_text = "Chenny is 10 years old. Alyana is 4 years younger than Chenny. Anne's age is 2 more than Alyana. What is the age of Anne?"
    problem_input = ModelInput(problem_text=problem_text, problem_id="demo-q1")
    logger.info(f"Using problem: {problem_text}")

    # 4. Solve the problem using the manager
    logger.info(f"Solving problem with {model_name}...")
    try:
        result = manager.solve_problem(model_name, problem_input)
    except Exception as e:
        logger.error(f"An error occurred during solving: {e}", exc_info=True)
        return

    # 5. Display results
    logger.info("Demonstration finished. Displaying results.")
    print("\n" + "="*50)
    print(" " * 20 + "FINAL RESULTS")
    print("="*50)
    print(f"\nProblem: {problem_input.problem_text}")
    
    if result:
        print("\n--- Reasoning Steps ---")
        if result.reasoning_chain:
            for i, step in enumerate(result.reasoning_chain):
                print(f"Step {i+1}: {step}")
        else:
            print("No reasoning steps were generated.")
        
        print("\n--- Final Answer ---")
        if result.answer:
            print(f"The calculated answer is: {result.answer}")
            print(f"Confidence: {result.confidence_score}")
        else:
            print("Could not determine the final answer.")
            if result.error_message:
                print(f"Error: {result.error_message}")
    else:
        print("\nNo result was returned from the model.")

    print("\n" + "="*50)

if __name__ == "__main__":
    run_demo()

