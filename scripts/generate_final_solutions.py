import json
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from final_cotdir_generator import FinalCOTDIRGenerator, RelationBasedSolution


def generate_all_solutions():
    """
    Uses the definitive V4 generator to re-process all problems and create a
    final, corrected solutions file.
    """
    print("🚀 Starting full regeneration of all solutions using FinalCOTDIRGenerator...")
    
    source_file = Path('full_relation_solutions_20250630_024146.json')
    if not source_file.exists():
        print(f"❌ ERROR: Source problem file not found at '{source_file}'. Cannot proceed.")
        return

    # --- Load Source Problems ---
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            problems = data.get('solutions', [])
        print(f"📚 Found {len(problems)} problems to process from '{source_file}'.")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"❌ ERROR: Could not read problems from '{source_file}'. Malformed file? Error: {e}")
        return

    # --- Initialize the Correct Generator ---
    generator = FinalCOTDIRGenerator()
    
    # --- Process All Problems ---
    start_time = time.time()
    final_solutions = []
    
    for problem in tqdm(problems, desc="🤖 Generating new solutions"):
        solution_obj = generator.generate_solution_for_problem(problem)
        # Convert the main object and all nested objects to dicts for JSON serialization
        solution_dict = solution_obj.__dict__
        solution_dict['relations'] = [rel.__dict__ for rel in solution_obj.relations]
        final_solutions.append(solution_dict)

    end_time = time.time()
    
    # --- Prepare Final Output ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"final_corrected_solutions_{timestamp}.json"
    
    final_output = {
        "metadata": {
            "generation_script": "generate_final_solutions.py",
            "generator_used": "FinalCOTDIRGenerator (V4)",
            "source_file": str(source_file),
            "total_problems": len(final_solutions),
            "timestamp": datetime.now().isoformat(),
            "total_processing_time_seconds": round(end_time - start_time, 2),
        },
        "solutions": final_solutions
    }
    
    # --- Save to File ---
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
        
    print("\n" + "="*80)
    print("✅ SUCCESS! Full regeneration complete.")
    print(f"📄 New master solutions file created: {output_filename}")
    print("="*80)

if __name__ == "__main__":
    generate_all_solutions() 