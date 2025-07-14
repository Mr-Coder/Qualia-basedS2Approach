# COT-DIR Examples\n\nThis document contains practical examples of using the COT-DIR system.\n\n## Basic Examples\n\n### Simple Arithmetic\n\nSolving basic mathematical operations\n\n```python\nfrom core import solve_problem_unified

# Simple addition
result = solve_problem_unified("What is 15 + 27?")
print(f"Answer: {result['final_answer']}")

# With chain of thought
result = solve_problem_unified(
    "Calculate 12 * 8 + 5",
    strategy="chain_of_thought"
)
print(f"Reasoning steps: {result['reasoning_steps']}")\n```\n\n### Word Problems\n\nSolving mathematical word problems\n\n```python\nfrom core import create_problem_solver

solver = create_problem_solver("chain_of_thought")

problem = """
A store sells apples for $2 each and oranges for $3 each.
If someone buys 5 apples and 3 oranges, how much do they pay in total?
"""

result = solver.solve_problem(problem)
print(f"Solution: {result.final_answer}")
print(f"Confidence: {result.confidence:.2f}")\n```\n\n### Batch Processing\n\nProcessing multiple problems efficiently\n\n```python\nfrom core import create_problem_solver

solver = create_problem_solver("direct_reasoning")

math_problems = [
    "Find the square root of 144",
    "What is 25% of 80?",
    "Solve for x: 2x + 5 = 13",
    "Calculate the perimeter of a rectangle with length 8 and width 5"
]

results = solver.batch_solve(math_problems)

for i, result in enumerate(results):
    print(f"Problem {i+1}: {result.final_answer}")
    print(f"Success: {result.success}")
    print("---")\n```\n