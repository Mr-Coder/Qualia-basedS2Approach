"""
ReasoningPipeline 多用例推理主线演示
"""

def demo_reasoning_pipeline_cases():
    from src.reasoning_core.reasoning_pipeline import ReasoningPipeline
    pipeline = ReasoningPipeline()
    cases = [
        {"problem": "Mary has 5 apples. She gives 2 apples to John. How many apples does Mary have left?", "expected_answer": 3},
        {"problem": "There are 12 birds in a tree. 3 more birds join them. How many birds are there now?", "expected_answer": 15},
        {"problem": "A store has 20 books. They sell 8 books. How many books remain?", "expected_answer": 12},
        {"problem": "John has 5 apples. He buys 3 more apples. How many apples does John have in total?", "expected_answer": 8},
        {"problem": "Sarah had 15 stickers. She gave 6 stickers to her friend. How many stickers does Sarah have left?", "expected_answer": 9},
        {"problem": "If a pizza has 8 slices and John eats 3 slices, how many slices are left?", "expected_answer": 5}
    ]
    print("\n🧩 ReasoningPipeline 统一推理主线多用例演示:")
    for i, prob in enumerate(cases, 1):
        print(f"\n--- 问题 {i} ---\n问题: {prob['problem']}")
        result = pipeline.solve(prob, mode='auto')
        final_answer = getattr(result, 'final_answer', None) or getattr(result, 'answer', None) or (result.get('final_answer', None) if isinstance(result, dict) else None)
        print(f"答案: {final_answer}")
        print(f"期望: {prob['expected_answer']}")
    print("\n✅ ReasoningPipeline 多用例 demo 完成!\n")

if __name__ == "__main__":
    demo_reasoning_pipeline_cases() 