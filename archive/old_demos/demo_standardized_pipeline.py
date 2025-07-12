"""
标准化数据流 end-to-end 演示：加载→预处理→推理→评测
"""

def demo_standardized_pipeline():
    print("\n" + "="*60)
    print("🚀 标准化数据流 End-to-End Demo")
    print("="*60)
    try:
        from src.data.loader import DataLoader
        from src.data.preprocessor import Preprocessor
        from src.evaluation.evaluator import Evaluator
        from src.bridge.reasoning_bridge import ReasoningEngine
    except ImportError as e:
        print(f"导入标准化接口失败: {e}")
        return False

    # 1. 加载数据
    loader = DataLoader()
    # 以 Math23K 为例，加载前5个样本
    try:
        raw_samples = loader.load(dataset_name="Math23K", max_samples=5)
    except Exception as e:
        print(f"数据加载失败: {e}")
        return False
    print(f"加载样本数: {len(raw_samples)}")

    # 2. 预处理
    preprocessor = Preprocessor()
    processed_samples = [preprocessor.process(s) for s in raw_samples]
    print(f"已完成预处理。示例: {processed_samples[0]}")

    # 3. 推理
    engine = ReasoningEngine()
    predictions = []
    for i, sample in enumerate(processed_samples, 1):
        print(f"\n--- 推理样本 {i} ---\n问题: {sample.get('problem')}")
        result = engine.solve(sample)
        print(f"推理结果: {result}")
        # 兼容多种推理结果格式
        answer = result.get('final_answer') or result.get('answer')
        predictions.append({**sample, 'answer': answer})

    # 4. 评测
    evaluator = Evaluator()
    eval_result = evaluator.evaluate(predictions, raw_samples)
    print(f"\n评测结果: {eval_result}")
    return True

if __name__ == "__main__":
    success = demo_standardized_pipeline()
    if success:
        print(f"\n🎉 标准化数据流 demo completed successfully!")
    else:
        print(f"\n⚠️  标准化数据流 demo failed.") 