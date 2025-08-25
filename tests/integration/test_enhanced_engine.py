#!/usr/bin/env python3
"""
Test script for Enhanced IRD Engine
"""
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_enhanced_engine():
    """Test the Enhanced IRD Engine"""
    print("🔍 Testing Enhanced IRD Engine...")
    print("=" * 50)
    
    try:
        from reasoning.qs2_enhancement.enhanced_ird_engine import EnhancedIRDEngine
        
        # Initialize engine
        config = {
            "min_strength_threshold": 0.3,
            "max_relations_per_entity": 5,
            "enable_parallel_processing": True
        }
        
        engine = EnhancedIRDEngine(config)
        print("✅ Engine initialized successfully")
        
        # Test problems
        test_problems = [
            "小明有10个苹果，给了小红3个，还剩多少个？",
            "一辆汽车以60公里/小时的速度行驶2小时，行驶了多少公里？",
            "班级有40个学生，其中60%是男生，男生有多少人？"
        ]
        
        print(f"\n📝 Testing {len(test_problems)} problems...")
        
        for i, problem in enumerate(test_problems, 1):
            print(f"\n--- Problem {i} ---")
            print(f"Text: {problem}")
            
            # Discover relations
            result = engine.discover_relations(problem)
            
            print(f"✅ Relations found: {len(result.relations)}")
            print(f"⏱️  Processing time: {result.processing_time:.3f}s")
            print(f"📊 Entity count: {result.entity_count}")
            print(f"🎯 High strength relations: {result.high_strength_relations}")
            
            # Show top relations
            for j, relation in enumerate(result.relations[:3], 1):
                print(f"  🔗 Relation {j}: {relation.entity1} -> {relation.entity2}")
                print(f"     Type: {relation.relation_type.value}")
                print(f"     Strength: {relation.strength:.2f}")
                print(f"     Confidence: {relation.confidence:.2f}")
                print(f"     Evidence: {len(relation.evidence)} items")
        
        # Show global statistics
        print(f"\n📈 Global Statistics:")
        stats = engine.get_global_stats()
        print(f"  Total discoveries: {stats['total_discoveries']}")
        print(f"  Total relations found: {stats['total_relations_found']}")
        print(f"  Average processing time: {stats['average_processing_time']:.3f}s")
        
        print(f"\n🎉 Enhanced IRD Engine test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_engine()
    sys.exit(0 if success else 1)