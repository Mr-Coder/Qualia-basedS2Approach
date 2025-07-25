import React, { useState, useMemo } from 'react'
import Layout from '@/components/layout/Layout'
import ErrorBoundary from '@/components/ErrorBoundary'
import InteractivePropertySmartSolver from '@/components/features/InteractivePropertySmartSolver'
import ActivationLearningGuide from '@/components/features/ActivationLearningGuide'
import ActivationErrorAnalysis from '@/components/features/ActivationErrorAnalysis'
import ActivationStrategyAnalysis from '@/components/features/ActivationStrategyAnalysis'
import ActivationAlgorithmRelationship from '@/components/features/ActivationAlgorithmRelationship'
import HistoryPanel from '@/components/features/HistoryPanel'
import KnowledgeMap from '@/components/features/KnowledgeMap'
import EntityRelationshipDiagram from '@/components/features/EntityRelationshipDiagram'
import SimpleEntityDiagram from '@/components/features/SimpleEntityDiagram'
import AlgorithmicEntityDiagram from '@/components/features/AlgorithmicEntityDiagram'
import AlgorithmExecutionTimeline from '@/components/features/AlgorithmExecutionTimeline'
// import AlgorithmDebugger from '@/components/features/AlgorithmDebugger' // 移至测试目录
import IRDAlgorithmVisualization from '@/components/features/IRDAlgorithmVisualization'
// import SimpleQS2Test from '@/components/features/SimpleQS2Test' // 移至测试目录
import PhysicalPropertyVisualization from '@/components/features/PhysicalPropertyVisualization'
import ReasoningStepsVisualization from '@/components/features/ReasoningStepsVisualization'
import PhysicsConstraintVisualization from '@/components/features/PhysicsConstraintVisualization'
import SimplifiedConstraintVisualization from '@/components/features/SimplifiedConstraintVisualization'
import StandalonePhysicalReasoning from '@/components/features/StandalonePhysicalReasoning'
import { useProblemStore } from '@/stores/problemStore'

type TabType = 'smart' | 'knowledge' | 'learning' | 'error' | 'strategy' | 'algorithm' | 'diagram' | 'physics' | 'reasoning'

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('smart')
  const [useSimpleDiagram, setUseSimpleDiagram] = useState(false) // 调试开关
  const { solveResult, currentProblem } = useProblemStore()

  // 生成算法步骤数据
  const algorithmSteps = useMemo(() => {
    // 如果有求解结果，使用实际数据
    if (solveResult?.entities && solveResult.entities.length > 0) {
      const steps = [
        {
          id: 'step1',
          step: 1,
          type: 'entity_recognition' as const,
          description: '识别数学实体：数字、对象、人物',
          highlightedElements: solveResult.entities?.slice(0, 3).map(e => e.id) || [],
          confidence: 0.9,
          timestamp: Date.now()
        },
        {
          id: 'step2', 
          step: 2,
          type: 'relation_discovery' as const,
          description: '发现隐式关系：算术、所有权关系',
          highlightedElements: solveResult.relationships?.slice(0, 2).map(r => r.id || `rel_${r.from}_${r.to}`) || [],
          confidence: 0.8,
          timestamp: Date.now() + 1000
        },
        {
          id: 'step3',
          step: 3, 
          type: 'semantic_analysis' as const,
          description: '语义分析：理解数学概念和上下文',
          highlightedElements: solveResult.entities?.filter(e => e.type === 'operation').map(e => e.id) || [],
          confidence: 0.85,
          timestamp: Date.now() + 2000
        },
        {
          id: 'step4',
          step: 4,
          type: 'confidence_calculation' as const,
          description: '置信度计算：评估解答可靠性',
          highlightedElements: [],
          confidence: solveResult.confidence || 0.75,
          timestamp: Date.now() + 3000
        }
      ]
      
      return steps
    }
    
    // 否则提供示例算法步骤用于演示
    return [
      {
        id: 'demo_step1',
        step: 1,
        type: 'entity_recognition' as const,
        description: '识别数学实体：小明、苹果、数字5、数字3',
        highlightedElements: ['xiaoming', 'apples', 'number_5'],
        confidence: 0.95,
        timestamp: Date.now()
      },
      {
        id: 'demo_step2',
        step: 2,
        type: 'relation_discovery' as const,
        description: '发现隐式关系：拥有关系、算术关系',
        highlightedElements: ['rel_xiaoming_apples', 'rel_number5_apples'],
        confidence: 0.90,
        timestamp: Date.now() + 1000
      },
      {
        id: 'demo_step3',
        step: 3,
        type: 'semantic_analysis' as const,
        description: '语义分析：理解加法运算和数学上下文',
        highlightedElements: ['addition', 'rel_addition_numbers'],
        confidence: 0.88,
        timestamp: Date.now() + 2000
      },
      {
        id: 'demo_step4',
        step: 4,
        type: 'confidence_calculation' as const,
        description: '置信度计算：评估解答的可靠性为92%',
        highlightedElements: ['result_8'],
        confidence: 0.92,
        timestamp: Date.now() + 3000
      }
    ]
  }, [solveResult])

  // 将求解结果转换为算法可视化需要的格式
  const algorithmEntities = useMemo(() => {
    // 如果有求解结果，使用实际数据
    if (solveResult?.entities && solveResult.entities.length > 0) {
      return solveResult.entities.map((entity, index) => ({
        id: entity.id,
        content: entity.value?.toString() || entity.label || `实体${index + 1}`,
        type: entity.type as 'number' | 'object' | 'person' | 'operation' | 'result' | 'constraint',
        semanticWeight: 0.8,
        cognitiveLoad: 0.3 + (index * 0.1),
        attentionPriority: index < 3 ? 'primary' as const : 'secondary' as const,
        confidence: 0.8 + Math.random() * 0.2
      }))
    }
    
    // 否则提供示例数据用于演示
    return [
      {
        id: 'xiaoming',
        content: '小明',
        type: 'person' as const,
        semanticWeight: 0.9,
        cognitiveLoad: 0.3,
        attentionPriority: 'primary' as const,
        confidence: 0.95
      },
      {
        id: 'apples',
        content: '苹果',
        type: 'object' as const,
        semanticWeight: 0.8,
        cognitiveLoad: 0.4,
        attentionPriority: 'primary' as const,
        confidence: 0.90
      },
      {
        id: 'number_5',
        content: '5',
        type: 'number' as const,
        semanticWeight: 0.7,
        cognitiveLoad: 0.2,
        attentionPriority: 'primary' as const,
        confidence: 0.98
      },
      {
        id: 'number_3',
        content: '3',
        type: 'number' as const,
        semanticWeight: 0.7,
        cognitiveLoad: 0.2,
        attentionPriority: 'secondary' as const,
        confidence: 0.98
      },
      {
        id: 'addition',
        content: '加法',
        type: 'operation' as const,
        semanticWeight: 0.6,
        cognitiveLoad: 0.5,
        attentionPriority: 'secondary' as const,
        confidence: 0.85
      },
      {
        id: 'result_8',
        content: '8',
        type: 'result' as const,
        semanticWeight: 0.9,
        cognitiveLoad: 0.3,
        attentionPriority: 'primary' as const,
        confidence: 0.92
      }
    ]
  }, [solveResult?.entities])

  const algorithmRelationships = useMemo(() => {
    // 如果有求解结果，使用实际数据
    if (solveResult?.relationships && solveResult.relationships.length > 0) {
      return solveResult.relationships.map((rel, index) => ({
        id: rel.id || `rel_${rel.from}_${rel.to}`,
        source: rel.from,
        target: rel.to, 
        type: rel.type as 'arithmetic' | 'ownership' | 'constraint' | 'causal',
        strength: rel.weight || 0.7,
        confidence: 0.7 + Math.random() * 0.3,
        discoveryStep: index + 1
      }))
    }
    
    // 否则提供示例关系用于演示
    return [
      {
        id: 'rel_xiaoming_apples',
        source: 'xiaoming',
        target: 'apples',
        type: 'ownership' as const,
        strength: 0.9,
        confidence: 0.95,
        discoveryStep: 1
      },
      {
        id: 'rel_number5_apples',
        source: 'number_5',
        target: 'apples',
        type: 'arithmetic' as const,
        strength: 0.8,
        confidence: 0.90,
        discoveryStep: 1
      },
      {
        id: 'rel_number3_apples',
        source: 'number_3',
        target: 'apples',
        type: 'arithmetic' as const,
        strength: 0.8,
        confidence: 0.90,
        discoveryStep: 1
      },
      {
        id: 'rel_addition_numbers',
        source: 'addition',
        target: 'result_8',
        type: 'causal' as const,
        strength: 0.95,
        confidence: 0.98,
        discoveryStep: 2
      },
      {
        id: 'rel_numbers_addition',
        source: 'number_5',
        target: 'addition',
        type: 'constraint' as const,
        strength: 0.7,
        confidence: 0.85,
        discoveryStep: 2
      },
      {
        id: 'rel_numbers_addition2',
        source: 'number_3',
        target: 'addition',
        type: 'constraint' as const,
        strength: 0.7,
        confidence: 0.85,
        discoveryStep: 2
      }
    ]
  }, [solveResult?.relationships])

  const renderContent = () => {
    switch (activeTab) {
      case 'smart':
        return (
          <div className="space-y-8">
            <InteractivePropertySmartSolver />
            <div className="max-w-md mx-auto">
              <HistoryPanel />
            </div>
          </div>
        )
      case 'knowledge':
        return <KnowledgeMap />
      case 'learning':
        return <ActivationLearningGuide />
      case 'error':
        return <ActivationErrorAnalysis />
      case 'strategy':
        return <ActivationStrategyAnalysis />
      case 'algorithm':
        return (
          <div className="space-y-6">
            {/* 算法执行时间线 */}
            <div className="bg-white rounded-lg border">
              <div className="p-4 border-b">
                <h3 className="font-semibold text-gray-800">⏱️ 算法执行时间线</h3>
                <p className="text-sm text-gray-600">可视化展示COT-DIR算法的执行过程和各步骤耗时</p>
              </div>
              <div className="p-4">
                <AlgorithmExecutionTimeline steps={algorithmSteps} />
              </div>
            </div>
            
            {/* IRD算法可视化 */}
            <div className="bg-white rounded-lg border">
              <div className="p-4 border-b">
                <h3 className="font-semibold text-gray-800">🔍 隐式关系发现过程</h3>
                <p className="text-sm text-gray-600">深入展示IRD（隐式关系发现）算法如何工作</p>
              </div>
              <div className="p-4">
                <IRDAlgorithmVisualization 
                  entities={algorithmEntities}
                  relations={algorithmRelationships}
                />
              </div>
            </div>
          </div>
        )
      case 'physics':
        return (
          <div className="space-y-6">
            {/* 物理约束验证 */}
            <div className="bg-white rounded-lg border">
              <div className="p-4 border-b">
                <h3 className="font-semibold text-gray-800">⚛️ 物理约束验证</h3>
                <p className="text-sm text-gray-600">基于守恒定律和数学约束的深度验证</p>
              </div>
              <div className="p-4">
                <PhysicsConstraintVisualization
                  problemText={currentProblem}
                  showGuides={true}
                  showValidation={true}
                />
              </div>
            </div>

            {/* 物性图谱深度分析 */}
            <div className="bg-white rounded-lg border">
              <div className="p-4 border-b">
                <h3 className="font-semibold text-gray-800">🔬 物性图谱深度分析</h3>
                <p className="text-sm text-gray-600">展示数学问题中的物理属性和约束关系</p>
              </div>
              <div className="p-4">
                <PhysicalPropertyVisualization
                  problemText={currentProblem}
                  enableRealTimeUpdate={true}
                  showValidation={true}
                  width={900}
                  height={500}
                />
              </div>
            </div>
          </div>
        )
      case 'diagram':
        return (
          <div className="space-y-6">
            {/* 实体关系图 - 核心组件 */}
            <div className="bg-white rounded-lg border">
              <div className="p-4 border-b">
                <h3 className="font-semibold text-gray-800">🧩 实体关系结构图</h3>
                <p className="text-sm text-gray-600">清晰展示问题中的实体、数量和它们之间的关系</p>
                <div className="mt-2">
                  <button
                    onClick={() => setUseSimpleDiagram(!useSimpleDiagram)}
                    className="text-xs px-3 py-1 bg-blue-100 text-blue-700 rounded-full hover:bg-blue-200"
                  >
                    {useSimpleDiagram ? '使用完整版' : '使用简化版（调试）'}
                  </button>
                </div>
              </div>
              <div className="p-4">
                <ErrorBoundary>
                  {useSimpleDiagram ? (
                    <SimpleEntityDiagram
                      entities={solveResult?.entities || [
                        { id: 'demo_xiaoming', name: '小明', type: 'person' },
                        { id: 'demo_apples', name: '苹果', type: 'object' },
                        { id: 'demo_num5', name: '5个', type: 'concept' },
                        { id: 'demo_num3', name: '3个', type: 'concept' }
                      ]}
                      relationships={solveResult?.relationships || [
                        { source: 'demo_xiaoming', target: 'demo_apples', type: '拥有关系', weight: 0.9 },
                        { source: 'demo_apples', target: 'demo_num5', type: '初始数量', weight: 0.8 },
                        { source: 'demo_apples', target: 'demo_num3', type: '增加数量', weight: 0.8 }
                      ]}
                    />
                  ) : (
                    <EntityRelationshipDiagram
                    entities={solveResult?.entities || [
                      { id: 'demo_xiaoming', name: '小明', type: 'person' },
                      { id: 'demo_apples', name: '苹果', type: 'object' },
                      { id: 'demo_num5', name: '5个', type: 'concept' },
                      { id: 'demo_num3', name: '3个', type: 'concept' }
                    ]}
                    relationships={solveResult?.relationships || [
                      { source: 'demo_xiaoming', target: 'demo_apples', type: '拥有关系', weight: 0.9 },
                      { source: 'demo_apples', target: 'demo_num5', type: '初始数量', weight: 0.8 },
                      { source: 'demo_apples', target: 'demo_num3', type: '增加数量', weight: 0.8 }
                    ]}
                    problemText={currentProblem || "小明有5个苹果，妈妈又给了他3个苹果，现在小明一共有几个苹果？"}
                    diagramMode="simple"
                    width={900}
                    height={500}
                  />
                  )}
                </ErrorBoundary>
              </div>
            </div>

            {/* 算法实体图 - 展示算法视角 */}
            <div className="bg-white rounded-lg border">
              <div className="p-4 border-b">
                <h3 className="font-semibold text-gray-800">🎯 算法实体分析</h3>
                <p className="text-sm text-gray-600">从算法角度展示实体的语义权重和认知负载</p>
              </div>
              <div className="p-4">
                <AlgorithmicEntityDiagram 
                  entities={algorithmEntities}
                  relations={algorithmRelationships}
                />
              </div>
            </div>
          </div>
        )
      case 'reasoning':
        return (
          <div className="space-y-6">
            <StandalonePhysicalReasoning />
          </div>
        )
      default:
        return <SmartSolver />
    }
  }

  return (
    <Layout activeTab={activeTab} setActiveTab={setActiveTab}>
      {renderContent()}
    </Layout>
  )
}

export default App