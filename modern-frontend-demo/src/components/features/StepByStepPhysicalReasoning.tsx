import React, { useState, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { PhysicalReasoner, buildPhysicalGraphFromProblem, PhysicalGraph, PhysicalEntity, PhysicalRelation } from '@/services/physicalReasoningAPI'
import { ArrowRight, ChevronRight, ChevronLeft, Zap, Network, Target, Lightbulb, AlertCircle } from 'lucide-react'

interface StepData {
  step: number
  title: string
  description: string
  entities: PhysicalEntity[]
  relations: PhysicalRelation[]
  highlight?: {
    entities?: string[]
    relations?: string[]
  }
  explanation: string
}

const StepByStepPhysicalReasoning: React.FC = () => {
  const [selectedExample, setSelectedExample] = useState<number>(0)
  const [currentStep, setCurrentStep] = useState<number>(0)
  const [stepData, setStepData] = useState<StepData[]>([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)

  const examples = [
    {
      title: "苹果数量问题",
      problem: "小明有5个苹果，妈妈又给了他3个苹果，现在小明一共有几个苹果？",
      analysis: "通过物性关系（守恒定律）连接已知和未知"
    },
    {
      title: "水池排水问题", 
      problem: "水池有水100升，每小时流出20升，几小时后水池为空？",
      analysis: "通过时间流逝的物性关系推理"
    },
    {
      title: "购物问题",
      problem: "小红买了3支笔，每支笔5元，她一共花了多少钱？",
      analysis: "通过价值交换的物性关系推理"
    }
  ]

  const analyzeStepByStep = useCallback(() => {
    const problem = examples[selectedExample].problem
    const baseGraph = buildPhysicalGraphFromProblem(problem)
    const steps: StepData[] = []
    
    // 步骤1: 提取实体和直陈关系
    const step1Entities = baseGraph.entities.filter(e => 
      e.type === 'person' || e.type === 'object' || e.type === 'quantity'
    )
    const step1Relations = baseGraph.relations.filter(r => 
      r.type === 'has_quantity'
    )
    
    steps.push({
      step: 1,
      title: "分解题目 → 提取entity",
      description: "识别题目中的人物、物体和数量",
      entities: step1Entities,
      relations: step1Relations,
      highlight: {
        entities: step1Entities.map(e => e.id)
      },
      explanation: `从题目中提取出：
• 人物：${step1Entities.filter(e => e.type === 'person').map(e => e.name).join(', ')}
• 物体：${step1Entities.filter(e => e.type === 'object').map(e => e.name).join(', ')}
• 数量：${step1Entities.filter(e => e.type === 'quantity').map(e => `${e.name}(${e.properties.value}${e.properties.unit || ''})`).join(', ')}`
    })
    
    // 步骤2: 加入概念节点
    const step2Entities = [...step1Entities, ...baseGraph.entities.filter(e => 
      e.type === 'concept'
    )]
    const step2Relations = [...step1Relations, ...baseGraph.relations.filter(r => 
      r.type === 'transformation'
    )]
    
    steps.push({
      step: 2,
      title: "加入物性关系 → 构建graph",
      description: "添加运算概念和转换关系",
      entities: step2Entities,
      relations: step2Relations,
      highlight: {
        entities: baseGraph.entities.filter(e => e.type === 'concept').map(e => e.id),
        relations: baseGraph.relations.filter(r => r.type === 'transformation').map((r, i) => `rel_${i}`)
      },
      explanation: `加入数学概念和物性关系：
• 运算概念：${baseGraph.entities.filter(e => e.type === 'concept').map(e => e.name).join(', ')}
• 建立转换关系，连接已知量和运算`
    })
    
    // 步骤3: 物性关系推理扩展
    const step3Relations = [...step2Relations, ...baseGraph.relations.filter(r => 
      r.type === 'conservation' || r.type === 'causal' || r.type === 'temporal'
    )]
    
    steps.push({
      step: 3,
      title: "物性关系推理 → 扩展连接",
      description: "通过守恒、因果等物理规律扩展关系",
      entities: step2Entities,
      relations: step3Relations,
      highlight: {
        relations: baseGraph.relations.filter(r => 
          r.type === 'conservation' || r.type === 'causal' || r.type === 'temporal'
        ).map((r, i) => `phys_${i}`)
      },
      explanation: `通过物性关系扩展：
• 守恒关系：部分之和等于整体
• 因果关系：原因导致结果
• 时间关系：随时间变化的规律`
    })
    
    // 步骤4: 寻找路径
    const knownEntities = baseGraph.known
    const unknownEntities = baseGraph.unknown
    const expandedGraph = PhysicalReasoner.expandGraphWithPhysicalRelations(baseGraph)
    const paths = PhysicalReasoner.findPathsFromKnownToUnknown(expandedGraph)
    
    steps.push({
      step: 4,
      title: "寻找通路 → 确定求解路径",
      description: "在已知和未知之间寻找连接路径",
      entities: step2Entities,
      relations: step3Relations,
      highlight: {
        entities: [...knownEntities, ...unknownEntities],
        relations: paths.length > 0 ? paths[0].relations.map((r, i) => `path_${i}`) : []
      },
      explanation: `寻找求解路径：
• 已知：${knownEntities.map(k => baseGraph.entities.find(e => e.id === k)?.name).join(', ')}
• 未知：${unknownEntities.map(u => baseGraph.entities.find(e => e.id === u)?.name).join(', ')}
• 找到路径：${paths.length > 0 ? paths[0].path.join(' → ') : '无'}`
    })
    
    // 步骤5: 沿路径推理
    if (paths.length > 0) {
      const bestPath = paths[0]
      const relevantRelations = step3Relations.filter(r => 
        bestPath.path.includes(r.source) && bestPath.path.includes(r.target)
      )
      
      steps.push({
        step: 5,
        title: "沿路径推理 → 得出答案",
        description: "忽略无关推理，专注求解路径",
        entities: step2Entities.filter(e => bestPath.path.includes(e.id)),
        relations: relevantRelations,
        highlight: {
          entities: bestPath.path,
          relations: relevantRelations.map((r, i) => `final_${i}`)
        },
        explanation: `沿最优路径推理：
• 推理路径：${bestPath.reasoning}
• 忽略无关节点和关系
• 得出答案：通过${bestPath.relations.map(r => r.type).join('→')}关系推导`
      })
    }
    
    setStepData(steps)
    setCurrentStep(0)
    setIsAnalyzing(true)
  }, [selectedExample])

  const renderStepGraph = (step: StepData) => {
    const width = 800
    const height = 400
    const centerX = width / 2
    const centerY = height / 2

    // 计算实体位置
    const entityPositions = step.entities.map((entity, index) => {
      const angle = (index * 2 * Math.PI) / step.entities.length
      const radius = 120
      return {
        ...entity,
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle)
      }
    })

    // 获取实体颜色
    const getEntityColor = (entity: any) => {
      const isHighlighted = step.highlight?.entities?.includes(entity.id)
      
      if (entity.id.includes('initial') || entity.id.includes('quantity') || entity.id.includes('unit_price')) {
        return isHighlighted ? '#10b981' : '#86efac' // 绿色 - 已知
      }
      if (entity.id.includes('total')) {
        return isHighlighted ? '#ef4444' : '#fca5a5' // 红色 - 未知
      }
      if (entity.type === 'concept') {
        return isHighlighted ? '#3b82f6' : '#93bbfc' // 蓝色 - 概念
      }
      return isHighlighted ? '#8b5cf6' : '#c4b5fd' // 紫色 - 其他
    }

    // 获取关系颜色和样式
    const getRelationStyle = (rel: any, index: number) => {
      const relId = `${rel.type}_${index}`
      const isHighlighted = step.highlight?.relations?.some(id => 
        relId.includes(id) || id.includes(rel.type)
      )
      
      let color = '#6b7280'
      let dashArray = 'none'
      let width = 2
      
      switch(rel.type) {
        case 'has_quantity':
          color = isHighlighted ? '#6366f1' : '#a5b4fc'
          break
        case 'transformation':
          color = isHighlighted ? '#ec4899' : '#f9a8d4'
          break
        case 'conservation':
          color = isHighlighted ? '#f59e0b' : '#fcd34d'
          dashArray = '5,5'
          break
        case 'causal':
          color = isHighlighted ? '#8b5cf6' : '#c4b5fd'
          dashArray = '8,4'
          break
        case 'temporal':
          color = isHighlighted ? '#06b6d4' : '#67e8f9'
          dashArray = '10,5'
          break
      }
      
      if (isHighlighted) width = 4
      
      return { color, dashArray, width }
    }

    return (
      <svg width={width} height={height} className="border border-gray-200 rounded-lg bg-gradient-to-br from-gray-50 to-blue-50">
        {/* 步骤标题 */}
        <text x={width/2} y={30} textAnchor="middle" fontSize="16" fontWeight="bold" fill="#374151">
          步骤 {step.step}: {step.title}
        </text>
        
        {/* 绘制关系线 */}
        {step.relations.map((rel, index) => {
          const source = entityPositions.find(e => e.id === rel.source)
          const target = entityPositions.find(e => e.id === rel.target)
          
          if (!source || !target) return null
          
          const style = getRelationStyle(rel, index)
          
          // 计算曲线控制点
          const midX = (source.x + target.x) / 2
          const midY = (source.y + target.y) / 2
          const dx = target.x - source.x
          const dy = target.y - source.y
          const distance = Math.sqrt(dx * dx + dy * dy)
          const offset = 20
          const controlX = midX - (dy / distance) * offset
          const controlY = midY + (dx / distance) * offset
          
          return (
            <g key={`rel-${index}`}>
              <path
                d={`M ${source.x} ${source.y} Q ${controlX} ${controlY} ${target.x} ${target.y}`}
                stroke={style.color}
                strokeWidth={style.width}
                fill="none"
                strokeDasharray={style.dashArray}
                opacity="0.8"
              />
              {step.step >= 2 && (
                <text
                  x={controlX}
                  y={controlY}
                  textAnchor="middle"
                  fontSize="10"
                  fill={style.color}
                  fontWeight="500"
                >
                  {rel.type}
                </text>
              )}
            </g>
          )
        })}
        
        {/* 绘制实体节点 */}
        {entityPositions.map((entity) => {
          const isHighlighted = step.highlight?.entities?.includes(entity.id)
          
          return (
            <g key={entity.id}>
              <circle
                cx={entity.x}
                cy={entity.y}
                r={isHighlighted ? 35 : 30}
                fill={getEntityColor(entity)}
                stroke="white"
                strokeWidth="2"
                className={isHighlighted ? "drop-shadow-lg" : "drop-shadow"}
              />
              <text
                x={entity.x}
                y={entity.y - 5}
                textAnchor="middle"
                fontSize="11"
                fill="white"
                fontWeight={isHighlighted ? "bold" : "normal"}
              >
                {entity.name}
              </text>
              {entity.properties.value !== null && entity.properties.value !== undefined && (
                <text
                  x={entity.x}
                  y={entity.y + 8}
                  textAnchor="middle"
                  fontSize="12"
                  fill="white"
                  fontWeight="bold"
                >
                  {entity.properties.value}{entity.properties.unit || ''}
                </text>
              )}
            </g>
          )
        })}
      </svg>
    )
  }

  return (
    <Card className="max-w-5xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Network className="h-6 w-6 text-purple-600" />
          基于物性关系的推理 - 分步演示
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* 例题选择 */}
        <div>
          <h3 className="font-semibold mb-3">选择示例问题：</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {examples.map((example, index) => (
              <button
                key={index}
                onClick={() => {
                  setSelectedExample(index)
                  setIsAnalyzing(false)
                  setCurrentStep(0)
                }}
                className={`p-4 rounded-lg border-2 transition-all ${
                  selectedExample === index 
                    ? 'border-purple-500 bg-purple-50' 
                    : 'border-gray-200 hover:border-purple-300'
                }`}
              >
                <h4 className="font-semibold mb-1">{example.title}</h4>
                <p className="text-sm text-gray-600">{example.problem}</p>
              </button>
            ))}
          </div>
        </div>

        {/* 分析按钮 */}
        {!isAnalyzing && (
          <div className="flex justify-center">
            <Button
              onClick={analyzeStepByStep}
              className="flex items-center gap-2"
              size="lg"
            >
              <Zap className="h-5 w-5" />
              开始分步推理
            </Button>
          </div>
        )}

        {/* 分步展示 */}
        {isAnalyzing && stepData.length > 0 && (
          <div className="space-y-4">
            {/* 步骤导航 */}
            <div className="flex items-center justify-between bg-gray-100 p-4 rounded-lg">
              <Button
                onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
                disabled={currentStep === 0}
                variant="outline"
                size="sm"
              >
                <ChevronLeft className="h-4 w-4 mr-1" />
                上一步
              </Button>
              
              <div className="flex space-x-2">
                {stepData.map((_, index) => (
                  <button
                    key={index}
                    onClick={() => setCurrentStep(index)}
                    className={`w-10 h-10 rounded-full transition-all ${
                      index === currentStep 
                        ? 'bg-purple-600 text-white' 
                        : index < currentStep
                        ? 'bg-purple-200 text-purple-800'
                        : 'bg-gray-300 text-gray-600'
                    }`}
                  >
                    {index + 1}
                  </button>
                ))}
              </div>
              
              <Button
                onClick={() => setCurrentStep(Math.min(stepData.length - 1, currentStep + 1))}
                disabled={currentStep === stepData.length - 1}
                variant="outline"
                size="sm"
              >
                下一步
                <ChevronRight className="h-4 w-4 ml-1" />
              </Button>
            </div>

            {/* 当前步骤内容 */}
            <div className="bg-white p-6 rounded-lg border">
              <h3 className="text-lg font-semibold mb-2 text-purple-700">
                {stepData[currentStep].description}
              </h3>
              
              {/* 图形展示 */}
              <div className="mb-4">
                {renderStepGraph(stepData[currentStep])}
              </div>
              
              {/* 步骤说明 */}
              <div className="bg-purple-50 p-4 rounded-lg">
                <pre className="text-sm whitespace-pre-wrap text-purple-900">
                  {stepData[currentStep].explanation}
                </pre>
              </div>
            </div>

            {/* 进度条 */}
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${((currentStep + 1) / stepData.length) * 100}%` }}
              />
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default StepByStepPhysicalReasoning