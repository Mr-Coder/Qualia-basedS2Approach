import React, { useState, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { PhysicalReasoner, buildPhysicalGraphFromProblem, PhysicalGraph } from '@/services/physicalReasoningAPI'
import { ArrowRight, Zap, Network, Target, Lightbulb, AlertCircle } from 'lucide-react'

const PhysicalReasoningDemo: React.FC = () => {
  const [selectedExample, setSelectedExample] = useState<number>(0)
  const [physicalGraph, setPhysicalGraph] = useState<PhysicalGraph | null>(null)
  const [reasoning, setReasoning] = useState<string>('')
  const [showProcess, setShowProcess] = useState(false)

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

  const analyzeWithPhysicalReasoning = useCallback(() => {
    const problem = examples[selectedExample].problem
    const graph = buildPhysicalGraphFromProblem(problem)
    
    // 扩展图并寻找路径
    const expandedGraph = PhysicalReasoner.expandGraphWithPhysicalRelations(graph)
    const paths = PhysicalReasoner.findPathsFromKnownToUnknown(expandedGraph)
    
    // 更新图的路径
    expandedGraph.paths = paths
    
    setPhysicalGraph(expandedGraph)
    setShowProcess(true)
    
    // 生成推理说明
    if (paths.length > 0) {
      const bestPath = paths.reduce((best, current) => 
        current.confidence > best.confidence ? current : best
      )
      
      setReasoning(`
        推理过程：
        1. 识别实体：${expandedGraph.entities.map(e => e.name).join(', ')}
        2. 已知量：${expandedGraph.known.map(k => {
          const entity = expandedGraph.entities.find(e => e.id === k)
          return `${entity?.name}(${entity?.properties.value}${entity?.properties.unit || ''})`
        }).join(', ')}
        3. 未知量：${expandedGraph.unknown.map(u => {
          const entity = expandedGraph.entities.find(e => e.id === u)
          return entity?.name
        }).join(', ')}
        4. 推理路径：${bestPath.reasoning}
        5. 置信度：${(bestPath.confidence * 100).toFixed(0)}%
      `)
    } else {
      setReasoning('未找到有效的推理路径')
    }
  }, [selectedExample])

  const renderPhysicalGraph = () => {
    if (!physicalGraph) return null

    const width = 800
    const height = 400
    const centerX = width / 2
    const centerY = height / 2

    // 计算实体位置
    const entityPositions = physicalGraph.entities.map((entity, index) => {
      const angle = (index * 2 * Math.PI) / physicalGraph.entities.length
      const radius = 150
      return {
        ...entity,
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle)
      }
    })

    // 获取实体颜色
    const getEntityColor = (entity: any) => {
      if (physicalGraph.known.includes(entity.id)) return '#10b981' // 绿色 - 已知
      if (physicalGraph.unknown.includes(entity.id)) return '#ef4444' // 红色 - 未知
      return '#3b82f6' // 蓝色 - 中间概念
    }

    // 获取关系颜色
    const getRelationColor = (type: string) => {
      switch(type) {
        case 'conservation': return '#f59e0b' // 橙色 - 守恒
        case 'causal': return '#8b5cf6' // 紫色 - 因果
        case 'temporal': return '#06b6d4' // 青色 - 时间
        case 'transformation': return '#ec4899' // 粉色 - 转换
        default: return '#6b7280' // 灰色
      }
    }

    return (
      <svg width={width} height={height} className="border border-gray-200 rounded-lg bg-gradient-to-br from-blue-50 to-purple-50">
        {/* 绘制关系线 */}
        {physicalGraph.relations.map((rel, index) => {
          const source = entityPositions.find(e => e.id === rel.source)
          const target = entityPositions.find(e => e.id === rel.target)
          
          if (!source || !target) return null
          
          // 计算曲线控制点
          const midX = (source.x + target.x) / 2
          const midY = (source.y + target.y) / 2
          const dx = target.x - source.x
          const dy = target.y - source.y
          const distance = Math.sqrt(dx * dx + dy * dy)
          const offset = 30
          const controlX = midX - (dy / distance) * offset
          const controlY = midY + (dx / distance) * offset
          
          return (
            <g key={`rel-${index}`}>
              <path
                d={`M ${source.x} ${source.y} Q ${controlX} ${controlY} ${target.x} ${target.y}`}
                stroke={getRelationColor(rel.type)}
                strokeWidth={2 + rel.properties.strength * 2}
                fill="none"
                opacity={0.7}
                strokeDasharray={rel.type === 'conservation' ? '5,5' : 'none'}
              />
              <text
                x={controlX}
                y={controlY}
                textAnchor="middle"
                fontSize="11"
                fill={getRelationColor(rel.type)}
                fontWeight="500"
              >
                {rel.type}
              </text>
            </g>
          )
        })}
        
        {/* 绘制推理路径（如果存在） */}
        {physicalGraph.paths.length > 0 && (
          <g>
            {physicalGraph.paths[0].path.map((nodeId, index) => {
              if (index === 0) return null
              const prevNode = entityPositions.find(e => e.id === physicalGraph.paths[0].path[index - 1])
              const currentNode = entityPositions.find(e => e.id === nodeId)
              
              if (!prevNode || !currentNode) return null
              
              return (
                <line
                  key={`path-${index}`}
                  x1={prevNode.x}
                  y1={prevNode.y}
                  x2={currentNode.x}
                  y2={currentNode.y}
                  stroke="#f59e0b"
                  strokeWidth="4"
                  opacity="0.5"
                  strokeDasharray="10,5"
                />
              )
            })}
          </g>
        )}
        
        {/* 绘制实体节点 */}
        {entityPositions.map((entity) => (
          <g key={entity.id}>
            <circle
              cx={entity.x}
              cy={entity.y}
              r="45"
              fill={getEntityColor(entity)}
              stroke="white"
              strokeWidth="3"
              className="drop-shadow-md"
            />
            <text
              x={entity.x}
              y={entity.y - 5}
              textAnchor="middle"
              fontSize="13"
              fill="white"
              fontWeight="bold"
            >
              {entity.name}
            </text>
            {entity.properties.value !== null && entity.properties.value !== undefined && (
              <text
                x={entity.x}
                y={entity.y + 10}
                textAnchor="middle"
                fontSize="14"
                fill="white"
                fontWeight="bold"
              >
                {entity.properties.value}{entity.properties.unit || ''}
              </text>
            )}
          </g>
        ))}
        
        {/* 图例 */}
        <g transform="translate(20, 20)">
          <rect x="0" y="0" width="150" height="90" fill="white" opacity="0.9" rx="5" />
          <text x="10" y="20" fontSize="12" fontWeight="bold">图例：</text>
          <circle cx="20" cy="35" r="6" fill="#10b981" />
          <text x="35" y="40" fontSize="11">已知量</text>
          <circle cx="20" cy="50" r="6" fill="#ef4444" />
          <text x="35" y="55" fontSize="11">未知量</text>
          <circle cx="20" cy="65" r="6" fill="#3b82f6" />
          <text x="35" y="70" fontSize="11">中间概念</text>
          <line x1="10" y1="80" x2="30" y2="80" stroke="#f59e0b" strokeWidth="3" strokeDasharray="5,3" />
          <text x="35" y="85" fontSize="11">推理路径</text>
        </g>
      </svg>
    )
  }

  return (
    <Card className="max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Network className="h-6 w-6 text-purple-600" />
          基于物性关系的推理演示
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* 算法说明 */}
        <div className="bg-gradient-to-r from-purple-50 to-blue-50 p-6 rounded-lg">
          <h3 className="font-semibold mb-3 flex items-center gap-2">
            <Lightbulb className="h-5 w-5 text-yellow-500" />
            算法核心思想
          </h3>
          <ol className="space-y-2 text-sm">
            <li className="flex items-start gap-2">
              <span className="font-bold text-purple-600">1.</span>
              <span>分解题目 → 提取phrase和entity → 识别直陈关系</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold text-purple-600">2.</span>
              <span>加入物性关系 → 构建复杂的物性graph</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold text-purple-600">3.</span>
              <span>通过物性关系推理 → 扩展graph连接</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold text-purple-600">4.</span>
              <span>寻找已知到未知的通路 → 确定求解路径</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold text-purple-600">5.</span>
              <span>沿着路径推理 → 忘记无关推理 → 得出答案</span>
            </li>
          </ol>
        </div>

        {/* 例题选择 */}
        <div>
          <h3 className="font-semibold mb-3">选择示例问题：</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {examples.map((example, index) => (
              <button
                key={index}
                onClick={() => {
                  setSelectedExample(index)
                  setShowProcess(false)
                  setPhysicalGraph(null)
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
        <div className="flex justify-center">
          <Button
            onClick={analyzeWithPhysicalReasoning}
            className="flex items-center gap-2"
            size="lg"
          >
            <Zap className="h-5 w-5" />
            使用物性关系推理
          </Button>
        </div>

        {/* 推理过程展示 */}
        {showProcess && (
          <div className="space-y-4">
            {/* 物性图可视化 */}
            <div className="bg-white p-4 rounded-lg border">
              <h3 className="font-semibold mb-3 flex items-center gap-2">
                <Network className="h-5 w-5" />
                物性关系图
              </h3>
              {renderPhysicalGraph()}
            </div>

            {/* 推理说明 */}
            <div className="bg-blue-50 p-4 rounded-lg">
              <h3 className="font-semibold mb-3 flex items-center gap-2">
                <Target className="h-5 w-5" />
                推理过程
              </h3>
              <pre className="text-sm whitespace-pre-wrap">{reasoning}</pre>
            </div>

            {/* 关键洞察 */}
            <div className="bg-yellow-50 p-4 rounded-lg">
              <h3 className="font-semibold mb-2 flex items-center gap-2">
                <AlertCircle className="h-5 w-5 text-yellow-600" />
                关键洞察
              </h3>
              <p className="text-sm">
                通过物性关系（如守恒定律、因果关系、时间流逝等），我们能够在已知量和未知量之间建立连接通路，
                即使题目中没有直接说明这些关系。这种方法模拟了人类基于物理常识的推理过程。
              </p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default PhysicalReasoningDemo