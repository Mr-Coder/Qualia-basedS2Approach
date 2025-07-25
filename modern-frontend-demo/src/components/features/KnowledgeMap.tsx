import React, { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'

interface KnowledgeNode {
  id: string
  name: string
  description: string
  category: 'concept' | 'strategy' | 'domain' | 'skill'
  details: string[]
  x: number
  y: number
  connections: string[]
}

interface KnowledgeConnection {
  from: string
  to: string
  label: string
  type: 'dependency' | 'application' | 'enhancement' | 'example'
}

const knowledgeNodes: KnowledgeNode[] = [
  // 核心概念
  {
    id: 'entity',
    name: '实体',
    description: '问题中的基本对象和要素',
    category: 'concept',
    details: [
      '人物、物品、数量等具体对象',
      '问题中的关键信息载体',
      '解题过程中的基本单位',
      '具有特定属性和状态'
    ],
    x: 150,
    y: 100,
    connections: ['relation', 'modeling', 'arithmetic']
  },
  {
    id: 'relation',
    name: '关系',
    description: '实体之间的连接和相互作用',
    category: 'concept',
    details: [
      '数量关系、空间关系、时间关系',
      '因果关系、逻辑关系',
      '网络化的关系结构',
      '动态变化的关系状态'
    ],
    x: 350,
    y: 100,
    connections: ['property', 'reasoning', 'got']
  },
  {
    id: 'property',
    name: '属性',
    description: '实体的特征和性质',
    category: 'concept',
    details: [
      '数值属性、类别属性',
      '静态属性、动态属性',
      '可观察属性、推导属性',
      '约束条件和限制'
    ],
    x: 550,
    y: 100,
    connections: ['constraint', 'verification', 'geometry']
  },
  {
    id: 'constraint',
    name: '约束',
    description: '问题的限制条件和边界',
    category: 'concept',
    details: [
      '显性约束、隐性约束',
      '物理约束、逻辑约束',
      '数值范围约束',
      '一致性约束'
    ],
    x: 750,
    y: 100,
    connections: ['reasoning', 'tot', 'application']
  },
  {
    id: 'reasoning',
    name: '推理',
    description: '从已知推导未知的思维过程',
    category: 'concept',
    details: [
      '逻辑推理、数学推理',
      '演绎推理、归纳推理',
      '类比推理、因果推理',
      '多步骤推理链'
    ],
    x: 950,
    y: 100,
    connections: ['cot', 'got', 'tot']
  },

  // 推理策略
  {
    id: 'cot',
    name: 'COT推理',
    description: '思维链推理策略',
    category: 'strategy',
    details: [
      '逐步分解问题',
      '建立清晰的推理链',
      '状态跟踪和转移',
      '适合顺序推理问题'
    ],
    x: 250,
    y: 300,
    connections: ['decomposition', 'verification', 'arithmetic']
  },
  {
    id: 'got',
    name: 'GOT推理',
    description: '思维图推理策略',
    category: 'strategy',
    details: [
      '构建关系网络',
      '发现隐含连接',
      '网络拓扑分析',
      '适合复杂关系问题'
    ],
    x: 450,
    y: 300,
    connections: ['modeling', 'analysis', 'application']
  },
  {
    id: 'tot',
    name: 'TOT推理',
    description: '思维树推理策略',
    category: 'strategy',
    details: [
      '多路径探索',
      '层次化分析',
      '方案比较选择',
      '适合开放性问题'
    ],
    x: 650,
    y: 300,
    connections: ['exploration', 'evaluation', 'percentage']
  },

  // 问题领域
  {
    id: 'arithmetic',
    name: '算术问题',
    description: '基础数学运算问题',
    category: 'domain',
    details: [
      '加减乘除运算',
      '数量关系分析',
      '基本应用题',
      '运算规律探索'
    ],
    x: 150,
    y: 500,
    connections: ['decomposition', 'verification']
  },
  {
    id: 'geometry',
    name: '几何问题',
    description: '图形和空间问题',
    category: 'domain',
    details: [
      '平面几何计算',
      '立体几何分析',
      '图形变换',
      '空间关系推理'
    ],
    x: 350,
    y: 500,
    connections: ['modeling', 'analysis']
  },
  {
    id: 'application',
    name: '应用题',
    description: '实际情境问题',
    category: 'domain',
    details: [
      '现实场景建模',
      '多约束条件处理',
      '复杂关系分析',
      '实际意义验证'
    ],
    x: 550,
    y: 500,
    connections: ['modeling', 'analysis', 'evaluation']
  },
  {
    id: 'percentage',
    name: '百分比问题',
    description: '比例和百分比计算',
    category: 'domain',
    details: [
      '比例关系计算',
      '百分比转换',
      '增长率分析',
      '比例应用题'
    ],
    x: 750,
    y: 500,
    connections: ['exploration', 'evaluation']
  },

  // 思维技能
  {
    id: 'decomposition',
    name: '分解',
    description: '将复杂问题分解为简单部分',
    category: 'skill',
    details: [
      '问题结构分析',
      '子问题识别',
      '分解策略选择',
      '分解结果整合'
    ],
    x: 250,
    y: 700,
    connections: []
  },
  {
    id: 'modeling',
    name: '建模',
    description: '构建问题的数学模型',
    category: 'skill',
    details: [
      '抽象化处理',
      '模型构建',
      '参数确定',
      '模型验证'
    ],
    x: 450,
    y: 700,
    connections: []
  },
  {
    id: 'analysis',
    name: '分析',
    description: '深入理解问题本质',
    category: 'skill',
    details: [
      '关系分析',
      '模式识别',
      '逻辑推理',
      '结果解释'
    ],
    x: 650,
    y: 700,
    connections: []
  },
  {
    id: 'verification',
    name: '验证',
    description: '检验解答的正确性',
    category: 'skill',
    details: [
      '结果检查',
      '逻辑验证',
      '合理性评估',
      '方法确认'
    ],
    x: 850,
    y: 700,
    connections: []
  },
  {
    id: 'exploration',
    name: '探索',
    description: '发现新的解题路径',
    category: 'skill',
    details: [
      '多方案尝试',
      '创新思维',
      '假设验证',
      '路径优化'
    ],
    x: 150,
    y: 900,
    connections: []
  },
  {
    id: 'evaluation',
    name: '评估',
    description: '比较和选择最优方案',
    category: 'skill',
    details: [
      '方案比较',
      '优劣分析',
      '选择标准',
      '决策制定'
    ],
    x: 350,
    y: 900,
    connections: []
  }
]

const connections: KnowledgeConnection[] = [
  // 核心概念链：认知的基础流程
  { from: 'entity', to: 'relation', label: '建立', type: 'dependency' },
  { from: 'relation', to: 'property', label: '体现', type: 'dependency' },
  { from: 'property', to: 'constraint', label: '约束', type: 'dependency' },
  { from: 'constraint', to: 'reasoning', label: '指导', type: 'dependency' },
  
  // 推理策略实现：从理论到实践
  { from: 'reasoning', to: 'cot', label: '线性推理', type: 'application' },
  { from: 'reasoning', to: 'got', label: '网络推理', type: 'application' },
  { from: 'reasoning', to: 'tot', label: '树形推理', type: 'application' },
  
  // 策略与领域应用：不同策略适用不同问题
  { from: 'cot', to: 'arithmetic', label: '最佳适用', type: 'application' },
  { from: 'cot', to: 'geometry', label: '步骤计算', type: 'application' },
  { from: 'got', to: 'application', label: '复杂关系', type: 'application' },
  { from: 'got', to: 'geometry', label: '空间关系', type: 'application' },
  { from: 'tot', to: 'percentage', label: '多路径', type: 'application' },
  { from: 'tot', to: 'application', label: '方案比较', type: 'application' },
  
  // 策略技能依赖：每种策略需要的核心技能
  { from: 'cot', to: 'decomposition', label: '核心技能', type: 'dependency' },
  { from: 'cot', to: 'verification', label: '验证步骤', type: 'dependency' },
  { from: 'got', to: 'modeling', label: '关系建模', type: 'dependency' },
  { from: 'got', to: 'analysis', label: '关系分析', type: 'dependency' },
  { from: 'tot', to: 'exploration', label: '路径探索', type: 'dependency' },
  { from: 'tot', to: 'evaluation', label: '方案评估', type: 'dependency' },
  
  // 技能间协作关系：技能之间的互补增强
  { from: 'decomposition', to: 'verification', label: '相互验证', type: 'enhancement' },
  { from: 'modeling', to: 'analysis', label: '模型分析', type: 'enhancement' },
  { from: 'exploration', to: 'evaluation', label: '探索评估', type: 'enhancement' },
  { from: 'analysis', to: 'verification', label: '分析验证', type: 'enhancement' },
  { from: 'decomposition', to: 'modeling', label: '分解建模', type: 'enhancement' },
  
  // 概念与技能的关系：概念如何指导技能应用
  { from: 'entity', to: 'decomposition', label: '识别分解', type: 'example' },
  { from: 'relation', to: 'modeling', label: '关系建模', type: 'example' },
  { from: 'property', to: 'analysis', label: '属性分析', type: 'example' },
  { from: 'constraint', to: 'verification', label: '约束验证', type: 'example' },
  
  // 跨领域应用：不同问题领域的相互关系
  { from: 'arithmetic', to: 'geometry', label: '计算基础', type: 'dependency' },
  { from: 'arithmetic', to: 'percentage', label: '数值基础', type: 'dependency' },
  { from: 'geometry', to: 'application', label: '实际应用', type: 'application' },
  { from: 'percentage', to: 'application', label: '比例应用', type: 'application' }
]

export const KnowledgeMap: React.FC = () => {
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)
  const [showRealData, setShowRealData] = useState(false)
  const [realTimeData, setRealTimeData] = useState<any>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  // 获取实时算法执行数据
  useEffect(() => {
    const fetchRealTimeData = async () => {
      try {
        const response = await fetch('/api/algorithm/execution')
        const data = await response.json()
        if (data.success && data.data) {
          setRealTimeData(data.data)
        }
      } catch (error) {
        console.error('获取实时数据失败:', error)
      }
    }

    if (showRealData) {
      fetchRealTimeData()
      const interval = setInterval(fetchRealTimeData, 5000) // 每5秒更新一次
      return () => clearInterval(interval)
    }
  }, [showRealData])

  const getNodeColor = (category: string) => {
    const colors = {
      concept: '#3b82f6',    // 蓝色
      strategy: '#10b981',   // 绿色
      domain: '#f59e0b',     // 橙色
      skill: '#ef4444'       // 红色
    }
    return colors[category] || '#6b7280'
  }

  const getNodeIcon = (category: string) => {
    const icons = {
      concept: '💡',
      strategy: '🎯',
      domain: '📚',
      skill: '🛠️'
    }
    return icons[category] || '📦'
  }

  // 🔗 获取与节点直接相关的节点（基于明确的连接关系）
  const getDirectlyRelatedNodes = (nodeId: string): string[] => {
    const relatedNodeIds = new Set<string>()
    
    // 查找所有直接连接的节点
    connections.forEach(conn => {
      if (conn.from === nodeId) {
        relatedNodeIds.add(conn.to)
      } else if (conn.to === nodeId) {
        relatedNodeIds.add(conn.from)
      }
    })
    
    return Array.from(relatedNodeIds)
  }

  const renderConnections = () => {
    return connections.map((conn, index) => {
      const fromNode = knowledgeNodes.find(n => n.id === conn.from)
      const toNode = knowledgeNodes.find(n => n.id === conn.to)
      
      if (!fromNode || !toNode) return null

      // 判断连接线是否应该高亮（当其中一个节点被选中时）
      const isRelated = selectedNode && (conn.from === selectedNode || conn.to === selectedNode)
      
      const strokeColor = {
        dependency: '#94a3b8',
        application: '#10b981',
        enhancement: '#f59e0b',
        example: '#8b5cf6'
      }[conn.type]

      return (
        <g key={index}>
          <line
            x1={fromNode.x}
            y1={fromNode.y}
            x2={toNode.x}
            y2={toNode.y}
            stroke={strokeColor}
            strokeWidth={isRelated ? 4 : 2}
            strokeDasharray={conn.type === 'dependency' ? '0' : '5,5'}
            opacity={!selectedNode || isRelated ? 0.8 : 0.3}
            className="transition-all duration-300"
          />
          
          <text
            x={(fromNode.x + toNode.x) / 2}
            y={(fromNode.y + toNode.y) / 2 - 5}
            textAnchor="middle"
            className="text-xs fill-gray-600"
            fontSize="10"
            opacity={!selectedNode || isRelated ? 0.8 : 0.4}
          >
            {conn.label}
          </text>
        </g>
      )
    })
  }

  const renderNodes = () => {
    return knowledgeNodes.map((node) => {
      const isSelected = selectedNode === node.id
      const isHovered = hoveredNode === node.id
      
      // 判断当前节点是否与选中节点直接相关
      const directlyRelatedNodes = selectedNode ? getDirectlyRelatedNodes(selectedNode) : []
      const isDirectlyRelated = directlyRelatedNodes.includes(node.id)
      
      const radius = isSelected ? 35 : (isHovered ? 33 : 30)
      const baseColor = getNodeColor(node.category)
      
      return (
        <g key={node.id}>
          <circle
            cx={node.x}
            cy={node.y}
            r={radius}
            fill={baseColor}
            stroke="#fff"
            strokeWidth="3"
            className="cursor-pointer drop-shadow-lg transition-all duration-300"
            opacity={!selectedNode || isSelected || isDirectlyRelated ? 1 : 0.3}
            onClick={() => setSelectedNode(isSelected ? null : node.id)}
            onMouseEnter={() => setHoveredNode(node.id)}
            onMouseLeave={() => setHoveredNode(null)}
          />
          
          {/* 节点图标 */}
          <text
            x={node.x}
            y={node.y - 5}
            textAnchor="middle"
            className="text-xs fill-white font-medium pointer-events-none"
            fontSize="12"
          >
            {getNodeIcon(node.category)}
          </text>
          
          {/* 节点名称 */}
          <text
            x={node.x}
            y={node.y + 8}
            textAnchor="middle"
            className="text-xs fill-white font-medium pointer-events-none"
            fontSize="9"
          >
            {node.name}
          </text>
        </g>
      )
    })
  }

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* 页面标题 */}
      <Card>
        <CardHeader>
          <CardTitle>🗺️ 物性图谱</CardTitle>
          <p className="text-gray-600">
            可视化展示COT-DIR系统的知识结构，了解各概念间的关系和应用
          </p>
        </CardHeader>
      </Card>

      {/* 知识域概览 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { category: 'concept', name: '核心概念', icon: '💡', color: 'bg-blue-500', count: 5 },
          { category: 'strategy', name: '推理策略', icon: '🎯', color: 'bg-green-500', count: 3 },
          { category: 'domain', name: '问题领域', icon: '📚', color: 'bg-orange-500', count: 4 },
          { category: 'skill', name: '思维技能', icon: '🛠️', color: 'bg-red-500', count: 6 }
        ].map((domain) => (
          <motion.div
            key={domain.category}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-lg p-4 border border-gray-200 hover:shadow-lg transition-shadow"
          >
            <div className="flex items-center gap-3 mb-3">
              <div className={`w-10 h-10 ${domain.color} rounded-lg flex items-center justify-center text-white text-xl`}>
                {domain.icon}
              </div>
              <div>
                <h3 className="font-semibold text-gray-800">{domain.name}</h3>
                <p className="text-sm text-gray-600">{domain.count} 个节点</p>
              </div>
            </div>
            <div className="text-xs text-gray-500">
              {knowledgeNodes
                .filter(n => n.category === domain.category)
                .map(n => n.name)
                .join(', ')}
            </div>
          </motion.div>
        ))}
      </div>

      {/* 实时算法数据面板 */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>📊 实时算法关系数据</CardTitle>
            <Button
              onClick={() => setShowRealData(!showRealData)}
              variant={showRealData ? 'default' : 'outline'}
              size="sm"
            >
              {showRealData ? '🔄 实时同步中' : '🔗 连接实时数据'}
            </Button>
          </div>
          <p className="text-gray-600">
            展示来自真实IRD算法执行的实体关系发现数据
          </p>
        </CardHeader>
        {showRealData && (
          <CardContent>
            {realTimeData ? (
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-blue-50 rounded-lg p-3">
                    <div className="text-sm text-blue-600 font-medium">执行ID</div>
                    <div className="text-xs text-blue-700 font-mono">
                      {realTimeData.execution_id.split('-')[0]}...
                    </div>
                  </div>
                  <div className="bg-green-50 rounded-lg p-3">
                    <div className="text-sm text-green-600 font-medium">发现关系</div>
                    <div className="text-lg font-bold text-green-700">
                      {realTimeData.execution_metrics?.total_relations_discovered || 0}
                    </div>
                  </div>
                  <div className="bg-purple-50 rounded-lg p-3">
                    <div className="text-sm text-purple-600 font-medium">平均置信度</div>
                    <div className="text-lg font-bold text-purple-700">
                      {((realTimeData.execution_metrics?.average_confidence || 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                <div className="border rounded-lg p-4">
                  <h4 className="font-medium text-gray-800 mb-3">🔍 最新执行阶段</h4>
                  <div className="space-y-2">
                    {realTimeData.stages?.map((stage: any, index: number) => (
                      <div key={index} className="flex items-center justify-between bg-gray-50 rounded p-2">
                        <div className="flex items-center gap-2">
                          <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs">
                            {index + 1}
                          </div>
                          <span className="text-sm font-medium">{stage.stage_name}</span>
                        </div>
                        <div className="flex items-center gap-3 text-xs text-gray-600">
                          <span>耗时: {stage.duration_ms.toFixed(2)}ms</span>
                          <span>置信度: {(stage.confidence * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="text-xs text-gray-500 text-center">
                  问题: {realTimeData.problem_text?.substring(0, 50)}...
                  <br />
                  执行时间: {new Date(realTimeData.start_time * 1000).toLocaleString()}
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-2"></div>
                正在获取实时算法执行数据...
              </div>
            )}
          </CardContent>
        )}
      </Card>

      {/* 关联关系说明面板 */}
      {selectedNode && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-4 border"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-gray-800">🔗 直接关联关系</h3>
            <div className="text-sm text-gray-600">
              选中节点: {knowledgeNodes.find(n => n.id === selectedNode)?.name}
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {connections
              .filter(conn => conn.from === selectedNode || conn.to === selectedNode)
              .map((conn, index) => {
                const relatedNodeId = conn.from === selectedNode ? conn.to : conn.from
                const relatedNode = knowledgeNodes.find(n => n.id === relatedNodeId)
                if (!relatedNode) return null
                
                const relationshipColor = {
                  'dependency': 'text-slate-600 bg-slate-100',
                  'application': 'text-green-600 bg-green-100', 
                  'enhancement': 'text-orange-600 bg-orange-100',
                  'example': 'text-purple-600 bg-purple-100'
                }[conn.type]
                
                const relationshipDesc = {
                  'dependency': '依赖关系 - 基础概念支撑',
                  'application': '应用关系 - 实际运用场景',
                  'enhancement': '增强关系 - 相互促进提升',
                  'example': '示例关系 - 具体案例展示'
                }[conn.type]
                
                return (
                  <div 
                    key={index}
                    className="bg-white rounded-lg p-3 border hover:shadow-md transition-shadow cursor-pointer"
                    onClick={() => setSelectedNode(relatedNodeId)}
                  >
                    <div className="flex items-center gap-3 mb-2">
                      <div className="text-xl">{getNodeIcon(relatedNode.category)}</div>
                      <div>
                        <div className="font-medium text-gray-800">{relatedNode.name}</div>
                        <div className="text-xs text-gray-500">{relatedNode.category === 'concept' ? '核心概念' : relatedNode.category === 'strategy' ? '推理策略' : relatedNode.category === 'domain' ? '问题领域' : '思维技能'}</div>
                      </div>
                    </div>
                    
                    <div className="space-y-1">
                      <div className={`inline-block px-2 py-1 rounded text-xs font-medium ${relationshipColor}`}>
                        {conn.label}
                      </div>
                      <div className="text-xs text-gray-600">
                        {relationshipDesc}
                      </div>
                    </div>
                  </div>
                )
              })}
          </div>
          
          <div className="mt-3 text-xs text-gray-500">
            💡 点击相关节点可以切换查看，所有关联基于明确的知识结构设计
          </div>
        </motion.div>
      )}

      {/* 交互式知识图谱 */}
      <Card>
        <CardHeader>
          <CardTitle>🕸️ 交互式物性图谱</CardTitle>
          <p className="text-sm text-gray-600">
            点击节点查看详细信息，体验基于激活扩散理论的智能关联激活
          </p>
        </CardHeader>
        <CardContent>
          <div className="relative">
            <svg
              ref={svgRef}
              width="100%"
              height="600"
              viewBox="0 0 1000 1000"
              className="border border-gray-200 rounded-lg bg-gray-50"
            >
              {/* 定义箭头标记 */}
              <defs>
                <marker
                  id="arrowhead"
                  markerWidth="10"
                  markerHeight="7"
                  refX="9"
                  refY="3.5"
                  orient="auto"
                >
                  <polygon
                    points="0 0, 10 3.5, 0 7"
                    fill="#94a3b8"
                  />
                </marker>
              </defs>

              {/* 渲染连接线 */}
              {renderConnections()}

              {/* 渲染节点 */}
              {renderNodes()}
            </svg>

            {/* 图例 */}
            <div className="mt-4 flex flex-wrap gap-4 text-sm">
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 rounded-full bg-blue-500"></div>
                <span>💡 核心概念</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 rounded-full bg-green-500"></div>
                <span>🎯 推理策略</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 rounded-full bg-orange-500"></div>
                <span>📚 问题领域</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 rounded-full bg-red-500"></div>
                <span>🛠️ 思维技能</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 节点详细信息 */}
      {selectedNode && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <Card>
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className={`w-10 h-10 rounded-lg flex items-center justify-center text-white text-xl`}
                     style={{ backgroundColor: getNodeColor(knowledgeNodes.find(n => n.id === selectedNode)!.category) }}>
                  {getNodeIcon(knowledgeNodes.find(n => n.id === selectedNode)!.category)}
                </div>
                <div>
                  <CardTitle>{knowledgeNodes.find(n => n.id === selectedNode)!.name}</CardTitle>
                  <p className="text-sm text-gray-600">
                    {knowledgeNodes.find(n => n.id === selectedNode)!.description}
                  </p>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold text-gray-800 mb-2">📝 详细说明</h4>
                  <ul className="space-y-1">
                    {knowledgeNodes.find(n => n.id === selectedNode)!.details.map((detail, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                        <span className="text-purple-500 mt-1">•</span>
                        <span>{detail}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                
                {knowledgeNodes.find(n => n.id === selectedNode)!.connections.length > 0 && (
                  <div>
                    <h4 className="font-semibold text-gray-800 mb-2">🔗 相关知识点</h4>
                    <div className="flex flex-wrap gap-2">
                      {knowledgeNodes.find(n => n.id === selectedNode)!.connections.map((connId) => {
                        const connNode = knowledgeNodes.find(n => n.id === connId)
                        return connNode ? (
                          <button
                            key={connId}
                            onClick={() => setSelectedNode(connId)}
                            className="px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-full text-sm transition-colors"
                          >
                            {getNodeIcon(connNode.category)} {connNode.name}
                          </button>
                        ) : null
                      })}
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* 智能推理过程流程 */}
      <Card>
        <CardHeader>
          <CardTitle>🔄 智能推理过程流程</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-6">
            <div className="flex items-center justify-between text-sm">
              <div className="flex flex-col items-center">
                <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold">
                  1
                </div>
                <span className="mt-2 text-center">实体识别</span>
              </div>
              <div className="flex-1 h-px bg-gray-300 mx-2"></div>
              <div className="flex flex-col items-center">
                <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center text-white font-bold">
                  2
                </div>
                <span className="mt-2 text-center">关系分析</span>
              </div>
              <div className="flex-1 h-px bg-gray-300 mx-2"></div>
              <div className="flex flex-col items-center">
                <div className="w-12 h-12 bg-orange-500 rounded-full flex items-center justify-center text-white font-bold">
                  3
                </div>
                <span className="mt-2 text-center">策略选择</span>
              </div>
              <div className="flex-1 h-px bg-gray-300 mx-2"></div>
              <div className="flex flex-col items-center">
                <div className="w-12 h-12 bg-purple-500 rounded-full flex items-center justify-center text-white font-bold">
                  4
                </div>
                <span className="mt-2 text-center">推理执行</span>
              </div>
              <div className="flex-1 h-px bg-gray-300 mx-2"></div>
              <div className="flex flex-col items-center">
                <div className="w-12 h-12 bg-red-500 rounded-full flex items-center justify-center text-white font-bold">
                  5
                </div>
                <span className="mt-2 text-center">结果验证</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default KnowledgeMap