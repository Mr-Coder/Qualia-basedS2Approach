import React, { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'

interface PropertyNode {
  id: string
  name: string
  description: string
  type: 'concept' | 'strategy' | 'domain' | 'skill'
  activation_level: number
  activation_state: 'inactive' | 'primed' | 'active' | 'decaying'
  x: number
  y: number
  details: string[]
}

interface PropertyConnection {
  from: string
  to: string
  type: string
  weight: number
  label: string
}

interface NetworkState {
  nodes: PropertyNode[]
  connections: PropertyConnection[]
  total_activation: number
  active_nodes_count: number
}

interface ActivationPropertyGraphProps {
  problemText?: string
  entities?: Array<{name: string, type: string}>
  onNodeActivation?: (nodeId: string, level: number) => void
}

const ActivationPropertyGraph: React.FC<ActivationPropertyGraphProps> = ({
  problemText = "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？",
  entities = [],
  onNodeActivation
}) => {
  const [networkState, setNetworkState] = useState<NetworkState | null>(null)
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [activationHistory, setActivationHistory] = useState<Array<{nodeId: string, level: number, timestamp: number}>>([])
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (problemText) {
      analyzeAndActivate()
    }
  }, [problemText, entities])

  const analyzeAndActivate = async () => {
    setIsLoading(true)
    try {
      // 模拟激活扩散分析
      const mockNetworkState: NetworkState = {
        nodes: [
          // 概念节点
          {
            id: "entity", name: "实体", description: "问题中的基本对象",
            type: "concept", activation_level: 0.8, activation_state: "active",
            x: 200, y: 150, details: ["人物", "物品", "数量", "单位"]
          },
          {
            id: "relation", name: "关系", description: "实体间的连接",
            type: "concept", activation_level: 0.6, activation_state: "primed",
            x: 400, y: 150, details: ["数量关系", "空间关系", "因果关系"]
          },
          {
            id: "property", name: "属性", description: "实体的特征",
            type: "concept", activation_level: 0.4, activation_state: "primed",
            x: 600, y: 150, details: ["数值属性", "类别属性", "约束条件"]
          },
          {
            id: "constraint", name: "约束", description: "问题的限制条件",
            type: "concept", activation_level: 0.3, activation_state: "primed",
            x: 800, y: 150, details: ["非负约束", "整数约束", "守恒约束"]
          },
          
          // 策略节点
          {
            id: "cot", name: "链式思维", description: "逐步推理",
            type: "strategy", activation_level: 0.9, activation_state: "active",
            x: 200, y: 350, details: ["步骤分解", "逻辑链条", "顺序执行"]
          },
          {
            id: "got", name: "图式思维", description: "关系网络推理",
            type: "strategy", activation_level: 0.5, activation_state: "primed",
            x: 400, y: 350, details: ["网络分析", "关系发现", "并行推理"]
          },
          {
            id: "tot", name: "树式思维", description: "多路径探索",
            type: "strategy", activation_level: 0.2, activation_state: "decaying",
            x: 600, y: 350, details: ["路径搜索", "方案评估", "最优选择"]
          },
          
          // 领域节点
          {
            id: "arithmetic", name: "算术", description: "基本数学运算",
            type: "domain", activation_level: 0.95, activation_state: "active",
            x: 200, y: 550, details: ["加减乘除", "数值计算", "运算规则"]
          },
          {
            id: "geometry", name: "几何", description: "空间形状关系",
            type: "domain", activation_level: 0.1, activation_state: "inactive",
            x: 400, y: 550, details: ["图形计算", "空间推理", "测量分析"]
          },
          
          // 技能节点
          {
            id: "modeling", name: "建模", description: "数学建模",
            type: "skill", activation_level: 0.7, activation_state: "active",
            x: 300, y: 750, details: ["抽象建模", "参数确定", "模型验证"]
          },
          {
            id: "verification", name: "验证", description: "结果验证",
            type: "skill", activation_level: 0.6, activation_state: "primed",
            x: 500, y: 750, details: ["结果检查", "约束验证", "合理性评估"]
          }
        ],
        connections: [
          { from: "entity", to: "relation", type: "dependency", weight: 0.8, label: "依赖" },
          { from: "relation", to: "property", type: "enhancement", weight: 0.7, label: "增强" },
          { from: "property", to: "constraint", type: "application", weight: 0.9, label: "应用" },
          { from: "entity", to: "cot", type: "application", weight: 0.7, label: "应用" },
          { from: "relation", to: "got", type: "application", weight: 0.9, label: "应用" },
          { from: "arithmetic", to: "entity", type: "dependency", weight: 0.8, label: "依赖" },
          { from: "cot", to: "modeling", type: "enhancement", weight: 0.9, label: "增强" },
          { from: "constraint", to: "verification", type: "application", weight: 0.9, label: "应用" }
        ],
        total_activation: 5.25,
        active_nodes_count: 4
      }
      
      setNetworkState(mockNetworkState)
      
      // 记录激活历史
      const timestamp = Date.now()
      const newHistory = mockNetworkState.nodes
        .filter(node => node.activation_level > 0.2)
        .map(node => ({
          nodeId: node.id,
          level: node.activation_level,
          timestamp
        }))
      
      setActivationHistory(prev => [...prev, ...newHistory].slice(-50))
      
    } finally {
      setIsLoading(false)
    }
  }

  const getNodeColor = (type: PropertyNode['type'], activation_level: number) => {
    const baseColors = {
      concept: '#3b82f6',    // 蓝色
      strategy: '#10b981',   // 绿色
      domain: '#f59e0b',     // 橙色
      skill: '#8b5cf6'       // 紫色
    }
    
    const baseColor = baseColors[type]
    const opacity = Math.max(0.3, activation_level)
    
    return `${baseColor}${Math.round(opacity * 255).toString(16).padStart(2, '0')}`
  }

  const getNodeSize = (activation_level: number) => {
    return 25 + (activation_level * 15) // 25-40px
  }

  const getActivationIntensity = (activation_level: number) => {
    if (activation_level > 0.8) return { intensity: '强', color: '#dc2626' }
    if (activation_level > 0.5) return { intensity: '中', color: '#f59e0b' }
    if (activation_level > 0.2) return { intensity: '弱', color: '#10b981' }
    return { intensity: '无', color: '#6b7280' }
  }

  const renderNodes = () => {
    if (!networkState) return null
    
    return networkState.nodes.map(node => {
      const isSelected = selectedNode === node.id
      const isHovered = hoveredNode === node.id
      const size = getNodeSize(node.activation_level)
      const color = getNodeColor(node.type, node.activation_level)
      
      // 获取直接相关的节点
      const relatedConnections = networkState.connections.filter(
        conn => conn.from === node.id || conn.to === node.id
      )
      const isRelated = selectedNode && relatedConnections.some(
        conn => conn.from === selectedNode || conn.to === selectedNode
      )

      return (
        <g key={node.id}>
          {/* 激活光晕效果 */}
          {node.activation_level > 0.5 && (
            <circle
              cx={node.x}
              cy={node.y}
              r={size + 8}
              fill={color}
              opacity={0.3}
              className="animate-pulse"
            />
          )}
          
          {/* 主节点 */}
          <motion.circle
            cx={node.x}
            cy={node.y}
            r={size}
            fill={color}
            stroke="#fff"
            strokeWidth="3"
            className="cursor-pointer drop-shadow-lg"
            opacity={!selectedNode || isSelected || isRelated ? 1 : 0.4}
            onClick={() => {
              setSelectedNode(isSelected ? null : node.id)
              onNodeActivation?.(node.id, node.activation_level)
            }}
            onMouseEnter={() => setHoveredNode(node.id)}
            onMouseLeave={() => setHoveredNode(null)}
            animate={{
              scale: isSelected ? 1.2 : (isHovered ? 1.1 : 1),
              r: size
            }}
            transition={{ duration: 0.2 }}
          />
          
          {/* 节点标签 */}
          <text
            x={node.x}
            y={node.y + size + 15}
            textAnchor="middle"
            className="text-xs font-medium fill-gray-700 pointer-events-none"
            opacity={!selectedNode || isSelected || isRelated ? 1 : 0.4}
          >
            {node.name}
          </text>
          
          {/* 激活强度指示器 */}
          {node.activation_level > 0.2 && (
            <text
              x={node.x}
              y={node.y - size - 5}
              textAnchor="middle"
              className="text-xs font-bold pointer-events-none"
              fill={getActivationIntensity(node.activation_level).color}
            >
              {(node.activation_level * 100).toFixed(0)}%
            </text>
          )}
        </g>
      )
    })
  }

  const renderConnections = () => {
    if (!networkState) return null
    
    return networkState.connections.map((conn, index) => {
      const fromNode = networkState.nodes.find(n => n.id === conn.from)
      const toNode = networkState.nodes.find(n => n.id === conn.to)
      
      if (!fromNode || !toNode) return null

      const isRelated = selectedNode && (conn.from === selectedNode || conn.to === selectedNode)
      const activationFlow = (fromNode.activation_level + toNode.activation_level) / 2
      
      const strokeColors = {
        dependency: '#94a3b8',
        application: '#10b981',
        enhancement: '#f59e0b'
      }
      
      return (
        <g key={index}>
          <motion.line
            x1={fromNode.x}
            y1={fromNode.y}
            x2={toNode.x}
            y2={toNode.y}
            stroke={strokeColors[conn.type as keyof typeof strokeColors] || '#94a3b8'}
            strokeWidth={2 + (activationFlow * 3)}
            strokeDasharray={conn.type === 'dependency' ? '0' : '5,5'}
            opacity={!selectedNode || isRelated ? 0.8 : 0.3}
            className="transition-all duration-300"
            animate={{
              strokeWidth: 2 + (activationFlow * 3)
            }}
          />
          
          {/* 连接标签 */}
          {isRelated && (
            <text
              x={(fromNode.x + toNode.x) / 2}
              y={(fromNode.y + toNode.y) / 2 - 5}
              textAnchor="middle"
              className="text-xs fill-gray-600 font-medium"
              fontSize="10"
            >
              {conn.label}
            </text>
          )}
        </g>
      )
    })
  }

  const renderActivationWaves = () => {
    if (!networkState) return null
    
    const activeNodes = networkState.nodes.filter(node => node.activation_level > 0.6)
    
    return activeNodes.map(node => (
      <motion.circle
        key={`wave-${node.id}`}
        cx={node.x}
        cy={node.y}
        r={30}
        fill="none"
        stroke={getNodeColor(node.type, 1.0)}
        strokeWidth="2"
        opacity={0.6}
        initial={{ r: 30, opacity: 0.6 }}
        animate={{ r: 60, opacity: 0 }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: "easeOut"
        }}
      />
    ))
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="flex items-center space-x-2">
          <motion.div
            className="w-4 h-4 bg-blue-500 rounded-full"
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 1, repeat: Infinity }}
          />
          <span>激活扩散分析中...</span>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* 网络状态总览 */}
      {networkState && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">激活扩散状态</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-4 gap-4 text-center">
              <div>
                <div className="text-2xl font-bold text-blue-600">
                  {networkState.active_nodes_count}
                </div>
                <div className="text-xs text-gray-600">激活节点</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-green-600">
                  {networkState.total_activation.toFixed(1)}
                </div>
                <div className="text-xs text-gray-600">总激活强度</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-purple-600">
                  {networkState.connections.length}
                </div>
                <div className="text-xs text-gray-600">连接数</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-orange-600">
                  {networkState.nodes.length}
                </div>
                <div className="text-xs text-gray-600">总节点数</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* 主要的激活扩散图谱 */}
      <Card>
        <CardHeader>
          <CardTitle>🧠 激活扩散物性图谱</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative">
            <svg
              ref={svgRef}
              width="100%"
              height="800"
              viewBox="0 0 1000 800"
              className="border rounded"
            >
              {/* 背景网格 */}
              <defs>
                <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
                  <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#f1f5f9" strokeWidth="1"/>
                </pattern>
              </defs>
              <rect width="100%" height="100%" fill="url(#grid)" />
              
              {/* 激活波纹效果 */}
              {renderActivationWaves()}
              
              {/* 连接线 */}
              {renderConnections()}
              
              {/* 节点 */}
              {renderNodes()}
            </svg>
          </div>
          
          {/* 节点详情面板 */}
          <AnimatePresence>
            {selectedNode && networkState && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="mt-4 p-4 bg-gray-50 rounded-lg"
              >
                {(() => {
                  const node = networkState.nodes.find(n => n.id === selectedNode)
                  if (!node) return null
                  
                  const intensity = getActivationIntensity(node.activation_level)
                  
                  return (
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-semibold text-lg">{node.name}</h4>
                        <span 
                          className="px-2 py-1 rounded text-xs font-medium text-white"
                          style={{ backgroundColor: intensity.color }}
                        >
                          激活强度: {intensity.intensity}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 mb-2">{node.description}</p>
                      <div className="space-y-1">
                        {node.details.map((detail, index) => (
                          <div key={index} className="text-xs text-gray-500">
                            • {detail}
                          </div>
                        ))}
                      </div>
                      <div className="mt-2 text-xs text-gray-500">
                        激活状态: {node.activation_state} | 激活水平: {(node.activation_level * 100).toFixed(1)}%
                      </div>
                    </div>
                  )
                })()}
              </motion.div>
            )}
          </AnimatePresence>
        </CardContent>
      </Card>

      {/* 重新分析按钮 */}
      <div className="text-center">
        <Button 
          onClick={analyzeAndActivate}
          disabled={isLoading}
          className="px-6"
        >
          重新激活分析
        </Button>
      </div>
    </div>
  )
}

export default ActivationPropertyGraph