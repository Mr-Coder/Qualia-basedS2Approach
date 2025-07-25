import React, { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'

// Icons
import { 
  BookOpen, 
  Target, 
  TrendingUp, 
  CheckCircle,
  ArrowRight,
  Lightbulb,
  MapPin,
  Zap,
  Play,
  BookMarked,
  Award
} from 'lucide-react'

// 🧠 基于交互式物性图谱的学习节点结构
interface LearningPropertyNode {
  id: string
  name: string
  description: string
  category: 'concept' | 'strategy' | 'domain' | 'skill'
  difficulty: 'beginner' | 'intermediate' | 'advanced'
  activation_level: number
  mastery_level: number  // 掌握程度 [0-1]
  learning_state: 'not_started' | 'learning' | 'practicing' | 'mastered'
  prerequisites: string[]
  learning_objectives: string[]
  practice_examples: string[]
  mastery_indicators: string[]
  x: number
  y: number
  connections: string[]
}

interface LearningConnection {
  from: string
  to: string
  label: string
  type: 'prerequisite' | 'builds_on' | 'reinforces' | 'applies'
  weight: number
  unlocked: boolean
}

const InteractivePropertyLearningGuide: React.FC = () => {
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)
  const [learningNodes, setLearningNodes] = useState<LearningPropertyNode[]>([])
  const [connections, setConnections] = useState<LearningConnection[]>([])
  const [currentLearningPath, setCurrentLearningPath] = useState<string[]>([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const svgRef = useRef<SVGSVGElement>(null)

  // 初始化学习图谱
  useEffect(() => {
    initializeLearningGraph()
  }, [])

  const initializeLearningGraph = () => {
    const nodes: LearningPropertyNode[] = [
      {
        id: 'entity_recognition',
        name: '实体识别',
        description: '识别数学问题中的基本对象',
        category: 'concept',
        difficulty: 'beginner',
        activation_level: 0.9,
        mastery_level: 0.8,
        learning_state: 'mastered',
        prerequisites: [],
        learning_objectives: [
          '识别问题中的人物、物品、数量',
          '理解实体在数学问题中的作用',
          '建立实体与数学运算的联系'
        ],
        practice_examples: [
          '小明有5个苹果 → 识别：小明(人物)、苹果(物品)、5(数量)',
          '书店有30本书 → 识别：书店(地点)、书(物品)、30(数量)'
        ],
        mastery_indicators: [
          '能快速识别所有实体',
          '准确分类实体类型',
          '理解实体间的基本关系'
        ],
        x: 100, y: 100,
        connections: ['relation_understanding', 'arithmetic_basics']
      },
      {
        id: 'relation_understanding',
        name: '关系理解',
        description: '理解实体间的数学关系',
        category: 'concept',
        difficulty: 'beginner',
        activation_level: 0.7,
        mastery_level: 0.6,
        learning_state: 'practicing',
        prerequisites: ['entity_recognition'],
        learning_objectives: [
          '识别数量关系、拥有关系',
          '理解关系的方向性',
          '建立关系网络思维'
        ],
        practice_examples: [
          '小明有5个，小红有3个 → 拥有关系',
          '一共有多少 → 总和关系'
        ],
        mastery_indicators: [
          '能识别各种数学关系',
          '理解关系的逻辑含义',
          '能构建关系图'
        ],
        x: 300, y: 100,
        connections: ['problem_modeling', 'pattern_recognition']
      },
      {
        id: 'arithmetic_basics',
        name: '算术基础',
        description: '掌握基本的数学运算',
        category: 'domain',
        difficulty: 'beginner',
        activation_level: 0.95,
        mastery_level: 0.9,
        learning_state: 'mastered',
        prerequisites: ['entity_recognition'],
        learning_objectives: [
          '熟练掌握四则运算',
          '理解运算符号的含义',
          '能够进行心算和笔算'
        ],
        practice_examples: [
          '5 + 3 = 8（加法：合并数量）',
          '10 - 4 = 6（减法：去除部分）',
          '3 × 4 = 12（乘法：重复相加）'
        ],
        mastery_indicators: [
          '运算速度和准确度达标',
          '理解运算的实际意义',
          '能选择合适的运算方法'
        ],
        x: 100, y: 300,
        connections: ['cot_reasoning', 'problem_solving']
      },
      {
        id: 'cot_reasoning',
        name: '链式推理',
        description: '学会逐步推理的思维方法',
        category: 'strategy',
        difficulty: 'intermediate',
        activation_level: 0.8,
        mastery_level: 0.4,
        learning_state: 'learning',
        prerequisites: ['arithmetic_basics', 'relation_understanding'],
        learning_objectives: [
          '掌握分步推理方法',
          '建立逻辑推理链条',
          '提高推理的系统性'
        ],
        practice_examples: [
          '步骤1：识别已知条件',
          '步骤2：确定运算关系',
          '步骤3：执行计算过程',
          '步骤4：验证结果合理性'
        ],
        mastery_indicators: [
          '能够分解复杂问题',
          '推理步骤清晰有逻辑',
          '善于检验推理过程'
        ],
        x: 300, y: 300,
        connections: ['problem_modeling', 'advanced_reasoning']
      },
      {
        id: 'problem_modeling',
        name: '问题建模',
        description: '将实际问题转化为数学模型',
        category: 'skill',
        difficulty: 'intermediate',
        activation_level: 0.6,
        mastery_level: 0.3,
        learning_state: 'learning',
        prerequisites: ['relation_understanding', 'cot_reasoning'],
        learning_objectives: [
          '学会抽象思维',
          '建立数学模型',
          '连接现实与数学'
        ],
        practice_examples: [
          '小明有5个苹果... → 设x为总数',
          '速度问题 → 距离 = 速度 × 时间',
          '比例问题 → 建立比例式'
        ],
        mastery_indicators: [
          '能快速建立数学模型',
          '模型准确反映问题本质',
          '善于选择合适的表示方法'
        ],
        x: 500, y: 200,
        connections: ['advanced_reasoning', 'problem_solving']
      },
      {
        id: 'pattern_recognition',
        name: '模式识别',
        description: '识别问题中的规律和模式',
        category: 'skill',
        difficulty: 'intermediate',
        activation_level: 0.5,
        mastery_level: 0.2,
        learning_state: 'not_started',
        prerequisites: ['relation_understanding'],
        learning_objectives: [
          '发现数学问题的规律',
          '识别常见问题类型',
          '提高解题效率'
        ],
        practice_examples: [
          '加法问题的共同特征',
          '比例问题的关键词',
          '几何问题的图形规律'
        ],
        mastery_indicators: [
          '能快速识别问题类型',
          '掌握各类问题的解法模式',
          '举一反三能力强'
        ],
        x: 500, y: 100,
        connections: ['advanced_reasoning']
      },
      {
        id: 'problem_solving',
        name: '综合解题',
        description: '综合运用各种方法解决数学问题',
        category: 'skill',
        difficulty: 'advanced',
        activation_level: 0.4,
        mastery_level: 0.1,
        learning_state: 'not_started',
        prerequisites: ['cot_reasoning', 'problem_modeling'],
        learning_objectives: [
          '综合运用多种策略',
          '解决复杂数学问题',
          '提升数学思维能力'
        ],
        practice_examples: [
          '多步骤复合问题',
          '需要多种方法的问题',
          '开放性数学问题'
        ],
        mastery_indicators: [
          '能解决各类数学问题',
          '方法选择恰当高效',
          '具备数学思维素养'
        ],
        x: 300, y: 500,
        connections: ['advanced_reasoning']
      },
      {
        id: 'advanced_reasoning',
        name: '高级推理',
        description: '掌握复杂的数学推理方法',
        category: 'strategy',
        difficulty: 'advanced',
        activation_level: 0.3,
        mastery_level: 0.0,
        learning_state: 'not_started',
        prerequisites: ['cot_reasoning', 'problem_modeling', 'pattern_recognition'],
        learning_objectives: [
          '掌握多种推理策略',
          '处理复杂逻辑关系',
          '发展创新思维'
        ],
        practice_examples: [
          '图式推理(GOT)',
          '树式推理(TOT)',
          '类比推理方法'
        ],
        mastery_indicators: [
          '能选择最优推理策略',
          '处理复杂推理问题',
          '具备创新解题能力'
        ],
        x: 500, y: 400,
        connections: []
      }
    ]

    const learningConnections: LearningConnection[] = [
      { from: 'entity_recognition', to: 'relation_understanding', label: '基础', type: 'prerequisite', weight: 0.9, unlocked: true },
      { from: 'entity_recognition', to: 'arithmetic_basics', label: '应用', type: 'prerequisite', weight: 0.8, unlocked: true },
      { from: 'relation_understanding', to: 'problem_modeling', label: '深化', type: 'builds_on', weight: 0.7, unlocked: true },
      { from: 'relation_understanding', to: 'pattern_recognition', label: '拓展', type: 'builds_on', weight: 0.6, unlocked: false },
      { from: 'arithmetic_basics', to: 'cot_reasoning', label: '方法', type: 'builds_on', weight: 0.8, unlocked: true },
      { from: 'cot_reasoning', to: 'problem_modeling', label: '结合', type: 'reinforces', weight: 0.9, unlocked: true },
      { from: 'cot_reasoning', to: 'problem_solving', label: '应用', type: 'applies', weight: 0.7, unlocked: false },
      { from: 'problem_modeling', to: 'advanced_reasoning', label: '升级', type: 'builds_on', weight: 0.8, unlocked: false },
      { from: 'pattern_recognition', to: 'advanced_reasoning', label: '整合', type: 'reinforces', weight: 0.7, unlocked: false },
      { from: 'problem_solving', to: 'advanced_reasoning', label: '提升', type: 'builds_on', weight: 0.9, unlocked: false }
    ]

    setLearningNodes(nodes)
    setConnections(learningConnections)
  }

  // 🔗 获取与节点直接相关的节点（复制KnowledgeMap.tsx的核心逻辑）
  const getDirectlyRelatedNodes = (nodeId: string): string[] => {
    const relatedNodeIds = new Set<string>()
    
    connections.forEach(conn => {
      if (conn.from === nodeId) {
        relatedNodeIds.add(conn.to)
      } else if (conn.to === nodeId) {
        relatedNodeIds.add(conn.from)
      }
    })
    
    return Array.from(relatedNodeIds)
  }

  // 节点样式函数
  const getNodeColor = (category: string, state: string) => {
    const baseColors = {
      concept: '#3b82f6',    // 蓝色
      strategy: '#10b981',   // 绿色
      domain: '#f59e0b',     // 橙色
      skill: '#ef4444'       // 红色
    }
    
    const baseColor = baseColors[category] || '#6b7280'
    
    // 根据学习状态调整颜色
    switch (state) {
      case 'mastered': return baseColor
      case 'practicing': return baseColor + 'CC' // 80% 透明度
      case 'learning': return baseColor + '99'   // 60% 透明度
      case 'not_started': return baseColor + '66' // 40% 透明度
      default: return baseColor
    }
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

  const getLearningStateIcon = (state: string) => {
    const icons = {
      mastered: '✅',
      practicing: '🔄',
      learning: '📖',
      not_started: '⭕'
    }
    return icons[state] || '❓'
  }

  // 生成个性化学习路径
  const generateLearningPath = async () => {
    setIsAnalyzing(true)
    
    try {
      // 模拟分析过程
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      // 基于当前掌握情况生成学习路径
      const unmastered = learningNodes.filter(node => node.learning_state !== 'mastered')
      const sortedByPrerequisites = unmastered.sort((a, b) => a.prerequisites.length - b.prerequisites.length)
      
      setCurrentLearningPath(sortedByPrerequisites.map(node => node.id))
      
      // 更新激活状态
      setLearningNodes(prevNodes => 
        prevNodes.map(node => {
          const pathIndex = sortedByPrerequisites.findIndex(n => n.id === node.id)
          if (pathIndex !== -1) {
            return {
              ...node,
              activation_level: Math.max(0.3, 1 - (pathIndex * 0.1))
            }
          }
          return node
        })
      )
      
    } catch (error) {
      console.error('Learning path generation failed:', error)
    } finally {
      setIsAnalyzing(false)
    }
  }

  // 开始学习节点
  const startLearningNode = (nodeId: string) => {
    setLearningNodes(prev => 
      prev.map(node => 
        node.id === nodeId 
          ? { ...node, learning_state: 'learning', activation_level: 0.9 }
          : node
      )
    )
  }

  // 渲染连接线
  const renderConnections = () => {
    return connections.map((conn, index) => {
      const fromNode = learningNodes.find(n => n.id === conn.from)
      const toNode = learningNodes.find(n => n.id === conn.to)
      
      if (!fromNode || !toNode) return null

      const isRelated = selectedNode && (conn.from === selectedNode || conn.to === selectedNode)
      
      const strokeColor = conn.unlocked ? {
        prerequisite: '#94a3b8',
        builds_on: '#10b981',
        reinforces: '#f59e0b',
        applies: '#8b5cf6'
      }[conn.type] : '#d1d5db'

      return (
        <g key={index}>
          <line
            x1={fromNode.x}
            y1={fromNode.y}
            x2={toNode.x}
            y2={toNode.y}
            stroke={strokeColor}
            strokeWidth={isRelated ? 4 : 2}
            strokeDasharray={conn.unlocked ? '0' : '8,4'}
            opacity={!selectedNode || isRelated ? 0.7 : 0.3}
            className="transition-all duration-300"
          />
          
          <text
            x={(fromNode.x + toNode.x) / 2}
            y={(fromNode.y + toNode.y) / 2 - 5}
            textAnchor="middle"
            className="text-xs fill-gray-600"
            fontSize="9"
            opacity={!selectedNode || isRelated ? 0.8 : 0.4}
          >
            {conn.label}
          </text>
        </g>
      )
    })
  }

  // 渲染学习节点
  const renderNodes = () => {
    return learningNodes.map((node) => {
      const isSelected = selectedNode === node.id
      const isHovered = hoveredNode === node.id
      
      const directlyRelatedNodes = selectedNode ? getDirectlyRelatedNodes(selectedNode) : []
      const isDirectlyRelated = directlyRelatedNodes.includes(node.id)
      
      // 根据掌握程度和激活水平调整节点大小
      const baseRadius = 25
      const masteryBonus = node.mastery_level * 10
      const activationBonus = node.activation_level * 8
      const radius = isSelected ? baseRadius + masteryBonus + activationBonus + 5 : 
                    (isHovered ? baseRadius + masteryBonus + activationBonus + 3 : 
                     baseRadius + masteryBonus + activationBonus)
      
      const nodeColor = getNodeColor(node.category, node.learning_state)
      
      return (
        <g key={node.id}>
          {/* 掌握程度环形指示器 */}
          <circle
            cx={node.x}
            cy={node.y}
            r={radius + 6}
            fill="none"
            stroke={nodeColor}
            strokeWidth="3"
            strokeDasharray={`${node.mastery_level * 2 * Math.PI * (radius + 6)} ${2 * Math.PI * (radius + 6)}`}
            opacity={0.6}
            transform={`rotate(-90 ${node.x} ${node.y})`}
          />
          
          {/* 激活脉冲效果 */}
          {node.activation_level > 0.6 && (
            <circle
              cx={node.x}
              cy={node.y}
              r={radius + 12}
              fill={nodeColor}
              opacity={0.2}
              className="animate-pulse"
            />
          )}
          
          {/* 主节点 */}
          <circle
            cx={node.x}
            cy={node.y}
            r={radius}
            fill={nodeColor}
            stroke="#fff"
            strokeWidth="2"
            className="cursor-pointer drop-shadow-lg transition-all duration-300"
            opacity={!selectedNode || isSelected || isDirectlyRelated ? 1 : 0.4}
            onClick={() => setSelectedNode(isSelected ? null : node.id)}
            onMouseEnter={() => setHoveredNode(node.id)}
            onMouseLeave={() => setHoveredNode(null)}
          />
          
          {/* 节点图标 */}
          <text
            x={node.x}
            y={node.y - 8}
            textAnchor="middle"
            className="text-xs fill-white font-medium pointer-events-none"
            fontSize="12"
          >
            {getNodeIcon(node.category)}
          </text>
          
          {/* 学习状态图标 */}
          <text
            x={node.x}
            y={node.y + 5}
            textAnchor="middle"
            className="text-xs fill-white font-medium pointer-events-none"
            fontSize="10"
          >
            {getLearningStateIcon(node.learning_state)}
          </text>
          
          {/* 节点名称 */}
          <text
            x={node.x}
            y={node.y + 16}
            textAnchor="middle"
            className="text-xs fill-white font-medium pointer-events-none"
            fontSize="8"
          >
            {node.name}
          </text>
          
          {/* 掌握程度百分比 */}
          <text
            x={node.x}
            y={node.y + 26}
            textAnchor="middle"
            className="text-xs fill-white font-bold pointer-events-none"
            fontSize="7"
          >
            {(node.mastery_level * 100).toFixed(0)}%
          </text>
        </g>
      )
    })
  }

  return (
    <div className="space-y-6">
      {/* 学习指导标题 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <BookOpen className="h-5 w-5" />
            <span>📚 交互式物性图谱学习指导</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <p className="text-gray-600">
              基于激活扩散理论的个性化学习路径，通过知识节点激活建立系统性思维网络
            </p>
            
            <div className="flex justify-between items-center">
              <div className="text-sm text-gray-600">
                点击节点查看学习详情，环形进度显示掌握程度
              </div>
              <Button
                onClick={generateLearningPath}
                disabled={isAnalyzing}
                className="flex items-center space-x-2"
              >
                {isAnalyzing ? (
                  <>
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    >
                      <Zap className="h-4 w-4" />
                    </motion.div>
                    <span>分析中...</span>
                  </>
                ) : (
                  <>
                    <Target className="h-4 w-4" />
                    <span>生成学习路径</span>
                  </>
                )}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 学习进度概览 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {[
          { category: 'concept', name: '概念理解', icon: '💡', color: 'bg-blue-500' },
          { category: 'strategy', name: '策略掌握', icon: '🎯', color: 'bg-green-500' },
          { category: 'domain', name: '领域知识', icon: '📚', color: 'bg-orange-500' },
          { category: 'skill', name: '技能训练', icon: '🛠️', color: 'bg-red-500' }
        ].map((domain) => {
          const categoryNodes = learningNodes.filter(n => n.category === domain.category)
          const avgMastery = categoryNodes.reduce((sum, n) => sum + n.mastery_level, 0) / categoryNodes.length || 0
          
          return (
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
                  <p className="text-sm text-gray-600">{categoryNodes.length} 个节点</p>
                </div>
              </div>
              <div className="mb-2">
                <div className="flex justify-between text-xs text-gray-600 mb-1">
                  <span>掌握程度</span>
                  <span>{(avgMastery * 100).toFixed(0)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full ${domain.color}`}
                    style={{ width: `${avgMastery * 100}%` }}
                  ></div>
                </div>
              </div>
            </motion.div>
          )
        })}
      </div>

      {/* 交互式学习图谱 */}
      <Card>
        <CardHeader>
          <CardTitle>🕸️ 交互式学习知识图谱</CardTitle>
          <p className="text-sm text-gray-600">
            点击节点查看学习详情，环形进度条显示掌握程度，虚线表示未解锁路径
          </p>
        </CardHeader>
        <CardContent>
          <div className="relative">
            <svg
              ref={svgRef}
              width="100%"
              height="500"
              viewBox="0 0 600 600"
              className="border border-gray-200 rounded-lg bg-gray-50"
            >
              {/* 渲染连接线 */}
              {renderConnections()}

              {/* 渲染节点 */}
              {renderNodes()}
            </svg>

            {/* 图例 */}
            <div className="mt-4 space-y-2">
              <div className="flex flex-wrap gap-4 text-sm">
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 rounded-full bg-blue-500"></div>
                  <span>💡 概念节点</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 rounded-full bg-green-500"></div>
                  <span>🎯 策略节点</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 rounded-full bg-orange-500"></div>
                  <span>📚 领域节点</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 rounded-full bg-red-500"></div>
                  <span>🛠️ 技能节点</span>
                </div>
              </div>
              <div className="flex flex-wrap gap-4 text-sm text-gray-600">
                <div className="flex items-center space-x-2">
                  <span>✅ 已掌握</span>
                  <span>🔄 练习中</span>
                  <span>📖 学习中</span>
                  <span>⭕ 未开始</span>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 个性化学习路径 */}
      {currentLearningPath.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <MapPin className="h-5 w-5" />
              <span>🎯 个性化学习路径</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {currentLearningPath.map((nodeId, index) => {
                const node = learningNodes.find(n => n.id === nodeId)
                if (!node) return null
                
                return (
                  <motion.div
                    key={nodeId}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="flex items-center justify-between p-3 bg-white border rounded-lg hover:shadow-md transition-shadow"
                  >
                    <div className="flex items-center gap-3">
                      <div className="flex items-center justify-center w-8 h-8 bg-blue-100 text-blue-600 rounded-full text-sm font-bold">
                        {index + 1}
                      </div>
                      <div>
                        <div className="font-medium text-gray-800">
                          {getNodeIcon(node.category)} {node.name}
                        </div>
                        <div className="text-sm text-gray-600">
                          {node.difficulty} • 掌握度: {(node.mastery_level * 100).toFixed(0)}%
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="text-sm text-gray-500">
                        {getLearningStateIcon(node.learning_state)}
                      </div>
                      {node.learning_state === 'not_started' && (
                        <Button
                          size="sm"
                          onClick={() => startLearningNode(nodeId)}
                          className="text-xs"
                        >
                          开始学习
                        </Button>
                      )}
                    </div>
                  </motion.div>
                )
              })}
            </div>
          </CardContent>
        </Card>
      )}

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
                <div 
                  className="w-12 h-12 rounded-lg flex items-center justify-center text-white text-xl"
                  style={{ backgroundColor: getNodeColor(learningNodes.find(n => n.id === selectedNode)!.category, learningNodes.find(n => n.id === selectedNode)!.learning_state) }}
                >
                  {getNodeIcon(learningNodes.find(n => n.id === selectedNode)!.category)}
                </div>
                <div>
                  <CardTitle className="flex items-center gap-2">
                    {learningNodes.find(n => n.id === selectedNode)!.name}
                    <span className="text-lg">
                      {getLearningStateIcon(learningNodes.find(n => n.id === selectedNode)!.learning_state)}
                    </span>
                  </CardTitle>
                  <p className="text-sm text-gray-600">
                    掌握程度: {(learningNodes.find(n => n.id === selectedNode)!.mastery_level * 100).toFixed(1)}% • 
                    难度: {learningNodes.find(n => n.id === selectedNode)!.difficulty}
                  </p>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* 学习目标 */}
                <div>
                  <h4 className="font-semibold text-gray-800 mb-2 flex items-center gap-2">
                    <Target className="h-4 w-4" />
                    学习目标
                  </h4>
                  <ul className="space-y-1">
                    {learningNodes.find(n => n.id === selectedNode)!.learning_objectives.map((objective, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                        <span className="text-blue-500 mt-1">•</span>
                        <span>{objective}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* 练习示例 */}
                <div>
                  <h4 className="font-semibold text-gray-800 mb-2 flex items-center gap-2">
                    <BookMarked className="h-4 w-4" />
                    练习示例
                  </h4>
                  <div className="space-y-2">
                    {learningNodes.find(n => n.id === selectedNode)!.practice_examples.map((example, i) => (
                      <div key={i} className="bg-gray-50 p-2 rounded text-sm text-gray-700">
                        {example}
                      </div>
                    ))}
                  </div>
                </div>

                {/* 掌握指标 */}
                <div>
                  <h4 className="font-semibold text-gray-800 mb-2 flex items-center gap-2">
                    <Award className="h-4 w-4" />
                    掌握指标
                  </h4>
                  <ul className="space-y-1">
                    {learningNodes.find(n => n.id === selectedNode)!.mastery_indicators.map((indicator, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                        <span className="text-green-500 mt-1">✓</span>
                        <span>{indicator}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* 前置条件 */}
                {learningNodes.find(n => n.id === selectedNode)!.prerequisites.length > 0 && (
                  <div>
                    <h4 className="font-semibold text-gray-800 mb-2">📋 前置条件</h4>
                    <div className="flex flex-wrap gap-2">
                      {learningNodes.find(n => n.id === selectedNode)!.prerequisites.map((prereqId) => {
                        const prereqNode = learningNodes.find(n => n.id === prereqId)
                        return prereqNode ? (
                          <button
                            key={prereqId}
                            onClick={() => setSelectedNode(prereqId)}
                            className="px-3 py-1 bg-blue-100 hover:bg-blue-200 text-blue-800 rounded-full text-sm transition-colors"
                          >
                            {getNodeIcon(prereqNode.category)} {prereqNode.name}
                          </button>
                        ) : null
                      })}
                    </div>
                  </div>
                )}

                {/* 学习操作 */}
                <div className="flex gap-2 pt-2 border-t">
                  {learningNodes.find(n => n.id === selectedNode)!.learning_state === 'not_started' && (
                    <Button onClick={() => startLearningNode(selectedNode!)} className="flex items-center gap-2">
                      <Play className="h-4 w-4" />
                      开始学习
                    </Button>
                  )}
                  {learningNodes.find(n => n.id === selectedNode)!.learning_state === 'learning' && (
                    <Button variant="outline" className="flex items-center gap-2">
                      <BookOpen className="h-4 w-4" />
                      继续学习
                    </Button>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}
    </div>
  )
}

export default InteractivePropertyLearningGuide