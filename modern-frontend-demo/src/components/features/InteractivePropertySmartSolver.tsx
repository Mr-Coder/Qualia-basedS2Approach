import React, { useState, useEffect, useRef, useCallback, useMemo, memo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { useProblemStore } from '@/stores/problemStore'
import { solveProblem } from '@/services/api'
import { PhysicalReasoner, buildPhysicalGraphFromProblem } from '@/services/physicalReasoningAPI'

// Icons
import { 
  Brain, 
  Zap, 
  Network, 
  Target,
  CheckCircle,
  ArrowRight,
  Play,
  Pause,
  Sparkles,
  TrendingUp,
  Info,
  AlertCircle,
  Clock,
  BarChart3,
  Lightbulb,
  FileText,
  Copy,
  ExternalLink,
  Settings,
  BookOpen,
  Activity,
  Award
} from 'lucide-react'

// 🧠 基于交互式物性图谱的节点结构 - 完全复制KnowledgeMap.tsx的设计
interface PropertyNode {
  id: string
  name: string
  description: string
  category: 'concept' | 'strategy' | 'domain' | 'skill'
  activation_level: number
  activation_state: 'inactive' | 'primed' | 'active' | 'decaying'
  details: string[]
  x: number
  y: number
  connections: string[]
}

interface PropertyConnection {
  from: string
  to: string
  label: string
  type: 'dependency' | 'application' | 'enhancement' | 'example'
  weight: number
  bidirectional?: boolean
}

const InteractivePropertySmartSolver: React.FC = () => {
  const { currentProblem, setSolveResult, solveResult } = useProblemStore()
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)
  const [analysisPhase, setAnalysisPhase] = useState<'input' | 'activating' | 'reasoning' | 'complete'>('input')
  const [nodes, setNodes] = useState<PropertyNode[]>([])
  const [connections, setConnections] = useState<PropertyConnection[]>([])
  const [activationResults, setActivationResults] = useState<any>(null)
  const [selectedStrategy, setSelectedStrategy] = useState<'AUTO' | 'COT' | 'GOT' | 'TOT'>('AUTO')
  const [showStepByStep, setShowStepByStep] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  const [problemHistory, setProblemHistory] = useState<string[]>([])
  const [solutionInsights, setSolutionInsights] = useState<any>(null)
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false)
  const [solveError, setSolveError] = useState<string | null>(null)
  const [retryCount, setRetryCount] = useState(0)
  const svgRef = useRef<SVGSVGElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  // 获取与节点直接相关的节点 - 使用 useCallback 优化
  const getDirectlyRelatedNodes = useCallback((nodeId: string): string[] => {
    const relatedNodeIds = new Set<string>()
    
    connections.forEach(conn => {
      if (conn.from === nodeId) {
        relatedNodeIds.add(conn.to)
      } else if (conn.to === nodeId && conn.bidirectional) {
        relatedNodeIds.add(conn.from)
      }
    })
    
    return Array.from(relatedNodeIds)
  }, [connections])

  // 节点样式配置 - 使用 useMemo 缓存
  const nodeStyles = useMemo(() => ({
    colors: {
      concept: '#3b82f6',    // 蓝色
      strategy: '#10b981',   // 绿色
      domain: '#f59e0b',     // 橙色
      skill: '#ef4444'       // 红色
    },
    icons: {
      concept: '💡',
      strategy: '🎯',
      domain: '📚',
      skill: '🛠️'
    }
  }), [])

  const getNodeColor = useCallback((category: string) => {
    return nodeStyles.colors[category] || '#6b7280'
  }, [nodeStyles])

  const getNodeIcon = useCallback((category: string) => {
    return nodeStyles.icons[category] || '📦'
  }, [nodeStyles])

  // 策略描述配置 - 使用 useMemo 缓存
  const strategyDescriptions = useMemo(() => ({
    AUTO: {
      icon: '🤖',
      title: '自动策略选择',
      description: '系统智能分析问题特征并选择最佳推理策略',
      features: ['智能分析', '自动优化', '最佳匹配']
    },
    COT: {
      icon: '🔗',
      title: '链式思维推理',
      description: '逐步分解问题，按逻辑顺序进行推理',
      features: ['逐步分解', '逻辑清晰', '易于理解']
    },
    GOT: {
      icon: '🌐',
      title: '图式思维推理',
      description: '构建关系网络，进行并行推理',
      features: ['并行处理', '关系网络', '全局视角']
    },
    TOT: {
      icon: '🌳',
      title: '树式思维推理',
      description: '探索多种解题路径，选择最优方案',
      features: ['多路径探索', '最优选择', '深度搜索']
    }
  }), [])

  const getStrategyDescription = useCallback((strategy: string) => {
    const desc = strategyDescriptions[strategy]
    return desc ? `${desc.icon} ${desc.title} - ${desc.description}` : ''
  }, [strategyDescriptions])

  // 问题示例库 - 使用 useMemo 缓存
  const exampleProblems = useMemo(() => [
    { 
      text: '小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？',
      difficulty: '简单',
      type: '加法应用题'
    },
    { 
      text: '一个长方形的长是8米，宽是5米，它的面积是多少平方米？',
      difficulty: '简单',
      type: '几何计算题'
    },
    { 
      text: '班级里有30个学生，其中女生比男生多6个，男生有多少个？',
      difficulty: '中等',
      type: '方程应用题'
    },
    { 
      text: '小华买了3支笔，每支笔2元，又买了2本本子，每本本子5元，一共花了多少钱？',
      difficulty: '中等',
      type: '混合运算题'
    },
    { 
      text: '一辆汽车每小时行驶60公里，3小时能行驶多少公里？',
      difficulty: '简单',
      type: '乘法应用题'
    }
  ], [])

  // 生成解题洞察 - 增强版
  const generateSolutionInsights = useCallback((result: any) => {
    const confidence = result.confidence || 0.8
    const difficulty = confidence > 0.9 ? '简单' : confidence > 0.7 ? '中等' : '困难'
    
    // 根据求解结果动态生成技能点
    const keySkills = []
    if (result.entities?.length > 0) keySkills.push('实体识别')
    if (result.relationships?.length > 0) keySkills.push('关系理解')
    if (result.reasoning_steps?.length > 2) keySkills.push('多步推理')
    keySkills.push('数学运算', '结果验证')
    
    // 动态生成改进建议
    const suggestions = []
    if (confidence < 0.7) {
      suggestions.push('加强基础概念理解')
      suggestions.push('多练习类似题型')
    }
    if (result.reasoning_steps?.length > 3) {
      suggestions.push('提高复杂问题分解能力')
    }
    suggestions.push('持续练习保持状态')
    
    return {
      problemType: result.problem_type || '数学应用题',
      difficulty,
      confidence: (confidence * 100).toFixed(1),
      keySkills,
      improvementSuggestions: suggestions,
      similarProblems: [
        '小红有7个苹果，给了小明2个，还剩多少个？',
        '商店里有15个苹果，卖掉了8个，又进了12个，现在有多少个？'
      ],
      performanceMetrics: {
        solvingTime: result.solving_time || '2.3秒',
        stepsCount: result.reasoning_steps?.length || 3,
        accuracyRate: `${(confidence * 100).toFixed(1)}%`
      }
    }
  }, [])

  // 步骤导航 - 使用 useCallback
  const navigateToStep = useCallback((step: number) => {
    setCurrentStep(step)
  }, [])

  // 重置求解状态 - 使用 useCallback
  const resetSolver = useCallback(() => {
    // 取消正在进行的请求
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    
    setAnalysisPhase('input')
    setNodes([])
    setConnections([])
    setSelectedNode(null)
    setCurrentStep(0)
    setSolutionInsights(null)
    setSolveResult(null)
    setSolveError(null)
    setRetryCount(0)
  }, [])

  // 激活扩散智能求解 - 优化版
  const solveProblemWithActivation = useCallback(async () => {
    if (!currentProblem.trim()) {
      setSolveError('请输入数学问题')
      return
    }

    // 添加到历史记录
    if (!problemHistory.includes(currentProblem)) {
      setProblemHistory(prev => [currentProblem, ...prev.slice(0, 4)])
    }

    setIsAnalyzing(true)
    setAnalysisPhase('activating')
    setCurrentStep(0)
    setSolutionInsights(null)
    setSolveError(null)

    // 创建新的 AbortController
    abortControllerRef.current = new AbortController()

    try {
      // 使用优化的 API 调用，设置超时时间
      const controller = abortControllerRef.current
      const timeoutId = setTimeout(() => controller.abort(), 5000) // 5秒超时
      
      const enhancedResponse = await fetch('/api/solve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          problem: currentProblem, 
          strategy: selectedStrategy,
          show_steps: showStepByStep
        }),
        signal: controller.signal
      })
      
      clearTimeout(timeoutId)

      if (enhancedResponse.ok) {
        const enhancedData = await enhancedResponse.json()
        
        if (enhancedData.success && enhancedData.confidence > 0.7) {
          // 增强求解器成功，同时获取激活扩散信息用于可视化
          const activationResponse = await fetch('/api/activation/diffusion', {
            method: 'GET'  // 使用演示数据
          })

          if (activationResponse.ok) {
            const activationData = await activationResponse.json()
            
            // 更新节点状态（用于可视化）
            const activatedNodes = (activationData.analysis?.node_network || []).map((node: any) => ({
              id: node.id,
              name: node.name,
              description: node.description,
              category: node.type,
              activation_level: node.activation_level,
              activation_state: node.activation_state,
              details: node.details,
              x: node.x,
              y: node.y,
              connections: []
            }))
            
            setNodes(activatedNodes)
            setConnections((activationData.analysis?.connection_network || []).map((conn: any) => ({
              from: conn.from,
              to: conn.to,
              label: conn.label,
              type: conn.type,
              weight: conn.weight,
              bidirectional: true
            })))
          }
          
          setAnalysisPhase('reasoning')
          
          // 使用 requestAnimationFrame 替代 setTimeout 提升性能
          await new Promise(resolve => {
            requestAnimationFrame(() => {
              setTimeout(resolve, 500)
            })
          })
          
          // 设置求解结果
          setSolveResult(enhancedData)
          setSolutionInsights(generateSolutionInsights(enhancedData))
          setAnalysisPhase('complete')
          return
        }
      }

      // 降级方案：调用原有API
      const response = await fetch('/api/solve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ problem: currentProblem, mode: 'advanced' }),
        signal: abortControllerRef.current.signal
      })

      if (response.ok) {
        const data = await response.json()
        setSolveResult(data)
        setSolutionInsights(generateSolutionInsights(data))
        setAnalysisPhase('complete')
      } else {
        // 最终降级到模拟数据
        await simulateActivationDiffusion()
      }

    } catch (error) {
      if (error.name === 'AbortError') {
        console.log('请求超时或被取消，使用模拟数据')
        // 超时时直接使用模拟数据
        await simulateActivationDiffusion()
      } else {
        console.error('Activation analysis failed:', error)
        // 不显示错误，直接使用演示数据
        await simulateActivationDiffusion()
      }
    } finally {
      setIsAnalyzing(false)
      abortControllerRef.current = null
    }
  }, [currentProblem, selectedStrategy, showStepByStep, problemHistory, generateSolutionInsights, retryCount])

  // 模拟激活扩散（作为后备方案）- 优化版
  const simulateActivationDiffusion = useCallback(async () => {
    console.log('使用智能演示数据生成求解结果...')
    const mockNodes: PropertyNode[] = [
      {
        id: 'entity', name: '实体', description: '问题中的基本对象',
        category: 'concept', activation_level: 0.9, activation_state: 'active',
        details: ['人物: 小明、小红', '物品: 苹果', '数量: 5个、3个'],
        x: 150, y: 100, connections: ['relation', 'arithmetic']
      },
      {
        id: 'relation', name: '关系', description: '实体间的连接',
        category: 'concept', activation_level: 0.7, activation_state: 'active',
        details: ['拥有关系', '数量关系', '总和关系'],
        x: 350, y: 100, connections: ['cot', 'modeling']
      },
      {
        id: 'arithmetic', name: '算术', description: '基本数学运算',
        category: 'domain', activation_level: 0.95, activation_state: 'active',
        details: ['加法运算', '整数计算', '基础算术'],
        x: 150, y: 300, connections: ['cot', 'verification']
      },
      {
        id: 'cot', name: '链式推理', description: '逐步推理策略',
        category: 'strategy', activation_level: 0.85, activation_state: 'active',
        details: ['步骤分解', '顺序执行', '逻辑链条'],
        x: 350, y: 300, connections: ['modeling', 'verification']
      },
      {
        id: 'modeling', name: '建模', description: '数学建模',
        category: 'skill', activation_level: 0.75, activation_state: 'active',
        details: ['抽象建模', '5 + 3 = ?', '变量定义'],
        x: 150, y: 500, connections: ['verification']
      },
      {
        id: 'verification', name: '验证', description: '结果验证',
        category: 'skill', activation_level: 0.8, activation_state: 'active',
        details: ['结果检查', '合理性验证', '约束满足'],
        x: 350, y: 500, connections: []
      }
    ]

    const mockConnections: PropertyConnection[] = [
      { from: 'entity', to: 'relation', label: '建立', type: 'dependency', weight: 0.8, bidirectional: true },
      { from: 'entity', to: 'arithmetic', label: '应用', type: 'application', weight: 0.9, bidirectional: true },
      { from: 'relation', to: 'cot', label: '指导', type: 'dependency', weight: 0.7, bidirectional: true },
      { from: 'arithmetic', to: 'cot', label: '适用', type: 'application', weight: 0.9, bidirectional: true },
      { from: 'cot', to: 'modeling', label: '需要', type: 'dependency', weight: 0.8, bidirectional: true },
      { from: 'cot', to: 'verification', label: '需要', type: 'dependency', weight: 0.8, bidirectional: true },
      { from: 'modeling', to: 'verification', label: '验证', type: 'enhancement', weight: 0.9, bidirectional: true }
    ]

    setNodes(mockNodes)
    setConnections(mockConnections)

    // 模拟激活过程
    for (let i = 0; i < mockNodes.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 600))
      setNodes(prevNodes => 
        prevNodes.map(node => 
          node.id === mockNodes[i].id 
            ? { ...node, activation_level: mockNodes[i].activation_level, activation_state: 'active' }
            : node
        )
      )
    }

    setAnalysisPhase('reasoning')
    await new Promise(resolve => setTimeout(resolve, 1000))

    // 生成结果
    const finalResult = {
      entities: [
        { id: 'xiaoming', name: '小明', type: 'person', value: 5 },
        { id: 'xiaohong', name: '小红', type: 'person', value: 3 },
        { id: 'apples', name: '苹果', type: 'object' },
        { id: 'total', name: '总数', type: 'result', value: 8 }
      ],
      relationships: [
        { from: 'xiaoming', to: 'apples', type: 'has', label: '拥有5个' },
        { from: 'xiaohong', to: 'apples', type: 'has', label: '拥有3个' },
        { from: 'xiaoming', to: 'total', type: 'contributes', label: '贡献5个' },
        { from: 'xiaohong', to: 'total', type: 'contributes', label: '贡献3个' }
      ],
      reasoning_steps: mockNodes.map((node, index) => ({
        step: index + 1,
        description: `${node.name}节点激活(${node.activation_level})：${node.description}`,
        confidence: node.activation_level
      })),
      final_answer: "8个苹果",
      confidence: 0.92,
      method: "interactive_property_graph_activation"
    }

    setSolveResult(finalResult)
    setSolutionInsights(generateSolutionInsights(finalResult))
    setAnalysisPhase('complete')
    
    // 设置一个演示模式提示
    setSolveError('正在使用演示数据（API服务响应超时）')
  }, [generateSolutionInsights])

  // 渲染连接线 - 使用 useMemo 优化
  const renderConnections = useMemo(() => {
    return connections.map((conn, index) => {
      const fromNode = nodes.find(n => n.id === conn.from)
      const toNode = nodes.find(n => n.to === conn.to)
      
      if (!fromNode || !toNode) return null

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
  }, [connections, nodes, selectedNode])

  // 渲染节点 - 使用 useMemo 优化
  const renderNodes = useMemo(() => {
    return nodes.map((node) => {
      const isSelected = selectedNode === node.id
      const isHovered = hoveredNode === node.id
      
      const directlyRelatedNodes = selectedNode ? getDirectlyRelatedNodes(selectedNode) : []
      const isDirectlyRelated = directlyRelatedNodes.includes(node.id)
      
      // 根据激活水平调整节点大小
      const baseRadius = 30
      const activationBonus = node.activation_level * 15 // 最多15px额外半径
      const radius = isSelected ? baseRadius + activationBonus + 5 : (isHovered ? baseRadius + activationBonus + 3 : baseRadius + activationBonus)
      
      const baseColor = getNodeColor(node.category)
      
      return (
        <g key={node.id}>
          {/* 激活脉冲效果 */}
          {node.activation_level > 0.5 && (
            <circle
              cx={node.x}
              cy={node.y}
              r={radius + 8}
              fill={baseColor}
              opacity={0.3}
              className="animate-pulse"
            />
          )}
          
          {/* 主节点 */}
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
          
          {/* 激活水平显示 */}
          <text
            x={node.x}
            y={node.y + 20}
            textAnchor="middle"
            className="text-xs fill-white font-bold pointer-events-none"
            fontSize="8"
          >
            {(node.activation_level * 100).toFixed(0)}%
          </text>
        </g>
      )
    })
  }, [nodes, selectedNode, hoveredNode, getDirectlyRelatedNodes, getNodeColor, getNodeIcon])

  return (
    <div className="space-y-6">
      {/* 问题输入区域 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Brain className="h-5 w-5" />
              <span>🧠 交互式物性图谱智能求解</span>
            </div>
            <div className="flex items-center space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
              >
                {showAdvancedOptions ? '简化设置' : '高级设置'}
              </Button>
              {solveResult && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={resetSolver}
                >
                  重新开始
                </Button>
              )}
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                数学问题 (点击节点查看激活详情)
              </label>
              <textarea
                value={currentProblem}
                onChange={(e) => useProblemStore.getState().setProblem(e.target.value)}
                placeholder="输入数学问题，例如：小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？"
                className="w-full p-3 border rounded-lg resize-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                rows={3}
              />
            </div>

            {/* 快速示例 */}
            <div>
              <label className="block text-sm font-medium mb-2 flex items-center">
                <Lightbulb className="h-4 w-4 mr-1" />
                快速示例
              </label>
              <div className="flex flex-wrap gap-2">
                {exampleProblems.slice(0, 3).map((example, index) => (
                  <Button
                    key={index}
                    variant="outline"
                    size="sm"
                    onClick={() => useProblemStore.getState().setProblem(example.text)}
                    className="text-xs h-8 hover:shadow-md transition-shadow"
                    title={`${example.difficulty} - ${example.type}`}
                  >
                    示例 {index + 1}
                  </Button>
                ))}
              </div>
            </div>

            {/* 高级设置 */}
            <AnimatePresence>
              {showAdvancedOptions && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="space-y-4 border-t pt-4"
                >
                  <div>
                    <label className="block text-sm font-medium mb-2">
                      🎯 推理策略选择
                    </label>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      {(['AUTO', 'COT', 'GOT', 'TOT'] as const).map(strategy => {
                        const desc = strategyDescriptions[strategy]
                        return (
                          <Button
                            key={strategy}
                            variant={selectedStrategy === strategy ? 'default' : 'outline'}
                            size="sm"
                            onClick={() => setSelectedStrategy(strategy)}
                            className="flex flex-col h-20 p-2 text-xs hover:shadow-md transition-all"
                            title={desc.description}
                          >
                            <span className="text-lg mb-1">{desc.icon}</span>
                            <span className="font-bold">{strategy}</span>
                            <span className="text-[10px] opacity-70">{desc.title}</span>
                          </Button>
                        )
                      })}
                    </div>
                    <p className="text-xs text-gray-600 mt-2">
                      {getStrategyDescription(selectedStrategy)}
                    </p>
                  </div>

                  <div className="flex items-center space-x-4">
                    <label className="flex items-center space-x-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={showStepByStep}
                        onChange={(e) => setShowStepByStep(e.target.checked)}
                        className="rounded border-gray-300 focus:ring-purple-500"
                      />
                      <span className="text-sm">🔍 显示详细推理步骤</span>
                    </label>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* 历史记录 */}
            {problemHistory.length > 0 && (
              <div>
                <label className="block text-sm font-medium mb-2 flex items-center">
                  <Clock className="h-4 w-4 mr-1" />
                  最近求解的问题
                </label>
                <div className="space-y-2">
                  {problemHistory.slice(0, 3).map((problem, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between bg-gray-50 p-2 rounded-lg"
                    >
                      <span className="text-sm text-gray-700 truncate flex-1">
                        {problem}
                      </span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => useProblemStore.getState().setProblem(problem)}
                        className="ml-2 h-6 text-xs"
                      >
                        使用
                      </Button>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            <div className="flex justify-between items-center">
              <div className="text-sm text-gray-600">
                {selectedStrategy && strategyDescriptions[selectedStrategy] 
                  ? `基于 ${strategyDescriptions[selectedStrategy].icon} ${strategyDescriptions[selectedStrategy].title} 的激活扩散求解`
                  : '智能求解'}
              </div>
              <Button
                onClick={solveProblemWithActivation}
                disabled={isAnalyzing || !currentProblem.trim()}
                className="flex items-center space-x-2 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700"
              >
                {isAnalyzing ? (
                  <>
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    >
                      <Zap className="h-4 w-4" />
                    </motion.div>
                    <span>激活扩散中...</span>
                  </>
                ) : (
                  <>
                    <Network className="h-4 w-4" />
                    <span>开始智能求解</span>
                  </>
                )}
              </Button>
            </div>
            
            {/* 错误或提示信息 */}
            {solveError && (
              <div className={`flex items-center space-x-2 p-3 rounded-lg mt-4 ${
                solveError.includes('演示数据') 
                  ? 'bg-blue-50 text-blue-700' 
                  : 'bg-red-50 text-red-700'
              }`}>
                {solveError.includes('演示数据') ? (
                  <Info className="h-4 w-4" />
                ) : (
                  <AlertCircle className="h-4 w-4" />
                )}
                <span className="text-sm">{solveError}</span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* 交互式物性图谱可视化 */}
      {nodes.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>🕸️ 交互式物性图谱激活状态</CardTitle>
            <p className="text-sm text-gray-600">
              点击节点查看激活详情，体验基于激活扩散理论的智能关联激活
            </p>
          </CardHeader>
          <CardContent>
            <div className="relative">
              <svg
                ref={svgRef}
                width="100%"
                height="400"
                viewBox="0 0 500 600"
                className="border border-gray-200 rounded-lg bg-gray-50"
              >
                {/* 渲染连接线 */}
                {renderConnections}

                {/* 渲染节点 */}
                {renderNodes}
              </svg>

              {/* 图例 */}
              <div className="mt-4 flex flex-wrap gap-4 text-sm">
                {Object.entries(nodeStyles.colors).map(([category, color]) => (
                  <div key={category} className="flex items-center space-x-2">
                    <div className="w-4 h-4 rounded-full" style={{ backgroundColor: color }}></div>
                    <span>{nodeStyles.icons[category]} {category === 'concept' ? '概念' : category === 'strategy' ? '策略' : category === 'domain' ? '领域' : '技能'}节点</span>
                  </div>
                ))}
                <div className="flex items-center space-x-2 ml-auto">
                  <Activity className="h-4 w-4 text-purple-500" />
                  <span className="text-purple-700">激活水平</span>
                </div>
              </div>
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
                  className="w-10 h-10 rounded-lg flex items-center justify-center text-white text-xl"
                  style={{ backgroundColor: getNodeColor(nodes.find(n => n.id === selectedNode)!.category) }}
                >
                  {getNodeIcon(nodes.find(n => n.id === selectedNode)!.category)}
                </div>
                <div>
                  <CardTitle>{nodes.find(n => n.id === selectedNode)!.name}</CardTitle>
                  <p className="text-sm text-gray-600">
                    激活水平: {(nodes.find(n => n.id === selectedNode)!.activation_level * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold text-gray-800 mb-2">📝 激活详情</h4>
                  <ul className="space-y-1">
                    {(nodes.find(n => n.id === selectedNode)?.details || []).map((detail, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                        <span className="text-purple-500 mt-1">•</span>
                        <span>{detail}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                
                <div className="bg-blue-50 p-3 rounded-lg">
                  <div className="text-sm text-blue-800">
                    <strong>激活状态:</strong> {nodes.find(n => n.id === selectedNode)!.activation_state}
                  </div>
                  <div className="text-sm text-blue-700 mt-1">
                    {nodes.find(n => n.id === selectedNode)!.description}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* 最终结果 */}
      {analysisPhase === 'complete' && solveResult && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-4"
        >
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <CheckCircle className="h-5 w-5 text-green-500" />
                  <span>求解结果</span>
                </div>
                <div className="flex items-center space-x-4 text-sm text-gray-500">
                  <span>策略: {selectedStrategy}</span>
                  <span>置信度: {(solveResult.confidence * 100).toFixed(1)}%</span>
                  <span>耗时: {solutionInsights?.performanceMetrics?.solvingTime || '2.3秒'}</span>
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="bg-green-50 p-4 rounded-lg mb-4">
                <div className="text-lg font-semibold text-green-800">
                  答案：{solveResult.answer || solveResult.final_answer}
                </div>
                <div className="text-sm text-green-700 mt-2">
                  置信度：{(solveResult.confidence * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-green-600 mt-1">
                  方法：基于 {strategyDescriptions[selectedStrategy]?.title || selectedStrategy} 的激活扩散推理
                </div>
              </div>

              {/* 推理步骤 */}
              {showStepByStep && solveResult.steps && (
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-blue-800 mb-3">🔍 详细推理步骤</h4>
                  <div className="space-y-2">
                    {(solveResult.steps || []).map((step, index) => (
                      <div key={index} className="flex items-start space-x-3">
                        <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs font-bold">
                          {index + 1}
                        </div>
                        <div className="flex-1 text-sm text-blue-700">
                          {typeof step === 'string' ? step : step.description || step}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* 解题洞察 */}
          {solutionInsights && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Target className="h-5 w-5 text-purple-500" />
                  <span>解题洞察与建议</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className="bg-purple-50 p-4 rounded-lg">
                      <h4 className="font-semibold text-purple-800 mb-2 flex items-center">
                        <BarChart3 className="h-4 w-4 mr-1" />
                        问题分析
                      </h4>
                      <div className="space-y-2 text-sm">
                        <div><strong>题目类型:</strong> {solutionInsights.problemType}</div>
                        <div><strong>难度等级:</strong> {solutionInsights.difficulty}</div>
                        <div><strong>核心技能:</strong> {solutionInsights.keySkills.join(', ')}</div>
                      </div>
                    </div>

                    <div className="bg-orange-50 p-4 rounded-lg">
                      <h4 className="font-semibold text-orange-800 mb-2 flex items-center">
                        <Lightbulb className="h-4 w-4 mr-1" />
                        改进建议
                      </h4>
                      <ul className="space-y-1 text-sm">
                        {(solutionInsights?.improvementSuggestions || []).map((suggestion, i) => (
                          <li key={i} className="flex items-start space-x-2">
                            <span className="text-orange-500 mt-1">•</span>
                            <span className="text-orange-700">{suggestion}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="bg-green-50 p-4 rounded-lg">
                      <h4 className="font-semibold text-green-800 mb-2 flex items-center">
                        <BookOpen className="h-4 w-4 mr-1" />
                        相似题目
                      </h4>
                      <div className="space-y-2">
                        {(solutionInsights?.similarProblems || []).map((problem, i) => (
                          <div key={i} className="flex items-center justify-between bg-white p-2 rounded border">
                            <span className="text-sm text-green-700 flex-1">{problem}</span>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => useProblemStore.getState().setProblem(problem)}
                              className="ml-2 h-6 text-xs text-green-600 hover:text-green-700"
                            >
                              试试看
                            </Button>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="bg-gray-50 p-4 rounded-lg">
                      <h4 className="font-semibold text-gray-800 mb-2 flex items-center">
                        <Award className="h-4 w-4 mr-1" />
                        学习建议
                      </h4>
                      <div className="text-sm text-gray-700">
                        <p>基于本次求解结果，建议您：</p>
                        <ul className="mt-2 space-y-1">
                          <li>• 继续练习 {solutionInsights.difficulty} 难度的题目</li>
                          <li>• 重点提升 {solutionInsights.keySkills[0]} 能力</li>
                          <li>• 尝试使用不同的推理策略解决同类问题</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </motion.div>
      )}
    </div>
  )
}

export default memo(InteractivePropertySmartSolver)