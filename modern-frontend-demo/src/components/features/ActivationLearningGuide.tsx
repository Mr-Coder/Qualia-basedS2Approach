import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import ActivationPropertyGraph from './ActivationPropertyGraph'
import { getPersonalizedLearningPaths, getLearningTechniques, type LearningPathRequest, type ActivationLearningResponse } from '@/services/api'

// Icons
import { 
  BookOpen, 
  Target, 
  TrendingUp, 
  CheckCircle,
  ArrowRight,
  Lightbulb,
  MapPin,
  Zap
} from 'lucide-react'

interface LearningNode {
  id: string
  name: string
  description: string
  type: 'concept' | 'strategy' | 'domain' | 'skill'
  difficulty: 'beginner' | 'intermediate' | 'advanced'
  activation_level: number
  prerequisites: string[]
  learning_objectives: string[]
  practice_examples: string[]
  mastery_indicators: string[]
}

interface LearningPath {
  id: string
  name: string
  description: string
  nodes: LearningNode[]
  estimated_time: string
  difficulty_level: string
}

const ActivationLearningGuide: React.FC = () => {
  const [selectedPath, setSelectedPath] = useState<string | null>(null)
  const [currentNode, setCurrentNode] = useState<number>(0)
  const [masteredNodes, setMasteredNodes] = useState<Set<string>>(new Set())
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [learningProgress, setLearningProgress] = useState<{[key: string]: number}>({})
  const [activeTab, setActiveTab] = useState<'pathways' | 'stages' | 'techniques'>('pathways')
  const [apiLearningData, setApiLearningData] = useState<ActivationLearningResponse | null>(null)
  const [userLevel, setUserLevel] = useState<'beginner' | 'intermediate' | 'advanced'>('beginner')
  const [userProgress, setUserProgress] = useState<{[key: number]: 'completed' | 'in_progress' | 'available' | 'locked'}>({1: 'available'})

  // 基于激活扩散的学习路径
  const learningPaths: LearningPath[] = [
    {
      id: 'basic_arithmetic',
      name: '基础算术激活路径',
      description: '通过激活扩散理论学习基础数学运算',
      estimated_time: '2-3小时',
      difficulty_level: '初级',
      nodes: [
        {
          id: 'entity_recognition',
          name: '实体识别',
          description: '学会识别数学问题中的基本对象',
          type: 'concept',
          difficulty: 'beginner',
          activation_level: 0.9,
          prerequisites: [],
          learning_objectives: [
            '能够识别问题中的人物、物品、数量',
            '理解实体在数学问题中的作用',
            '建立实体与数学运算的联系'
          ],
          practice_examples: [
            '小明有5个苹果 → 识别：小明(人物)、苹果(物品)、5(数量)',
            '书店有30本书 → 识别：书店(地点)、书(物品)、30(数量)',
            '小红买了3支笔 → 识别：小红(人物)、笔(物品)、3(数量)'
          ],
          mastery_indicators: [
            '能快速识别所有实体',
            '准确分类实体类型',
            '理解实体间的基本关系'
          ]
        },
        {
          id: 'arithmetic_operations',
          name: '算术运算',
          description: '掌握基本的加减乘除运算',
          type: 'domain',
          difficulty: 'beginner',
          activation_level: 0.95,
          prerequisites: ['entity_recognition'],
          learning_objectives: [
            '熟练掌握四则运算',
            '理解运算符号的含义',
            '能够进行心算和笔算'
          ],
          practice_examples: [
            '5 + 3 = 8（加法：合并两个数量）',
            '10 - 4 = 6（减法：从总数中去除部分）',
            '3 × 4 = 12（乘法：重复相加）'
          ],
          mastery_indicators: [
            '运算速度和准确度达标',
            '理解运算的实际意义',
            '能解释运算过程'
          ]
        },
        {
          id: 'step_by_step_reasoning',
          name: '逐步推理',
          description: '学习链式思维的推理方法',
          type: 'strategy',
          difficulty: 'intermediate',
          activation_level: 0.85,
          prerequisites: ['entity_recognition', 'arithmetic_operations'],
          learning_objectives: [
            '学会分解复杂问题',
            '掌握逻辑推理顺序',
            '建立清晰的思维链条'
          ],
          practice_examples: [
            '步骤1：识别已知条件 → 步骤2：确定运算类型 → 步骤3：执行计算',
            '复杂问题分解为多个子问题',
            '每一步都有明确的理由和依据'
          ],
          mastery_indicators: [
            '能够独立分解问题',
            '推理过程清晰有序',
            '每步都有逻辑支撑'
          ]
        },
        {
          id: 'result_verification',
          name: '结果验证',
          description: '学习检查和验证答案的方法',
          type: 'skill',
          difficulty: 'intermediate',
          activation_level: 0.75,
          prerequisites: ['step_by_step_reasoning'],
          learning_objectives: [
            '掌握多种验证方法',
            '培养自我检查意识',
            '理解合理性判断'
          ],
          practice_examples: [
            '反向验证：8 - 3 = 5，8 - 5 = 3',
            '常识检查：答案是否符合实际情况',
            '约束检查：数量是否为正整数'
          ],
          mastery_indicators: [
            '主动进行结果验证',
            '能发现并纠正错误',
            '建立质量意识'
          ]
        }
      ]
    },
    {
      id: 'advanced_reasoning',
      name: '高级推理激活路径',
      description: '通过图式思维和树式思维进行复杂推理',
      estimated_time: '4-5小时',
      difficulty_level: '高级',
      nodes: [
        {
          id: 'relationship_analysis',
          name: '关系分析',
          description: '理解和分析实体间的复杂关系',
          type: 'concept',
          difficulty: 'advanced',
          activation_level: 0.8,
          prerequisites: ['entity_recognition'],
          learning_objectives: [
            '识别多种关系类型',
            '理解关系的传递性',
            '构建关系网络'
          ],
          practice_examples: [
            '因果关系：小明买苹果 → 苹果数量增加',
            '比较关系：A比B多3个 → A = B + 3',
            '时序关系：先买5个，再买3个 → 总数 = 5 + 3'
          ],
          mastery_indicators: [
            '能识别隐含关系',
            '理解关系的数学表达',
            '构建完整的关系图'
          ]
        },
        {
          id: 'graph_reasoning',
          name: '图式推理',
          description: '使用图网络方法进行并行推理',
          type: 'strategy',
          difficulty: 'advanced',
          activation_level: 0.9,
          prerequisites: ['relationship_analysis'],
          learning_objectives: [
            '掌握网络化思维',
            '进行并行信息处理',
            '建立全局视角'
          ],
          practice_examples: [
            '同时考虑多个实体和关系',
            '从不同角度分析同一问题',
            '发现隐藏的连接模式'
          ],
          mastery_indicators: [
            '能建立问题的图模型',
            '同时处理多个信息',
            '发现系统性规律'
          ]
        },
        {
          id: 'multi_path_exploration',
          name: '多路径探索',
          description: '树式思维的多方案求解',
          type: 'strategy',
          difficulty: 'advanced',
          activation_level: 0.85,
          prerequisites: ['graph_reasoning'],
          learning_objectives: [
            '生成多种解题方案',
            '比较不同方案优劣',
            '选择最优解决路径'
          ],
          practice_examples: [
            '方案A：直接相加；方案B：分组相加',
            '路径1：从左到右；路径2：从整体到部分',
            '策略比较：速度vs准确性'
          ],
          mastery_indicators: [
            '能生成多个有效方案',
            '客观评估方案质量',
            '做出最优选择'
          ]
        }
      ]
    }
  ]

  const analyzeCurrentProblem = async () => {
    setIsAnalyzing(true)
    
    // 模拟基于当前问题的学习建议分析
    await new Promise(resolve => setTimeout(resolve, 1500))
    
    // 根据问题复杂度推荐学习路径
    const problemComplexity = Math.random() > 0.5 ? 'basic' : 'advanced'
    const recommendedPath = problemComplexity === 'basic' ? 'basic_arithmetic' : 'advanced_reasoning'
    
    setSelectedPath(recommendedPath)
    setCurrentNode(0)
    setIsAnalyzing(false)
  }

  const markNodeAsMastered = (nodeId: string) => {
    setMasteredNodes(prev => new Set([...prev, nodeId]))
    setLearningProgress(prev => ({
      ...prev,
      [nodeId]: 100
    }))
  }

  const getCurrentPath = () => {
    return learningPaths.find(path => path.id === selectedPath)
  }

  const getDifficultyColor = (difficulty: string) => {
    const colors = {
      beginner: 'bg-green-100 text-green-800',
      intermediate: 'bg-yellow-100 text-yellow-800',
      advanced: 'bg-red-100 text-red-800'
    }
    return colors[difficulty as keyof typeof colors] || colors.beginner
  }

  const getNodeTypeIcon = (type: string) => {
    const icons = {
      concept: '💡',
      strategy: '🎯',
      domain: '📚',
      skill: '🛠️'
    }
    return icons[type as keyof typeof icons] || '⚡'
  }

  // 获取API数据
  useEffect(() => {
    const fetchLearningData = async () => {
      try {
        const data = await getPersonalizedLearningPaths({
          user_level: userLevel,
          learning_goal: '数学推理能力提升',
          preferences: {}
        })
        setApiLearningData(data)
      } catch (error) {
        console.error('获取学习数据失败:', error)
      }
    }
    
    fetchLearningData()
  }, [userLevel])

  // 处理阶段完成
  const handleStageComplete = (stageId: number) => {
    setUserProgress(prev => ({ ...prev, [stageId]: 'completed' }))
  }

  // 获取状态图标
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return '✅'
      case 'in_progress': return '🔄'
      case 'available': return '🔓'
      case 'locked': return '🔒'
      default: return '⭕'
    }
  }

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* 页面标题和激活扩散学习指导 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-3">
            <span className="text-3xl">🧠</span>
            <div>
              <h1 className="text-2xl font-bold">激活扩散学习指导</h1>
              <p className="text-sm text-gray-600 mt-1">
                基于激活扩散理论，为您推荐个性化的学习路径，通过激活相关知识点，建立系统的数学思维网络。
              </p>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center space-x-4 mb-4">
            <Button 
              variant={activeTab === 'pathways' ? 'default' : 'outline'}
              onClick={() => setActiveTab('pathways')}
              className="flex items-center space-x-2"
            >
              <span>🎯</span>
              <span>智能推荐学习路径</span>
            </Button>
            <Button 
              variant={activeTab === 'stages' ? 'default' : 'outline'}
              onClick={() => setActiveTab('stages')}
              className="flex items-center space-x-2"
            >
              <span>📖</span>
              <span>学习路径（6个阶段）</span>
            </Button>
            <Button 
              variant={activeTab === 'techniques' ? 'default' : 'outline'}
              onClick={() => setActiveTab('techniques')}
              className="flex items-center space-x-2"
            >
              <span>💡</span>
              <span>学习技巧建议</span>
            </Button>
          </div>
        </CardContent>
      </Card>

      <AnimatePresence mode="wait">
        {activeTab === 'pathways' && (
          <motion.div
            key="pathways"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {/* 智能推荐学习路径 */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <span className="text-xl">🎯</span>
                  <span>智能推荐学习路径</span>
                </CardTitle>
                <p className="text-gray-600 mt-2">
                  基于激活扩散理论，为您推荐个性化的学习路径。通过激活相关知识点，建立系统的数学思维网络。
                </p>
              </CardHeader>
              <CardContent>
                <div className="mb-6">
                  <label className="block text-sm font-medium text-gray-700 mb-2">选择您的学习水平</label>
                  <div className="flex space-x-4">
                    {(['beginner', 'intermediate', 'advanced'] as const).map(level => (
                      <Button
                        key={level}
                        variant={userLevel === level ? 'default' : 'outline'}
                        onClick={() => setUserLevel(level)}
                        className="capitalize"
                      >
                        {level === 'beginner' ? '初级' : level === 'intermediate' ? '中级' : '高级'}
                      </Button>
                    ))}
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {(apiLearningData?.recommended_paths || learningPaths).map((path, index) => (
                    <motion.div
                      key={path.id}
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: index * 0.1 }}
                      className="border border-gray-200 rounded-lg p-6 hover:shadow-lg transition-all cursor-pointer bg-gradient-to-br from-white to-gray-50"
                      onClick={() => setSelectedPath(path.id)}
                    >
                      <div className="flex items-start justify-between mb-4">
                        <div className="flex items-center space-x-3">
                          <div className="text-3xl">{path.icon || '🧮'}</div>
                          <div>
                            <h3 className="font-semibold text-gray-800">{path.name || path.title}</h3>
                            <span className={`inline-block px-2 py-1 rounded-full text-xs font-medium mt-1 ${
                              getDifficultyColor(path.difficulty || path.difficulty_level)
                            }`}>
                              难度: {path.difficulty === 'beginner' ? '初级' : path.difficulty === 'intermediate' ? '中级' : path.difficulty === 'advanced' ? '高级' : path.difficulty_level}
                            </span>
                          </div>
                        </div>
                      </div>
                      
                      <p className="text-sm text-gray-600 mb-4">{path.description}</p>
                      
                      <div className="flex items-center justify-between text-sm text-gray-500">
                        <div className="flex items-center space-x-4">
                          <span>⏱️ 预计时间: {path.estimatedTime || path.estimated_time}</span>
                          <span>📚 {path.stages || (path.nodes ? path.nodes.length : 4)} 个学习节点</span>
                        </div>
                        <Button size="sm" className="bg-blue-500 hover:bg-blue-600">
                          开始学习
                        </Button>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {activeTab === 'stages' && (
          <motion.div
            key="stages"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {/* 学习路径（6个阶段） */}
            <Card>
              <CardHeader>
                <CardTitle>📖 学习路径（6个阶段）</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {(apiLearningData?.personalized_stages || [
                    { id: 1, title: '实体识别阶段', difficulty: 'beginner', estimatedTime: '30-45分钟', status: 'available' },
                    { id: 2, title: '关系理解阶段', difficulty: 'beginner', estimatedTime: '45-60分钟', status: 'locked' },
                    { id: 3, title: '策略选择阶段', difficulty: 'intermediate', estimatedTime: '60-75分钟', status: 'locked' },
                    { id: 4, title: '深度推理阶段', difficulty: 'intermediate', estimatedTime: '75-90分钟', status: 'locked' },
                    { id: 5, title: '结果验证阶段', difficulty: 'intermediate', estimatedTime: '30-45分钟', status: 'locked' },
                    { id: 6, title: '反思改进阶段', difficulty: 'advanced', estimatedTime: '45-60分钟', status: 'locked' }
                  ]).map((stage, index) => {
                    const stageStatus = userProgress[stage.id] || stage.status
                    const isLocked = stageStatus === 'locked'
                    
                    return (
                      <motion.div
                        key={stage.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className={`border rounded-lg p-4 transition-all ${
                          isLocked 
                            ? 'border-gray-200 bg-gray-50 opacity-60' 
                            : 'border-gray-200 hover:shadow-md'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex-1">
                            <div className="flex items-center gap-3 mb-2">
                              <div className={`w-10 h-10 rounded-lg flex items-center justify-center text-white text-sm font-bold ${
                                isLocked ? 'bg-gray-400' : 'bg-blue-500'
                              }`}>
                                {getStatusIcon(stageStatus)}
                              </div>
                              <div className="flex-1">
                                <div className="flex items-center space-x-3">
                                  <h3 className="font-semibold text-gray-800">
                                    第{stage.id}阶段：{stage.title}
                                  </h3>
                                  <span className={`inline-block px-2 py-1 rounded-full text-xs font-medium ${
                                    getDifficultyColor(stage.difficulty)
                                  }`}>
                                    {stage.difficulty === 'beginner' ? '初级' : stage.difficulty === 'intermediate' ? '中级' : '高级'}
                                  </span>
                                </div>
                                <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                                  <span>⏱️ {stage.estimatedTime}</span>
                                </div>
                              </div>
                            </div>
                          </div>
                          {!isLocked && (
                            <Button 
                              onClick={() => handleStageComplete(stage.id)}
                              className="bg-green-500 hover:bg-green-600"
                              disabled={stageStatus === 'completed'}
                            >
                              {stageStatus === 'completed' ? '已完成' : '标记完成'}
                            </Button>
                          )}
                        </div>
                      </motion.div>
                    )
                  })}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {activeTab === 'techniques' && (
          <motion.div
            key="techniques"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {/* 学习技巧建议 */}
            <Card>
              <CardHeader>
                <CardTitle>💡 学习技巧建议</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {(apiLearningData?.activation_based_techniques || [
                    {
                      category: '激活扩散识别技巧',
                      icon: '🔍',
                      color: 'blue',
                      techniques: [
                        '通过关键词激活相关概念网络',
                        '利用语义相似性发现隐含实体',
                        '使用激活强度判断实体重要性',
                        '通过激活路径追踪实体关系'
                      ]
                    },
                    {
                      category: '网络化关系理解方法',
                      icon: '🕸️',
                      color: 'green',
                      techniques: [
                        '构建激活扩散的关系网络',
                        '通过激活强度评估关系重要性',
                        '利用激活路径发现隐式关系',
                        '基于激活模式识别关系类型'
                      ]
                    }
                  ]).map((category, index) => (
                    <motion.div
                      key={category.category}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className={`bg-${category.color}-50 rounded-lg p-6 border border-${category.color}-200`}
                    >
                      <div className="flex items-center space-x-3 mb-4">
                        <span className="text-2xl">{category.icon}</span>
                        <h4 className={`font-semibold text-${category.color}-800`}>{category.category}</h4>
                      </div>
                      <ul className="space-y-3">
                        {category.techniques.map((technique, i) => (
                          <li key={i} className="flex items-start gap-3">
                            <span className={`text-${category.color}-500 mt-1`}>•</span>
                            <span className={`text-sm text-${category.color}-700`}>{technique}</span>
                          </li>
                        ))}
                      </ul>
                      {category.activation_methods && (
                        <div className="mt-4">
                          <h5 className={`font-medium text-${category.color}-800 mb-2`}>激活方法：</h5>
                          <ul className="space-y-2">
                            {category.activation_methods.map((method, i) => (
                              <li key={i} className="flex items-start gap-2">
                                <span className={`text-${category.color}-400 mt-1`}>→</span>
                                <span className={`text-xs text-${category.color}-600`}>{method}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </motion.div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 学习网络状态展示 */}
      {apiLearningData?.learning_network_state && (
        <Card>
          <CardHeader>
            <CardTitle>🧠 学习网络状态</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <h4 className="font-semibold text-blue-800 mb-2">当前激活概念</h4>
                <div className="flex flex-wrap gap-2">
                  {apiLearningData.learning_network_state.activated_concepts.slice(0, 5).map((concept, i) => (
                    <span key={i} className="px-2 py-1 bg-blue-200 text-blue-800 text-xs rounded">
                      {concept}
                    </span>
                  ))}
                </div>
              </div>
              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="font-semibold text-green-800 mb-2">激活强度</h4>
                <div className="w-full bg-green-200 rounded-full h-2 mb-2">
                  <div 
                    className="bg-green-600 h-2 rounded-full" 
                    style={{ width: `${apiLearningData.learning_network_state.activation_strength * 100}%` }}
                  />
                </div>
                <span className="text-green-700 text-sm">
                  {(apiLearningData.learning_network_state.activation_strength * 100).toFixed(0)}%
                </span>
              </div>
              <div className="bg-purple-50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-800 mb-2">推荐重点</h4>
                <span className="text-purple-700 text-sm">
                  {apiLearningData.learning_network_state.recommended_focus}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* 详细学习路径 */}
      <AnimatePresence>
        {selectedPath && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-6"
          >
            {/* 路径概览 */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span className="flex items-center space-x-2">
                    <MapPin className="h-5 w-5" />
                    <span>{getCurrentPath()?.name}</span>
                  </span>
                  <div className="text-sm text-gray-500">
                    {masteredNodes.size} / {getCurrentPath()?.nodes.length || 0} 已掌握
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <p className="text-gray-600">{getCurrentPath()?.description}</p>
                  
                  {/* 进度条 */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>学习进度</span>
                      <span>
                        {Math.round((masteredNodes.size / (getCurrentPath()?.nodes.length || 1)) * 100)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <motion.div
                        className="bg-blue-600 h-2 rounded-full"
                        initial={{ width: 0 }}
                        animate={{ 
                          width: `${(masteredNodes.size / (getCurrentPath()?.nodes.length || 1)) * 100}%` 
                        }}
                        transition={{ duration: 0.5 }}
                      />
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* 学习节点详情 */}
            <div className="space-y-4">
              {getCurrentPath()?.nodes.map((node, index) => (
                <motion.div
                  key={node.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <Card className={`${
                    masteredNodes.has(node.id) 
                      ? 'bg-green-50 border-green-200' 
                      : 'bg-white'
                  }`}>
                    <CardHeader>
                      <CardTitle className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <span className="text-2xl">{getNodeTypeIcon(node.type)}</span>
                          <div>
                            <h3 className="font-semibold">{node.name}</h3>
                            <p className="text-sm text-gray-600">{node.description}</p>
                          </div>
                        </div>
                        <div className="flex items-center space-x-2">
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getDifficultyColor(node.difficulty)}`}>
                            {node.difficulty}
                          </span>
                          <span className="text-xs text-gray-500">
                            激活度: {(node.activation_level * 100).toFixed(0)}%
                          </span>
                        </div>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        {/* 学习目标 */}
                        <div>
                          <h4 className="font-medium mb-2 flex items-center">
                            <Target className="h-4 w-4 mr-2" />
                            学习目标
                          </h4>
                          <ul className="text-sm space-y-1">
                            {node.learning_objectives.map((objective, i) => (
                              <li key={i} className="flex items-start">
                                <ArrowRight className="h-3 w-3 mt-0.5 mr-2 text-blue-500" />
                                {objective}
                              </li>
                            ))}
                          </ul>
                        </div>

                        {/* 练习示例 */}
                        <div>
                          <h4 className="font-medium mb-2 flex items-center">
                            <Lightbulb className="h-4 w-4 mr-2" />
                            练习示例
                          </h4>
                          <div className="text-sm space-y-2">
                            {node.practice_examples.map((example, i) => (
                              <div key={i} className="bg-gray-50 p-2 rounded font-mono text-xs">
                                {example}
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* 掌握指标 */}
                        <div>
                          <h4 className="font-medium mb-2 flex items-center">
                            <CheckCircle className="h-4 w-4 mr-2" />
                            掌握指标
                          </h4>
                          <div className="text-sm space-y-1">
                            {node.mastery_indicators.map((indicator, i) => (
                              <div key={i} className="flex items-center">
                                <div className={`w-2 h-2 rounded-full mr-2 ${
                                  masteredNodes.has(node.id) ? 'bg-green-500' : 'bg-gray-300'
                                }`} />
                                {indicator}
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* 操作按钮 */}
                        <div className="flex space-x-2 pt-2">
                          {!masteredNodes.has(node.id) ? (
                            <Button
                              size="sm"
                              onClick={() => markNodeAsMastered(node.id)}
                              className="flex items-center space-x-1"
                            >
                              <CheckCircle className="h-4 w-4" />
                              <span>标记为已掌握</span>
                            </Button>
                          ) : (
                            <div className="flex items-center space-x-2 text-green-600">
                              <CheckCircle className="h-4 w-4" />
                              <span className="text-sm font-medium">已掌握</span>
                            </div>
                          )}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>

            {/* 激活扩散图谱展示学习网络 */}
            <Card>
              <CardHeader>
                <CardTitle>🧠 学习知识激活网络</CardTitle>
              </CardHeader>
              <CardContent>
                <ActivationPropertyGraph
                  problemText="学习知识网络激活展示"
                  entities={getCurrentPath()?.nodes.map(node => ({
                    name: node.name,
                    type: node.type
                  })) || []}
                />
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default ActivationLearningGuide