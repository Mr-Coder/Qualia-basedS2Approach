import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'

interface LearningStage {
  id: number
  title: string
  description: string
  skills: string[]
  practices: string[]
  tips: string[]
  estimatedTime: string
  difficulty: 'beginner' | 'intermediate' | 'advanced'
  prerequisites?: number[]
  status: 'locked' | 'available' | 'in_progress' | 'completed'
}

interface LearningPath {
  id: string
  title: string
  description: string
  estimatedTime: string
  difficulty: 'beginner' | 'advanced'
  stages: number
  icon: string
}

interface SkillTechnique {
  category: string
  icon: string
  color: string
  techniques: string[]
}

const learningPaths: LearningPath[] = [
  {
    id: 'basic_activation',
    title: '基础算术激活路径',
    description: '通过激活扩散理论学习基础数学运算',
    estimatedTime: '2-3小时',
    difficulty: 'beginner',
    stages: 4,
    icon: '🧮'
  },
  {
    id: 'advanced_reasoning',
    title: '高级推理激活路径',
    description: '通过函数式思维和网络思维进行复杂推理',
    estimatedTime: '4-5小时',
    difficulty: 'advanced',
    stages: 3,
    icon: '🧠'
  }
]

const skillTechniques: SkillTechnique[] = [
  {
    category: '实体识别技巧',
    icon: '🔍',
    color: 'blue',
    techniques: [
      '仔细阅读问题，标记关键信息',
      '区分数量词和描述词',
      '注意时间和空间的表述',
      '识别隐含的实体和条件'
    ]
  },
  {
    category: '关系理解方法',
    icon: '🔗',
    color: 'green',
    techniques: [
      '用图形化方式表示关系',
      '注意关系的方向性',
      '识别因果关系和并列关系',
      '检查关系的完整性和一致性'
    ]
  }
]

const learningStages: LearningStage[] = [
  {
    id: 1,
    title: '实体识别阶段',
    description: '学会识别问题中的关键实体和对象，建立基本的问题理解框架',
    estimatedTime: '30-45分钟',
    difficulty: 'beginner',
    status: 'available',
    skills: [
      '识别问题中的人物、物品、数量',
      '区分已知条件和未知条件',
      '理解实体的属性和特征',
      '建立实体的基本分类'
    ],
    practices: [
      '阅读问题，圈出关键词',
      '列出所有实体和数量',
      '分类整理已知和未知',
      '用表格整理实体信息'
    ],
    tips: [
      '多读几遍问题，不要急于求解',
      '用不同颜色标记不同类型的实体',
      '注意隐含的实体和条件'
    ]
  },
  {
    id: 2,
    title: '关系理解阶段',
    description: '深入理解实体间的关系，掌握各种数学关系的表达方式',
    estimatedTime: '45-60分钟',
    difficulty: 'beginner',
    prerequisites: [1],
    status: 'locked',
    skills: [
      '识别实体间的数量关系',
      '理解时间、空间关系',
      '掌握因果关系分析',
      '建立关系网络图'
    ],
    practices: [
      '画关系图连接实体',
      '用箭头表示关系方向',
      '标注关系的类型和强度',
      '检查关系的完整性'
    ],
    tips: [
      '关系比实体更重要',
      '注意隐含的关系',
      '用图形化方式表示关系'
    ]
  },
  {
    id: 3,
    title: '策略选择阶段',
    description: '根据问题特点选择合适的推理策略，提高解题效率',
    estimatedTime: '60-75分钟',
    difficulty: 'intermediate',
    prerequisites: [2],
    status: 'locked',
    skills: [
      '判断问题的复杂度',
      '识别问题的类型特征',
      '选择合适的推理策略',
      '评估策略的适用性'
    ],
    practices: [
      '分析问题的结构特点',
      '比较不同策略的优缺点',
      '根据自己的能力选择策略',
      '记录策略使用的效果'
    ],
    tips: [
      '简单问题用COT，复杂问题用GOT',
      '开放性问题考虑TOT',
      '可以尝试多种策略组合'
    ]
  },
  {
    id: 4,
    title: '深度推理阶段',
    description: '运用选定的策略进行深入推理，逐步解决问题',
    estimatedTime: '75-90分钟',
    difficulty: 'intermediate',
    prerequisites: [3],
    status: 'locked',
    skills: [
      '按策略要求进行推理',
      '保持推理逻辑的连贯性',
      '处理推理中的分支情况',
      '监控推理过程的正确性'
    ],
    practices: [
      '记录每一步推理过程',
      '检查推理的逻辑性',
      '处理推理中的障碍',
      '保持推理的系统性'
    ],
    tips: [
      '不要跳步，保持完整性',
      '遇到困难时回到上一步',
      '用多种方法验证推理结果'
    ]
  },
  {
    id: 5,
    title: '结果验证阶段',
    description: '对推理结果进行验证，确保答案的正确性和合理性',
    estimatedTime: '30-45分钟',
    difficulty: 'intermediate',
    prerequisites: [4],
    status: 'locked',
    skills: [
      '检查计算的准确性',
      '验证结果的合理性',
      '确认答案符合题意',
      '评估解题过程的有效性'
    ],
    practices: [
      '重新代入原问题检验',
      '用不同方法验证结果',
      '检查单位和量纲',
      '评估结果的现实意义'
    ],
    tips: [
      '验证是解题的必要环节',
      '不合理的结果要重新思考',
      '养成验证的良好习惯'
    ]
  },
  {
    id: 6,
    title: '反思改进阶段',
    description: '反思解题过程，总结经验教训，持续改进解题能力',
    estimatedTime: '45-60分钟',
    difficulty: 'advanced',
    prerequisites: [5],
    status: 'locked',
    skills: [
      '分析解题过程的优缺点',
      '总结解题的关键步骤',
      '识别常见的错误模式',
      '制定改进计划'
    ],
    practices: [
      '写解题心得体会',
      '整理错题和难题',
      '分享解题经验',
      '制定练习计划'
    ],
    tips: [
      '每道题都要有收获',
      '错题是最好的老师',
      '持续改进解题方法'
    ]
  }
]

export const LearningGuide: React.FC = () => {
  const [selectedStage, setSelectedStage] = useState<number | null>(null)
  const [activeTab, setActiveTab] = useState<'pathways' | 'stages' | 'techniques'>('pathways')
  const [userProgress, setUserProgress] = useState<{[key: number]: 'completed' | 'in_progress' | 'available' | 'locked'}>({1: 'available'})

  // Update stage status based on user progress
  useEffect(() => {
    const updatedStages = learningStages.map(stage => {
      if (userProgress[stage.id]) {
        return { ...stage, status: userProgress[stage.id] }
      }
      if (stage.prerequisites && stage.prerequisites.every(prereq => userProgress[prereq] === 'completed')) {
        return { ...stage, status: 'available' as const }
      }
      return stage
    })
  }, [userProgress])

  const handleStageComplete = (stageId: number) => {
    setUserProgress(prev => ({ ...prev, [stageId]: 'completed' }))
  }

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'bg-green-100 text-green-800'
      case 'intermediate': return 'bg-yellow-100 text-yellow-800'
      case 'advanced': return 'bg-red-100 text-red-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

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
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {learningPaths.map((path, index) => (
                    <motion.div
                      key={path.id}
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: index * 0.1 }}
                      className="border border-gray-200 rounded-lg p-6 hover:shadow-lg transition-all cursor-pointer bg-gradient-to-br from-white to-gray-50"
                    >
                      <div className="flex items-start justify-between mb-4">
                        <div className="flex items-center space-x-3">
                          <div className="text-3xl">{path.icon}</div>
                          <div>
                            <h3 className="font-semibold text-gray-800">{path.title}</h3>
                            <span className={`inline-block px-2 py-1 rounded-full text-xs font-medium mt-1 ${
                              getDifficultyColor(path.difficulty)
                            }`}>
                              难度: {path.difficulty === 'beginner' ? '初级' : '高级'}
                            </span>
                          </div>
                        </div>
                      </div>
                      
                      <p className="text-sm text-gray-600 mb-4">{path.description}</p>
                      
                      <div className="flex items-center justify-between text-sm text-gray-500">
                        <div className="flex items-center space-x-4">
                          <span>⏱️ 预计时间: {path.estimatedTime}</span>
                          <span>📚 {path.stages} 个学习节点</span>
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
                  {learningStages.map((stage, index) => {
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
                            : `border-gray-200 cursor-pointer hover:shadow-md ${
                                selectedStage === stage.id ? 'border-blue-500 bg-blue-50' : ''
                              }`
                        }`}
                        onClick={() => !isLocked && setSelectedStage(
                          selectedStage === stage.id ? null : stage.id
                        )}
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
                                <p className="text-sm text-gray-600 mt-1">{stage.description}</p>
                                <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                                  <span>⏱️ {stage.estimatedTime}</span>
                                  {stage.prerequisites && (
                                    <span>📋 需完成阶段: {stage.prerequisites.join(', ')}</span>
                                  )}
                                </div>
                              </div>
                            </div>
                          </div>
                          {!isLocked && (
                            <Button variant="ghost" size="sm" className="text-blue-600">
                              {selectedStage === stage.id ? '收起' : '展开'}
                            </Button>
                          )}
                        </div>
                        
                        {selectedStage === stage.id && !isLocked && (
                          <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            transition={{ duration: 0.3 }}
                            className="mt-4 ml-13 grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t border-gray-200"
                          >
                            <div>
                              <h4 className="font-medium text-gray-800 mb-2">🎯 核心技能</h4>
                              <ul className="text-sm text-gray-600 space-y-1">
                                {stage.skills.map((skill, i) => (
                                  <li key={i} className="flex items-start gap-2">
                                    <span className="text-blue-500 mt-1">•</span>
                                    <span>{skill}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                            
                            <div>
                              <h4 className="font-medium text-gray-800 mb-2">📝 练习方法</h4>
                              <ul className="text-sm text-gray-600 space-y-1">
                                {stage.practices.map((practice, i) => (
                                  <li key={i} className="flex items-start gap-2">
                                    <span className="text-green-500 mt-1">•</span>
                                    <span>{practice}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                            
                            <div>
                              <h4 className="font-medium text-gray-800 mb-2">💡 学习提示</h4>
                              <ul className="text-sm text-gray-600 space-y-1">
                                {stage.tips.map((tip, i) => (
                                  <li key={i} className="flex items-start gap-2">
                                    <span className="text-orange-500 mt-1">•</span>
                                    <span>{tip}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                            
                            <div className="md:col-span-3 pt-4 border-t border-gray-200">
                              <div className="flex justify-between items-center">
                                <div className="text-sm text-gray-600">
                                  完成此阶段后将解锁后续学习内容
                                </div>
                                <Button 
                                  onClick={() => handleStageComplete(stage.id)}
                                  className="bg-green-500 hover:bg-green-600"
                                  disabled={stageStatus === 'completed'}
                                >
                                  {stageStatus === 'completed' ? '已完成' : '标记完成'}
                                </Button>
                              </div>
                            </div>
                          </motion.div>
                        )}
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
                  {skillTechniques.map((category, index) => (
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
                    </motion.div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

    </div>
  )
}

export default LearningGuide