import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'

interface ProblemType {
  id: string
  name: string
  description: string
  characteristics: string[]
  examples: string[]
  strategies: string[]
  color: string
  icon: string
}

interface LearningStage {
  id: number
  title: string
  description: string
  skills: string[]
  practices: string[]
  tips: string[]
}

const problemTypes: ProblemType[] = [
  {
    id: 'arithmetic',
    name: '算术问题',
    description: '涉及基本数学运算的问题，注重数量关系和运算规则',
    characteristics: [
      '数量关系明确',
      '运算规则清晰',
      '实体聚合关系',
      '步骤相对简单'
    ],
    examples: [
      '小明有10个苹果，吃了3个，还剩多少？',
      '商店有45本书，卖出18本，又进了23本',
      '班级有男生15人，女生12人，总共多少人？'
    ],
    strategies: ['COT推理', '逐步分解', '状态跟踪'],
    color: 'bg-blue-500',
    icon: '🔢'
  },
  {
    id: 'geometry',
    name: '几何问题',
    description: '涉及图形和空间关系的问题，注重空间思维和公式应用',
    characteristics: [
      '空间关系分析',
      '公式应用为主',
      '维度转换思维',
      '图形可视化'
    ],
    examples: [
      '长方形长12cm，宽8cm，求面积',
      '圆形半径5cm，求周长和面积',
      '正方形边长6cm，求对角线长度'
    ],
    strategies: ['GOT推理', '空间建模', '关系网络'],
    color: 'bg-green-500',
    icon: '📐'
  },
  {
    id: 'application',
    name: '应用题',
    description: '结合实际情境的问题，注重现实映射和逻辑推理',
    characteristics: [
      '现实情境映射',
      '多约束条件',
      '逻辑推理复杂',
      '实体关系丰富'
    ],
    examples: [
      '小王去超市买东西，苹果5元/斤，买了3斤...',
      '从甲地到乙地，汽车速度60km/h，需要多长时间？',
      '工厂每天生产零件200个，5天能生产多少？'
    ],
    strategies: ['GOT推理', 'TOT推理', '多路径探索'],
    color: 'bg-purple-500',
    icon: '🌍'
  },
  {
    id: 'percentage',
    name: '百分比问题',
    description: '涉及比例和百分比计算的问题，注重比例关系和转换',
    characteristics: [
      '比例关系计算',
      '整体部分关系',
      '转换思维要求',
      '实际应用广泛'
    ],
    examples: [
      '某商品原价100元，打8折后价格是多少？',
      '学校有学生500人，男生占60%，女生多少人？',
      '存款1000元，年利率5%，一年后多少钱？'
    ],
    strategies: ['COT推理', '比例分析', '转换计算'],
    color: 'bg-orange-500',
    icon: '📊'
  }
]

const learningStages: LearningStage[] = [
  {
    id: 1,
    title: '实体识别阶段',
    description: '学会识别问题中的关键实体和对象，建立基本的问题理解框架',
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

const learningTips = [
  {
    category: '实体识别技巧',
    tips: [
      '仔细阅读问题，标记关键信息',
      '区分数量词和描述词',
      '注意时间和空间的表述',
      '识别隐含的实体和条件'
    ]
  },
  {
    category: '关系理解方法',
    tips: [
      '用图形化方式表示关系',
      '注意关系的方向性',
      '识别因果关系和并列关系',
      '检查关系的完整性和一致性'
    ]
  },
  {
    category: '策略选择指导',
    tips: [
      '根据问题复杂度选择策略',
      '考虑自己的能力水平',
      '可以尝试多种策略组合',
      '记录策略使用的效果'
    ]
  },
  {
    category: '练习建议',
    tips: [
      '从简单问题开始练习',
      '逐步增加问题难度',
      '定期回顾和总结',
      '与同学交流解题经验'
    ]
  }
]

export const LearningGuide: React.FC = () => {
  const [selectedProblemType, setSelectedProblemType] = useState<string | null>(null)
  const [selectedStage, setSelectedStage] = useState<number | null>(null)

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* 页面标题 */}
      <Card>
        <CardHeader>
          <CardTitle>📚 学习指导</CardTitle>
          <p className="text-gray-600">
            系统化的学习路径和实用技巧，帮助您掌握数学问题的智能推理方法
          </p>
        </CardHeader>
      </Card>

      {/* 问题类型指导 */}
      <Card>
        <CardHeader>
          <CardTitle>🎯 问题类型指导</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {problemTypes.map((type, index) => (
              <motion.div
                key={type.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                  selectedProblemType === type.id 
                    ? 'border-purple-500 bg-purple-50' 
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => setSelectedProblemType(
                  selectedProblemType === type.id ? null : type.id
                )}
              >
                <div className="flex items-center gap-3 mb-3">
                  <div className={`w-10 h-10 ${type.color} rounded-lg flex items-center justify-center text-white text-xl`}>
                    {type.icon}
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-800">{type.name}</h3>
                    <p className="text-sm text-gray-600">{type.description}</p>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-1">特征</h4>
                    <div className="flex flex-wrap gap-1">
                      {type.characteristics.slice(0, 2).map((char, i) => (
                        <span key={i} className="text-xs px-2 py-1 bg-gray-100 rounded">
                          {char}
                        </span>
                      ))}
                    </div>
                  </div>
                  
                  <Button 
                    variant="outline" 
                    size="sm" 
                    className="w-full"
                    onClick={(e) => {
                      e.stopPropagation()
                      setSelectedProblemType(
                        selectedProblemType === type.id ? null : type.id
                      )
                    }}
                  >
                    {selectedProblemType === type.id ? '收起' : '展开'}
                  </Button>
                </div>
              </motion.div>
            ))}
          </div>
          
          {/* 问题类型详细信息 */}
          {selectedProblemType && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
              className="mt-6 p-4 bg-gray-50 rounded-lg"
            >
              {(() => {
                const type = problemTypes.find(t => t.id === selectedProblemType)!
                return (
                  <div className="space-y-4">
                    <div className="flex items-center gap-3">
                      <div className={`w-8 h-8 ${type.color} rounded-lg flex items-center justify-center text-white`}>
                        {type.icon}
                      </div>
                      <h3 className="text-lg font-semibold">{type.name} 详细指导</h3>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <h4 className="font-medium text-gray-800 mb-2">📋 问题特征</h4>
                        <ul className="text-sm text-gray-600 space-y-1">
                          {type.characteristics.map((char, i) => (
                            <li key={i} className="flex items-start gap-2">
                              <span className="text-purple-500 mt-1">•</span>
                              <span>{char}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                      
                      <div>
                        <h4 className="font-medium text-gray-800 mb-2">🛠️ 推荐策略</h4>
                        <ul className="text-sm text-gray-600 space-y-1">
                          {type.strategies.map((strategy, i) => (
                            <li key={i} className="flex items-start gap-2">
                              <span className="text-green-500 mt-1">•</span>
                              <span>{strategy}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="font-medium text-gray-800 mb-2">📝 典型例题</h4>
                      <div className="space-y-2">
                        {type.examples.map((example, i) => (
                          <div key={i} className="p-3 bg-white rounded border text-sm">
                            {example}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )
              })()}
            </motion.div>
          )}
        </CardContent>
      </Card>

      {/* 学习路径 */}
      <Card>
        <CardHeader>
          <CardTitle>🛤️ 学习路径（6个阶段）</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {learningStages.map((stage, index) => (
              <motion.div
                key={stage.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className={`border-l-4 pl-4 py-3 cursor-pointer transition-all ${
                  selectedStage === stage.id 
                    ? 'border-purple-500 bg-purple-50' 
                    : 'border-gray-300 hover:border-gray-400'
                }`}
                onClick={() => setSelectedStage(
                  selectedStage === stage.id ? null : stage.id
                )}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-semibold text-gray-800">
                      第{stage.id}阶段：{stage.title}
                    </h3>
                    <p className="text-sm text-gray-600 mt-1">{stage.description}</p>
                  </div>
                  <Button variant="ghost" size="sm">
                    {selectedStage === stage.id ? '收起' : '展开'}
                  </Button>
                </div>
                
                {selectedStage === stage.id && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    transition={{ duration: 0.3 }}
                    className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4"
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
                  </motion.div>
                )}
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* 学习技巧建议 */}
      <Card>
        <CardHeader>
          <CardTitle>💡 学习技巧建议</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {learningTips.map((category, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="bg-gray-50 rounded-lg p-4"
              >
                <h3 className="font-semibold text-gray-800 mb-3">{category.category}</h3>
                <ul className="space-y-2">
                  {category.tips.map((tip, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                      <span className="text-purple-500 mt-1">▸</span>
                      <span>{tip}</span>
                    </li>
                  ))}
                </ul>
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* 学习建议总结 */}
      <Card>
        <CardHeader>
          <CardTitle>📈 学习建议总结</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg p-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold text-gray-800 mb-3">🎯 学习要点</h3>
                <ul className="space-y-2 text-sm text-gray-700">
                  <li className="flex items-start gap-2">
                    <span className="text-green-500 mt-1">✓</span>
                    <span>循序渐进，从简单问题开始</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-green-500 mt-1">✓</span>
                    <span>重视实体识别和关系分析</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-green-500 mt-1">✓</span>
                    <span>选择合适的推理策略</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-green-500 mt-1">✓</span>
                    <span>坚持练习和反思改进</span>
                  </li>
                </ul>
              </div>
              
              <div>
                <h3 className="font-semibold text-gray-800 mb-3">⚠️ 注意事项</h3>
                <ul className="space-y-2 text-sm text-gray-700">
                  <li className="flex items-start gap-2">
                    <span className="text-orange-500 mt-1">!</span>
                    <span>不要急于求解，先理解问题</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-orange-500 mt-1">!</span>
                    <span>避免跳步，保持推理完整性</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-orange-500 mt-1">!</span>
                    <span>重视验证环节，确保答案正确</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-orange-500 mt-1">!</span>
                    <span>错题是宝贵的学习资源</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default LearningGuide