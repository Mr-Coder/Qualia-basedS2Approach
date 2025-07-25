import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'

interface Strategy {
  id: string
  name: string
  description: string
  characteristics: string[]
  icon: string
  color: string
}

const strategies: Strategy[] = [
  {
    id: 'cot',
    name: 'COT - 思维链推理',
    description: '基于分析特性的深层实体识别和交互推理策略，直观性探索，适合基础推理问题',
    characteristics: [
      '线性推理：按照逻辑顺序进行步骤化推理',
      '实体分析：深入分析每个实体的特性',
      '关系建立：明确实体间的直接关系',
      '步骤验证：每个推理步骤都有明确验证'
    ],
    icon: '🔗',
    color: 'blue'
  },
  {
    id: 'got',
    name: 'GOT - 思维图推理',
    description: '基于图网络的多维度关系系统建模推理，多角度探索实体关联的综合策略',
    characteristics: [
      '网络构建：将问题构建为关系网络',
      '多维分析：从多个角度分析实体关系',
      '隐含发现：识别隐含的实体关联',
      '系统整合：整合所有关系进行推理'
    ],
    icon: '🕸️',
    color: 'green'
  },
  {
    id: 'tot',
    name: 'TOT - 思维树推理',
    description: '基于分支探索的多路径推理策略，通过树状结构探索多种解决方案和可能性',
    characteristics: [
      '分支探索：探索多种可能的推理路径',
      '路径评估：评估不同路径的可行性',
      '最优选择：选择最优的推理路径',
      '方案比较：比较不同解决方案的优劣'
    ],
    icon: '🌳',
    color: 'purple'
  }
]

export const StrategyAnalysis: React.FC = () => {
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null)

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* 页面标题 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <span className="text-2xl">🎯</span>
            <span>策略分析</span>
          </CardTitle>
          <p className="text-gray-600 mt-2">
            深入了解COT、GOT、TOT三种推理策略的核心特性 (已更新 - 简化版本)
          </p>
        </CardHeader>
      </Card>

      {/* 策略卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {strategies.map((strategy, index) => (
          <motion.div
            key={strategy.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <Card className="h-full cursor-pointer hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="flex items-center gap-3 mb-3">
                  <div className={`w-12 h-12 bg-${strategy.color}-500 rounded-lg flex items-center justify-center text-white text-2xl`}>
                    {strategy.icon}
                  </div>
                  <div>
                    <CardTitle className="text-lg">{strategy.name}</CardTitle>
                  </div>
                </div>
                <p className="text-sm text-gray-600 leading-relaxed">
                  {strategy.description}
                </p>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div>
                    <h4 className="font-medium text-gray-800 mb-2">核心特性</h4>
                    <ul className="text-sm text-gray-600 space-y-1">
                      {strategy.characteristics.map((char, i) => (
                        <li key={i} className="flex items-start gap-2">
                          <span className={`text-${strategy.color}-500 mt-1`}>•</span>
                          <span>{char}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                  
                  <Button 
                    variant="outline" 
                    size="sm" 
                    className="w-full mt-4"
                    onClick={() => setSelectedStrategy(
                      selectedStrategy === strategy.id ? null : strategy.id
                    )}
                  >
                    查看详情
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* 策略详细信息（如果有选中的策略） */}
      {selectedStrategy && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <Card>
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className={`w-10 h-10 bg-${strategies.find(s => s.id === selectedStrategy)!.color}-500 rounded-lg flex items-center justify-center text-white text-xl`}>
                  {strategies.find(s => s.id === selectedStrategy)!.icon}
                </div>
                <CardTitle>{strategies.find(s => s.id === selectedStrategy)!.name} 详细分析</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-gray-600">
                <p>详细的策略分析内容将在这里显示...</p>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}
    </div>
  )
}

export default StrategyAnalysis