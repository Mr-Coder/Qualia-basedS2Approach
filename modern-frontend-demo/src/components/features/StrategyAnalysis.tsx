import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'

interface Strategy {
  id: string
  name: string
  description: string
  characteristics: string[]
  advantages: string[]
  disadvantages: string[]
  bestFor: string[]
  examples: string[]
  color: string
  icon: string
}

const strategies: Strategy[] = [
  {
    id: 'cot',
    name: 'COT - 思维链推理',
    description: '基于链式推理的深层实体物性关系建模',
    characteristics: [
      '链式分解：将复杂问题分解为有序的推理链条',
      '实体物性分析：深入分析每个实体的物理属性和能力',
      '状态转移跟踪：追踪实体在推理过程中的状态变化',
      '多层验证：从数值、逻辑、物理三个层面验证结果'
    ],
    advantages: [
      '推理过程清晰可见',
      '适合顺序推理问题',
      '容易理解和验证',
      '适合教学演示'
    ],
    disadvantages: [
      '无法处理复杂关系网络',
      '可能错过并行推理路径',
      '对复杂问题效率较低'
    ],
    bestFor: [
      '算术问题',
      '简单几何问题',
      '线性推理问题',
      '教学场景'
    ],
    examples: [
      '小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？',
      '一个长方形长10cm，宽5cm，求面积',
      '从A地到B地，速度60km/h，需要2小时，求距离'
    ],
    color: 'bg-blue-500',
    icon: '🔗'
  },
  {
    id: 'got',
    name: 'GOT - 思维图推理',
    description: '基于图网络的多维实体关系建模',
    characteristics: [
      '网络拓扑构建：将实体和关系构建为多层网络结构',
      '隐含边发现：识别实体间的隐含连接和依赖关系',
      '流动分析：分析实体、信息、能量在网络中的流动',
      '子图推理：通过子网络推理局部和全局性质'
    ],
    advantages: [
      '能处理复杂关系网络',
      '发现隐含关系',
      '适合多实体问题',
      '推理结果更全面'
    ],
    disadvantages: [
      '计算复杂度高',
      '需要更多推理资源',
      '结果可能过于复杂'
    ],
    bestFor: [
      '复杂应用题',
      '多实体关系问题',
      '网络分析问题',
      '系统性问题'
    ],
    examples: [
      '班级有45个学生，男生占60%，参加数学竞赛的占40%，求各种组合',
      '工厂生产线问题，涉及多个工序和资源',
      '复杂的购物问题，涉及多种商品和优惠'
    ],
    color: 'bg-green-500',
    icon: '🕸️'
  },
  {
    id: 'tot',
    name: 'TOT - 思维树推理',
    description: '基于分层树结构的多路径实体关系探索',
    characteristics: [
      '层次树构建：建立分层的实体关系分类树',
      '多路径探索：探索多种解决方案和推理路径',
      '约束传播：在树结构中传播隐含约束',
      '最优选择：基于实体关系复杂度选择最优路径'
    ],
    advantages: [
      '探索多种解决方案',
      '找到最优路径',
      '适合开放性问题',
      '思维发散能力强'
    ],
    disadvantages: [
      '计算量大',
      '可能过度复杂化',
      '需要评估能力'
    ],
    bestFor: [
      '开放性问题',
      '创新思维问题',
      '多方案比较',
      '决策支持'
    ],
    examples: [
      '设计一个学校运动会的安排方案',
      '如何分配有限资源解决多个问题',
      '探索数学问题的多种解法'
    ],
    color: 'bg-purple-500',
    icon: '🌳'
  }
]

export const StrategyAnalysis: React.FC = () => {
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null)
  const [comparisonMode, setComparisonMode] = useState(false)

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* 页面标题 */}
      <Card>
        <CardHeader>
          <CardTitle>🎯 策略分析</CardTitle>
          <p className="text-gray-600">
            深入了解COT、GOT、TOT三种推理策略的特点、优势和适用场景
          </p>
        </CardHeader>
      </Card>

      {/* 策略概览 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {strategies.map((strategy, index) => (
          <motion.div
            key={strategy.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className={`relative p-6 rounded-lg border-2 cursor-pointer transition-all ${
              selectedStrategy === strategy.id
                ? 'border-purple-500 bg-purple-50 shadow-lg'
                : 'border-gray-200 hover:border-gray-300 hover:shadow-md'
            }`}
            onClick={() => setSelectedStrategy(
              selectedStrategy === strategy.id ? null : strategy.id
            )}
          >
            <div className="flex items-center gap-4 mb-4">
              <div className={`w-12 h-12 ${strategy.color} rounded-lg flex items-center justify-center text-white text-2xl`}>
                {strategy.icon}
              </div>
              <div>
                <h3 className="text-lg font-semibold text-gray-800">{strategy.name}</h3>
                <p className="text-sm text-gray-600">{strategy.description}</p>
              </div>
            </div>
            
            <div className="space-y-3">
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-1">核心特征</h4>
                <ul className="text-xs text-gray-600 space-y-1">
                  {strategy.characteristics.slice(0, 2).map((char, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="text-purple-500 mt-1">•</span>
                      <span>{char}</span>
                    </li>
                  ))}
                </ul>
              </div>
              
              <Button 
                variant="outline" 
                size="sm" 
                className="w-full"
                onClick={(e) => {
                  e.stopPropagation()
                  setSelectedStrategy(
                    selectedStrategy === strategy.id ? null : strategy.id
                  )
                }}
              >
                {selectedStrategy === strategy.id ? '收起详情' : '查看详情'}
              </Button>
            </div>
          </motion.div>
        ))}
      </div>

      {/* 策略详细分析 */}
      {selectedStrategy && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <Card>
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className={`w-10 h-10 ${strategies.find(s => s.id === selectedStrategy)!.color} rounded-lg flex items-center justify-center text-white text-xl`}>
                  {strategies.find(s => s.id === selectedStrategy)!.icon}
                </div>
                <CardTitle>{strategies.find(s => s.id === selectedStrategy)!.name} 详细分析</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              {(() => {
                const strategy = strategies.find(s => s.id === selectedStrategy)!
                return (
                  <div className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <h4 className="font-semibold text-gray-800 mb-3">✅ 优势</h4>
                        <ul className="space-y-2">
                          {strategy.advantages.map((advantage, i) => (
                            <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                              <span className="text-green-500 mt-1">✓</span>
                              <span>{advantage}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                      
                      <div>
                        <h4 className="font-semibold text-gray-800 mb-3">❌ 劣势</h4>
                        <ul className="space-y-2">
                          {strategy.disadvantages.map((disadvantage, i) => (
                            <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                              <span className="text-red-500 mt-1">✗</span>
                              <span>{disadvantage}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="font-semibold text-gray-800 mb-3">🎯 适用场景</h4>
                      <div className="flex flex-wrap gap-2">
                        {strategy.bestFor.map((scenario, i) => (
                          <span key={i} className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm">
                            {scenario}
                          </span>
                        ))}
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="font-semibold text-gray-800 mb-3">📝 典型例题</h4>
                      <div className="space-y-3">
                        {strategy.examples.map((example, i) => (
                          <div key={i} className="p-4 bg-gray-50 rounded-lg border">
                            <p className="text-sm text-gray-700">{example}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )
              })()}
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* 策略对比分析 */}
      <Card>
        <CardHeader>
          <CardTitle>📊 策略对比分析</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b">
                  <th className="text-left p-3 font-semibold">对比维度</th>
                  <th className="text-center p-3 font-semibold text-blue-600">COT</th>
                  <th className="text-center p-3 font-semibold text-green-600">GOT</th>
                  <th className="text-center p-3 font-semibold text-purple-600">TOT</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b">
                  <td className="p-3 font-medium">复杂度处理</td>
                  <td className="p-3 text-center">⭐⭐</td>
                  <td className="p-3 text-center">⭐⭐⭐⭐</td>
                  <td className="p-3 text-center">⭐⭐⭐⭐⭐</td>
                </tr>
                <tr className="border-b">
                  <td className="p-3 font-medium">推理效率</td>
                  <td className="p-3 text-center">⭐⭐⭐⭐⭐</td>
                  <td className="p-3 text-center">⭐⭐⭐</td>
                  <td className="p-3 text-center">⭐⭐</td>
                </tr>
                <tr className="border-b">
                  <td className="p-3 font-medium">结果准确性</td>
                  <td className="p-3 text-center">⭐⭐⭐⭐</td>
                  <td className="p-3 text-center">⭐⭐⭐⭐⭐</td>
                  <td className="p-3 text-center">⭐⭐⭐⭐</td>
                </tr>
                <tr className="border-b">
                  <td className="p-3 font-medium">可解释性</td>
                  <td className="p-3 text-center">⭐⭐⭐⭐⭐</td>
                  <td className="p-3 text-center">⭐⭐⭐</td>
                  <td className="p-3 text-center">⭐⭐</td>
                </tr>
                <tr>
                  <td className="p-3 font-medium">创新性</td>
                  <td className="p-3 text-center">⭐⭐</td>
                  <td className="p-3 text-center">⭐⭐⭐⭐</td>
                  <td className="p-3 text-center">⭐⭐⭐⭐⭐</td>
                </tr>
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* 策略选择建议 */}
      <Card>
        <CardHeader>
          <CardTitle>🎯 策略选择建议</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h4 className="font-semibold text-blue-800 mb-2">🔗 选择COT的情况</h4>
              <ul className="text-sm text-blue-700 space-y-1">
                <li>• 问题结构简单清晰</li>
                <li>• 需要详细的推理过程</li>
                <li>• 教学演示场景</li>
                <li>• 注重可解释性</li>
              </ul>
            </div>
            
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <h4 className="font-semibold text-green-800 mb-2">🕸️ 选择GOT的情况</h4>
              <ul className="text-sm text-green-700 space-y-1">
                <li>• 多实体复杂关系</li>
                <li>• 需要发现隐含关系</li>
                <li>• 系统性分析问题</li>
                <li>• 网络结构问题</li>
              </ul>
            </div>
            
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
              <h4 className="font-semibold text-purple-800 mb-2">🌳 选择TOT的情况</h4>
              <ul className="text-sm text-purple-700 space-y-1">
                <li>• 开放性问题</li>
                <li>• 需要创新解决方案</li>
                <li>• 多方案比较选择</li>
                <li>• 决策支持场景</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default StrategyAnalysis