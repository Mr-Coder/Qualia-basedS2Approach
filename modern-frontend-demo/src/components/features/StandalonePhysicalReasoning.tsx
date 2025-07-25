import React, { useState, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { ArrowRight, ChevronRight, ChevronLeft, Zap, Network, Target, Lightbulb, AlertCircle, BookOpen } from 'lucide-react'

interface Entity {
  id: string
  name: string
  type: 'person' | 'object' | 'quantity' | 'concept'
  value?: number
  unit?: string
}

interface Relation {
  id: string
  source: string
  target: string
  type: 'has_quantity' | 'transformation' | 'conservation' | 'causal' | 'temporal'
  label: string
}

interface StepData {
  step: number
  title: string
  description: string
  entities: Entity[]
  relations: Relation[]
  highlight?: {
    entities?: string[]
    relations?: string[]
  }
  explanation: string
}

const StandalonePhysicalReasoning: React.FC = () => {
  const [selectedExample, setSelectedExample] = useState<number>(0)
  const [currentStep, setCurrentStep] = useState<number>(0)
  const [stepData, setStepData] = useState<StepData[]>([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [showAlgorithmExplanation, setShowAlgorithmExplanation] = useState(true)

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

  // 生成步骤数据的纯函数，不依赖任何API
  const generateStepData = useCallback((problemIndex: number): StepData[] => {
    const problem = examples[problemIndex]
    
    if (problemIndex === 0) { // 苹果问题
      return [
        {
          step: 1,
          title: "分解题目 → 提取entity",
          description: "识别题目中的人物、物体和数量",
          entities: [
            { id: 'xiaoming', name: '小明', type: 'person' },
            { id: 'apples', name: '苹果', type: 'object' },
            { id: 'initial_5', name: '5个', type: 'quantity', value: 5, unit: '个' },
            { id: 'given_3', name: '3个', type: 'quantity', value: 3, unit: '个' }
          ],
          relations: [
            { id: 'r1', source: 'xiaoming', target: 'apples', type: 'has_quantity', label: '拥有' }
          ],
          highlight: {
            entities: ['xiaoming', 'apples', 'initial_5', 'given_3']
          },
          explanation: `从题目中提取出：
• 人物：小明
• 物体：苹果
• 数量：5个(初始)、3个(新增)`
        },
        {
          step: 2,
          title: "加入物性关系 → 构建graph",
          description: "添加运算概念和转换关系",
          entities: [
            { id: 'xiaoming', name: '小明', type: 'person' },
            { id: 'apples', name: '苹果', type: 'object' },
            { id: 'initial_5', name: '5个', type: 'quantity', value: 5, unit: '个' },
            { id: 'given_3', name: '3个', type: 'quantity', value: 3, unit: '个' },
            { id: 'addition', name: '加法', type: 'concept' },
            { id: 'total', name: '总数', type: 'concept' }
          ],
          relations: [
            { id: 'r1', source: 'xiaoming', target: 'apples', type: 'has_quantity', label: '拥有' },
            { id: 'r2', source: 'initial_5', target: 'addition', type: 'transformation', label: '输入' },
            { id: 'r3', source: 'given_3', target: 'addition', type: 'transformation', label: '输入' },
            { id: 'r4', source: 'addition', target: 'total', type: 'transformation', label: '产生' }
          ],
          highlight: {
            entities: ['addition', 'total'],
            relations: ['r2', 'r3', 'r4']
          },
          explanation: `加入数学概念和物性关系：
• 运算概念：加法、总数
• 建立转换关系，连接已知量和运算`
        },
        {
          step: 3,
          title: "物性关系推理 → 扩展连接",
          description: "通过守恒、因果等物理规律扩展关系",
          entities: [
            { id: 'xiaoming', name: '小明', type: 'person' },
            { id: 'apples', name: '苹果', type: 'object' },
            { id: 'initial_5', name: '5个', type: 'quantity', value: 5, unit: '个' },
            { id: 'given_3', name: '3个', type: 'quantity', value: 3, unit: '个' },
            { id: 'addition', name: '加法', type: 'concept' },
            { id: 'total', name: '总数', type: 'concept' }
          ],
          relations: [
            { id: 'r1', source: 'xiaoming', target: 'apples', type: 'has_quantity', label: '拥有' },
            { id: 'r2', source: 'initial_5', target: 'addition', type: 'transformation', label: '输入' },
            { id: 'r3', source: 'given_3', target: 'addition', type: 'transformation', label: '输入' },
            { id: 'r4', source: 'addition', target: 'total', type: 'transformation', label: '产生' },
            { id: 'r5', source: 'initial_5', target: 'total', type: 'conservation', label: '守恒' },
            { id: 'r6', source: 'given_3', target: 'total', type: 'conservation', label: '守恒' }
          ],
          highlight: {
            relations: ['r5', 'r6']
          },
          explanation: `通过物性关系扩展：
• 守恒关系：部分之和等于整体
• 5个 + 3个 = 总数（守恒定律）`
        },
        {
          step: 4,
          title: "寻找通路 → 确定求解路径",
          description: "在已知和未知之间寻找连接路径",
          entities: [
            { id: 'xiaoming', name: '小明', type: 'person' },
            { id: 'apples', name: '苹果', type: 'object' },
            { id: 'initial_5', name: '5个', type: 'quantity', value: 5, unit: '个' },
            { id: 'given_3', name: '3个', type: 'quantity', value: 3, unit: '个' },
            { id: 'addition', name: '加法', type: 'concept' },
            { id: 'total', name: '总数', type: 'concept' }
          ],
          relations: [
            { id: 'r1', source: 'xiaoming', target: 'apples', type: 'has_quantity', label: '拥有' },
            { id: 'r2', source: 'initial_5', target: 'addition', type: 'transformation', label: '输入' },
            { id: 'r3', source: 'given_3', target: 'addition', type: 'transformation', label: '输入' },
            { id: 'r4', source: 'addition', target: 'total', type: 'transformation', label: '产生' },
            { id: 'r5', source: 'initial_5', target: 'total', type: 'conservation', label: '守恒' },
            { id: 'r6', source: 'given_3', target: 'total', type: 'conservation', label: '守恒' }
          ],
          highlight: {
            entities: ['initial_5', 'given_3', 'total'],
            relations: ['r5', 'r6']
          },
          explanation: `寻找求解路径：
• 已知：5个、3个
• 未知：总数
• 找到路径：5个 → 守恒 → 总数 ← 守恒 ← 3个`
        },
        {
          step: 5,
          title: "沿路径推理 → 得出答案",
          description: "忽略无关推理，专注求解路径",
          entities: [
            { id: 'initial_5', name: '5个', type: 'quantity', value: 5, unit: '个' },
            { id: 'given_3', name: '3个', type: 'quantity', value: 3, unit: '个' },
            { id: 'total', name: '8个', type: 'quantity', value: 8, unit: '个' }
          ],
          relations: [
            { id: 'r5', source: 'initial_5', target: 'total', type: 'conservation', label: '守恒' },
            { id: 'r6', source: 'given_3', target: 'total', type: 'conservation', label: '守恒' }
          ],
          highlight: {
            entities: ['initial_5', 'given_3', 'total'],
            relations: ['r5', 'r6']
          },
          explanation: `沿最优路径推理：
• 推理路径：5个 + 3个 = 8个
• 忽略无关节点（小明、苹果、加法概念）
• 得出答案：8个苹果`
        }
      ]
    } else if (problemIndex === 1) { // 水池问题
      return [
        {
          step: 1,
          title: "分解题目 → 提取entity",
          description: "识别题目中的物体、数量和时间",
          entities: [
            { id: 'pool', name: '水池', type: 'object' },
            { id: 'water_100', name: '100升', type: 'quantity', value: 100, unit: '升' },
            { id: 'flow_20', name: '20升/时', type: 'quantity', value: 20, unit: '升/时' }
          ],
          relations: [
            { id: 'r1', source: 'pool', target: 'water_100', type: 'has_quantity', label: '包含' }
          ],
          highlight: {
            entities: ['pool', 'water_100', 'flow_20']
          },
          explanation: `从题目中提取出：
• 物体：水池
• 初始量：100升水
• 变化率：20升/小时`
        },
        {
          step: 2,
          title: "加入物性关系 → 构建graph",
          description: "添加时间和流出概念",
          entities: [
            { id: 'pool', name: '水池', type: 'object' },
            { id: 'water_100', name: '100升', type: 'quantity', value: 100, unit: '升' },
            { id: 'flow_20', name: '20升/时', type: 'quantity', value: 20, unit: '升/时' },
            { id: 'division', name: '除法', type: 'concept' },
            { id: 'time', name: '时间', type: 'concept' }
          ],
          relations: [
            { id: 'r1', source: 'pool', target: 'water_100', type: 'has_quantity', label: '包含' },
            { id: 'r2', source: 'water_100', target: 'division', type: 'transformation', label: '被除数' },
            { id: 'r3', source: 'flow_20', target: 'division', type: 'transformation', label: '除数' },
            { id: 'r4', source: 'division', target: 'time', type: 'transformation', label: '得到' }
          ],
          highlight: {
            entities: ['division', 'time'],
            relations: ['r2', 'r3', 'r4']
          },
          explanation: `加入数学概念：
• 运算：除法
• 目标：求时间
• 建立：总量÷速率=时间`
        },
        {
          step: 3,
          title: "物性关系推理 → 扩展连接",
          description: "通过时间流逝规律扩展关系",
          entities: [
            { id: 'pool', name: '水池', type: 'object' },
            { id: 'water_100', name: '100升', type: 'quantity', value: 100, unit: '升' },
            { id: 'flow_20', name: '20升/时', type: 'quantity', value: 20, unit: '升/时' },
            { id: 'division', name: '除法', type: 'concept' },
            { id: 'time', name: '时间', type: 'concept' }
          ],
          relations: [
            { id: 'r1', source: 'pool', target: 'water_100', type: 'has_quantity', label: '包含' },
            { id: 'r2', source: 'water_100', target: 'division', type: 'transformation', label: '被除数' },
            { id: 'r3', source: 'flow_20', target: 'division', type: 'transformation', label: '除数' },
            { id: 'r4', source: 'division', target: 'time', type: 'transformation', label: '得到' },
            { id: 'r5', source: 'water_100', target: 'time', type: 'temporal', label: '时间关系' },
            { id: 'r6', source: 'flow_20', target: 'time', type: 'temporal', label: '时间关系' }
          ],
          highlight: {
            relations: ['r5', 'r6']
          },
          explanation: `通过物性关系扩展：
• 时间关系：水量随时间线性减少
• 建立时间连续性约束`
        },
        {
          step: 4,
          title: "寻找通路 → 确定求解路径",
          description: "在已知和未知之间寻找连接路径",
          entities: [
            { id: 'pool', name: '水池', type: 'object' },
            { id: 'water_100', name: '100升', type: 'quantity', value: 100, unit: '升' },
            { id: 'flow_20', name: '20升/时', type: 'quantity', value: 20, unit: '升/时' },
            { id: 'division', name: '除法', type: 'concept' },
            { id: 'time', name: '时间', type: 'concept' }
          ],
          relations: [
            { id: 'r1', source: 'pool', target: 'water_100', type: 'has_quantity', label: '包含' },
            { id: 'r2', source: 'water_100', target: 'division', type: 'transformation', label: '被除数' },
            { id: 'r3', source: 'flow_20', target: 'division', type: 'transformation', label: '除数' },
            { id: 'r4', source: 'division', target: 'time', type: 'transformation', label: '得到' },
            { id: 'r5', source: 'water_100', target: 'time', type: 'temporal', label: '时间关系' },
            { id: 'r6', source: 'flow_20', target: 'time', type: 'temporal', label: '时间关系' }
          ],
          highlight: {
            entities: ['water_100', 'flow_20', 'time'],
            relations: ['r5', 'r6']
          },
          explanation: `寻找求解路径：
• 已知：100升、20升/时
• 未知：时间
• 路径：100升 → 时间关系 → 时间 ← 时间关系 ← 20升/时`
        },
        {
          step: 5,
          title: "沿路径推理 → 得出答案",
          description: "忽略无关推理，专注求解路径",
          entities: [
            { id: 'water_100', name: '100升', type: 'quantity', value: 100, unit: '升' },
            { id: 'flow_20', name: '20升/时', type: 'quantity', value: 20, unit: '升/时' },
            { id: 'time', name: '5小时', type: 'quantity', value: 5, unit: '小时' }
          ],
          relations: [
            { id: 'r5', source: 'water_100', target: 'time', type: 'temporal', label: '时间关系' },
            { id: 'r6', source: 'flow_20', target: 'time', type: 'temporal', label: '时间关系' }
          ],
          highlight: {
            entities: ['water_100', 'flow_20', 'time'],
            relations: ['r5', 'r6']
          },
          explanation: `沿最优路径推理：
• 推理：100升 ÷ 20升/时 = 5小时
• 忽略无关概念（水池、除法）
• 得出答案：5小时后水池为空`
        }
      ]
    } else { // 购物问题
      return [
        {
          step: 1,
          title: "分解题目 → 提取entity",
          description: "识别题目中的人物、物品和价格",
          entities: [
            { id: 'xiaohong', name: '小红', type: 'person' },
            { id: 'pens', name: '笔', type: 'object' },
            { id: 'quantity_3', name: '3支', type: 'quantity', value: 3, unit: '支' },
            { id: 'price_5', name: '5元/支', type: 'quantity', value: 5, unit: '元/支' }
          ],
          relations: [
            { id: 'r1', source: 'xiaohong', target: 'pens', type: 'has_quantity', label: '购买' }
          ],
          highlight: {
            entities: ['xiaohong', 'pens', 'quantity_3', 'price_5']
          },
          explanation: `从题目中提取出：
• 人物：小红
• 物品：笔
• 数量：3支
• 单价：5元/支`
        },
        {
          step: 2,
          title: "加入物性关系 → 构建graph",
          description: "添加乘法运算概念",
          entities: [
            { id: 'xiaohong', name: '小红', type: 'person' },
            { id: 'pens', name: '笔', type: 'object' },
            { id: 'quantity_3', name: '3支', type: 'quantity', value: 3, unit: '支' },
            { id: 'price_5', name: '5元/支', type: 'quantity', value: 5, unit: '元/支' },
            { id: 'multiplication', name: '乘法', type: 'concept' },
            { id: 'total_cost', name: '总价', type: 'concept' }
          ],
          relations: [
            { id: 'r1', source: 'xiaohong', target: 'pens', type: 'has_quantity', label: '购买' },
            { id: 'r2', source: 'quantity_3', target: 'multiplication', type: 'transformation', label: '数量' },
            { id: 'r3', source: 'price_5', target: 'multiplication', type: 'transformation', label: '单价' },
            { id: 'r4', source: 'multiplication', target: 'total_cost', type: 'transformation', label: '计算' }
          ],
          highlight: {
            entities: ['multiplication', 'total_cost'],
            relations: ['r2', 'r3', 'r4']
          },
          explanation: `加入数学概念：
• 运算：乘法
• 目标：总价
• 建立：数量×单价=总价`
        },
        {
          step: 3,
          title: "物性关系推理 → 扩展连接",
          description: "通过价值交换规律扩展关系",
          entities: [
            { id: 'xiaohong', name: '小红', type: 'person' },
            { id: 'pens', name: '笔', type: 'object' },
            { id: 'quantity_3', name: '3支', type: 'quantity', value: 3, unit: '支' },
            { id: 'price_5', name: '5元/支', type: 'quantity', value: 5, unit: '元/支' },
            { id: 'multiplication', name: '乘法', type: 'concept' },
            { id: 'total_cost', name: '总价', type: 'concept' }
          ],
          relations: [
            { id: 'r1', source: 'xiaohong', target: 'pens', type: 'has_quantity', label: '购买' },
            { id: 'r2', source: 'quantity_3', target: 'multiplication', type: 'transformation', label: '数量' },
            { id: 'r3', source: 'price_5', target: 'multiplication', type: 'transformation', label: '单价' },
            { id: 'r4', source: 'multiplication', target: 'total_cost', type: 'transformation', label: '计算' },
            { id: 'r5', source: 'quantity_3', target: 'total_cost', type: 'causal', label: '因果' },
            { id: 'r6', source: 'price_5', target: 'total_cost', type: 'causal', label: '因果' }
          ],
          highlight: {
            relations: ['r5', 'r6']
          },
          explanation: `通过物性关系扩展：
• 因果关系：数量和单价共同决定总价
• 价值守恒：支付金额等于商品价值`
        },
        {
          step: 4,
          title: "寻找通路 → 确定求解路径",
          description: "在已知和未知之间寻找连接路径",
          entities: [
            { id: 'xiaohong', name: '小红', type: 'person' },
            { id: 'pens', name: '笔', type: 'object' },
            { id: 'quantity_3', name: '3支', type: 'quantity', value: 3, unit: '支' },
            { id: 'price_5', name: '5元/支', type: 'quantity', value: 5, unit: '元/支' },
            { id: 'multiplication', name: '乘法', type: 'concept' },
            { id: 'total_cost', name: '总价', type: 'concept' }
          ],
          relations: [
            { id: 'r1', source: 'xiaohong', target: 'pens', type: 'has_quantity', label: '购买' },
            { id: 'r2', source: 'quantity_3', target: 'multiplication', type: 'transformation', label: '数量' },
            { id: 'r3', source: 'price_5', target: 'multiplication', type: 'transformation', label: '单价' },
            { id: 'r4', source: 'multiplication', target: 'total_cost', type: 'transformation', label: '计算' },
            { id: 'r5', source: 'quantity_3', target: 'total_cost', type: 'causal', label: '因果' },
            { id: 'r6', source: 'price_5', target: 'total_cost', type: 'causal', label: '因果' }
          ],
          highlight: {
            entities: ['quantity_3', 'price_5', 'total_cost'],
            relations: ['r5', 'r6']
          },
          explanation: `寻找求解路径：
• 已知：3支、5元/支
• 未知：总价
• 路径：3支 → 因果 → 总价 ← 因果 ← 5元/支`
        },
        {
          step: 5,
          title: "沿路径推理 → 得出答案",
          description: "忽略无关推理，专注求解路径",
          entities: [
            { id: 'quantity_3', name: '3支', type: 'quantity', value: 3, unit: '支' },
            { id: 'price_5', name: '5元/支', type: 'quantity', value: 5, unit: '元/支' },
            { id: 'total_cost', name: '15元', type: 'quantity', value: 15, unit: '元' }
          ],
          relations: [
            { id: 'r5', source: 'quantity_3', target: 'total_cost', type: 'causal', label: '因果' },
            { id: 'r6', source: 'price_5', target: 'total_cost', type: 'causal', label: '因果' }
          ],
          highlight: {
            entities: ['quantity_3', 'price_5', 'total_cost'],
            relations: ['r5', 'r6']
          },
          explanation: `沿最优路径推理：
• 推理：3支 × 5元/支 = 15元
• 忽略无关概念（小红、笔、乘法）
• 得出答案：一共花了15元`
        }
      ]
    }
  }, [examples])

  const analyzeStepByStep = useCallback(() => {
    const steps = generateStepData(selectedExample)
    setStepData(steps)
    setCurrentStep(0)
    setIsAnalyzing(true)
  }, [selectedExample, generateStepData])

  const renderStepGraph = (step: StepData) => {
    const width = 800
    const height = 400
    const centerX = width / 2
    const centerY = height / 2

    // 计算实体位置
    const entityPositions = step.entities.map((entity, index) => {
      const angle = (index * 2 * Math.PI) / step.entities.length
      const radius = 120
      return {
        ...entity,
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle)
      }
    })

    // 获取实体颜色
    const getEntityColor = (entity: Entity) => {
      const isHighlighted = step.highlight?.entities?.includes(entity.id)
      
      if (entity.type === 'quantity' && entity.value !== undefined) {
        return isHighlighted ? '#10b981' : '#86efac' // 绿色 - 已知数量
      }
      if (entity.id.includes('total') || entity.id.includes('time')) {
        return isHighlighted ? '#ef4444' : '#fca5a5' // 红色 - 未知
      }
      if (entity.type === 'concept') {
        return isHighlighted ? '#3b82f6' : '#93bbfc' // 蓝色 - 概念
      }
      return isHighlighted ? '#8b5cf6' : '#c4b5fd' // 紫色 - 其他
    }

    // 获取关系颜色和样式
    const getRelationStyle = (rel: Relation) => {
      const isHighlighted = step.highlight?.relations?.includes(rel.id)
      
      let color = '#6b7280'
      let dashArray = 'none'
      let width = 2
      
      switch(rel.type) {
        case 'has_quantity':
          color = isHighlighted ? '#6366f1' : '#a5b4fc'
          break
        case 'transformation':
          color = isHighlighted ? '#ec4899' : '#f9a8d4'
          break
        case 'conservation':
          color = isHighlighted ? '#f59e0b' : '#fcd34d'
          dashArray = '5,5'
          break
        case 'causal':
          color = isHighlighted ? '#8b5cf6' : '#c4b5fd'
          dashArray = '8,4'
          break
        case 'temporal':
          color = isHighlighted ? '#06b6d4' : '#67e8f9'
          dashArray = '10,5'
          break
      }
      
      if (isHighlighted) width = 4
      
      return { color, dashArray, width }
    }

    return (
      <svg width={width} height={height} className="border border-gray-200 rounded-lg bg-gradient-to-br from-gray-50 to-blue-50">
        {/* 步骤标题 */}
        <text x={width/2} y={30} textAnchor="middle" fontSize="16" fontWeight="bold" fill="#374151">
          步骤 {step.step}: {step.title}
        </text>
        
        {/* 绘制关系线 */}
        {step.relations.map((rel) => {
          const source = entityPositions.find(e => e.id === rel.source)
          const target = entityPositions.find(e => e.id === rel.target)
          
          if (!source || !target) return null
          
          const style = getRelationStyle(rel)
          
          // 计算曲线控制点
          const midX = (source.x + target.x) / 2
          const midY = (source.y + target.y) / 2
          const dx = target.x - source.x
          const dy = target.y - source.y
          const distance = Math.sqrt(dx * dx + dy * dy)
          const offset = 20
          const controlX = midX - (dy / distance) * offset
          const controlY = midY + (dx / distance) * offset
          
          return (
            <g key={rel.id}>
              <path
                d={`M ${source.x} ${source.y} Q ${controlX} ${controlY} ${target.x} ${target.y}`}
                stroke={style.color}
                strokeWidth={style.width}
                fill="none"
                strokeDasharray={style.dashArray}
                opacity="0.8"
              />
              {step.step >= 2 && (
                <text
                  x={controlX}
                  y={controlY}
                  textAnchor="middle"
                  fontSize="10"
                  fill={style.color}
                  fontWeight="500"
                >
                  {rel.label}
                </text>
              )}
            </g>
          )
        })}
        
        {/* 绘制实体节点 */}
        {entityPositions.map((entity) => {
          const isHighlighted = step.highlight?.entities?.includes(entity.id)
          
          return (
            <g key={entity.id}>
              <circle
                cx={entity.x}
                cy={entity.y}
                r={isHighlighted ? 35 : 30}
                fill={getEntityColor(entity)}
                stroke="white"
                strokeWidth="2"
                className={isHighlighted ? "drop-shadow-lg" : "drop-shadow"}
              />
              <text
                x={entity.x}
                y={entity.y - 5}
                textAnchor="middle"
                fontSize="11"
                fill="white"
                fontWeight={isHighlighted ? "bold" : "normal"}
              >
                {entity.name}
              </text>
              {entity.value !== undefined && (
                <text
                  x={entity.x}
                  y={entity.y + 8}
                  textAnchor="middle"
                  fontSize="10"
                  fill="white"
                  fontWeight="bold"
                >
                  {entity.value}{entity.unit || ''}
                </text>
              )}
            </g>
          )
        })}
      </svg>
    )
  }

  return (
    <div className="space-y-6">
      {/* 算法讲解部分 */}
      {showAlgorithmExplanation && (
        <Card className="max-w-5xl mx-auto">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <BookOpen className="h-6 w-6 text-purple-600" />
                基于物性关系的推理算法讲解
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowAlgorithmExplanation(false)}
              >
                收起
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="bg-blue-50 border-l-4 border-blue-400 p-4">
                <p className="text-blue-900">
                  <strong>算法精髓：</strong>通过引入物性关系，在断开的已知和未知之间建立连接通路，
                  然后沿着找到的路径进行推理，忽略无关的推理分支。
                </p>
              </div>

              {/* 算法步骤 */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="font-semibold mb-3">算法五步骤</h3>
                <div className="space-y-2">
                  {[
                    '1. 题目分解 → 提取phrase和entity，获得直陈关系',
                    '2. 构建物性Graph → 直陈关系 + 物性entity',
                    '3. 物性关系推理 → 通过物性规律扩展Graph',
                    '4. 寻找通路 → 找到已知到未知的路径',
                    '5. 精简推理 → 只保留路径，忽略无关推理'
                  ].map((step, i) => (
                    <div key={i} className="flex items-start">
                      <span className="text-purple-600 font-semibold mr-2">•</span>
                      <span>{step}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* 推理规则 */}
              <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4">
                <h4 className="font-semibold text-yellow-900 mb-2">三条核心推理规则</h4>
                <div className="space-y-2 text-sm">
                  <div>
                    <strong className="text-yellow-900">规则1（守恒扩展）：</strong>
                    看到"一共"、"总共"时，建立部分与整体的守恒关系
                    <div className="text-xs text-yellow-700 ml-4 mt-1">应用时机：识别总量概念</div>
                  </div>
                  <div>
                    <strong className="text-yellow-900">规则2（因果链接）：</strong>
                    存在动作或变化时，建立原因→结果的因果关系
                    <div className="text-xs text-yellow-700 ml-4 mt-1">应用时机：看到动词、变化</div>
                  </div>
                  <div>
                    <strong className="text-yellow-900">规则3（时空连续）：</strong>
                    涉及时间或空间变化时，建立连续性关系
                    <div className="text-xs text-yellow-700 ml-4 mt-1">应用时机：涉及速率、位移</div>
                  </div>
                </div>
              </div>

              <div className="bg-green-50 p-4 rounded-lg">
                <p className="text-green-900 font-medium">
                  <strong>记住：</strong>我们要的不是所有可能的推理，而是找到一条从已知到未知的通路，
                  然后沿着这条路径推理即可。其余的推理都可以"忘掉"。
                </p>
              </div>

              {/* 算法总结公式 */}
              <div className="bg-gradient-to-r from-purple-50 to-blue-50 p-4 rounded-lg">
                <h3 className="font-semibold text-gray-900 mb-2">算法总结公式</h3>
                <div className="bg-white p-3 rounded font-mono text-sm">
                  <p>题目 → Phrase + Entity → 直陈关系 → +物性Entity → 物性Graph</p>
                  <p className="mt-1">→ 物性推理扩展 → 找通路(已知→未知) → 沿路径推理 → 答案</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* 分步演示部分 */}
      <Card className="max-w-5xl mx-auto">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Network className="h-6 w-6 text-purple-600" />
            基于物性关系的推理 - 分步演示
            {!showAlgorithmExplanation && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowAlgorithmExplanation(true)}
                className="ml-auto"
              >
                查看算法讲解
              </Button>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* 例题选择 */}
          <div>
            <h3 className="font-semibold mb-3">选择示例问题：</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {examples.map((example, index) => (
                <button
                  key={index}
                  onClick={() => {
                    setSelectedExample(index)
                    setIsAnalyzing(false)
                    setCurrentStep(0)
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
          {!isAnalyzing && (
            <div className="flex justify-center">
              <Button
                onClick={analyzeStepByStep}
                className="flex items-center gap-2"
                size="lg"
              >
                <Zap className="h-5 w-5" />
                开始分步推理
              </Button>
            </div>
          )}

          {/* 分步展示 */}
          {isAnalyzing && stepData.length > 0 && (
            <div className="space-y-4">
              {/* 步骤导航 */}
              <div className="flex items-center justify-between bg-gray-100 p-4 rounded-lg">
                <Button
                  onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
                  disabled={currentStep === 0}
                  variant="outline"
                  size="sm"
                >
                  <ChevronLeft className="h-4 w-4 mr-1" />
                  上一步
                </Button>
                
                <div className="flex space-x-2">
                  {stepData.map((_, index) => (
                    <button
                      key={index}
                      onClick={() => setCurrentStep(index)}
                      className={`w-10 h-10 rounded-full transition-all ${
                        index === currentStep 
                          ? 'bg-purple-600 text-white' 
                          : index < currentStep
                          ? 'bg-purple-200 text-purple-800'
                          : 'bg-gray-300 text-gray-600'
                      }`}
                    >
                      {index + 1}
                    </button>
                  ))}
                </div>
                
                <Button
                  onClick={() => setCurrentStep(Math.min(stepData.length - 1, currentStep + 1))}
                  disabled={currentStep === stepData.length - 1}
                  variant="outline"
                  size="sm"
                >
                  下一步
                  <ChevronRight className="h-4 w-4 ml-1" />
                </Button>
              </div>

              {/* 当前步骤内容 */}
              <div className="bg-white p-6 rounded-lg border">
                <h3 className="text-lg font-semibold mb-2 text-purple-700">
                  {stepData[currentStep].description}
                </h3>
                
                {/* 图形展示 */}
                <div className="mb-4">
                  {renderStepGraph(stepData[currentStep])}
                </div>
                
                {/* 步骤说明 */}
                <div className="bg-purple-50 p-4 rounded-lg">
                  <pre className="text-sm whitespace-pre-wrap text-purple-900">
                    {stepData[currentStep].explanation}
                  </pre>
                </div>
              </div>

              {/* 进度条 */}
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${((currentStep + 1) / stepData.length) * 100}%` }}
                />
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

export default StandalonePhysicalReasoning