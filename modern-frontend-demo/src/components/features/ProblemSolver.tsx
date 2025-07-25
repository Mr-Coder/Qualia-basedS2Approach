import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Textarea } from '@/components/ui/Textarea'
import { useProblemStore } from '@/stores/problemStore'
import { generateId } from '@/utils/helpers'
import EntityRelationshipDiagram from './EntityRelationshipDiagram'

const exampleProblems = [
  "小明有10个苹果，他给了小红3个，又买了5个，请问小明现在有多少个苹果？",
  "一个长方形的长是12厘米，宽是8厘米，求这个长方形的面积。",
  "班级里有30名学生，其中男生比女生多4人，请问男生和女生各有多少人？"
]

export const ProblemSolver: React.FC = () => {
  const {
    solveResult,
    isLoading,
    error,
    setSolveResult,
    setLoading,
    setError,
    addToHistory
  } = useProblemStore()

  const [localProblem, setLocalProblem] = useState('')

  const handleSolve = async () => {
    if (!localProblem.trim()) {
      setError('请输入问题')
      return
    }

    setLoading(true)
    setError(null)

    try {
      // 模拟处理时间
      await new Promise(resolve => setTimeout(resolve, 800))
      
      // 创建简化的模拟数据
      const result = {
        answer: "8个苹果",
        confidence: 0.95,
        strategy: 'auto',
        steps: [
          "步骤1: 识别问题中的实体和关系",
          "步骤2: 建立深度隐含关系模型",
          "步骤3: 应用数学运算求解"
        ],
        entities: [
          { id: "xiaoming", name: "小明", type: "person" as const },
          { id: "xiaohong", name: "小红", type: "person" as const },
          { id: "apple", name: "苹果", type: "object" as const },
          { id: "five", name: "5", type: "concept" as const },
          { id: "three", name: "3", type: "concept" as const }
        ],
        relationships: [
          { source: "xiaoming", target: "apple", type: "拥有关系", weight: 5 },
          { source: "xiaohong", target: "apple", type: "拥有关系", weight: 3 }
        ],
        constraints: [
          "数量守恒定律：苹果总数 = 各部分之和",
          "非负性约束：物品数量必须 ≥ 0"
        ],
        physicalConstraints: [
          "数量守恒定律：苹果总数 = 各部分之和",
          "非负性约束：物品数量必须 ≥ 0",
          "整数约束：可数物品为整数个"
        ],
        physicalProperties: {
          conservationLaws: ["物质守恒", "数量守恒"],
          spatialRelations: ["拥有关系", "位置分布"],
          temporalConstraints: ["操作顺序"],
          materialProperties: ["可数性", "物理存在"]
        },
        deepRelations: [
          {
            id: "deep_rel_1",
            source: "小明",
            target: "苹果",
            type: "implicit_dependency",
            depth: "shallow" as const,
            confidence: 0.85,
            label: "拥有关系",
            evidence: ["语义模式匹配", "实体类型推理"],
            constraints: ["非负数量约束", "整数约束"],
            visualization: {
              depth_color: "#3b82f6",
              confidence_size: 40,
              relation_width: 3,
              animation_delay: 0.2,
              hover_info: {
                title: "拥有关系",
                details: ["语义模式匹配", "实体类型推理"],
                constraints: ["非负数量约束", "整数约束"]
              }
            }
          }
        ],
        implicitConstraints: [
          {
            id: "constraint_1",
            type: "non_negativity",
            description: "数量必须为非负整数",
            entities: ["苹果"],
            expression: "count(苹果) ≥ 0",
            confidence: 0.95,
            color: "#10b981",
            icon: "🔢",
            visualization: {
              constraint_priority: 1,
              visualization_layer: "primary",
              animation_type: "fade_in",
              detail_panel: {
                title: "非负性约束",
                expression: "count(苹果) ≥ 0",
                method: "实体类型分析",
                entities: ["苹果"]
              }
            }
          }
        ],
        visualizationConfig: {
          show_depth_indicators: true,
          show_constraint_panels: true,
          enable_interactive_exploration: true,
          animation_sequence: true
        },
        processingTime: 0.13
      }

      setSolveResult(result)
      addToHistory({
        id: generateId(),
        problem: localProblem,
        answer: result.answer,
        strategy: 'auto',
        timestamp: new Date(),
        confidence: result.confidence
      })

    } catch (error) {
      console.error('❌ 解题失败:', error)
      setError('解题过程中发生错误')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <span>🧮</span>
            <span>智能数学问题求解器</span>
            <span className="text-sm font-normal text-gray-500">
              • 深度隐含关系发现算法
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {/* 示例问题 */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                📚 示例问题（点击使用）
              </label>
              <div className="grid grid-cols-1 gap-2">
                {exampleProblems.map((example, index) => (
                  <motion.button
                    key={index}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => setLocalProblem(example)}
                    className="text-left p-3 bg-gray-50 hover:bg-blue-50 border border-gray-200 hover:border-blue-300 rounded-lg text-sm transition-colors"
                  >
                    {example}
                  </motion.button>
                ))}
              </div>
            </div>

            {/* 问题输入 */}
            <Textarea
              label="输入数学问题"
              placeholder="请输入您要解决的数学问题..."
              value={localProblem}
              onChange={(e) => setLocalProblem(e.target.value)}
              className="min-h-[120px]"
              error={error}
            />

            {/* 解题按钮 */}
            <Button
              onClick={handleSolve}
              loading={isLoading}
              disabled={!localProblem.trim()}
              size="lg"
              className="w-full"
            >
              {isLoading ? '正在解题...' : '🚀 开始解题'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* 结果展示 */}
      <AnimatePresence>
        {solveResult && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            <div className="grid grid-cols-1 gap-6">
              {/* 基本结果 */}
              <Card>
                <CardHeader>
                  <CardTitle>✅ 解答结果</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                    <div className="text-2xl font-bold text-green-800">
                      答案：{solveResult.answer}
                    </div>
                    <div className="text-sm text-green-600 mt-2">
                      置信度：{(solveResult.confidence * 100).toFixed(1)}%
                    </div>
                    {solveResult.processingTime && (
                      <div className="text-sm text-green-600">
                        处理时间：{solveResult.processingTime.toFixed(2)}秒
                      </div>
                    )}
                  </div>
                  
                  {/* 深度关系统计 */}
                  {solveResult.deepRelations && solveResult.deepRelations.length > 0 && (
                    <div className="bg-purple-50 border border-purple-200 rounded-lg p-4 mt-4">
                      <div className="text-sm font-medium text-purple-800 mb-2">
                        🔬 深度隐含关系发现
                      </div>
                      <div className="text-sm text-purple-600">
                        发现 {solveResult.deepRelations.length} 个深度关系
                      </div>
                      <div className="text-sm text-purple-600">
                        挖掘 {solveResult.implicitConstraints?.length || 0} 个隐含约束
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* 物性关系图 */}
              <EntityRelationshipDiagram
                entities={solveResult.entities}
                relationships={solveResult.relationships}
                physicalConstraints={solveResult.physicalConstraints || []}
                physicalProperties={solveResult.physicalProperties}
                deepRelations={solveResult.deepRelations || []}
                implicitConstraints={solveResult.implicitConstraints || []}
                visualizationConfig={solveResult.visualizationConfig}
                width={700}
                height={400}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default ProblemSolver