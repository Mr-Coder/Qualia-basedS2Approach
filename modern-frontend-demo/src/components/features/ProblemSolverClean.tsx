import React, { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Textarea } from '@/components/ui/Textarea'

export const ProblemSolverClean: React.FC = () => {
  const [problem, setProblem] = useState('')
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const handleSolve = async () => {
    console.log('🧪 测试解题 - 不使用状态管理')
    setError('')
    setIsLoading(true)
    setResult(null)
    
    try {
      // 直接设置模拟数据，不通过API调用
      await new Promise(resolve => setTimeout(resolve, 500))
      
      const mockResult = {
        answer: "8个苹果",
        confidence: 0.95,
        strategy: 'auto',
        entities: [
          { id: "xiaoming", name: "小明", type: "person" },
          { id: "apple", name: "苹果", type: "object" }
        ],
        relationships: [
          { source: "xiaoming", target: "apple", type: "拥有关系", weight: 5 }
        ],
        deepRelations: [
          {
            id: "deep_rel_1",
            source: "小明",
            target: "苹果",
            type: "implicit_dependency",
            depth: "shallow",
            confidence: 0.85,
            label: "拥有关系",
            evidence: ["语义模式匹配"],
            constraints: ["非负数量约束"],
            visualization: {
              depth_color: "#3b82f6",
              confidence_size: 40,
              relation_width: 3,
              animation_delay: 0.2,
              hover_info: {
                title: "拥有关系",
                details: ["语义模式匹配"],
                constraints: ["非负数量约束"]
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
        }
      }
      
      console.log('✅ 设置模拟结果:', mockResult)
      setResult(mockResult)
      
    } catch (err) {
      console.error('❌ 错误:', err)
      setError(`错误: ${err}`)
    } finally {
      setIsLoading(false)
    }
  }

  const clearResult = () => {
    setResult(null)
    setError('')
    console.log('🧹 清除结果')
  }

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>🧪 纯净测试版解题器</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Textarea
              label="数学问题"
              value={problem}
              onChange={(e) => setProblem(e.target.value)}
              placeholder="小明有10个苹果，他给了小红3个，又买了5个，请问小明现在有多少个苹果？"
            />
            
            <div className="flex gap-2">
              <Button 
                onClick={handleSolve}
                disabled={!problem.trim() || isLoading}
                className="flex-1"
              >
                {isLoading ? '解题中...' : '🚀 开始解题'}
              </Button>
              
              <Button 
                onClick={clearResult}
                variant="outline"
                disabled={!result}
              >
                🧹 清除结果
              </Button>
            </div>
            
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="text-red-800 font-medium">错误:</div>
                <div className="text-red-600">{error}</div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {result && (
        <Card>
          <CardHeader>
            <CardTitle>✅ 解题结果</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <div className="text-2xl font-bold text-green-800 mb-2">
                答案：{result.answer}
              </div>
              <div className="text-sm text-green-600">
                置信度：{(result.confidence * 100).toFixed(1)}%
              </div>
              
              <div className="mt-4 bg-white rounded p-3 border">
                <div className="text-sm font-medium mb-2">📊 数据结构检查</div>
                <div className="text-xs space-y-1">
                  <div>实体数量: {result.entities?.length || 0}</div>
                  <div>关系数量: {result.relationships?.length || 0}</div>
                  <div>深度关系: {result.deepRelations?.length || 0}</div>
                  <div>隐含约束: {result.implicitConstraints?.length || 0}</div>
                  <div>可视化配置: {result.visualizationConfig ? '✅' : '❌'}</div>
                </div>
              </div>
              
              <details className="mt-4">
                <summary className="cursor-pointer text-sm font-medium">🔍 详细数据结构</summary>
                <pre className="text-xs bg-gray-50 p-2 rounded mt-2 overflow-auto max-h-60">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </details>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

export default ProblemSolverClean