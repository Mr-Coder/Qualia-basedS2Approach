import React, { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Textarea } from '@/components/ui/Textarea'
import { apiClient, testAPIConnection } from '@/services/unifiedAPI'


export const ProblemSolverDebug: React.FC = () => {
  const [problem, setProblem] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const handleSolve = async () => {
    console.log('🐛 开始调试解题流程')
    setError('')
    setIsLoading(true)
    
    try {
      console.log('🐛 问题:', problem)
      
      // 最简单的API调用
      const response = await fetch('http://localhost:5004/api/solve', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          problem: problem,
          strategy: 'auto'
        }),
      })
      
      console.log('🐛 响应状态:', response.status)
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      
      const data = await response.json()
      console.log('🐛 响应数据:', data)
      
      setResult(data)
      console.log('🐛 设置结果成功')
      
    } catch (err) {
      console.error('🐛 错误:', err)
      setError(err.message)
    } finally {
      setIsLoading(false)
      console.log('🐛 解题流程结束')
    }
  }

  console.log('🐛 组件渲染状态:', { result, error, isLoading })

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>🐛 调试版问题求解器</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Textarea
              label="测试问题"
              value={problem}
              onChange={(e) => setProblem(e.target.value)}
              placeholder="输入数学问题进行调试..."
            />
            
            <Button 
              onClick={handleSolve}
              disabled={!problem.trim() || isLoading}
              className="w-full"
            >
              {isLoading ? '调试中...' : '🐛 调试解题'}
            </Button>
            
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="text-red-800 font-medium">错误:</div>
                <div className="text-red-600">{error}</div>
              </div>
            )}
            
            {result && (
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <div className="text-green-800 font-medium mb-2">调试结果:</div>
                <pre className="text-sm bg-white p-2 rounded border overflow-auto">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default ProblemSolverDebug