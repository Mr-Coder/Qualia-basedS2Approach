import React, { useState } from 'react'
import Layout from '@/components/layout/Layout'
import { useProblemStore } from '@/stores/problemStore'

const StoreDebugger: React.FC = () => {
  const [testResult, setTestResult] = useState<string>('')
  const [error, setError] = useState<string | null>(null)
  
  const store = useProblemStore()

  const testBasicStore = () => {
    try {
      setError(null)
      setTestResult('开始测试基础Store操作...')
      
      // 测试基础操作
      store.setProblem('测试问题')
      setTestResult(prev => prev + '\n✅ setProblem 成功')
      
      store.setStrategy('cot')
      setTestResult(prev => prev + '\n✅ setStrategy 成功')
      
      store.setLoading(true)
      setTestResult(prev => prev + '\n✅ setLoading 成功')
      
      store.setLoading(false)
      setTestResult(prev => prev + '\n✅ setLoading(false) 成功')
      
      setTestResult(prev => prev + '\n🎉 基础Store操作全部正常')
      
    } catch (err) {
      setError(`基础Store测试失败: ${String(err)}`)
    }
  }

  const testSolveResult = () => {
    try {
      setError(null)
      setTestResult('开始测试setSolveResult...')
      
      const mockResult = {
        answer: '5个苹果',
        confidence: 0.95,
        strategy: 'cot' as const,
        steps: ['步骤1', '步骤2'],
        entities: [
          { id: 'test1', name: '测试实体1', type: 'person' as const }
        ],
        relationships: [
          { source: 'test1', target: 'test2', type: '测试关系' }
        ],
        constraints: ['测试约束']
      }
      
      console.log('尝试设置solve result:', mockResult)
      store.setSolveResult(mockResult)
      setTestResult(prev => prev + '\n✅ setSolveResult 成功')
      
    } catch (err) {
      setError(`setSolveResult测试失败: ${String(err)}`)
      console.error('setSolveResult错误:', err)
    }
  }

  const testAddHistory = () => {
    try {
      setError(null)
      setTestResult('开始测试addToHistory...')
      
      const historyItem = {
        id: `test-${Date.now()}`,
        problem: '测试问题',
        answer: '测试答案',
        strategy: 'cot' as const,
        timestamp: new Date(),
        confidence: 0.95
      }
      
      console.log('尝试添加历史记录:', historyItem)
      store.addToHistory(historyItem)
      setTestResult(prev => prev + '\n✅ addToHistory 成功')
      
    } catch (err) {
      setError(`addToHistory测试失败: ${String(err)}`)
      console.error('addToHistory错误:', err)
    }
  }

  const testCompleteFlow = () => {
    try {
      setError(null)
      setTestResult('开始测试完整流程...')
      
      // 1. 设置问题
      store.setProblem('完整测试问题')
      setTestResult(prev => prev + '\n✅ 1. 设置问题成功')
      
      // 2. 设置结果
      const result = {
        answer: '完整测试答案',
        confidence: 0.9,
        strategy: 'cot' as const,
        steps: ['完整测试步骤'],
        entities: [],
        relationships: [],
        constraints: []
      }
      
      console.log('设置完整结果:', result)
      store.setSolveResult(result)
      setTestResult(prev => prev + '\n✅ 2. 设置结果成功')
      
      // 3. 添加历史
      const historyItem = {
        id: `complete-${Date.now()}`,
        problem: '完整测试问题',
        answer: '完整测试答案',
        strategy: 'cot' as const,
        timestamp: new Date(),
        confidence: 0.9
      }
      
      console.log('添加完整历史:', historyItem)
      store.addToHistory(historyItem)
      setTestResult(prev => prev + '\n✅ 3. 添加历史成功')
      
      setTestResult(prev => prev + '\n🎉 完整流程测试成功！')
      
    } catch (err) {
      setError(`完整流程测试失败: ${String(err)}`)
      console.error('完整流程错误:', err)
    }
  }

  return (
    <Layout activeTab="smart" setActiveTab={() => {}}>
      <div style={{ padding: '20px' }}>
        <h1>🔧 Store状态调试器</h1>
        
        <div style={{ background: '#f0f8ff', padding: '15px', margin: '15px 0', border: '1px solid #0066cc' }}>
          <h3>当前Store状态:</h3>
          <p>问题: {store.currentProblem || '无'}</p>
          <p>策略: {store.selectedStrategy}</p>
          <p>加载中: {store.isLoading ? '是' : '否'}</p>
          <p>结果: {store.solveResult ? '有' : '无'}</p>
          <p>历史记录: {store.history.length} 条</p>
          <p>错误: {store.error || '无'}</p>
        </div>

        <div style={{ display: 'flex', gap: '10px', margin: '15px 0', flexWrap: 'wrap' }}>
          <button 
            onClick={testBasicStore}
            style={{ padding: '10px 20px', backgroundColor: '#007bff', color: 'white', border: 'none', borderRadius: '4px' }}
          >
            测试基础Store操作
          </button>
          <button 
            onClick={testSolveResult}
            style={{ padding: '10px 20px', backgroundColor: '#ffc107', color: 'black', border: 'none', borderRadius: '4px' }}
          >
            测试setSolveResult
          </button>
          <button 
            onClick={testAddHistory}
            style={{ padding: '10px 20px', backgroundColor: '#28a745', color: 'white', border: 'none', borderRadius: '4px' }}
          >
            测试addToHistory
          </button>
          <button 
            onClick={testCompleteFlow}
            style={{ padding: '10px 20px', backgroundColor: '#dc3545', color: 'white', border: 'none', borderRadius: '4px' }}
          >
            测试完整流程
          </button>
        </div>

        {error && (
          <div style={{ background: '#f8d7da', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
            <h3>❌ 错误信息:</h3>
            <p>{error}</p>
          </div>
        )}

        {testResult && (
          <div style={{ background: '#e8f5e8', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
            <h3>📝 测试结果:</h3>
            <pre style={{ fontSize: '12px', whiteSpace: 'pre-wrap' }}>
              {testResult}
            </pre>
          </div>
        )}

        <div style={{ background: '#fff3cd', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
          <h3>💡 调试提示:</h3>
          <p>请打开浏览器开发者工具(F12) → Console标签页查看详细日志</p>
          <p>如果某个测试失败，说明Store操作有问题</p>
        </div>
      </div>
    </Layout>
  )
}

export default StoreDebugger