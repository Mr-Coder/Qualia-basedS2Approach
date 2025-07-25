import React from 'react'
import Layout from '@/components/layout/Layout'
import { useHistory, useProblemStore } from '@/stores/problemStore'

const StoreTest: React.FC = () => {
  try {
    return (
      <Layout activeTab="smart" setActiveTab={() => {}}>
        <div style={{ padding: '20px', fontSize: '18px' }}>
          <h1>🔍 Store状态管理测试</h1>
          <StoreStatus />
          <StoreOperations />
        </div>
      </Layout>
    )
  } catch (error) {
    return (
      <div style={{ padding: '20px', background: '#f8d7da' }}>
        <h1>❌ Store测试失败</h1>
        <p>错误: {String(error)}</p>
        <pre style={{ fontSize: '12px', overflow: 'auto' }}>
          {error instanceof Error ? error.stack : String(error)}
        </pre>
      </div>
    )
  }
}

const StoreStatus: React.FC = () => {
  try {
    const history = useHistory()
    const currentProblem = useProblemStore(state => state.currentProblem)
    const selectedStrategy = useProblemStore(state => state.selectedStrategy)
    
    return (
      <div>
        <div style={{ background: '#d4edda', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
          <h2>✅ Store状态读取正常</h2>
          <p>历史记录数量: {history.length}</p>
          <p>当前问题: {currentProblem || '无'}</p>
          <p>选择策略: {selectedStrategy}</p>
        </div>
        
        <div style={{ background: '#f0f8ff', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
          <h3>测试进度:</h3>
          <ul>
            <li>✓ React基础渲染</li>
            <li>✓ Layout组件</li>
            <li>✓ Store状态管理</li>
            <li>? HistoryPanel组件</li>
          </ul>
        </div>
      </div>
    )
  } catch (error) {
    return (
      <div style={{ background: '#f8d7da', padding: '15px', margin: '15px 0' }}>
        <h2>❌ Store状态读取失败</h2>
        <p>错误: {String(error)}</p>
      </div>
    )
  }
}

const StoreOperations: React.FC = () => {
  const addToHistory = useProblemStore(state => state.addToHistory)
  const clearHistory = useProblemStore(state => state.clearHistory)
  const setProblem = useProblemStore(state => state.setProblem)
  
  const handleAddTest = () => {
    try {
      const testEntry = {
        id: `test-${Date.now()}`,
        problem: '测试问题：小明有3个苹果，小红有2个苹果，一共有多少个苹果？',
        answer: '5个苹果',
        strategy: 'cot' as const,
        timestamp: new Date(),
        confidence: 0.95
      }
      addToHistory(testEntry)
    } catch (error) {
      console.error('添加历史记录失败:', error)
    }
  }

  const handleSetProblem = () => {
    try {
      setProblem('这是一个测试问题')
    } catch (error) {
      console.error('设置问题失败:', error)
    }
  }

  return (
    <div style={{ background: '#fff3cd', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
      <h3>Store操作测试:</h3>
      <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
        <button 
          onClick={handleAddTest}
          style={{ padding: '8px 16px', backgroundColor: '#28a745', color: 'white', border: 'none', borderRadius: '4px' }}
        >
          添加测试历史记录
        </button>
        <button 
          onClick={handleSetProblem}
          style={{ padding: '8px 16px', backgroundColor: '#007bff', color: 'white', border: 'none', borderRadius: '4px' }}
        >
          设置测试问题
        </button>
        <button 
          onClick={clearHistory}
          style={{ padding: '8px 16px', backgroundColor: '#dc3545', color: 'white', border: 'none', borderRadius: '4px' }}
        >
          清空历史记录
        </button>
      </div>
    </div>
  )
}

export default StoreTest