import React, { useState } from 'react'
import Layout from '@/components/layout/Layout'
import { useProblemStore } from '@/stores/problemStore'

const SolveDebugger: React.FC = () => {
  const [testStep, setTestStep] = useState(1)
  const [debugError, setDebugError] = useState<string | null>(null)
  
  const { 
    setProblem, 
    setStrategy, 
    setSolveResult, 
    setLoading, 
    setError: setStoreError,
    addToHistory,
    currentProblem,
    selectedStrategy,
    isLoading
  } = useProblemStore()

  return (
    <Layout activeTab="smart" setActiveTab={() => {}}>
      <div style={{ padding: '20px' }}>
        <h1>🔍 解题功能调试器</h1>
        
        <div style={{ background: '#f0f8ff', padding: '15px', margin: '15px 0', border: '1px solid #0066cc' }}>
          <h3>当前状态:</h3>
          <p>问题: {currentProblem || '无'}</p>
          <p>策略: {selectedStrategy}</p>
          <p>加载中: {isLoading ? '是' : '否'}</p>
        </div>

        <div style={{ display: 'flex', gap: '10px', margin: '15px 0', flexWrap: 'wrap' }}>
          <button 
            onClick={() => setTestStep(1)}
            style={{ 
              padding: '8px 16px', 
              backgroundColor: testStep === 1 ? '#007bff' : '#6c757d', 
              color: 'white', 
              border: 'none', 
              borderRadius: '4px' 
            }}
          >
            步骤1: 设置问题
          </button>
          <button 
            onClick={() => setTestStep(2)}
            style={{ 
              padding: '8px 16px', 
              backgroundColor: testStep === 2 ? '#ffc107' : '#6c757d', 
              color: 'black', 
              border: 'none', 
              borderRadius: '4px' 
            }}
          >
            步骤2: 模拟解题API
          </button>
          <button 
            onClick={() => setTestStep(3)}
            style={{ 
              padding: '8px 16px', 
              backgroundColor: testStep === 3 ? '#dc3545' : '#6c757d', 
              color: 'white', 
              border: 'none', 
              borderRadius: '4px' 
            }}
          >
            步骤3: 测试真实解题
          </button>
        </div>

        {testStep === 1 && <Step1SetProblem />}
        {testStep === 2 && <Step2MockSolve />}
        {testStep === 3 && <Step3RealSolve />}
      </div>
    </Layout>
  )
}

const Step1SetProblem: React.FC = () => {
  const { setProblem, setStrategy } = useProblemStore()

  const handleSetProblem = () => {
    try {
      setProblem('小明有3个苹果，小红有2个苹果，一共有多少个苹果？')
      setStrategy('cot')
    } catch (error) {
      console.error('设置问题失败:', error)
    }
  }

  return (
    <div>
      <div style={{ background: '#d4edda', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
        <h2>步骤1: 设置问题</h2>
        <p>测试Store的基本操作</p>
      </div>
      
      <button 
        onClick={handleSetProblem}
        style={{ padding: '10px 20px', backgroundColor: '#28a745', color: 'white', border: 'none', borderRadius: '4px' }}
      >
        设置测试问题
      </button>
      
      <div style={{ background: '#f8f9fa', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
        <h3>如果成功，上面的"当前状态"应该显示问题文本</h3>
      </div>
    </div>
  )
}

const Step2MockSolve: React.FC = () => {
  const { setSolveResult, addToHistory } = useProblemStore()

  const handleMockSolve = () => {
    try {
      const mockResult = {
        answer: '5个苹果',
        confidence: 0.95,
        strategy: 'cot',
        steps: [
          '分析问题：求两个人苹果的总数',
          '小明有3个苹果',
          '小红有2个苹果',
          '总数 = 3 + 2 = 5个苹果'
        ],
        entities: [
          { id: 'xiaoming', name: '小明', type: 'person' as const },
          { id: 'xiaohong', name: '小红', type: 'person' as const },
          { id: 'apples', name: '苹果', type: 'object' as const }
        ],
        relationships: [
          { source: 'xiaoming', target: 'apples', type: '拥有', weight: 3 },
          { source: 'xiaohong', target: 'apples', type: '拥有', weight: 2 }
        ],
        constraints: ['苹果数量为非负整数'],
        processingTime: 1200
      }

      setSolveResult(mockResult)
      
      // 添加到历史记录
      addToHistory({
        id: `mock-${Date.now()}`,
        problem: '小明有3个苹果，小红有2个苹果，一共有多少个苹果？',
        answer: '5个苹果',
        strategy: 'cot',
        timestamp: new Date(),
        confidence: 0.95
      })

    } catch (error) {
      console.error('模拟解题失败:', error)
    }
  }

  return (
    <div>
      <div style={{ background: '#fff3cd', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
        <h2>步骤2: 模拟解题API</h2>
        <p>测试解题结果的处理，不调用真实API</p>
      </div>
      
      <button 
        onClick={handleMockSolve}
        style={{ padding: '10px 20px', backgroundColor: '#ffc107', color: 'black', border: 'none', borderRadius: '4px' }}
      >
        模拟解题成功
      </button>
      
      <div style={{ background: '#f8f9fa', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
        <h3>如果成功，应该可以看到解题结果和历史记录</h3>
      </div>
    </div>
  )
}

const Step3RealSolve: React.FC = () => {
  const [apiError, setApiError] = useState<string | null>(null)
  const [apiResponse, setApiResponse] = useState<any>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const { currentProblem, selectedStrategy, setLoading, setSolveResult, setError } = useProblemStore()

  const handleRealSolve = async () => {
    if (!currentProblem) {
      setApiError('请先在步骤1设置问题')
      return
    }

    try {
      setIsProcessing(true)
      setApiError(null)
      setApiResponse(null)
      
      console.log('开始真实解题API调用...')
      
      // 尝试调用真实的解题API
      const response = await fetch('/api/solve', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          problem: currentProblem,
          strategy: selectedStrategy
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const result = await response.json()
      console.log('解题API响应:', result)
      
      // 先保存原始响应，不立即设置到Store
      setApiResponse(result)
      
    } catch (error) {
      console.error('真实解题API失败:', error)
      setApiError(`解题API调用失败: ${String(error)}`)
    } finally {
      setIsProcessing(false)
    }
  }

  const handleApplyResult = () => {
    try {
      if (apiResponse) {
        setLoading(true)
        setSolveResult(apiResponse)
        setApiError(null)
      }
    } catch (error) {
      setApiError(`应用结果失败: ${String(error)}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div style={{ background: '#f8d7da', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
        <h2>⚠️ 步骤3: 测试真实解题API</h2>
        <p>分步安全测试：先获取API响应，再应用到Store</p>
      </div>
      
      <div style={{ display: 'flex', gap: '10px', margin: '15px 0' }}>
        <button 
          onClick={handleRealSolve}
          disabled={isProcessing}
          style={{ 
            padding: '10px 20px', 
            backgroundColor: isProcessing ? '#6c757d' : '#dc3545', 
            color: 'white', 
            border: 'none', 
            borderRadius: '4px' 
          }}
        >
          {isProcessing ? '调用中...' : '调用真实解题API'}
        </button>
        
        {apiResponse && (
          <button 
            onClick={handleApplyResult}
            style={{ 
              padding: '10px 20px', 
              backgroundColor: '#28a745', 
              color: 'white', 
              border: 'none', 
              borderRadius: '4px' 
            }}
          >
            应用结果到Store
          </button>
        )}
      </div>
      
      {apiError && (
        <div style={{ background: '#f8d7da', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
          <h3>❌ 错误:</h3>
          <p>{apiError}</p>
        </div>
      )}
      
      {apiResponse && (
        <div style={{ background: '#d4edda', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
          <h3>✅ API响应成功</h3>
          <p>答案: {apiResponse.answer || '无'}</p>
          <p>策略: {apiResponse.strategy_used || apiResponse.strategy || '无'}</p>
          <p>置信度: {apiResponse.confidence || '无'}</p>
          <details>
            <summary>查看完整响应</summary>
            <pre style={{ fontSize: '12px', overflow: 'auto', maxHeight: '200px' }}>
              {JSON.stringify(apiResponse, null, 2)}
            </pre>
          </details>
        </div>
      )}
      
      <div style={{ background: '#f8f9fa', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
        <h3>现在可以安全测试API响应处理过程</h3>
      </div>
    </div>
  )
}

export default SolveDebugger