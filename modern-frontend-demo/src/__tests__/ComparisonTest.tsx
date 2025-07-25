import React, { useState } from 'react'
import Layout from '@/components/layout/Layout'
import HistoryPanel from '@/components/features/HistoryPanel'
import HistoryPanelFixed from '@/components/features/HistoryPanelFixed'
import { useProblemStore } from '@/stores/problemStore'

const ComparisonTest: React.FC = () => {
  const [testStep, setTestStep] = useState(1)
  const addToHistory = useProblemStore(state => state.addToHistory)
  const clearHistory = useProblemStore(state => state.clearHistory)

  const addTestData = () => {
    const testEntry = {
      id: `test-${Date.now()}`,
      problem: '测试问题：小明有3个苹果，小红有2个苹果，一共有多少个苹果？',
      answer: '5个苹果',
      strategy: 'cot' as const,
      timestamp: new Date(),
      confidence: 0.95
    }
    addToHistory(testEntry)
  }

  return (
    <Layout activeTab="smart" setActiveTab={() => {}}>
      <div style={{ padding: '20px' }}>
        <h1>🔍 HistoryPanel 问题对比测试</h1>
        
        <div style={{ background: '#f0f8ff', padding: '15px', margin: '15px 0', border: '1px solid #0066cc' }}>
          <h3>测试控制台</h3>
          <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
            <button 
              onClick={() => setTestStep(1)}
              style={{ padding: '8px 16px', backgroundColor: testStep === 1 ? '#0066cc' : '#ccc', color: 'white', border: 'none', borderRadius: '4px' }}
            >
              测试1: 修复版HistoryPanel
            </button>
            <button 
              onClick={() => setTestStep(2)}
              style={{ padding: '8px 16px', backgroundColor: testStep === 2 ? '#0066cc' : '#ccc', color: 'white', border: 'none', borderRadius: '4px' }}
            >
              测试2: 原版HistoryPanel
            </button>
            <button 
              onClick={addTestData}
              style={{ padding: '8px 16px', backgroundColor: '#28a745', color: 'white', border: 'none', borderRadius: '4px' }}
            >
              添加测试数据
            </button>
            <button 
              onClick={clearHistory}
              style={{ padding: '8px 16px', backgroundColor: '#dc3545', color: 'white', border: 'none', borderRadius: '4px' }}
            >
              清空历史
            </button>
          </div>
        </div>

        {testStep === 1 && <TestFixedVersion />}
        {testStep === 2 && <TestOriginalVersion />}
      </div>
    </Layout>
  )
}

const TestFixedVersion: React.FC = () => {
  try {
    return (
      <div>
        <div style={{ background: '#d4edda', padding: '10px', margin: '10px 0' }}>
          <h2>✅ 测试修复版 HistoryPanel</h2>
          <p>这个版本有安全的时间戳处理</p>
        </div>
        
        <HistoryPanelFixed />
        
        <div style={{ background: '#d4edda', padding: '10px', margin: '10px 0' }}>
          <h2>🎉 修复版渲染成功！</h2>
          <p>如果你能看到这个，说明修复版没有问题</p>
        </div>
      </div>
    )
  } catch (error) {
    return (
      <div style={{ background: '#f8d7da', padding: '10px' }}>
        <h2>❌ 修复版也失败了</h2>
        <p>错误: {String(error)}</p>
        <pre style={{ fontSize: '12px', overflow: 'auto' }}>
          {error instanceof Error ? error.stack : String(error)}
        </pre>
      </div>
    )
  }
}

const TestOriginalVersion: React.FC = () => {
  try {
    return (
      <div>
        <div style={{ background: '#fff3cd', padding: '10px', margin: '10px 0' }}>
          <h2>⚠️ 测试原版 HistoryPanel</h2>
          <p>这可能会导致白屏</p>
        </div>
        
        <HistoryPanel />
        
        <div style={{ background: '#d4edda', padding: '10px', margin: '10px 0' }}>
          <h2>✅ 原版竟然也成功了？</h2>
          <p>如果你能看到这个，说明原版也没问题，问题可能在别处</p>
        </div>
      </div>
    )
  } catch (error) {
    return (
      <div style={{ background: '#f8d7da', padding: '10px' }}>
        <h2>❌ 原版失败了 (如预期)</h2>
        <p>错误: {String(error)}</p>
        <pre style={{ fontSize: '12px', overflow: 'auto' }}>
          {error instanceof Error ? error.stack : String(error)}
        </pre>
      </div>
    )
  }
}

export default ComparisonTest