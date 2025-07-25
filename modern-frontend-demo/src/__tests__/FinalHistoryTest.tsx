import React, { useState } from 'react'
import Layout from '@/components/layout/Layout'
import HistoryPanel from '@/components/features/HistoryPanel'
import HistoryPanelFixed from '@/components/features/HistoryPanelFixed'
import { useProblemStore } from '@/stores/problemStore'

const FinalHistoryTest: React.FC = () => {
  const [testStep, setTestStep] = useState(1)
  const addToHistory = useProblemStore(state => state.addToHistory)
  const clearHistory = useProblemStore(state => state.clearHistory)

  const addTestData = () => {
    const testEntry = {
      id: `test-${Date.now()}`,
      problem: '新测试问题：小王买了5支铅笔，每支2元，一共花了多少钱？',
      answer: '10元',
      strategy: 'got' as const,
      timestamp: new Date(),
      confidence: 0.98
    }
    addToHistory(testEntry)
  }

  return (
    <Layout activeTab="smart" setActiveTab={() => {}}>
      <div style={{ padding: '20px' }}>
        <h1>🔍 HistoryPanel 最终测试</h1>
        
        <div style={{ background: '#d4edda', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
          <h2>✅ 前面所有测试都通过了！</h2>
          <p>React ✓ Layout ✓ Store ✓</p>
          <p>现在测试HistoryPanel - 这是导致白屏的组件</p>
        </div>

        <div style={{ background: '#f0f8ff', padding: '15px', margin: '15px 0', border: '1px solid #0066cc' }}>
          <h3>测试控制台</h3>
          <div style={{ display: 'flex', gap: '10px', marginTop: '10px', flexWrap: 'wrap' }}>
            <button 
              onClick={() => setTestStep(1)}
              style={{ 
                padding: '8px 16px', 
                backgroundColor: testStep === 1 ? '#28a745' : '#6c757d', 
                color: 'white', 
                border: 'none', 
                borderRadius: '4px' 
              }}
            >
              测试1: 修复版HistoryPanel
            </button>
            <button 
              onClick={() => setTestStep(2)}
              style={{ 
                padding: '8px 16px', 
                backgroundColor: testStep === 2 ? '#dc3545' : '#6c757d', 
                color: 'white', 
                border: 'none', 
                borderRadius: '4px' 
              }}
            >
              测试2: 原版HistoryPanel (可能白屏)
            </button>
            <button 
              onClick={addTestData}
              style={{ padding: '8px 16px', backgroundColor: '#007bff', color: 'white', border: 'none', borderRadius: '4px' }}
            >
              添加测试数据
            </button>
            <button 
              onClick={clearHistory}
              style={{ padding: '8px 16px', backgroundColor: '#ffc107', color: 'black', border: 'none', borderRadius: '4px' }}
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
        <div style={{ background: '#d4edda', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
          <h2>🛡️ 测试修复版 HistoryPanel</h2>
          <p>这个版本有安全的timestamp处理，应该不会白屏</p>
        </div>
        
        <div style={{ border: '2px solid #28a745', padding: '15px', borderRadius: '5px' }}>
          <HistoryPanelFixed />
        </div>
        
        <div style={{ background: '#d4edda', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
          <h2>🎉 修复版测试完成！</h2>
          <p>如果你能看到这个消息，说明修复版工作正常</p>
          <p>现在可以尝试点击"测试2"按钮测试原版（小心可能白屏）</p>
        </div>
      </div>
    )
  } catch (error) {
    return (
      <div style={{ background: '#f8d7da', padding: '15px', margin: '15px 0' }}>
        <h2>❌ 修复版也失败了！</h2>
        <p>错误: {String(error)}</p>
        <pre style={{ fontSize: '12px', overflow: 'auto', maxHeight: '200px' }}>
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
        <div style={{ background: '#fff3cd', padding: '15px', margin: '15px 0', borderRadius: '5px', border: '2px solid #ffc107' }}>
          <h2>⚠️ 危险：测试原版 HistoryPanel</h2>
          <p>这是导致白屏的原版组件！</p>
          <p>如果页面变白屏，请刷新浏览器回到测试1</p>
        </div>
        
        <div style={{ border: '2px solid #dc3545', padding: '15px', borderRadius: '5px' }}>
          <p style={{ background: '#fff3cd', padding: '10px', margin: '10px 0' }}>
            正在渲染原版HistoryPanel...
          </p>
          <HistoryPanel />
        </div>
        
        <div style={{ background: '#d4edda', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
          <h2>😲 意外！原版居然也能工作？</h2>
          <p>如果你能看到这个，说明原版在当前条件下没有崩溃</p>
          <p>问题可能在特定的数据条件或环境下才出现</p>
        </div>
      </div>
    )
  } catch (error) {
    return (
      <div style={{ background: '#f8d7da', padding: '15px', margin: '15px 0' }}>
        <h2>❌ 原版失败了（如预期）</h2>
        <p>这就是导致白屏的错误：</p>
        <p><strong>错误:</strong> {String(error)}</p>
        <pre style={{ fontSize: '12px', overflow: 'auto', maxHeight: '200px' }}>
          {error instanceof Error ? error.stack : String(error)}
        </pre>
      </div>
    )
  }
}

export default FinalHistoryTest