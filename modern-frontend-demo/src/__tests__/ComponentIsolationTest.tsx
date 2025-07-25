import React, { useState } from 'react'
import Layout from '@/components/layout/Layout'
import SmartSolver from '@/components/features/SmartSolver'
import HistoryPanel from '@/components/features/HistoryPanel'

type TabType = 'smart' | 'knowledge' | 'learning' | 'error' | 'strategy' | 'diagram'

const ComponentIsolationTest: React.FC = () => {
  const [step, setStep] = useState(1)

  if (step === 1) {
    return (
      <div style={{ padding: '20px' }}>
        <h1>🔍 组件隔离测试</h1>
        <p>分别测试SmartSolver和HistoryPanel</p>
        <button onClick={() => setStep(2)}>测试: 仅SmartSolver</button>
      </div>
    )
  }

  if (step === 2) return <TestSmartSolverOnly setStep={setStep} />
  if (step === 3) return <TestHistoryPanelOnly setStep={setStep} />
  if (step === 4) return <TestBothSeparately setStep={setStep} />
  if (step === 5) return <TestBothTogether setStep={setStep} />

  return <div>测试完成</div>
}

// 仅测试SmartSolver
const TestSmartSolverOnly: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  try {
    const [activeTab, setActiveTab] = useState<TabType>('smart')

    return (
      <Layout activeTab={activeTab} setActiveTab={setActiveTab}>
        <div style={{ padding: '20px' }}>
          <h1>🔍 仅测试SmartSolver</h1>
          <div style={{ background: '#fff3cd', padding: '10px', margin: '10px 0' }}>
            <p>准备渲染SmartSolver...</p>
          </div>
          
          <SmartSolver />
          
          <div style={{ background: '#d4edda', padding: '10px', margin: '10px 0' }}>
            <h2>✅ SmartSolver单独渲染成功</h2>
            <button onClick={() => setStep(3)}>测试: 仅HistoryPanel</button>
            <button onClick={() => setStep(1)}>返回</button>
          </div>
        </div>
      </Layout>
    )
  } catch (error) {
    return (
      <div style={{ padding: '20px', background: '#f8d7da' }}>
        <h1>❌ SmartSolver渲染失败</h1>
        <p>错误: {String(error)}</p>
        <pre style={{ fontSize: '12px', overflow: 'auto' }}>
          {error instanceof Error ? error.stack : ''}
        </pre>
        <button onClick={() => setStep(1)}>返回</button>
      </div>
    )
  }
}

// 仅测试HistoryPanel
const TestHistoryPanelOnly: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  try {
    const [activeTab, setActiveTab] = useState<TabType>('smart')

    return (
      <Layout activeTab={activeTab} setActiveTab={setActiveTab}>
        <div style={{ padding: '20px' }}>
          <h1>🔍 仅测试HistoryPanel</h1>
          <div style={{ background: '#fff3cd', padding: '10px', margin: '10px 0' }}>
            <p>准备渲染HistoryPanel...</p>
          </div>
          
          <HistoryPanel />
          
          <div style={{ background: '#d4edda', padding: '10px', margin: '10px 0' }}>
            <h2>✅ HistoryPanel单独渲染成功</h2>
            <button onClick={() => setStep(4)}>测试: 两个组件分开渲染</button>
            <button onClick={() => setStep(2)}>返回SmartSolver测试</button>
          </div>
        </div>
      </Layout>
    )
  } catch (error) {
    return (
      <div style={{ padding: '20px', background: '#f8d7da' }}>
        <h1>❌ HistoryPanel渲染失败</h1>
        <p>错误: {String(error)}</p>
        <pre style={{ fontSize: '12px', overflow: 'auto' }}>
          {error instanceof Error ? error.stack : ''}
        </pre>
        <button onClick={() => setStep(2)}>返回SmartSolver测试</button>
      </div>
    )
  }
}

// 测试两个组件分开渲染
const TestBothSeparately: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  try {
    const [activeTab, setActiveTab] = useState<TabType>('smart')

    return (
      <Layout activeTab={activeTab} setActiveTab={setActiveTab}>
        <div style={{ padding: '20px' }}>
          <h1>🔍 两个组件分开渲染</h1>
          
          <div style={{ marginBottom: '40px' }}>
            <h2>第一个: SmartSolver</h2>
            <div style={{ border: '2px solid #007bff', padding: '10px' }}>
              <SmartSolver />
            </div>
          </div>

          <div style={{ marginBottom: '40px' }}>
            <h2>第二个: HistoryPanel</h2>
            <div style={{ border: '2px solid #28a745', padding: '10px' }}>
              <HistoryPanel />
            </div>
          </div>

          <div style={{ background: '#d4edda', padding: '10px' }}>
            <h2>✅ 两个组件分开渲染成功</h2>
            <button onClick={() => setStep(5)}>测试: 两个组件紧密结合</button>
            <button onClick={() => setStep(3)}>返回</button>
          </div>
        </div>
      </Layout>
    )
  } catch (error) {
    return (
      <div style={{ padding: '20px', background: '#f8d7da' }}>
        <h1>❌ 分开渲染失败</h1>
        <p>错误: {String(error)}</p>
        <pre style={{ fontSize: '12px', overflow: 'auto' }}>
          {error instanceof Error ? error.stack : ''}
        </pre>
        <button onClick={() => setStep(3)}>返回HistoryPanel测试</button>
      </div>
    )
  }
}

// 测试两个组件紧密结合 (模拟原始App.tsx的结构)
const TestBothTogether: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  try {
    const [activeTab, setActiveTab] = useState<TabType>('smart')

    return (
      <Layout activeTab={activeTab} setActiveTab={setActiveTab}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
          <SmartSolver />
          <div style={{ maxWidth: '384px', margin: '0 auto' }}>
            <HistoryPanel />
          </div>
          <div style={{ background: '#d4edda', padding: '10px' }}>
            <h2>✅ 紧密结合渲染成功</h2>
            <p>如果你能看到这个，说明问题在其他地方</p>
            <button onClick={() => setStep(1)}>重新开始</button>
          </div>
        </div>
      </Layout>
    )
  } catch (error) {
    return (
      <div style={{ padding: '20px', background: '#f8d7da' }}>
        <h1>❌ 紧密结合渲染失败</h1>
        <p>这就是问题所在！</p>
        <p>错误: {String(error)}</p>
        <pre style={{ fontSize: '12px', overflow: 'auto' }}>
          {error instanceof Error ? error.stack : ''}
        </pre>
        <button onClick={() => setStep(4)}>返回分开渲染测试</button>
      </div>
    )
  }
}

export default ComponentIsolationTest