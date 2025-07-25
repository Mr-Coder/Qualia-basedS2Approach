import React, { useState } from 'react'
import { useProblemStore } from '@/stores/problemStore'
import Layout from '@/components/layout/Layout'
import SmartSolver from '@/components/features/SmartSolver'
import HistoryPanel from '@/components/features/HistoryPanel'

type TabType = 'smart' | 'knowledge' | 'learning' | 'error' | 'strategy' | 'diagram'

const AppRenderTest: React.FC = () => {
  const [step, setStep] = useState(1)

  if (step === 1) {
    return (
      <div style={{ padding: '20px' }}>
        <h1>🔍 App完整渲染测试</h1>
        <p>模拟App.tsx的完整渲染流程</p>
        <button onClick={() => setStep(2)}>测试基础App结构</button>
      </div>
    )
  }

  if (step === 2) return <TestBasicAppStructure setStep={setStep} />
  if (step === 3) return <TestWithLayout setStep={setStep} />
  if (step === 4) return <TestWithSmartSolver setStep={setStep} />
  if (step === 5) return <TestCompleteApp setStep={setStep} />

  return <div>测试完成</div>
}

// 测试基础App结构
const TestBasicAppStructure: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  try {
    const [activeTab, setActiveTab] = useState<TabType>('smart')
    const { solveResult, currentProblem } = useProblemStore()

    return (
      <div style={{ padding: '20px', background: '#d4edda' }}>
        <h1>✅ 基础App结构正常</h1>
        <p>useState和useProblemStore正常工作</p>
        <p>当前Tab: {activeTab}</p>
        <p>当前问题: {currentProblem || '无'}</p>
        <button onClick={() => setStep(3)}>测试Layout包装</button>
        <button onClick={() => setStep(1)}>返回</button>
      </div>
    )
  } catch (error) {
    return (
      <div style={{ padding: '20px', background: '#f8d7da' }}>
        <h1>❌ 基础App结构失败</h1>
        <p>错误: {String(error)}</p>
        <button onClick={() => setStep(1)}>返回</button>
      </div>
    )
  }
}

// 测试Layout包装
const TestWithLayout: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  try {
    const [activeTab, setActiveTab] = useState<TabType>('smart')

    return (
      <div style={{ padding: '20px' }}>
        <h1>🔍 测试Layout包装</h1>
        <Layout activeTab={activeTab} setActiveTab={setActiveTab}>
          <div style={{ padding: '20px', background: '#d4edda' }}>
            <h2>✅ Layout包装成功</h2>
            <p>Layout组件正常渲染内容</p>
            <button onClick={() => setStep(4)}>测试SmartSolver渲染</button>
            <button onClick={() => setStep(2)}>返回</button>
          </div>
        </Layout>
      </div>
    )
  } catch (error) {
    return (
      <div style={{ padding: '20px', background: '#f8d7da' }}>
        <h1>❌ Layout包装失败</h1>
        <p>错误: {String(error)}</p>
        <pre style={{ fontSize: '12px', overflow: 'auto' }}>
          {error instanceof Error ? error.stack : ''}
        </pre>
        <button onClick={() => setStep(2)}>返回</button>
      </div>
    )
  }
}

// 测试SmartSolver渲染
const TestWithSmartSolver: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  try {
    const [activeTab, setActiveTab] = useState<TabType>('smart')

    return (
      <div style={{ padding: '20px' }}>
        <h1>🔍 测试SmartSolver渲染</h1>
        <Layout activeTab={activeTab} setActiveTab={setActiveTab}>
          <div style={{ background: '#fff3cd', padding: '10px', margin: '10px 0' }}>
            <p>准备渲染SmartSolver组件...</p>
          </div>
          <SmartSolverRenderTest setStep={setStep} />
        </Layout>
      </div>
    )
  } catch (error) {
    return (
      <div style={{ padding: '20px', background: '#f8d7da' }}>
        <h1>❌ SmartSolver测试失败</h1>
        <p>错误: {String(error)}</p>
        <pre style={{ fontSize: '12px', overflow: 'auto' }}>
          {error instanceof Error ? error.stack : ''}
        </pre>
        <button onClick={() => setStep(3)}>返回Layout测试</button>
      </div>
    )
  }
}

const SmartSolverRenderTest: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  try {
    return (
      <div>
        <div style={{ background: '#d4edda', padding: '10px', margin: '10px 0' }}>
          <h3>✅ 准备渲染SmartSolver</h3>
        </div>
        <SmartSolver />
        <div style={{ background: '#d4edda', padding: '10px', margin: '10px 0' }}>
          <h3>✅ SmartSolver渲染成功！</h3>
          <button onClick={() => setStep(5)}>测试完整App</button>
          <button onClick={() => setStep(3)}>返回</button>
        </div>
      </div>
    )
  } catch (error) {
    return (
      <div style={{ background: '#f8d7da', padding: '10px' }}>
        <h3>❌ SmartSolver渲染失败</h3>
        <p>错误: {String(error)}</p>
        <pre style={{ fontSize: '12px', overflow: 'auto' }}>
          {error instanceof Error ? error.stack : ''}
        </pre>
        <button onClick={() => setStep(3)}>返回</button>
      </div>
    )
  }
}

// 测试完整App
const TestCompleteApp: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  try {
    return (
      <div style={{ padding: '20px' }}>
        <h1>🔍 测试完整App渲染</h1>
        <CompleteAppTest setStep={setStep} />
      </div>
    )
  } catch (error) {
    return (
      <div style={{ padding: '20px', background: '#f8d7da' }}>
        <h1>❌ 完整App测试失败</h1>
        <p>错误: {String(error)}</p>
        <button onClick={() => setStep(4)}>返回SmartSolver测试</button>
      </div>
    )
  }
}

const CompleteAppTest: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  const [activeTab, setActiveTab] = useState<TabType>('smart')
  const { solveResult, currentProblem } = useProblemStore()

  const renderContent = () => {
    switch (activeTab) {
      case 'smart':
        return (
          <div className="space-y-8">
            <SmartSolver />
            <div className="max-w-md mx-auto">
              <HistoryPanel />
            </div>
          </div>
        )
      default:
        return <SmartSolver />
    }
  }

  return (
    <div>
      <div style={{ background: '#d4edda', padding: '10px', margin: '10px 0' }}>
        <h3>✅ 完整App结构创建成功</h3>
        <p>准备渲染Layout + 内容...</p>
      </div>
      
      <Layout activeTab={activeTab} setActiveTab={setActiveTab}>
        {renderContent()}
      </Layout>
      
      <div style={{ background: '#d4edda', padding: '10px', margin: '10px 0' }}>
        <h3>🎉 完整App渲染成功！</h3>
        <p>如果你能看到这个，说明App.tsx的逻辑本身没有问题</p>
        <button onClick={() => setStep(1)}>重新开始</button>
      </div>
    </div>
  )
}

export default AppRenderTest