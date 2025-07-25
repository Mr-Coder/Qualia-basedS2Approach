import React, { useState } from 'react'
import { useProblemStore } from '@/stores/problemStore'
import Layout from '@/components/layout/Layout'
import SmartSolver from '@/components/features/SmartSolver'
import HistoryPanel from '@/components/features/HistoryPanel'

type TabType = 'smart' | 'knowledge' | 'learning' | 'error' | 'strategy' | 'diagram'

const SafeAppTest: React.FC = () => {
  const [step, setStep] = useState(1)

  if (step === 1) {
    return (
      <div style={{ padding: '20px' }}>
        <h1>🔍 安全App测试</h1>
        <p>逐步测试App组件的渲染问题</p>
        <button onClick={() => setStep(2)}>测试1: Layout + 简单内容</button>
      </div>
    )
  }

  if (step === 2) return <Test1SimpleLayoutContent setStep={setStep} />
  if (step === 3) return <Test2SmartSolverOnly setStep={setStep} />
  if (step === 4) return <Test3SmartSolverWithHistory setStep={setStep} />
  if (step === 5) return <Test4WithTailwindClasses setStep={setStep} />

  return <div>测试完成</div>
}

// 测试1: Layout + 简单内容
const Test1SimpleLayoutContent: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  try {
    const [activeTab, setActiveTab] = useState<TabType>('smart')

    return (
      <Layout activeTab={activeTab} setActiveTab={setActiveTab}>
        <div style={{ padding: '20px', background: '#d4edda' }}>
          <h1>✅ 测试1成功: Layout + 简单内容</h1>
          <button onClick={() => setStep(3)}>测试2: 单独SmartSolver</button>
          <button onClick={() => setStep(1)}>返回</button>
        </div>
      </Layout>
    )
  } catch (error) {
    return (
      <div style={{ padding: '20px', background: '#f8d7da' }}>
        <h1>❌ 测试1失败</h1>
        <p>错误: {String(error)}</p>
        <button onClick={() => setStep(1)}>返回</button>
      </div>
    )
  }
}

// 测试2: 单独SmartSolver
const Test2SmartSolverOnly: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  try {
    const [activeTab, setActiveTab] = useState<TabType>('smart')

    return (
      <Layout activeTab={activeTab} setActiveTab={setActiveTab}>
        <div style={{ padding: '20px' }}>
          <h1>🔍 测试2: 单独SmartSolver</h1>
          <div style={{ border: '2px solid #007bff', padding: '10px', margin: '10px 0' }}>
            <SmartSolver />
          </div>
          <div style={{ background: '#d4edda', padding: '10px' }}>
            <h2>✅ 测试2成功: SmartSolver渲染正常</h2>
            <button onClick={() => setStep(4)}>测试3: SmartSolver + HistoryPanel</button>
            <button onClick={() => setStep(2)}>返回</button>
          </div>
        </div>
      </Layout>
    )
  } catch (error) {
    return (
      <div style={{ padding: '20px', background: '#f8d7da' }}>
        <h1>❌ 测试2失败: SmartSolver问题</h1>
        <p>错误: {String(error)}</p>
        <pre style={{ fontSize: '12px', overflow: 'auto' }}>
          {error instanceof Error ? error.stack : ''}
        </pre>
        <button onClick={() => setStep(2)}>返回测试1</button>
      </div>
    )
  }
}

// 测试3: SmartSolver + HistoryPanel (不使用CSS类)
const Test3SmartSolverWithHistory: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  try {
    const [activeTab, setActiveTab] = useState<TabType>('smart')

    return (
      <Layout activeTab={activeTab} setActiveTab={setActiveTab}>
        <div style={{ padding: '20px' }}>
          <h1>🔍 测试3: SmartSolver + HistoryPanel</h1>
          
          <div style={{ marginBottom: '32px' }}>
            <h2>SmartSolver组件:</h2>
            <div style={{ border: '1px solid #ccc', padding: '10px' }}>
              <SmartSolver />
            </div>
          </div>

          <div style={{ maxWidth: '384px', margin: '0 auto' }}>
            <h2>HistoryPanel组件:</h2>
            <div style={{ border: '1px solid #ccc', padding: '10px' }}>
              <HistoryPanel />
            </div>
          </div>

          <div style={{ background: '#d4edda', padding: '10px', marginTop: '20px' }}>
            <h2>✅ 测试3成功</h2>
            <button onClick={() => setStep(5)}>测试4: 使用CSS类</button>
            <button onClick={() => setStep(3)}>返回</button>
          </div>
        </div>
      </Layout>
    )
  } catch (error) {
    return (
      <div style={{ padding: '20px', background: '#f8d7da' }}>
        <h1>❌ 测试3失败: SmartSolver + HistoryPanel问题</h1>
        <p>错误: {String(error)}</p>
        <pre style={{ fontSize: '12px', overflow: 'auto' }}>
          {error instanceof Error ? error.stack : ''}
        </pre>
        <button onClick={() => setStep(3)}>返回测试2</button>
      </div>
    )
  }
}

// 测试4: 使用Tailwind CSS类
const Test4WithTailwindClasses: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  try {
    const [activeTab, setActiveTab] = useState<TabType>('smart')

    return (
      <Layout activeTab={activeTab} setActiveTab={setActiveTab}>
        <div className="space-y-8">
          <SmartSolver />
          <div className="max-w-md mx-auto">
            <HistoryPanel />
          </div>
          <div style={{ background: '#d4edda', padding: '10px' }}>
            <h2>✅ 测试4成功: Tailwind CSS类正常</h2>
            <p>如果你能看到这个，说明问题不在CSS类</p>
            <button onClick={() => setStep(1)}>重新开始</button>
          </div>
        </div>
      </Layout>
    )
  } catch (error) {
    return (
      <div style={{ padding: '20px', background: '#f8d7da' }}>
        <h1>❌ 测试4失败: CSS类问题</h1>
        <p>错误: {String(error)}</p>
        <p>可能是Tailwind CSS配置或类名问题</p>
        <pre style={{ fontSize: '12px', overflow: 'auto' }}>
          {error instanceof Error ? error.stack : ''}
        </pre>
        <button onClick={() => setStep(4)}>返回测试3</button>
      </div>
    )
  }
}

export default SafeAppTest