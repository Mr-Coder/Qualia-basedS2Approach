import React, { useState } from 'react'

// 直接测试static imports
import { useProblemStore } from '@/stores/problemStore'

const QuickTestApp: React.FC = () => {
  const [step, setStep] = useState(1)

  if (step === 1) {
    return (
      <div style={{ padding: '20px' }}>
        <h1>✅ 基础React正常</h1>
        <button onClick={() => setStep(2)}>测试Store</button>
      </div>
    )
  }

  if (step === 2) {
    return <TestStore setStep={setStep} />
  }

  if (step === 3) {
    return <TestLayout setStep={setStep} />
  }

  return <div>测试完成</div>
}

const TestStore: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  try {
    const store = useProblemStore()
    return (
      <div style={{ padding: '20px', background: '#d4edda' }}>
        <h1>✅ Store测试成功</h1>
        <p>当前问题: {store.currentProblem || '无'}</p>
        <button onClick={() => setStep(3)}>测试Layout</button>
        <button onClick={() => setStep(1)}>返回</button>
      </div>
    )
  } catch (error) {
    return (
      <div style={{ padding: '20px', background: '#f8d7da' }}>
        <h1>❌ Store测试失败</h1>
        <p>错误: {String(error)}</p>
        <button onClick={() => setStep(1)}>返回</button>
      </div>
    )
  }
}

const TestLayout: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  return (
    <div style={{ padding: '20px' }}>
      <h1>🔍 Layout测试</h1>
      <TestLayoutContent setStep={setStep} />
    </div>
  )
}

const TestLayoutContent: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  return (
    <div style={{ background: '#d1ecf1', padding: '10px' }}>
      <h2>Layout组件正在加载...</h2>
      <button onClick={() => setStep(2)}>返回Store测试</button>
    </div>
  )
}

export default QuickTestApp