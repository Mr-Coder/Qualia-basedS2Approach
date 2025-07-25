import React, { useState } from 'react'

// 第一步：测试基本的React和状态
const DebugApp: React.FC = () => {
  const [step, setStep] = useState(1)
  const [error, setError] = useState<string | null>(null)

  if (step === 1) {
    return (
      <div style={{ padding: '20px' }}>
        <h1>🔍 步骤1: 基本React + useState 测试</h1>
        <p>当前步骤: {step}</p>
        <button onClick={() => setStep(2)}>下一步: 测试Store导入</button>
      </div>
    )
  }

  return <DebugStep2 step={step} setStep={setStep} />
}

// 步骤2：测试store导入
const DebugStep2: React.FC<{step: number, setStep: (n: number) => void}> = ({ step, setStep }) => {
  if (step === 2) {
    return <TestStoreImport setStep={setStep} />
  }
  if (step === 3) {
    return <TestLayoutImport setStep={setStep} />
  }
  if (step === 4) {
    return <TestSmartSolverImport setStep={setStep} />
  }
  return <div>未知步骤</div>
}

// 测试Store导入组件
const TestStoreImport: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  try {
    const { useProblemStore } = require('@/stores/problemStore')
    return (
      <div style={{ padding: '20px' }}>
        <h1>✅ 步骤2: Store导入成功</h1>
        <button onClick={() => setStep(3)}>下一步: 测试Layout组件</button>
      </div>
    )
  } catch (error) {
    return (
      <div style={{ padding: '20px', background: '#ffebee' }}>
        <h1>❌ 步骤2: Store导入失败</h1>
        <p>错误: {String(error)}</p>
        <button onClick={() => setStep(1)}>返回步骤1</button>
      </div>
    )
  }
}

const TestLayoutImport: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  return (
    <div style={{ padding: '20px', background: '#fff3cd' }}>
      <h1>🔍 步骤3: 正在测试Layout组件导入...</h1>
      <button onClick={() => setStep(2)}>返回步骤2</button>
    </div>
  )
}

const TestSmartSolverImport: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  return (
    <div style={{ padding: '20px', background: '#fff3cd' }}>
      <h1>🔍 步骤4: 正在测试SmartSolver组件导入...</h1>
      <button onClick={() => setStep(3)}>返回步骤3</button>
    </div>
  )
}

export default DebugApp