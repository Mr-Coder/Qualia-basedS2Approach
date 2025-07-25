import React, { useState } from 'react'

// 分步导入测试
const StepByStepApp: React.FC = () => {
  const [step, setStep] = useState(1)

  if (step === 1) {
    return (
      <div style={{ padding: '20px' }}>
        <h1>✅ 步骤1: 基础React正常</h1>
        <button onClick={() => setStep(2)}>下一步: 测试Store</button>
      </div>
    )
  }

  if (step === 2) {
    return <Step2Store setStep={setStep} />
  }

  if (step === 3) {
    return <Step3Layout setStep={setStep} />
  }

  if (step === 4) {
    return <Step4SmartSolver setStep={setStep} />
  }

  return <div>未知步骤</div>
}

// 步骤2: 测试Store
const Step2Store: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  try {
    // 这里会触发实际的导入
    return <Step2StoreContent setStep={setStep} />
  } catch (error) {
    return (
      <div style={{ padding: '20px', background: '#f8d7da' }}>
        <h1>❌ 步骤2: Store导入失败</h1>
        <p>错误: {String(error)}</p>
        <button onClick={() => setStep(1)}>返回步骤1</button>
      </div>
    )
  }
}

// 单独的组件来导入store
const Step2StoreContent: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  // 静态导入
  const { useProblemStore } = require('@/stores/problemStore')
  
  return (
    <div style={{ padding: '20px', background: '#d4edda' }}>
      <h1>✅ 步骤2: Store导入成功</h1>
      <button onClick={() => setStep(3)}>下一步: 测试Layout</button>
      <button onClick={() => setStep(1)}>返回步骤1</button>
    </div>
  )
}

// 步骤3: 测试Layout
const Step3Layout: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  try {
    return <Step3LayoutContent setStep={setStep} />
  } catch (error) {
    return (
      <div style={{ padding: '20px', background: '#f8d7da' }}>
        <h1>❌ 步骤3: Layout导入失败</h1>
        <p>错误: {String(error)}</p>
        <button onClick={() => setStep(2)}>返回步骤2</button>
      </div>
    )
  }
}

const Step3LayoutContent: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  const Layout = require('@/components/layout/Layout').default
  
  return (
    <div style={{ padding: '20px', background: '#d4edda' }}>
      <h1>✅ 步骤3: Layout导入成功</h1>
      <button onClick={() => setStep(4)}>下一步: 测试SmartSolver</button>
      <button onClick={() => setStep(2)}>返回步骤2</button>
    </div>
  )
}

// 步骤4: 测试SmartSolver
const Step4SmartSolver: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  try {
    return <Step4SmartSolverContent setStep={setStep} />
  } catch (error) {
    return (
      <div style={{ padding: '20px', background: '#f8d7da' }}>
        <h1>❌ 步骤4: SmartSolver导入失败</h1>
        <p>错误: {String(error)}</p>
        <button onClick={() => setStep(3)}>返回步骤3</button>
      </div>
    )
  }
}

const Step4SmartSolverContent: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  const SmartSolver = require('@/components/features/SmartSolver').default
  
  return (
    <div style={{ padding: '20px', background: '#d4edda' }}>
      <h1>✅ 步骤4: SmartSolver导入成功</h1>
      <SmartSolver />
      <button onClick={() => setStep(3)}>返回步骤3</button>
      <button onClick={() => setStep(1)}>重新开始</button>
    </div>
  )
}

export default StepByStepApp