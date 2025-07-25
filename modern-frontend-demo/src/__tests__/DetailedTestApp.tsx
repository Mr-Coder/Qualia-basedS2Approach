import React, { useState } from 'react'
import { useProblemStore } from '@/stores/problemStore'

const DetailedTestApp: React.FC = () => {
  const [step, setStep] = useState(1)

  if (step === 1) {
    return (
      <div style={{ padding: '20px' }}>
        <h1>✅ 基础React + Store正常</h1>
        <button onClick={() => setStep(2)}>测试Layout导入</button>
      </div>
    )
  }

  if (step === 2) {
    return <TestLayoutImport setStep={setStep} />
  }

  if (step === 3) {
    return <TestLayoutRender setStep={setStep} />
  }

  return <div>测试完成</div>
}

// 测试Layout导入
const TestLayoutImport: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  return (
    <div style={{ padding: '20px', background: '#d4edda' }}>
      <h1>✅ Layout导入测试</h1>
      <p>准备测试Layout组件渲染...</p>
      <button onClick={() => setStep(3)}>测试Layout渲染</button>
      <button onClick={() => setStep(1)}>返回</button>
    </div>
  )
}

// 测试Layout渲染
const TestLayoutRender: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  return (
    <div style={{ padding: '20px' }}>
      <h1>🔍 测试Layout渲染</h1>
      <LayoutRenderTest setStep={setStep} />
    </div>
  )
}

const LayoutRenderTest: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  try {
    return <ActualLayoutTest setStep={setStep} />
  } catch (error) {
    return (
      <div style={{ background: '#f8d7da', padding: '10px' }}>
        <h2>❌ Layout渲染失败</h2>
        <p>错误: {String(error)}</p>
        <pre style={{ fontSize: '12px', overflow: 'auto' }}>
          {error instanceof Error ? error.stack : ''}
        </pre>
        <button onClick={() => setStep(2)}>返回Layout导入测试</button>
      </div>
    )
  }
}

// 实际Layout组件测试
const ActualLayoutTest: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  try {
    // 使用动态import
    const [layoutLoaded, setLayoutLoaded] = useState(false)
    const [layoutError, setLayoutError] = useState<string | null>(null)
    
    React.useEffect(() => {
      import('@/components/layout/Layout')
        .then(() => {
          setLayoutLoaded(true)
        })
        .catch((error) => {
          setLayoutError(String(error))
        })
    }, [])
    
    if (layoutError) {
      return (
        <div style={{ background: '#f8d7da', padding: '10px' }}>
          <h2>❌ Layout动态导入失败</h2>
          <p>错误: {layoutError}</p>
          <button onClick={() => setStep(2)}>返回</button>
        </div>
      )
    }
    
    if (!layoutLoaded) {
      return (
        <div style={{ background: '#fff3cd', padding: '10px' }}>
          <h2>⏳ Layout组件加载中...</h2>
          <button onClick={() => setStep(2)}>返回</button>
        </div>
      )
    }
    
    return (
      <div style={{ background: '#d4edda', padding: '10px' }}>
        <h2>✅ Layout组件加载成功</h2>
        <p>Layout组件可以正常导入</p>
        <button onClick={() => setStep(2)}>返回</button>
        <button onClick={() => setStep(1)}>重新开始</button>
      </div>
    )
  } catch (error) {
    return (
      <div style={{ background: '#f8d7da', padding: '10px' }}>
        <h2>❌ Layout测试异常</h2>
        <p>错误: {String(error)}</p>
        <button onClick={() => setStep(2)}>返回</button>
      </div>
    )
  }
}

export default DetailedTestApp