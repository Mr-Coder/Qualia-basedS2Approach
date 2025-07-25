import React, { useState } from 'react'
import { useProblemStore } from '@/stores/problemStore'

const FeatureTestApp: React.FC = () => {
  const [step, setStep] = useState(1)

  if (step === 1) {
    return (
      <div style={{ padding: '20px' }}>
        <h1>🔍 Feature组件逐个测试</h1>
        <p>测试App.tsx中导入的所有feature组件</p>
        <button onClick={() => setStep(2)}>测试SmartSolver</button>
      </div>
    )
  }

  if (step === 2) return <TestSmartSolver setStep={setStep} />
  if (step === 3) return <TestHistoryPanel setStep={setStep} />
  if (step === 4) return <TestKnowledgeMap setStep={setStep} />
  if (step === 5) return <TestLearningGuide setStep={setStep} />
  if (step === 6) return <TestErrorAnalysis setStep={setStep} />
  if (step === 7) return <TestStrategyAnalysis setStep={setStep} />
  if (step === 8) return <TestEntityRelationshipDiagram setStep={setStep} />

  return <div>所有测试完成</div>
}

// 测试SmartSolver
const TestSmartSolver: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading')
  const [error, setError] = useState<string>('')

  React.useEffect(() => {
    import('@/components/features/SmartSolver')
      .then(() => setStatus('success'))
      .catch((err) => {
        setStatus('error')
        setError(String(err))
      })
  }, [])

  if (status === 'loading') {
    return (
      <div style={{ padding: '20px', background: '#fff3cd' }}>
        <h1>⏳ 测试SmartSolver导入...</h1>
      </div>
    )
  }

  if (status === 'error') {
    return (
      <div style={{ padding: '20px', background: '#f8d7da' }}>
        <h1>❌ SmartSolver导入失败</h1>
        <p>错误: {error}</p>
        <button onClick={() => setStep(1)}>返回</button>
      </div>
    )
  }

  return (
    <div style={{ padding: '20px', background: '#d4edda' }}>
      <h1>✅ SmartSolver导入成功</h1>
      <button onClick={() => setStep(3)}>测试HistoryPanel</button>
      <button onClick={() => setStep(1)}>返回</button>
    </div>
  )
}

// 测试HistoryPanel
const TestHistoryPanel: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading')
  const [error, setError] = useState<string>('')

  React.useEffect(() => {
    import('@/components/features/HistoryPanel')
      .then(() => setStatus('success'))
      .catch((err) => {
        setStatus('error')
        setError(String(err))
      })
  }, [])

  if (status === 'loading') {
    return (
      <div style={{ padding: '20px', background: '#fff3cd' }}>
        <h1>⏳ 测试HistoryPanel导入...</h1>
      </div>
    )
  }

  if (status === 'error') {
    return (
      <div style={{ padding: '20px', background: '#f8d7da' }}>
        <h1>❌ HistoryPanel导入失败</h1>
        <p>错误: {error}</p>
        <button onClick={() => setStep(2)}>返回SmartSolver</button>
      </div>
    )
  }

  return (
    <div style={{ padding: '20px', background: '#d4edda' }}>
      <h1>✅ HistoryPanel导入成功</h1>
      <button onClick={() => setStep(4)}>测试KnowledgeMap</button>
      <button onClick={() => setStep(2)}>返回</button>
    </div>
  )
}

// 测试KnowledgeMap
const TestKnowledgeMap: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading')
  const [error, setError] = useState<string>('')

  React.useEffect(() => {
    import('@/components/features/KnowledgeMap')
      .then(() => setStatus('success'))
      .catch((err) => {
        setStatus('error')
        setError(String(err))
      })
  }, [])

  if (status === 'loading') {
    return (
      <div style={{ padding: '20px', background: '#fff3cd' }}>
        <h1>⏳ 测试KnowledgeMap导入...</h1>
      </div>
    )
  }

  if (status === 'error') {
    return (
      <div style={{ padding: '20px', background: '#f8d7da' }}>
        <h1>❌ KnowledgeMap导入失败</h1>
        <p>错误: {error}</p>
        <button onClick={() => setStep(3)}>返回HistoryPanel</button>
      </div>
    )
  }

  return (
    <div style={{ padding: '20px', background: '#d4edda' }}>
      <h1>✅ KnowledgeMap导入成功</h1>
      <button onClick={() => setStep(5)}>测试LearningGuide</button>
      <button onClick={() => setStep(3)}>返回</button>
    </div>
  )
}

// 测试LearningGuide
const TestLearningGuide: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading')
  const [error, setError] = useState<string>('')

  React.useEffect(() => {
    import('@/components/features/LearningGuide')
      .then(() => setStatus('success'))
      .catch((err) => {
        setStatus('error')
        setError(String(err))
      })
  }, [])

  if (status === 'loading') {
    return (
      <div style={{ padding: '20px', background: '#fff3cd' }}>
        <h1>⏳ 测试LearningGuide导入...</h1>
      </div>
    )
  }

  if (status === 'error') {
    return (
      <div style={{ padding: '20px', background: '#f8d7da' }}>
        <h1>❌ LearningGuide导入失败</h1>
        <p>错误: {error}</p>
        <button onClick={() => setStep(4)}>返回KnowledgeMap</button>
      </div>
    )
  }

  return (
    <div style={{ padding: '20px', background: '#d4edda' }}>
      <h1>✅ LearningGuide导入成功</h1>
      <button onClick={() => setStep(6)}>测试ErrorAnalysis</button>
      <button onClick={() => setStep(4)}>返回</button>
    </div>
  )
}

// 测试ErrorAnalysis
const TestErrorAnalysis: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading')
  const [error, setError] = useState<string>('')

  React.useEffect(() => {
    import('@/components/features/ErrorAnalysis')
      .then(() => setStatus('success'))
      .catch((err) => {
        setStatus('error')
        setError(String(err))
      })
  }, [])

  if (status === 'loading') {
    return (
      <div style={{ padding: '20px', background: '#fff3cd' }}>
        <h1>⏳ 测试ErrorAnalysis导入...</h1>
      </div>
    )
  }

  if (status === 'error') {
    return (
      <div style={{ padding: '20px', background: '#f8d7da' }}>
        <h1>❌ ErrorAnalysis导入失败</h1>
        <p>错误: {error}</p>
        <button onClick={() => setStep(5)}>返回LearningGuide</button>
      </div>
    )
  }

  return (
    <div style={{ padding: '20px', background: '#d4edda' }}>
      <h1>✅ ErrorAnalysis导入成功</h1>
      <button onClick={() => setStep(7)}>测试StrategyAnalysis</button>
      <button onClick={() => setStep(5)}>返回</button>
    </div>
  )
}

// 测试StrategyAnalysis
const TestStrategyAnalysis: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading')
  const [error, setError] = useState<string>('')

  React.useEffect(() => {
    import('@/components/features/StrategyAnalysis')
      .then(() => setStatus('success'))
      .catch((err) => {
        setStatus('error')
        setError(String(err))
      })
  }, [])

  if (status === 'loading') {
    return (
      <div style={{ padding: '20px', background: '#fff3cd' }}>
        <h1>⏳ 测试StrategyAnalysis导入...</h1>
      </div>
    )
  }

  if (status === 'error') {
    return (
      <div style={{ padding: '20px', background: '#f8d7da' }}>
        <h1>❌ StrategyAnalysis导入失败</h1>
        <p>错误: {error}</p>
        <button onClick={() => setStep(6)}>返回ErrorAnalysis</button>
      </div>
    )
  }

  return (
    <div style={{ padding: '20px', background: '#d4edda' }}>
      <h1>✅ StrategyAnalysis导入成功</h1>
      <button onClick={() => setStep(8)}>测试EntityRelationshipDiagram</button>
      <button onClick={() => setStep(6)}>返回</button>
    </div>
  )
}

// 测试EntityRelationshipDiagram
const TestEntityRelationshipDiagram: React.FC<{setStep: (n: number) => void}> = ({ setStep }) => {
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading')
  const [error, setError] = useState<string>('')

  React.useEffect(() => {
    import('@/components/features/EntityRelationshipDiagram')
      .then(() => setStatus('success'))
      .catch((err) => {
        setStatus('error')
        setError(String(err))
      })
  }, [])

  if (status === 'loading') {
    return (
      <div style={{ padding: '20px', background: '#fff3cd' }}>
        <h1>⏳ 测试EntityRelationshipDiagram导入...</h1>
      </div>
    )
  }

  if (status === 'error') {
    return (
      <div style={{ padding: '20px', background: '#f8d7da' }}>
        <h1>❌ EntityRelationshipDiagram导入失败</h1>
        <p>错误: {error}</p>
        <button onClick={() => setStep(7)}>返回StrategyAnalysis</button>
      </div>
    )
  }

  return (
    <div style={{ padding: '20px', background: '#d4edda' }}>
      <h1>✅ EntityRelationshipDiagram导入成功</h1>
      <p>所有组件测试完成！</p>
      <button onClick={() => setStep(1)}>重新开始</button>
    </div>
  )
}

export default FeatureTestApp