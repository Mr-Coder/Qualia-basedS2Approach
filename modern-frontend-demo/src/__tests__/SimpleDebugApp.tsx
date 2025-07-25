import React, { useState } from 'react'

// 直接测试各个导入
const SimpleDebugApp: React.FC = () => {
  const [currentTest, setCurrentTest] = useState('basic')

  if (currentTest === 'basic') {
    return (
      <div style={{ padding: '20px' }}>
        <h1>✅ 基础React测试成功</h1>
        <button onClick={() => setCurrentTest('store')}>测试Store导入</button>
      </div>
    )
  }

  if (currentTest === 'store') {
    return <StoreTest setCurrentTest={setCurrentTest} />
  }

  if (currentTest === 'layout') {
    return <LayoutTest setCurrentTest={setCurrentTest} />
  }

  if (currentTest === 'smartsolver') {
    return <SmartSolverTest setCurrentTest={setCurrentTest} />
  }

  return <div>未知测试</div>
}

// 测试Store
const StoreTest: React.FC<{setCurrentTest: (test: string) => void}> = ({ setCurrentTest }) => {
  return (
    <div style={{ padding: '20px' }}>
      <h1>🔍 正在测试Store...</h1>
      <StoreTestContent setCurrentTest={setCurrentTest} />
    </div>
  )
}

const StoreTestContent: React.FC<{setCurrentTest: (test: string) => void}> = ({ setCurrentTest }) => {
  try {
    // 直接导入测试
    const { useProblemStore } = require('@/stores/problemStore')
    const hook = useProblemStore()
    
    return (
      <div style={{ background: '#d4edda', padding: '10px' }}>
        <h2>✅ Store导入成功</h2>
        <p>Store state keys: {Object.keys(hook).join(', ')}</p>
        <button onClick={() => setCurrentTest('layout')}>测试Layout</button>
        <button onClick={() => setCurrentTest('basic')}>返回</button>
      </div>
    )
  } catch (error) {
    return (
      <div style={{ background: '#f8d7da', padding: '10px' }}>
        <h2>❌ Store导入失败</h2>
        <p>错误: {String(error)}</p>
        <pre>{error instanceof Error ? error.stack : ''}</pre>
        <button onClick={() => setCurrentTest('basic')}>返回</button>
      </div>
    )
  }
}

// 测试Layout
const LayoutTest: React.FC<{setCurrentTest: (test: string) => void}> = ({ setCurrentTest }) => {
  return (
    <div style={{ padding: '20px' }}>
      <h1>🔍 正在测试Layout...</h1>
      <LayoutTestContent setCurrentTest={setCurrentTest} />
    </div>
  )
}

const LayoutTestContent: React.FC<{setCurrentTest: (test: string) => void}> = ({ setCurrentTest }) => {
  try {
    const Layout = require('@/components/layout/Layout').default
    
    return (
      <div style={{ background: '#d4edda', padding: '10px' }}>
        <h2>✅ Layout导入成功</h2>
        <Layout activeTab="smart" setActiveTab={() => {}}>
          <div>测试内容</div>
        </Layout>
        <button onClick={() => setCurrentTest('smartsolver')}>测试SmartSolver</button>
        <button onClick={() => setCurrentTest('store')}>返回</button>
      </div>
    )
  } catch (error) {
    return (
      <div style={{ background: '#f8d7da', padding: '10px' }}>
        <h2>❌ Layout导入失败</h2>
        <p>错误: {String(error)}</p>
        <pre>{error instanceof Error ? error.stack : ''}</pre>
        <button onClick={() => setCurrentTest('store')}>返回</button>
      </div>
    )
  }
}

// 测试SmartSolver
const SmartSolverTest: React.FC<{setCurrentTest: (test: string) => void}> = ({ setCurrentTest }) => {
  return (
    <div style={{ padding: '20px' }}>
      <h1>🔍 正在测试SmartSolver...</h1>
      <SmartSolverTestContent setCurrentTest={setCurrentTest} />
    </div>
  )
}

const SmartSolverTestContent: React.FC<{setCurrentTest: (test: string) => void}> = ({ setCurrentTest }) => {
  try {
    const SmartSolver = require('@/components/features/SmartSolver').default
    
    return (
      <div style={{ background: '#d4edda', padding: '10px' }}>
        <h2>✅ SmartSolver导入成功</h2>
        <SmartSolver />
        <button onClick={() => setCurrentTest('layout')}>返回</button>
      </div>
    )
  } catch (error) {
    return (
      <div style={{ background: '#f8d7da', padding: '10px' }}>
        <h2>❌ SmartSolver导入失败</h2>
        <p>错误: {String(error)}</p>
        <pre>{error instanceof Error ? error.stack : ''}</pre>
        <button onClick={() => setCurrentTest('layout')}>返回</button>
      </div>
    )
  }
}

export default SimpleDebugApp