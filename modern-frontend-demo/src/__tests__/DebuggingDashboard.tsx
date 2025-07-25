import React, { useState } from 'react'
import Layout from '@/components/layout/Layout'

const DebuggingDashboard: React.FC = () => {
  const [activeTest, setActiveTest] = useState('main')

  return (
    <Layout activeTab="smart" setActiveTab={() => {}}>
      <div style={{ padding: '20px' }}>
        <h1>🔍 白屏问题调试面板</h1>
        
        <div style={{ background: '#f0f8ff', padding: '15px', margin: '15px 0', border: '1px solid #0066cc' }}>
          <h3>已知问题:</h3>
          <ul>
            <li>✅ HistoryPanel - 已修复</li>
            <li>❌ 解题功能 - 导致白屏</li>
            <li>❌ 物性关系图 - 导致白屏</li>
          </ul>
        </div>

        <div style={{ display: 'flex', gap: '10px', margin: '15px 0', flexWrap: 'wrap' }}>
          <button 
            onClick={() => setActiveTest('main')}
            style={{ 
              padding: '8px 16px', 
              backgroundColor: activeTest === 'main' ? '#007bff' : '#6c757d', 
              color: 'white', 
              border: 'none', 
              borderRadius: '4px' 
            }}
          >
            主面板
          </button>
          <button 
            onClick={() => setActiveTest('solver')}
            style={{ 
              padding: '8px 16px', 
              backgroundColor: activeTest === 'solver' ? '#dc3545' : '#6c757d', 
              color: 'white', 
              border: 'none', 
              borderRadius: '4px' 
            }}
          >
            测试SmartSolver
          </button>
          <button 
            onClick={() => setActiveTest('diagram')}
            style={{ 
              padding: '8px 16px', 
              backgroundColor: activeTest === 'diagram' ? '#ffc107' : '#6c757d', 
              color: 'black', 
              border: 'none', 
              borderRadius: '4px' 
            }}
          >
            测试物性关系图
          </button>
        </div>

        {activeTest === 'main' && <MainPanel />}
        {activeTest === 'solver' && <SolverTest />}
        {activeTest === 'diagram' && <DiagramTest />}
      </div>
    </Layout>
  )
}

const MainPanel: React.FC = () => {
  return (
    <div>
      <div style={{ background: '#d4edda', padding: '15px', borderRadius: '5px' }}>
        <h2>🏠 主调试面板</h2>
        <p>使用上面的按钮分别测试可能导致白屏的组件</p>
        <p>这样我们可以逐个定位问题</p>
      </div>
    </div>
  )
}

const SolverTest: React.FC = () => {
  try {
    // 动态导入以防止立即崩溃
    const SmartSolver = React.lazy(() => import('@/components/features/SmartSolver'))
    
    return (
      <div>
        <div style={{ background: '#fff3cd', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
          <h2>⚠️ SmartSolver 测试</h2>
          <p>正在测试SmartSolver组件...</p>
        </div>
        
        <React.Suspense fallback={<div>加载SmartSolver中...</div>}>
          <div style={{ border: '2px solid #dc3545', padding: '15px', borderRadius: '5px' }}>
            <SmartSolver />
          </div>
        </React.Suspense>
        
        <div style={{ background: '#d4edda', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
          <h2>✅ SmartSolver加载成功！</h2>
          <p>如果你能看到这个，说明SmartSolver没有立即崩溃</p>
        </div>
      </div>
    )
  } catch (error) {
    return (
      <div style={{ background: '#f8d7da', padding: '15px', margin: '15px 0' }}>
        <h2>❌ SmartSolver测试失败</h2>
        <p>错误: {String(error)}</p>
        <pre style={{ fontSize: '12px', overflow: 'auto', maxHeight: '200px' }}>
          {error instanceof Error ? error.stack : String(error)}
        </pre>
      </div>
    )
  }
}

const DiagramTest: React.FC = () => {
  try {
    // 使用关系图组件
    const EntityRelationshipDiagram = React.lazy(() => import('@/components/features/EntityRelationshipDiagram'))
    
    return (
      <div>
        <div style={{ background: '#fff3cd', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
          <h2>⚠️ 物性关系图测试</h2>
          <p>正在测试EntityRelationshipDiagram组件...</p>
        </div>
        
        <React.Suspense fallback={<div>加载物性关系图中...</div>}>
          <div style={{ border: '2px solid #ffc107', padding: '15px', borderRadius: '5px' }}>
            <EntityRelationshipDiagram />
          </div>
        </React.Suspense>
        
        <div style={{ background: '#d4edda', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
          <h2>✅ 物性关系图加载成功！</h2>
          <p>如果你能看到这个，说明物性关系图没有立即崩溃</p>
        </div>
      </div>
    )
  } catch (error) {
    return (
      <div style={{ background: '#f8d7da', padding: '15px', margin: '15px 0' }}>
        <h2>❌ 物性关系图测试失败</h2>
        <p>错误: {String(error)}</p>
        <pre style={{ fontSize: '12px', overflow: 'auto', maxHeight: '200px' }}>
          {error instanceof Error ? error.stack : String(error)}
        </pre>
      </div>
    )
  }
}

export default DebuggingDashboard