import React from 'react'
import Layout from '@/components/layout/Layout'

const LayoutTest: React.FC = () => {
  try {
    return (
      <Layout activeTab="smart" setActiveTab={() => {}}>
        <div style={{ padding: '20px', fontSize: '18px' }}>
          <h1>🔍 Layout组件测试</h1>
          <div style={{ background: '#d4edda', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
            <h2>✅ Layout组件正常工作</h2>
            <p>时间: {new Date().toLocaleString()}</p>
          </div>
          <div style={{ background: '#f0f8ff', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
            <h3>测试进度:</h3>
            <ul>
              <li>✓ React基础渲染</li>
              <li>✓ Layout组件</li>
              <li>? Store状态管理</li>
              <li>? HistoryPanel组件</li>
            </ul>
          </div>
        </div>
      </Layout>
    )
  } catch (error) {
    return (
      <div style={{ padding: '20px', background: '#f8d7da' }}>
        <h1>❌ Layout组件测试失败</h1>
        <p>错误: {String(error)}</p>
        <pre style={{ fontSize: '12px', overflow: 'auto' }}>
          {error instanceof Error ? error.stack : String(error)}
        </pre>
      </div>
    )
  }
}

export default LayoutTest