import React from 'react'

const SimpleTest: React.FC = () => {
  return (
    <div style={{ padding: '20px', fontSize: '18px' }}>
      <h1>🔍 简单测试页面</h1>
      <p>如果你能看到这个页面，说明基本的React渲染正常。</p>
      <div style={{ background: '#d4edda', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
        <h2>✅ React正常工作</h2>
        <p>时间: {new Date().toLocaleString()}</p>
      </div>
      <div style={{ background: '#f0f8ff', padding: '15px', margin: '15px 0', borderRadius: '5px' }}>
        <h3>下一步测试:</h3>
        <ul>
          <li>✓ React基础渲染</li>
          <li>? Layout组件</li>
          <li>? Store状态管理</li>
          <li>? HistoryPanel组件</li>
        </ul>
      </div>
    </div>
  )
}

export default SimpleTest