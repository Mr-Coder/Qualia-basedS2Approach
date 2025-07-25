import React from 'react'

const TestApp: React.FC = () => {
  return (
    <div style={{ padding: '20px', backgroundColor: '#f0f0f0', minHeight: '100vh' }}>
      <h1 style={{ color: 'green' }}>✅ React 正常工作!</h1>
      <p>如果你能看到这个页面，说明React和Vite都在正常运行。</p>
      <p>时间: {new Date().toLocaleString()}</p>
      <button onClick={() => alert('按钮点击正常!')}>测试交互</button>
    </div>
  )
}

export default TestApp