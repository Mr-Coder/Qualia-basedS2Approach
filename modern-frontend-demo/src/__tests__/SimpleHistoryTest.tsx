import React from 'react'
import Layout from '@/components/layout/Layout'

const SimpleHistoryTest: React.FC = () => {
  return (
    <Layout activeTab="smart" setActiveTab={() => {}}>
      <div style={{ padding: '20px' }}>
        <h1>🔍 HistoryPanel分步测试</h1>
        <HistoryTestStep1 />
      </div>
    </Layout>
  )
}

// 步骤1: 测试基础导入
const HistoryTestStep1: React.FC = () => {
  try {
    // 测试store导入
    import { useHistory } from '@/stores/problemStore'
    
    return (
      <div style={{ background: '#d4edda', padding: '10px', margin: '10px 0' }}>
        <h2>✅ Store导入成功</h2>
        <HistoryTestStep2 />
      </div>
    )
  } catch (error) {
    return (
      <div style={{ background: '#f8d7da', padding: '10px' }}>
        <h2>❌ Store导入失败</h2>
        <p>错误: {String(error)}</p>
      </div>
    )
  }
}

// 步骤2: 测试UI组件导入
const HistoryTestStep2: React.FC = () => {
  try {
    import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
    
    return (
      <div style={{ background: '#d4edda', padding: '10px', margin: '10px 0' }}>
        <h2>✅ UI组件导入成功</h2>
        <HistoryTestStep3 />
      </div>
    )
  } catch (error) {
    return (
      <div style={{ background: '#f8d7da', padding: '10px' }}>
        <h2>❌ UI组件导入失败</h2>
        <p>错误: {String(error)}</p>
      </div>
    )
  }
}

// 步骤3: 测试helpers导入
const HistoryTestStep3: React.FC = () => {
  try {
    import { formatTimestamp } from '@/utils/helpers'
    
    return (
      <div style={{ background: '#d4edda', padding: '10px', margin: '10px 0' }}>
        <h2>✅ Helpers导入成功</h2>
        <HistoryTestStep4 />
      </div>
    )
  } catch (error) {
    return (
      <div style={{ background: '#f8d7da', padding: '10px' }}>
        <h2>❌ Helpers导入失败</h2>
        <p>错误: {String(error)}</p>
      </div>
    )
  }
}

// 步骤4: 测试framer-motion导入
const HistoryTestStep4: React.FC = () => {
  try {
    import { motion } from 'framer-motion'
    
    return (
      <div style={{ background: '#d4edda', padding: '10px', margin: '10px 0' }}>
        <h2>✅ Framer Motion导入成功</h2>
        <HistoryTestStep5 />
      </div>
    )
  } catch (error) {
    return (
      <div style={{ background: '#f8d7da', padding: '10px' }}>
        <h2>❌ Framer Motion导入失败</h2>
        <p>错误: {String(error)}</p>
        <pre style={{ fontSize: '12px' }}>{String(error)}</pre>
      </div>
    )
  }
}

// 步骤5: 测试useHistory hook
const HistoryTestStep5: React.FC = () => {
  try {
    import { useHistory } from '@/stores/problemStore'
    const history = useHistory()
    
    return (
      <div style={{ background: '#d4edda', padding: '10px', margin: '10px 0' }}>
        <h2>✅ useHistory Hook正常</h2>
        <p>历史记录数量: {history.length}</p>
        <HistoryTestStep6 />
      </div>
    )
  } catch (error) {
    return (
      <div style={{ background: '#f8d7da', padding: '10px' }}>
        <h2>❌ useHistory Hook失败</h2>
        <p>错误: {String(error)}</p>
        <pre style={{ fontSize: '12px' }}>{String(error)}</pre>
      </div>
    )
  }
}

// 步骤6: 测试简单HistoryPanel渲染
const HistoryTestStep6: React.FC = () => {
  try {
    import { useHistory } from '@/stores/problemStore'
    import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
    const history = useHistory()
    
    return (
      <div>
        <Card>
          <CardHeader>
            <CardTitle>📚 简单历史面板测试</CardTitle>
          </CardHeader>
          <CardContent>
            <div style={{ textAlign: 'center', padding: '32px 0', color: '#6b7280' }}>
              <div style={{ fontSize: '2rem', marginBottom: '16px' }}>🔍</div>
              <p>历史记录数量: {history.length}</p>
              <p style={{ fontSize: '14px', marginTop: '8px' }}>
                {history.length === 0 ? '还没有解题记录' : '有解题记录'}
              </p>
            </div>
          </CardContent>
        </Card>
        
        <div style={{ background: '#d4edda', padding: '10px', margin: '10px 0' }}>
          <h2>🎉 HistoryPanel基础渲染成功！</h2>
          <p>问题不在基础渲染，可能在motion动画或复杂逻辑</p>
        </div>
      </div>
    )
  } catch (error) {
    return (
      <div style={{ background: '#f8d7da', padding: '10px' }}>
        <h2>❌ HistoryPanel渲染失败</h2>
        <p>错误: {String(error)}</p>
        <pre style={{ fontSize: '12px', overflow: 'auto' }}>
          {error instanceof Error ? error.stack : String(error)}
        </pre>
      </div>
    )
  }
}

export default SimpleHistoryTest