import React from 'react'
import Layout from '@/components/layout/Layout'
import { useHistory } from '@/stores/problemStore'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'

// 不用motion，不用formatTimestamp，只测试最基础的渲染
const HistoryPanelSimpleTest: React.FC = () => {
  return (
    <Layout activeTab="smart" setActiveTab={() => {}}>
      <div style={{ padding: '20px' }}>
        <h1>🔍 HistoryPanel最简测试</h1>
        <SimpleHistoryPanel />
      </div>
    </Layout>
  )
}

const SimpleHistoryPanel: React.FC = () => {
  try {
    const history = useHistory()

    return (
      <div>
        <div style={{ background: '#d4edda', padding: '10px', margin: '10px 0' }}>
          <h2>✅ useHistory正常，历史记录数量: {history.length}</h2>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>📚 解题历史 (简化版)</CardTitle>
          </CardHeader>
          <CardContent>
            {history.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '32px 0', color: '#6b7280' }}>
                <div style={{ fontSize: '2rem', marginBottom: '16px' }}>🔍</div>
                <p>还没有解题记录</p>
                <p style={{ fontSize: '14px', marginTop: '8px' }}>
                  开始解题后，历史记录将显示在这里
                </p>
              </div>
            ) : (
              <div style={{ maxHeight: '384px', overflowY: 'auto' }}>
                <div style={{ background: '#fff3cd', padding: '10px', marginBottom: '10px' }}>
                  <p>有 {history.length} 条历史记录，开始渲染...</p>
                </div>
                
                {history.map((item, index) => (
                  <SimpleHistoryItem key={item.id || index} item={item} index={index} />
                ))}
                
                <div style={{ background: '#d4edda', padding: '10px', marginTop: '10px' }}>
                  <p>✅ 所有历史记录渲染完成</p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <div style={{ background: '#d4edda', padding: '10px', margin: '10px 0' }}>
          <h2>🎉 SimpleHistoryPanel渲染成功！</h2>
        </div>
      </div>
    )
  } catch (error) {
    return (
      <div style={{ background: '#f8d7da', padding: '10px' }}>
        <h2>❌ SimpleHistoryPanel失败</h2>
        <p>错误: {String(error)}</p>
        <pre style={{ fontSize: '12px', overflow: 'auto' }}>
          {error instanceof Error ? error.stack : String(error)}
        </pre>
      </div>
    )
  }
}

// 最简单的历史记录项，不用motion，不用复杂样式
const SimpleHistoryItem: React.FC<{item: any, index: number}> = ({ item, index }) => {
  try {
    return (
      <div style={{ 
        border: '1px solid #e5e7eb', 
        borderRadius: '8px', 
        padding: '16px',
        marginBottom: '16px',
        backgroundColor: '#f9fafb'
      }}>
        <div style={{ background: '#e3f2fd', padding: '5px', marginBottom: '10px' }}>
          <small>正在渲染第 {index + 1} 条记录...</small>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: '14px', fontWeight: '500', color: '#111827', marginBottom: '8px' }}>
              问题: {item.problem || '未知问题'}
            </div>
            <div style={{ fontSize: '12px', color: '#6b7280' }}>
              <div>策略: {item.strategy || '未知'}</div>
              <div>答案: {item.result?.answer || '无答案'}</div>
              <div>时间: {item.timestamp ? String(item.timestamp) : '无时间'}</div>
            </div>
          </div>
          <div style={{ marginLeft: '16px' }}>
            <div style={{ width: '8px', height: '8px', backgroundColor: '#10b981', borderRadius: '50%' }}></div>
          </div>
        </div>
        
        <div style={{ background: '#d4edda', padding: '5px', marginTop: '10px' }}>
          <small>✅ 第 {index + 1} 条记录渲染完成</small>
        </div>
      </div>
    )
  } catch (error) {
    return (
      <div style={{ background: '#f8d7da', padding: '10px', margin: '5px 0' }}>
        <p>❌ 历史记录项 {index + 1} 渲染失败</p>
        <p>错误: {String(error)}</p>
      </div>
    )
  }
}

export default HistoryPanelSimpleTest