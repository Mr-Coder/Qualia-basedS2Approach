import React from 'react'
import Layout from '@/components/layout/Layout'
import { useHistory, useProblemStore } from '@/stores/problemStore'
import { formatTimestamp } from '@/utils/helpers'

const TimestampTest: React.FC = () => {
  const history = useHistory()

  return (
    <Layout activeTab="smart" setActiveTab={() => {}}>
      <div style={{ padding: '20px' }}>
        <h1>🔍 Timestamp 问题调试</h1>
        
        <div style={{ background: '#fff3cd', padding: '10px', margin: '10px 0' }}>
          <p>历史记录数量: {history.length}</p>
        </div>

        {history.length === 0 ? (
          <div style={{ background: '#d4edda', padding: '10px' }}>
            <h2>✅ 无历史记录，测试添加一条</h2>
            <TestAddHistory />
          </div>
        ) : (
          <div>
            <h2>历史记录分析:</h2>
            {history.map((item, index) => (
              <TimestampDebugItem key={item.id || index} item={item} index={index} />
            ))}
          </div>
        )}
      </div>
    </Layout>
  )
}

const TestAddHistory: React.FC = () => {
  const addToHistory = useProblemStore(state => state.addToHistory)

  const handleAddTest = () => {
    try {
      const testEntry = {
        id: `test-${Date.now()}`,
        problem: '测试问题：小明有3个苹果，小红有2个苹果，一共有多少个苹果？',
        answer: '5个苹果',
        strategy: 'cot' as const,
        timestamp: new Date(),
        confidence: 0.95
      }
      
      addToHistory(testEntry)
      
      return (
        <div style={{ background: '#d4edda', padding: '10px', margin: '10px 0' }}>
          <p>✅ 测试记录已添加</p>
        </div>
      )
    } catch (error) {
      return (
        <div style={{ background: '#f8d7da', padding: '10px' }}>
          <p>❌ 添加测试记录失败: {String(error)}</p>
        </div>
      )
    }
  }

  return (
    <div>
      <button onClick={handleAddTest} style={{ padding: '10px', backgroundColor: '#007bff', color: 'white', border: 'none', borderRadius: '4px' }}>
        添加测试记录
      </button>
    </div>
  )
}

const TimestampDebugItem: React.FC<{item: any, index: number}> = ({ item, index }) => {
  return (
    <div style={{ 
      border: '1px solid #ccc', 
      padding: '15px', 
      margin: '10px 0',
      backgroundColor: '#f9f9f9'
    }}>
      <h3>记录 #{index + 1}</h3>
      
      <div style={{ marginBottom: '10px' }}>
        <strong>问题:</strong> {item.problem}
      </div>
      
      <div style={{ marginBottom: '10px' }}>
        <strong>答案:</strong> {item.answer}
      </div>
      
      <div style={{ marginBottom: '10px' }}>
        <strong>策略:</strong> {item.strategy}
      </div>
      
      <div style={{ marginBottom: '10px' }}>
        <strong>Timestamp 原始值:</strong>
        <pre style={{ background: '#eee', padding: '5px', fontSize: '12px' }}>
          {JSON.stringify(item.timestamp, null, 2)}
        </pre>
      </div>
      
      <div style={{ marginBottom: '10px' }}>
        <strong>Timestamp 类型:</strong> {typeof item.timestamp}
      </div>
      
      <div style={{ marginBottom: '10px' }}>
        <strong>是否为Date对象:</strong> {item.timestamp instanceof Date ? '是' : '否'}
      </div>
      
      <TimestampFormatTest timestamp={item.timestamp} />
    </div>
  )
}

const TimestampFormatTest: React.FC<{timestamp: any}> = ({ timestamp }) => {
  try {
    // 测试1: 直接使用formatTimestamp
    const formatted1 = formatTimestamp(timestamp)
    
    return (
      <div style={{ background: '#d4edda', padding: '10px' }}>
        <strong>✅ formatTimestamp成功:</strong> {formatted1}
      </div>
    )
  } catch (error1) {
    try {
      // 测试2: 转换为Date再格式化
      const dateObj = new Date(timestamp)
      const formatted2 = formatTimestamp(dateObj)
      
      return (
        <div style={{ background: '#fff3cd', padding: '10px' }}>
          <strong>⚠️ 需要转换为Date:</strong> {formatted2}
          <div style={{ fontSize: '12px', marginTop: '5px' }}>
            原始错误: {String(error1)}
          </div>
        </div>
      )
    } catch (error2) {
      return (
        <div style={{ background: '#f8d7da', padding: '10px' }}>
          <strong>❌ formatTimestamp完全失败:</strong>
          <div style={{ fontSize: '12px' }}>
            <div>错误1: {String(error1)}</div>
            <div>错误2: {String(error2)}</div>
          </div>
        </div>
      )
    }
  }
}

export default TimestampTest