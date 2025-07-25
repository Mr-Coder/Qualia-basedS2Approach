import React from 'react'
import Layout from '@/components/layout/Layout'

// 首先导入所有需要的依赖来测试
import { useHistory } from '@/stores/problemStore'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { motion } from 'framer-motion'

const HistoryPanelDebugger: React.FC = () => {
  return (
    <Layout activeTab="smart" setActiveTab={() => {}}>
      <div style={{ padding: '20px' }}>
        <h1>🔍 HistoryPanel组件调试器</h1>
        
        <div style={{ background: '#d4edda', padding: '10px', margin: '10px 0' }}>
          <h2>✅ 步骤1: 所有依赖导入成功</h2>
          <p>useHistory, Card组件, motion - 都已成功导入</p>
        </div>

        <TestBasicHook />
      </div>
    </Layout>
  )
}

// 测试基础Hook
const TestBasicHook: React.FC = () => {
  try {
    const history = useHistory()
    
    return (
      <div>
        <div style={{ background: '#d4edda', padding: '10px', margin: '10px 0' }}>
          <h2>✅ 步骤2: useHistory Hook正常</h2>
          <p>历史记录数量: {history.length}</p>
        </div>
        
        <TestBasicCard />
      </div>
    )
  } catch (error) {
    return (
      <div style={{ background: '#f8d7da', padding: '10px' }}>
        <h2>❌ useHistory Hook失败</h2>
        <p>错误: {String(error)}</p>
      </div>
    )
  }
}

// 测试基础Card组件
const TestBasicCard: React.FC = () => {
  try {
    const history = useHistory()
    
    return (
      <div>
        <div style={{ background: '#d4edda', padding: '10px', margin: '10px 0' }}>
          <h2>✅ 步骤3: Card组件渲染正常</h2>
        </div>
        
        <Card>
          <CardHeader>
            <CardTitle>📚 测试Card</CardTitle>
          </CardHeader>
          <CardContent>
            <p>Card组件正常工作, 历史记录数量: {history.length}</p>
          </CardContent>
        </Card>
        
        <TestBasicMotion />
      </div>
    )
  } catch (error) {
    return (
      <div style={{ background: '#f8d7da', padding: '10px' }}>
        <h2>❌ Card组件渲染失败</h2>
        <p>错误: {String(error)}</p>
      </div>
    )
  }
}

// 测试基础Motion
const TestBasicMotion: React.FC = () => {
  try {
    return (
      <div>
        <div style={{ background: '#d4edda', padding: '10px', margin: '10px 0' }}>
          <h2>✅ 步骤4: 测试简单Motion</h2>
        </div>
        
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          style={{ background: '#e3f2fd', padding: '10px', margin: '10px 0' }}
        >
          简单Motion动画正常
        </motion.div>
        
        <TestHistoryList />
      </div>
    )
  } catch (error) {
    return (
      <div style={{ background: '#f8d7da', padding: '10px' }}>
        <h2>❌ Motion动画失败</h2>
        <p>错误: {String(error)}</p>
      </div>
    )
  }
}

// 测试历史记录列表渲染 (这是最关键的部分)
const TestHistoryList: React.FC = () => {
  try {
    const history = useHistory()
    
    return (
      <div>
        <div style={{ background: '#d4edda', padding: '10px', margin: '10px 0' }}>
          <h2>✅ 步骤5: 开始测试历史记录列表</h2>
        </div>
        
        <Card>
          <CardHeader>
            <CardTitle>📚 历史记录测试</CardTitle>
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
              <TestHistoryItemsWithMotion history={history} />
            )}
          </CardContent>
        </Card>
        
        <div style={{ background: '#d4edda', padding: '10px', margin: '10px 0' }}>
          <h2>🎉 所有测试完成!</h2>
          <p>如果能看到这个，说明HistoryPanel的核心逻辑都没问题</p>
        </div>
      </div>
    )
  } catch (error) {
    return (
      <div style={{ background: '#f8d7da', padding: '10px' }}>
        <h2>❌ 历史记录列表失败</h2>
        <p>错误: {String(error)}</p>
        <pre style={{ fontSize: '12px', overflow: 'auto' }}>
          {error instanceof Error ? error.stack : String(error)}
        </pre>
      </div>
    )
  }
}

// 测试带Motion的历史记录项
const TestHistoryItemsWithMotion: React.FC<{history: any[]}> = ({ history }) => {
  try {
    return (
      <div style={{ maxHeight: '96px', overflowY: 'auto' }}>
        {history.map((item, index) => (
          <motion.div
            key={item.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            style={{ 
              border: '1px solid #e5e7eb', 
              borderRadius: '8px', 
              padding: '16px',
              marginBottom: '16px',
              backgroundColor: '#f9fafb'
            }}
          >
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: '14px', fontWeight: '500', color: '#111827', marginBottom: '8px' }}>
                  {item.problem}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px', fontSize: '12px', color: '#6b7280' }}>
                  <span>
                    策略: {item.strategy === 'auto' ? '自动' : 
                          item.strategy === 'cot' ? 'COT' :
                          item.strategy === 'got' ? 'GOT' : 'TOT'}
                  </span>
                  <span>答案: {item.result.answer}</span>
                  <span>{new Date(item.timestamp).toLocaleString()}</span>
                </div>
              </div>
              <div style={{ marginLeft: '16px' }}>
                <div style={{ width: '8px', height: '8px', backgroundColor: '#10b981', borderRadius: '50%' }}></div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    )
  } catch (error) {
    return (
      <div style={{ background: '#f8d7da', padding: '10px' }}>
        <h2>❌ Motion历史项失败</h2>
        <p>错误: {String(error)}</p>
        <pre style={{ fontSize: '12px' }}>{String(error)}</pre>
      </div>
    )
  }
}

export default HistoryPanelDebugger