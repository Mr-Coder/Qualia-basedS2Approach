import React from 'react'
import { motion } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { useHistory } from '@/stores/problemStore'

// 安全的时间戳格式化函数 - 修复白屏问题
const safeFormatTimestamp = (timestamp: any): string => {
  try {
    // 如果已经是Date对象，直接格式化
    if (timestamp instanceof Date) {
      return new Intl.DateTimeFormat('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
      }).format(timestamp)
    }
    
    // 如果是字符串或数字，尝试转换为Date
    const date = new Date(timestamp)
    if (!isNaN(date.getTime())) {
      return new Intl.DateTimeFormat('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
      }).format(date)
    }
    
    // 如果都失败了，返回原始值的字符串
    return String(timestamp)
  } catch (error) {
    // 出错时返回默认值
    return '时间未知'
  }
}

const HistoryPanel: React.FC = () => {
  const history = useHistory()
  
  // 确保 history 是数组
  const safeHistory = Array.isArray(history) ? history : []

  if (safeHistory.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>📚 解题历史</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500">
            <div className="text-4xl mb-4">🔍</div>
            <p>还没有解题记录</p>
            <p className="text-sm mt-2">开始解题后，历史记录将显示在这里</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>📚 解题历史</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4 max-h-96 overflow-y-auto">
          {safeHistory.map((item, index) => (
            <motion.div
              key={item.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="text-sm font-medium text-gray-900 mb-2 line-clamp-2">
                    {item.problem}
                  </div>
                  <div className="flex items-center space-x-4 text-xs text-gray-500">
                    <span>
                      策略: {item.strategy === 'auto' ? '自动' : 
                            item.strategy === 'cot' ? 'COT' :
                            item.strategy === 'got' ? 'GOT' : 'TOT'}
                    </span>
                    <span>答案: {item.answer}</span>
                    <span>{safeFormatTimestamp(item.timestamp)}</span>
                  </div>
                </div>
                <div className="ml-4">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

export default HistoryPanel