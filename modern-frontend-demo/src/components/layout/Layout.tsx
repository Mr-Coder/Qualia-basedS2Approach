import React from 'react'
import { motion } from 'framer-motion'

type TabType = 'solver' | 'knowledge' | 'learning' | 'error' | 'strategy' | 'diagram'

interface LayoutProps {
  children: React.ReactNode
  activeTab: TabType
  setActiveTab: (tab: TabType) => void
}

const Layout: React.FC<LayoutProps> = ({ children, activeTab, setActiveTab }) => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50">
      {/* 背景装饰 */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-300 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse-slow"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-blue-300 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse-slow"></div>
      </div>

      {/* 主要内容 */}
      <div className="relative z-10">
        <Header activeTab={activeTab} setActiveTab={setActiveTab} />
        <main className="container mx-auto px-4 py-8">
          {children}
        </main>
      </div>
    </div>
  )
}

interface HeaderProps {
  activeTab: TabType
  setActiveTab: (tab: TabType) => void
}

const Header: React.FC<HeaderProps> = ({ activeTab, setActiveTab }) => {
  const tabs = [
    { id: 'solver' as TabType, name: '🧮 解题器', description: '智能数学问题求解' },
    { id: 'knowledge' as TabType, name: '🗺️ 知识图谱', description: '知识结构可视化' },
    { id: 'learning' as TabType, name: '📚 学习指导', description: '系统学习路径' },
    { id: 'error' as TabType, name: '🔍 错题分析', description: '智能错误诊断' },
    { id: 'strategy' as TabType, name: '🎯 策略分析', description: '推理策略详解' },
    { id: 'diagram' as TabType, name: '🔬 物性关系图', description: '物性关系情景可视化' }
  ]

  return (
    <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-50">
      <div className="container mx-auto px-4">
        {/* 顶部品牌区域 */}
        <div className="flex items-center justify-between h-16 border-b border-gray-100">
          <motion.div 
            className="flex items-center space-x-3"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
          >
            <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">🧠</span>
            </div>
            <div>
              <h1 className="text-xl font-bold gradient-text">COT-DIR</h1>
              <p className="text-xs text-gray-600">智能数学推理系统</p>
            </div>
          </motion.div>

          <motion.div 
            className="text-sm text-gray-500"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            现代化前端框架 • React + TypeScript
          </motion.div>
        </div>

        {/* 导航标签页 */}
        <div className="flex items-center space-x-1 py-3 overflow-x-auto">
          {tabs.map((tab, index) => (
            <motion.button
              key={tab.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
              onClick={() => setActiveTab(tab.id)}
              className={`relative px-4 py-2 rounded-lg text-sm font-medium transition-all whitespace-nowrap group ${
                activeTab === tab.id
                  ? 'bg-purple-100 text-purple-700 shadow-sm'
                  : 'text-gray-600 hover:text-purple-600 hover:bg-purple-50'
              }`}
            >
              <span className="relative z-10">{tab.name}</span>
              
              {/* 活跃状态指示器 */}
              {activeTab === tab.id && (
                <motion.div
                  layoutId="activeTab"
                  className="absolute inset-0 bg-purple-100 rounded-lg border border-purple-200"
                  transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                />
              )}
              
              {/* 悬浮提示 */}
              <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-gray-800 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                {tab.description}
                <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-2 border-r-2 border-t-2 border-transparent border-t-gray-800"></div>
              </div>
            </motion.button>
          ))}
        </div>
      </div>
    </header>
  )
}

export default Layout