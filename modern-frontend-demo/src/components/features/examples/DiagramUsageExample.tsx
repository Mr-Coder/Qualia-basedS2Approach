/**
 * 增强版实体关系图使用示例
 * 展示如何使用推理路径动画、智能布局和协作学习功能
 */

import React, { useState, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import EnhancedEntityRelationshipDiagram from '../EnhancedEntityRelationshipDiagram'
import {
  Entity,
  Relationship,
  ReasoningPath,
  ReasoningStep,
  EnhancedDiagramProps,
  AnimatedEntity,
  CollaborationEvent
} from '../types/DiagramTypes'

// 示例数据生成器
const createMathProblemExample = () => {
  // 问题: "小明有10个苹果，给了小红3个，又买了5个，现在有多少个苹果？"
  
  const entities: Entity[] = [
    { id: 'xiaoming', name: '小明', type: 'person' },
    { id: 'xiaohong', name: '小红', type: 'person' },
    { id: 'apple_initial', name: '初始苹果(10个)', type: 'object' },
    { id: 'apple_given', name: '给出苹果(3个)', type: 'object' },
    { id: 'apple_bought', name: '购买苹果(5个)', type: 'object' },
    { id: 'apple_final', name: '最终苹果(?个)', type: 'object' },
    { id: 'money', name: '购买费用', type: 'money' }
  ]
  
  const relationships: Relationship[] = [
    { source: 'xiaoming', target: 'apple_initial', type: '拥有', weight: 1.0 },
    { source: 'xiaoming', target: 'apple_given', type: '给出', weight: 0.8 },
    { source: 'xiaohong', target: 'apple_given', type: '接收', weight: 0.8 },
    { source: 'xiaoming', target: 'money', type: '支付', weight: 0.9 },
    { source: 'money', target: 'apple_bought', type: '购买', weight: 0.9 },
    { source: 'xiaoming', target: 'apple_bought', type: '获得', weight: 1.0 },
    { source: 'xiaoming', target: 'apple_final', type: '最终拥有', weight: 1.0 }
  ]
  
  const reasoningSteps: ReasoningStep[] = [
    {
      id: 'step1',
      sequence: 1,
      description: '确认小明的初始苹果数量',
      inputEntities: ['xiaoming'],
      outputEntities: ['apple_initial'],
      operation: 'derive',
      confidence: 1.0,
      timestamp: 0,
      duration: 1000,
      metadata: {
        mathOperation: '初始化: 10个苹果',
        logicType: 'deductive'
      }
    },
    {
      id: 'step2', 
      sequence: 2,
      description: '计算给出苹果后的数量',
      inputEntities: ['apple_initial', 'apple_given'],
      outputEntities: ['xiaoming'],
      operation: 'combine',
      confidence: 0.95,
      timestamp: 1500,
      duration: 1200,
      metadata: {
        mathOperation: '减法: 10 - 3 = 7',
        logicType: 'deductive'
      }
    },
    {
      id: 'step3',
      sequence: 3, 
      description: '加上新购买的苹果',
      inputEntities: ['xiaoming', 'apple_bought'],
      outputEntities: ['apple_final'],
      operation: 'combine',
      confidence: 0.98,
      timestamp: 3000,
      duration: 1000,
      metadata: {
        mathOperation: '加法: 7 + 5 = 12',
        logicType: 'deductive'
      }
    },
    {
      id: 'step4',
      sequence: 4,
      description: '验证最终结果',
      inputEntities: ['apple_final'],
      outputEntities: ['xiaoming'],
      operation: 'validate',
      confidence: 1.0,
      timestamp: 4500,
      duration: 800,
      metadata: {
        mathOperation: '验证: 12个苹果',
        logicType: 'deductive'
      }
    }
  ]
  
  const reasoningPath: ReasoningPath = {
    id: 'main-reasoning',
    steps: reasoningSteps,
    totalConfidence: 0.98,
    pathType: 'main',
    startTime: 0,
    endTime: 6000,
    isOptimal: true,
    visualization: {
      color: '#4CAF50',
      width: 3,
      style: 'solid',
      animationType: 'flow'
    }
  }
  
  return { entities, relationships, reasoningPaths: [reasoningPath] }
}

const createGeometryProblemExample = () => {
  // 问题: "长方形长12cm，宽8cm，求面积"
  
  const entities: Entity[] = [
    { id: 'rectangle', name: '长方形', type: 'concept' },
    { id: 'length', name: '长(12cm)', type: 'concept' },
    { id: 'width', name: '宽(8cm)', type: 'concept' }, 
    { id: 'area', name: '面积(?)', type: 'concept' },
    { id: 'formula', name: '面积公式', type: 'concept' }
  ]
  
  const relationships: Relationship[] = [
    { source: 'rectangle', target: 'length', type: '具有', weight: 1.0 },
    { source: 'rectangle', target: 'width', type: '具有', weight: 1.0 },
    { source: 'formula', target: 'area', type: '计算', weight: 1.0 },
    { source: 'length', target: 'area', type: '参与计算', weight: 0.9 },
    { source: 'width', target: 'area', type: '参与计算', weight: 0.9 }
  ]
  
  const reasoningSteps: ReasoningStep[] = [
    {
      id: 'geo_step1',
      sequence: 1,
      description: '识别几何图形和已知条件',
      inputEntities: ['rectangle', 'length', 'width'],
      outputEntities: ['formula'],
      operation: 'derive',
      confidence: 1.0,
      timestamp: 0,
      duration: 1000,
      metadata: {
        mathOperation: '识别: 长方形面积 = 长 × 宽',
        logicType: 'deductive'
      }
    },
    {
      id: 'geo_step2',
      sequence: 2,
      description: '应用面积公式计算',
      inputEntities: ['formula', 'length', 'width'],
      outputEntities: ['area'],
      operation: 'combine',
      confidence: 1.0,
      timestamp: 1500,
      duration: 1200,
      metadata: {
        mathOperation: '计算: 12 × 8 = 96 cm²',
        logicType: 'deductive'
      }
    }
  ]
  
  const reasoningPath: ReasoningPath = {
    id: 'geometry-reasoning',
    steps: reasoningSteps,
    totalConfidence: 1.0,
    pathType: 'main',
    startTime: 0,
    endTime: 3000,
    isOptimal: true,
    visualization: {
      color: '#2196F3',
      width: 3,
      style: 'solid',
      animationType: 'flow'
    }
  }
  
  return { entities, relationships, reasoningPaths: [reasoningPath] }
}

// 主示例组件
const DiagramUsageExample: React.FC = () => {
  const [currentExample, setCurrentExample] = useState<'math' | 'geometry' | 'custom'>('math')
  const [layoutType, setLayoutType] = useState<'force' | 'hierarchical' | 'circular' | 'timeline' | 'clustered'>('force')
  const [collaborationEnabled, setCollaborationEnabled] = useState(false)
  const [logs, setLogs] = useState<string[]>([])
  
  // 获取当前示例数据
  const getCurrentExampleData = () => {
    switch (currentExample) {
      case 'math':
        return createMathProblemExample()
      case 'geometry':
        return createGeometryProblemExample()
      case 'custom':
        return {
          entities: [
            { id: 'custom1', name: '自定义实体1', type: 'concept' as const },
            { id: 'custom2', name: '自定义实体2', type: 'concept' as const }
          ],
          relationships: [
            { source: 'custom1', target: 'custom2', type: '自定义关系', weight: 1.0 }
          ],
          reasoningPaths: []
        }
      default:
        return createMathProblemExample()
    }
  }
  
  const exampleData = getCurrentExampleData()
  
  // 构建完整的图表配置
  const diagramProps: EnhancedDiagramProps = {
    entities: exampleData.entities,
    relationships: exampleData.relationships,
    reasoningPaths: exampleData.reasoningPaths,
    
    layoutConfig: {
      type: layoutType,
      params: {
        forceStrength: 0.6,
        linkDistance: 100,
        centerForce: 0.3,
        direction: 'top-down',
        timelineOrientation: 'horizontal',
        clusterMethod: 'type'
      }
    },
    
    autoOptimizeLayout: true,
    
    animationConfig: {
      enablePathAnimation: true,
      enableEntityAnimation: true,
      animationSpeed: 1.0,
      simultaneousSteps: 1,
      pauseBetweenSteps: 800
    },
    
    collaborationConfig: collaborationEnabled ? {
      enabled: true,
      sessionId: 'demo-session-' + Date.now(),
      currentUserId: 'demo-user',
      realTimeSync: true
    } : undefined,
    
    interactionConfig: {
      enableDrag: true,
      enableZoom: true,
      enableSelection: true,
      enableComments: collaborationEnabled
    },
    
    width: 900,
    height: 600,
    
    // 事件回调
    onEntitySelect: useCallback((entity: AnimatedEntity) => {
      const message = `选中实体: ${entity.name} (${entity.type})`
      setLogs(prev => [message, ...prev.slice(0, 9)])
    }, []),
    
    onLayoutChange: useCallback((layout) => {
      const message = `布局更新: 质量评分 ${layout.quality.toFixed(2)}, 计算时间 ${layout.computationTime.toFixed(0)}ms`
      setLogs(prev => [message, ...prev.slice(0, 9)])
    }, []),
    
    onCollaborationEvent: useCallback((event: CollaborationEvent) => {
      const message = `协作事件: ${event.type} by ${event.userId}`
      setLogs(prev => [message, ...prev.slice(0, 9)])
    }, [])
  }
  
  return (
    <div className=\"space-y-6\">
      <Card>
        <CardHeader>
          <CardTitle className=\"flex items-center gap-2\">
            🚀 增强版实体关系图演示
            <div className=\"flex-1\" />
            <span className=\"text-sm text-gray-500\">
              使用示例和功能展示
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className=\"grid grid-cols-1 md:grid-cols-3 gap-4 mb-6\">
            {/* 示例选择 */}
            <div>
              <label className=\"block text-sm font-medium text-gray-700 mb-2\">
                📚 选择示例问题
              </label>
              <select
                value={currentExample}
                onChange={(e) => setCurrentExample(e.target.value as any)}
                className=\"w-full border rounded-md px-3 py-2 text-sm\"
              >
                <option value=\"math\">🧮 数学应用题</option>
                <option value=\"geometry\">📐 几何问题</option>
                <option value=\"custom\">🎨 自定义示例</option>
              </select>
            </div>
            
            {/* 布局选择 */}
            <div>
              <label className=\"block text-sm font-medium text-gray-700 mb-2\">
                🎯 智能布局算法
              </label>
              <select
                value={layoutType}
                onChange={(e) => setLayoutType(e.target.value as any)}
                className=\"w-full border rounded-md px-3 py-2 text-sm\"
              >
                <option value=\"force\">🔥 力导向布局</option>
                <option value=\"hierarchical\">🌳 分层布局</option>
                <option value=\"circular\">⭕ 圆形布局</option>
                <option value=\"timeline\">📅 时间轴布局</option>
                <option value=\"clustered\">📊 聚类布局</option>
              </select>
            </div>
            
            {/* 协作模式 */}
            <div>
              <label className=\"block text-sm font-medium text-gray-700 mb-2\">
                👥 协作学习模式
              </label>
              <div className=\"flex items-center space-x-3\">
                <Button
                  variant={collaborationEnabled ? 'primary' : 'outline'}
                  onClick={() => setCollaborationEnabled(!collaborationEnabled)}
                  size=\"sm\"
                  className=\"w-full\"
                >
                  {collaborationEnabled ? '✅ 协作模式开启' : '🔒 单人模式'}
                </Button>
              </div>
            </div>
          </div>
          
          {/* 当前示例描述 */}
          <div className=\"bg-blue-50 rounded-lg p-4 mb-4\">
            <h3 className=\"font-medium text-blue-900 mb-2\">
              {currentExample === 'math' && '🧮 数学应用题示例'}
              {currentExample === 'geometry' && '📐 几何问题示例'}
              {currentExample === 'custom' && '🎨 自定义示例'}
            </h3>
            <p className=\"text-blue-800 text-sm\">
              {currentExample === 'math' && 
                '演示如何通过推理路径动画展示"小明有10个苹果，给了小红3个，又买了5个，现在有多少个苹果"的解题过程。'}
              {currentExample === 'geometry' && 
                '展示几何问题的实体关系和计算步骤，包括长方形面积计算的完整推理过程。'}
              {currentExample === 'custom' && 
                '自定义示例，您可以修改代码来添加自己的实体、关系和推理步骤。'}
            </p>
          </div>
        </CardContent>
      </Card>
      
      {/* 主图表 */}
      <EnhancedEntityRelationshipDiagram {...diagramProps} />
      
      {/* 功能特色展示 */}
      <div className=\"grid grid-cols-1 md:grid-cols-2 gap-6\">
        {/* 核心功能 */}
        <Card>
          <CardHeader>
            <CardTitle className=\"text-lg\">✨ 核心功能特色</CardTitle>
          </CardHeader>
          <CardContent>
            <div className=\"space-y-3 text-sm\">
              <div className=\"flex items-start gap-3\">
                <span className=\"text-lg\">🎬</span>
                <div>
                  <div className=\"font-medium\">推理路径动画</div>
                  <div className=\"text-gray-600\">逐步展示思维过程，帮助学生理解解题步骤</div>
                </div>
              </div>
              
              <div className=\"flex items-start gap-3\">
                <span className=\"text-lg\">🧠</span>
                <div>
                  <div className=\"font-medium\">智能布局优化</div>
                  <div className=\"text-gray-600\">自动选择最适合的布局算法，优化视觉效果</div>
                </div>
              </div>
              
              <div className=\"flex items-start gap-3\">
                <span className=\"text-lg\">👥</span>
                <div>
                  <div className=\"font-medium\">协作学习模式</div>
                  <div className=\"text-gray-600\">支持多用户实时协作，师生互动讨论</div>
                </div>
              </div>
              
              <div className=\"flex items-start gap-3\">
                <span className=\"text-lg\">🎯</span>
                <div>
                  <div className=\"font-medium\">交互式探索</div>
                  <div className=\"text-gray-600\">点击、缩放、拖拽，多维度探索关系网络</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
        
        {/* 操作日志 */}
        <Card>
          <CardHeader>
            <CardTitle className=\"text-lg\">📋 操作日志</CardTitle>
          </CardHeader>
          <CardContent>
            <div className=\"space-y-1 text-xs font-mono max-h-48 overflow-y-auto bg-gray-50 p-3 rounded\">
              {logs.length === 0 ? (
                <div className=\"text-gray-500 text-center py-4\">
                  与图表交互查看操作日志
                </div>
              ) : (
                logs.map((log, index) => (
                  <div key={index} className=\"text-gray-700\">
                    {new Date().toLocaleTimeString()} - {log}
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>
      </div>
      
      {/* 使用方法 */}
      <Card>
        <CardHeader>
          <CardTitle className=\"text-lg\">📖 使用方法</CardTitle>
        </CardHeader>
        <CardContent>
          <div className=\"space-y-4 text-sm\">
            <div>
              <h4 className=\"font-medium mb-2\">🎮 基础操作</h4>
              <ul className=\"list-disc list-inside space-y-1 text-gray-600 ml-4\">
                <li>点击播放按钮开始推理路径动画</li>
                <li>使用缩放控制按钮调整视图大小</li>
                <li>点击实体节点查看详细信息</li>
                <li>切换不同布局算法观察效果</li>
              </ul>
            </div>
            
            <div>
              <h4 className=\"font-medium mb-2\">⚙️ 高级配置</h4>
              <ul className=\"list-disc list-inside space-y-1 text-gray-600 ml-4\">
                <li>修改 animationConfig 调整动画速度和效果</li>
                <li>自定义 layoutConfig 参数优化布局</li>
                <li>启用 collaborationConfig 开启协作模式</li>
                <li>实现自定义事件回调处理交互</li>
              </ul>
            </div>
            
            <div>
              <h4 className=\"font-medium mb-2\">💡 集成提示</h4>
              <div className=\"bg-yellow-50 p-3 rounded text-yellow-800\">
                <p>
                  将 EnhancedEntityRelationshipDiagram 组件集成到您的项目中，
                  只需提供 entities、relationships 和 reasoningPaths 数据即可。
                  组件会自动处理布局计算、动画播放和用户交互。
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* 代码示例 */}
      <Card>
        <CardHeader>
          <CardTitle className=\"text-lg\">💻 代码示例</CardTitle>
        </CardHeader>
        <CardContent>
          <pre className=\"bg-gray-100 p-4 rounded-lg text-xs overflow-x-auto\">
{`import EnhancedEntityRelationshipDiagram from './EnhancedEntityRelationshipDiagram'

const MyComponent = () => {
  const diagramProps = {
    entities: [
      { id: 'entity1', name: '实体1', type: 'person' },
      { id: 'entity2', name: '实体2', type: 'object' }
    ],
    relationships: [
      { source: 'entity1', target: 'entity2', type: '关系', weight: 1.0 }
    ],
    reasoningPaths: [{
      id: 'path1',
      steps: [/* 推理步骤 */],
      // ... 其他配置
    }],
    layoutConfig: {
      type: 'force',
      params: { forceStrength: 0.6 }
    },
    animationConfig: {
      enablePathAnimation: true,
      animationSpeed: 1.0
    },
    onEntitySelect: (entity) => {
      console.log('选中实体:', entity)
    }
  }
  
  return <EnhancedEntityRelationshipDiagram {...diagramProps} />
}`}
          </pre>
        </CardContent>
      </Card>
    </div>
  )
}

export default DiagramUsageExample