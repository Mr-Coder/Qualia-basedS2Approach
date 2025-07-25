import React, { useState, useEffect, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

/**
 * 算法可视化实体关系图组件
 * 
 * 核心功能：
 * 1. 实时算法过程可视化 - IRD算法逐步发现关系
 * 2. 认知友好指引系统 - 基于7±2法则的信息管理
 * 3. 置信度反馈可视化 - 算法"思考"过程视觉展现
 * 4. 语义理解可视化 - 语义分析结果空间布局
 */

// 数据类型定义
interface Entity {
  id: string
  content: string
  type: 'number' | 'object' | 'person' | 'operation' | 'result' | 'constraint'
  semanticWeight: number
  cognitiveLoad: number
  attentionPriority: 'primary' | 'secondary' | 'background'
  position?: { x: number; y: number }
  confidence?: number
}

interface Relationship {
  id: string
  source: string
  target: string
  type: 'arithmetic' | 'ownership' | 'constraint' | 'causal'
  strength: number
  confidence: number
  discoveryStep?: number
}

interface AlgorithmStep {
  id: string
  step: number
  type: 'entity_recognition' | 'relation_discovery' | 'semantic_analysis' | 'confidence_calculation'
  description: string
  highlightedElements: string[]
  confidence: number
  timestamp: number
}

interface AlgorithmicEntityDiagramProps {
  entities: Entity[]
  relationships: Relationship[]
  algorithmSteps?: AlgorithmStep[]
  enableRealTimeVisualization?: boolean
  cognitiveLoadThreshold?: number
  onStepComplete?: (step: AlgorithmStep) => void
}

// 认知负荷管理器
class CognitiveLoadManager {
  private maxPrimaryNodes = 5  // 7±2法则的下限
  private maxSecondaryNodes = 7
  
  assessCognitiveLoad(entities: Entity[]): {
    level: 'low' | 'medium' | 'high'
    recommendations: string[]
  } {
    const totalLoad = entities.reduce((sum, entity) => sum + entity.cognitiveLoad, 0)
    const avgLoad = totalLoad / entities.length
    
    let level: 'low' | 'medium' | 'high' = 'low'
    const recommendations: string[] = []
    
    if (avgLoad > 0.7) {
      level = 'high'
      recommendations.push('建议分层显示信息')
      recommendations.push('减少同时显示的主要元素')
    } else if (avgLoad > 0.4) {
      level = 'medium'
      recommendations.push('适当分组相关元素')
    }
    
    if (entities.filter(e => e.attentionPriority === 'primary').length > this.maxPrimaryNodes) {
      recommendations.push(`主要焦点元素建议不超过${this.maxPrimaryNodes}个`)
    }
    
    return { level, recommendations }
  }
  
  optimizeAttentionPriority(entities: Entity[]): Entity[] {
    // 按语义权重和认知负荷排序
    const sorted = [...entities].sort((a, b) => 
      (b.semanticWeight - b.cognitiveLoad) - (a.semanticWeight - a.cognitiveLoad)
    )
    
    return sorted.map((entity, index) => ({
      ...entity,
      attentionPriority: 
        index < this.maxPrimaryNodes ? 'primary' :
        index < this.maxPrimaryNodes + this.maxSecondaryNodes ? 'secondary' :
        'background'
    }))
  }
}

// 语义空间映射器
class SemanticSpatialMapper {
  private width = 800
  private height = 600
  
  // 语义区域定义
  private semanticRegions = {
    input: { center: [160, 180], radius: 120, color: '#E3F2FD' },      // 输入区域（左侧）
    process: { center: [400, 300], radius: 150, color: '#E8F5E8' },    // 处理区域（中央）
    output: { center: [640, 180], radius: 120, color: '#FFF3E0' },     // 输出区域（右侧）
    constraint: { center: [400, 480], radius: 100, color: '#FCE4EC' }   // 约束区域（下方）
  } as const
  
  mapEntitiesToSemanticSpace(entities: Entity[]): Entity[] {
    return entities.map(entity => {
      const position = this.getSemanticPosition(entity)
      return { ...entity, position }
    })
  }
  
  private getSemanticPosition(entity: Entity): { x: number; y: number } {
    // 根据实体类型分配到不同语义区域
    let regionKey: keyof typeof this.semanticRegions = 'process'
    
    switch (entity.type) {
      case 'number':
      case 'object':
      case 'person':
        regionKey = 'input'
        break
      case 'operation':
        regionKey = 'process'
        break
      case 'result':
        regionKey = 'output'
        break
      case 'constraint':
        regionKey = 'constraint'
        break
    }
    
    const region = this.semanticRegions[regionKey]
    const [centerX, centerY] = region.center
    const radius = region.radius
    
    // 在区域内随机分布，但避免重叠
    const angle = Math.random() * 2 * Math.PI
    const distance = Math.random() * radius * 0.8
    
    return {
      x: centerX + distance * Math.cos(angle),
      y: centerY + distance * Math.sin(angle)
    }
  }
  
  getSemanticRegions() {
    return this.semanticRegions
  }
}

// 算法过程可视化组件
const AlgorithmProcessVisualizer: React.FC<{
  steps: AlgorithmStep[]
  currentStep: number
  onStepChange: (step: number) => void
}> = ({ steps, currentStep, onStepChange }) => {
  return (
    <div className="algorithm-process-panel">
      <h4 className="text-sm font-semibold mb-3 text-gray-700">🧠 算法执行过程</h4>
      
      {/* 进度条 */}
      <div className="mb-4">
        <div className="w-full bg-gray-200 rounded-full h-2">
          <motion.div 
            className="bg-blue-500 h-2 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
        <div className="text-xs text-gray-500 mt-1">
          步骤 {currentStep + 1} / {steps.length}
        </div>
      </div>
      
      {/* 步骤列表 */}
      <div className="space-y-2 max-h-40 overflow-y-auto">
        {steps.map((step, index) => (
          <motion.div
            key={step.id}
            className={`p-2 rounded cursor-pointer transition-colors ${
              index === currentStep 
                ? 'bg-blue-100 border-l-4 border-blue-500' 
                : index < currentStep
                ? 'bg-green-50 border-l-4 border-green-400'
                : 'bg-gray-50 hover:bg-gray-100'
            }`}
            onClick={() => onStepChange(index)}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm font-medium">
                  {step.type === 'entity_recognition' && '🔍 实体识别'}
                  {step.type === 'relation_discovery' && '🔗 关系发现'}
                  {step.type === 'semantic_analysis' && '🧠 语义分析'}
                  {step.type === 'confidence_calculation' && '📊 置信度计算'}
                </div>
                <div className="text-xs text-gray-600">{step.description}</div>
              </div>
              <div className="text-xs text-blue-600 font-medium">
                {(step.confidence * 100).toFixed(0)}%
              </div>
            </div>
          </motion.div>
        ))}
      </div>
      
      {/* 控制按钮 */}
      <div className="flex space-x-2 mt-4">
        <button
          onClick={() => onStepChange(Math.max(0, currentStep - 1))}
          disabled={currentStep === 0}
          className="px-3 py-1 text-xs bg-gray-200 hover:bg-gray-300 disabled:opacity-50 rounded"
        >
          上一步
        </button>
        <button
          onClick={() => onStepChange(Math.min(steps.length - 1, currentStep + 1))}
          disabled={currentStep >= steps.length - 1}
          className="px-3 py-1 text-xs bg-blue-500 hover:bg-blue-600 disabled:opacity-50 text-white rounded"
        >
          下一步
        </button>
      </div>
    </div>
  )
}

// 认知负荷指示器
const CognitiveLoadIndicator: React.FC<{
  level: 'low' | 'medium' | 'high'
  recommendations: string[]
}> = ({ level, recommendations }) => {
  const getColorAndIcon = () => {
    switch (level) {
      case 'low': return { color: 'text-green-600 bg-green-50', icon: '😊', label: '认知负荷: 低' }
      case 'medium': return { color: 'text-yellow-600 bg-yellow-50', icon: '🤔', label: '认知负荷: 中等' }
      case 'high': return { color: 'text-red-600 bg-red-50', icon: '😵', label: '认知负荷: 高' }
    }
  }
  
  const { color, icon, label } = getColorAndIcon()
  
  return (
    <div className={`p-3 rounded-lg ${color}`}>
      <div className="flex items-center space-x-2 mb-2">
        <span className="text-lg">{icon}</span>
        <span className="font-medium text-sm">{label}</span>
      </div>
      
      {recommendations.length > 0 && (
        <div className="text-xs space-y-1">
          <div className="font-medium">💡 优化建议:</div>
          {recommendations.map((rec, index) => (
            <div key={index} className="flex items-start space-x-1">
              <span>•</span>
              <span>{rec}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// 置信度反馈可视化
const ConfidenceFeedback: React.FC<{
  entities: Entity[]
  relationships: Relationship[]
  overallConfidence: number
}> = ({ entities, relationships, overallConfidence }) => {
  return (
    <div className="confidence-feedback-panel">
      <h4 className="text-sm font-semibold mb-3 text-gray-700">📊 置信度反馈</h4>
      
      {/* 整体置信度 */}
      <div className="mb-4">
        <div className="flex justify-between items-center mb-1">
          <span className="text-sm">整体置信度</span>
          <span className="text-sm font-bold text-blue-600">
            {(overallConfidence * 100).toFixed(1)}%
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <motion.div
            className={`h-2 rounded-full ${
              overallConfidence >= 0.8 ? 'bg-green-500' :
              overallConfidence >= 0.6 ? 'bg-yellow-500' :
              'bg-red-500'
            }`}
            initial={{ width: 0 }}
            animate={{ width: `${overallConfidence * 100}%` }}
            transition={{ duration: 1 }}
          />
        </div>
      </div>
      
      {/* 实体置信度 */}
      <div className="mb-3">
        <div className="text-xs font-medium text-gray-600 mb-1">实体识别置信度</div>
        <div className="space-y-1 max-h-24 overflow-y-auto">
          {entities.filter(e => e.confidence !== undefined).map(entity => (
            <div key={entity.id} className="flex justify-between items-center text-xs">
              <span className="truncate max-w-20">{entity.content}</span>
              <div className="flex items-center space-x-1">
                <div className="w-12 bg-gray-200 rounded-full h-1">
                  <div 
                    className="bg-blue-400 h-1 rounded-full"
                    style={{ width: `${(entity.confidence || 0) * 100}%` }}
                  />
                </div>
                <span>{((entity.confidence || 0) * 100).toFixed(0)}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      {/* 关系置信度 */}
      <div>
        <div className="text-xs font-medium text-gray-600 mb-1">关系发现置信度</div>
        <div className="space-y-1 max-h-24 overflow-y-auto">
          {relationships.map(rel => (
            <div key={rel.id} className="flex justify-between items-center text-xs">
              <span className="truncate max-w-20">{rel.type}</span>
              <div className="flex items-center space-x-1">
                <div className="w-12 bg-gray-200 rounded-full h-1">
                  <div 
                    className="bg-green-400 h-1 rounded-full"
                    style={{ width: `${rel.confidence * 100}%` }}
                  />
                </div>
                <span>{(rel.confidence * 100).toFixed(0)}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// 主组件
const AlgorithmicEntityDiagram: React.FC<AlgorithmicEntityDiagramProps> = ({
  entities: initialEntities,
  relationships: initialRelationships,
  algorithmSteps = [],
  enableRealTimeVisualization = true,
  cognitiveLoadThreshold = 0.6,
  onStepComplete
}) => {
  const [entities, setEntities] = useState<Entity[]>(initialEntities)
  const [relationships, setRelationships] = useState<Relationship[]>(initialRelationships)
  const [currentStep, setCurrentStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  
  const cognitiveManager = useRef(new CognitiveLoadManager())
  const spatialMapper = useRef(new SemanticSpatialMapper())
  
  // 认知负荷评估
  const cognitiveLoad = cognitiveManager.current.assessCognitiveLoad(entities)
  
  // 整体置信度计算
  const overallConfidence = entities.reduce((sum, e) => sum + (e.confidence || 0), 0) / entities.length || 0
  
  // 语义空间映射
  const mappedEntities = spatialMapper.current.mapEntitiesToSemanticSpace(
    cognitiveManager.current.optimizeAttentionPriority(entities)
  )
  
  // 语义区域
  const semanticRegions = spatialMapper.current.getSemanticRegions()
  
  // 自动播放算法步骤
  useEffect(() => {
    if (isPlaying && currentStep < algorithmSteps.length - 1) {
      const timer = setTimeout(() => {
        setCurrentStep(prev => prev + 1)
        if (onStepComplete && algorithmSteps[currentStep + 1]) {
          onStepComplete(algorithmSteps[currentStep + 1])
        }
      }, 2000)
      return () => clearTimeout(timer)
    } else {
      setIsPlaying(false)
    }
  }, [isPlaying, currentStep, algorithmSteps, onStepComplete])
  
  // 当前步骤的高亮元素
  const currentStepHighlights = algorithmSteps[currentStep]?.highlightedElements || []
  
  // 渲染实体节点
  const renderEntity = (entity: Entity) => {
    const isHighlighted = currentStepHighlights.includes(entity.id)
    const position = entity.position || { x: 400, y: 300 }
    
    // 根据注意力优先级确定样式
    const getEntityStyles = () => {
      switch (entity.attentionPriority) {
        case 'primary':
          return {
            size: 60,
            opacity: 1,
            borderWidth: 3,
            glowIntensity: 0.8
          }
        case 'secondary':
          return {
            size: 45,
            opacity: 0.8,
            borderWidth: 2,
            glowIntensity: 0.4
          }
        case 'background':
          return {
            size: 30,
            opacity: 0.5,
            borderWidth: 1,
            glowIntensity: 0
          }
      }
    }
    
    const styles = getEntityStyles()
    
    // 实体类型颜色映射
    const getEntityColor = (type: Entity['type']) => {
      const colors = {
        number: '#4CAF50',
        object: '#FF9800', 
        person: '#9C27B0',
        operation: '#2196F3',
        result: '#F44336',
        constraint: '#607D8B'
      }
      return colors[type]
    }
    
    return (
      <motion.g
        key={entity.id}
        initial={{ opacity: 0, scale: 0 }}
        animate={{ 
          opacity: styles.opacity,
          scale: 1,
          x: position.x,
          y: position.y
        }}
        transition={{ duration: 0.5 }}
      >
        {/* 高亮光晕效果 */}
        {isHighlighted && (
          <motion.circle
            cx={0}
            cy={0}
            r={styles.size * 0.8}
            fill="none"
            stroke="#FFD700"
            strokeWidth="3"
            opacity={0.6}
            animate={{
              r: [styles.size * 0.8, styles.size * 1.2, styles.size * 0.8],
              opacity: [0.6, 0.3, 0.6]
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
        )}
        
        {/* 置信度脉动效果 */}
        {entity.confidence && entity.confidence > 0.8 && (
          <motion.circle
            cx={0}
            cy={0}
            r={styles.size * 0.6}
            fill="none"
            stroke={getEntityColor(entity.type)}
            strokeWidth="1"
            opacity={0.3}
            animate={{
              r: [styles.size * 0.6, styles.size * 0.9, styles.size * 0.6],
              opacity: [0.3, 0.1, 0.3]
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
        )}
        
        {/* 主要节点 */}
        <circle
          cx={0}
          cy={0}
          r={styles.size / 2}
          fill={getEntityColor(entity.type)}
          fillOpacity={0.8}
          stroke={isHighlighted ? "#FFD700" : getEntityColor(entity.type)}
          strokeWidth={styles.borderWidth}
          style={{
            filter: styles.glowIntensity > 0 ? 
              `drop-shadow(0 0 ${styles.glowIntensity * 10}px ${getEntityColor(entity.type)})` : 
              'none'
          }}
        />
        
        {/* 文本标签 */}
        <text
          x={0}
          y={5}
          textAnchor="middle"
          fontSize={12}
          fill="white"
          fontWeight="bold"
        >
          {entity.content.length > 6 ? entity.content.slice(0, 6) + '...' : entity.content}
        </text>
        
        {/* 置信度指示器 */}
        {entity.confidence !== undefined && (
          <text
            x={0}
            y={styles.size / 2 + 15}
            textAnchor="middle"
            fontSize={10}
            fill={getEntityColor(entity.type)}
            fontWeight="bold"
          >
            {(entity.confidence * 100).toFixed(0)}%
          </text>
        )}
      </motion.g>
    )
  }
  
  // 渲染关系边
  const renderRelationship = (relationship: Relationship) => {
    const sourceEntity = mappedEntities.find(e => e.id === relationship.source)
    const targetEntity = mappedEntities.find(e => e.id === relationship.target)
    
    if (!sourceEntity || !targetEntity || !sourceEntity.position || !targetEntity.position) {
      return null
    }
    
    const isHighlighted = currentStepHighlights.includes(relationship.id)
    const shouldShow = !relationship.discoveryStep || relationship.discoveryStep <= currentStep
    
    if (!shouldShow) return null
    
    return (
      <motion.g
        key={relationship.id}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        {/* 置信度流动动画 */}
        {relationship.confidence > 0.7 && (
          <motion.circle
            r="3"
            fill="#FFD700"
            opacity={0.8}
            animate={{
              cx: [sourceEntity.position.x, targetEntity.position.x],
              cy: [sourceEntity.position.y, targetEntity.position.y]
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              ease: "linear"
            }}
          />
        )}
        
        <line
          x1={sourceEntity.position.x}
          y1={sourceEntity.position.y}
          x2={targetEntity.position.x}
          y2={targetEntity.position.y}
          stroke={isHighlighted ? "#FFD700" : "#666"}
          strokeWidth={Math.max(1, relationship.strength * 4)}
          strokeOpacity={relationship.confidence}
          markerEnd="url(#arrowhead)"
        />
        
        {/* 关系标签 */}
        <text
          x={(sourceEntity.position.x + targetEntity.position.x) / 2}
          y={(sourceEntity.position.y + targetEntity.position.y) / 2}
          textAnchor="middle"
          fontSize={10}
          fill="#333"
          dy="-5"
        >
          {relationship.type}
        </text>
      </motion.g>
    )
  }
  
  return (
    <div className="w-full h-full flex">
      {/* 主可视化区域 */}
      <div className="flex-1 relative">
        <svg 
          viewBox="0 0 800 600" 
          className="w-full h-full border rounded-lg bg-gradient-to-br from-blue-50 to-indigo-50"
        >
          {/* 箭头标记定义 */}
          <defs>
            <marker
              id="arrowhead"
              markerWidth="10"
              markerHeight="7"
              refX="9"
              refY="3.5"
              orient="auto"
            >
              <polygon
                points="0 0, 10 3.5, 0 7"
                fill="#666"
              />
            </marker>
          </defs>
          
          {/* 语义区域背景 */}
          {Object.entries(semanticRegions).map(([key, region]) => (
            <motion.circle
              key={key}
              cx={region.center[0]}
              cy={region.center[1]}
              r={region.radius}
              fill={region.color}
              fillOpacity={0.2}
              stroke={region.color.replace('50', '200')}
              strokeWidth={1}
              strokeDasharray="5,5"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
            />
          ))}
          
          {/* 区域标签 */}
          <text x={160} y={120} textAnchor="middle" fontSize={14} fill="#666" fontWeight="bold">
            📥 输入区域
          </text>
          <text x={400} y={200} textAnchor="middle" fontSize={14} fill="#666" fontWeight="bold">
            ⚙️ 处理区域
          </text>
          <text x={640} y={120} textAnchor="middle" fontSize={14} fill="#666" fontWeight="bold">
            📤 输出区域
          </text>
          <text x={400} y={420} textAnchor="middle" fontSize={14} fill="#666" fontWeight="bold">
            🔒 约束区域
          </text>
          
          {/* 关系线 */}
          {initialRelationships.map(renderRelationship)}
          
          {/* 实体节点 */}
          {mappedEntities.map(renderEntity)}
        </svg>
        
        {/* 播放控制 */}
        {algorithmSteps.length > 0 && (
          <div className="absolute bottom-4 left-4">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className={`px-4 py-2 rounded-lg text-white font-medium ${
                isPlaying ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'
              }`}
            >
              {isPlaying ? '⏸️ 暂停' : '▶️ 播放'} 算法过程
            </button>
          </div>
        )}
      </div>
      
      {/* 右侧控制面板 */}
      <div className="w-80 p-4 bg-gray-50 border-l space-y-4 overflow-y-auto">
        {/* 算法过程可视化 */}
        {algorithmSteps.length > 0 && (
          <AlgorithmProcessVisualizer
            steps={algorithmSteps}
            currentStep={currentStep}
            onStepChange={setCurrentStep}
          />
        )}
        
        {/* 认知负荷指示器 */}
        <CognitiveLoadIndicator
          level={cognitiveLoad.level}
          recommendations={cognitiveLoad.recommendations}
        />
        
        {/* 置信度反馈 */}
        <ConfidenceFeedback
          entities={entities}
          relationships={relationships}
          overallConfidence={overallConfidence}
        />
        
        {/* 统计信息 */}
        <div className="p-3 bg-white rounded-lg border">
          <h4 className="text-sm font-semibold mb-2 text-gray-700">📈 统计信息</h4>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span>实体数量:</span>
              <span className="font-medium">{entities.length}</span>
            </div>
            <div className="flex justify-between">
              <span>关系数量:</span>
              <span className="font-medium">{relationships.length}</span>
            </div>
            <div className="flex justify-between">
              <span>主要焦点:</span>
              <span className="font-medium">
                {entities.filter(e => e.attentionPriority === 'primary').length}
              </span>
            </div>
            <div className="flex justify-between">
              <span>语义区域:</span>
              <span className="font-medium">{Object.keys(semanticRegions).length}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default AlgorithmicEntityDiagram