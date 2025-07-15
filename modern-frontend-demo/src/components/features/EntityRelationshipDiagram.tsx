import React, { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Entity, Relationship, DeepRelation, ImplicitConstraint } from '@/stores/problemStore'

interface EntityRelationshipDiagramProps {
  entities: Entity[]
  relationships: Relationship[]
  physicalConstraints?: string[]
  physicalProperties?: {
    conservationLaws: string[]
    spatialRelations: string[]
    temporalConstraints: string[]
    materialProperties: string[]
  }
  // 新增：深度隐含关系增强属性
  deepRelations?: DeepRelation[]
  implicitConstraints?: ImplicitConstraint[]
  visualizationConfig?: {
    show_depth_indicators: boolean
    show_constraint_panels: boolean
    enable_interactive_exploration: boolean
    animation_sequence: boolean
  }
  width?: number
  height?: number
}

interface Position {
  x: number
  y: number
}

interface EntityWithPosition extends Entity {
  position: Position
}

// 物性关系情景图组件
const EntityRelationshipDiagram: React.FC<EntityRelationshipDiagramProps> = ({
  entities,
  relationships,
  physicalConstraints = [],
  physicalProperties,
  deepRelations = [],
  implicitConstraints = [],
  visualizationConfig = {
    show_depth_indicators: true,
    show_constraint_panels: true,
    enable_interactive_exploration: true,
    animation_sequence: true
  },
  width = 800,
  height = 600
}) => {
  const svgRef = useRef<SVGSVGElement>(null)
  const [entitiesWithPositions, setEntitiesWithPositions] = useState<EntityWithPosition[]>([])
  const [hoveredEntity, setHoveredEntity] = useState<string | null>(null)
  const [selectedDepthLayer, setSelectedDepthLayer] = useState<string>('all')
  const [showConstraintDetails, setShowConstraintDetails] = useState<boolean>(false)

  // 计算节点位置 - 使用物性关系优化的布局
  const calculatePositions = (entities: Entity[]): EntityWithPosition[] => {
    const centerX = width / 2
    const centerY = height / 2
    const radius = Math.min(width, height) / 3

    if (entities.length === 0) {
      return []
    }

    if (entities.length === 1) {
      return [{ ...entities[0], position: { x: centerX, y: centerY } }]
    }

    // 物性关系导向的布局：按实体类型分层
    const personEntities = entities.filter(e => e.type === 'person')
    const objectEntities = entities.filter(e => e.type === 'object')
    const conceptEntities = entities.filter(e => e.type === 'concept')
    const moneyEntities = entities.filter(e => e.type === 'money')

    const positioned: EntityWithPosition[] = []

    // 人物实体放在上方
    personEntities.forEach((entity, index) => {
      const x = centerX + (index - (personEntities.length - 1) / 2) * 120
      const y = centerY - 150
      positioned.push({ ...entity, position: { x, y } })
    })

    // 物品实体放在中间
    objectEntities.forEach((entity, index) => {
      const x = centerX + (index - (objectEntities.length - 1) / 2) * 100
      const y = centerY
      positioned.push({ ...entity, position: { x, y } })
    })

    // 概念实体放在下方
    conceptEntities.forEach((entity, index) => {
      const x = centerX + (index - (conceptEntities.length - 1) / 2) * 140
      const y = centerY + 150
      positioned.push({ ...entity, position: { x, y } })
    })

    // 货币实体放在右侧
    moneyEntities.forEach((entity, index) => {
      const x = centerX + 200
      const y = centerY + (index - (moneyEntities.length - 1) / 2) * 80
      positioned.push({ ...entity, position: { x, y } })
    })

    return positioned
  }

  useEffect(() => {
    const positioned = calculatePositions(entities)
    setEntitiesWithPositions(positioned)
  }, [entities, width, height])

  // 如果没有数据，显示物性关系说明
  if (!entities || entities.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>🔬 物性关系情景图</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12 text-gray-500">
            <div className="text-6xl mb-4">⚗️</div>
            <div className="text-lg font-medium mb-2">暂无物性关系数据</div>
            <div className="text-sm mb-4">请先解决一个数学问题来生成物性关系情景图</div>
            <div className="bg-blue-50 p-4 rounded-lg text-left max-w-md mx-auto">
              <div className="text-sm text-blue-800">
                <strong>物性关系包括：</strong>
                <ul className="mt-2 space-y-1">
                  <li>• 拥有关系 - 实体对物体的所有权</li>
                  <li>• 物理守恒 - 物质不灭定律</li>
                  <li>• 数量约束 - 非负整数性质</li>
                  <li>• 空间关系 - 实体的空间分布</li>
                  <li>• 状态转移 - 物理状态的变化</li>
                </ul>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  // 获取物性关系的实体颜色
  const getPhysicalEntityColor = (type: Entity['type']): string => {
    const physicalColors = {
      person: '#e74c3c',    // 红色 - 具有能动性的实体
      object: '#27ae60',    // 绿色 - 物理实体
      money: '#f39c12',     // 橙色 - 价值载体
      concept: '#9b59b6'    // 紫色 - 抽象概念
    }
    return physicalColors[type] || '#6b7280'
  }

  // 获取物性关系的实体图标
  const getPhysicalEntityIcon = (type: Entity['type']): string => {
    const physicalIcons = {
      person: '👤',
      object: '🧮',  // 使用算盘表示可计算的物理对象
      money: '💰',
      concept: '⚛️'  // 使用原子符号表示抽象概念
    }
    return physicalIcons[type] || '🔷'
  }

  // 获取物性关系描述
  const getPhysicalRelationLabel = (relationship: Relationship): string => {
    const { type, weight } = relationship
    if (type.includes('拥有')) return `拥有关系 (${weight || 1}个)`
    if (type.includes('聚合') || type.includes('总')) return `聚合关系 (守恒)`
    if (type.includes('购买')) return `交易关系 (价值转移)`
    if (type.includes('几何')) return `几何关系 (公式)`
    return `物性关系 (${(weight || 0.5) * 100}%)`
  }

  // 计算物性关系连接线路径
  const calculatePhysicalPath = (source: EntityWithPosition, target: EntityWithPosition) => {
    const dx = target.position.x - source.position.x
    const dy = target.position.y - source.position.y
    const distance = Math.sqrt(dx * dx + dy * dy)
    
    // 计算节点边缘的连接点
    const radius = 40
    const sourceX = source.position.x + (dx / distance) * radius
    const sourceY = source.position.y + (dy / distance) * radius
    const targetX = target.position.x - (dx / distance) * radius
    const targetY = target.position.y - (dy / distance) * radius
    
    // 如果是物性关系，使用曲线路径表示物理作用
    const midX = (sourceX + targetX) / 2
    const midY = (sourceY + targetY) / 2
    const controlX = midX + (dy / distance) * 30  // 垂直偏移创建曲线
    const controlY = midY - (dx / distance) * 30
    
    return `M ${sourceX} ${sourceY} Q ${controlX} ${controlY} ${targetX} ${targetY}`
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>🔬 物性关系情景图</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="relative">
          <svg
            ref={svgRef}
            width={width}
            height={height}
            className="border border-gray-200 rounded-lg bg-gradient-to-br from-blue-50 to-purple-50"
            viewBox={`0 0 ${width} ${height}`}
          >
            {/* 定义物性关系箭头标记 */}
            <defs>
              <marker
                id="physicalArrow"
                markerWidth="12"
                markerHeight="8"
                refX="10"
                refY="4"
                orient="auto"
              >
                <polygon
                  points="0 0, 12 4, 0 8"
                  fill="#2563eb"
                  stroke="#1d4ed8"
                  strokeWidth="1"
                />
              </marker>
              
              {/* 物理作用力指示器 */}
              <marker
                id="forceIndicator"
                markerWidth="8"
                markerHeight="8"
                refX="4"
                refY="4"
                orient="auto"
              >
                <circle cx="4" cy="4" r="3" fill="#ef4444" opacity="0.8" />
              </marker>
            </defs>

          {/* 深度隐含关系可视化层 */}
          {visualizationConfig.show_depth_indicators && deepRelations.map((deepRel, index) => {
            const sourceEntity = entitiesWithPositions.find(e => e.id === deepRel.source)
            const targetEntity = entitiesWithPositions.find(e => e.id === deepRel.target)
            
            if (!sourceEntity || !targetEntity) return null
            
            // 根据深度层级过滤
            if (selectedDepthLayer !== 'all' && deepRel.depth !== selectedDepthLayer) return null

            const path = calculatePhysicalPath(sourceEntity, targetEntity)
            const midX = (sourceEntity.position.x + targetEntity.position.x) / 2
            const midY = (sourceEntity.position.y + targetEntity.position.y) / 2

            return (
              <g key={`deep-${index}`}>
                <motion.path
                  d={path}
                  stroke={deepRel.visualization.depth_color}
                  strokeWidth={deepRel.visualization.relation_width}
                  fill="none"
                  strokeDasharray={deepRel.depth === 'deep' ? "8,4" : "none"}
                  markerEnd="url(#physicalArrow)"
                  initial={{ pathLength: 0, opacity: 0 }}
                  animate={{ pathLength: 1, opacity: 0.9 }}
                  transition={{ 
                    duration: visualizationConfig.animation_sequence ? 1.2 : 0.5, 
                    delay: deepRel.visualization.animation_delay 
                  }}
                  className="cursor-pointer drop-shadow-sm"
                  onMouseEnter={() => setHoveredEntity(`deep-${deepRel.id}`)}
                  onMouseLeave={() => setHoveredEntity(null)}
                />
                <motion.text
                  x={midX}
                  y={midY - 12}
                  textAnchor="middle"
                  className="text-xs font-medium"
                  fill={deepRel.visualization.depth_color}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 1.0 + deepRel.visualization.animation_delay }}
                >
                  {deepRel.label}
                </motion.text>
                
                {/* 深度指示器 */}
                <motion.circle
                  cx={midX + 20}
                  cy={midY}
                  r={3}
                  fill={deepRel.visualization.depth_color}
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 1.2 + deepRel.visualization.animation_delay }}
                  className="cursor-pointer"
                >
                  <title>{`深度: ${deepRel.depth}, 置信度: ${(deepRel.confidence * 100).toFixed(1)}%`}</title>
                </motion.circle>
              </g>
            )
          })}

          {/* 绘制物性关系连接线 */}
          {relationships.map((rel, index) => {
            const sourceEntity = entitiesWithPositions.find(e => e.id === rel.source)
            const targetEntity = entitiesWithPositions.find(e => e.id === rel.target)
            
            if (!sourceEntity || !targetEntity) return null

            const path = calculatePhysicalPath(sourceEntity, targetEntity)
            const midX = (sourceEntity.position.x + targetEntity.position.x) / 2
            const midY = (sourceEntity.position.y + targetEntity.position.y) / 2

            return (
              <g key={index}>
                <motion.path
                  d={path}
                  stroke="#2563eb"
                  strokeWidth="3"
                  fill="none"
                  markerEnd="url(#physicalArrow)"
                  markerMid="url(#forceIndicator)"
                  initial={{ pathLength: 0, opacity: 0 }}
                  animate={{ pathLength: 1, opacity: 0.8 }}
                  transition={{ duration: 1.0, delay: index * 0.2 }}
                  className="drop-shadow-sm"
                />
                <motion.text
                  x={midX}
                  y={midY - 8}
                  textAnchor="middle"
                  className="text-xs font-medium fill-blue-700"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.8 + index * 0.2 }}
                >
                  {getPhysicalRelationLabel(rel)}
                </motion.text>
              </g>
            )
          })}

          {/* 绘制物性实体节点 */}
          {entitiesWithPositions.map((entity, index) => (
            <g key={entity.id}>
              <motion.circle
                cx={entity.position.x}
                cy={entity.position.y}
                r={hoveredEntity === entity.id ? 45 : 40}
                fill={getPhysicalEntityColor(entity.type)}
                stroke="#fff"
                strokeWidth="4"
                className="cursor-pointer drop-shadow-lg"
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 0.9 }}
                transition={{ duration: 0.6, delay: index * 0.15 }}
                onMouseEnter={() => setHoveredEntity(entity.id)}
                onMouseLeave={() => setHoveredEntity(null)}
              />
              
              {/* 物性实体图标 */}
              <motion.text
                x={entity.position.x}
                y={entity.position.y - 8}
                textAnchor="middle"
                className="text-lg pointer-events-none"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4 + index * 0.15 }}
              >
                {getPhysicalEntityIcon(entity.type)}
              </motion.text>
              
              {/* 物性实体名称 */}
              <motion.text
                x={entity.position.x}
                y={entity.position.y + 12}
                textAnchor="middle"
                className="text-xs font-bold fill-white pointer-events-none"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4 + index * 0.15 }}
              >
                {entity.name}
              </motion.text>
              
              {/* 物性实体类型标签 */}
              <motion.text
                x={entity.position.x}
                y={entity.position.y + 24}
                textAnchor="middle"
                className="text-xs fill-white opacity-80 pointer-events-none"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.6 + index * 0.15 }}
              >
                {entity.type === 'person' ? '能动实体' : 
                 entity.type === 'object' ? '物理实体' :
                 entity.type === 'concept' ? '抽象实体' : '价值实体'}
              </motion.text>
            </g>
          ))}
        </svg>

        {/* 深度层级控制器 */}
        {visualizationConfig.show_depth_indicators && deepRelations.length > 0 && (
          <div className="mt-4 bg-gradient-to-r from-purple-50 to-blue-50 p-4 rounded-lg border border-purple-200">
            <h4 className="text-sm font-medium text-purple-800 mb-3 flex items-center">
              ⚡ 深度关系层级控制
            </h4>
            <div className="flex flex-wrap gap-2">
              <button
                className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                  selectedDepthLayer === 'all' 
                    ? 'bg-purple-600 text-white' 
                    : 'bg-white text-purple-600 border border-purple-300'
                }`}
                onClick={() => setSelectedDepthLayer('all')}
              >
                显示全部
              </button>
              {['surface', 'shallow', 'medium', 'deep'].map(depth => {
                const depthCount = deepRelations.filter(r => r.depth === depth).length
                if (depthCount === 0) return null
                
                return (
                  <button
                    key={depth}
                    className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                      selectedDepthLayer === depth 
                        ? 'bg-purple-600 text-white' 
                        : 'bg-white text-purple-600 border border-purple-300'
                    }`}
                    onClick={() => setSelectedDepthLayer(depth)}
                  >
                    {depth === 'surface' ? '📄 表层' :
                     depth === 'shallow' ? '🔍 浅层' :
                     depth === 'medium' ? '🧠 中层' : '⚡ 深层'} ({depthCount})
                  </button>
                )
              })}
            </div>
          </div>
        )}

        {/* 物性关系图例 */}
        <div className="mt-6 bg-white p-4 rounded-lg border border-gray-200">
          <h4 className="text-sm font-medium text-gray-800 mb-3">🔬 物性关系图例</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="space-y-2">
              <div className="flex items-center space-x-3">
                <div className="w-4 h-4 rounded-full bg-red-500"></div>
                <span>👤 能动实体 - 具有主观能动性</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-4 h-4 rounded-full bg-green-500"></div>
                <span>🧮 物理实体 - 具有物理属性</span>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex items-center space-x-3">
                <div className="w-4 h-4 rounded-full bg-orange-500"></div>
                <span>💰 价值实体 - 价值载体媒介</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-4 h-4 rounded-full bg-purple-500"></div>
                <span>⚛️ 抽象实体 - 概念性存在</span>
              </div>
            </div>
          </div>
          
          <div className="mt-4 pt-3 border-t border-gray-200">
            <div className="text-xs text-gray-600">
              <strong>物性关系类型：</strong>
              <span className="ml-2">拥有关系 • 聚合关系 • 交易关系 • 几何关系 • 守恒关系</span>
            </div>
            {deepRelations.length > 0 && (
              <div className="text-xs text-purple-600 mt-1">
                <strong>深度关系：</strong>
                <span className="ml-2">
                  📄 表层 • 🔍 浅层 • 🧠 中层 • ⚡ 深层
                </span>
              </div>
            )}
          </div>
        </div>

        {/* 隐含约束展示面板 */}
        {visualizationConfig.show_constraint_panels && implicitConstraints.length > 0 && (
          <div className="mt-6 bg-gradient-to-r from-amber-50 to-orange-50 p-4 rounded-lg border border-amber-200">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-medium text-amber-800 flex items-center">
                🔒 隐含约束发现
              </h4>
              <button
                className="text-xs text-amber-600 hover:text-amber-800 transition-colors"
                onClick={() => setShowConstraintDetails(!showConstraintDetails)}
              >
                {showConstraintDetails ? '收起详情' : '展开详情'}
              </button>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {implicitConstraints.map((constraint, index) => (
                <motion.div
                  key={constraint.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-white p-3 rounded border-l-4"
                  style={{ borderLeftColor: constraint.color }}
                >
                  <div className="text-sm text-gray-800 flex items-start space-x-2">
                    <span className="text-lg">{constraint.icon}</span>
                    <div className="flex-1">
                      <div className="font-medium">{constraint.description}</div>
                      {showConstraintDetails && (
                        <div className="mt-2 text-xs text-gray-600">
                          <div><strong>表达式:</strong> {constraint.expression}</div>
                          <div><strong>影响实体:</strong> {constraint.entities.join(', ')}</div>
                          <div><strong>置信度:</strong> {(constraint.confidence * 100).toFixed(1)}%</div>
                        </div>
                      )}
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}

        {/* 物性约束展示面板 */}
        {physicalConstraints.length > 0 && (
          <div className="mt-6 bg-gradient-to-r from-blue-50 to-purple-50 p-4 rounded-lg border border-blue-200">
            <h4 className="text-sm font-medium text-blue-800 mb-3 flex items-center">
              ⚛️ 物性约束与守恒定律
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {physicalConstraints.map((constraint, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-white p-3 rounded border-l-4 border-blue-500"
                >
                  <div className="text-sm text-gray-800">
                    {constraint.includes('守恒') && '⚖️ '}
                    {constraint.includes('连续性') && '🔗 '}
                    {constraint.includes('拥有') && '🤝 '}
                    {constraint.includes('单调性') && '📈 '}
                    <span className="font-medium">{constraint}</span>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}

        {/* 物性属性分类展示 */}
        {physicalProperties && (
          <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
            {physicalProperties.conservationLaws.length > 0 && (
              <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                <h5 className="text-sm font-medium text-green-800 mb-2 flex items-center">
                  ⚖️ 守恒定律
                </h5>
                <ul className="space-y-1">
                  {physicalProperties.conservationLaws.map((law, index) => (
                    <li key={index} className="text-xs text-green-700">• {law}</li>
                  ))}
                </ul>
              </div>
            )}
            
            {physicalProperties.materialProperties.length > 0 && (
              <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                <h5 className="text-sm font-medium text-orange-800 mb-2 flex items-center">
                  🧱 物质属性
                </h5>
                <ul className="space-y-1">
                  {physicalProperties.materialProperties.map((prop, index) => (
                    <li key={index} className="text-xs text-orange-700">• {prop}</li>
                  ))}
                </ul>
              </div>
            )}
            
            {physicalProperties.spatialRelations.length > 0 && (
              <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                <h5 className="text-sm font-medium text-purple-800 mb-2 flex items-center">
                  📍 空间关系
                </h5>
                <ul className="space-y-1">
                  {physicalProperties.spatialRelations.map((relation, index) => (
                    <li key={index} className="text-xs text-purple-700">• {relation}</li>
                  ))}
                </ul>
              </div>
            )}
            
            {physicalProperties.temporalConstraints.length > 0 && (
              <div className="bg-pink-50 p-4 rounded-lg border border-pink-200">
                <h5 className="text-sm font-medium text-pink-800 mb-2 flex items-center">
                  ⏰ 时间约束
                </h5>
                <ul className="space-y-1">
                  {physicalProperties.temporalConstraints.map((constraint, index) => (
                    <li key={index} className="text-xs text-pink-700">• {constraint}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </CardContent>
  </Card>
)
}

export default EntityRelationshipDiagram