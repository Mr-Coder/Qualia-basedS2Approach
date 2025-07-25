import React from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'

interface SimpleEntity {
  id: string
  name: string
  type: 'person' | 'object' | 'money' | 'concept'
}

interface SimpleRelationship {
  source: string
  target: string
  type: string
  weight?: number
}

interface SimpleEntityDiagramProps {
  entities?: SimpleEntity[]
  relationships?: SimpleRelationship[]
}

// 极简版实体关系图，不使用任何第三方动画库
const SimpleEntityDiagram: React.FC<SimpleEntityDiagramProps> = ({
  entities = [],
  relationships = []
}) => {
  // 确保数据是数组
  const safeEntities = Array.isArray(entities) ? entities : []
  const safeRelationships = Array.isArray(relationships) ? relationships : []

  // 如果没有数据，显示默认内容
  if (safeEntities.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>🔬 实体关系图（简化版）</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500">
            <div className="text-4xl mb-4">📊</div>
            <p>暂无数据</p>
            <p className="text-sm mt-2">这是一个没有动画的简化版本</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  // 计算实体位置
  const width = 800
  const height = 400
  const centerX = width / 2
  const centerY = height / 2
  const radius = 150

  const entitiesWithPositions = safeEntities.map((entity, index) => {
    const angle = (index * 2 * Math.PI) / safeEntities.length
    return {
      ...entity,
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle)
    }
  })

  // 获取实体颜色
  const getEntityColor = (type: SimpleEntity['type']) => {
    const colors = {
      person: '#e74c3c',
      object: '#27ae60',
      money: '#f39c12',
      concept: '#9b59b6'
    }
    return colors[type] || '#6b7280'
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>🔬 实体关系图（简化版）</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="bg-gray-50 rounded-lg p-4 mb-4">
          <p className="text-sm text-gray-600">
            ⚠️ 这是一个不使用动画库的简化版本，用于调试目的
          </p>
        </div>
        
        <svg width={width} height={height} className="border border-gray-200 rounded-lg bg-white">
          {/* 绘制关系线 */}
          {safeRelationships.map((rel, index) => {
            const source = entitiesWithPositions.find(e => e.id === rel.source)
            const target = entitiesWithPositions.find(e => e.id === rel.target)
            
            if (!source || !target) return null
            
            return (
              <g key={`rel-${index}`}>
                <line
                  x1={source.x}
                  y1={source.y}
                  x2={target.x}
                  y2={target.y}
                  stroke="#94a3b8"
                  strokeWidth="2"
                />
                <text
                  x={(source.x + target.x) / 2}
                  y={(source.y + target.y) / 2 - 5}
                  textAnchor="middle"
                  fontSize="12"
                  fill="#64748b"
                >
                  {rel.type}
                </text>
              </g>
            )
          })}
          
          {/* 绘制实体节点 */}
          {entitiesWithPositions.map((entity) => (
            <g key={entity.id}>
              <circle
                cx={entity.x}
                cy={entity.y}
                r="40"
                fill={getEntityColor(entity.type)}
                stroke="#fff"
                strokeWidth="3"
              />
              <text
                x={entity.x}
                y={entity.y + 5}
                textAnchor="middle"
                fontSize="14"
                fill="white"
                fontWeight="bold"
              >
                {entity.name}
              </text>
            </g>
          ))}
        </svg>
        
        <div className="mt-4 text-xs text-gray-500">
          <p>实体数量: {safeEntities.length}</p>
          <p>关系数量: {safeRelationships.length}</p>
        </div>
      </CardContent>
    </Card>
  )
}

export default SimpleEntityDiagram