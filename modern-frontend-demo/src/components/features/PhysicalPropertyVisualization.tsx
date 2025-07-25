import React, { useEffect, useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { 
  getPhysicalPropertyDemo, 
  getPhysicalPropertyAnalysis,
  getBackendDrivenConfig,
  validatePhysicalConsistency,
  PhysicalGraphAnalysis,
  PhysicalProperty,
  PhysicalConstraint,
  PhysicalRelation
} from '@/services/physicalPropertyAPI'

interface PhysicalPropertyVisualizationProps {
  problemText?: string
  width?: number
  height?: number
  enableRealTimeUpdate?: boolean
  showValidation?: boolean
}

const PhysicalPropertyVisualization: React.FC<PhysicalPropertyVisualizationProps> = ({
  problemText,
  width = 800,
  height = 600,
  enableRealTimeUpdate = true,
  showValidation = true
}) => {
  const [physicalAnalysis, setPhysicalAnalysis] = useState<PhysicalGraphAnalysis | null>(null)
  const [backendConfig, setBackendConfig] = useState<any>(null)
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [selectedPropertyType, setSelectedPropertyType] = useState<string>('all')
  const [selectedConstraintType, setSelectedConstraintType] = useState<string>('all')
  const [validationResult, setValidationResult] = useState<any>(null)
  const [expandedProperty, setExpandedProperty] = useState<string | null>(null)

  // 物性属性颜色映射（基于后端驱动的配置）
  const getPropertyColor = (type: PhysicalProperty['type']): string => {
    const colors = {
      conservation: '#ef4444',    // 红色 - 守恒性
      discreteness: '#3b82f6',    // 蓝色 - 离散性
      continuity: '#10b981',      // 绿色 - 连续性
      additivity: '#f59e0b',      // 橙色 - 可加性
      measurability: '#8b5cf6',   // 紫色 - 可测性
      locality: '#06b6d4',        // 青色 - 局域性
      temporality: '#ec4899',     // 粉色 - 时序性
      causality: '#84cc16'        // 黄绿 - 因果性
    }
    return colors[type] || '#6b7280'
  }

  // 约束类型图标映射
  const getConstraintIcon = (type: PhysicalConstraint['type']): string => {
    const icons = {
      conservation_law: '⚖️',
      non_negative: '➕',
      integer_constraint: '#️⃣',
      upper_bound: '🔝',
      lower_bound: '🔻',
      equivalence: '⚖️',
      ordering: '📊',
      exclusivity: '🚫'
    }
    return icons[type] || '🔒'
  }

  // 从后端获取物性图谱分析
  const loadPhysicalAnalysis = useCallback(async () => {
    setIsLoading(true)
    try {
      let analysis: PhysicalGraphAnalysis | null = null
      
      if (problemText) {
        console.log('🔍 获取特定问题的物性图谱分析:', problemText)
        analysis = await getPhysicalPropertyAnalysis(problemText)
      } else {
        console.log('🔍 获取物性图谱演示数据')
        analysis = await getPhysicalPropertyDemo()
      }
      
      if (analysis) {
        setPhysicalAnalysis(analysis)
        
        // 验证物性一致性
        if (showValidation) {
          const validation = validatePhysicalConsistency(analysis)
          setValidationResult(validation)
        }
        
        console.log('✅ 物性图谱分析加载成功:', {
          properties: analysis.physical_properties.length,
          constraints: analysis.physical_constraints.length,
          relations: analysis.physical_relations.length,
          consistency: analysis.consistency_score
        })
      }
      
      // 获取后端驱动的配置
      const config = await getBackendDrivenConfig()
      if (config) {
        setBackendConfig(config)
        console.log('🎨 后端驱动配置加载成功:', config)
      }
      
    } catch (error) {
      console.error('❌ 物性图谱分析加载失败:', error)
    } finally {
      setIsLoading(false)
    }
  }, [problemText, showValidation])

  // 初始加载和定时更新
  useEffect(() => {
    loadPhysicalAnalysis()
    
    if (enableRealTimeUpdate) {
      const interval = setInterval(loadPhysicalAnalysis, 10000) // 10秒更新一次
      return () => clearInterval(interval)
    }
  }, [loadPhysicalAnalysis, enableRealTimeUpdate])

  // 筛选物性属性
  const filteredProperties = physicalAnalysis?.physical_properties.filter(prop => 
    selectedPropertyType === 'all' || prop.type === selectedPropertyType
  ) || []

  // 筛选约束条件
  const filteredConstraints = physicalAnalysis?.physical_constraints.filter(constraint => 
    selectedConstraintType === 'all' || constraint.type === selectedConstraintType
  ) || []

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>🔬 物性图谱可视化</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12">
            <div className="text-6xl mb-4 animate-spin">⚛️</div>
            <div className="text-lg font-medium mb-2">正在分析物性关系...</div>
            <div className="text-sm text-blue-600">后端物性推理引擎处理中</div>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!physicalAnalysis) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>🔬 物性图谱可视化</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12 text-gray-500">
            <div className="text-6xl mb-4">⚛️</div>
            <div className="text-lg font-medium mb-2">暂无物性图谱数据</div>
            <div className="text-sm mb-4">请先在智能求解模块解决问题以生成物性图谱</div>
            <button
              onClick={loadPhysicalAnalysis}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
            >
              重新加载
            </button>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      {/* 主要物性图谱展示 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>🔬 物性图谱可视化</span>
            <div className="flex items-center space-x-2">
              <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded-full">
                一致性: {(physicalAnalysis.consistency_score * 100).toFixed(1)}%
              </span>
              {backendConfig && (
                <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full">
                  后端驱动
                </span>
              )}
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {/* 筛选控制器 */}
          <div className="mb-6 bg-gray-50 p-4 rounded-lg">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium text-gray-700 mb-2 block">
                  物性属性类型
                </label>
                <select
                  value={selectedPropertyType}
                  onChange={(e) => setSelectedPropertyType(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                >
                  <option value="all">显示全部</option>
                  <option value="conservation">守恒性</option>
                  <option value="discreteness">离散性</option>
                  <option value="continuity">连续性</option>
                  <option value="additivity">可加性</option>
                  <option value="measurability">可测性</option>
                  <option value="locality">局域性</option>
                  <option value="temporality">时序性</option>
                  <option value="causality">因果性</option>
                </select>
              </div>
              
              <div>
                <label className="text-sm font-medium text-gray-700 mb-2 block">
                  约束条件类型
                </label>
                <select
                  value={selectedConstraintType}
                  onChange={(e) => setSelectedConstraintType(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                >
                  <option value="all">显示全部</option>
                  <option value="conservation_law">守恒定律</option>
                  <option value="non_negative">非负约束</option>
                  <option value="integer_constraint">整数约束</option>
                  <option value="upper_bound">上界约束</option>
                  <option value="lower_bound">下界约束</option>
                  <option value="equivalence">等价约束</option>
                  <option value="ordering">顺序约束</option>
                  <option value="exclusivity">排斥约束</option>
                </select>
              </div>
            </div>
          </div>

          {/* 物性属性网格 */}
          <div className="mb-6">
            <h4 className="text-lg font-medium text-gray-800 mb-4 flex items-center">
              ⚛️ 物性属性
              <span className="ml-2 text-sm text-gray-500">
                ({filteredProperties.length}/{physicalAnalysis.physical_properties.length})
              </span>
            </h4>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredProperties.map((property, index) => (
                <motion.div
                  key={property.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-white border rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow cursor-pointer"
                  style={{ borderLeftColor: getPropertyColor(property.type), borderLeftWidth: '4px' }}
                  onClick={() => setExpandedProperty(expandedProperty === property.id ? null : property.id)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-800">
                      {property.entity}
                    </span>
                    <div 
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: getPropertyColor(property.type) }}
                    />
                  </div>
                  
                  <div className="text-xs text-gray-600 mb-2">
                    类型: {property.type}
                  </div>
                  
                  <div className="text-sm text-gray-800 mb-2">
                    值: {typeof property.value === 'object' ? JSON.stringify(property.value) : property.value} {property.unit}
                  </div>
                  
                  <div className="text-xs text-green-600">
                    确定性: {(property.certainty * 100).toFixed(1)}%
                  </div>
                  
                  {expandedProperty === property.id && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      className="mt-3 pt-3 border-t border-gray-200"
                    >
                      <div className="text-xs text-gray-600">
                        <strong>约束条件:</strong>
                        <ul className="mt-1 space-y-1">
                          {property.constraints.map((constraint, i) => (
                            <li key={i} className="flex items-start">
                              <span className="text-blue-500 mr-1">•</span>
                              {constraint}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </motion.div>
                  )}
                </motion.div>
              ))}
            </div>
          </div>

          {/* 物性约束展示 */}
          <div className="mb-6">
            <h4 className="text-lg font-medium text-gray-800 mb-4 flex items-center">
              🔒 物性约束
              <span className="ml-2 text-sm text-gray-500">
                ({filteredConstraints.length}/{physicalAnalysis.physical_constraints.length})
              </span>
            </h4>
            
            <div className="space-y-3">
              {filteredConstraints.map((constraint, index) => (
                <motion.div
                  key={constraint.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-sm transition-shadow"
                >
                  <div className="flex items-start space-x-3">
                    <span className="text-2xl flex-shrink-0">
                      {getConstraintIcon(constraint.type)}
                    </span>
                    
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-gray-800">
                          {constraint.description}
                        </span>
                        <div className="flex items-center space-x-2">
                          <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">
                            {constraint.type}
                          </span>
                          <div className="w-full bg-gray-200 rounded-full h-2 max-w-20">
                            <div
                              className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                              style={{ width: `${constraint.strength * 100}%` }}
                            />
                          </div>
                        </div>
                      </div>
                      
                      <div className="text-xs text-gray-600 mb-2">
                        <strong>表达式:</strong> {constraint.expression}
                      </div>
                      
                      <div className="text-xs text-blue-600">
                        <strong>影响实体:</strong> {constraint.entities.join(', ')}
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>

          {/* 物性关系网络 */}
          <div className="mb-6">
            <h4 className="text-lg font-medium text-gray-800 mb-4 flex items-center">
              🔗 物性关系网络
              <span className="ml-2 text-sm text-gray-500">
                ({physicalAnalysis.physical_relations.length})
              </span>
            </h4>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {physicalAnalysis.physical_relations.map((relation, index) => (
                <motion.div
                  key={relation.id}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.05 }}
                  className="bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-lg p-3"
                >
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs font-bold">
                      {relation.source.slice(-1)}
                    </div>
                    <div className="flex-1">
                      <div className="text-xs text-blue-600 font-medium">
                        {relation.type}
                      </div>
                      <div className="text-xs text-gray-600">
                        {relation.physical_basis}
                      </div>
                    </div>
                    <div className="text-xs text-gray-500">
                      {(relation.strength * 100).toFixed(0)}%
                    </div>
                    <div className="w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center text-xs font-bold">
                      {relation.target.slice(-1)}
                    </div>
                  </div>
                  
                  {relation.causal_direction && (
                    <div className="mt-2 text-xs text-green-600 flex items-center">
                      <span className="mr-1">🔄</span>
                      因果方向: {relation.causal_direction}
                    </div>
                  )}
                </motion.div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 验证结果面板 */}
      {showValidation && validationResult && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <span className={validationResult.isConsistent ? 'text-green-600' : 'text-orange-600'}>
                {validationResult.isConsistent ? '✅' : '⚠️'} 物性一致性验证
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {validationResult.violations.length > 0 && (
              <div className="mb-4">
                <h5 className="text-sm font-medium text-red-600 mb-2">⚠️ 发现的违背:</h5>
                <ul className="space-y-1">
                  {validationResult.violations.map((violation: string, index: number) => (
                    <li key={index} className="text-sm text-red-600 flex items-start">
                      <span className="mr-2">•</span>
                      {violation}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            
            {validationResult.warnings.length > 0 && (
              <div>
                <h5 className="text-sm font-medium text-orange-600 mb-2">💡 建议:</h5>
                <ul className="space-y-1">
                  {validationResult.warnings.map((warning: string, index: number) => (
                    <li key={index} className="text-sm text-orange-600 flex items-start">
                      <span className="mr-2">•</span>
                      {warning}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            
            {validationResult.isConsistent && validationResult.violations.length === 0 && validationResult.warnings.length === 0 && (
              <div className="text-center py-4">
                <div className="text-green-600 text-sm">
                  ✅ 物性图谱一致性良好，未发现违背和警告
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* 后端驱动的优化信息 */}
      {backendConfig && (
        <Card>
          <CardHeader>
            <CardTitle>🎨 后端驱动的前端优化</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {backendConfig.backend_features && (
                <div>
                  <h5 className="text-sm font-medium text-blue-800 mb-3">🔧 后端特性</h5>
                  <div className="space-y-2">
                    {Object.entries(backendConfig.backend_features).map(([key, value]: [string, any]) => (
                      <div key={key} className="text-sm">
                        <span className="font-medium text-gray-700">{key}:</span>
                        <span className="text-gray-600 ml-2">{value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {backendConfig.visualization && (
                <div>
                  <h5 className="text-sm font-medium text-green-800 mb-3">📊 可视化优化</h5>
                  <div className="space-y-2">
                    {Object.entries(backendConfig.visualization).map(([key, value]: [string, any]) => (
                      <div key={key} className="text-sm">
                        <span className="font-medium text-gray-700">{key}:</span>
                        <span className="text-gray-600 ml-2">{value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

export default PhysicalPropertyVisualization