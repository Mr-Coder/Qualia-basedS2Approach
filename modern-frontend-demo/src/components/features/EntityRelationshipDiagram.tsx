import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Entity, Relationship, DeepRelation, ImplicitConstraint } from '@/stores/problemStore'
import { 
  getPhysicalPropertyDemo,
  getPhysicalPropertyAnalysis,
  PhysicalGraphAnalysis,
  PhysicalProperty,
  PhysicalConstraint,
  PhysicalRelation
} from '@/services/physicalPropertyAPI'

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
  // 新增：物性关系图专用属性
  problemText?: string  // 用于调用物性关系分析API
  enablePhysicalAnalysis?: boolean  // 是否启用物性关系分析
  diagramMode?: 'standard' | 'physical' | 'enhanced' | 'simple' | 'safe'  // 图表模式
  width?: number
  height?: number
  // 简化模式选项
  enableAnimations?: boolean  // 是否启用动画 (simple模式默认false)
  showConstraints?: boolean   // 是否显示约束 (simple模式默认false)
  interactive?: boolean       // 是否启用交互 (safe模式会限制某些交互)
  // 新增的参数
  reasoningPath?: any
  layoutConfig?: any
  animationConfig?: {
    enablePathAnimation: boolean
    animationSpeed: number
    stepDuration: number
    highlightIntensity: number
    autoPlay: boolean
  }
  onEntitySelect?: (entity: any) => void
  onStepChange?: (step: any) => void
}

// 新增：物性分析结果接口
interface PhysicalAnalysis {
  conservationViolations: string[]
  quantityConsistency: boolean
  spatialConstraints: string[]
  temporalOrder: string[]
  energyFlow: string[]
}

// 新增：实体集群接口
interface EntityCluster {
  id: string
  entities: EntityWithPosition[]
  center: Position
  type: 'conservation' | 'interaction' | 'transformation'
  strength: number
}

interface Position {
  x: number
  y: number
}

interface EntityWithPosition extends Entity {
  position: Position
}

// 物性关系情景图组件 - 增强版本支持推理路径动画和智能布局
const EntityRelationshipDiagram: React.FC<EntityRelationshipDiagramProps> = ({
  entities = [],
  relationships = [],
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
  problemText,
  enablePhysicalAnalysis = false,
  diagramMode = 'standard',
  width = 800,
  height = 600,
  
  // 新增参数
  reasoningPath,
  layoutConfig,
  animationConfig = {
    enablePathAnimation: true,
    animationSpeed: 1.0,
    stepDuration: 1000,
    highlightIntensity: 0.8,
    autoPlay: false
  },
  onEntitySelect,
  onStepChange
}) => {
  // 调试日志 - 打印所有接收到的 props
  console.log('🔍 EntityRelationshipDiagram Props:', {
    entities: entities,
    entitiesLength: entities?.length,
    entitiesIsArray: Array.isArray(entities),
    relationships: relationships,
    relationshipsLength: relationships?.length,
    relationshipsIsArray: Array.isArray(relationships),
    problemText: problemText,
    diagramMode: diagramMode
  })

  // 安全检查：确保必要的数组参数不为 undefined
  const safeEntities = Array.isArray(entities) ? entities : []
  const safeRelationships = Array.isArray(relationships) ? relationships : []
  const safeDeepRelations = Array.isArray(deepRelations) ? deepRelations : []
  const safeImplicitConstraints = Array.isArray(implicitConstraints) ? implicitConstraints : []
  const safePhysicalConstraints = Array.isArray(physicalConstraints) ? physicalConstraints : []
  
  const svgRef = useRef<SVGSVGElement>(null)
  const [entitiesWithPositions, setEntitiesWithPositions] = useState<EntityWithPosition[]>([])
  const [hoveredEntity, setHoveredEntity] = useState<string | null>(null)
  const [selectedDepthLayer, setSelectedDepthLayer] = useState<string>('all')
  const [showConstraintDetails, setShowConstraintDetails] = useState<boolean>(false)
  const [selectedRelationType, setSelectedRelationType] = useState<string>('all')
  const [physicalAnalysis, setPhysicalAnalysis] = useState<PhysicalAnalysis | null>(null)
  const [entityClusters, setEntityClusters] = useState<EntityCluster[]>([])
  const [animationStep, setAnimationStep] = useState<number>(0)
  const [showPhysicsSimulation, setShowPhysicsSimulation] = useState<boolean>(false)
  
  // 新增：物性关系分析状态
  const [physicalGraphAnalysis, setPhysicalGraphAnalysis] = useState<PhysicalGraphAnalysis | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false)
  const [physicalVizConfig, setPhysicalVizConfig] = useState<any>(null)

  // 物性关系分析主函数
  const performPhysicalAnalysis = useCallback(async () => {
    if (!problemText || !enablePhysicalAnalysis) return

    console.log('开始物性关系分析:', { problemText, enablePhysicalAnalysis })
    setIsAnalyzing(true)
    try {
      // 调用物性图谱分析API
      const analysisResult = await getPhysicalPropertyAnalysis(problemText)
      
      if (analysisResult) {
        setPhysicalGraphAnalysis(analysisResult)
        console.log('物性关系分析完成:', {
          properties: analysisResult.physical_properties?.length || 0,
          constraints: analysisResult.physical_constraints?.length || 0,
          relations: analysisResult.physical_relations?.length || 0,
          consistency: analysisResult.consistency_score || 0
        })
      } else {
        // 如果没有获取到特定分析，尝试获取演示数据
        const demoResult = await getPhysicalPropertyDemo()
        if (demoResult) {
          setPhysicalGraphAnalysis(demoResult)
        }
      }
    } catch (error) {
      console.error('物性关系分析失败:', error)
    } finally {
      setIsAnalyzing(false)
    }
  }, [problemText, enablePhysicalAnalysis])

  // 合并关系数据 - 需要在 analyzePhysicalProperties 之前定义
  const allRelationships = useMemo(() => {
    // 如果有物性分析结果，使用它
    if (physicalGraphAnalysis?.physical_relations) {
      return (physicalGraphAnalysis.physical_relations || []).map(pr => ({
        source: pr.source,
        target: pr.target,
        type: pr.type,
        weight: pr.strength
      }))
    }
    // 否则使用传入的关系数据
    if (relationships && relationships.length > 0) {
      return relationships
    }
    // 如果都没有，返回空数组
    return []
  }, [physicalGraphAnalysis, relationships])

  // 智能物性分析函数
  const analyzePhysicalProperties = useCallback((): PhysicalAnalysis => {
    const analysis: PhysicalAnalysis = {
      conservationViolations: [],
      quantityConsistency: true,
      spatialConstraints: [],
      temporalOrder: [],
      energyFlow: []
    }

    // 分析守恒定律违背
    const moneyEntities = safeEntities.filter(e => e.type === 'money')
    const objectEntities = safeEntities.filter(e => e.type === 'object')
    
    if (moneyEntities.length > 0) {
      analysis.energyFlow.push('货币价值流动检测')
      analysis.conservationViolations.push('检查货币总量守恒')
    }
    
    if (objectEntities.length > 0) {
      analysis.spatialConstraints.push('物体空间分布约束')
      analysis.temporalOrder.push('物理状态时序约束')
    }

    // 分析关系的数量一致性
    const quantityRelations = (allRelationships || []).filter(r => r.type.includes('总') || r.type.includes('和'))
    if (quantityRelations.length > 0) {
      analysis.conservationViolations.push('数量关系一致性验证')
    }

    return analysis
  }, [safeEntities, allRelationships])

  // 智能实体集群算法
  const calculateEntityClusters = useCallback((entities: EntityWithPosition[]): EntityCluster[] => {
    const clusters: EntityCluster[] = []
    
    // 基于关系强度的聚类
    const relationshipMap = new Map<string, string[]>()
    (allRelationships || []).forEach(rel => {
      if (!relationshipMap.has(rel.source)) relationshipMap.set(rel.source, [])
      if (!relationshipMap.has(rel.target)) relationshipMap.set(rel.target, [])
      relationshipMap.get(rel.source)?.push(rel.target)
      relationshipMap.get(rel.target)?.push(rel.source)
    })

    // 创建守恒集群
    const conservationEntities = safeEntities.filter(e => 
      (allRelationships || []).some(r => 
        (r.source === e.id || r.target === e.id) && 
        (r.type.includes('拥有') || r.type.includes('总'))
      )
    )
    
    if (conservationEntities.length > 1) {
      const centerX = conservationEntities.reduce((sum, e) => sum + e.position.x, 0) / conservationEntities.length
      const centerY = conservationEntities.reduce((sum, e) => sum + e.position.y, 0) / conservationEntities.length
      
      clusters.push({
        id: 'conservation-cluster',
        entities: conservationEntities,
        center: { x: centerX, y: centerY },
        type: 'conservation',
        strength: conservationEntities.length / Math.max(safeEntities.length, 1)
      })
    }

    return clusters
  }, [allRelationships])

  // 合并实体数据：优先使用传入的实体，其次使用物性分析结果
  const allEntities = useMemo(() => {
    // 优先使用传入的实体数据
    if (safeEntities && safeEntities.length > 0) {
      return safeEntities
    }
    
    // 如果没有传入的实体，尝试从物性分析结果提取
    if (physicalGraphAnalysis?.physical_properties) {
      const uniqueEntityIds = new Set<string>()
      const entities: Entity[] = []
      
      (physicalGraphAnalysis.physical_properties || []).forEach(prop => {
        if (!uniqueEntityIds.has(prop.entity)) {
          uniqueEntityIds.add(prop.entity)
          // 推断实体类型
          let type: Entity['type'] = 'concept'
          if (['小明', '小红', '小张'].some(name => prop.entity.includes(name))) {
            type = 'person'
          } else if (['苹果', '笔', '书'].some(obj => prop.entity.includes(obj))) {
            type = 'object'
          } else if (prop.entity.includes('元') || prop.type === 'conservation') {
            type = 'money'
          }
          
          entities.push({
            id: prop.entity,
            name: prop.value?.toString() || prop.entity,
            type
          })
        }
      })
      
      if (entities.length > 0) {
        return entities
      }
    }
    
    // 如果都没有，返回空数组
    return []
  }, [safeEntities, physicalGraphAnalysis])

  // 计算节点位置 - 使用物性关系优化的布局
  const calculatePositions = useCallback((entities: Entity[]): EntityWithPosition[] => {
    const centerX = width / 2
    const centerY = height / 2
    const radius = Math.min(width, height) / 3

    if (entities.length === 0) {
      return []
    }

    if (entities.length === 1) {
      return [{ ...entities[0], position: { x: centerX, y: centerY } }]
    }

    // 增强的物性关系导向布局：考虑关系密度
    const relationshipDensity = new Map<string, number>()
    safeEntities.forEach(entity => {
      const density = (allRelationships || []).filter(r => r.source === entity.id || r.target === entity.id).length
      relationshipDensity.set(entity.id, density)
    })

    // 按实体类型和关系密度分层
    const personEntities = safeEntities.filter(e => e.type === 'person')
    const objectEntities = safeEntities.filter(e => e.type === 'object')
    const conceptEntities = safeEntities.filter(e => e.type === 'concept')
    const moneyEntities = safeEntities.filter(e => e.type === 'money')

    const positioned: EntityWithPosition[] = []

    // 人物实体放在上方 - 根据关系密度调整位置
    personEntities.forEach((entity, index) => {
      const density = relationshipDensity.get(entity.id) || 0
      const x = centerX + (index - (personEntities.length - 1) / 2) * (120 + density * 10)
      const y = centerY - 150 - density * 15
      positioned.push({ ...entity, position: { x, y } })
    })

    // 物品实体放在中间 - 形成物理网络
    objectEntities.forEach((entity, index) => {
      const density = relationshipDensity.get(entity.id) || 0
      const angle = (index / objectEntities.length) * 2 * Math.PI
      const adjustedRadius = 80 + density * 20
      const x = centerX + Math.cos(angle) * adjustedRadius
      const y = centerY + Math.sin(angle) * adjustedRadius
      positioned.push({ ...entity, position: { x, y } })
    })

    // 概念实体放在下方 - 抽象层次布局
    conceptEntities.forEach((entity, index) => {
      const density = relationshipDensity.get(entity.id) || 0
      const x = centerX + (index - (conceptEntities.length - 1) / 2) * (140 + density * 5)
      const y = centerY + 150 + density * 10
      positioned.push({ ...entity, position: { x, y } })
    })

    // 货币实体放在右侧 - 价值流动区
    moneyEntities.forEach((entity, index) => {
      const density = relationshipDensity.get(entity.id) || 0
      const x = centerX + 200 + density * 15
      const y = centerY + (index - (moneyEntities.length - 1) / 2) * (80 + density * 10)
      positioned.push({ ...entity, position: { x, y } })
    })

    return positioned
  }, [width, height, allRelationships])

  useEffect(() => {
    // 确保 allEntities 是数组
    if (Array.isArray(allEntities)) {
      const positioned = calculatePositions(allEntities)
      setEntitiesWithPositions(positioned)
    } else {
      setEntitiesWithPositions([])
    }
  }, [allEntities, width, height, calculatePositions])

  // 从算法执行数据获取真实的物理关系
  const [algorithmData, setAlgorithmData] = useState<any>(null)
  
  // QS²算法专用状态
  const [qs2Data, setQS2Data] = useState<any>(null)
  const [qualiaStructures, setQualiaStructures] = useState<any[]>([])
  const [showQualiaDetails, setShowQualiaDetails] = useState<boolean>(false)
  const [isQS2Enhanced, setIsQS2Enhanced] = useState<boolean>(false)
  const [forceUpdate, setForceUpdate] = useState<number>(0)

  const extractMathematicalRelations = (problemText: string, algorithmStages: any[]) => {
    const extractedEntities: PhysicalEntity[] = []
    const extractedRelationships: PhysicalRelationship[] = []
    
    // 从问题文本智能提取数学实体
    const numbers = problemText.match(/\d+/g) || []
    const entities_keywords = ['小明', '小红', '苹果', '笔', '元', '米', '小时', '班级', '学生', '男生', '女生']
    
    // 创建数学实体
    let entityId = 0
    
    // 添加人物实体
    entities_keywords.forEach(keyword => {
      if (problemText.includes(keyword)) {
        const type = ['小明', '小红', '学生', '男生', '女生'].includes(keyword) ? 'person' : 
                    ['苹果', '笔'].includes(keyword) ? 'object' :
                    ['元'].includes(keyword) ? 'money' : 'concept'
        
        extractedEntities.push({
          id: `entity_${entityId++}`,
          name: keyword,
          type: type,
          properties: { 
            value: keyword === '元' ? '货币单位' : 
                   ['小明', '小红'].includes(keyword) ? '人物' : keyword
          },
          physicalAttributes: {
            mass: 1,
            position: { x: 0, y: 0, z: 0 },
            velocity: { x: 0, y: 0, z: 0 }
          }
        })
      }
    })
    
    // 添加数字实体
    numbers.forEach((num, index) => {
      extractedEntities.push({
        id: `num_${entityId++}`,
        name: num,
        type: 'concept',
        properties: { 
          value: parseInt(num),
          unit: problemText.includes('元') ? '元' : problemText.includes('个') ? '个' : '数量'
        },
        physicalAttributes: {
          mass: parseInt(num),
          position: { x: 0, y: 0, z: 0 },
          velocity: { x: 0, y: 0, z: 0 }
        }
      })
    })
    
    // 生成数学关系
    let relationId = 0
    
    // 如果是购买问题
    if (problemText.includes('买') && problemText.includes('元')) {
      const personEntity = extractedEntities.find(e => e.type === 'person')
      const objectEntity = extractedEntities.find(e => e.type === 'object')
      const moneyEntity = extractedEntities.find(e => e.name.includes('元') || e.type === 'money')
      const priceNum = extractedEntities.find(e => e.type === 'concept' && e.properties?.unit === '元')
      
      if (personEntity && objectEntity) {
        extractedRelationships.push({
          source: personEntity.id,
          target: objectEntity.id,
          type: '购买关系',
          strength: 0.9,
          physicalLaw: 'transaction',
          constraints: ['货币守恒', '物品转移']
        })
      }
      
      if (objectEntity && priceNum) {
        extractedRelationships.push({
          source: objectEntity.id,
          target: priceNum.id,
          type: '价格关系',
          strength: 0.8,
          physicalLaw: 'value_mapping',
          constraints: ['价值对应']
        })
      }
    }
    
    // 如果是计数问题
    if (problemText.includes('有') && problemText.includes('个')) {
      const personEntities = extractedEntities.filter(e => e.type === 'person')
      const objectEntity = extractedEntities.find(e => e.type === 'object')
      const numEntities = extractedEntities.filter(e => e.type === 'concept' && e.properties?.unit === '个')
      
      personEntities.forEach((person, index) => {
        if (objectEntity && numEntities[index]) {
          extractedRelationships.push({
            source: person.id,
            target: objectEntity.id,
            type: '拥有关系',
            strength: 0.9,
            physicalLaw: 'possession',
            constraints: ['数量守恒']
          })
          
          extractedRelationships.push({
            source: person.id,
            target: numEntities[index].id,
            type: '数量关系',
            strength: 0.8,
            physicalLaw: 'quantity',
            constraints: ['非负整数']
          })
        }
      })
      
      // 添加总和关系
      if (numEntities.length > 1) {
        extractedRelationships.push({
          source: numEntities[0].id,
          target: numEntities[1].id,
          type: '聚合关系',
          strength: 0.9,
          physicalLaw: 'summation',
          constraints: ['加法运算', '数量守恒']
        })
      }
    }
    
    // 如果是几何问题
    if (problemText.includes('长方形') && problemText.includes('面积')) {
      const shapeEntity = extractedEntities.find(e => e.name === '长方形')
      const lengthNum = extractedEntities.find(e => e.type === 'concept' && problemText.includes('长是' + e.name))
      const widthNum = extractedEntities.find(e => e.type === 'concept' && problemText.includes('宽是' + e.name))
      
      if (shapeEntity && lengthNum) {
        extractedRelationships.push({
          source: shapeEntity.id,
          target: lengthNum.id,
          type: '长度关系',
          strength: 0.9,
          physicalLaw: 'geometric',
          constraints: ['几何约束']
        })
      }
      
      if (shapeEntity && widthNum) {
        extractedRelationships.push({
          source: shapeEntity.id,
          target: widthNum.id,
          type: '宽度关系',
          strength: 0.9,
          physicalLaw: 'geometric',
          constraints: ['几何约束']
        })
      }
      
      if (lengthNum && widthNum) {
        extractedRelationships.push({
          source: lengthNum.id,
          target: widthNum.id,
          type: '面积计算',
          strength: 0.95,
          physicalLaw: 'multiplication',
          constraints: ['面积公式: 长×宽']
        })
      }
    }
    
    return { extractedEntities, extractedRelationships }
  }

  // QS²算法数据获取
  const fetchQS2Data = useCallback(async () => {
    try {
      // 动态导入QS²API
      const { qs2API } = await import('@/services/qs2API')
      
      // 获取QS²关系数据
      const qs2Response = await qs2API.getQS2Relations()
      
      if (qs2Response.success && qs2Response.data) {
        console.log('🧠 QS²数据获取成功:', qs2Response)
        
        setQS2Data(qs2Response.data)
        setIsQS2Enhanced(true)
        
        // 转换QS²数据为物性图谱格式
        const physicalAnalysis: PhysicalGraphAnalysis = {
          problem: '基于QS²算法的物性分析',
          physical_properties: (qs2Response.data.entities || []).map((entity: any, index: number) => ({
            id: `qs2_prop_${index}`,
            type: 'locality' as const,
            entity: entity.id,
            value: entity.name,
            unit: 'entity',
            certainty: entity.confidence || 0.8,
            constraints: entity.qualia_roles ? Object.keys(entity.qualia_roles) : []
          })),
          physical_constraints: [],
          physical_relations: (qs2Response.data.relationships || []).map((rel: any, index: number) => ({
            id: `qs2_rel_${index}`,
            source: rel.source,
            target: rel.target,
            type: rel.type,
            physical_basis: 'qs2_qualia_semantic',
            strength: rel.strength || 0.7,
            causal_direction: 'bidirectional'
          })),
          graph_metrics: {
            entity_count: (qs2Response.data.entities || []).length,
            relation_count: (qs2Response.data.relationships || []).length
          },
          consistency_score: 0.9
        }
        
        setPhysicalGraphAnalysis(physicalAnalysis)
        
        // 获取Qualia结构
        const qualiaResponse = await qs2API.getQualiaStructures()
        if (qualiaResponse.success) {
          setQualiaStructures(qualiaResponse.data)
        }
        
        console.log('🔬 QS²算法数据处理完成:', {
          properties: physicalAnalysis.physical_properties.length,
          relations: physicalAnalysis.physical_relations.length,
          qualiaStructures: qualiaResponse.data?.length || 0,
          isQS2Enhanced: true
        })
      }
    } catch (error) {
      console.error('❌ QS²数据获取失败:', error)
    }
  }, [])

  useEffect(() => {
    const fetchAlgorithmData = async () => {
      try {
        const response = await fetch('/api/algorithm/execution')
        const data = await response.json()
        if (data.success && data.data) {
          console.log('📥 获取到算法数据:', data.data)
          setAlgorithmData(data.data)
          
          // 🎯 强制启用QS²模式 - 检查是否是QS²增强算法
          const hasQS2Flags = data.data.is_qs2_enhanced || data.data.algorithm_type === 'QS2_Enhanced'
          const hasQS2Stages = data.data.stages?.some((stage: any) => 
            stage.stage_name?.includes('语义结构构建') ||
            stage.output_data?.qualia_structures ||
            stage.decisions?.some((d: any) => d.method === 'qualia_based')
          )
          const hasQS2Features = hasQS2Flags || hasQS2Stages
          
          console.log('🔍 QS²特征检测结果:', {
            hasQS2Flags,
            hasQS2Stages,
            hasQS2Features,
            stages: data.data.stages?.map((s: any) => s.stage_name)
          })
          
          // 🚀 强制启用QS²模式
          console.log('🧠 强制启用QS²模式...')
          setIsQS2Enhanced(true)
          await fetchQS2Data()
          setForceUpdate(prev => prev + 1) // 强制更新
          
          if (hasQS2Features) {
            console.log('✅ 检测到QS²增强算法特征，确认QS²模式...')
          } else {
            // 尝试获取QS²演示数据作为后备
            console.log('🔄 未检测到QS²特征，尝试获取QS²演示数据...')
            try {
              await fetchQS2Data()
              // 如果获取成功，仍然启用QS²模式
              if (qs2Data) {
                setIsQS2Enhanced(true)
                setForceUpdate(prev => prev + 1) // 强制更新
                console.log('✅ 使用QS²演示数据，启用QS²模式')
                return // 跳过原有逻辑
              }
            } catch (error) {
              console.log('⚠️ QS²演示数据获取失败，使用标准模式')
            }
            // 使用原有的数学关系提取逻辑
            const problemText = data.data.problem_text || '小明有5个苹果，小红有3个苹果，一共有多少个苹果？'
            
            console.log('生成数学关系，问题文本:', problemText)
            
            const { extractedEntities, extractedRelationships } = extractMathematicalRelations(
              problemText, 
              data.data.stages || []
            )
            
            if (extractedEntities.length > 0) {
              // 转换为物性图谱格式
              const fallbackAnalysis: PhysicalGraphAnalysis = {
                problem: problemText,
                physical_properties: (extractedEntities || []).map((entity, index) => ({
                  id: entity.id,
                  type: 'locality' as const,
                  entity: entity.id,
                  value: entity.name,
                  unit: entity.properties?.unit || 'entity',
                  certainty: 0.8,
                  constraints: []
                })),
                physical_constraints: [],
                physical_relations: (extractedRelationships || []).map((rel, index) => ({
                  id: `math_rel_${index}`,
                  source: rel.source,
                  target: rel.target,
                  type: rel.type,
                  physical_basis: rel.physicalLaw || 'mathematical',
                  strength: rel.strength,
                  causal_direction: 'unidirectional'
                })),
                graph_metrics: {
                  entity_count: extractedEntities.length,
                  relation_count: extractedRelationships.length
                },
                consistency_score: 0.75
              }
              
              setPhysicalGraphAnalysis(fallbackAnalysis)
              console.log('生成数学关系成功:', {
                properties: fallbackAnalysis.physical_properties.length,
                relations: fallbackAnalysis.physical_relations.length,
                entityNames: (extractedEntities || []).map(e => e.name),
                relationshipTypes: (extractedRelationships || []).map(r => r.type)
              })
            }
          }
        }
      } catch (error) {
        console.error('获取算法数据失败:', error)
      }
    }

    // 定时获取算法数据
    fetchAlgorithmData()
    const interval = setInterval(fetchAlgorithmData, 5000) // 增加间隔以减少请求频率
    return () => clearInterval(interval)
  }, [fetchQS2Data])

  // 触发物性关系分析
  useEffect(() => {
    if (enablePhysicalAnalysis && problemText && diagramMode === 'physical') {
      console.log('触发物性关系分析:', { problemText, diagramMode })
      // 延迟执行，确保组件稳定后再进行分析
      const timer = setTimeout(() => {
        performPhysicalAnalysis()
      }, 100)
      return () => clearTimeout(timer)
    }
  }, [performPhysicalAnalysis, enablePhysicalAnalysis, problemText, diagramMode])

  // 当传入新的实体数据时，如果没有物性分析结果且启用了物性分析，触发分析
  useEffect(() => {
    if (enablePhysicalAnalysis && 
        problemText && 
        diagramMode === 'physical' && 
        entities && entities.length > 0 && 
        !physicalGraphAnalysis &&
        !isAnalyzing) {
      console.log('接收到新实体数据，触发物性关系分析')
      performPhysicalAnalysis()
    }
  }, [entities, enablePhysicalAnalysis, problemText, diagramMode, physicalGraphAnalysis, isAnalyzing, performPhysicalAnalysis])

  // 数据持久化：当接收到新数据时，保持物性分析结果
  useEffect(() => {
    console.log('组件数据更新:', {
      entitiesCount: entities?.length || 0,
      relationshipsCount: relationships?.length || 0,
      physicalAnalysisAvailable: !!physicalGraphAnalysis,
      physicalPropertiesCount: physicalGraphAnalysis?.physical_properties.length || 0,
      physicalRelationsCount: physicalGraphAnalysis?.physical_relations.length || 0,
      problemText: problemText?.substring(0, 20) + '...',
      componentKey: 'physical-relations-diagram'
    })
  }, [entities, relationships, physicalGraphAnalysis, problemText])

  // 调试数据状态
  useEffect(() => {
    console.log('EntityRelationshipDiagram 数据状态:', {
      entities: entities?.length || 0,
      relationships: relationships?.length || 0,
      physicalAnalysis: !!physicalGraphAnalysis,
      allEntities: allEntities?.length || 0,
      allRelationships: allRelationships?.length || 0,
      problemText,
      enablePhysicalAnalysis,
      diagramMode,
      isAnalyzing
    })
  }, [entities, relationships, physicalGraphAnalysis, allEntities, allRelationships, problemText, enablePhysicalAnalysis, diagramMode, isAnalyzing])

  // 显示分析状态
  if (isAnalyzing) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>🔬 数学关系情景图</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12">
            <div className="text-6xl mb-4 animate-pulse">🧮</div>
            <div className="text-lg font-medium mb-2">正在解析数学关系...</div>
            <div className="text-sm text-blue-600">数学关系分析引擎正在处理算法数据</div>
            {algorithmData && (
              <div className="mt-4 text-xs text-gray-500">
                检测到算法执行数据: {algorithmData.stages?.length || 0}个阶段
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    )
  }

  // 如果没有数据，显示示例或提示
  if (!allEntities || allEntities.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>🔬 数学关系情景图</CardTitle>
        </CardHeader>
        <CardContent className="min-h-[400px]">
          <div className="text-center py-12 text-gray-500">
            <div className="text-6xl mb-4">🧮</div>
            <div className="text-lg font-medium mb-2">暂无数学关系数据</div>
            <div className="text-sm mb-4">请先在智能求解模块解决一个数学问题</div>
            
            {/* 添加示例按钮 */}
            <button
              onClick={() => {
                // 使用示例数据
                const exampleEntities = [
                  { id: 'xiaoming', name: '小明', type: 'person' as const },
                  { id: 'apples', name: '苹果', type: 'object' as const },
                  { id: 'num5', name: '5个', type: 'concept' as const }
                ];
                const exampleRelationships = [
                  { source: 'xiaoming', target: 'apples', type: '拥有关系', weight: 0.9 },
                  { source: 'apples', target: 'num5', type: '数量关系', weight: 0.8 }
                ];
                // 这里需要通过父组件传递数据，暂时只能显示调试信息
                console.log('示例数据:', { exampleEntities, exampleRelationships });
              }}
              className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
            >
              查看示例图表
            </button>
            {algorithmData && (
              <div className="text-xs text-green-600 mb-2">
                检测到算法数据 - 正在解析数学关系...
              </div>
            )}
            {/* 调试信息 */}
            <div className="text-xs bg-gray-100 p-2 rounded mt-4">
              <div>实体: {entities?.length || 0} | 物性分析: {physicalGraphAnalysis ? '可用' : '无'}</div>
              <div>关系: {relationships?.length || 0} | 物性关系: {physicalGraphAnalysis?.physical_relations.length || 0}</div>
              <div>合并实体: {allEntities.length} | 合并关系: {allRelationships.length}</div>
              <div>算法数据: {algorithmData ? '已获取' : '获取中'}</div>
              {algorithmData && (
                <div className="mt-2 text-blue-600">
                  问题: {algorithmData.problem_text?.substring(0, 30) || '未知'}...
                </div>
              )}
              {physicalGraphAnalysis && (
                <div className="mt-2 text-green-600">
                  物性属性: {physicalGraphAnalysis.physical_properties.length}个，
                  约束: {physicalGraphAnalysis.physical_constraints.length}个，
                  一致性: {(physicalGraphAnalysis.consistency_score * 100).toFixed(1)}%
                </div>
              )}
            </div>
            <div className="bg-blue-50 p-4 rounded-lg text-left max-w-md mx-auto">
              <div className="text-sm text-blue-800">
                <strong>数学关系包括：</strong>
                <ul className="mt-2 space-y-1">
                  <li>• 拥有关系 - 数量归属关系</li>
                  <li>• 聚合关系 - 数学运算关系</li>
                  <li>• 购买关系 - 价值交换关系</li>
                  <li>• 几何关系 - 空间度量关系</li>
                  <li>• 约束关系 - 数学约束条件</li>
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
    if (type.includes('拥有关系')) return `拥有关系 (守恒)`
    if (type.includes('数量关系')) return `数量关系 (${weight || 0.8})`
    if (type.includes('聚合关系')) return `聚合关系 (求和)`
    if (type.includes('购买关系')) return `购买关系 (交易)`
    if (type.includes('价格关系')) return `价格关系 (对应)`
    if (type.includes('长度关系')) return `长度关系 (尺寸)`
    if (type.includes('宽度关系')) return `宽度关系 (尺寸)`
    if (type.includes('面积计算')) return `面积计算 (长×宽)`
    if (type.includes('几何')) return `几何关系 (公式)`
    return `数学关系 (${((weight || 0.5) * 100).toFixed(0)}%)`
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
        <CardTitle className="flex items-center justify-between">
          <span>
            {isQS2Enhanced ? '🧠 QS²语义关系图' : '🔬 数学关系情景图'}
          </span>
          {isQS2Enhanced && (
            <div className="flex items-center space-x-2">
              <span className="text-xs bg-purple-100 text-purple-800 px-2 py-1 rounded-full">
                QS²增强算法
              </span>
              {qualiaStructures.length > 0 && (
                <button
                  onClick={() => setShowQualiaDetails(!showQualiaDetails)}
                  className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full hover:bg-blue-200 transition-colors"
                >
                  {showQualiaDetails ? '隐藏' : '显示'}Qualia结构
                </button>
              )}
            </div>
          )}
        </CardTitle>
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
          {visualizationConfig.show_depth_indicators && (safeDeepRelations || []).map((deepRel, index) => {
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
          {(allRelationships || []).map((rel, index) => {
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
          {(entitiesWithPositions || []).map((entity, index) => (
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
        {visualizationConfig.show_depth_indicators && safeDeepRelations.length > 0 && (
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
                const depthCount = safeDeepRelations.filter(r => r.depth === depth).length
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

        {/* QS²算法执行阶段显示 */}
        {isQS2Enhanced && qs2Data && qs2Data.algorithm_stages && (
          <div className="mt-6 bg-gradient-to-r from-purple-50 to-indigo-50 p-4 rounded-lg border border-purple-200">
            <h4 className="text-sm font-medium text-purple-800 mb-3 flex items-center">
              🧠 QS²算法执行阶段
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {(qs2Data.algorithm_stages || []).map((stage: any, index: number) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-white p-3 rounded border-l-4 border-purple-500"
                >
                  <div className="flex justify-between items-start mb-2">
                    <span className="text-sm font-medium text-gray-800">{stage.name}</span>
                    <span className="text-xs text-purple-600">{stage.duration_ms.toFixed(1)}ms</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-purple-600 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${stage.confidence * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-xs text-gray-600 min-w-fit">{(stage.confidence * 100).toFixed(0)}%</span>
                  </div>
                  {stage.visual_elements && stage.visual_elements.length > 0 && (
                    <div className="mt-2 text-xs text-gray-500">
                      发现元素: {stage.visual_elements.length}个
                    </div>
                  )}
                </motion.div>
              ))}
            </div>
          </div>
        )}

        {/* Qualia语义结构详情 */}
        {isQS2Enhanced && showQualiaDetails && qualiaStructures.length > 0 && (
          <div className="mt-6 bg-gradient-to-r from-blue-50 to-cyan-50 p-4 rounded-lg border border-blue-200">
            <h4 className="text-sm font-medium text-blue-800 mb-3 flex items-center">
              🔬 Qualia四维语义结构
            </h4>
            <div className="space-y-4">
              {(qualiaStructures || []).map((structure: any, index: number) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.2 }}
                  className="bg-white p-4 rounded-lg border border-blue-200"
                >
                  <div className="flex justify-between items-start mb-3">
                    <h5 className="text-sm font-semibold text-gray-800 flex items-center">
                      {getPhysicalEntityIcon(structure.entity_type as any)} {structure.entity}
                    </h5>
                    <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full">
                      置信度: {(structure.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div className="space-y-2">
                      <div className="text-xs">
                        <span className="font-medium text-red-600">📋 Formal (形式):</span>
                        <div className="mt-1 flex flex-wrap gap-1">
                          {(structure.qualia_roles?.formal || []).map((role: string, i: number) => (
                            <span key={i} className="text-xs bg-red-100 text-red-700 px-2 py-1 rounded">
                              {role}
                            </span>
                          ))}
                        </div>
                      </div>
                      
                      <div className="text-xs">
                        <span className="font-medium text-green-600">🎯 Telic (目的):</span>
                        <div className="mt-1 flex flex-wrap gap-1">
                          {(structure.qualia_roles?.telic || []).map((role: string, i: number) => (
                            <span key={i} className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded">
                              {role}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                    
                    <div className="space-y-2">
                      <div className="text-xs">
                        <span className="font-medium text-blue-600">🔨 Agentive (起源):</span>
                        <div className="mt-1 flex flex-wrap gap-1">
                          {(structure.qualia_roles?.agentive || []).map((role: string, i: number) => (
                            <span key={i} className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded">
                              {role}
                            </span>
                          ))}
                        </div>
                      </div>
                      
                      <div className="text-xs">
                        <span className="font-medium text-purple-600">🧩 Constitutive (构成):</span>
                        <div className="mt-1 flex flex-wrap gap-1">
                          {(structure.qualia_roles?.constitutive || []).map((role: string, i: number) => (
                            <span key={i} className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded">
                              {role}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}

        {/* 数学关系图例 */}
        <div className="mt-6 bg-white p-4 rounded-lg border border-gray-200">
          <h4 className="text-sm font-medium text-gray-800 mb-3">
            {isQS2Enhanced ? '🧠 QS²语义关系图例' : '🔬 数学关系图例'}
          </h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="space-y-2">
              <div className="flex items-center space-x-3">
                <div className="w-4 h-4 rounded-full bg-red-500"></div>
                <span>👤 人物实体 - 问题主体</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-4 h-4 rounded-full bg-green-500"></div>
                <span>🧮 物体实体 - 可计算对象</span>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex items-center space-x-3">
                <div className="w-4 h-4 rounded-full bg-orange-500"></div>
                <span>💰 价值实体 - 货币单位</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-4 h-4 rounded-full bg-purple-500"></div>
                <span>⚛️ 数值实体 - 数学概念</span>
              </div>
            </div>
          </div>
          
          <div className="mt-4 pt-3 border-t border-gray-200">
            <div className="text-xs text-gray-600">
              <strong>数学关系类型：</strong>
              <span className="ml-2">拥有关系 • 数量关系 • 聚合关系 • 购买关系 • 几何关系</span>
            </div>
            {safeDeepRelations.length > 0 && (
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
        {visualizationConfig.show_constraint_panels && safeImplicitConstraints.length > 0 && (
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
              {(safeImplicitConstraints || []).map((constraint, index) => (
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
        {safePhysicalConstraints.length > 0 && (
          <div className="mt-6 bg-gradient-to-r from-blue-50 to-purple-50 p-4 rounded-lg border border-blue-200">
            <h4 className="text-sm font-medium text-blue-800 mb-3 flex items-center">
              ⚛️ 物性约束与守恒定律
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {(safePhysicalConstraints || []).map((constraint, index) => (
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
            {physicalProperties?.conservationLaws && physicalProperties.conservationLaws.length > 0 && (
              <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                <h5 className="text-sm font-medium text-green-800 mb-2 flex items-center">
                  ⚖️ 守恒定律
                </h5>
                <ul className="space-y-1">
                  {(physicalProperties?.conservationLaws || []).map((law, index) => (
                    <li key={index} className="text-xs text-green-700">• {law}</li>
                  ))}
                </ul>
              </div>
            )}
            
            {physicalProperties?.materialProperties && physicalProperties.materialProperties.length > 0 && (
              <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                <h5 className="text-sm font-medium text-orange-800 mb-2 flex items-center">
                  🧱 物质属性
                </h5>
                <ul className="space-y-1">
                  {(physicalProperties?.materialProperties || []).map((prop, index) => (
                    <li key={index} className="text-xs text-orange-700">• {prop}</li>
                  ))}
                </ul>
              </div>
            )}
            
            {physicalProperties?.spatialRelations && physicalProperties.spatialRelations.length > 0 && (
              <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                <h5 className="text-sm font-medium text-purple-800 mb-2 flex items-center">
                  📍 空间关系
                </h5>
                <ul className="space-y-1">
                  {(physicalProperties?.spatialRelations || []).map((relation, index) => (
                    <li key={index} className="text-xs text-purple-700">• {relation}</li>
                  ))}
                </ul>
              </div>
            )}
            
            {physicalProperties?.temporalConstraints && physicalProperties.temporalConstraints.length > 0 && (
              <div className="bg-pink-50 p-4 rounded-lg border border-pink-200">
                <h5 className="text-sm font-medium text-pink-800 mb-2 flex items-center">
                  ⏰ 时间约束
                </h5>
                <ul className="space-y-1">
                  {(physicalProperties?.temporalConstraints || []).map((constraint, index) => (
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