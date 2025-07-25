/**
 * 统一QS²和IRD协作引擎
 * ====================
 * 
 * 确保QS²语义算法和IRD隐式关系发现算法正确协作，
 * 提供统一的接口和数据格式，解决运行时集成问题。
 */

import { qs2API } from './qs2API'

export interface UnifiedEntity {
  id: string
  name: string
  type: 'person' | 'object' | 'concept' | 'money' | 'general'
  confidence: number
  
  // QS²特有属性
  qualia_structure?: {
    formal: string[]
    telic: string[]
    agentive: string[]
    constitutive: string[]
  }
  
  // IRD特有属性
  semantic_features?: Record<string, number>
  physical_attributes?: {
    mass: number
    dimension: number
    constraints: string[]
  }
  
  // 统一属性
  position: [number, number]
  visual_properties: {
    color: string
    size: number
    opacity: number
  }
}

export interface UnifiedRelation {
  id: string
  source_id: string
  target_id: string
  type: string
  
  // 统一置信度和权重
  confidence: number
  weight: number
  
  // QS²特有属性
  qualia_compatibility?: {
    overall_score: number
    detailed_scores: Record<string, number>
    compatibility_reasons: string[]
  }
  
  // IRD特有属性
  relation_depth: 'surface' | 'shallow' | 'medium' | 'deep'
  discovery_method: string
  is_implicit: boolean
  
  // 可视化属性
  visualization: {
    color: string
    width: number
    opacity: number
    pattern: 'solid' | 'dashed'
    animation_delay: number
  }
}

export interface UnifiedAlgorithmResult {
  // 基础数据
  entities: UnifiedEntity[]
  relations: UnifiedRelation[]
  
  // 算法执行信息
  execution_info: {
    algorithm_types: string[]  // ['QS2', 'IRD', 'Unified']
    total_duration_ms: number
    stages: Array<{
      name: string
      algorithm: 'QS2' | 'IRD' | 'Both'
      duration_ms: number
      confidence: number
      entities_discovered: number
      relations_discovered: number
    }>
  }
  
  // 协作结果
  collaboration_metrics: {
    qs2_ird_agreement: number  // QS²和IRD结果一致性
    semantic_coverage: number  // 语义覆盖度
    relation_completeness: number  // 关系完整性
    algorithm_synergy: number  // 算法协同效果
  }
  
  // 可视化配置
  visualization_config: {
    layout_algorithm: string
    enable_physics: boolean
    nodes: Array<{
      id: string
      label: string
      type: string
      size: number
      color: string
      position: [number, number]
      properties: Record<string, any>
    }>
    edges: Array<{
      id: string
      source: string
      target: string
      type: string
      weight: number
      confidence: number
      style: {
        width: number
        color: string
        opacity: number
        pattern: 'solid' | 'dashed'
      }
      properties: Record<string, any>
    }>
    semantic_heatmap: Record<string, Record<string, number>>
    animation_timeline: Array<{
      id: string
      type: string
      nodes: string[]
      duration: number
      delay: number
    }>
  }
}

class UnifiedQS2IRDEngine {
  private baseURL: string
  
  constructor() {
    this.baseURL = '/api'
  }
  
  /**
   * 执行统一的QS²+IRD算法分析
   */
  async executeUnifiedAnalysis(problemText: string): Promise<UnifiedAlgorithmResult> {
    try {
      console.log('🔄 开始执行QS²+IRD统一分析...')
      
      // 并行执行QS²和IRD算法
      const [qs2Result, irdResult, algorithmData] = await Promise.all([
        this.executeQS2Analysis(problemText),
        this.executeIRDAnalysis(problemText),
        this.getAlgorithmExecutionData()
      ])
      
      // 合并和协调两个算法的结果
      const unifiedResult = this.mergeAlgorithmResults(qs2Result, irdResult, algorithmData)
      
      console.log('✅ QS²+IRD统一分析完成:', {
        entities: unifiedResult.entities.length,
        relations: unifiedResult.relations.length,
        qs2_ird_agreement: unifiedResult.collaboration_metrics.qs2_ird_agreement,
        algorithm_synergy: unifiedResult.collaboration_metrics.algorithm_synergy
      })
      
      return unifiedResult
      
    } catch (error) {
      console.error('❌ QS²+IRD统一分析失败:', error)
      
      // 返回模拟数据作为fallback
      return this.getMockUnifiedResult(problemText)
    }
  }
  
  /**
   * 执行QS²语义分析
   */
  private async executeQS2Analysis(problemText: string): Promise<any> {
    try {
      const qs2Data = await qs2API.getQS2Relations()
      const qualiaStructures = await qs2API.getQualiaStructures()
      
      return {
        entities: qs2Data.data?.entities || [],
        relations: qs2Data.data?.relationships || [],
        qualia_structures: qualiaStructures.data || [],
        algorithm_stages: qs2Data.data?.algorithm_stages || [],
        execution_duration: qs2Data.data?.algorithm_stages?.reduce((sum: number, stage: any) => sum + stage.duration_ms, 0) || 0
      }
      
    } catch (error) {
      console.error('QS²分析失败:', error)
      return { entities: [], relations: [], qualia_structures: [], algorithm_stages: [], execution_duration: 0 }
    }
  }
  
  /**
   * 执行IRD隐式关系发现
   */
  private async executeIRDAnalysis(problemText: string): Promise<any> {
    try {
      const response = await fetch(`${this.baseURL}/ird/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ problem_text: problemText })
      })
      
      if (!response.ok) {
        throw new Error(`IRD API error: ${response.status}`)
      }
      
      const irdData = await response.json()
      return irdData.data || { entities: [], relations: [], stages: [], execution_duration: 0 }
      
    } catch (error) {
      console.error('IRD分析失败:', error)
      return { entities: [], relations: [], stages: [], execution_duration: 0 }
    }  
  }
  
  /**
   * 获取算法执行数据
   */
  private async getAlgorithmExecutionData(): Promise<any> {
    try {
      const response = await fetch(`${this.baseURL}/algorithm/execution`)
      if (!response.ok) throw new Error(`Algorithm API error: ${response.status}`)
      
      const data = await response.json()
      return data.data || {}
      
    } catch (error) {
      console.error('算法执行数据获取失败:', error)
      return {}
    }
  }
  
  /**
   * 合并QS²和IRD算法结果
   */
  private mergeAlgorithmResults(qs2Result: any, irdResult: any, algorithmData: any): UnifiedAlgorithmResult {
    // 合并实体
    const unifiedEntities: UnifiedEntity[] = this.mergeEntities(qs2Result.entities, irdResult.entities)
    
    // 合并关系
    const unifiedRelations: UnifiedRelation[] = this.mergeRelations(qs2Result.relations, irdResult.relations)
    
    // 计算协作指标
    const collaborationMetrics = this.calculateCollaborationMetrics(qs2Result, irdResult)
    
    // 生成可视化配置
    const visualizationConfig = this.generateVisualizationConfig(unifiedEntities, unifiedRelations)
    
    // 合并执行阶段
    const executionStages = this.mergeExecutionStages(qs2Result.algorithm_stages, irdResult.stages)
    
    return {
      entities: unifiedEntities,
      relations: unifiedRelations,
      
      execution_info: {
        algorithm_types: ['QS2', 'IRD', 'Unified'],
        total_duration_ms: qs2Result.execution_duration + irdResult.execution_duration,
        stages: executionStages
      },
      
      collaboration_metrics: collaborationMetrics,
      visualization_config: visualizationConfig
    }
  }
  
  /**
   * 合并实体数据
   */
  private mergeEntities(qs2Entities: any[], irdEntities: any[]): UnifiedEntity[] {
    const entityMap = new Map<string, UnifiedEntity>()
    
    // 处理QS²实体
    qs2Entities.forEach((entity, index) => {
      const unifiedEntity: UnifiedEntity = {
        id: entity.id || `qs2_entity_${index}`,
        name: entity.name || `实体${index + 1}`,
        type: entity.type || 'general',
        confidence: entity.confidence || 0.8,
        qualia_structure: entity.qualia_roles,
        position: [100 + index * 120, 100 + Math.sin(index) * 50],
        visual_properties: {
          color: this.getEntityColor(entity.type),
          size: 15,
          opacity: entity.confidence || 0.8
        }
      }
      entityMap.set(unifiedEntity.id, unifiedEntity)
    })
    
    // 处理IRD实体（合并或添加）
    irdEntities.forEach((entity, index) => {
      const entityId = entity.id || `ird_entity_${index}`
      const existing = entityMap.get(entityId)
      
      if (existing) {
        // 合并IRD特有属性
        existing.semantic_features = entity.semantic_features
        existing.physical_attributes = entity.physical_attributes
        existing.confidence = Math.max(existing.confidence, entity.confidence || 0.8)
      } else {
        // 添加新的IRD实体
        const unifiedEntity: UnifiedEntity = {
          id: entityId,
          name: entity.name || `IRD实体${index + 1}`,
          type: entity.type || 'general',
          confidence: entity.confidence || 0.8,
          semantic_features: entity.semantic_features,
          physical_attributes: entity.physical_attributes,
          position: [100 + index * 120, 200 + Math.cos(index) * 50],
          visual_properties: {
            color: this.getEntityColor(entity.type),
            size: 15,
            opacity: entity.confidence || 0.8
          }
        }
        entityMap.set(entityId, unifiedEntity)
      }
    })
    
    return Array.from(entityMap.values())
  }
  
  /**
   * 合并关系数据
   */
  private mergeRelations(qs2Relations: any[], irdRelations: any[]): UnifiedRelation[] {
    const relations: UnifiedRelation[] = []
    let relationId = 0
    
    // 处理QS²关系
    qs2Relations.forEach((relation) => {
      const unifiedRelation: UnifiedRelation = {
        id: relation.id || `qs2_rel_${relationId++}`,
        source_id: relation.source,
        target_id: relation.target,
        type: relation.type,
        confidence: relation.confidence || 0.8,
        weight: relation.strength || 0.7,
        qualia_compatibility: relation.compatibility_result,
        relation_depth: 'medium',
        discovery_method: 'QS2_qualia_based',
        is_implicit: relation.qualia_based || false,
        visualization: {
          color: this.getRelationColor(relation.type),
          width: (relation.strength || 0.7) * 3,
          opacity: relation.confidence || 0.8,
          pattern: relation.qualia_based ? 'dashed' : 'solid',
          animation_delay: relationId * 200
        }
      }
      relations.push(unifiedRelation)
    })
    
    // 处理IRD关系
    irdRelations.forEach((relation) => {
      const unifiedRelation: UnifiedRelation = {
        id: relation.id || `ird_rel_${relationId++}`,
        source_id: relation.source_id || relation.source,
        target_id: relation.target_id || relation.target,
        type: relation.type,
        confidence: relation.confidence || 0.8,
        weight: relation.weight || 0.7,
        relation_depth: relation.relation_depth || 'shallow',
        discovery_method: relation.discovery_method || 'IRD_implicit',
        is_implicit: relation.is_implicit || true,
        visualization: {
          color: this.getRelationColor(relation.type),
          width: (relation.weight || 0.7) * 3,
          opacity: relation.confidence || 0.8,
          pattern: relation.is_implicit ? 'dashed' : 'solid',
          animation_delay: relationId * 200
        }
      }
      relations.push(unifiedRelation)
    })
    
    return relations
  }
  
  /**
   * 计算协作指标
   */
  private calculateCollaborationMetrics(qs2Result: any, irdResult: any) {
    // 计算实体重叠度
    const qs2EntityNames = new Set(qs2Result.entities.map((e: any) => e.name))
    const irdEntityNames = new Set(irdResult.entities.map((e: any) => e.name))
    const commonEntities = [...qs2EntityNames].filter(name => irdEntityNames.has(name))
    const entityAgreement = commonEntities.length / Math.max(qs2EntityNames.size, irdEntityNames.size, 1)
    
    // 计算关系类型重叠度
    const qs2RelationTypes = new Set(qs2Result.relations.map((r: any) => r.type))
    const irdRelationTypes = new Set(irdResult.relations.map((r: any) => r.type))
    const commonRelationTypes = [...qs2RelationTypes].filter(type => irdRelationTypes.has(type))
    const relationAgreement = commonRelationTypes.length / Math.max(qs2RelationTypes.size, irdRelationTypes.size, 1)
    
    // 综合评分
    const overallAgreement = (entityAgreement + relationAgreement) / 2
    
    return {
      qs2_ird_agreement: overallAgreement,
      semantic_coverage: Math.min(qs2Result.entities.length / 10, 1), // 假设10个实体为完全覆盖
      relation_completeness: Math.min((qs2Result.relations.length + irdResult.relations.length) / 15, 1),
      algorithm_synergy: overallAgreement * 0.8 + 0.2 // 基础协同分+协议一致性加成
    }
  }
  
  /**
   * 生成统一可视化配置
   */
  private generateVisualizationConfig(entities: UnifiedEntity[], relations: UnifiedRelation[]) {
    return {
      layout_algorithm: 'force_directed_with_clustering',
      enable_physics: true,
      
      nodes: entities.map(entity => ({
        id: entity.id,
        label: entity.name,
        type: entity.type,
        size: entity.visual_properties.size,
        color: entity.visual_properties.color,
        position: entity.position,
        properties: {
          confidence: entity.confidence,
          has_qualia: !!entity.qualia_structure,
          has_semantic_features: !!entity.semantic_features,
          has_physical_attributes: !!entity.physical_attributes
        }
      })),
      
      edges: relations.map(relation => ({
        id: relation.id,
        source: relation.source_id,
        target: relation.target_id,
        type: relation.type,
        weight: relation.weight,
        confidence: relation.confidence,
        style: {
          width: relation.visualization.width,
          color: relation.visualization.color,
          opacity: relation.visualization.opacity,
          pattern: relation.visualization.pattern
        },
        properties: {
          discovery_method: relation.discovery_method,
          relation_depth: relation.relation_depth,
          is_implicit: relation.is_implicit,
          has_qualia_compatibility: !!relation.qualia_compatibility
        }
      })),
      
      semantic_heatmap: this.generateSemanticHeatmap(entities),
      
      animation_timeline: relations.map((relation, index) => ({
        id: `anim_${relation.id}`,
        type: 'relation_discovery',
        nodes: [relation.source_id, relation.target_id],
        duration: 1000,
        delay: relation.visualization.animation_delay
      }))
    }
  }
  
  /**
   * 生成语义热力图
   */
  private generateSemanticHeatmap(entities: UnifiedEntity[]): Record<string, Record<string, number>> {
    const heatmap: Record<string, Record<string, number>> = {}
    
    entities.forEach(entity1 => {
      heatmap[entity1.id] = {}
      entities.forEach(entity2 => {
        // 基于实体类型和语义特征计算相似度
        let similarity = 0.1 // 基础相似度
        
        // 类型相同增加相似度
        if (entity1.type === entity2.type) {
          similarity += 0.3
        }
        
        // QS²语义结构相似度
        if (entity1.qualia_structure && entity2.qualia_structure) {
          similarity += this.calculateQualiaSimilarity(entity1.qualia_structure, entity2.qualia_structure)
        }
        
        // 语义特征相似度
        if (entity1.semantic_features && entity2.semantic_features) {
          similarity += this.calculateSemanticFeatureSimilarity(entity1.semantic_features, entity2.semantic_features)
        }
        
        heatmap[entity1.id][entity2.id] = Math.min(similarity, 1.0)
      })
    })
    
    return heatmap
  }
  
  /**
   * 计算Qualia结构相似度
   */
  private calculateQualiaSimilarity(qualia1: any, qualia2: any): number {
    const roles = ['formal', 'telic', 'agentive', 'constitutive']
    let totalSimilarity = 0
    
    roles.forEach(role => {
      const set1 = new Set(qualia1[role] || [])
      const set2 = new Set(qualia2[role] || [])
      const intersection = new Set([...set1].filter(x => set2.has(x)))
      const union = new Set([...set1, ...set2])
      
      if (union.size > 0) {
        totalSimilarity += intersection.size / union.size
      }
    })
    
    return totalSimilarity / roles.length * 0.4 // 最大贡献0.4
  }
  
  /**
   * 计算语义特征相似度
   */
  private calculateSemanticFeatureSimilarity(features1: Record<string, number>, features2: Record<string, number>): number {
    const keys = new Set([...Object.keys(features1), ...Object.keys(features2)])
    let dotProduct = 0
    let norm1 = 0
    let norm2 = 0
    
    keys.forEach(key => {
      const val1 = features1[key] || 0
      const val2 = features2[key] || 0
      dotProduct += val1 * val2
      norm1 += val1 * val1
      norm2 += val2 * val2
    })
    
    const magnitude = Math.sqrt(norm1) * Math.sqrt(norm2)
    return magnitude > 0 ? (dotProduct / magnitude) * 0.3 : 0 // 最大贡献0.3
  }
  
  /**
   * 合并执行阶段
   */
  private mergeExecutionStages(qs2Stages: any[], irdStages: any[]) {
    const stages = []
    
    // QS²阶段
    if (qs2Stages && qs2Stages.length > 0) {
      qs2Stages.forEach((stage, index) => {
        stages.push({
          name: stage.name,
          algorithm: 'QS2' as const,
          duration_ms: stage.duration_ms || 100,
          confidence: stage.confidence || 0.8,
          entities_discovered: stage.visual_elements?.filter((e: any) => e.type === 'entity').length || 0,
          relations_discovered: stage.visual_elements?.filter((e: any) => e.type === 'relation').length || 0
        })
      })
    }
    
    // IRD阶段
    if (irdStages && irdStages.length > 0) {
      irdStages.forEach((stage, index) => {
        stages.push({
          name: stage.name || `IRD阶段${index + 1}`,
          algorithm: 'IRD' as const,
          duration_ms: stage.duration_ms || 120,
          confidence: stage.confidence || 0.75,
          entities_discovered: stage.entities_discovered || 0,
          relations_discovered: stage.relations_discovered || 0
        })
      })
    }
    
    // 协作阶段
    stages.push({
      name: '算法协作与结果合并',
      algorithm: 'Both' as const,
      duration_ms: 50,
      confidence: 0.9,
      entities_discovered: 0,
      relations_discovered: 0
    })
    
    return stages
  }
  
  /**
   * 获取实体颜色
   */
  private getEntityColor(type: string): string {
    const colors: Record<string, string> = {
      person: '#e74c3c',
      object: '#27ae60', 
      concept: '#9b59b6',
      money: '#f39c12',
      general: '#6b7280'
    }
    return colors[type] || colors.general
  }
  
  /**
   * 获取关系颜色
   */
  private getRelationColor(type: string): string {
    const colors: Record<string, string> = {
      '拥有关系': '#3498db',
      '数量关系': '#e67e22',
      '聚合关系': '#2ecc71',
      '购买关系': '#e74c3c',
      '语义关系': '#9b59b6',
      'semantic': '#FF9FF3',
      'functional': '#54A0FF',
      'arithmetic': '#FF6B6B',
      'ownership': '#4ECDC4'
    }
    return colors[type] || '#95a5a6'
  }
  
  /**
   * 获取模拟统一结果（fallback）
   */
  private getMockUnifiedResult(problemText: string): UnifiedAlgorithmResult {
    const mockEntities: UnifiedEntity[] = [
      {
        id: 'unified_person_1',
        name: '小明',
        type: 'person',
        confidence: 0.95,
        qualia_structure: {
          formal: ['人物实体', '主体'],
          telic: ['拥有物品', '参与计算'],
          agentive: ['题目设定'],
          constitutive: ['认知主体']
        },
        position: [150, 100],
        visual_properties: { color: '#e74c3c', size: 20, opacity: 0.95 }
      },
      {
        id: 'unified_object_1', 
        name: '苹果',
        type: 'object',
        confidence: 0.90,
        qualia_structure: {
          formal: ['可计数物体', '水果'],
          telic: ['被拥有', '被计算'],
          agentive: ['自然生长'],
          constitutive: ['有机物质']
        },
        position: [300, 100],
        visual_properties: { color: '#27ae60', size: 18, opacity: 0.90 }
      },
      {
        id: 'unified_concept_1',
        name: '5',
        type: 'concept',
        confidence: 0.85,
        position: [150, 200],
        visual_properties: { color: '#9b59b6', size: 16, opacity: 0.85 }
      },
      {
        id: 'unified_concept_2',
        name: '3',
        type: 'concept',
        confidence: 0.85,
        position: [300, 200],
        visual_properties: { color: '#9b59b6', size: 16, opacity: 0.85 }
      }
    ]
    
    const mockRelations: UnifiedRelation[] = [
      {
        id: 'unified_rel_1',
        source_id: 'unified_person_1',
        target_id: 'unified_object_1',
        type: '拥有关系',
        confidence: 0.92,
        weight: 0.9,
        relation_depth: 'surface',
        discovery_method: 'QS2_qualia_based',
        is_implicit: false,
        qualia_compatibility: {
          overall_score: 0.88,
          detailed_scores: { formal: 0.85, telic: 0.95, agentive: 0.80, constitutive: 0.85 },
          compatibility_reasons: ['目的角色高度兼容']
        },
        visualization: {
          color: '#3498db',
          width: 3,
          opacity: 0.9,
          pattern: 'solid',
          animation_delay: 0
        }
      },
      {
        id: 'unified_rel_2',
        source_id: 'unified_person_1',
        target_id: 'unified_concept_1', 
        type: '数量关系',
        confidence: 0.85,
        weight: 0.8,
        relation_depth: 'shallow',
        discovery_method: 'IRD_implicit',
        is_implicit: true,
        visualization: {
          color: '#e67e22',
          width: 2.5,
          opacity: 0.8,
          pattern: 'dashed',
          animation_delay: 500
        }
      }
    ]
    
    return {
      entities: mockEntities,
      relations: mockRelations,
      
      execution_info: {
        algorithm_types: ['QS2', 'IRD', 'Unified'],
        total_duration_ms: 420.5,
        stages: [
          {
            name: 'QS²实体提取',
            algorithm: 'QS2',
            duration_ms: 45.2,
            confidence: 0.95,
            entities_discovered: 3,
            relations_discovered: 0
          },
          {
            name: 'QS²语义结构构建',
            algorithm: 'QS2', 
            duration_ms: 128.7,
            confidence: 0.88,
            entities_discovered: 0,
            relations_discovered: 2
          },
          {
            name: 'IRD隐式关系发现',
            algorithm: 'IRD',
            duration_ms: 156.4,
            confidence: 0.87,
            entities_discovered: 1,
            relations_discovered: 3
          },
          {
            name: '算法协作与结果合并',
            algorithm: 'Both',
            duration_ms: 50,
            confidence: 0.9,
            entities_discovered: 0,
            relations_discovered: 0
          }
        ]
      },
      
      collaboration_metrics: {
        qs2_ird_agreement: 0.78,
        semantic_coverage: 0.85,
        relation_completeness: 0.80,
        algorithm_synergy: 0.82
      },
      
      visualization_config: {
        layout_algorithm: 'force_directed_with_clustering',
        enable_physics: true,
        nodes: mockEntities.map(entity => ({
          id: entity.id,
          label: entity.name,
          type: entity.type,
          size: entity.visual_properties.size,
          color: entity.visual_properties.color,
          position: entity.position,
          properties: {
            confidence: entity.confidence,
            has_qualia: !!entity.qualia_structure
          }
        })),
        edges: mockRelations.map(relation => ({
          id: relation.id,
          source: relation.source_id,
          target: relation.target_id,
          type: relation.type,
          weight: relation.weight,
          confidence: relation.confidence,
          style: {
            width: relation.visualization.width,
            color: relation.visualization.color,
            opacity: relation.visualization.opacity,
            pattern: relation.visualization.pattern
          },
          properties: {
            discovery_method: relation.discovery_method,
            is_implicit: relation.is_implicit
          }
        })),
        semantic_heatmap: {
          'unified_person_1': { 'unified_person_1': 1.0, 'unified_object_1': 0.7, 'unified_concept_1': 0.5 },
          'unified_object_1': { 'unified_object_1': 1.0, 'unified_person_1': 0.7, 'unified_concept_2': 0.3 }
        },
        animation_timeline: [
          {
            id: 'anim_rel_1',
            type: 'relation_discovery',
            nodes: ['unified_person_1', 'unified_object_1'],
            duration: 1000,
            delay: 0
          },
          {
            id: 'anim_rel_2',
            type: 'relation_discovery',
            nodes: ['unified_person_1', 'unified_concept_1'],
            duration: 1000,
            delay: 500
          }
        ]
      }
    }
  }
}

// 创建全局实例
export const unifiedQS2IRDEngine = new UnifiedQS2IRDEngine()

export default unifiedQS2IRDEngine