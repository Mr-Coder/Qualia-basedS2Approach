/**
 * 增强版实体关系图引擎
 * 核心处理逻辑：推理路径动画、智能布局、协作管理
 */

import {
  EnhancedDiagramProps,
  DiagramState,
  ReasoningPath,
  AnimatedEntity,
  AnimatedRelationship,
  LayoutConfig,
  LayoutResult,
  AnimationFrame,
  CollaborationEvent,
  Position,
  DiagramError
} from '../types/DiagramTypes'
import { LayoutEngine } from './LayoutEngine'
import { AnimationEngine } from './AnimationEngine'
import { CollaborationEngine } from './CollaborationEngine'

export class DiagramEngine {
  private state: DiagramState
  private layoutEngine: LayoutEngine
  private animationEngine: AnimationEngine
  private collaborationEngine: CollaborationEngine
  private canvas: HTMLCanvasElement | null = null
  private context: CanvasRenderingContext2D | null = null
  private animationId: number | null = null
  
  constructor(private props: EnhancedDiagramProps) {
    // Validate input props before initialization
    this.validateProps(props)
    
    this.layoutEngine = new LayoutEngine()
    this.animationEngine = new AnimationEngine()
    this.collaborationEngine = new CollaborationEngine()
    
    this.state = {
      entities: new Map(),
      relationships: new Map(),
      currentLayout: this.createEmptyLayout(),
      animationQueue: [],
      collaborationState: null,
      isPlaying: false,
      currentStep: 0,
      totalSteps: 0
    }
    
    this.initialize()
  }
  
  // ====== 输入验证 ======
  
  private validateProps(props: EnhancedDiagramProps): void {
    if (!props) {
      throw new DiagramError('Props cannot be null or undefined', 'PROPS_NULL')
    }
    
    // 验证实体
    if (!Array.isArray(props.entities)) {
      throw new DiagramError('Entities must be an array', 'ENTITIES_INVALID_TYPE')
    }
    
    props.entities.forEach((entity, index) => {
      if (!entity.id || typeof entity.id !== 'string' || entity.id.trim() === '') {
        throw new DiagramError(
          `Entity at index ${index} has invalid id`, 
          'ENTITY_INVALID_ID',
          { index, entity }
        )
      }
      
      if (!entity.name || typeof entity.name !== 'string' || entity.name.trim() === '') {
        throw new DiagramError(
          `Entity "${entity.id}" has invalid name`, 
          'ENTITY_INVALID_NAME',
          { entityId: entity.id }
        )
      }
      
      if (!entity.type || typeof entity.type !== 'string') {
        throw new DiagramError(
          `Entity "${entity.id}" has invalid type`, 
          'ENTITY_INVALID_TYPE',
          { entityId: entity.id }
        )
      }
    })
    
    // 检查实体ID重复
    const entityIds = props.entities.map(e => e.id)
    const duplicateIds = entityIds.filter((id, index) => entityIds.indexOf(id) !== index)
    if (duplicateIds.length > 0) {
      throw new DiagramError(
        'Duplicate entity IDs found', 
        'ENTITIES_DUPLICATE_ID',
        { duplicateIds }
      )
    }
    
    // 验证关系
    if (!Array.isArray(props.relationships)) {
      throw new DiagramError('Relationships must be an array', 'RELATIONSHIPS_INVALID_TYPE')
    }
    
    props.relationships.forEach((rel, index) => {
      if (!rel.source || typeof rel.source !== 'string' || rel.source.trim() === '') {
        throw new DiagramError(
          `Relationship at index ${index} has invalid source`, 
          'RELATIONSHIP_INVALID_SOURCE',
          { index, relationship: rel }
        )
      }
      
      if (!rel.target || typeof rel.target !== 'string' || rel.target.trim() === '') {
        throw new DiagramError(
          `Relationship at index ${index} has invalid target`, 
          'RELATIONSHIP_INVALID_TARGET',
          { index, relationship: rel }
        )
      }
      
      if (!rel.type || typeof rel.type !== 'string' || rel.type.trim() === '') {
        throw new DiagramError(
          `Relationship at index ${index} has invalid type`, 
          'RELATIONSHIP_INVALID_TYPE',
          { index, relationship: rel }
        )
      }
      
      // 验证关系权重
      if (rel.weight !== undefined) {
        if (typeof rel.weight !== 'number' || isNaN(rel.weight)) {
          throw new DiagramError(
            `Relationship at index ${index} has invalid weight`, 
            'RELATIONSHIP_INVALID_WEIGHT',
            { index, relationship: rel }
          )
        }
        
        if (rel.weight < 0 || rel.weight > 1) {
          console.warn(`Relationship weight should be between 0 and 1, got ${rel.weight}`)
        }
      }
      
      // 验证引用的实体存在
      const entityExists = (entityId: string) => 
        props.entities.some(e => e.id === entityId)
      
      if (!entityExists(rel.source)) {
        throw new DiagramError(
          `Relationship references non-existent source entity: ${rel.source}`, 
          'RELATIONSHIP_ORPHANED_SOURCE',
          { index, relationship: rel }
        )
      }
      
      if (!entityExists(rel.target)) {
        throw new DiagramError(
          `Relationship references non-existent target entity: ${rel.target}`, 
          'RELATIONSHIP_ORPHANED_TARGET',
          { index, relationship: rel }
        )
      }
    })
    
    // 验证推理路径
    if (props.reasoningPaths && !Array.isArray(props.reasoningPaths)) {
      throw new DiagramError('ReasoningPaths must be an array', 'REASONING_PATHS_INVALID_TYPE')
    }
    
    props.reasoningPaths?.forEach((path, pathIndex) => {
      if (!path.id || typeof path.id !== 'string') {
        throw new DiagramError(
          `ReasoningPath at index ${pathIndex} has invalid id`, 
          'REASONING_PATH_INVALID_ID',
          { pathIndex, path }
        )
      }
      
      if (!Array.isArray(path.steps)) {
        throw new DiagramError(
          `ReasoningPath "${path.id}" has invalid steps`, 
          'REASONING_PATH_INVALID_STEPS',
          { pathIndex, path }
        )
      }
      
      path.steps.forEach((step, stepIndex) => {
        if (!step.id || typeof step.id !== 'string') {
          throw new DiagramError(
            `ReasoningStep at path ${pathIndex}, step ${stepIndex} has invalid id`, 
            'REASONING_STEP_INVALID_ID',
            { pathIndex, stepIndex, step }
          )
        }
        
        if (!Array.isArray(step.inputEntities) || !Array.isArray(step.outputEntities)) {
          throw new DiagramError(
            `ReasoningStep "${step.id}" has invalid entity arrays`, 
            'REASONING_STEP_INVALID_ENTITIES',
            { pathIndex, stepIndex, step }
          )
        }
        
        // 验证步骤中引用的实体存在
        const allReferencedEntities = [...step.inputEntities, ...step.outputEntities]
        allReferencedEntities.forEach(entityId => {
          if (!props.entities.some(e => e.id === entityId)) {
            throw new DiagramError(
              `ReasoningStep "${step.id}" references non-existent entity: ${entityId}`, 
              'REASONING_STEP_ORPHANED_ENTITY',
              { pathIndex, stepIndex, step, entityId }
            )
          }
        })
        
        // 验证时间和置信度
        if (step.confidence !== undefined && 
            (typeof step.confidence !== 'number' || step.confidence < 0 || step.confidence > 1)) {
          console.warn(`Invalid confidence value for step "${step.id}": ${step.confidence}`)
        }
        
        if (step.duration !== undefined && 
            (typeof step.duration !== 'number' || step.duration < 0)) {
          console.warn(`Invalid duration value for step "${step.id}": ${step.duration}`)
        }
      })
    })
    
    // 验证布局配置
    if (props.layoutConfig) {
      const validLayoutTypes = ['force', 'hierarchical', 'circular', 'timeline', 'clustered']
      if (!validLayoutTypes.includes(props.layoutConfig.type)) {
        throw new DiagramError(
          `Invalid layout type: ${props.layoutConfig.type}`, 
          'LAYOUT_INVALID_TYPE',
          { layoutConfig: props.layoutConfig }
        )
      }
    }
    
    // 验证动画配置
    if (props.animationConfig) {
      const { animationSpeed, simultaneousSteps, pauseBetweenSteps } = props.animationConfig
      
      if (animationSpeed !== undefined && 
          (typeof animationSpeed !== 'number' || animationSpeed <= 0)) {
        throw new DiagramError(
          `Invalid animation speed: ${animationSpeed}`, 
          'ANIMATION_INVALID_SPEED'
        )
      }
      
      if (simultaneousSteps !== undefined && 
          (typeof simultaneousSteps !== 'number' || simultaneousSteps < 1)) {
        throw new DiagramError(
          `Invalid simultaneousSteps: ${simultaneousSteps}`, 
          'ANIMATION_INVALID_SIMULTANEOUS_STEPS'
        )
      }
      
      if (pauseBetweenSteps !== undefined && 
          (typeof pauseBetweenSteps !== 'number' || pauseBetweenSteps < 0)) {
        throw new DiagramError(
          `Invalid pauseBetweenSteps: ${pauseBetweenSteps}`, 
          'ANIMATION_INVALID_PAUSE'
        )
      }
    }
    
    // 验证协作配置
    if (props.collaborationConfig?.enabled) {
      if (!props.collaborationConfig.sessionId || 
          typeof props.collaborationConfig.sessionId !== 'string') {
        throw new DiagramError(
          'Valid sessionId required for collaboration', 
          'COLLABORATION_INVALID_SESSION_ID'
        )
      }
      
      if (!props.collaborationConfig.currentUserId || 
          typeof props.collaborationConfig.currentUserId !== 'string') {
        throw new DiagramError(
          'Valid currentUserId required for collaboration', 
          'COLLABORATION_INVALID_USER_ID'
        )
      }
    }
  }
  
  // ====== 初始化方法 ======
  
  private async initialize(): Promise<void> {
    try {
      // 验证数据完整性
      this.validateDataIntegrity()
      
      // 1. 转换基础数据为动画实体
      this.convertEntitiesToAnimated()
      this.convertRelationshipsToAnimated()
      
      // 2. 计算智能布局
      await this.computeOptimalLayout()
      
      // 3. 生成推理路径动画序列
      if (this.props.reasoningPaths && this.props.reasoningPaths.length > 0) {
        this.generateAnimationSequence()
      }
      
      // 4. 初始化协作模式
      if (this.props.collaborationConfig?.enabled) {
        await this.initializeCollaboration()
      }
      
    } catch (error) {
      const diagramError = error instanceof DiagramError 
        ? error 
        : new DiagramError(
            'Failed to initialize diagram engine',
            'INIT_ERROR',
            error
          )
      this.handleError(diagramError)
      throw diagramError
    }
  }
  
  private validateDataIntegrity(): void {
    // 检查是否有孤立实体（没有任何关系的实体）
    const connectedEntities = new Set<string>()
    this.props.relationships.forEach(rel => {
      connectedEntities.add(rel.source)
      connectedEntities.add(rel.target)
    })
    
    const isolatedEntities = this.props.entities
      .filter(entity => !connectedEntities.has(entity.id))
      .map(entity => entity.id)
    
    if (isolatedEntities.length > 0) {
      console.warn('Found isolated entities (no relationships):', isolatedEntities)
    }
    
    // 检查循环引用
    this.detectCircularReferences()
    
    // 验证推理路径的逻辑连贯性
    if (this.props.reasoningPaths) {
      this.validateReasoningLogic()
    }
  }
  
  private detectCircularReferences(): void {
    const visited = new Set<string>()
    const recursionStack = new Set<string>()
    
    const hasCycle = (entityId: string): boolean => {
      if (recursionStack.has(entityId)) return true
      if (visited.has(entityId)) return false
      
      visited.add(entityId)
      recursionStack.add(entityId)
      
      // 查找所有从当前实体出发的关系
      const outgoingRelationships = this.props.relationships
        .filter(rel => rel.source === entityId)
      
      for (const rel of outgoingRelationships) {
        if (hasCycle(rel.target)) {
          return true
        }
      }
      
      recursionStack.delete(entityId)
      return false
    }
    
    for (const entity of this.props.entities) {
      if (!visited.has(entity.id) && hasCycle(entity.id)) {
        console.warn(`Circular reference detected involving entity: ${entity.id}`)
      }
    }
  }
  
  private validateReasoningLogic(): void {
    this.props.reasoningPaths?.forEach((path, pathIndex) => {
      // 检查步骤序列的连贯性
      for (let i = 1; i < path.steps.length; i++) {
        const prevStep = path.steps[i - 1]
        const currentStep = path.steps[i]
        
        // 检查时间序列
        if (prevStep.timestamp && currentStep.timestamp && 
            prevStep.timestamp >= currentStep.timestamp) {
          console.warn(
            `Time sequence issue in path "${path.id}": step ${i-1} occurs after step ${i}`
          )
        }
        
        // 检查逻辑连接
        const hasLogicalConnection = prevStep.outputEntities.some(outputEntity => 
          currentStep.inputEntities.includes(outputEntity)
        )
        
        if (!hasLogicalConnection && prevStep.outputEntities.length > 0) {
          console.warn(
            `Logical gap in path "${path.id}": step "${prevStep.id}" output doesn't connect to step "${currentStep.id}" input`
          )
        }
      }
      
      // 验证置信度合理性
      const stepConfidences = path.steps
        .map(step => step.confidence)
        .filter(conf => conf !== undefined) as number[]
      
      if (stepConfidences.length > 0) {
        const avgConfidence = stepConfidences.reduce((sum, conf) => sum + conf, 0) / stepConfidences.length
        
        if (path.totalConfidence && Math.abs(path.totalConfidence - avgConfidence) > 0.2) {
          console.warn(
            `Total confidence (${path.totalConfidence}) differs significantly from average step confidence (${avgConfidence.toFixed(2)}) in path "${path.id}"`
          )
        }
      }
    })
  }
  
  private convertEntitiesToAnimated(): void {
    this.props.entities.forEach(entity => {
      try {
        const animatedEntity: AnimatedEntity = {
          ...entity,
          position: { x: 0, y: 0 }, // 将在布局计算中设置
          animationState: 'idle',
          appearanceTime: 0,
          lifecycle: {
            created: Date.now(),
            modified: [],
            accessed: []
          },
          visualProps: {
            size: this.calculateEntitySize(entity),
            opacity: 1,
            borderWidth: 2,
            glowIntensity: 0
          }
        }
        
        // 验证计算结果
        if (!isFinite(animatedEntity.visualProps.size) || animatedEntity.visualProps.size <= 0) {
          console.warn(`Invalid entity size for "${entity.id}", using default`)
          animatedEntity.visualProps.size = 30
        }
        
        this.state.entities.set(entity.id, animatedEntity)
      } catch (error) {
        throw new DiagramError(
          `Failed to convert entity "${entity.id}" to animated entity`,
          'ENTITY_CONVERSION_ERROR',
          { entity, error }
        )
      }
    })
  }
  
  private convertRelationshipsToAnimated(): void {
    this.props.relationships.forEach((relationship, index) => {
      try {
        const animatedRelationship: AnimatedRelationship = {
          ...relationship,
          startEntity: relationship.source,
          endEntity: relationship.target,
          animationState: 'idle',
          flowDirection: 'forward',
          activationTime: 0,
          strength: relationship.weight || 0.5,
          visualProps: {
            thickness: this.calculateRelationshipThickness(relationship),
            opacity: 0.8,
            curvature: 0
          }
        }
        
        // 验证计算结果
        if (!isFinite(animatedRelationship.visualProps.thickness) || animatedRelationship.visualProps.thickness <= 0) {
          console.warn(`Invalid relationship thickness for relationship ${index}, using default`)
          animatedRelationship.visualProps.thickness = 2
        }
        
        if (!isFinite(animatedRelationship.strength) || animatedRelationship.strength < 0) {
          console.warn(`Invalid relationship strength for relationship ${index}, using default`)
          animatedRelationship.strength = 0.5
        }
        
        const relationshipId = `${relationship.source}-${relationship.target}`
        this.state.relationships.set(relationshipId, animatedRelationship)
      } catch (error) {
        throw new DiagramError(
          `Failed to convert relationship at index ${index} to animated relationship`,
          'RELATIONSHIP_CONVERSION_ERROR',
          { relationship, index, error }
        )
      }
    })
  }
  
  // ====== 智能布局系统 ======
  
  public async computeOptimalLayout(): Promise<LayoutResult> {
    try {
      const { layoutConfig, autoOptimizeLayout } = this.props
      
      // 如果启用自动优化，先分析最佳布局类型
      let optimalConfig = layoutConfig
      if (autoOptimizeLayout) {
        optimalConfig = await this.analyzeOptimalLayoutType()
      }
      
      // 计算布局
      const layoutResult = await this.layoutEngine.computeLayout(
        Array.from(this.state.entities.values()),
        Array.from(this.state.relationships.values()),
        optimalConfig
      )
      
      // 应用布局结果到实体位置
      this.applyLayoutToEntities(layoutResult)
      
      this.state.currentLayout = layoutResult
      this.props.onLayoutChange?.(layoutResult)
      
      return layoutResult
      
    } catch (error) {
      throw new DiagramError(
        'Layout computation failed',
        'LAYOUT_ERROR',
        error
      )
    }
  }
  
  private async analyzeOptimalLayoutType(): Promise<LayoutConfig> {
    const entityCount = this.state.entities.size
    const relationshipCount = this.state.relationships.size
    const hasTimeSequence = this.props.reasoningPaths && 
      this.props.reasoningPaths.some(path => path.steps.length > 1)
    
    // 智能选择布局算法
    if (hasTimeSequence) {
      return {
        type: 'timeline',
        params: {
          timelineOrientation: 'horizontal',
          timeScale: 100
        }
      }
    } else if (entityCount > 20) {
      return {
        type: 'clustered',
        params: {
          clusterMethod: 'type',
          clusterSpacing: 150
        }
      }
    } else if (relationshipCount / entityCount > 2) {
      return {
        type: 'force',
        params: {
          forceStrength: 0.8,
          linkDistance: 100,
          centerForce: 0.3
        }
      }
    } else {
      return {
        type: 'hierarchical',
        params: {
          direction: 'top-down',
          levelSeparation: 120,
          nodeSeparation: 80
        }
      }
    }
  }
  
  private applyLayoutToEntities(layoutResult: LayoutResult): void {
    layoutResult.entityPositions.forEach((position, entityId) => {
      const entity = this.state.entities.get(entityId)
      if (entity) {
        entity.targetPosition = position
        // 如果是初始化，直接设置位置；否则会触发动画
        if (entity.position.x === 0 && entity.position.y === 0) {
          entity.position = { ...position }
        }
      }
    })
  }
  
  // ====== 推理路径动画系统 ======
  
  private generateAnimationSequence(): void {
    if (!this.props.reasoningPaths) return
    
    const animationFrames: AnimationFrame[] = []
    let currentTime = 0
    
    // 为每个推理路径生成动画帧
    this.props.reasoningPaths.forEach(path => {
      path.steps.forEach((step, stepIndex) => {
        // 步骤开始帧
        const startFrame: AnimationFrame = {
          timestamp: currentTime,
          entityUpdates: new Map(),
          relationshipUpdates: new Map(),
          highlights: [...step.inputEntities, ...step.outputEntities]
        }
        
        // 高亮输入实体
        step.inputEntities.forEach(entityId => {
          startFrame.entityUpdates.set(entityId, {
            animationState: 'highlighted',
            visualProps: { glowIntensity: 0.8 }
          })
        })
        
        // 激活相关关系
        this.findRelationshipsForStep(step).forEach(relationshipId => {
          startFrame.relationshipUpdates.set(relationshipId, {
            animationState: 'active',
            visualProps: { opacity: 1.0 }
          })
        })
        
        animationFrames.push(startFrame)
        currentTime += step.duration
        
        // 步骤结束帧
        const endFrame: AnimationFrame = {
          timestamp: currentTime,
          entityUpdates: new Map(),
          relationshipUpdates: new Map(),
          highlights: step.outputEntities
        }
        
        // 高亮输出实体
        step.outputEntities.forEach(entityId => {
          endFrame.entityUpdates.set(entityId, {
            animationState: 'processing',
            visualProps: { glowIntensity: 1.0 }
          })
        })
        
        animationFrames.push(endFrame)
        currentTime += this.props.animationConfig.pauseBetweenSteps
      })
    })
    
    this.state.animationQueue = animationFrames
    this.state.totalSteps = animationFrames.length
  }
  
  private findRelationshipsForStep(step: ReasoningStep): string[] {
    const relationshipIds: string[] = []
    
    // 查找涉及输入和输出实体的所有关系
    step.inputEntities.forEach(inputId => {
      step.outputEntities.forEach(outputId => {
        const forwardId = `${inputId}-${outputId}`
        const backwardId = `${outputId}-${inputId}`
        
        if (this.state.relationships.has(forwardId)) {
          relationshipIds.push(forwardId)
        }
        if (this.state.relationships.has(backwardId)) {
          relationshipIds.push(backwardId)
        }
      })
    })
    
    return relationshipIds
  }
  
  // ====== 动画播放控制 ======
  
  public playAnimation(): void {
    if (this.state.animationQueue.length === 0) {
      console.warn('No animation sequence available')
      return
    }
    
    this.state.isPlaying = true
    this.state.currentStep = 0
    
    const startTime = performance.now()
    
    const animate = (currentTime: number) => {
      const elapsed = (currentTime - startTime) * this.props.animationConfig.animationSpeed
      
      // 执行当前时间点的所有动画帧
      while (
        this.state.currentStep < this.state.animationQueue.length &&
        this.state.animationQueue[this.state.currentStep].timestamp <= elapsed
      ) {
        this.executeAnimationFrame(this.state.animationQueue[this.state.currentStep])
        this.state.currentStep++
      }
      
      // 继续动画或结束
      if (this.state.currentStep < this.state.animationQueue.length) {
        this.animationId = requestAnimationFrame(animate)
      } else {
        this.state.isPlaying = false
        this.onAnimationComplete()
      }
    }
    
    this.animationId = requestAnimationFrame(animate)
  }
  
  public pauseAnimation(): void {
    this.state.isPlaying = false
    if (this.animationId) {
      cancelAnimationFrame(this.animationId)
      this.animationId = null
    }
  }
  
  public resetAnimation(): void {
    this.pauseAnimation()
    this.state.currentStep = 0
    this.resetEntityStates()
    this.resetRelationshipStates()
  }
  
  private executeAnimationFrame(frame: AnimationFrame): void {
    // 更新实体状态
    frame.entityUpdates.forEach((updates, entityId) => {
      const entity = this.state.entities.get(entityId)
      if (entity) {
        Object.assign(entity, updates)
      }
    })
    
    // 更新关系状态
    frame.relationshipUpdates.forEach((updates, relationshipId) => {
      const relationship = this.state.relationships.get(relationshipId)
      if (relationship) {
        Object.assign(relationship, updates)
      }
    })
  }
  
  private onAnimationComplete(): void {
    console.log('Animation sequence completed')
    // 触发完成事件
    this.props.onReasoningStepClick?.({} as any) // 可以传递完成状态
  }
  
  // ====== 协作学习系统 ======
  
  private async initializeCollaboration(): Promise<void> {
    if (!this.props.collaborationConfig?.sessionId) return
    
    try {
      this.state.collaborationState = await this.collaborationEngine.joinSession(
        this.props.collaborationConfig.sessionId,
        this.props.collaborationConfig.currentUserId!
      )
      
      // 监听协作事件
      this.collaborationEngine.onEvent((event: CollaborationEvent) => {
        this.handleCollaborationEvent(event)
        this.props.onCollaborationEvent?.(event)
      })
      
    } catch (error) {
      throw new DiagramError(
        'Failed to initialize collaboration',
        'COLLABORATION_ERROR',
        error
      )
    }
  }
  
  private handleCollaborationEvent(event: CollaborationEvent): void {
    switch (event.type) {
      case 'entity_move':
        this.handleRemoteEntityMove(event)
        break
      case 'step_highlight':
        this.handleRemoteStepHighlight(event)
        break
      case 'comment_add':
        this.handleRemoteComment(event)
        break
    }
  }
  
  private handleRemoteEntityMove(event: CollaborationEvent): void {
    const { entityId, newPosition } = event.data
    const entity = this.state.entities.get(entityId)
    if (entity) {
      entity.targetPosition = newPosition
      entity.animationState = 'moving'
    }
  }
  
  private handleRemoteStepHighlight(event: CollaborationEvent): void {
    const { stepIndex } = event.data
    if (stepIndex >= 0 && stepIndex < this.state.animationQueue.length) {
      this.executeAnimationFrame(this.state.animationQueue[stepIndex])
    }
  }
  
  private handleRemoteComment(event: CollaborationEvent): void {
    // 处理远程评论添加
    console.log('Remote comment added:', event.data)
  }
  
  // ====== 工具方法 ======
  
  private calculateEntitySize(entity: any): number {
    // 根据实体类型和重要性计算大小
    const baseSize = 30
    const typeMultiplier = {
      'person': 1.2,
      'object': 1.0,
      'money': 0.8,
      'concept': 1.1
    }
    return baseSize * (typeMultiplier[entity.type as keyof typeof typeMultiplier] || 1.0)
  }
  
  private calculateRelationshipThickness(relationship: any): number {
    const baseThickness = 2
    return baseThickness * (relationship.weight || 0.5) * 2
  }
  
  private resetEntityStates(): void {
    this.state.entities.forEach(entity => {
      entity.animationState = 'idle'
      entity.visualProps.glowIntensity = 0
    })
  }
  
  private resetRelationshipStates(): void {
    this.state.relationships.forEach(relationship => {
      relationship.animationState = 'idle'
      relationship.visualProps.opacity = 0.8
    })
  }
  
  private createEmptyLayout(): LayoutResult {
    return {
      entityPositions: new Map(),
      relationshipPaths: new Map(),
      boundingBox: { x: 0, y: 0, width: 0, height: 0 },
      quality: 0,
      computationTime: 0,
      warnings: []
    }
  }
  
  private handleError(error: DiagramError): void {
    console.error(`DiagramEngine Error [${error.code}]:`, error.message, error.details)
    // 可以添加错误上报逻辑
  }
  
  // ====== 公共接口 ======
  
  public getState(): DiagramState {
    return { ...this.state }
  }
  
  public updateProps(newProps: Partial<EnhancedDiagramProps>): void {
    try {
      // 验证新属性
      if (newProps.entities || newProps.relationships) {
        const mergedProps = { ...this.props, ...newProps }
        this.validateProps(mergedProps)
      }
      
      Object.assign(this.props, newProps)
      
      // 重新计算必要的部分
      if (newProps.entities || newProps.relationships) {
        this.initialize()
      } else if (newProps.layoutConfig) {
        this.computeOptimalLayout()
      } else if (newProps.reasoningPaths) {
        this.generateAnimationSequence()
      }
    } catch (error) {
      const diagramError = error instanceof DiagramError 
        ? error 
        : new DiagramError(
            'Failed to update props',
            'PROPS_UPDATE_ERROR',
            error
          )
      this.handleError(diagramError)
      throw diagramError
    }
  }
  
  public dispose(): void {
    this.pauseAnimation()
    this.collaborationEngine.disconnect()
  }
}