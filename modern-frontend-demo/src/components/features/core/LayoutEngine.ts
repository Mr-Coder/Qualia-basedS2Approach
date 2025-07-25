/**
 * 智能布局引擎
 * 实现多种布局算法：力导向、分层、聚类、时间轴
 */

import {
  LayoutConfig,
  LayoutResult,
  AnimatedEntity,
  AnimatedRelationship,
  Position,
  BoundingBox,
  DiagramError
} from '../types/DiagramTypes'

interface ForceNode {
  id: string
  x: number
  y: number
  vx: number
  vy: number
  fx?: number // 固定位置
  fy?: number
  mass: number
}

interface ForceLink {
  source: string
  target: string
  distance: number
  strength: number
}

export class LayoutEngine {
  private width: number = 800
  private height: number = 600
  private maxIterations: number = 300
  
  public async computeLayout(
    entities: AnimatedEntity[],
    relationships: AnimatedRelationship[],
    config: LayoutConfig
  ): Promise<LayoutResult> {
    const startTime = performance.now()
    
    try {
      let result: LayoutResult
      
      switch (config.type) {
        case 'force':
          result = await this.computeForceDirectedLayout(entities, relationships, config)
          break
        case 'hierarchical':
          result = await this.computeHierarchicalLayout(entities, relationships, config)
          break
        case 'circular':
          result = await this.computeCircularLayout(entities, relationships, config)
          break
        case 'timeline':
          result = await this.computeTimelineLayout(entities, relationships, config)
          break
        case 'clustered':
          result = await this.computeClusteredLayout(entities, relationships, config)
          break
        default:
          throw new DiagramError('Unsupported layout type', 'LAYOUT_TYPE_ERROR', config.type)
      }
      
      result.computationTime = performance.now() - startTime
      return result
      
    } catch (error) {
      throw new DiagramError(
        'Layout computation failed',
        'LAYOUT_COMPUTATION_ERROR',
        { error, config }
      )
    }
  }
  
  // ====== 力导向布局 ======
  
  private async computeForceDirectedLayout(
    entities: AnimatedEntity[],
    relationships: AnimatedRelationship[],
    config: LayoutConfig
  ): Promise<LayoutResult> {
    const params = config.params
    const forceStrength = params?.forceStrength || 0.6
    const linkDistance = params?.linkDistance || 80
    const centerForce = params?.centerForce || 0.3
    
    // 初始化节点
    const nodes: ForceNode[] = entities.map(entity => ({
      id: entity.id,
      x: Math.random() * this.width,
      y: Math.random() * this.height,
      vx: 0,
      vy: 0,
      mass: entity.visualProps.size / 30 // 归一化质量
    }))
    
    // 初始化连接
    const links: ForceLink[] = relationships.map(rel => ({
      source: rel.startEntity,
      target: rel.endEntity,
      distance: linkDistance * (rel.strength || 1),
      strength: forceStrength
    }))
    
    // 力模拟迭代
    for (let iteration = 0; iteration < this.maxIterations; iteration++) {
      // 重置力
      nodes.forEach(node => {
        node.vx = 0
        node.vy = 0
      })
      
      // 计算排斥力 (所有节点间)
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const nodeA = nodes[i]
          const nodeB = nodes[j]
          
          const dx = nodeB.x - nodeA.x
          const dy = nodeB.y - nodeA.y
          const distance = Math.sqrt(dx * dx + dy * dy) || 0.1
          
          const force = (forceStrength * 100) / (distance * distance)
          const fx = (dx / distance) * force
          const fy = (dy / distance) * force
          
          nodeA.vx -= fx / nodeA.mass
          nodeA.vy -= fy / nodeA.mass
          nodeB.vx += fx / nodeB.mass
          nodeB.vy += fy / nodeB.mass
        }
      }
      
      // 计算吸引力 (连接的节点间)
      links.forEach(link => {
        const sourceNode = nodes.find(n => n.id === link.source)
        const targetNode = nodes.find(n => n.id === link.target)
        
        if (sourceNode && targetNode) {
          const dx = targetNode.x - sourceNode.x
          const dy = targetNode.y - sourceNode.y
          const distance = Math.sqrt(dx * dx + dy * dy) || 0.1
          
          const force = link.strength * (distance - link.distance) / distance
          const fx = dx * force
          const fy = dy * force
          
          sourceNode.vx += fx / sourceNode.mass
          sourceNode.vy += fy / sourceNode.mass
          targetNode.vx -= fx / targetNode.mass
          targetNode.vy -= fy / targetNode.mass
        }
      })
      
      // 向心力
      if (centerForce > 0) {
        const centerX = this.width / 2
        const centerY = this.height / 2
        
        nodes.forEach(node => {
          const dx = centerX - node.x
          const dy = centerY - node.y
          
          node.vx += dx * centerForce * 0.01
          node.vy += dy * centerForce * 0.01
        })
      }
      
      // 更新位置
      nodes.forEach(node => {
        if (node.fx === undefined) {
          node.x += node.vx * 0.1 // 阻尼系数
        } else {
          node.x = node.fx
        }
        
        if (node.fy === undefined) {
          node.y += node.vy * 0.1
        } else {
          node.y = node.fy
        }
        
        // 边界约束
        node.x = Math.max(50, Math.min(this.width - 50, node.x))
        node.y = Math.max(50, Math.min(this.height - 50, node.y))
      })
      
      // 收敛检查
      const totalEnergy = nodes.reduce((sum, node) => 
        sum + node.vx * node.vx + node.vy * node.vy, 0
      )
      
      if (totalEnergy < 0.01) break
    }
    
    // 构建结果
    const entityPositions = new Map<string, Position>()
    nodes.forEach(node => {
      entityPositions.set(node.id, { x: node.x, y: node.y })
    })
    
    return {
      entityPositions,
      relationshipPaths: new Map(),
      boundingBox: this.calculateBoundingBox(entityPositions),
      quality: this.evaluateLayoutQuality(entityPositions, links),
      computationTime: 0,
      warnings: []
    }
  }
  
  // ====== 分层布局 ======
  
  private async computeHierarchicalLayout(
    entities: AnimatedEntity[],
    relationships: AnimatedRelationship[],
    config: LayoutConfig
  ): Promise<LayoutResult> {
    const params = config.params
    const direction = params?.direction || 'top-down'
    const levelSeparation = params?.levelSeparation || 120
    const nodeSeparation = params?.nodeSeparation || 80
    
    // 构建有向图
    const graph = this.buildDirectedGraph(entities, relationships)
    
    // 拓扑排序分层
    const levels = this.topologicalLayers(graph)
    
    // 计算位置
    const entityPositions = new Map<string, Position>()
    
    if (direction === 'top-down') {
      levels.forEach((level, levelIndex) => {
        const levelY = levelIndex * levelSeparation
        const startX = (this.width - (level.length - 1) * nodeSeparation) / 2
        
        level.forEach((nodeId, nodeIndex) => {
          entityPositions.set(nodeId, {
            x: startX + nodeIndex * nodeSeparation,
            y: levelY + 50
          })
        })
      })
    } else if (direction === 'left-right') {
      levels.forEach((level, levelIndex) => {
        const levelX = levelIndex * levelSeparation
        const startY = (this.height - (level.length - 1) * nodeSeparation) / 2
        
        level.forEach((nodeId, nodeIndex) => {
          entityPositions.set(nodeId, {
            x: levelX + 50,
            y: startY + nodeIndex * nodeSeparation
          })
        })
      })
    } else if (direction === 'radial') {
      // 径向布局
      const center = { x: this.width / 2, y: this.height / 2 }
      const radiusStep = Math.min(this.width, this.height) / 4 / levels.length
      
      levels.forEach((level, levelIndex) => {
        const radius = (levelIndex + 1) * radiusStep
        const angleStep = (2 * Math.PI) / level.length
        
        level.forEach((nodeId, nodeIndex) => {
          const angle = nodeIndex * angleStep
          entityPositions.set(nodeId, {
            x: center.x + radius * Math.cos(angle),
            y: center.y + radius * Math.sin(angle)
          })
        })
      })
    }
    
    return {
      entityPositions,
      relationshipPaths: new Map(),
      boundingBox: this.calculateBoundingBox(entityPositions),
      quality: this.evaluateHierarchicalQuality(levels),
      computationTime: 0,
      warnings: []
    }
  }
  
  // ====== 时间轴布局 ======
  
  private async computeTimelineLayout(
    entities: AnimatedEntity[],
    relationships: AnimatedRelationship[],
    config: LayoutConfig
  ): Promise<LayoutResult> {
    const params = config.params
    const orientation = params?.timelineOrientation || 'horizontal'
    const timeScale = params?.timeScale || 100
    
    // 按实体的生命周期时间排序
    const sortedEntities = entities.sort((a, b) => 
      a.lifecycle.created - b.lifecycle.created
    )
    
    const entityPositions = new Map<string, Position>()
    const startTime = Math.min(...entities.map(e => e.lifecycle.created))
    
    if (orientation === 'horizontal') {
      sortedEntities.forEach((entity, index) => {
        const timeOffset = (entity.lifecycle.created - startTime) * timeScale / 1000
        entityPositions.set(entity.id, {
          x: 50 + timeOffset,
          y: 100 + (index % 4) * 100 // 多行排列避免重叠
        })
      })
    } else {
      sortedEntities.forEach((entity, index) => {
        const timeOffset = (entity.lifecycle.created - startTime) * timeScale / 1000
        entityPositions.set(entity.id, {
          x: 100 + (index % 4) * 150,
          y: 50 + timeOffset
        })
      })
    }
    
    return {
      entityPositions,
      relationshipPaths: new Map(),
      boundingBox: this.calculateBoundingBox(entityPositions),
      quality: 0.8, // 时间轴布局质量较高
      computationTime: 0,
      warnings: []
    }
  }
  
  // ====== 聚类布局 ======
  
  private async computeClusteredLayout(
    entities: AnimatedEntity[],
    relationships: AnimatedRelationship[],
    config: LayoutConfig
  ): Promise<LayoutResult> {
    const params = config.params
    const clusterMethod = params?.clusterMethod || 'type'
    const clusterSpacing = params?.clusterSpacing || 200
    
    // 按方法聚类
    const clusters = this.clusterEntities(entities, clusterMethod)
    
    const entityPositions = new Map<string, Position>()
    const clusterCenters = this.calculateClusterCenters(clusters.length, clusterSpacing)
    
    clusters.forEach((cluster, clusterIndex) => {
      const center = clusterCenters[clusterIndex]
      
      // 在每个聚类内部使用圆形布局
      cluster.forEach((entity, entityIndex) => {
        const angle = (2 * Math.PI * entityIndex) / cluster.length
        const radius = Math.min(60, cluster.length * 8)
        
        entityPositions.set(entity.id, {
          x: center.x + radius * Math.cos(angle),
          y: center.y + radius * Math.sin(angle)
        })
      })
    })
    
    return {
      entityPositions,
      relationshipPaths: new Map(),
      boundingBox: this.calculateBoundingBox(entityPositions),
      quality: this.evaluateClusteringQuality(clusters),
      computationTime: 0,
      warnings: []
    }
  }
  
  // ====== 圆形布局 ======
  
  private async computeCircularLayout(
    entities: AnimatedEntity[],
    relationships: AnimatedRelationship[],
    config: LayoutConfig
  ): Promise<LayoutResult> {
    const center = { x: this.width / 2, y: this.height / 2 }
    const radius = Math.min(this.width, this.height) / 3
    const angleStep = (2 * Math.PI) / entities.length
    
    const entityPositions = new Map<string, Position>()
    
    entities.forEach((entity, index) => {
      const angle = index * angleStep
      entityPositions.set(entity.id, {
        x: center.x + radius * Math.cos(angle),
        y: center.y + radius * Math.sin(angle)
      })
    })
    
    return {
      entityPositions,
      relationshipPaths: new Map(),
      boundingBox: this.calculateBoundingBox(entityPositions),
      quality: 0.6, // 圆形布局质量中等
      computationTime: 0,
      warnings: []
    }
  }
  
  // ====== 辅助方法 ======
  
  private buildDirectedGraph(
    entities: AnimatedEntity[],
    relationships: AnimatedRelationship[]
  ): Map<string, string[]> {
    const graph = new Map<string, string[]>()
    
    entities.forEach(entity => {
      graph.set(entity.id, [])
    })
    
    relationships.forEach(rel => {
      const sources = graph.get(rel.startEntity) || []
      sources.push(rel.endEntity)
      graph.set(rel.startEntity, sources)
    })
    
    return graph
  }
  
  private topologicalLayers(graph: Map<string, string[]>): string[][] {
    const inDegree = new Map<string, number>()
    const nodes = Array.from(graph.keys())
    
    // 计算入度
    nodes.forEach(node => inDegree.set(node, 0))
    graph.forEach(neighbors => {
      neighbors.forEach(neighbor => {
        inDegree.set(neighbor, (inDegree.get(neighbor) || 0) + 1)
      })
    })
    
    const layers: string[][] = []
    const remaining = new Set(nodes)
    
    while (remaining.size > 0) {
      const currentLayer = Array.from(remaining).filter(node => 
        inDegree.get(node) === 0
      )
      
      if (currentLayer.length === 0) {
        // 处理循环依赖，选择入度最小的节点
        const minInDegree = Math.min(...Array.from(remaining).map(node => 
          inDegree.get(node) || 0
        ))
        const breakCycleNode = Array.from(remaining).find(node => 
          inDegree.get(node) === minInDegree
        )
        if (breakCycleNode) {
          currentLayer.push(breakCycleNode)
        }
      }
      
      layers.push(currentLayer)
      
      // 移除当前层节点并更新入度
      currentLayer.forEach(node => {
        remaining.delete(node)
        const neighbors = graph.get(node) || []
        neighbors.forEach(neighbor => {
          inDegree.set(neighbor, (inDegree.get(neighbor) || 0) - 1)
        })
      })
    }
    
    return layers
  }
  
  private clusterEntities(
    entities: AnimatedEntity[],
    method: string
  ): AnimatedEntity[][] {
    switch (method) {
      case 'type':
        return this.clusterByType(entities)
      case 'time':
        return this.clusterByTime(entities)
      case 'importance':
        return this.clusterByImportance(entities)
      default:
        return [entities] // 单个聚类
    }
  }
  
  private clusterByType(entities: AnimatedEntity[]): AnimatedEntity[][] {
    const clusters = new Map<string, AnimatedEntity[]>()
    
    entities.forEach(entity => {
      const cluster = clusters.get(entity.type) || []
      cluster.push(entity)
      clusters.set(entity.type, cluster)
    })
    
    return Array.from(clusters.values())
  }
  
  private clusterByTime(entities: AnimatedEntity[]): AnimatedEntity[][] {
    // 按创建时间聚类（简单的时间窗口方法）
    const sortedEntities = entities.sort((a, b) => 
      a.lifecycle.created - b.lifecycle.created
    )
    
    const clusters: AnimatedEntity[][] = []
    let currentCluster: AnimatedEntity[] = []
    let lastTime = 0
    const timeWindow = 60000 // 1分钟窗口
    
    sortedEntities.forEach(entity => {
      if (entity.lifecycle.created - lastTime > timeWindow && currentCluster.length > 0) {
        clusters.push(currentCluster)
        currentCluster = []
      }
      
      currentCluster.push(entity)
      lastTime = entity.lifecycle.created
    })
    
    if (currentCluster.length > 0) {
      clusters.push(currentCluster)
    }
    
    return clusters
  }
  
  private clusterByImportance(entities: AnimatedEntity[]): AnimatedEntity[][] {
    // 按重要性（大小）聚类
    const highImportance: AnimatedEntity[] = []
    const mediumImportance: AnimatedEntity[] = []
    const lowImportance: AnimatedEntity[] = []
    
    entities.forEach(entity => {
      if (entity.visualProps.size > 40) {
        highImportance.push(entity)
      } else if (entity.visualProps.size > 30) {
        mediumImportance.push(entity)
      } else {
        lowImportance.push(entity)
      }
    })
    
    return [highImportance, mediumImportance, lowImportance].filter(cluster => 
      cluster.length > 0
    )
  }
  
  private calculateClusterCenters(
    clusterCount: number,
    spacing: number
  ): Position[] {
    const centers: Position[] = []
    const cols = Math.ceil(Math.sqrt(clusterCount))
    const rows = Math.ceil(clusterCount / cols)
    
    for (let i = 0; i < clusterCount; i++) {
      const row = Math.floor(i / cols)
      const col = i % cols
      
      centers.push({
        x: (col + 1) * spacing + spacing / 2,
        y: (row + 1) * spacing + spacing / 2
      })
    }
    
    return centers
  }
  
  private calculateBoundingBox(
    positions: Map<string, Position>
  ): BoundingBox {
    const positionArray = Array.from(positions.values())
    
    if (positionArray.length === 0) {
      return { x: 0, y: 0, width: 0, height: 0 }
    }
    
    const minX = Math.min(...positionArray.map(p => p.x))
    const maxX = Math.max(...positionArray.map(p => p.x))
    const minY = Math.min(...positionArray.map(p => p.y))
    const maxY = Math.max(...positionArray.map(p => p.y))
    
    return {
      x: minX - 50,
      y: minY - 50,
      width: maxX - minX + 100,
      height: maxY - minY + 100
    }
  }
  
  private evaluateLayoutQuality(
    positions: Map<string, Position>,
    links: ForceLink[]
  ): number {
    // 计算布局质量指标
    let totalOverlap = 0
    let totalLinkLength = 0
    let optimalLinkLength = 0
    
    const positionArray = Array.from(positions.values())
    const minDistance = 60 // 最小节点间距
    
    // 检查节点重叠
    for (let i = 0; i < positionArray.length; i++) {
      for (let j = i + 1; j < positionArray.length; j++) {
        const dist = this.distance(positionArray[i], positionArray[j])
        if (dist < minDistance) {
          totalOverlap += minDistance - dist
        }
      }
    }
    
    // 检查连接长度
    links.forEach(link => {
      const sourcePos = positions.get(link.source)
      const targetPos = positions.get(link.target)
      
      if (sourcePos && targetPos) {
        const actualLength = this.distance(sourcePos, targetPos)
        totalLinkLength += actualLength
        optimalLinkLength += link.distance
      }
    })
    
    // 计算质量分数 (0-1)
    const overlapPenalty = Math.min(totalOverlap / 1000, 0.5)
    const lengthVariation = Math.abs(totalLinkLength - optimalLinkLength) / 
      Math.max(optimalLinkLength, 1)
    const lengthPenalty = Math.min(lengthVariation / 5, 0.3)
    
    return Math.max(0, 1 - overlapPenalty - lengthPenalty)
  }
  
  private evaluateHierarchicalQuality(levels: string[][]): number {
    // 评估分层布局质量
    const evenness = this.calculateLevelEvenness(levels)
    const separation = this.calculateLevelSeparation(levels)
    
    return (evenness + separation) / 2
  }
  
  private evaluateClusteringQuality(clusters: AnimatedEntity[][]): number {
    // 评估聚类质量
    const balance = this.calculateClusterBalance(clusters)
    const cohesion = this.calculateClusterCohesion(clusters)
    
    return (balance + cohesion) / 2
  }
  
  private calculateLevelEvenness(levels: string[][]): number {
    if (levels.length === 0) return 1
    
    const avgSize = levels.reduce((sum, level) => sum + level.length, 0) / levels.length
    const variance = levels.reduce((sum, level) => 
      sum + Math.pow(level.length - avgSize, 2), 0
    ) / levels.length
    
    return Math.max(0, 1 - variance / (avgSize * avgSize))
  }
  
  private calculateLevelSeparation(levels: string[][]): number {
    // 简化计算，假设层次分离良好
    return 0.8
  }
  
  private calculateClusterBalance(clusters: AnimatedEntity[][]): number {
    if (clusters.length === 0) return 1
    
    const sizes = clusters.map(cluster => cluster.length)
    const avgSize = sizes.reduce((sum, size) => sum + size, 0) / sizes.length
    const variance = sizes.reduce((sum, size) => 
      sum + Math.pow(size - avgSize, 2), 0
    ) / sizes.length
    
    return Math.max(0, 1 - variance / (avgSize * avgSize))
  }
  
  private calculateClusterCohesion(clusters: AnimatedEntity[][]): number {
    // 简化计算，评估聚类内聚性
    return 0.7
  }
  
  private distance(a: Position, b: Position): number {
    const dx = a.x - b.x
    const dy = a.y - b.y
    return Math.sqrt(dx * dx + dy * dy)
  }
  
  public setDimensions(width: number, height: number): void {
    this.width = width
    this.height = height
  }
}