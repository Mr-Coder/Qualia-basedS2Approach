/**
 * 增强QS²关系图组件 (Enhanced QS² Relationship Diagram)
 * 
 * 基于QS²量子语义和IRD隐式关系发现算法的高级数学关系图可视化组件。
 * 支持多维语义空间可视化、隐式关系动画、约束网络展示等高级特性。
 */

import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import * as d3 from 'd3'

// 类型定义
interface QS2Entity {
  id: string
  text: string
  entity_type: string
  confidence: number
  position: [number, number]
  semantic_features: Record<string, number>
  physical_attributes: {
    mass: number
    dimension: number
    constraints: string[]
  }
  qualia_structure?: {
    formal: Record<string, number>
    telic: Record<string, number>
    agentive: Record<string, number>
    constitutive: Record<string, number>
  }
}

interface QS2Relation {
  id: string
  source_id: string
  target_id: string
  relation_type: string
  relation_depth: 'surface' | 'shallow' | 'medium' | 'deep'
  confidence: number
  weight: number
  semantic_compatibility?: number
  qualia_alignment?: Record<string, number>
  discovery_method: string
  is_implicit?: boolean
  discovery_path?: Array<{
    step_id: string
    method: string
    premises: string[]
    conclusion: string
    confidence: number
  }>
}

interface SemanticSpace {
  [entityId: string]: {
    dimensions: Record<string, number>
    magnitude: number
    uncertainty: number
  }
}

interface VisualizationData {
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

interface EnhancedQS2RelationshipDiagramProps {
  // 核心数据
  entities: QS2Entity[]
  relations: QS2Relation[]
  semantic_space: SemanticSpace
  
  // 可视化配置
  visualization_data?: VisualizationData
  width?: number
  height?: number
  
  // 显示选项
  showImplicitRelations?: boolean
  showSemanticHeatmap?: boolean
  showQualiaStructures?: boolean
  showInferenceAnimations?: boolean
  showConstraintNetwork?: boolean
  
  // 交互选项
  enableNodeDragging?: boolean
  enableZoomPan?: boolean
  enableHover?: boolean
  enableSelection?: boolean
  
  // 回调函数
  onNodeClick?: (entity: QS2Entity) => void
  onRelationClick?: (relation: QS2Relation) => void
  onSemanticAnalysis?: (entityId: string) => void
}

const EnhancedQS2RelationshipDiagram: React.FC<EnhancedQS2RelationshipDiagramProps> = ({
  entities,
  relations,
  semantic_space,
  visualization_data,
  width = 800,
  height = 600,
  showImplicitRelations = true,
  showSemanticHeatmap = false,
  showQualiaStructures = false,
  showInferenceAnimations = true,
  showConstraintNetwork = false,
  enableNodeDragging = true,
  enableZoomPan = true,
  enableHover = true,
  enableSelection = true,
  onNodeClick,
  onRelationClick,
  onSemanticAnalysis
}) => {
  // 引用和状态
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [selectedEntity, setSelectedEntity] = useState<string | null>(null)
  const [hoveredEntity, setHoveredEntity] = useState<string | null>(null)
  const [animationPlaying, setAnimationPlaying] = useState(false)
  const [currentAnimationStep, setCurrentAnimationStep] = useState(0)
  const [semanticAnalysisMode, setSemanticAnalysisMode] = useState(false)
  
  // 模拟状态
  const [simulation, setSimulation] = useState<d3.Simulation<any, any> | null>(null)
  const [transform, setTransform] = useState(d3.zoomIdentity)
  
  // 处理后的数据
  const processedData = useMemo(() => {
    // 合并可视化数据和原始数据
    const nodes = visualization_data?.nodes || entities.map(entity => ({
      id: entity.id,
      label: entity.text,
      type: entity.entity_type,
      size: 15,
      color: getNodeColor(entity.entity_type),
      position: entity.position,
      properties: {
        confidence: entity.confidence,
        semantic_features: entity.semantic_features,
        qualia_structure: entity.qualia_structure
      }
    }))
    
    const edges = visualization_data?.edges || relations.map(relation => ({
      id: relation.id,
      source: relation.source_id,
      target: relation.target_id,
      type: relation.relation_type,
      weight: relation.weight,
      confidence: relation.confidence,
      style: {
        width: relation.weight * 2,
        color: getEdgeColor(relation.relation_type),
        opacity: relation.confidence,
        pattern: relation.is_implicit ? 'dashed' as const : 'solid' as const
      },
      properties: {
        relation_depth: relation.relation_depth,
        discovery_method: relation.discovery_method,
        is_implicit: relation.is_implicit,
        semantic_compatibility: relation.semantic_compatibility
      }
    }))
    
    return { nodes, edges }
  }, [entities, relations, visualization_data])
  
  // 初始化D3力导向图
  useEffect(() => {
    if (!svgRef.current || !processedData.nodes.length) return
    
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()
    
    // 创建主要的g元素
    const mainGroup = svg.append('g').attr('class', 'main-group')
    
    // 设置缩放
    if (enableZoomPan) {
      const zoom = d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 4])
        .on('zoom', (event) => {
          setTransform(event.transform)
          mainGroup.attr('transform', event.transform)
        })
      
      svg.call(zoom)
    }
    
    // 创建力模拟
    const sim = d3.forceSimulation(processedData.nodes)
      .force('link', d3.forceLink(processedData.edges).id((d: any) => d.id).strength(0.5))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius((d: any) => d.size + 5))
    
    setSimulation(sim)
    
    // 绘制语义热力图背景（如果启用）
    if (showSemanticHeatmap && visualization_data?.semantic_heatmap) {
      drawSemanticHeatmap(mainGroup, visualization_data.semantic_heatmap)
    }
    
    // 绘制边
    const linkGroup = mainGroup.append('g').attr('class', 'links')
    drawEdges(linkGroup, processedData.edges)
    
    // 绘制节点
    const nodeGroup = mainGroup.append('g').attr('class', 'nodes')
    drawNodes(nodeGroup, processedData.nodes, sim)
    
    // 绘制Qualia结构（如果启用）
    if (showQualiaStructures) {
      const qualiaGroup = mainGroup.append('g').attr('class', 'qualia-structures')
      drawQualiaStructures(qualiaGroup, processedData.nodes)
    }
    
    // 启动推理动画（如果启用）
    if (showInferenceAnimations && visualization_data?.animation_timeline) {
      startInferenceAnimations(mainGroup, visualization_data.animation_timeline)
    }
    
    return () => {
      sim.stop()
    }
  }, [processedData, width, height, showSemanticHeatmap, showQualiaStructures, showInferenceAnimations])
  
  // 绘制语义热力图
  const drawSemanticHeatmap = useCallback((container: d3.Selection<SVGGElement, unknown, null, undefined>, 
                                         heatmapData: Record<string, Record<string, number>>) => {
    const entityIds = Object.keys(heatmapData)
    const cellSize = Math.min(width, height) / (entityIds.length * 2)
    
    const heatmapGroup = container.append('g')
      .attr('class', 'semantic-heatmap')
      .attr('opacity', 0.3)
    
    entityIds.forEach((entityId1, i) => {
      entityIds.forEach((entityId2, j) => {
        const similarity = heatmapData[entityId1]?.[entityId2] || 0
        
        heatmapGroup.append('rect')
          .attr('x', i * cellSize)
          .attr('y', j * cellSize)
          .attr('width', cellSize)
          .attr('height', cellSize)
          .attr('fill', d3.interpolateRdYlBu(1 - similarity))
          .attr('opacity', similarity)
          .on('mouseover', () => {
            // 显示相似度信息
            showTooltip(`语义相似度: ${similarity.toFixed(3)}`)
          })
      })
    })
  }, [width, height])
  
  // 绘制边
  const drawEdges = useCallback((container: d3.Selection<SVGGElement, unknown, null, undefined>, 
                               edges: any[]) => {
    const links = container.selectAll('.link')
      .data(edges)
      .enter()
      .append('g')
      .attr('class', 'link')
    
    // 主要连线
    links.append('line')
      .attr('class', 'link-line')
      .attr('stroke', d => d.style.color)
      .attr('stroke-width', d => d.style.width)
      .attr('stroke-opacity', d => d.style.opacity)
      .attr('stroke-dasharray', d => d.style.pattern === 'dashed' ? '5,5' : 'none')
      .on('click', (event, d) => {
        event.stopPropagation()
        if (onRelationClick) {
          const relation = relations.find(r => r.id === d.id)
          if (relation) onRelationClick(relation)
        }
      })
      .on('mouseover', (event, d) => {
        if (enableHover) {
          showRelationTooltip(event, d)
        }
      })
    
    // 隐式关系特殊效果
    links.filter(d => d.properties.is_implicit)
      .append('line')
      .attr('class', 'implicit-glow')
      .attr('stroke', d => d.style.color)
      .attr('stroke-width', d => d.style.width + 2)
      .attr('stroke-opacity', 0.3)
      .attr('filter', 'url(#glow)')
    
    // 语义兼容性指示器
    links.filter(d => d.properties.semantic_compatibility > 0.7)
      .append('circle')
      .attr('class', 'semantic-indicator')
      .attr('r', 3)
      .attr('fill', '#4ECDC4')
      .attr('opacity', 0.8)
    
    // 更新位置
    if (simulation) {
      simulation.on('tick', () => {
        links.selectAll('line')
          .attr('x1', (d: any) => d.source.x)
          .attr('y1', (d: any) => d.source.y)
          .attr('x2', (d: any) => d.target.x)
          .attr('y2', (d: any) => d.target.y)
        
        links.selectAll('.semantic-indicator')
          .attr('cx', (d: any) => (d.source.x + d.target.x) / 2)
          .attr('cy', (d: any) => (d.source.y + d.target.y) / 2)
      })
    }
  }, [simulation, relations, onRelationClick, enableHover])
  
  // 绘制节点
  const drawNodes = useCallback((container: d3.Selection<SVGGElement, unknown, null, undefined>, 
                               nodes: any[], sim: d3.Simulation<any, any>) => {
    const nodeGroups = container.selectAll('.node')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', 'node')
      .style('cursor', 'pointer')
    
    // 主要节点圆圈
    nodeGroups.append('circle')
      .attr('class', 'node-circle')
      .attr('r', d => d.size)
      .attr('fill', d => d.color)
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .attr('opacity', d => d.properties.confidence)
    
    // 置信度环
    nodeGroups.append('circle')
      .attr('class', 'confidence-ring')
      .attr('r', d => d.size + 3)
      .attr('fill', 'none')
      .attr('stroke', d => d.color)
      .attr('stroke-width', 2)
      .attr('stroke-opacity', d => d.properties.confidence * 0.5)
      .attr('stroke-dasharray', d => `${d.properties.confidence * 20}, ${20 - d.properties.confidence * 20}`)
    
    // 节点标签
    nodeGroups.append('text')
      .attr('class', 'node-label')
      .attr('text-anchor', 'middle')
      .attr('dy', d => d.size + 15)
      .attr('font-size', '12px')
      .attr('font-weight', 'bold')
      .attr('fill', '#333')
      .text(d => d.label)
    
    // QS²语义维度指示器
    nodeGroups.filter(d => d.properties.qualia_structure)
      .each(function(d) {
        const group = d3.select(this)
        const qualia = d.properties.qualia_structure
        const angles = [0, Math.PI/2, Math.PI, 3*Math.PI/2] // 四个方向
        const qualiaTypes = ['formal', 'telic', 'agentive', 'constitutive']
        
        qualiaTypes.forEach((type, i) => {
          const value = Object.values(qualia[type] || {}).reduce((sum: number, v: any) => sum + (typeof v === 'number' ? v : 0), 0)
          const angle = angles[i]
          const radius = d.size + 8 + value * 10
          
          group.append('circle')
            .attr('class', `qualia-${type}`)
            .attr('cx', Math.cos(angle) * radius)
            .attr('cy', Math.sin(angle) * radius)
            .attr('r', 2 + value * 3)
            .attr('fill', getQualiaColor(type))
            .attr('opacity', 0.7)
        })
      })
    
    // 交互事件
    nodeGroups
      .on('click', (event, d) => {
        event.stopPropagation()
        setSelectedEntity(d.id)
        if (onNodeClick) {
          const entity = entities.find(e => e.id === d.id)
          if (entity) onNodeClick(entity)
        }
      })
      .on('mouseover', (event, d) => {
        if (enableHover) {
          setHoveredEntity(d.id)
          showNodeTooltip(event, d)
        }
      })
      .on('mouseout', () => {
        setHoveredEntity(null)
        hideTooltip()
      })
      .on('dblclick', (event, d) => {
        if (onSemanticAnalysis) {
          onSemanticAnalysis(d.id)
        }
      })
    
    // 拖拽行为
    if (enableNodeDragging) {
      const drag = d3.drag<SVGGElement, any>()
        .on('start', (event, d) => {
          if (!event.active) sim.alphaTarget(0.3).restart()
          d.fx = d.x
          d.fy = d.y
        })
        .on('drag', (event, d) => {
          d.fx = event.x
          d.fy = event.y
        })
        .on('end', (event, d) => {
          if (!event.active) sim.alphaTarget(0)
          d.fx = null
          d.fy = null
        })
      
      nodeGroups.call(drag)
    }
    
    // 更新位置
    sim.on('tick', () => {
      nodeGroups.attr('transform', d => `translate(${d.x},${d.y})`)
    })
  }, [entities, enableHover, enableNodeDragging, onNodeClick, onSemanticAnalysis])
  
  // 绘制Qualia结构
  const drawQualiaStructures = useCallback((container: d3.Selection<SVGGElement, unknown, null, undefined>, 
                                          nodes: any[]) => {
    const qualiaNodes = nodes.filter(d => d.properties.qualia_structure)
    
    qualiaNodes.forEach(node => {
      const group = container.append('g')
        .attr('class', 'qualia-structure')
        .attr('transform', `translate(${node.x || 0}, ${node.y || 0})`)
      
      const qualia = node.properties.qualia_structure
      const qualiaTypes = ['formal', 'telic', 'agentive', 'constitutive']
      
      // 绘制Qualia轮廓
      const radius = node.size + 20
      const points = qualiaTypes.map((type, i) => {
        const angle = (i * 2 * Math.PI) / qualiaTypes.length
        const value = Object.values(qualia[type] || {}).reduce((sum: number, v: any) => sum + (typeof v === 'number' ? v : 0), 0)
        const r = radius + value * 15
        return [Math.cos(angle) * r, Math.sin(angle) * r]
      })
      
      group.append('polygon')
        .attr('class', 'qualia-outline')
        .attr('points', points.map(p => p.join(',')).join(' '))
        .attr('fill', node.color)
        .attr('fill-opacity', 0.1)
        .attr('stroke', node.color)
        .attr('stroke-width', 1)
        .attr('stroke-opacity', 0.3)
    })
  }, [])
  
  // 启动推理动画
  const startInferenceAnimations = useCallback((container: d3.Selection<SVGGElement, unknown, null, undefined>, 
                                              timeline: any[]) => {
    if (!timeline.length) return
    
    setAnimationPlaying(true)
    let currentStep = 0
    
    const playNextAnimation = () => {
      if (currentStep >= timeline.length) {
        setAnimationPlaying(false)
        setCurrentAnimationStep(0)
        return
      }
      
      const animation = timeline[currentStep]
      setCurrentAnimationStep(currentStep)
      
      // 高亮推理路径
      if (animation.type === 'causal_flow') {
        highlightCausalChain(container, animation.nodes, animation.duration)
      }
      
      setTimeout(() => {
        currentStep++
        playNextAnimation()
      }, animation.duration + animation.delay)
    }
    
    playNextAnimation()
  }, [])
  
  // 高亮因果链
  const highlightCausalChain = useCallback((container: d3.Selection<SVGGElement, unknown, null, undefined>, 
                                          nodeIds: string[], duration: number) => {
    // 创建动画路径
    const path = container.append('path')
      .attr('class', 'causal-flow')
      .attr('fill', 'none')
      .attr('stroke', '#FF6B6B')
      .attr('stroke-width', 3)
      .attr('stroke-opacity', 0.8)
      .attr('marker-end', 'url(#arrowhead)')
    
    // 计算路径坐标
    const pathData = nodeIds.map((nodeId, i) => {
      const node = processedData.nodes.find(n => n.id === nodeId)
      if (!node) return null
      
      const command = i === 0 ? 'M' : 'L'
      return `${command} ${node.position[0]} ${node.position[1]}`
    }).filter(Boolean).join(' ')
    
    path.attr('d', pathData)
    
    // 动画效果
    const pathLength = (path.node() as SVGPathElement)?.getTotalLength() || 0
    
    path
      .attr('stroke-dasharray', `0 ${pathLength}`)
      .transition()
      .duration(duration)
      .attr('stroke-dasharray', `${pathLength} 0`)
      .on('end', () => {
        path.remove()
      })
  }, [processedData.nodes])
  
  // 工具提示
  const [tooltip, setTooltip] = useState<{
    visible: boolean
    x: number
    y: number
    content: string
  }>({ visible: false, x: 0, y: 0, content: '' })
  
  const showTooltip = useCallback((content: string, event?: MouseEvent) => {
    setTooltip({
      visible: true,
      x: event?.clientX || 0,
      y: event?.clientY || 0,
      content
    })
  }, [])
  
  const showNodeTooltip = useCallback((event: any, node: any) => {
    const content = `
      <div>
        <strong>${node.label}</strong><br/>
        类型: ${node.type}<br/>
        置信度: ${(node.properties.confidence * 100).toFixed(1)}%<br/>
        ${node.properties.qualia_structure ? '包含Qualia结构' : ''}
      </div>
    `
    showTooltip(content, event)
  }, [showTooltip])
  
  const showRelationTooltip = useCallback((event: any, edge: any) => {
    const content = `
      <div>
        <strong>${edge.type}</strong><br/>
        深度: ${edge.properties.relation_depth}<br/>
        置信度: ${(edge.confidence * 100).toFixed(1)}%<br/>
        发现方法: ${edge.properties.discovery_method}<br/>
        ${edge.properties.is_implicit ? '隐式关系' : '显式关系'}
      </div>
    `
    showTooltip(content, event)
  }, [showTooltip])
  
  const hideTooltip = useCallback(() => {
    setTooltip(prev => ({ ...prev, visible: false }))
  }, [])
  
  // 辅助函数
  const getNodeColor = (entityType: string): string => {
    const colors: Record<string, string> = {
      'number': '#FF6B6B',
      'object': '#4ECDC4',
      'action': '#45B7D1',
      'semantic': '#96CEB4',
      'context': '#FFEAA7'
    }
    return colors[entityType] || '#CCCCCC'
  }
  
  const getEdgeColor = (relationType: string): string => {
    const colors: Record<string, string> = {
      'arithmetic': '#FF9999',
      'semantic_similarity': '#99CCFF',
      'constraint_based': '#99FF99',
      'causal': '#FFCC99',
      'temporal': '#CC99FF'
    }
    return colors[relationType] || '#999999'
  }
  
  const getQualiaColor = (qualiaType: string): string => {
    const colors: Record<string, string> = {
      'formal': '#FF6B6B',
      'telic': '#4ECDC4',
      'agentive': '#45B7D1',
      'constitutive': '#96CEB4'
    }
    return colors[qualiaType] || '#CCCCCC'
  }
  
  return (
    <Card className=\"w-full h-full\">
      <CardHeader>
        <CardTitle className=\"flex items-center justify-between\">
          <span>增强QS²数学关系图</span>
          <div className=\"flex gap-2\">
            <button
              onClick={() => setSemanticAnalysisMode(!semanticAnalysisMode)}
              className={`px-3 py-1 rounded text-sm ${
                semanticAnalysisMode 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-200 text-gray-700'
              }`}
            >
              语义分析
            </button>
            <button
              onClick={() => setAnimationPlaying(!animationPlaying)}
              className={`px-3 py-1 rounded text-sm ${
                animationPlaying 
                  ? 'bg-red-500 text-white' 
                  : 'bg-green-500 text-white'
              }`}
            >
              {animationPlaying ? '停止动画' : '播放动画'}
            </button>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className=\"p-0\">
        <div 
          ref={containerRef}
          className=\"relative w-full h-full\"
          style={{ minHeight: height }}
        >
          <svg
            ref={svgRef}
            width={width}
            height={height}
            className=\"w-full h-full\"
          >
            {/* 定义滤镜和标记 */}
            <defs>
              <filter id=\"glow\">
                <feGaussianBlur stdDeviation=\"3\" result=\"coloredBlur\"/>
                <feMerge> 
                  <feMergeNode in=\"coloredBlur\"/>
                  <feMergeNode in=\"SourceGraphic\"/>
                </feMerge>
              </filter>
              
              <marker
                id=\"arrowhead\"
                markerWidth=\"10\"
                markerHeight=\"7\"
                refX=\"9\"
                refY=\"3.5\"
                orient=\"auto\"
              >
                <polygon
                  points=\"0 0, 10 3.5, 0 7\"
                  fill=\"#FF6B6B\"
                />
              </marker>
            </defs>
          </svg>
          
          {/* 工具提示 */}
          <AnimatePresence>
            {tooltip.visible && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className=\"absolute bg-black text-white p-2 rounded shadow-lg pointer-events-none z-10\"
                style={{
                  left: tooltip.x + 10,
                  top: tooltip.y - 10
                }}
                dangerouslySetInnerHTML={{ __html: tooltip.content }}
              />
            )}
          </AnimatePresence>
          
          {/* 状态面板 */}
          {semanticAnalysisMode && (
            <motion.div
              initial={{ opacity: 0, x: -100 }}
              animate={{ opacity: 1, x: 0 }}
              className=\"absolute top-4 left-4 bg-white p-4 rounded shadow-lg\"
            >
              <h3 className=\"font-bold mb-2\">语义空间分析</h3>
              <div className=\"space-y-2 text-sm\">
                <div>实体数量: {entities.length}</div>
                <div>关系数量: {relations.length}</div>
                <div>隐式关系: {relations.filter(r => r.is_implicit).length}</div>
                <div>语义维度: {Object.keys(semantic_space).length}</div>
              </div>
            </motion.div>
          )}
          
          {/* 动画状态 */}
          {animationPlaying && (
            <motion.div
              initial={{ opacity: 0, y: -50 }}
              animate={{ opacity: 1, y: 0 }}
              className=\"absolute top-4 right-4 bg-blue-500 text-white p-3 rounded shadow-lg\"
            >
              <div className=\"text-sm font-bold\">推理动画</div>
              <div className=\"text-xs\">步骤 {currentAnimationStep + 1}</div>
            </motion.div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

export default EnhancedQS2RelationshipDiagram"