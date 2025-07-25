// 基于物性关系的推理引擎 API
export interface PhysicalEntity {
  id: string
  name: string
  type: 'object' | 'quantity' | 'person' | 'concept'
  properties: {
    value?: number
    unit?: string
    physical_state?: string
    [key: string]: any
  }
}

export interface PhysicalRelation {
  source: string
  target: string
  type: 'has_quantity' | 'conservation' | 'transformation' | 'spatial' | 'temporal' | 'causal'
  properties: {
    strength: number
    reversible?: boolean
    condition?: string
  }
}

export interface ReasoningPath {
  from: string
  to: string
  path: string[]
  relations: PhysicalRelation[]
  confidence: number
  reasoning: string
}

export interface PhysicalGraph {
  entities: PhysicalEntity[]
  relations: PhysicalRelation[]
  known: string[]     // 已知实体
  unknown: string[]   // 未知实体
  paths: ReasoningPath[]  // 推理路径
}

// 物性关系推理器
export class PhysicalReasoner {
  // 通过物性关系扩展图
  static expandGraphWithPhysicalRelations(graph: PhysicalGraph): PhysicalGraph {
    const expandedRelations = [...graph.relations]
    
    // 1. 守恒关系推理
    graph.entities.forEach(entity => {
      if (entity.type === 'quantity') {
        // 如果是可分割的量，添加部分-整体关系
        const relatedEntities = graph.relations
          .filter(r => r.source === entity.id || r.target === entity.id)
          .map(r => r.source === entity.id ? r.target : r.source)
        
        // 添加传递性关系
        relatedEntities.forEach(related => {
          const transitiveRelations = graph.relations
            .filter(r => r.source === related && r.target !== entity.id)
          
          transitiveRelations.forEach(tr => {
            if (!expandedRelations.some(r => 
              r.source === entity.id && r.target === tr.target
            )) {
              expandedRelations.push({
                source: entity.id,
                target: tr.target,
                type: 'causal',
                properties: {
                  strength: 0.7,
                  reasoning: '传递性推理'
                }
              })
            }
          })
        })
      }
    })
    
    // 2. 因果关系推理
    const causalChains = this.findCausalChains(graph)
    causalChains.forEach(chain => {
      expandedRelations.push({
        source: chain[0],
        target: chain[chain.length - 1],
        type: 'causal',
        properties: {
          strength: 0.8,
          reasoning: `因果链: ${chain.join(' → ')}`
        }
      })
    })
    
    return {
      ...graph,
      relations: expandedRelations
    }
  }
  
  // 查找因果链
  static findCausalChains(graph: PhysicalGraph): string[][] {
    const chains: string[][] = []
    
    graph.entities.forEach(start => {
      const visited = new Set<string>()
      const currentChain: string[] = []
      
      const dfs = (current: string) => {
        if (visited.has(current)) return
        visited.add(current)
        currentChain.push(current)
        
        const nextNodes = graph.relations
          .filter(r => r.source === current && r.type === 'causal')
          .map(r => r.target)
        
        if (nextNodes.length === 0 && currentChain.length > 1) {
          chains.push([...currentChain])
        } else {
          nextNodes.forEach(next => dfs(next))
        }
        
        currentChain.pop()
      }
      
      dfs(start.id)
    })
    
    return chains
  }
  
  // 寻找已知到未知的路径
  static findPathsFromKnownToUnknown(graph: PhysicalGraph): ReasoningPath[] {
    const paths: ReasoningPath[] = []
    const expandedGraph = this.expandGraphWithPhysicalRelations(graph)
    
    graph.known.forEach(knownId => {
      graph.unknown.forEach(unknownId => {
        const path = this.findPath(expandedGraph, knownId, unknownId)
        if (path) {
          paths.push(path)
        }
      })
    })
    
    return paths
  }
  
  // 使用 BFS 寻找最短路径
  static findPath(graph: PhysicalGraph, start: string, end: string): ReasoningPath | null {
    const queue: Array<{node: string, path: string[], relations: PhysicalRelation[]}> = [
      {node: start, path: [start], relations: []}
    ]
    const visited = new Set<string>()
    
    while (queue.length > 0) {
      const current = queue.shift()!
      
      if (current.node === end) {
        return {
          from: start,
          to: end,
          path: current.path,
          relations: current.relations,
          confidence: this.calculatePathConfidence(current.relations),
          reasoning: this.generateReasoningExplanation(current.path, current.relations)
        }
      }
      
      if (visited.has(current.node)) continue
      visited.add(current.node)
      
      const nextRelations = graph.relations.filter(r => r.source === current.node)
      
      nextRelations.forEach(rel => {
        queue.push({
          node: rel.target,
          path: [...current.path, rel.target],
          relations: [...current.relations, rel]
        })
      })
    }
    
    return null
  }
  
  // 计算路径置信度
  static calculatePathConfidence(relations: PhysicalRelation[]): number {
    if (relations.length === 0) return 1
    
    const totalStrength = relations.reduce((sum, rel) => sum + rel.properties.strength, 0)
    return totalStrength / relations.length
  }
  
  // 生成推理解释
  static generateReasoningExplanation(path: string[], relations: PhysicalRelation[]): string {
    let explanation = `推理路径: ${path[0]}`
    
    for (let i = 0; i < relations.length; i++) {
      const rel = relations[i]
      explanation += ` --[${rel.type}]--> ${path[i + 1]}`
    }
    
    return explanation
  }
}

// 示例问题的物性图构建
export function buildPhysicalGraphFromProblem(problemText: string): PhysicalGraph {
  // 示例1: 小明有5个苹果，又买了3个，一共有几个？
  if (problemText.includes('苹果')) {
    return {
      entities: [
        { id: 'xiaoming', name: '小明', type: 'person', properties: {} },
        { id: 'apples_initial', name: '初始苹果', type: 'quantity', properties: { value: 5, unit: '个' } },
        { id: 'apples_bought', name: '购买苹果', type: 'quantity', properties: { value: 3, unit: '个' } },
        { id: 'apples_total', name: '总苹果数', type: 'quantity', properties: { value: null, unit: '个' } },
        { id: 'addition', name: '加法运算', type: 'concept', properties: {} }
      ],
      relations: [
        { source: 'xiaoming', target: 'apples_initial', type: 'has_quantity', properties: { strength: 1 } },
        { source: 'xiaoming', target: 'apples_bought', type: 'has_quantity', properties: { strength: 1 } },
        { source: 'apples_initial', target: 'addition', type: 'transformation', properties: { strength: 0.9 } },
        { source: 'apples_bought', target: 'addition', type: 'transformation', properties: { strength: 0.9 } },
        { source: 'addition', target: 'apples_total', type: 'transformation', properties: { strength: 1 } },
        // 物性关系：守恒
        { source: 'apples_initial', target: 'apples_total', type: 'conservation', properties: { strength: 0.8, condition: '部分到整体' } },
        { source: 'apples_bought', target: 'apples_total', type: 'conservation', properties: { strength: 0.8, condition: '部分到整体' } }
      ],
      known: ['apples_initial', 'apples_bought'],
      unknown: ['apples_total'],
      paths: []
    }
  }
  
  // 示例2: 水池有水100升，每小时流出20升，几小时后水池为空？
  if (problemText.includes('水池')) {
    return {
      entities: [
        { id: 'pool', name: '水池', type: 'object', properties: {} },
        { id: 'water_initial', name: '初始水量', type: 'quantity', properties: { value: 100, unit: '升' } },
        { id: 'flow_rate', name: '流出速率', type: 'quantity', properties: { value: 20, unit: '升/小时' } },
        { id: 'time', name: '时间', type: 'quantity', properties: { value: null, unit: '小时' } },
        { id: 'empty_state', name: '空状态', type: 'concept', properties: { value: 0, unit: '升' } }
      ],
      relations: [
        { source: 'pool', target: 'water_initial', type: 'has_quantity', properties: { strength: 1 } },
        { source: 'water_initial', target: 'flow_rate', type: 'causal', properties: { strength: 0.9 } },
        { source: 'flow_rate', target: 'time', type: 'causal', properties: { strength: 0.9 } },
        { source: 'time', target: 'empty_state', type: 'causal', properties: { strength: 1 } },
        // 物性关系：时间流逝
        { source: 'water_initial', target: 'empty_state', type: 'temporal', properties: { strength: 0.8, condition: '随时间减少' } }
      ],
      known: ['water_initial', 'flow_rate'],
      unknown: ['time'],
      paths: []
    }
  }
  
  // 示例3: 购物问题
  if (problemText.includes('笔') && problemText.includes('元')) {
    return {
      entities: [
        { id: 'xiaohong', name: '小红', type: 'person', properties: {} },
        { id: 'pens', name: '笔', type: 'object', properties: {} },
        { id: 'quantity', name: '数量', type: 'quantity', properties: { value: 3, unit: '支' } },
        { id: 'unit_price', name: '单价', type: 'quantity', properties: { value: 5, unit: '元/支' } },
        { id: 'total_price', name: '总价', type: 'quantity', properties: { value: null, unit: '元' } },
        { id: 'multiplication', name: '乘法运算', type: 'concept', properties: {} },
        { id: 'exchange', name: '价值交换', type: 'concept', properties: {} }
      ],
      relations: [
        { source: 'xiaohong', target: 'pens', type: 'has_quantity', properties: { strength: 1 } },
        { source: 'pens', target: 'quantity', type: 'has_quantity', properties: { strength: 1 } },
        { source: 'pens', target: 'unit_price', type: 'has_quantity', properties: { strength: 1 } },
        { source: 'quantity', target: 'multiplication', type: 'transformation', properties: { strength: 0.9 } },
        { source: 'unit_price', target: 'multiplication', type: 'transformation', properties: { strength: 0.9 } },
        { source: 'multiplication', target: 'total_price', type: 'transformation', properties: { strength: 1 } },
        // 物性关系：价值守恒
        { source: 'exchange', target: 'total_price', type: 'conservation', properties: { strength: 0.8, condition: '价值守恒' } },
        { source: 'quantity', target: 'total_price', type: 'causal', properties: { strength: 0.7, condition: '数量影响总价' } },
        { source: 'unit_price', target: 'total_price', type: 'causal', properties: { strength: 0.7, condition: '单价影响总价' } }
      ],
      known: ['quantity', 'unit_price'],
      unknown: ['total_price'],
      paths: []
    }
  }
  
  // 默认示例
  return {
    entities: [],
    relations: [],
    known: [],
    unknown: [],
    paths: []
  }
}