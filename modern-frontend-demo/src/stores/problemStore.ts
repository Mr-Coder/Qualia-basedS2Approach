import { create } from 'zustand'
import { devtools, persist } from 'zustand/middleware'

export interface Entity {
  id: string
  name: string
  type: 'person' | 'object' | 'money' | 'concept'
}

export interface Relationship {
  source: string
  target: string
  type: string
  weight?: number
}

// 增强引擎的扩展信息
export interface EnhancedInfo {
  algorithm: string
  relationsFound: number
  semanticDepth: number
  processingMethod: string
}

export interface SolveResult {
  answer: string
  confidence: number
  strategy: string
  steps: string[]
  entities: Entity[]
  relationships: Relationship[]
  constraints: string[]
  processingTime?: number
  enhancedInfo?: EnhancedInfo
  physicalConstraints?: string[]  // 新增：物性约束
  physicalProperties?: {          // 新增：物性属性
    conservationLaws: string[]
    spatialRelations: string[]
    temporalConstraints: string[]
    materialProperties: string[]
  }
  // 新增：深度隐含关系增强数据
  deepRelations?: DeepRelation[]
  implicitConstraints?: ImplicitConstraint[]
  reasoningLayers?: ReasoningLayer[]
  visualizationConfig?: VisualizationConfig
  
  // 新增：物性图谱相关数据
  physicalGraph?: PhysicalGraph
  physicalAnalysis?: PhysicalAnalysis
}

// 新增：深度关系接口
export interface DeepRelation {
  id: string
  source: string
  target: string
  type: string
  depth: 'surface' | 'shallow' | 'medium' | 'deep'
  confidence: number
  label: string
  evidence: string[]
  constraints: string[]
  visualization: {
    depth_color: string
    confidence_size: number
    relation_width: number
    animation_delay: number
    hover_info: {
      title: string
      details: string[]
      constraints: string[]
    }
  }
}

// 新增：隐含约束接口
export interface ImplicitConstraint {
  id: string
  type: string
  description: string
  entities: string[]
  expression: string
  confidence: number
  icon: string
  color: string
  visualization: {
    constraint_priority: number
    visualization_layer: string
    animation_type: string
    detail_panel: {
      title: string
      expression: string
      method: string
      entities: string[]
    }
  }
}

// 新增：推理层级接口
export interface ReasoningLayer {
  [key: string]: Array<{
    step_id: number
    description: string
    confidence: number
    metadata: any
  }>
}

// 新增：可视化配置接口
export interface VisualizationConfig {
  show_depth_indicators: boolean
  show_constraint_panels: boolean
  enable_interactive_exploration: boolean
  animation_sequence: boolean
}

// 新增：物性图谱接口
export interface PhysicalGraph {
  entities: Entity[]
  properties: PhysicalProperty[]
  constraints: PhysicalConstraint[]
  relations: PhysicalRelation[]
  graph_metrics: Record<string, number>
  consistency_score: number
}

// 新增：物理属性接口
export interface PhysicalProperty {
  id: string
  type: 'conservation' | 'discreteness' | 'continuity' | 'additivity' | 'measurability' | 'locality' | 'temporality' | 'causality'
  entity: string
  value: any
  unit: string
  certainty: number
  constraints: string[]
}

// 新增：物理约束接口
export interface PhysicalConstraint {
  id: string
  type: 'conservation_law' | 'non_negative' | 'integer_constraint' | 'upper_bound' | 'lower_bound' | 'equivalence' | 'ordering' | 'exclusivity'
  description: string
  expression: string
  strength: number
  entities: string[]
}

// 新增：物理关系接口
export interface PhysicalRelation {
  id: string
  source: string
  target: string
  type: string
  physical_basis: string
  strength: number
  causal_direction?: string
}

// 新增：物性分析接口
export interface PhysicalAnalysis {
  problem: string
  physical_properties: PhysicalProperty[]
  physical_constraints: PhysicalConstraint[]
  physical_relations: PhysicalRelation[]
  graph_metrics: Record<string, number>
  consistency_score: number
  backend_driven_features?: Record<string, string>
  frontend_optimization?: Record<string, string>
}

export interface SolveHistory {
  id: string
  problem: string
  answer: string
  strategy: string
  timestamp: Date
  confidence: number
}

export type Strategy = 'auto' | 'cot' | 'got' | 'tot'

interface ProblemState {
  // 当前状态
  currentProblem: string
  selectedStrategy: Strategy
  solveResult: SolveResult | null
  history: SolveHistory[]
  isLoading: boolean
  error: string | null
}

interface ProblemActions {
  // 基本操作
  setProblem: (problem: string) => void
  setStrategy: (strategy: Strategy) => void
  setSolveResult: (result: SolveResult) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
  
  // 历史管理
  addToHistory: (entry: SolveHistory) => void
  removeFromHistory: (id: string) => void
  clearHistory: () => void
  
  // 实用方法
  reset: () => void
  clearAllCache: () => void
  getHistoryByStrategy: (strategy: Strategy) => SolveHistory[]
}

type ProblemStore = ProblemState & ProblemActions

export const useProblemStore = create<ProblemStore>()(
  devtools(
    persist(
      (set, get) => ({
        // 初始状态
        currentProblem: '',
        selectedStrategy: 'auto',
        solveResult: null,
        history: [],
        isLoading: false,
        error: null,

        // 基本操作
        setProblem: (problem) => set({ currentProblem: problem }),
        setStrategy: (strategy) => set({ selectedStrategy: strategy }),
        setSolveResult: (result) => set({ solveResult: result }),
        setLoading: (loading) => set({ isLoading: loading }),
        setError: (error) => set({ error }),

        // 历史管理
        addToHistory: (entry) => set((state) => ({
          history: [entry, ...state.history].slice(0, 100) // 保留最近100条
        })),
        
        removeFromHistory: (id) => set((state) => ({
          history: state.history.filter(item => item.id !== id)
        })),
        
        clearHistory: () => set({ history: [] }),

        // 实用方法
        reset: () => set({
          currentProblem: '',
          selectedStrategy: 'auto',
          solveResult: null,
          isLoading: false,
          error: null
        }),

        // 清除所有缓存和存储
        clearAllCache: () => {
          // 清除localStorage
          localStorage.removeItem('problem-store')
          // 清除sessionStorage
          sessionStorage.clear()
          // 重置状态
          set({
            currentProblem: '',
            selectedStrategy: 'auto',
            solveResult: null,
            history: [],
            isLoading: false,
            error: null
          })
        },

        getHistoryByStrategy: (strategy) => {
          const state = get()
          return state.history.filter(item => item.strategy === strategy)
        }
      }),
      {
        name: 'problem-store',
        partialize: (state) => ({
          history: state.history,
          selectedStrategy: state.selectedStrategy
        }),
        // 添加数据验证
        onRehydrateStorage: () => (state) => {
          if (state) {
            // 确保 history 是数组
            if (!Array.isArray(state.history)) {
              state.history = []
            }
            // 验证每个历史记录项
            state.history = state.history.filter(item => 
              item && typeof item === 'object' && item.id && item.problem
            )
          }
        }
      }
    ),
    {
      name: 'problem-store'
    }
  )
)

// 选择器
export const useCurrentProblem = () => useProblemStore(state => state.currentProblem)
export const useSelectedStrategy = () => useProblemStore(state => state.selectedStrategy)
export const useSolveResult = () => useProblemStore(state => state.solveResult)
export const useIsLoading = () => useProblemStore(state => state.isLoading)
export const useError = () => useProblemStore(state => state.error)
export const useHistory = () => useProblemStore(state => state.history || [])