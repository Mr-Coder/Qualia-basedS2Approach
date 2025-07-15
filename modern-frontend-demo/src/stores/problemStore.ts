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
  label: string
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

export interface SolveHistory {
  id: string
  problem: string
  strategy: string
  result: SolveResult
  timestamp: Date
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
        })
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
export const useHistory = () => useProblemStore(state => state.history)