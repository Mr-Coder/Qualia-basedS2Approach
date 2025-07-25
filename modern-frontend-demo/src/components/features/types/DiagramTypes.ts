/**
 * 增强版实体关系图类型定义
 * 支持推理路径动画、智能布局、协作学习
 */

// ====== 核心数据类型 ======

export interface ReasoningStep {
  id: string
  sequence: number
  description: string
  inputEntities: string[]
  outputEntities: string[]
  operation: 'combine' | 'transform' | 'derive' | 'validate' | 'assume'
  confidence: number // 0-1
  timestamp: number
  duration: number // 动画持续时间(ms)
  metadata: {
    mathOperation?: string
    logicType?: 'deductive' | 'inductive' | 'abductive'
    difficulty?: 'easy' | 'medium' | 'hard'
  }
}

export interface ReasoningPath {
  id: string
  steps: ReasoningStep[]
  totalConfidence: number
  pathType: 'main' | 'alternative' | 'verification'
  startTime: number
  endTime: number
  isOptimal: boolean
  visualization: {
    color: string
    width: number
    style: 'solid' | 'dashed' | 'dotted'
    animationType: 'flow' | 'pulse' | 'glow' | 'trace'
  }
}

export interface AnimatedEntity extends Entity {
  position: Position
  targetPosition?: Position
  animationState: 'idle' | 'moving' | 'highlighted' | 'processing'
  appearanceTime: number
  lifecycle: {
    created: number
    modified: number[]
    accessed: number[]
  }
  visualProps: {
    size: number
    opacity: number
    borderWidth: number
    glowIntensity: number
  }
}

export interface AnimatedRelationship extends Relationship {
  startEntity: string
  endEntity: string
  animationState: 'idle' | 'active' | 'flowing' | 'pulsing'
  flowDirection: 'forward' | 'backward' | 'bidirectional'
  activationTime: number
  strength: number // 关系强度 0-1
  visualProps: {
    thickness: number
    opacity: number
    curvature: number
    dashPattern?: number[]
  }
}

// ====== 布局系统类型 ======

export interface LayoutConfig {
  type: 'force' | 'hierarchical' | 'circular' | 'timeline' | 'clustered'
  params: {
    // 力导向布局参数
    forceStrength?: number
    linkDistance?: number
    centerForce?: number
    
    // 分层布局参数
    levelSeparation?: number
    nodeSeparation?: number
    direction?: 'top-down' | 'left-right' | 'radial'
    
    // 聚类布局参数
    clusterMethod?: 'type' | 'time' | 'importance' | 'custom'
    clusterSpacing?: number
    
    // 时间轴布局参数
    timelineOrientation?: 'horizontal' | 'vertical'
    timeScale?: number
  }
}

export interface LayoutResult {
  entityPositions: Map<string, Position>
  relationshipPaths: Map<string, Position[]>
  boundingBox: BoundingBox
  quality: number // 布局质量评分 0-1
  computationTime: number
  warnings: string[]
}

// ====== 协作学习类型 ======

export interface CollaborationUser {
  id: string
  name: string
  role: 'teacher' | 'student' | 'observer'
  avatar?: string
  color: string
  cursor?: Position
  isActive: boolean
  joinTime: number
}

export interface CollaborationSession {
  id: string
  title: string
  users: CollaborationUser[]
  createdBy: string
  createdAt: number
  status: 'active' | 'paused' | 'completed'
  permissions: {
    canEdit: string[] // user IDs
    canComment: string[]
    canView: string[]
  }
}

export interface CollaborationEvent {
  id: string
  type: 'user_join' | 'user_leave' | 'entity_add' | 'entity_move' | 'comment_add' | 'step_highlight'
  userId: string
  timestamp: number
  data: any
  isUndoable: boolean
}

export interface Comment {
  id: string
  authorId: string
  content: string
  position: Position
  targetId?: string // entity/relationship ID
  timestamp: number
  isResolved: boolean
  replies: Comment[]
}

// ====== 主组件接口 ======

export interface EnhancedDiagramProps {
  // 基础数据
  entities: Entity[]
  relationships: Relationship[]
  reasoningPaths?: ReasoningPath[]
  
  // 布局配置
  layoutConfig: LayoutConfig
  autoOptimizeLayout?: boolean
  
  // 动画配置
  animationConfig: {
    enablePathAnimation: boolean
    enableEntityAnimation: boolean
    animationSpeed: number // 0.5-2.0
    simultaneousSteps: number // 同时播放的推理步骤数
    pauseBetweenSteps: number // 步骤间暂停时间(ms)
  }
  
  // 协作配置
  collaborationConfig?: {
    enabled: boolean
    sessionId?: string
    currentUserId?: string
    realTimeSync: boolean
  }
  
  // 交互配置
  interactionConfig: {
    enableDrag: boolean
    enableZoom: boolean
    enableSelection: boolean
    enableComments: boolean
  }
  
  // 尺寸配置
  width?: number
  height?: number
  
  // 事件回调
  onReasoningStepClick?: (step: ReasoningStep) => void
  onEntitySelect?: (entity: AnimatedEntity) => void
  onLayoutChange?: (layout: LayoutResult) => void
  onCollaborationEvent?: (event: CollaborationEvent) => void
}

// ====== 输出接口 ======

export interface DiagramState {
  entities: Map<string, AnimatedEntity>
  relationships: Map<string, AnimatedRelationship>
  currentLayout: LayoutResult
  animationQueue: AnimationFrame[]
  collaborationState: CollaborationSession | null
  isPlaying: boolean
  currentStep: number
  totalSteps: number
}

export interface AnimationFrame {
  timestamp: number
  entityUpdates: Map<string, Partial<AnimatedEntity>>
  relationshipUpdates: Map<string, Partial<AnimatedRelationship>>
  cameraPosition?: { x: number, y: number, zoom: number }
  highlights: string[] // entity/relationship IDs
}

// ====== 工具类型 ======

export interface Position {
  x: number
  y: number
}

export interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
}

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

// ====== 错误类型 ======

export class DiagramError extends Error {
  constructor(
    message: string,
    public code: string,
    public details?: any
  ) {
    super(message)
    this.name = 'DiagramError'
  }
}

export interface ErrorInfo {
  code: string
  message: string
  timestamp: number
  context?: any
  severity: 'low' | 'medium' | 'high' | 'critical'
}