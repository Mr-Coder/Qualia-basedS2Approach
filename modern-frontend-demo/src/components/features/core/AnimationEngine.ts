/**
 * 动画引擎 - 处理推理路径动画和实体转换
 */

import {
  AnimationFrame,
  AnimatedEntity,
  AnimatedRelationship,
  ReasoningStep,
  ReasoningPath,
  Position
} from '../types/DiagramTypes'

interface Tween {
  id: string
  startTime: number
  duration: number
  easing: EasingFunction
  from: any
  to: any
  onUpdate: (value: any) => void
  onComplete?: () => void
}

type EasingFunction = (t: number) => number

export class AnimationEngine {
  private activeTweens: Map<string, Tween> = new Map()
  private animationId: number | null = null
  private isRunning: boolean = false
  
  // ====== 缓动函数 ======
  
  private readonly easings: Record<string, EasingFunction> = {
    linear: (t: number) => t,
    
    easeInQuad: (t: number) => t * t,
    easeOutQuad: (t: number) => t * (2 - t),
    easeInOutQuad: (t: number) => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
    
    easeInCubic: (t: number) => t * t * t,
    easeOutCubic: (t: number) => (--t) * t * t + 1,
    easeInOutCubic: (t: number) => 
      t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1,
    
    easeInElastic: (t: number) => 
      t === 0 ? 0 : t === 1 ? 1 : -Math.pow(2, 10 * (t - 1)) * Math.sin((t - 1.1) * 5 * Math.PI),
    
    easeOutElastic: (t: number) => 
      t === 0 ? 0 : t === 1 ? 1 : Math.pow(2, -10 * t) * Math.sin((t - 0.1) * 5 * Math.PI) + 1,
    
    easeInBounce: (t: number) => 1 - this.easings.easeOutBounce(1 - t),
    easeOutBounce: (t: number) => {
      if (t < 1 / 2.75) {
        return 7.5625 * t * t
      } else if (t < 2 / 2.75) {
        return 7.5625 * (t -= 1.5 / 2.75) * t + 0.75
      } else if (t < 2.5 / 2.75) {
        return 7.5625 * (t -= 2.25 / 2.75) * t + 0.9375
      } else {
        return 7.5625 * (t -= 2.625 / 2.75) * t + 0.984375
      }
    }
  }
  
  // ====== 动画控制 ======
  
  public startAnimation(): void {
    if (this.isRunning) return
    
    this.isRunning = true
    this.animationLoop()
  }
  
  public stopAnimation(): void {
    this.isRunning = false
    if (this.animationId) {
      cancelAnimationFrame(this.animationId)
      this.animationId = null
    }
  }
  
  public pauseAnimation(): void {
    this.isRunning = false
  }
  
  public resumeAnimation(): void {
    if (!this.isRunning) {
      this.isRunning = true
      this.animationLoop()
    }
  }
  
  private animationLoop = (): void => {
    const currentTime = performance.now()
    
    // 更新所有活动的补间动画
    this.activeTweens.forEach((tween, id) => {
      const elapsed = currentTime - tween.startTime
      const progress = Math.min(elapsed / tween.duration, 1)
      const easedProgress = tween.easing(progress)
      
      // 插值计算
      const currentValue = this.interpolate(tween.from, tween.to, easedProgress)
      tween.onUpdate(currentValue)
      
      // 检查动画完成
      if (progress >= 1) {
        tween.onComplete?.()
        this.activeTweens.delete(id)
      }
    })
    
    // 继续循环
    if (this.isRunning) {
      this.animationId = requestAnimationFrame(this.animationLoop)
    }
  }
  
  // ====== 实体动画 ======
  
  public animateEntityMovement(
    entity: AnimatedEntity,
    targetPosition: Position,
    duration: number = 1000,
    easing: string = 'easeOutCubic'
  ): Promise<void> {
    return new Promise((resolve) => {
      const tweenId = `entity-move-${entity.id}-${Date.now()}`
      
      const tween: Tween = {
        id: tweenId,
        startTime: performance.now(),
        duration,
        easing: this.easings[easing] || this.easings.easeOutCubic,
        from: { ...entity.position },
        to: { ...targetPosition },
        onUpdate: (position: Position) => {
          entity.position = position
          entity.animationState = 'moving'
        },
        onComplete: () => {
          entity.animationState = 'idle'
          resolve()
        }
      }
      
      this.activeTweens.set(tweenId, tween)
      this.startAnimation()
    })
  }
  
  public animateEntityHighlight(
    entity: AnimatedEntity,
    intensity: number = 1.0,
    duration: number = 500,
    easing: string = 'easeInOutQuad'
  ): Promise<void> {
    return new Promise((resolve) => {
      const tweenId = `entity-highlight-${entity.id}-${Date.now()}`
      const originalGlow = entity.visualProps.glowIntensity
      
      // 高亮阶段
      const highlightTween: Tween = {
        id: tweenId,
        startTime: performance.now(),
        duration: duration / 2,
        easing: this.easings[easing] || this.easings.easeInOutQuad,
        from: { glow: originalGlow, size: entity.visualProps.size },
        to: { glow: intensity, size: entity.visualProps.size * 1.2 },
        onUpdate: (value: any) => {
          entity.visualProps.glowIntensity = value.glow
          entity.visualProps.size = value.size
          entity.animationState = 'highlighted'
        },
        onComplete: () => {
          // 恢复阶段
          const restoreTween: Tween = {
            id: `${tweenId}-restore`,
            startTime: performance.now(),
            duration: duration / 2,
            easing: this.easings[easing] || this.easings.easeInOutQuad,
            from: { glow: intensity, size: entity.visualProps.size },
            to: { glow: originalGlow, size: entity.visualProps.size / 1.2 },
            onUpdate: (value: any) => {
              entity.visualProps.glowIntensity = value.glow
              entity.visualProps.size = value.size
            },
            onComplete: () => {
              entity.animationState = 'idle'
              resolve()
            }
          }
          
          this.activeTweens.set(`${tweenId}-restore`, restoreTween)
        }
      }
      
      this.activeTweens.set(tweenId, highlightTween)
      this.startAnimation()
    })
  }
  
  public animateEntityAppearance(
    entity: AnimatedEntity,
    duration: number = 800,
    easing: string = 'easeOutElastic'
  ): Promise<void> {
    return new Promise((resolve) => {
      const tweenId = `entity-appear-${entity.id}-${Date.now()}`
      
      // 初始化为不可见状态
      const originalSize = entity.visualProps.size
      entity.visualProps.size = 0
      entity.visualProps.opacity = 0
      
      const tween: Tween = {
        id: tweenId,
        startTime: performance.now(),
        duration,
        easing: this.easings[easing] || this.easings.easeOutElastic,
        from: { size: 0, opacity: 0 },
        to: { size: originalSize, opacity: 1 },
        onUpdate: (value: any) => {
          entity.visualProps.size = value.size
          entity.visualProps.opacity = value.opacity
          entity.animationState = 'processing'
        },
        onComplete: () => {
          entity.animationState = 'idle'
          resolve()
        }
      }
      
      this.activeTweens.set(tweenId, tween)
      this.startAnimation()
    })
  }
  
  // ====== 关系动画 ======
  
  public animateRelationshipFlow(
    relationship: AnimatedRelationship,
    duration: number = 2000,
    direction: 'forward' | 'backward' | 'bidirectional' = 'forward'
  ): Promise<void> {
    return new Promise((resolve) => {
      const tweenId = `relationship-flow-${relationship.source}-${relationship.target}-${Date.now()}`
      
      relationship.flowDirection = direction
      relationship.animationState = 'flowing'
      
      const tween: Tween = {
        id: tweenId,
        startTime: performance.now(),
        duration,
        easing: this.easings.linear,
        from: { progress: 0 },
        to: { progress: 1 },
        onUpdate: (value: any) => {
          // 这里可以更新关系的流动效果
          // 实际渲染时会根据progress值绘制流动动画
        },
        onComplete: () => {
          relationship.animationState = 'idle'
          resolve()
        }
      }
      
      this.activeTweens.set(tweenId, tween)
      this.startAnimation()
    })
  }
  
  public animateRelationshipPulse(
    relationship: AnimatedRelationship,
    intensity: number = 2.0,
    duration: number = 600,
    cycles: number = 2
  ): Promise<void> {
    return new Promise((resolve) => {
      const tweenId = `relationship-pulse-${relationship.source}-${relationship.target}-${Date.now()}`
      const originalThickness = relationship.visualProps.thickness
      
      relationship.animationState = 'pulsing'
      
      const cycleDuration = duration / cycles
      let currentCycle = 0
      
      const createPulseCycle = () => {
        if (currentCycle >= cycles) {
          relationship.animationState = 'idle'
          relationship.visualProps.thickness = originalThickness
          resolve()
          return
        }
        
        const pulseTween: Tween = {
          id: `${tweenId}-cycle-${currentCycle}`,
          startTime: performance.now(),
          duration: cycleDuration,
          easing: this.easings.easeInOutQuad,
          from: { thickness: originalThickness },
          to: { thickness: originalThickness * intensity },
          onUpdate: (value: any) => {
            relationship.visualProps.thickness = value.thickness
          },
          onComplete: () => {
            currentCycle++
            createPulseCycle()
          }
        }
        
        this.activeTweens.set(`${tweenId}-cycle-${currentCycle}`, pulseTween)
      }
      
      createPulseCycle()
      this.startAnimation()
    })
  }
  
  // ====== 推理步骤动画 ======
  
  public async animateReasoningStep(
    step: ReasoningStep,
    entities: Map<string, AnimatedEntity>,
    relationships: Map<string, AnimatedRelationship>
  ): Promise<void> {
    // 1. 高亮输入实体
    const inputAnimations = step.inputEntities.map(entityId => {
      const entity = entities.get(entityId)
      return entity ? this.animateEntityHighlight(entity, 0.8, 400) : Promise.resolve()
    })
    
    await Promise.all(inputAnimations)
    
    // 2. 激活相关关系
    const relationshipAnimations = this.findRelevantRelationships(
      step, relationships
    ).map(rel => this.animateRelationshipFlow(rel, 1000))
    
    await Promise.all(relationshipAnimations)
    
    // 3. 高亮输出实体
    const outputAnimations = step.outputEntities.map(entityId => {
      const entity = entities.get(entityId)
      return entity ? this.animateEntityHighlight(entity, 1.0, 600) : Promise.resolve()
    })
    
    await Promise.all(outputAnimations)
    
    // 4. 等待步骤完成
    await this.delay(200)
  }
  
  public async animateReasoningPath(
    path: ReasoningPath,
    entities: Map<string, AnimatedEntity>,
    relationships: Map<string, AnimatedRelationship>
  ): Promise<void> {
    for (const step of path.steps) {
      await this.animateReasoningStep(step, entities, relationships)
      await this.delay(300) // 步骤间暂停
    }
  }
  
  // ====== 批量动画 ======
  
  public async animateLayoutTransition(
    entities: Map<string, AnimatedEntity>,
    newPositions: Map<string, Position>,
    duration: number = 1500,
    stagger: number = 50
  ): Promise<void> {
    const animations: Promise<void>[] = []
    let delay = 0
    
    entities.forEach((entity, entityId) => {
      const newPosition = newPositions.get(entityId)
      if (newPosition) {
        // 添加交错延迟
        setTimeout(() => {
          animations.push(
            this.animateEntityMovement(entity, newPosition, duration)
          )
        }, delay)
        delay += stagger
      }
    })
    
    await Promise.all(animations)
  }
  
  public async animateSceneEntrance(
    entities: Map<string, AnimatedEntity>,
    stagger: number = 100
  ): Promise<void> {
    const animations: Promise<void>[] = []
    let delay = 0
    
    entities.forEach(entity => {
      setTimeout(() => {
        animations.push(this.animateEntityAppearance(entity))
      }, delay)
      delay += stagger
    })
    
    await Promise.all(animations)
  }
  
  // ====== 辅助方法 ======
  
  private interpolate(from: any, to: any, progress: number): any {
    if (typeof from === 'number' && typeof to === 'number') {
      return from + (to - from) * progress
    }
    
    if (typeof from === 'object' && typeof to === 'object') {
      const result: any = {}
      for (const key in from) {
        if (key in to) {
          result[key] = this.interpolate(from[key], to[key], progress)
        }
      }
      return result
    }
    
    return progress < 0.5 ? from : to
  }
  
  private findRelevantRelationships(
    step: ReasoningStep,
    relationships: Map<string, AnimatedRelationship>
  ): AnimatedRelationship[] {
    const relevant: AnimatedRelationship[] = []
    
    step.inputEntities.forEach(inputId => {
      step.outputEntities.forEach(outputId => {
        const forwardKey = `${inputId}-${outputId}`
        const backwardKey = `${outputId}-${inputId}`
        
        const forward = relationships.get(forwardKey)
        const backward = relationships.get(backwardKey)
        
        if (forward) relevant.push(forward)
        if (backward) relevant.push(backward)
      })
    })
    
    return relevant
  }
  
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms))
  }
  
  // ====== 清理方法 ======
  
  public clearAllAnimations(): void {
    this.activeTweens.clear()
    this.stopAnimation()
  }
  
  public cancelAnimation(id: string): void {
    this.activeTweens.delete(id)
  }
  
  public getActiveAnimationCount(): number {
    return this.activeTweens.size
  }
}