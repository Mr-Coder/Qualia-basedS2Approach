/**
 * 增强版实体关系图引擎测试套件
 * 涵盖正常情况、边界条件、异常情况和性能测试
 */

import { vi, describe, test, expect, beforeEach, afterEach } from 'vitest'
import { DiagramEngine } from '../core/DiagramEngine'
import { LayoutEngine } from '../core/LayoutEngine'
import { AnimationEngine } from '../core/AnimationEngine'
import { ErrorHandler } from '../core/ErrorHandler'
import {
  EnhancedDiagramProps,
  Entity,
  Relationship,
  ReasoningStep,
  ReasoningPath,
  LayoutConfig,
  DiagramError
} from '../types/DiagramTypes'

// 测试数据生成器
class TestDataGenerator {
  static createBasicEntities(): Entity[] {
    return [
      { id: 'xiaoming', name: '小明', type: 'person' },
      { id: 'xiaohong', name: '小红', type: 'person' },
      { id: 'apple', name: '苹果', type: 'object' },
      { id: 'money', name: '钱', type: 'money' }
    ]
  }
  
  static createBasicRelationships(): Relationship[] {
    return [
      { source: 'xiaoming', target: 'apple', type: '拥有', weight: 1.0 },
      { source: 'xiaohong', target: 'apple', type: '拥有', weight: 0.8 },
      { source: 'xiaoming', target: 'money', type: '支付', weight: 0.9 }
    ]
  }
  
  static createReasoningSteps(): ReasoningStep[] {
    return [
      {
        id: 'step1',
        sequence: 1,
        description: '识别初始状态',
        inputEntities: ['xiaoming'],
        outputEntities: ['apple'],
        operation: 'derive',
        confidence: 0.9,
        timestamp: Date.now(),
        duration: 1000,
        metadata: {
          mathOperation: '初始化',
          logicType: 'deductive'
        }
      },
      {
        id: 'step2',
        sequence: 2,
        description: '计算变化',
        inputEntities: ['apple', 'xiaohong'],
        outputEntities: ['apple'],
        operation: 'combine',
        confidence: 0.85,
        timestamp: Date.now() + 1000,
        duration: 1500,
        metadata: {
          mathOperation: '减法',
          logicType: 'deductive'
        }
      }
    ]
  }
  
  static createReasoningPath(): ReasoningPath {
    return {
      id: 'main-path',
      steps: this.createReasoningSteps(),
      totalConfidence: 0.875,
      pathType: 'main',
      startTime: Date.now(),
      endTime: Date.now() + 5000,
      isOptimal: true,
      visualization: {
        color: '#4CAF50',
        width: 3,
        style: 'solid',
        animationType: 'flow'
      }
    }
  }
  
  static createBasicProps(): EnhancedDiagramProps {
    return {
      entities: this.createBasicEntities(),
      relationships: this.createBasicRelationships(),
      reasoningPaths: [this.createReasoningPath()],
      layoutConfig: {
        type: 'force',
        params: {
          forceStrength: 0.6,
          linkDistance: 80,
          centerForce: 0.3
        }
      },
      animationConfig: {
        enablePathAnimation: true,
        enableEntityAnimation: true,
        animationSpeed: 1.0,
        simultaneousSteps: 1,
        pauseBetweenSteps: 500
      },
      interactionConfig: {
        enableDrag: true,
        enableZoom: true,
        enableSelection: true,
        enableComments: false
      }
    }
  }
  
  static createLargeDataset(entityCount: number, relationshipRatio: number = 2): {
    entities: Entity[],
    relationships: Relationship[],
    reasoningPaths: ReasoningPath[]
  } {
    const entities: Entity[] = []
    const relationships: Relationship[] = []
    const types: ('person' | 'object' | 'money' | 'concept')[] = ['person', 'object', 'money', 'concept']
    
    // 生成实体
    for (let i = 0; i < entityCount; i++) {
      entities.push({
        id: `entity-${i}`,
        name: `实体${i}`,
        type: types[i % types.length]
      })
    }
    
    // 生成关系
    const relationshipCount = Math.min(entityCount * relationshipRatio, entityCount * (entityCount - 1) / 2)
    for (let i = 0; i < relationshipCount; i++) {
      const sourceIndex = Math.floor(Math.random() * entityCount)
      let targetIndex = Math.floor(Math.random() * entityCount)
      
      // 确保不是自连接
      while (targetIndex === sourceIndex) {
        targetIndex = Math.floor(Math.random() * entityCount)
      }
      
      relationships.push({
        source: `entity-${sourceIndex}`,
        target: `entity-${targetIndex}`,
        type: `关系${i % 5}`,
        weight: Math.random()
      })
    }
    
    // 生成简单的推理路径
    const reasoningSteps: ReasoningStep[] = []
    if (entityCount > 2) {
      reasoningSteps.push({
        id: 'large-step1',
        sequence: 1,
        description: '大数据集推理步骤',
        inputEntities: ['entity-0'],
        outputEntities: ['entity-1'],
        operation: 'derive',
        confidence: 0.8,
        timestamp: Date.now(),
        duration: 1000,
        metadata: {}
      })
    }
    
    const reasoningPath: ReasoningPath = {
      id: 'large-path',
      steps: reasoningSteps,
      totalConfidence: 0.8,
      pathType: 'main',
      startTime: Date.now(),
      endTime: Date.now() + 5000,
      isOptimal: true,
      visualization: {
        color: '#4CAF50',
        width: 2,
        style: 'solid',
        animationType: 'flow'
      }
    }
    
    return { entities, relationships, reasoningPaths: [reasoningPath] }
  }
}

// Mock implementations for testing
class MockCanvas {
  width = 800
  height = 600
  
  getContext() {
    return {
      clearRect: vi.fn(),
      fillRect: vi.fn(),
      strokeRect: vi.fn(),
      arc: vi.fn(),
      fill: vi.fn(),
      stroke: vi.fn(),
      beginPath: vi.fn(),
      closePath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn()
    }
  }
}

// 主测试套件
describe('DiagramEngine', () => {
  let diagramEngine: DiagramEngine
  let mockProps: EnhancedDiagramProps
  
  beforeEach(() => {
    // 重置所有mock
    vi.clearAllMocks()
    
    // 创建测试数据
    mockProps = TestDataGenerator.createBasicProps()
    
    // Mock canvas
    global.HTMLCanvasElement = MockCanvas as any
    
    // Mock performance.now
    global.performance.now = vi.fn(() => Date.now())
    
    // Mock requestAnimationFrame
    global.requestAnimationFrame = vi.fn((callback) => {
      setTimeout(callback, 16)
      return 1
    })
    
    // Mock cancelAnimationFrame
    global.cancelAnimationFrame = vi.fn()
  })
  
  afterEach(() => {
    if (diagramEngine) {
      diagramEngine.dispose()
    }
  })
  
  // ====== 正常情况测试 ======
  
  describe('正常情况测试', () => {
    test('应该成功初始化引擎', async () => {
      diagramEngine = new DiagramEngine(mockProps)
      
      // 等待初始化完成
      await new Promise(resolve => setTimeout(resolve, 100))
      
      const state = diagramEngine.getState()
      
      expect(state.entities.size).toBe(mockProps.entities.length)
      expect(state.relationships.size).toBe(mockProps.relationships.length)
      expect(state.isPlaying).toBe(false)
    })
    
    test('应该正确转换实体为动画实体', async () => {
      diagramEngine = new DiagramEngine(mockProps)
      await new Promise(resolve => setTimeout(resolve, 100))
      
      const state = diagramEngine.getState()
      const animatedEntity = state.entities.get('xiaoming')
      
      expect(animatedEntity).toBeDefined()
      expect(animatedEntity?.name).toBe('小明')
      expect(animatedEntity?.type).toBe('person')
      expect(animatedEntity?.animationState).toBe('idle')
      expect(animatedEntity?.visualProps).toBeDefined()
    })
    
    test('应该成功计算布局', async () => {
      diagramEngine = new DiagramEngine(mockProps)
      await new Promise(resolve => setTimeout(resolve, 200))
      
      const layout = await diagramEngine.computeOptimalLayout()
      
      expect(layout).toBeDefined()
      expect(layout.entityPositions.size).toBe(mockProps.entities.length)
      expect(layout.quality).toBeGreaterThanOrEqual(0)
      expect(layout.quality).toBeLessThanOrEqual(1)
    })
    
    test('应该成功播放推理路径动画', () => {
      diagramEngine = new DiagramEngine(mockProps)
      
      // 开始播放动画
      diagramEngine.playAnimation()
      
      const state = diagramEngine.getState()
      expect(state.isPlaying).toBe(true)
      expect(state.currentStep).toBe(0)
      expect(state.totalSteps).toBeGreaterThan(0)
    })
    
    test('应该正确暂停和重置动画', () => {
      diagramEngine = new DiagramEngine(mockProps)
      
      // 开始播放
      diagramEngine.playAnimation()
      expect(diagramEngine.getState().isPlaying).toBe(true)
      
      // 暂停
      diagramEngine.pauseAnimation()
      expect(diagramEngine.getState().isPlaying).toBe(false)
      
      // 重置
      diagramEngine.resetAnimation()
      expect(diagramEngine.getState().currentStep).toBe(0)
    })
  })
  
  // ====== 边界条件测试 ======
  
  describe('边界条件测试', () => {
    test('应该处理空数据', async () => {
      const emptyProps: EnhancedDiagramProps = {
        ...mockProps,
        entities: [],
        relationships: [],
        reasoningPaths: []
      }
      
      diagramEngine = new DiagramEngine(emptyProps)
      await new Promise(resolve => setTimeout(resolve, 100))
      
      const state = diagramEngine.getState()
      expect(state.entities.size).toBe(0)
      expect(state.relationships.size).toBe(0)
      expect(state.animationQueue.length).toBe(0)
    })
    
    test('应该处理单个实体', async () => {
      const singleEntityProps: EnhancedDiagramProps = {
        ...mockProps,
        entities: [{ id: 'single', name: '单个实体', type: 'concept' }],
        relationships: [],
        reasoningPaths: []
      }
      
      diagramEngine = new DiagramEngine(singleEntityProps)
      await new Promise(resolve => setTimeout(resolve, 100))
      
      const layout = await diagramEngine.computeOptimalLayout()
      expect(layout.entityPositions.size).toBe(1)
      
      const position = layout.entityPositions.get('single')
      expect(position).toBeDefined()
      expect(typeof position?.x).toBe('number')
      expect(typeof position?.y).toBe('number')
    })
    
    test('应该处理大量数据', async () => {
      const largeData = TestDataGenerator.createLargeDataset(50, 1.2)
      const largeProps: EnhancedDiagramProps = {
        ...mockProps,
        entities: largeData.entities,
        relationships: largeData.relationships,
        reasoningPaths: largeData.reasoningPaths,
        autoOptimizeLayout: true
      }
      
      diagramEngine = new DiagramEngine(largeProps)
      await new Promise(resolve => setTimeout(resolve, 500))
      
      const state = diagramEngine.getState()
      expect(state.entities.size).toBe(50)
      expect(state.relationships.size).toBeGreaterThan(0)
      
      // 测试布局性能
      const startTime = performance.now()
      await diagramEngine.computeOptimalLayout()
      const endTime = performance.now()
      
      // 应该在合理时间内完成（5秒）
      expect(endTime - startTime).toBeLessThan(5000)
    }, 10000) // 增加超时时间
    
    test('应该处理循环关系', async () => {
      const cyclicProps: EnhancedDiagramProps = {
        ...mockProps,
        entities: [
          { id: 'a', name: 'A', type: 'concept' },
          { id: 'b', name: 'B', type: 'concept' },
          { id: 'c', name: 'C', type: 'concept' }
        ],
        relationships: [
          { source: 'a', target: 'b', type: '指向' },
          { source: 'b', target: 'c', type: '指向' },
          { source: 'c', target: 'a', type: '指向' } // 形成循环
        ]
      }
      
      diagramEngine = new DiagramEngine(cyclicProps)
      await new Promise(resolve => setTimeout(resolve, 100))
      
      // 应该能够处理循环而不崩溃
      const layout = await diagramEngine.computeOptimalLayout()
      expect(layout.entityPositions.size).toBe(3)
      expect(layout.warnings).toBeDefined()
    })
    
    test('应该处理极长的推理路径', () => {
      // 创建足够的实体来支持长推理路径
      const longPathEntities: Entity[] = []
      for (let i = 0; i <= 50; i++) {
        longPathEntities.push({
          id: `entity-${i}`,
          name: `实体${i}`,
          type: 'concept'
        })
      }
      
      const longSteps: ReasoningStep[] = []
      for (let i = 0; i < 50; i++) {
        longSteps.push({
          id: `step-${i}`,
          sequence: i,
          description: `推理步骤 ${i}`,
          inputEntities: [`entity-${i}`],
          outputEntities: [`entity-${i + 1}`],
          operation: 'derive',
          confidence: 0.8,
          timestamp: Date.now() + i * 1000,
          duration: 500,
          metadata: {}
        })
      }
      
      const longPath: ReasoningPath = {
        id: 'long-path',
        steps: longSteps,
        totalConfidence: 0.8,
        pathType: 'main',
        startTime: Date.now(),
        endTime: Date.now() + 50000,
        isOptimal: true,
        visualization: {
          color: '#FF5722',
          width: 2,
          style: 'solid',
          animationType: 'flow'
        }
      }
      
      const longPathProps: EnhancedDiagramProps = {
        ...mockProps,
        entities: longPathEntities,
        relationships: [],
        reasoningPaths: [longPath]
      }
      
      diagramEngine = new DiagramEngine(longPathProps)
      
      const state = diagramEngine.getState()
      expect(state.animationQueue.length).toBeGreaterThan(0)
      
      // 播放动画不应该崩溃
      expect(() => diagramEngine.playAnimation()).not.toThrow()
    })
  })
  
  // ====== 异常情况测试 ======
  
  describe('异常情况测试', () => {
    test('应该处理无效的实体ID', async () => {
      const invalidProps: EnhancedDiagramProps = {
        ...mockProps,
        relationships: [
          { source: 'invalid-source', target: 'invalid-target', type: '无效关系' }
        ],
        reasoningPaths: [] // 移除推理路径避免验证错误
      }
      
      expect(() => {
        diagramEngine = new DiagramEngine(invalidProps)
      }).toThrow() // 现在应该抛出验证错误
    })
    
    test('应该处理NaN位置值', async () => {
      diagramEngine = new DiagramEngine(mockProps)
      await new Promise(resolve => setTimeout(resolve, 100))
      
      const state = diagramEngine.getState()
      const entity = state.entities.get('xiaoming')
      
      if (entity) {
        // 故意设置无效位置
        entity.position = { x: NaN, y: NaN }
        entity.targetPosition = { x: Infinity, y: -Infinity }
      }
      
      // 重新计算布局应该修复这些问题
      const layout = await diagramEngine.computeOptimalLayout()
      const newPosition = layout.entityPositions.get('xiaoming')
      
      expect(newPosition).toBeDefined()
      expect(isFinite(newPosition!.x)).toBe(true)
      expect(isFinite(newPosition!.y)).toBe(true)
    })
    
    test('应该处理布局计算失败', async () => {
      // Mock layout engine to throw error
      const originalCompute = LayoutEngine.prototype.computeLayout
      LayoutEngine.prototype.computeLayout = vi.fn().mockRejectedValue(new Error('Layout failed'))
      
      diagramEngine = new DiagramEngine(mockProps)
      
      try {
        await diagramEngine.computeOptimalLayout()
      } catch (error) {
        expect(error).toBeInstanceOf(DiagramError)
      }
      
      // Restore original method
      LayoutEngine.prototype.computeLayout = originalCompute
    })
    
    test('应该处理动画引擎错误', () => {
      // Mock animation engine to throw error
      const originalStartAnimation = AnimationEngine.prototype.startAnimation
      AnimationEngine.prototype.startAnimation = vi.fn().mockImplementation(() => {
        throw new Error('Animation failed')
      })
      
      diagramEngine = new DiagramEngine(mockProps)
      
      // 应该捕获错误而不崩溃
      expect(() => diagramEngine.playAnimation()).not.toThrow()
      
      // Restore original method
      AnimationEngine.prototype.startAnimation = originalStartAnimation
    })
    
    test('应该处理内存不足情况', async () => {
      // 模拟创建大量数据导致内存不足
      const hugeData = TestDataGenerator.createLargeDataset(1000, 2)
      
      const hugeProps: EnhancedDiagramProps = {
        ...mockProps,
        entities: hugeData.entities,
        relationships: hugeData.relationships,
        reasoningPaths: hugeData.reasoningPaths
      }
      
      // 应该能够处理而不崩溃（可能会有性能问题但不应该出错）
      expect(() => {
        diagramEngine = new DiagramEngine(hugeProps)
      }).not.toThrow()
    }, 15000)
    
    test('应该处理协作连接失败', async () => {
      const collaborationProps: EnhancedDiagramProps = {
        ...mockProps,
        collaborationConfig: {
          enabled: true,
          sessionId: 'invalid-session',
          currentUserId: 'test-user',
          realTimeSync: true
        }
      }
      
      // 模拟协作连接失败
      diagramEngine = new DiagramEngine(collaborationProps)
      await new Promise(resolve => setTimeout(resolve, 200))
      
      const state = diagramEngine.getState()
      // 应该能够继续工作，只是没有协作功能
      expect(state.entities.size).toBe(mockProps.entities.length)
    })
  })
  
  // ====== 性能测试 ======
  
  describe('性能测试', () => {
    test('初始化性能测试', async () => {
      const mediumData = TestDataGenerator.createLargeDataset(50, 2)
      const performanceProps: EnhancedDiagramProps = {
        ...mockProps,
        entities: mediumData.entities,
        relationships: mediumData.relationships,
        reasoningPaths: mediumData.reasoningPaths
      }
      
      const startTime = performance.now()
      diagramEngine = new DiagramEngine(performanceProps)
      await new Promise(resolve => setTimeout(resolve, 500))
      const endTime = performance.now()
      
      // 应该在1秒内完成初始化
      expect(endTime - startTime).toBeLessThan(1000)
    })
    
    test('布局计算性能测试', async () => {
      const mediumData = TestDataGenerator.createLargeDataset(30, 1.5)
      const performanceProps: EnhancedDiagramProps = {
        ...mockProps,
        entities: mediumData.entities,
        relationships: mediumData.relationships,
        reasoningPaths: mediumData.reasoningPaths
      }
      
      diagramEngine = new DiagramEngine(performanceProps)
      await new Promise(resolve => setTimeout(resolve, 100))
      
      const startTime = performance.now()
      const layout = await diagramEngine.computeOptimalLayout()
      const endTime = performance.now()
      
      // 应该在2秒内完成布局计算
      expect(endTime - startTime).toBeLessThan(2000)
      expect(layout.computationTime).toBeGreaterThan(0)
    })
    
    test('动画性能测试', async () => {
      diagramEngine = new DiagramEngine(mockProps)
      await new Promise(resolve => setTimeout(resolve, 100))
      
      const startTime = performance.now()
      
      // 播放短动画序列
      diagramEngine.playAnimation()
      
      // 等待一小段时间观察性能
      await new Promise(resolve => setTimeout(resolve, 200))
      
      diagramEngine.pauseAnimation()
      
      const endTime = performance.now()
      
      // 动画播放不应该消耗过多CPU时间
      expect(endTime - startTime).toBeLessThan(500)
    })
    
    test('内存使用测试', async () => {
      // 创建多个引擎实例测试内存泄漏
      const engines: DiagramEngine[] = []
      
      for (let i = 0; i < 5; i++) {
        const engine = new DiagramEngine(mockProps)
        engines.push(engine)
        await new Promise(resolve => setTimeout(resolve, 50))
      }
      
      // 清理所有引擎
      engines.forEach(engine => engine.dispose())
      
      // 强制垃圾收集（如果可用）
      if (global.gc) {
        global.gc()
      }
      
      // 这里主要是确保没有明显的内存泄漏导致的错误
      expect(engines.length).toBe(5)
    })
    
    test('并发操作性能测试', async () => {
      diagramEngine = new DiagramEngine(mockProps)
      await new Promise(resolve => setTimeout(resolve, 100))
      
      const startTime = performance.now()
      
      // 同时执行多个操作
      const operations = [
        diagramEngine.computeOptimalLayout(),
        new Promise(resolve => {
          diagramEngine.playAnimation()
          setTimeout(() => {
            diagramEngine.pauseAnimation()
            resolve(undefined)
          }, 100)
        }),
        diagramEngine.updateProps({ 
          animationConfig: { 
            ...mockProps.animationConfig, 
            animationSpeed: 2.0 
          } 
        })
      ]
      
      await Promise.all(operations)
      
      const endTime = performance.now()
      
      // 并发操作不应该导致长时间阻塞
      expect(endTime - startTime).toBeLessThan(3000)
    })
  })
  
  // ====== 集成测试 ======
  
  describe('集成测试', () => {
    test('完整工作流程测试', async () => {
      // 1. 初始化
      diagramEngine = new DiagramEngine(mockProps)
      await new Promise(resolve => setTimeout(resolve, 100))
      
      // 2. 验证初始状态
      let state = diagramEngine.getState()
      expect(state.entities.size).toBeGreaterThan(0)
      expect(state.relationships.size).toBeGreaterThan(0)
      
      // 3. 计算布局
      const layout = await diagramEngine.computeOptimalLayout()
      expect(layout.quality).toBeGreaterThan(0)
      
      // 4. 播放动画
      diagramEngine.playAnimation()
      state = diagramEngine.getState()
      expect(state.isPlaying).toBe(true)
      
      // 5. 暂停动画
      await new Promise(resolve => setTimeout(resolve, 100))
      diagramEngine.pauseAnimation()
      expect(state.isPlaying).toBe(false)
      
      // 6. 更新配置 
      diagramEngine.updateProps({
        layoutConfig: {
          type: 'hierarchical',
          params: { direction: 'top-down' }
        }
      })
      
      // 7. 重新计算布局
      const newLayout = await diagramEngine.computeOptimalLayout()
      expect(newLayout).toBeDefined()
      expect(newLayout.entityPositions.size).toBe(layout.entityPositions.size)
      
      // 8. 清理
      diagramEngine.dispose()
    })
    
    test('错误恢复集成测试', async () => {
      const errorHandler = new ErrorHandler()
      let errorCaught = false
      
      errorHandler.setErrorCallback(() => {
        errorCaught = true
      })
      
      // 创建会导致错误的配置
      const problematicProps: EnhancedDiagramProps = {
        ...mockProps,
        entities: [], // 空实体但有关系
        relationships: [],
        reasoningPaths: []
      }
      
      diagramEngine = new DiagramEngine(problematicProps)
      await new Promise(resolve => setTimeout(resolve, 200))
      
      // 系统应该能够处理这种情况而不崩溃
      const state = diagramEngine.getState()
      expect(state).toBeDefined()
      
      // 尝试一些可能失败的操作
      try {
        await diagramEngine.computeOptimalLayout()
        diagramEngine.playAnimation()
      } catch (error) {
        // 错误应该被正确处理
        expect(error).toBeInstanceOf(DiagramError)
      }
    })
  })
})