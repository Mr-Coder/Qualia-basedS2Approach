/**
 * 增强版实体关系图错误处理系统
 * 提供完整的错误捕获、恢复和用户反馈机制
 */

import { DiagramError, ErrorInfo } from '../types/DiagramTypes'

interface ErrorRecoveryStrategy {
  canRecover: (error: DiagramError) => boolean
  recover: (error: DiagramError, context: any) => Promise<boolean>
  fallback?: () => void
}

interface ErrorReportData {
  error: DiagramError
  context: {
    userAgent: string
    timestamp: number
    sessionId?: string
    userId?: string
    diagramState?: any
  }
  stackTrace: string
  reproductionSteps?: string[]
}

export class ErrorHandler {
  private errorLog: ErrorInfo[] = []
  private recoveryStrategies: Map<string, ErrorRecoveryStrategy> = new Map()
  private errorReportingEnabled: boolean = true
  private maxErrorLogSize: number = 100
  private onErrorCallback?: (error: DiagramError) => void
  private recentlyHandledErrors: Map<string, number> = new Map()
  private readonly DUPLICATE_ERROR_WINDOW = 5000 // 5秒内不重复处理
  
  constructor() {
    this.initializeRecoveryStrategies()
    this.setupGlobalErrorHandling()
  }
  
  // ====== 错误去重复机制 ======
  
  private isRecentlyHandled(errorKey: string): boolean {
    const lastHandled = this.recentlyHandledErrors.get(errorKey)
    if (!lastHandled) return false
    
    return (Date.now() - lastHandled) < this.DUPLICATE_ERROR_WINDOW
  }
  
  private markAsHandled(errorKey: string): void {
    this.recentlyHandledErrors.set(errorKey, Date.now())
    
    // 清理过期的标记
    this.cleanupHandledMarkers()
  }
  
  private clearHandledMarker(errorKey: string): void {
    this.recentlyHandledErrors.delete(errorKey)
  }
  
  private cleanupHandledMarkers(): void {
    const now = Date.now()
    const toDelete: string[] = []
    
    for (const [key, timestamp] of this.recentlyHandledErrors) {
      if (now - timestamp > this.DUPLICATE_ERROR_WINDOW) {
        toDelete.push(key)
      }
    }
    
    toDelete.forEach(key => this.recentlyHandledErrors.delete(key))
  }
  
  // ====== 错误处理核心方法 ======
  
  public async handleError(
    error: DiagramError,
    context?: any,
    recoveryAttempts: number = 0
  ): Promise<boolean> {
    try {
      // 防止无限递归和错误洪浌
      if (recoveryAttempts > 5) {
        console.error('Too many recovery attempts, giving up:', error)
        this.executeCriticalFallback()
        return false
      }
      
      // 防重复处理
      const errorKey = `${error.code}-${error.message}`
      if (this.isRecentlyHandled(errorKey)) {
        console.warn('Duplicate error detected, skipping:', error.code)
        return false
      }
      this.markAsHandled(errorKey)
      
      // 记录错误
      this.logError(error, context)
      
      // 通知回调（使用try-catch防止回调错误）
      try {
        this.onErrorCallback?.(error)
      } catch (callbackError) {
        console.error('Error in error callback:', callbackError)
      }
      
      // 尝试恢复
      if (recoveryAttempts < 3) {
        const recovered = await this.attemptRecovery(error, context, recoveryAttempts)
        if (recovered) {
          console.log(`Successfully recovered from error: ${error.code}`)
          this.clearHandledMarker(errorKey)
          return true
        }
      }
      
      // 恢复失败，执行降级策略
      await this.executeFallbackStrategy(error, context)
      
      // 上报错误（如果启用）
      if (this.errorReportingEnabled) {
        try {
          this.reportError(error, context)
        } catch (reportError) {
          console.error('Failed to report error:', reportError)
        }
      }
      
      return false
      
    } catch (handlingError) {
      // 错误处理器本身出错，记录并执行最终降级
      console.error('Error in error handler:', handlingError)
      this.executeCriticalFallback()
      return false
    }
  }
  
  private async attemptRecovery(
    error: DiagramError,
    context: any,
    attempts: number
  ): Promise<boolean> {
    const strategy = this.recoveryStrategies.get(error.code)
    
    if (strategy && strategy.canRecover(error)) {
      try {
        console.log(`Attempting recovery for error: ${error.code} (attempt ${attempts + 1})`)
        const recovered = await strategy.recover(error, context)
        
        if (recovered) {
          this.logRecovery(error, attempts + 1)
          return true
        }
      } catch (recoveryError) {
        console.error('Recovery attempt failed:', recoveryError)
        
        // 递归重试（最多3次）
        if (attempts < 2) {
          await this.delay(1000 * (attempts + 1)) // 指数退避
          return this.attemptRecovery(error, context, attempts + 1)
        }
      }
    }
    
    return false
  }
  
  // ====== 恢复策略 ======
  
  private initializeRecoveryStrategies(): void {
    // 初始化错误恢复策略
    this.recoveryStrategies.set('INIT_ERROR', {
      canRecover: () => true,
      recover: async (error, context) => {
        console.log('Attempting to reinitialize diagram engine')
        // 重新初始化基础状态
        if (context?.diagramEngine) {
          try {
            await context.diagramEngine.initialize()
            return true
          } catch {
            return false
          }
        }
        return false
      },
      fallback: () => {
        console.log('Falling back to basic diagram display')
      }
    })
    
    this.recoveryStrategies.set('LAYOUT_ERROR', {
      canRecover: () => true,
      recover: async (error, context) => {
        console.log('Attempting to recover from layout error')
        // 尝试使用更简单的布局算法
        if (context?.layoutEngine) {
          try {
            const simpleConfig = {
              type: 'circular' as const,
              params: {}
            }
            await context.layoutEngine.computeLayout(
              context.entities || [],
              context.relationships || [],
              simpleConfig
            )
            return true
          } catch {
            return false
          }
        }
        return false
      },
      fallback: () => {
        console.log('Using random layout as fallback')
      }
    })
    
    this.recoveryStrategies.set('ANIMATION_ERROR', {
      canRecover: () => true,
      recover: async (error, context) => {
        console.log('Attempting to recover from animation error')
        // 禁用动画并使用静态显示
        if (context?.animationEngine) {
          try {
            context.animationEngine.clearAllAnimations()
            // 可以设置一个标志禁用后续动画
            context.animationDisabled = true
            return true
          } catch {
            return false
          }
        }
        return false
      },
      fallback: () => {
        console.log('Disabling all animations')
      }
    })
    
    this.recoveryStrategies.set('COLLABORATION_ERROR', {
      canRecover: () => true,
      recover: async (error, context) => {
        console.log('Attempting to recover from collaboration error')
        // 切换到本地模式
        if (context?.collaborationEngine) {
          try {
            context.collaborationEngine.disconnect()
            context.collaborationMode = false
            return true
          } catch {
            return false
          }
        }
        return false
      },
      fallback: () => {
        console.log('Switching to local-only mode')
      }
    })
    
    this.recoveryStrategies.set('RENDER_ERROR', {
      canRecover: () => true,
      recover: async (error, context) => {
        console.log('Attempting to recover from render error')
        // 尝试重新创建渲染上下文
        if (context?.canvas) {
          try {
            const canvas = context.canvas
            const ctx = canvas.getContext('2d')
            if (ctx) {
              ctx.clearRect(0, 0, canvas.width, canvas.height)
              return true
            }
          } catch {
            return false
          }
        }
        return false
      },
      fallback: () => {
        console.log('Using simplified rendering')
      }
    })
    
    // 数据相关错误
    this.recoveryStrategies.set('DATA_VALIDATION_ERROR', {
      canRecover: (error) => {
        // 检查是否有有效的备用数据
        return error.details?.hasBackupData === true
      },
      recover: async (error, context) => {
        console.log('Attempting to recover from data validation error')
        // 使用备用数据或默认数据
        if (context?.setFallbackData) {
          try {
            context.setFallbackData()
            return true
          } catch {
            return false
          }
        }
        return false
      }
    })
    
    // 网络相关错误
    this.recoveryStrategies.set('NETWORK_ERROR', {
      canRecover: () => true,
      recover: async (error, context) => {
        console.log('Attempting to recover from network error')
        // 尝试重新连接或使用缓存数据
        if (context?.networkManager) {
          try {
            await context.networkManager.reconnect()
            return true
          } catch {
            // 尝试使用缓存数据
            if (context?.cacheManager?.hasCachedData()) {
              context.cacheManager.useCachedData()
              return true
            }
            return false
          }
        }
        return false
      }
    })
    
    // 内存不足错误
    this.recoveryStrategies.set('MEMORY_ERROR', {
      canRecover: () => true,
      recover: async (error, context) => {
        console.log('Attempting to recover from memory error')
        // 清理内存和缓存
        if (context?.memoryManager) {
          try {
            context.memoryManager.clearCache()
            context.memoryManager.forceGarbageCollection()
            return true
          } catch {
            return false
          }
        }
        
        // 通用内存清理
        if (typeof window !== 'undefined' && (window as any).gc) {
          (window as any).gc()
        }
        
        // 减少数据规模
        if (context?.diagramEngine) {
          context.diagramEngine.simplifyData()
          return true
        }
        
        return false
      },
      fallback: () => {
        console.log('Enabling low-memory mode')
      }
    })
    
    // 性能问题
    this.recoveryStrategies.set('PERFORMANCE_ERROR', {
      canRecover: () => true,
      recover: async (error, context) => {
        console.log('Attempting to recover from performance error')
        // 禁用高性能消耗功能
        if (context?.performanceManager) {
          try {
            context.performanceManager.enablePerformanceMode()
            return true
          } catch {
            return false
          }
        }
        
        // 降低动画质量
        if (context?.animationEngine) {
          context.animationEngine.setQuality('low')
          return true
        }
        
        return false
      },
      fallback: () => {
        console.log('Enabling performance mode')
      }
    })
  }
  
  // ====== 降级策略 ======
  
  private async executeFallbackStrategy(error: DiagramError, context?: any): Promise<void> {
    const strategy = this.recoveryStrategies.get(error.code)
    
    if (strategy?.fallback) {
      try {
        strategy.fallback()
      } catch (fallbackError) {
        console.error('Fallback strategy failed:', fallbackError)
        this.executeCriticalFallback()
      }
    } else {
      // 通用降级策略
      this.executeGenericFallback(error, context)
    }
  }
  
  private executeGenericFallback(error: DiagramError, context?: any): void {
    console.log(`Executing generic fallback for error: ${error.code}`)
    
    // 显示错误消息给用户
    this.showUserErrorMessage(error)
    
    // 尝试保存当前状态
    this.saveCurrentState(context)
    
    // 切换到安全模式
    this.enableSafeMode(context)
  }
  
  private executeCriticalFallback(): void {
    console.error('Executing critical fallback - system in unstable state')
    
    try {
      // 显示严重错误消息
      this.showCriticalErrorMessage()
      
      // 尝试清理资源
      this.emergencyCleanup()
      
      // 尝试重新加载页面（最后手段）
      if (typeof window !== 'undefined' && window.location) {
        setTimeout(() => {
          try {
            if (confirm('系统遇到严重错误，是否重新加载页面？')) {
              window.location.reload()
            }
          } catch (reloadError) {
            console.error('Failed to reload page:', reloadError)
            // 最终后备方案
            alert('请手动刷新页面来恢复系统')
          }
        }, 1000)
      }
    } catch (fallbackError) {
      console.error('Critical fallback failed:', fallbackError)
      // 最后的最后手段
      if (typeof console !== 'undefined') {
        console.error('System is in critical state - manual intervention required')
      }
    }
  }
  
  private emergencyCleanup(): void {
    try {
      // 清理事件监听器
      if (typeof window !== 'undefined') {
        const events = ['error', 'unhandledrejection']
        events.forEach(eventType => {
          try {
            window.removeEventListener(eventType, null as any)
          } catch {}
        })
      }
      
      // 清理定时器
      if (typeof window !== 'undefined') {
        const highestId = window.setTimeout(() => {}, 0)
        for (let i = 0; i < highestId; i++) {
          try {
            window.clearTimeout(i)
            window.clearInterval(i)
          } catch {}
        }
      }
      
      // 清理内存
      this.errorLog = []
      this.recentlyHandledErrors.clear()
      
    } catch (cleanupError) {
      console.error('Emergency cleanup failed:', cleanupError)
    }
  }
  
  // ====== 用户反馈 ======
  
  private showUserErrorMessage(error: DiagramError): void {
    const userMessage = this.getUserFriendlyMessage(error)
    
    // 这里可以集成到UI系统显示用户友好的错误消息
    console.warn('User message:', userMessage)
    
    // 可以通过事件系统通知UI组件显示错误
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('diagram-error', {
        detail: {
          message: userMessage,
          severity: this.getErrorSeverity(error),
          canRetry: this.canRetry(error)
        }
      }))
    }
  }
  
  private showCriticalErrorMessage(): void {
    const message = '系统遇到严重错误，建议刷新页面重试。'
    
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('diagram-critical-error', {
        detail: { message }
      }))
    }
  }
  
  private getUserFriendlyMessage(error: DiagramError): string {
    const messages: Record<string, string> = {
      'INIT_ERROR': '图表初始化失败，请稍后重试',
      'LAYOUT_ERROR': '图表布局计算出错，已切换到简化模式',
      'ANIMATION_ERROR': '动画播放出错，已切换到静态模式',
      'COLLABORATION_ERROR': '协作功能连接失败，已切换到本地模式',
      'RENDER_ERROR': '图表渲染出错，请检查浏览器兼容性',
      'DATA_VALIDATION_ERROR': '数据格式不正确，请检查输入数据',
      'NETWORK_ERROR': '网络连接失败，请检查网络状态',
      'PERMISSION_DENIED': '权限不足，无法执行此操作'
    }
    
    return messages[error.code] || '未知错误，请联系技术支持'
  }
  
  private getErrorSeverity(error: DiagramError): 'low' | 'medium' | 'high' | 'critical' {
    const severityMap: Record<string, 'low' | 'medium' | 'high' | 'critical'> = {
      'ANIMATION_ERROR': 'low',
      'DATA_VALIDATION_ERROR': 'medium',
      'LAYOUT_ERROR': 'medium',
      'COLLABORATION_ERROR': 'medium',
      'RENDER_ERROR': 'high',
      'INIT_ERROR': 'high',
      'NETWORK_ERROR': 'high'
    }
    
    return severityMap[error.code] || 'medium'
  }
  
  private canRetry(error: DiagramError): boolean {
    const retryableErrors = [
      'NETWORK_ERROR',
      'COLLABORATION_ERROR',
      'INIT_ERROR'
    ]
    
    return retryableErrors.includes(error.code)
  }
  
  // ====== 状态管理 ======
  
  private saveCurrentState(context?: any): void {
    try {
      if (context?.diagramState) {
        const stateSnapshot = JSON.stringify(context.diagramState)
        localStorage.setItem('diagram-error-recovery-state', stateSnapshot)
        console.log('Current state saved for error recovery')
      }
    } catch (saveError) {
      console.error('Failed to save current state:', saveError)
    }
  }
  
  private enableSafeMode(context?: any): void {
    console.log('Enabling safe mode')
    
    if (context) {
      // 禁用高级功能
      context.animationEnabled = false
      context.collaborationEnabled = false
      context.advancedLayoutEnabled = false
      
      // 使用最基本的配置
      context.safeMode = true
    }
  }
  
  // ====== 错误日志 ======
  
  private logError(error: DiagramError, context?: any): void {
    const errorInfo: ErrorInfo = {
      code: error.code,
      message: error.message,
      timestamp: Date.now(),
      context: this.sanitizeContext(context),
      severity: this.getErrorSeverity(error)
    }
    
    this.errorLog.push(errorInfo)
    
    // 限制日志大小
    if (this.errorLog.length > this.maxErrorLogSize) {
      this.errorLog.shift()
    }
    
    console.error(`DiagramError [${error.code}]:`, error.message, error.details)
  }
  
  private logRecovery(error: DiagramError, attempts: number): void {
    console.log(`Error recovery successful: ${error.code} after ${attempts} attempts`)
    
    // 可以记录恢复统计信息
    const recoveryInfo = {
      errorCode: error.code,
      attempts,
      timestamp: Date.now()
    }
    
    // 存储恢复统计以便分析
    this.storeRecoveryStats(recoveryInfo)
  }
  
  private sanitizeContext(context: any): any {
    if (!context) return null
    
    // 移除敏感信息和循环引用
    try {
      return {
        timestamp: Date.now(),
        userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'unknown',
        entitiesCount: context.entities?.length || 0,
        relationshipsCount: context.relationships?.length || 0,
        isAnimating: context.isAnimating || false,
        isCollaborating: context.isCollaborating || false
      }
    } catch {
      return { sanitizationFailed: true }
    }
  }
  
  // ====== 错误上报 ======
  
  private reportError(error: DiagramError, context?: any): void {
    if (!this.errorReportingEnabled) return
    
    const reportData: ErrorReportData = {
      error,
      context: {
        userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'unknown',
        timestamp: Date.now(),
        sessionId: context?.sessionId,
        userId: context?.userId,
        diagramState: this.sanitizeContext(context)
      },
      stackTrace: error.stack || 'No stack trace available'
    }
    
    // 在真实应用中，这里会发送到错误收集服务
    console.log('Error report data:', reportData)
    
    // 可以发送到服务器
    // this.sendErrorReport(reportData)
  }
  
  // ====== 全局错误处理 ======
  
  private setupGlobalErrorHandling(): void {
    if (typeof window === 'undefined') return
    
    // 捕获未处理的Promise错误
    window.addEventListener('unhandledrejection', (event) => {
      const error = new DiagramError(
        event.reason?.message || 'Unhandled promise rejection',
        'UNHANDLED_PROMISE_REJECTION',
        event.reason
      )
      
      this.handleError(error)
    })
    
    // 捕获全局JavaScript错误
    window.addEventListener('error', (event) => {
      const error = new DiagramError(
        event.message || 'Global JavaScript error',
        'GLOBAL_JS_ERROR',
        {
          filename: event.filename,
          lineno: event.lineno,
          colno: event.colno,
          error: event.error
        }
      )
      
      this.handleError(error)
    })
  }
  
  // ====== 工具方法 ======
  
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms))
  }
  
  private storeRecoveryStats(info: any): void {
    try {
      const stats = JSON.parse(localStorage.getItem('diagram-recovery-stats') || '[]')
      stats.push(info)
      
      // 只保留最近100条记录
      if (stats.length > 100) {
        stats.splice(0, stats.length - 100)
      }
      
      localStorage.setItem('diagram-recovery-stats', JSON.stringify(stats))
    } catch {
      // 忽略存储错误
    }
  }
  
  // ====== 公共接口 ======
  
  public setErrorCallback(callback: (error: DiagramError) => void): void {
    this.onErrorCallback = callback
  }
  
  public enableErrorReporting(enabled: boolean = true): void {
    this.errorReportingEnabled = enabled
  }
  
  public getErrorLog(): ErrorInfo[] {
    return [...this.errorLog]
  }
  
  public clearErrorLog(): void {
    this.errorLog = []
  }
  
  public addRecoveryStrategy(
    errorCode: string,
    strategy: ErrorRecoveryStrategy
  ): void {
    this.recoveryStrategies.set(errorCode, strategy)
  }
  
  public removeRecoveryStrategy(errorCode: string): void {
    this.recoveryStrategies.delete(errorCode)
  }
  
  public createError(
    message: string,
    code: string,
    details?: any
  ): DiagramError {
    return new DiagramError(message, code, details)
  }
  
  // ====== 系统检查方法 ======
  
  public getSystemHealth(): {
    status: 'healthy' | 'degraded' | 'critical'
    errors: number
    lastError?: ErrorInfo
    memoryUsage?: number
  } {
    const recentErrors = this.errorLog.filter(
      error => Date.now() - error.timestamp < 60000 // 近1分钟内的错误
    )
    
    let status: 'healthy' | 'degraded' | 'critical' = 'healthy'
    
    if (recentErrors.length > 10) {
      status = 'critical'
    } else if (recentErrors.length > 3) {
      status = 'degraded'
    }
    
    const result = {
      status,
      errors: recentErrors.length,
      lastError: this.errorLog[this.errorLog.length - 1]
    }
    
    // 尝试获取内存使用情况
    if (typeof window !== 'undefined' && (performance as any).memory) {
      const memory = (performance as any).memory
      ;(result as any).memoryUsage = memory.usedJSHeapSize / memory.jsHeapSizeLimit
    }
    
    return result
  }
  
  public isSystemStable(): boolean {
    const health = this.getSystemHealth()
    return health.status === 'healthy'
  }
  
  public getRecoveryStatistics(): {
    totalRecoveries: number
    successfulRecoveries: number
    failedRecoveries: number
    recoveryRate: number
  } {
    try {
      const stats = JSON.parse(localStorage.getItem('diagram-recovery-stats') || '[]')
      const totalRecoveries = stats.length
      const successfulRecoveries = stats.filter((s: any) => s.success).length
      const failedRecoveries = totalRecoveries - successfulRecoveries
      const recoveryRate = totalRecoveries > 0 ? successfulRecoveries / totalRecoveries : 0
      
      return {
        totalRecoveries,
        successfulRecoveries,
        failedRecoveries,
        recoveryRate
      }
    } catch {
      return {
        totalRecoveries: 0,
        successfulRecoveries: 0,
        failedRecoveries: 0,
        recoveryRate: 0
      }
    }
  }
}