/**
 * 协作学习引擎
 * 实现师生互动、实时同步、评论系统
 */

import {
  CollaborationUser,
  CollaborationSession,
  CollaborationEvent,
  Comment,
  Position,
  DiagramError
} from '../types/DiagramTypes'

interface WebSocketMessage {
  type: string
  sessionId: string
  userId: string
  timestamp: number
  data: any
}

interface PresenceInfo {
  cursor?: Position
  selectedEntities: string[]
  currentStep: number
  lastActivity: number
}

export class CollaborationEngine {
  private ws: WebSocket | null = null
  private currentSession: CollaborationSession | null = null
  private currentUserId: string | null = null
  private eventHandlers: Map<string, (event: CollaborationEvent) => void> = new Map()
  private presenceMap: Map<string, PresenceInfo> = new Map()
  private comments: Map<string, Comment> = new Map()
  private isConnected: boolean = false
  
  // 模拟服务器端点
  private readonly WS_ENDPOINT = 'ws://localhost:8765'
  private readonly HTTP_ENDPOINT = 'http://localhost:8765'
  
  // ====== 连接管理 ======
  
  public async connect(sessionId: string, userId: string): Promise<void> {
    try {
      this.currentUserId = userId
      
      // 在真实应用中，这里会连接到WebSocket服务器
      // 现在我们模拟连接
      await this.simulateConnection(sessionId, userId)
      
    } catch (error) {
      throw new DiagramError(
        'Failed to connect to collaboration server',
        'COLLABORATION_CONNECT_ERROR',
        error
      )
    }
  }
  
  private async simulateConnection(sessionId: string, userId: string): Promise<void> {
    // 模拟连接延迟
    await this.delay(200)
    
    this.isConnected = true
    console.log(`Connected to collaboration session: ${sessionId} as user: ${userId}`)
    
    // 模拟接收现有会话数据
    this.currentSession = {
      id: sessionId,
      title: 'Math Problem Solving Session',
      users: [
        {
          id: userId,
          name: 'Current User',
          role: 'student',
          color: this.generateUserColor(userId),
          isActive: true,
          joinTime: Date.now()
        }
      ],
      createdBy: userId,
      createdAt: Date.now(),
      status: 'active',
      permissions: {
        canEdit: [userId],
        canComment: [userId],
        canView: [userId]
      }
    }
    
    // 开始心跳机制
    this.startHeartbeat()
  }
  
  public disconnect(): void {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
    
    this.isConnected = false
    this.currentSession = null
    this.currentUserId = null
    
    console.log('Disconnected from collaboration session')
  }
  
  // ====== 会话管理 ======
  
  public async joinSession(sessionId: string, userId: string): Promise<CollaborationSession> {
    if (!this.isConnected) {
      await this.connect(sessionId, userId)
    }
    
    const joinEvent: CollaborationEvent = {
      id: this.generateId(),
      type: 'user_join',
      userId,
      timestamp: Date.now(),
      data: {
        sessionId,
        user: {
          id: userId,
          name: `User ${userId.slice(0, 6)}`,
          role: 'student',
          color: this.generateUserColor(userId),
          isActive: true,
          joinTime: Date.now()
        }
      },
      isUndoable: false
    }
    
    await this.sendEvent(joinEvent)
    
    return this.currentSession!
  }
  
  public async leaveSession(): Promise<void> {
    if (!this.currentSession || !this.currentUserId) return
    
    const leaveEvent: CollaborationEvent = {
      id: this.generateId(),
      type: 'user_leave',
      userId: this.currentUserId,
      timestamp: Date.now(),
      data: {
        sessionId: this.currentSession.id
      },
      isUndoable: false
    }
    
    await this.sendEvent(leaveEvent)
    this.disconnect()
  }
  
  // ====== 实时同步 ======
  
  public async syncEntityPosition(
    entityId: string, 
    newPosition: Position
  ): Promise<void> {
    if (!this.canEdit()) return
    
    const event: CollaborationEvent = {
      id: this.generateId(),
      type: 'entity_move',
      userId: this.currentUserId!,
      timestamp: Date.now(),
      data: {
        entityId,
        newPosition,
        oldPosition: null // 可以保存用于撤销
      },
      isUndoable: true
    }
    
    await this.sendEvent(event)
  }
  
  public async syncStepHighlight(stepIndex: number): Promise<void> {
    const event: CollaborationEvent = {
      id: this.generateId(),
      type: 'step_highlight',
      userId: this.currentUserId!,
      timestamp: Date.now(),
      data: {
        stepIndex,
        timestamp: Date.now()
      },
      isUndoable: false
    }
    
    await this.sendEvent(event)
    this.updatePresence({ currentStep: stepIndex })
  }
  
  public async syncCursorPosition(position: Position): Promise<void> {
    this.updatePresence({ cursor: position })
    
    // 节流发送光标位置更新
    this.throttledSendCursorUpdate(position)
  }
  
  private throttledSendCursorUpdate = this.throttle((position: Position) => {
    const event: CollaborationEvent = {
      id: this.generateId(),
      type: 'cursor_move',
      userId: this.currentUserId!,
      timestamp: Date.now(),
      data: { position },
      isUndoable: false
    }
    
    this.sendEvent(event)
  }, 100)
  
  // ====== 评论系统 ======
  
  public async addComment(
    content: string,
    position: Position,
    targetId?: string
  ): Promise<Comment> {
    if (!this.canComment()) {
      throw new DiagramError(
        'No permission to add comments',
        'PERMISSION_DENIED',
        { userId: this.currentUserId }
      )
    }
    
    const comment: Comment = {
      id: this.generateId(),
      authorId: this.currentUserId!,
      content,
      position,
      targetId,
      timestamp: Date.now(),
      isResolved: false,
      replies: []
    }
    
    this.comments.set(comment.id, comment)
    
    const event: CollaborationEvent = {
      id: this.generateId(),
      type: 'comment_add',
      userId: this.currentUserId!,
      timestamp: Date.now(),
      data: { comment },
      isUndoable: true
    }
    
    await this.sendEvent(event)
    return comment
  }
  
  public async resolveComment(commentId: string): Promise<void> {
    const comment = this.comments.get(commentId)
    if (!comment) return
    
    comment.isResolved = true
    
    const event: CollaborationEvent = {
      id: this.generateId(),
      type: 'comment_resolve',
      userId: this.currentUserId!,
      timestamp: Date.now(),
      data: { commentId },
      isUndoable: true
    }
    
    await this.sendEvent(event)
  }
  
  public async replyToComment(
    commentId: string,
    content: string
  ): Promise<Comment> {
    const parentComment = this.comments.get(commentId)
    if (!parentComment) {
      throw new DiagramError(
        'Parent comment not found',
        'COMMENT_NOT_FOUND',
        { commentId }
      )
    }
    
    const reply: Comment = {
      id: this.generateId(),
      authorId: this.currentUserId!,
      content,
      position: parentComment.position,
      timestamp: Date.now(),
      isResolved: false,
      replies: []
    }
    
    parentComment.replies.push(reply)
    
    const event: CollaborationEvent = {
      id: this.generateId(),
      type: 'comment_reply',
      userId: this.currentUserId!,
      timestamp: Date.now(),
      data: { commentId, reply },
      isUndoable: true
    }
    
    await this.sendEvent(event)
    return reply
  }
  
  // ====== 权限管理 ======
  
  public canEdit(): boolean {
    if (!this.currentSession || !this.currentUserId) return false
    return this.currentSession.permissions.canEdit.includes(this.currentUserId)
  }
  
  public canComment(): boolean {
    if (!this.currentSession || !this.currentUserId) return false
    return this.currentSession.permissions.canComment.includes(this.currentUserId)
  }
  
  public canView(): boolean {
    if (!this.currentSession || !this.currentUserId) return false
    return this.currentSession.permissions.canView.includes(this.currentUserId)
  }
  
  public async updatePermissions(
    userId: string,
    permissions: {
      canEdit?: boolean
      canComment?: boolean
      canView?: boolean
    }
  ): Promise<void> {
    if (!this.currentSession || !this.isOwner()) return
    
    const { canEdit, canComment, canView } = permissions
    
    if (canEdit !== undefined) {
      if (canEdit) {
        this.currentSession.permissions.canEdit.push(userId)
      } else {
        this.currentSession.permissions.canEdit = 
          this.currentSession.permissions.canEdit.filter(id => id !== userId)
      }
    }
    
    if (canComment !== undefined) {
      if (canComment) {
        this.currentSession.permissions.canComment.push(userId)
      } else {
        this.currentSession.permissions.canComment = 
          this.currentSession.permissions.canComment.filter(id => id !== userId)
      }
    }
    
    if (canView !== undefined) {
      if (canView) {
        this.currentSession.permissions.canView.push(userId)
      } else {
        this.currentSession.permissions.canView = 
          this.currentSession.permissions.canView.filter(id => id !== userId)
      }
    }
    
    const event: CollaborationEvent = {
      id: this.generateId(),
      type: 'permissions_update',
      userId: this.currentUserId!,
      timestamp: Date.now(),
      data: { userId, permissions },
      isUndoable: true
    }
    
    await this.sendEvent(event)
  }
  
  private isOwner(): boolean {
    return this.currentSession?.createdBy === this.currentUserId
  }
  
  // ====== 在线状态 ======
  
  private updatePresence(info: Partial<PresenceInfo>): void {
    if (!this.currentUserId) return
    
    const currentPresence = this.presenceMap.get(this.currentUserId) || {
      selectedEntities: [],
      currentStep: 0,
      lastActivity: Date.now()
    }
    
    const updatedPresence = {
      ...currentPresence,
      ...info,
      lastActivity: Date.now()
    }
    
    this.presenceMap.set(this.currentUserId, updatedPresence)
  }
  
  public getActiveUsers(): CollaborationUser[] {
    if (!this.currentSession) return []
    
    const now = Date.now()
    const activeThreshold = 30000 // 30秒内有活动认为在线
    
    return this.currentSession.users.filter(user => {
      const presence = this.presenceMap.get(user.id)
      return presence && (now - presence.lastActivity) < activeThreshold
    })
  }
  
  public getUserPresence(userId: string): PresenceInfo | null {
    return this.presenceMap.get(userId) || null
  }
  
  // ====== 事件处理 ======
  
  public onEvent(handler: (event: CollaborationEvent) => void): void {
    const id = this.generateId()
    this.eventHandlers.set(id, handler)
  }
  
  public offEvent(handlerId: string): void {
    this.eventHandlers.delete(handlerId)
  }
  
  private async sendEvent(event: CollaborationEvent): Promise<void> {
    // 在真实应用中，这里会通过WebSocket发送事件
    // 现在我们模拟发送并立即触发事件处理
    
    console.log('Sending collaboration event:', event)
    
    // 模拟网络延迟
    await this.delay(50)
    
    // 触发本地事件处理器
    this.eventHandlers.forEach(handler => {
      try {
        handler(event)
      } catch (error) {
        console.error('Error in collaboration event handler:', error)
      }
    })
  }
  
  private handleIncomingEvent(event: CollaborationEvent): void {
    console.log('Received collaboration event:', event)
    
    switch (event.type) {
      case 'user_join':
        this.handleUserJoin(event)
        break
      case 'user_leave':
        this.handleUserLeave(event)
        break
      case 'entity_move':
        this.handleEntityMove(event)
        break
      case 'comment_add':
        this.handleCommentAdd(event)
        break
      case 'cursor_move':
        this.handleCursorMove(event)
        break
    }
    
    // 更新用户活动状态
    this.updatePresence({ lastActivity: event.timestamp })
  }
  
  private handleUserJoin(event: CollaborationEvent): void {
    if (!this.currentSession) return
    
    const userData = event.data.user
    const existingUser = this.currentSession.users.find(u => u.id === userData.id)
    
    if (!existingUser) {
      this.currentSession.users.push(userData)
    } else {
      existingUser.isActive = true
      existingUser.joinTime = userData.joinTime
    }
  }
  
  private handleUserLeave(event: CollaborationEvent): void {
    if (!this.currentSession) return
    
    const user = this.currentSession.users.find(u => u.id === event.userId)
    if (user) {
      user.isActive = false
    }
    
    this.presenceMap.delete(event.userId)
  }
  
  private handleEntityMove(event: CollaborationEvent): void {
    // 实体移动会由主引擎处理
    console.log('Entity moved by remote user:', event.data)
  }
  
  private handleCommentAdd(event: CollaborationEvent): void {
    const comment = event.data.comment
    this.comments.set(comment.id, comment)
  }
  
  private handleCursorMove(event: CollaborationEvent): void {
    const presence = this.presenceMap.get(event.userId) || {
      selectedEntities: [],
      currentStep: 0,
      lastActivity: Date.now()
    }
    
    presence.cursor = event.data.position
    presence.lastActivity = event.timestamp
    
    this.presenceMap.set(event.userId, presence)
  }
  
  // ====== 心跳机制 ======
  
  private startHeartbeat(): void {
    setInterval(() => {
      if (this.isConnected && this.currentUserId) {
        this.updatePresence({ lastActivity: Date.now() })
      }
    }, 10000) // 每10秒发送心跳
  }
  
  // ====== 工具方法 ======
  
  private generateId(): string {
    return Date.now().toString(36) + Math.random().toString(36).substr(2)
  }
  
  private generateUserColor(userId: string): string {
    const colors = [
      '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
      '#FFEAA7', '#DDA0DD', '#FFB347', '#87CEEB'
    ]
    
    const index = userId.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0) % colors.length
    return colors[index]
  }
  
  private throttle<T extends (...args: any[]) => void>(
    func: T,
    limit: number
  ): (...args: Parameters<T>) => void {
    let inThrottle: boolean
    return function(this: any, ...args: Parameters<T>) {
      if (!inThrottle) {
        func.apply(this, args)
        inThrottle = true
        setTimeout(() => inThrottle = false, limit)
      }
    }
  }
  
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms))
  }
  
  // ====== 公共接口 ======
  
  public getCurrentSession(): CollaborationSession | null {
    return this.currentSession
  }
  
  public getAllComments(): Comment[] {
    return Array.from(this.comments.values())
  }
  
  public getCommentsByTarget(targetId: string): Comment[] {
    return Array.from(this.comments.values()).filter(comment => 
      comment.targetId === targetId
    )
  }
  
  public isConnected(): boolean {
    return this.isConnected
  }
}