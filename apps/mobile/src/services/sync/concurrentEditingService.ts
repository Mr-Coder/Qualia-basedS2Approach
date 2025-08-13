import { EventEmitter } from 'events';

export interface Operation {
  id: string;
  type: 'insert' | 'delete' | 'update' | 'move';
  position: number;
  content?: string;
  length?: number;
  oldPosition?: number;
  timestamp: number;
  userId: string;
  deviceId: string;
}

export interface TransformResult {
  operation1: Operation;
  operation2: Operation;
}

export interface EditingSession {
  documentId: string;
  version: number;
  operations: Operation[];
  participants: Set<string>;
}

export class ConcurrentEditingService extends EventEmitter {
  private static instance: ConcurrentEditingService;
  private sessions: Map<string, EditingSession> = new Map();
  private pendingOperations: Map<string, Operation[]> = new Map();
  private localOperations: Map<string, Operation[]> = new Map();

  private constructor() {
    super();
  }

  static getInstance(): ConcurrentEditingService {
    if (!ConcurrentEditingService.instance) {
      ConcurrentEditingService.instance = new ConcurrentEditingService();
    }
    return ConcurrentEditingService.instance;
  }

  createSession(documentId: string): EditingSession {
    const session: EditingSession = {
      documentId,
      version: 0,
      operations: [],
      participants: new Set(),
    };

    this.sessions.set(documentId, session);
    return session;
  }

  joinSession(documentId: string, userId: string): void {
    const session = this.sessions.get(documentId);
    if (session) {
      session.participants.add(userId);
      this.emit('participant:joined', { documentId, userId });
    }
  }

  leaveSession(documentId: string, userId: string): void {
    const session = this.sessions.get(documentId);
    if (session) {
      session.participants.delete(userId);
      this.emit('participant:left', { documentId, userId });
    }
  }

  applyLocalOperation(documentId: string, operation: Operation): Operation {
    const session = this.sessions.get(documentId);
    if (!session) {
      throw new Error('Session not found');
    }

    // Store local operation
    const localOps = this.localOperations.get(documentId) || [];
    localOps.push(operation);
    this.localOperations.set(documentId, localOps);

    // Transform against pending remote operations
    const pendingOps = this.pendingOperations.get(documentId) || [];
    let transformedOp = operation;

    for (const pendingOp of pendingOps) {
      const result = this.transformOperations(transformedOp, pendingOp);
      transformedOp = result.operation1;
    }

    // Apply to session
    session.operations.push(transformedOp);
    session.version++;

    this.emit('operation:applied', { documentId, operation: transformedOp });
    return transformedOp;
  }

  applyRemoteOperation(documentId: string, operation: Operation): void {
    const session = this.sessions.get(documentId);
    if (!session) {
      throw new Error('Session not found');
    }

    // Transform against local operations
    const localOps = this.localOperations.get(documentId) || [];
    let transformedOp = operation;

    for (const localOp of localOps) {
      const result = this.transformOperations(localOp, transformedOp);
      transformedOp = result.operation2;
    }

    // Apply to session
    session.operations.push(transformedOp);
    session.version++;

    // Store as pending for future local operations
    const pendingOps = this.pendingOperations.get(documentId) || [];
    pendingOps.push(transformedOp);
    this.pendingOperations.set(documentId, pendingOps);

    this.emit('operation:remote', { documentId, operation: transformedOp });
  }

  /**
   * Operational Transformation algorithm
   * Transforms two concurrent operations to maintain consistency
   */
  transformOperations(op1: Operation, op2: Operation): TransformResult {
    // Clone operations to avoid mutation
    const transformed1 = { ...op1 };
    const transformed2 = { ...op2 };

    if (op1.type === 'insert' && op2.type === 'insert') {
      if (op1.position < op2.position) {
        transformed2.position += op1.content?.length || 0;
      } else if (op1.position > op2.position) {
        transformed1.position += op2.content?.length || 0;
      } else {
        // Same position - use timestamp to determine order
        if (op1.timestamp < op2.timestamp) {
          transformed2.position += op1.content?.length || 0;
        } else {
          transformed1.position += op2.content?.length || 0;
        }
      }
    } else if (op1.type === 'insert' && op2.type === 'delete') {
      if (op1.position <= op2.position) {
        transformed2.position += op1.content?.length || 0;
      } else if (op1.position > op2.position + (op2.length || 0)) {
        transformed1.position -= op2.length || 0;
      } else {
        // Insert is within delete range - adjust
        transformed1.position = op2.position;
      }
    } else if (op1.type === 'delete' && op2.type === 'insert') {
      if (op2.position <= op1.position) {
        transformed1.position += op2.content?.length || 0;
      } else if (op2.position > op1.position + (op1.length || 0)) {
        transformed2.position -= op1.length || 0;
      } else {
        // Insert is within delete range
        transformed2.position = op1.position;
      }
    } else if (op1.type === 'delete' && op2.type === 'delete') {
      if (op1.position + (op1.length || 0) <= op2.position) {
        transformed2.position -= op1.length || 0;
      } else if (op2.position + (op2.length || 0) <= op1.position) {
        transformed1.position -= op2.length || 0;
      } else {
        // Overlapping deletes
        const overlapStart = Math.max(op1.position, op2.position);
        const overlapEnd = Math.min(
          op1.position + (op1.length || 0),
          op2.position + (op2.length || 0)
        );
        const overlapLength = overlapEnd - overlapStart;

        if (op1.position < op2.position) {
          transformed1.length = (transformed1.length || 0) - overlapLength;
          transformed2.position = op1.position;
          transformed2.length = (transformed2.length || 0) - overlapLength;
        } else {
          transformed2.length = (transformed2.length || 0) - overlapLength;
          transformed1.position = op2.position;
          transformed1.length = (transformed1.length || 0) - overlapLength;
        }
      }
    } else if (op1.type === 'update' || op2.type === 'update') {
      // For updates, last write wins based on timestamp
      if (op1.position === op2.position && op1.timestamp > op2.timestamp) {
        // op1 wins, no transformation needed
      } else if (op1.position === op2.position && op2.timestamp > op1.timestamp) {
        // op2 wins, no transformation needed
      }
    }

    return {
      operation1: transformed1,
      operation2: transformed2,
    };
  }

  /**
   * Merge concurrent text edits using operational transformation
   */
  mergeTextEdits(
    baseText: string,
    localOps: Operation[],
    remoteOps: Operation[]
  ): string {
    let text = baseText;
    const allOps: Operation[] = [];

    // Transform and merge operations
    for (const localOp of localOps) {
      let transformedOp = localOp;
      
      for (const remoteOp of remoteOps) {
        const result = this.transformOperations(transformedOp, remoteOp);
        transformedOp = result.operation1;
      }
      
      allOps.push(transformedOp);
    }

    for (const remoteOp of remoteOps) {
      let transformedOp = remoteOp;
      
      for (const localOp of localOps) {
        const result = this.transformOperations(localOp, transformedOp);
        transformedOp = result.operation2;
      }
      
      allOps.push(transformedOp);
    }

    // Sort operations by position (descending) to apply from end to start
    allOps.sort((a, b) => b.position - a.position);

    // Apply operations to text
    for (const op of allOps) {
      switch (op.type) {
        case 'insert':
          text = text.slice(0, op.position) + op.content + text.slice(op.position);
          break;
        case 'delete':
          text = text.slice(0, op.position) + text.slice(op.position + (op.length || 0));
          break;
        case 'update':
          text = text.slice(0, op.position) + 
                 op.content + 
                 text.slice(op.position + (op.content?.length || 0));
          break;
      }
    }

    return text;
  }

  /**
   * Get the current state of a document
   */
  getDocumentState(documentId: string): string {
    const session = this.sessions.get(documentId);
    if (!session) {
      throw new Error('Session not found');
    }

    // Reconstruct document from operations
    let text = '';
    const sortedOps = [...session.operations].sort((a, b) => a.timestamp - b.timestamp);

    for (const op of sortedOps) {
      switch (op.type) {
        case 'insert':
          text = text.slice(0, op.position) + op.content + text.slice(op.position);
          break;
        case 'delete':
          text = text.slice(0, op.position) + text.slice(op.position + (op.length || 0));
          break;
        case 'update':
          text = text.slice(0, op.position) + 
                 op.content + 
                 text.slice(op.position + (op.content?.length || 0));
          break;
      }
    }

    return text;
  }

  /**
   * Acknowledge that operations have been synced
   */
  acknowledgeSync(documentId: string, version: number): void {
    const localOps = this.localOperations.get(documentId) || [];
    const pendingOps = this.pendingOperations.get(documentId) || [];

    // Remove acknowledged operations
    const remainingLocal = localOps.filter(op => op.timestamp > version);
    const remainingPending = pendingOps.filter(op => op.timestamp > version);

    this.localOperations.set(documentId, remainingLocal);
    this.pendingOperations.set(documentId, remainingPending);
  }

  clearSession(documentId: string): void {
    this.sessions.delete(documentId);
    this.localOperations.delete(documentId);
    this.pendingOperations.delete(documentId);
  }
}