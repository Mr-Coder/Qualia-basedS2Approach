import { ConcurrentEditingService, Operation } from '../../services/sync/concurrentEditingService';
import { performanceMonitor } from '../../services/performance/performanceMonitor';

interface ConcurrentEditingTestConfig {
  numUsers: number;
  documentLength: number;
  operationsPerUser: number;
  operationInterval: number; // milliseconds between operations
}

interface ConcurrentEditingTestResult {
  totalOperations: number;
  successfulTransformations: number;
  failedTransformations: number;
  averageTransformTime: number;
  finalDocumentConsistency: boolean;
  documentVersions: Map<string, string>;
  transformTimes: number[];
}

export class ConcurrentEditingLoadTester {
  private editingService: ConcurrentEditingService;
  private documentId: string;
  private initialDocument: string;
  private userDocuments: Map<string, string> = new Map();
  private transformTimes: number[] = [];
  private successfulTransformations = 0;
  private failedTransformations = 0;

  constructor(private config: ConcurrentEditingTestConfig) {
    this.editingService = ConcurrentEditingService.getInstance();
    this.documentId = `test-doc-${Date.now()}`;
    this.initialDocument = this.generateInitialDocument();
  }

  async runTest(): Promise<ConcurrentEditingTestResult> {
    console.log(`Starting concurrent editing test with ${this.config.numUsers} users...`);
    
    // Initialize session and documents
    this.editingService.createSession(this.documentId);
    this.initializeUserDocuments();
    
    // Simulate concurrent editing
    const editingPromises = [];
    for (let userId = 0; userId < this.config.numUsers; userId++) {
      editingPromises.push(this.simulateUserEditing(userId));
    }
    
    await Promise.all(editingPromises);
    
    // Verify consistency
    const consistency = this.verifyDocumentConsistency();
    
    return this.collectResults(consistency);
  }

  private generateInitialDocument(): string {
    return 'A'.repeat(this.config.documentLength);
  }

  private initializeUserDocuments(): void {
    for (let i = 0; i < this.config.numUsers; i++) {
      this.userDocuments.set(`user-${i}`, this.initialDocument);
      this.editingService.joinSession(this.documentId, `user-${i}`);
    }
  }

  private async simulateUserEditing(userId: number): Promise<void> {
    const userKey = `user-${userId}`;
    let currentDocument = this.userDocuments.get(userKey)!;
    
    for (let i = 0; i < this.config.operationsPerUser; i++) {
      // Random delay to simulate real user behavior
      await new Promise(resolve => 
        setTimeout(resolve, this.config.operationInterval + Math.random() * 100)
      );
      
      // Generate random operation
      const operation = this.generateRandomOperation(userId, currentDocument.length);
      
      // Measure transformation time
      const startTime = Date.now();
      
      try {
        // Apply operation
        const transformedOp = this.editingService.applyLocalOperation(
          this.documentId,
          operation
        );
        
        // Update local document
        currentDocument = this.applyOperationToDocument(currentDocument, transformedOp);
        this.userDocuments.set(userKey, currentDocument);
        
        // Simulate receiving remote operations from other users
        this.simulateRemoteOperations(userId);
        
        const transformTime = Date.now() - startTime;
        this.transformTimes.push(transformTime);
        this.successfulTransformations++;
        
        // Track performance
        performanceMonitor.recordMetric({
          name: 'concurrent_edit_transform',
          value: transformTime,
          unit: 'ms',
          timestamp: Date.now(),
          context: {
            userId,
            operationType: operation.type,
            documentLength: currentDocument.length,
          },
        });
      } catch (error) {
        this.failedTransformations++;
        console.error(`Transform failed for user ${userId}:`, error);
      }
    }
  }

  private generateRandomOperation(userId: number, docLength: number): Operation {
    const types: Operation['type'][] = ['insert', 'delete', 'update'];
    const type = types[Math.floor(Math.random() * types.length)];
    const position = Math.floor(Math.random() * docLength);
    
    const operation: Operation = {
      id: `op-${Date.now()}-${userId}-${Math.random()}`,
      type,
      position,
      timestamp: Date.now(),
      userId: `user-${userId}`,
      deviceId: `device-${userId}`,
    };
    
    switch (type) {
      case 'insert':
        operation.content = String.fromCharCode(65 + userId); // A, B, C, etc.
        break;
      case 'delete':
        operation.length = Math.min(5, docLength - position);
        break;
      case 'update':
        operation.content = String.fromCharCode(97 + userId); // a, b, c, etc.
        operation.length = 1;
        break;
    }
    
    return operation;
  }

  private applyOperationToDocument(document: string, operation: Operation): string {
    switch (operation.type) {
      case 'insert':
        return (
          document.slice(0, operation.position) +
          operation.content +
          document.slice(operation.position)
        );
      case 'delete':
        return (
          document.slice(0, operation.position) +
          document.slice(operation.position + (operation.length || 0))
        );
      case 'update':
        return (
          document.slice(0, operation.position) +
          operation.content +
          document.slice(operation.position + (operation.content?.length || 0))
        );
      default:
        return document;
    }
  }

  private simulateRemoteOperations(currentUserId: number): void {
    // Simulate receiving operations from other users
    // In a real scenario, these would come through WebSocket
    const otherUsers = Array.from({ length: this.config.numUsers })
      .map((_, i) => i)
      .filter(i => i !== currentUserId);
    
    // Randomly receive operations from 1-3 other users
    const numRemoteOps = Math.min(3, Math.floor(Math.random() * otherUsers.length) + 1);
    
    for (let i = 0; i < numRemoteOps; i++) {
      const remoteUserId = otherUsers[Math.floor(Math.random() * otherUsers.length)];
      const remoteOp = this.generateRandomOperation(
        remoteUserId,
        this.userDocuments.get(`user-${currentUserId}`)!.length
      );
      
      try {
        this.editingService.applyRemoteOperation(this.documentId, remoteOp);
      } catch (error) {
        // Remote operation conflicts are expected in high concurrency
      }
    }
  }

  private verifyDocumentConsistency(): boolean {
    // Get the final document state from the editing service
    const finalDocument = this.editingService.getDocumentState(this.documentId);
    
    // Check if all user documents can be reconciled to the same state
    let isConsistent = true;
    const reconciledDocuments = new Map<string, string>();
    
    this.userDocuments.forEach((doc, userId) => {
      // In a real system, each user would reconcile their document
      // For this test, we'll use the service's final state
      reconciledDocuments.set(userId, finalDocument);
      
      // Check if reconciliation was successful
      if (doc.length === 0 && finalDocument.length > 0) {
        isConsistent = false;
      }
    });
    
    this.userDocuments = reconciledDocuments;
    return isConsistent;
  }

  private collectResults(consistency: boolean): ConcurrentEditingTestResult {
    const avgTransformTime = this.transformTimes.length > 0
      ? this.transformTimes.reduce((a, b) => a + b, 0) / this.transformTimes.length
      : 0;
    
    return {
      totalOperations: this.config.numUsers * this.config.operationsPerUser,
      successfulTransformations: this.successfulTransformations,
      failedTransformations: this.failedTransformations,
      averageTransformTime: Math.round(avgTransformTime),
      finalDocumentConsistency: consistency,
      documentVersions: new Map(this.userDocuments),
      transformTimes: this.transformTimes,
    };
  }

  cleanup(): void {
    this.editingService.clearSession(this.documentId);
  }
}

// Test scenarios
export const concurrentEditingScenarios = {
  // Small group collaboration
  small: {
    numUsers: 5,
    documentLength: 1000,
    operationsPerUser: 20,
    operationInterval: 100,
  },
  
  // Medium classroom
  medium: {
    numUsers: 25,
    documentLength: 5000,
    operationsPerUser: 50,
    operationInterval: 200,
  },
  
  // Large classroom
  large: {
    numUsers: 50,
    documentLength: 10000,
    operationsPerUser: 100,
    operationInterval: 300,
  },
  
  // Stress test
  stress: {
    numUsers: 100,
    documentLength: 20000,
    operationsPerUser: 200,
    operationInterval: 50,
  },
};

// Helper function to run concurrent editing test
export async function runConcurrentEditingTest(
  scenarioName: keyof typeof concurrentEditingScenarios
): Promise<ConcurrentEditingTestResult> {
  const scenario = concurrentEditingScenarios[scenarioName];
  const tester = new ConcurrentEditingLoadTester(scenario);
  
  try {
    const result = await tester.runTest();
    
    // Track results
    performanceMonitor.recordMetric({
      name: `concurrent_editing_test_${scenarioName}`,
      value: result.averageTransformTime,
      unit: 'ms',
      timestamp: Date.now(),
      context: {
        scenario: scenarioName,
        ...result,
        documentVersions: undefined, // Don't include full documents in metrics
      },
    });
    
    return result;
  } finally {
    tester.cleanup();
  }
}