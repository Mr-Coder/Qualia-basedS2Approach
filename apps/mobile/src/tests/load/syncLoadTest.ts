import { io, Socket } from 'socket.io-client';
import { performanceMonitor } from '../../services/performance/performanceMonitor';

interface LoadTestConfig {
  serverUrl: string;
  numUsers: number;
  testDuration: number; // in seconds
  messagesPerSecond: number;
  roomId: string;
}

interface LoadTestResult {
  totalMessages: number;
  successfulMessages: number;
  failedMessages: number;
  averageLatency: number;
  p95Latency: number;
  p99Latency: number;
  messagesPerSecond: number;
  errors: string[];
}

export class SyncLoadTester {
  private sockets: Socket[] = [];
  private latencies: number[] = [];
  private errors: string[] = [];
  private messagesSent = 0;
  private messagesReceived = 0;
  private startTime = 0;

  constructor(private config: LoadTestConfig) {}

  async runTest(): Promise<LoadTestResult> {
    console.log(`Starting load test with ${this.config.numUsers} users...`);
    
    this.startTime = Date.now();
    
    try {
      // Create user connections
      await this.createConnections();
      
      // Start sending messages
      await this.startMessageFlow();
      
      // Wait for test duration
      await new Promise(resolve => setTimeout(resolve, this.config.testDuration * 1000));
      
      // Stop test and collect results
      return this.collectResults();
    } finally {
      this.cleanup();
    }
  }

  private async createConnections(): Promise<void> {
    const connectionPromises = [];
    
    for (let i = 0; i < this.config.numUsers; i++) {
      connectionPromises.push(this.createUserConnection(i));
    }
    
    await Promise.all(connectionPromises);
    console.log(`Created ${this.sockets.length} connections`);
  }

  private createUserConnection(userId: number): Promise<void> {
    return new Promise((resolve, reject) => {
      const socket = io(this.config.serverUrl, {
        transports: ['websocket'],
        auth: {
          token: `test-token-${userId}`,
          userId: `load-test-user-${userId}`,
        },
      });

      socket.on('connect', () => {
        // Join room
        socket.emit('join-room', { roomId: this.config.roomId });
        
        // Listen for messages
        socket.on('message', (data) => {
          const receivedTime = Date.now();
          if (data.metadata && data.metadata.sentTime) {
            const latency = receivedTime - data.metadata.sentTime;
            this.latencies.push(latency);
            performanceMonitor.trackWebSocketLatency('message', latency);
          }
          this.messagesReceived++;
        });

        socket.on('error', (error) => {
          this.errors.push(`Socket ${userId}: ${error.message}`);
        });

        this.sockets.push(socket);
        resolve();
      });

      socket.on('connect_error', (error) => {
        reject(new Error(`Failed to connect user ${userId}: ${error.message}`));
      });

      // Timeout connection attempt
      setTimeout(() => {
        if (!socket.connected) {
          reject(new Error(`Connection timeout for user ${userId}`));
        }
      }, 10000);
    });
  }

  private async startMessageFlow(): Promise<void> {
    const messageInterval = 1000 / this.config.messagesPerSecond;
    const intervals: NodeJS.Timeout[] = [];

    // Each user sends messages at regular intervals
    this.sockets.forEach((socket, index) => {
      const interval = setInterval(() => {
        if (socket.connected) {
          const message = {
            id: `msg-${Date.now()}-${index}`,
            text: `Load test message from user ${index}`,
            roomId: this.config.roomId,
            userId: `load-test-user-${index}`,
            metadata: {
              sentTime: Date.now(),
            },
          };

          socket.emit('send-message', message);
          this.messagesSent++;
        }
      }, messageInterval * this.config.numUsers); // Distribute load across users

      intervals.push(interval);
    });

    // Clear intervals after test duration
    setTimeout(() => {
      intervals.forEach(interval => clearInterval(interval));
    }, this.config.testDuration * 1000);
  }

  private collectResults(): LoadTestResult {
    const duration = (Date.now() - this.startTime) / 1000;
    const sortedLatencies = [...this.latencies].sort((a, b) => a - b);
    
    return {
      totalMessages: this.messagesSent,
      successfulMessages: this.messagesReceived,
      failedMessages: this.messagesSent - this.messagesReceived,
      averageLatency: this.calculateAverage(this.latencies),
      p95Latency: this.calculatePercentile(sortedLatencies, 0.95),
      p99Latency: this.calculatePercentile(sortedLatencies, 0.99),
      messagesPerSecond: this.messagesSent / duration,
      errors: this.errors,
    };
  }

  private calculateAverage(values: number[]): number {
    if (values.length === 0) return 0;
    const sum = values.reduce((acc, val) => acc + val, 0);
    return Math.round(sum / values.length);
  }

  private calculatePercentile(sortedValues: number[], percentile: number): number {
    if (sortedValues.length === 0) return 0;
    const index = Math.ceil(sortedValues.length * percentile) - 1;
    return sortedValues[index] || 0;
  }

  private cleanup(): void {
    this.sockets.forEach(socket => {
      if (socket.connected) {
        socket.disconnect();
      }
    });
    this.sockets = [];
  }
}

// Load test scenarios
export const loadTestScenarios = {
  // Light load: 10 users, 1 message/second each
  light: {
    numUsers: 10,
    messagesPerSecond: 1,
    testDuration: 60,
  },
  
  // Medium load: 50 users, 2 messages/second each
  medium: {
    numUsers: 50,
    messagesPerSecond: 2,
    testDuration: 120,
  },
  
  // Heavy load: 100 users, 5 messages/second each
  heavy: {
    numUsers: 100,
    messagesPerSecond: 5,
    testDuration: 180,
  },
  
  // Stress test: 200 users, 10 messages/second each
  stress: {
    numUsers: 200,
    messagesPerSecond: 10,
    testDuration: 300,
  },
};

// Helper function to run a load test scenario
export async function runLoadTestScenario(
  scenarioName: keyof typeof loadTestScenarios,
  serverUrl: string,
  roomId: string
): Promise<LoadTestResult> {
  const scenario = loadTestScenarios[scenarioName];
  const config: LoadTestConfig = {
    serverUrl,
    roomId,
    ...scenario,
  };
  
  const tester = new SyncLoadTester(config);
  const result = await tester.runTest();
  
  // Track results in performance monitor
  performanceMonitor.recordMetric({
    name: `load_test_${scenarioName}`,
    value: result.averageLatency,
    unit: 'ms',
    timestamp: Date.now(),
    context: {
      scenario: scenarioName,
      ...result,
    },
  });
  
  return result;
}