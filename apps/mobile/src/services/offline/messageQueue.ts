import { OfflineStorage } from './offlineStorage';
import NetInfo, { NetInfoState } from '@react-native-community/netinfo';

export interface QueuedMessage {
  id: string;
  type: 'message' | 'reaction' | 'typing' | 'whiteboard' | 'ai_request';
  payload: any;
  roomId: string;
  timestamp: number;
  retryCount: number;
  maxRetries: number;
  priority: 'high' | 'normal' | 'low';
}

export interface QueuedOperation {
  id: string;
  type: string;
  payload: any;
  timestamp: number;
  retryCount: number;
}

interface MessageQueueCallbacks {
  onQueueChange: (size: number) => void;
  onSyncStart: () => void;
  onSyncComplete: (successCount: number, failureCount: number) => void;
  onSyncError: (error: Error) => void;
}

export class MessageQueue {
  private static instance: MessageQueue;
  private storage: OfflineStorage;
  private queue: QueuedMessage[] = [];
  private isOnline = true;
  private isSyncing = false;
  private callbacks: MessageQueueCallbacks | null = null;
  private syncTimer: NodeJS.Timeout | null = null;
  private netInfoUnsubscribe: (() => void) | null = null;

  private readonly QUEUE_KEY = 'message_queue';
  private readonly MAX_QUEUE_SIZE = 1000;
  private readonly SYNC_INTERVAL = 30000; // 30 seconds
  private readonly BATCH_SIZE = 50;

  private constructor() {
    this.storage = OfflineStorage.getInstance();
    this.initialize();
  }

  static getInstance(): MessageQueue {
    if (!MessageQueue.instance) {
      MessageQueue.instance = new MessageQueue();
    }
    return MessageQueue.instance;
  }

  private async initialize() {
    // Load existing queue from storage
    await this.loadQueue();

    // Monitor network connectivity
    this.netInfoUnsubscribe = NetInfo.addEventListener((state: NetInfoState) => {
      const wasOffline = !this.isOnline;
      this.isOnline = state.isConnected && state.isInternetReachable !== false;

      if (wasOffline && this.isOnline) {
        // Back online - trigger sync
        this.startSync();
      }
    });

    // Check initial network state
    const netState = await NetInfo.fetch();
    this.isOnline = netState.isConnected && netState.isInternetReachable !== false;

    // Start periodic sync if online
    if (this.isOnline) {
      this.startPeriodicSync();
    }
  }

  setCallbacks(callbacks: MessageQueueCallbacks) {
    this.callbacks = callbacks;
  }

  async enqueue(message: Omit<QueuedMessage, 'id' | 'retryCount'>): Promise<void> {
    if (this.queue.length >= this.MAX_QUEUE_SIZE) {
      // Remove oldest low-priority messages
      this.queue = this.queue
        .sort((a, b) => {
          if (a.priority !== b.priority) {
            const priorityOrder = { high: 0, normal: 1, low: 2 };
            return priorityOrder[a.priority] - priorityOrder[b.priority];
          }
          return b.timestamp - a.timestamp;
        })
        .slice(0, this.MAX_QUEUE_SIZE - 1);
    }

    const queuedMessage: QueuedMessage = {
      ...message,
      id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      retryCount: 0,
    };

    this.queue.push(queuedMessage);
    await this.saveQueue();

    this.callbacks?.onQueueChange(this.queue.length);

    // Try to send immediately if online
    if (this.isOnline && !this.isSyncing) {
      this.startSync();
    }
  }

  async processQueue(sendFunction: (message: QueuedMessage) => Promise<boolean>): Promise<void> {
    if (this.isSyncing || this.queue.length === 0) return;

    this.isSyncing = true;
    this.callbacks?.onSyncStart();

    const messages = [...this.queue].sort((a, b) => {
      // Sort by priority first, then by timestamp
      if (a.priority !== b.priority) {
        const priorityOrder = { high: 0, normal: 1, low: 2 };
        return priorityOrder[a.priority] - priorityOrder[b.priority];
      }
      return a.timestamp - b.timestamp;
    });

    let successCount = 0;
    let failureCount = 0;
    const failedMessages: QueuedMessage[] = [];

    // Process in batches
    for (let i = 0; i < messages.length; i += this.BATCH_SIZE) {
      const batch = messages.slice(i, i + this.BATCH_SIZE);
      
      await Promise.all(
        batch.map(async (message) => {
          try {
            const success = await sendFunction(message);
            
            if (success) {
              successCount++;
              // Remove from queue
              this.queue = this.queue.filter(m => m.id !== message.id);
            } else {
              message.retryCount++;
              
              if (message.retryCount >= message.maxRetries) {
                // Max retries reached, remove from queue
                this.queue = this.queue.filter(m => m.id !== message.id);
                failureCount++;
              } else {
                failedMessages.push(message);
              }
            }
          } catch (error) {
            console.error('Failed to send message:', error);
            message.retryCount++;
            
            if (message.retryCount >= message.maxRetries) {
              this.queue = this.queue.filter(m => m.id !== message.id);
              failureCount++;
            } else {
              failedMessages.push(message);
            }
          }
        })
      );
    }

    // Update queue with failed messages that haven't exceeded retry limit
    this.queue = failedMessages;
    await this.saveQueue();

    this.callbacks?.onQueueChange(this.queue.length);
    this.callbacks?.onSyncComplete(successCount, failureCount);
    
    this.isSyncing = false;
  }

  getQueueSize(): number {
    return this.queue.length;
  }

  getQueue(): QueuedMessage[] {
    return [...this.queue];
  }

  async clearQueue(): Promise<void> {
    this.queue = [];
    await this.saveQueue();
    this.callbacks?.onQueueChange(0);
  }

  private async loadQueue(): Promise<void> {
    const saved = await this.storage.load<QueuedMessage[]>(this.QUEUE_KEY);
    if (saved) {
      this.queue = saved;
      this.callbacks?.onQueueChange(this.queue.length);
    }
  }

  private async saveQueue(): Promise<void> {
    await this.storage.save(this.QUEUE_KEY, this.queue);
  }

  private startSync() {
    if (!this.isOnline || this.isSyncing) return;
    
    // Sync will be triggered by the component using this queue
    // This is just a placeholder for the sync trigger
  }

  private startPeriodicSync() {
    this.stopPeriodicSync();
    
    this.syncTimer = setInterval(() => {
      if (this.isOnline && !this.isSyncing && this.queue.length > 0) {
        this.startSync();
      }
    }, this.SYNC_INTERVAL);
  }

  private stopPeriodicSync() {
    if (this.syncTimer) {
      clearInterval(this.syncTimer);
      this.syncTimer = null;
    }
  }

  destroy() {
    this.stopPeriodicSync();
    
    if (this.netInfoUnsubscribe) {
      this.netInfoUnsubscribe();
      this.netInfoUnsubscribe = null;
    }
    
    this.callbacks = null;
  }
}