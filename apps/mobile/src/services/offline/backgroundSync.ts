import * as BackgroundFetch from 'expo-background-fetch';
import * as TaskManager from 'expo-task-manager';
import { MessageQueue } from './messageQueue';
import { OfflineStorage } from './offlineStorage';
import { Platform } from 'react-native';

const BACKGROUND_SYNC_TASK = 'MATHSOLVER_BACKGROUND_SYNC';
const SYNC_INTERVAL = 15 * 60; // 15 minutes in seconds

export interface BackgroundSyncConfig {
  enabled: boolean;
  interval: number;
  requiresCharging: boolean;
  requiresDeviceIdle: boolean;
  requiresNetworkConnectivity: boolean;
}

export class BackgroundSyncManager {
  private static instance: BackgroundSyncManager;
  private isRegistered = false;
  private config: BackgroundSyncConfig = {
    enabled: true,
    interval: SYNC_INTERVAL,
    requiresCharging: false,
    requiresDeviceIdle: false,
    requiresNetworkConnectivity: true,
  };

  static getInstance(): BackgroundSyncManager {
    if (!BackgroundSyncManager.instance) {
      BackgroundSyncManager.instance = new BackgroundSyncManager();
    }
    return BackgroundSyncManager.instance;
  }

  async initialize(config?: Partial<BackgroundSyncConfig>) {
    if (config) {
      this.config = { ...this.config, ...config };
    }

    // Define the background task
    TaskManager.defineTask(BACKGROUND_SYNC_TASK, async () => {
      try {
        console.log('Background sync task running...');
        
        const result = await this.performBackgroundSync();
        
        return result.success
          ? BackgroundFetch.BackgroundFetchResult.NewData
          : BackgroundFetch.BackgroundFetchResult.Failed;
      } catch (error) {
        console.error('Background sync error:', error);
        return BackgroundFetch.BackgroundFetchResult.Failed;
      }
    });

    // Register the background task
    if (this.config.enabled) {
      await this.register();
    }
  }

  async register(): Promise<void> {
    if (this.isRegistered) {
      console.log('Background sync already registered');
      return;
    }

    try {
      // Check if background fetch is available
      const status = await BackgroundFetch.getStatusAsync();
      
      if (status === BackgroundFetch.BackgroundFetchStatus.Restricted) {
        console.warn('Background fetch is restricted');
        return;
      }

      if (status === BackgroundFetch.BackgroundFetchStatus.Denied) {
        console.warn('Background fetch permission denied');
        return;
      }

      // Register the task
      await BackgroundFetch.registerTaskAsync(BACKGROUND_SYNC_TASK, {
        minimumInterval: this.config.interval,
        stopOnTerminate: false,
        startOnBoot: true,
      });

      this.isRegistered = true;
      console.log('Background sync registered successfully');

      // iOS specific: Set minimum background fetch interval
      if (Platform.OS === 'ios') {
        await BackgroundFetch.setMinimumIntervalAsync(this.config.interval);
      }
    } catch (error) {
      console.error('Failed to register background sync:', error);
    }
  }

  async unregister(): Promise<void> {
    if (!this.isRegistered) {
      return;
    }

    try {
      await BackgroundFetch.unregisterTaskAsync(BACKGROUND_SYNC_TASK);
      this.isRegistered = false;
      console.log('Background sync unregistered');
    } catch (error) {
      console.error('Failed to unregister background sync:', error);
    }
  }

  async updateConfig(config: Partial<BackgroundSyncConfig>): Promise<void> {
    this.config = { ...this.config, ...config };

    if (this.config.enabled && !this.isRegistered) {
      await this.register();
    } else if (!this.config.enabled && this.isRegistered) {
      await this.unregister();
    }
  }

  private async performBackgroundSync(): Promise<{ success: boolean; syncedItems: number }> {
    try {
      const messageQueue = MessageQueue.getInstance();
      const storage = OfflineStorage.getInstance();

      // Get all pending items
      const pendingMessages = messageQueue.getQueue();
      const offlineKeys = await storage.getAllKeys();

      console.log(`Background sync: ${pendingMessages.length} messages, ${offlineKeys.length} offline items`);

      let syncedItems = 0;

      // Process message queue
      if (pendingMessages.length > 0) {
        await messageQueue.processQueue(async (message) => {
          // This is a placeholder - actual implementation would send to server
          console.log('Background sync processing message:', message.id);
          
          // Simulate network request
          await new Promise(resolve => setTimeout(resolve, 100));
          
          // Return success/failure based on some condition
          return Math.random() > 0.1; // 90% success rate for demo
        });

        syncedItems += pendingMessages.length - messageQueue.getQueueSize();
      }

      // Sync offline data
      for (const key of offlineKeys) {
        const lastSync = await storage.getLastSyncTime(key);
        
        if (lastSync && Date.now() - lastSync > this.config.interval * 1000) {
          // Data needs syncing
          const data = await storage.load(key);
          
          if (data) {
            // Placeholder for actual sync logic
            console.log(`Background sync: syncing ${key}`);
            
            // Update sync time
            await storage.save(key, data);
            syncedItems++;
          }
        }
      }

      console.log(`Background sync completed: ${syncedItems} items synced`);

      return {
        success: true,
        syncedItems,
      };
    } catch (error) {
      console.error('Background sync failed:', error);
      return {
        success: false,
        syncedItems: 0,
      };
    }
  }

  async triggerSync(): Promise<void> {
    if (!this.isRegistered) {
      console.warn('Background sync not registered');
      return;
    }

    try {
      // Manually trigger the background task
      const result = await TaskManager.getTaskOptionsAsync(BACKGROUND_SYNC_TASK);
      
      if (result) {
        await this.performBackgroundSync();
      }
    } catch (error) {
      console.error('Failed to trigger sync:', error);
    }
  }

  async getStatus(): Promise<{
    registered: boolean;
    enabled: boolean;
    lastSync?: Date;
    nextSync?: Date;
  }> {
    const isTaskRegistered = await TaskManager.isTaskRegisteredAsync(BACKGROUND_SYNC_TASK);

    return {
      registered: isTaskRegistered,
      enabled: this.config.enabled,
      // These would be tracked in a real implementation
      lastSync: undefined,
      nextSync: undefined,
    };
  }
}