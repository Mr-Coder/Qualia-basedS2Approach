import AsyncStorage from '@react-native-async-storage/async-storage';

export interface OfflineData {
  version: number;
  lastSync: number;
  data: Record<string, any>;
}

const STORAGE_VERSION = 1;
const STORAGE_PREFIX = '@mathsolver_offline_';

export class OfflineStorage {
  private static instance: OfflineStorage;

  static getInstance(): OfflineStorage {
    if (!OfflineStorage.instance) {
      OfflineStorage.instance = new OfflineStorage();
    }
    return OfflineStorage.instance;
  }

  async save(key: string, data: any): Promise<void> {
    try {
      const offlineData: OfflineData = {
        version: STORAGE_VERSION,
        lastSync: Date.now(),
        data,
      };
      
      await AsyncStorage.setItem(
        `${STORAGE_PREFIX}${key}`,
        JSON.stringify(offlineData)
      );
    } catch (error) {
      console.error('Failed to save offline data:', error);
      throw error;
    }
  }

  async load<T>(key: string): Promise<T | null> {
    try {
      const stored = await AsyncStorage.getItem(`${STORAGE_PREFIX}${key}`);
      if (!stored) return null;

      const offlineData: OfflineData = JSON.parse(stored);
      
      // Check version compatibility
      if (offlineData.version !== STORAGE_VERSION) {
        console.warn('Offline data version mismatch, clearing old data');
        await this.remove(key);
        return null;
      }

      return offlineData.data as T;
    } catch (error) {
      console.error('Failed to load offline data:', error);
      return null;
    }
  }

  async remove(key: string): Promise<void> {
    try {
      await AsyncStorage.removeItem(`${STORAGE_PREFIX}${key}`);
    } catch (error) {
      console.error('Failed to remove offline data:', error);
    }
  }

  async clear(): Promise<void> {
    try {
      const allKeys = await AsyncStorage.getAllKeys();
      const offlineKeys = allKeys.filter(key => key.startsWith(STORAGE_PREFIX));
      await AsyncStorage.multiRemove(offlineKeys);
    } catch (error) {
      console.error('Failed to clear offline data:', error);
    }
  }

  async getStorageSize(): Promise<number> {
    try {
      const allKeys = await AsyncStorage.getAllKeys();
      const offlineKeys = allKeys.filter(key => key.startsWith(STORAGE_PREFIX));
      
      let totalSize = 0;
      for (const key of offlineKeys) {
        const value = await AsyncStorage.getItem(key);
        if (value) {
          totalSize += value.length;
        }
      }
      
      return totalSize;
    } catch (error) {
      console.error('Failed to calculate storage size:', error);
      return 0;
    }
  }

  async getAllKeys(): Promise<string[]> {
    try {
      const allKeys = await AsyncStorage.getAllKeys();
      return allKeys
        .filter(key => key.startsWith(STORAGE_PREFIX))
        .map(key => key.replace(STORAGE_PREFIX, ''));
    } catch (error) {
      console.error('Failed to get all keys:', error);
      return [];
    }
  }

  async getLastSyncTime(key: string): Promise<number | null> {
    try {
      const stored = await AsyncStorage.getItem(`${STORAGE_PREFIX}${key}`);
      if (!stored) return null;

      const offlineData: OfflineData = JSON.parse(stored);
      return offlineData.lastSync;
    } catch (error) {
      console.error('Failed to get last sync time:', error);
      return null;
    }
  }
}