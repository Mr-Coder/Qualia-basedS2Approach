import AsyncStorage from '@react-native-async-storage/async-storage';
import { OfflineStorage } from '../../../src/services/offline/offlineStorage';

// Mock AsyncStorage
jest.mock('@react-native-async-storage/async-storage', () => ({
  setItem: jest.fn(() => Promise.resolve()),
  getItem: jest.fn(() => Promise.resolve(null)),
  removeItem: jest.fn(() => Promise.resolve()),
  getAllKeys: jest.fn(() => Promise.resolve([])),
  multiRemove: jest.fn(() => Promise.resolve()),
}));

describe('OfflineStorage', () => {
  let storage: OfflineStorage;
  const mockAsyncStorage = AsyncStorage as jest.Mocked<typeof AsyncStorage>;

  beforeEach(() => {
    jest.clearAllMocks();
    storage = OfflineStorage.getInstance();
  });

  describe('save', () => {
    it('should save data with metadata', async () => {
      const testData = { message: 'Hello, World!' };
      await storage.save('test_key', testData);

      expect(mockAsyncStorage.setItem).toHaveBeenCalledWith(
        '@mathsolver_offline_test_key',
        expect.stringContaining('"message":"Hello, World!"')
      );

      const savedData = JSON.parse(mockAsyncStorage.setItem.mock.calls[0][1]);
      expect(savedData).toMatchObject({
        version: 1,
        data: testData,
      });
      expect(savedData.lastSync).toBeDefined();
    });

    it('should handle save errors', async () => {
      mockAsyncStorage.setItem.mockRejectedValueOnce(new Error('Storage error'));

      await expect(storage.save('test_key', {})).rejects.toThrow('Storage error');
    });
  });

  describe('load', () => {
    it('should load saved data', async () => {
      const testData = { message: 'Test' };
      const storedData = {
        version: 1,
        lastSync: Date.now(),
        data: testData,
      };

      mockAsyncStorage.getItem.mockResolvedValueOnce(JSON.stringify(storedData));

      const loaded = await storage.load('test_key');
      expect(loaded).toEqual(testData);
      expect(mockAsyncStorage.getItem).toHaveBeenCalledWith('@mathsolver_offline_test_key');
    });

    it('should return null for non-existent data', async () => {
      mockAsyncStorage.getItem.mockResolvedValueOnce(null);

      const loaded = await storage.load('missing_key');
      expect(loaded).toBeNull();
    });

    it('should handle version mismatch', async () => {
      const oldData = {
        version: 0, // Old version
        lastSync: Date.now(),
        data: { old: 'data' },
      };

      mockAsyncStorage.getItem.mockResolvedValueOnce(JSON.stringify(oldData));

      const loaded = await storage.load('old_key');
      expect(loaded).toBeNull();
      expect(mockAsyncStorage.removeItem).toHaveBeenCalledWith('@mathsolver_offline_old_key');
    });

    it('should handle corrupted data', async () => {
      mockAsyncStorage.getItem.mockResolvedValueOnce('corrupted json');

      const loaded = await storage.load('corrupted_key');
      expect(loaded).toBeNull();
    });
  });

  describe('remove', () => {
    it('should remove stored data', async () => {
      await storage.remove('test_key');

      expect(mockAsyncStorage.removeItem).toHaveBeenCalledWith('@mathsolver_offline_test_key');
    });

    it('should handle remove errors gracefully', async () => {
      mockAsyncStorage.removeItem.mockRejectedValueOnce(new Error('Remove error'));

      // Should not throw
      await expect(storage.remove('test_key')).resolves.not.toThrow();
    });
  });

  describe('clear', () => {
    it('should clear all offline data', async () => {
      mockAsyncStorage.getAllKeys.mockResolvedValueOnce([
        '@mathsolver_offline_key1',
        '@mathsolver_offline_key2',
        'other_key',
      ]);

      await storage.clear();

      expect(mockAsyncStorage.multiRemove).toHaveBeenCalledWith([
        '@mathsolver_offline_key1',
        '@mathsolver_offline_key2',
      ]);
    });

    it('should handle clear errors gracefully', async () => {
      mockAsyncStorage.getAllKeys.mockRejectedValueOnce(new Error('Clear error'));

      await expect(storage.clear()).resolves.not.toThrow();
    });
  });

  describe('getStorageSize', () => {
    it('should calculate total storage size', async () => {
      mockAsyncStorage.getAllKeys.mockResolvedValueOnce([
        '@mathsolver_offline_key1',
        '@mathsolver_offline_key2',
      ]);

      mockAsyncStorage.getItem
        .mockResolvedValueOnce('{"data": "test1"}') // 18 chars
        .mockResolvedValueOnce('{"data": "test2"}'); // 18 chars

      const size = await storage.getStorageSize();
      expect(size).toBe(36);
    });

    it('should handle null values', async () => {
      mockAsyncStorage.getAllKeys.mockResolvedValueOnce(['@mathsolver_offline_key1']);
      mockAsyncStorage.getItem.mockResolvedValueOnce(null);

      const size = await storage.getStorageSize();
      expect(size).toBe(0);
    });
  });

  describe('getAllKeys', () => {
    it('should return all offline keys without prefix', async () => {
      mockAsyncStorage.getAllKeys.mockResolvedValueOnce([
        '@mathsolver_offline_messages',
        '@mathsolver_offline_whiteboard',
        'other_app_key',
      ]);

      const keys = await storage.getAllKeys();
      expect(keys).toEqual(['messages', 'whiteboard']);
    });
  });

  describe('getLastSyncTime', () => {
    it('should return last sync timestamp', async () => {
      const timestamp = Date.now();
      const storedData = {
        version: 1,
        lastSync: timestamp,
        data: {},
      };

      mockAsyncStorage.getItem.mockResolvedValueOnce(JSON.stringify(storedData));

      const lastSync = await storage.getLastSyncTime('test_key');
      expect(lastSync).toBe(timestamp);
    });

    it('should return null for non-existent data', async () => {
      mockAsyncStorage.getItem.mockResolvedValueOnce(null);

      const lastSync = await storage.getLastSyncTime('missing_key');
      expect(lastSync).toBeNull();
    });
  });
});