import * as Notifications from 'expo-notifications';
import * as Device from 'expo-device';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as SecureStore from 'expo-secure-store';
import notificationService from '../../../src/services/notifications/notificationService';

// Mock dependencies
jest.mock('expo-notifications');
jest.mock('expo-device', () => ({
  isDevice: true,
  deviceName: 'Test Device',
}));
jest.mock('@react-native-async-storage/async-storage');
jest.mock('expo-secure-store');
jest.mock('expo-constants', () => ({
  default: {
    expoConfig: {
      extra: {
        eas: {
          projectId: 'test-project-id',
        },
      },
    },
  },
}));

// Mock fetch
global.fetch = jest.fn();

describe('NotificationService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (global.fetch as jest.Mock).mockClear();
  });

  describe('initialize', () => {
    it('should request permissions and register for push notifications', async () => {
      // Mock permissions granted
      (Notifications.getPermissionsAsync as jest.Mock).mockResolvedValue({
        status: 'granted',
      });
      
      // Mock push token
      (Notifications.getExpoPushTokenAsync as jest.Mock).mockResolvedValue({
        data: 'test-push-token',
      });
      
      // Mock auth token
      (SecureStore.getItemAsync as jest.Mock).mockResolvedValue('test-auth-token');
      
      // Mock successful API call
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({}),
      });

      await notificationService.initialize();

      expect(Notifications.getPermissionsAsync).toHaveBeenCalled();
      expect(Notifications.getExpoPushTokenAsync).toHaveBeenCalled();
      expect(AsyncStorage.setItem).toHaveBeenCalledWith('pushToken', 'test-push-token');
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/notifications/register'),
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Authorization': 'Bearer test-auth-token',
          }),
        })
      );
    });

    it('should handle permission denied', async () => {
      (Notifications.getPermissionsAsync as jest.Mock).mockResolvedValue({
        status: 'denied',
      });
      (Notifications.requestPermissionsAsync as jest.Mock).mockResolvedValue({
        status: 'denied',
      });

      await notificationService.initialize();

      expect(Notifications.getExpoPushTokenAsync).not.toHaveBeenCalled();
      expect(global.fetch).not.toHaveBeenCalled();
    });
  });

  describe('scheduleLocalNotification', () => {
    it('should schedule a notification', async () => {
      const title = 'Test Notification';
      const body = 'This is a test';
      const data = { type: 'test' };

      await notificationService.scheduleLocalNotification(title, body, data);

      expect(Notifications.scheduleNotificationAsync).toHaveBeenCalledWith({
        content: {
          title,
          body,
          data,
          sound: 'default',
        },
        trigger: null,
      });
    });
  });

  describe('notification preferences', () => {
    it('should get default preferences when none saved', async () => {
      (AsyncStorage.getItem as jest.Mock).mockResolvedValue(null);

      const prefs = await notificationService.getNotificationPreferences();

      expect(prefs).toEqual({
        messages: true,
        aiResponses: true,
        collaborationRequests: true,
        gradeUpdates: true,
        sound: true,
        vibration: true,
        groupNotifications: true,
      });
    });

    it('should get saved preferences', async () => {
      const savedPrefs = {
        messages: false,
        aiResponses: true,
        collaborationRequests: true,
        gradeUpdates: false,
        sound: false,
        vibration: true,
        groupNotifications: false,
      };

      (AsyncStorage.getItem as jest.Mock).mockResolvedValue(JSON.stringify(savedPrefs));

      const prefs = await notificationService.getNotificationPreferences();

      expect(prefs).toEqual(savedPrefs);
    });

    it('should save preferences', async () => {
      const newPrefs = {
        messages: true,
        aiResponses: false,
        collaborationRequests: true,
        gradeUpdates: true,
        sound: false,
        vibration: false,
        groupNotifications: true,
      };

      await notificationService.setNotificationPreferences(newPrefs);

      expect(AsyncStorage.setItem).toHaveBeenCalledWith(
        'notificationPreferences',
        JSON.stringify(newPrefs)
      );
    });
  });

  describe('badge count', () => {
    it('should set badge count on iOS', async () => {
      Object.defineProperty(require('react-native').Platform, 'OS', {
        value: 'ios',
        configurable: true,
      });

      await notificationService.setBadgeCount(5);

      expect(Notifications.setBadgeCountAsync).toHaveBeenCalledWith(5);
    });

    it('should not set badge count on Android', async () => {
      Object.defineProperty(require('react-native').Platform, 'OS', {
        value: 'android',
        configurable: true,
      });

      await notificationService.setBadgeCount(5);

      expect(Notifications.setBadgeCountAsync).not.toHaveBeenCalled();
    });
  });
});