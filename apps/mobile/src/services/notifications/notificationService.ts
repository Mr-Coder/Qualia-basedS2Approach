import * as Notifications from 'expo-notifications';
import * as Device from 'expo-device';
import Constants from 'expo-constants';
import { Platform } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as SecureStore from 'expo-secure-store';
import { API_BASE_URL } from '../config';

// Configure notification handler
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: true,
  }),
});

interface NotificationData {
  type: 'message' | 'ai_response' | 'collaboration_request' | 'grade_update';
  roomId?: string;
  messageId?: string;
  deepLink?: string;
  senderId?: string;
  senderName?: string;
}

class NotificationService {
  private notificationListener: any;
  private responseListener: any;

  async initialize() {
    // Request permissions
    const permissionStatus = await this.requestPermissions();
    if (!permissionStatus) {
      console.log('Notification permissions denied');
      return;
    }

    // Get push token
    const token = await this.registerForPushNotifications();
    if (token) {
      await this.sendTokenToServer(token);
    }

    // Set up notification listeners
    this.setupNotificationListeners();
  }

  async requestPermissions(): Promise<boolean> {
    if (!Device.isDevice) {
      console.log('Must use physical device for Push Notifications');
      return false;
    }

    const { status: existingStatus } = await Notifications.getPermissionsAsync();
    let finalStatus = existingStatus;

    if (existingStatus !== 'granted') {
      const { status } = await Notifications.requestPermissionsAsync();
      finalStatus = status;
    }

    if (finalStatus !== 'granted') {
      return false;
    }

    return true;
  }

  async registerForPushNotifications(): Promise<string | null> {
    try {
      const projectId = Constants.expoConfig?.extra?.eas?.projectId ?? Constants.easConfig?.projectId;
      
      if (!projectId) {
        console.log('Project ID not found');
        return null;
      }

      const token = (await Notifications.getExpoPushTokenAsync({ projectId })).data;
      console.log('Push token:', token);

      // Store token locally
      await AsyncStorage.setItem('pushToken', token);
      
      return token;
    } catch (error) {
      console.error('Error getting push token:', error);
      return null;
    }
  }

  async sendTokenToServer(token: string) {
    try {
      const authToken = await SecureStore.getItemAsync('accessToken');
      if (!authToken) {
        console.log('No auth token available');
        return;
      }

      const response = await fetch(`${API_BASE_URL}/api/notifications/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`,
        },
        body: JSON.stringify({
          token,
          platform: Platform.OS,
          deviceId: Device.deviceName,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to register push token');
      }

      console.log('Push token registered with server');
    } catch (error) {
      console.error('Error sending token to server:', error);
    }
  }

  setupNotificationListeners() {
    // Handle notifications when app is in foreground
    this.notificationListener = Notifications.addNotificationReceivedListener(
      this.handleNotificationReceived
    );

    // Handle notification taps
    this.responseListener = Notifications.addNotificationResponseReceivedListener(
      this.handleNotificationResponse
    );
  }

  handleNotificationReceived = (notification: Notifications.Notification) => {
    console.log('Notification received:', notification);
    
    const data = notification.request.content.data as NotificationData;
    
    // Handle different notification types
    switch (data.type) {
      case 'message':
        // New message received - handled by chat screen if open
        break;
      case 'ai_response':
        // AI response ready - could update UI
        break;
      case 'collaboration_request':
        // Someone wants to collaborate
        break;
      case 'grade_update':
        // Grade has been updated
        break;
    }
  };

  handleNotificationResponse = (response: Notifications.NotificationResponse) => {
    console.log('Notification tapped:', response);
    
    const data = response.notification.request.content.data as NotificationData;
    
    // Handle deep linking based on notification type
    if (data.deepLink) {
      // Navigate to specific screen
      // This would be handled by navigation service
      console.log('Navigate to:', data.deepLink);
    }
  };

  async scheduleLocalNotification(
    title: string,
    body: string,
    data?: any,
    trigger?: Notifications.NotificationTriggerInput
  ) {
    await Notifications.scheduleNotificationAsync({
      content: {
        title,
        body,
        data,
        sound: 'default',
      },
      trigger: trigger || null,
    });
  }

  async cancelAllNotifications() {
    await Notifications.cancelAllScheduledNotificationsAsync();
  }

  async setBadgeCount(count: number) {
    if (Platform.OS === 'ios') {
      await Notifications.setBadgeCountAsync(count);
    }
  }

  async getNotificationPreferences() {
    try {
      const prefs = await AsyncStorage.getItem('notificationPreferences');
      return prefs ? JSON.parse(prefs) : this.getDefaultPreferences();
    } catch (error) {
      console.error('Error getting notification preferences:', error);
      return this.getDefaultPreferences();
    }
  }

  async setNotificationPreferences(preferences: any) {
    try {
      await AsyncStorage.setItem('notificationPreferences', JSON.stringify(preferences));
    } catch (error) {
      console.error('Error saving notification preferences:', error);
    }
  }

  getDefaultPreferences() {
    return {
      messages: true,
      aiResponses: true,
      collaborationRequests: true,
      gradeUpdates: true,
      sound: true,
      vibration: true,
      groupNotifications: true,
    };
  }

  cleanup() {
    if (this.notificationListener) {
      Notifications.removeNotificationSubscription(this.notificationListener);
    }
    if (this.responseListener) {
      Notifications.removeNotificationSubscription(this.responseListener);
    }
  }
}

export default new NotificationService();