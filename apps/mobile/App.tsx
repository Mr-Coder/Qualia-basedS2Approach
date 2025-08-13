import React, { useEffect, useRef } from 'react';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { NavigationContainerRef } from '@react-navigation/native';
import * as Notifications from 'expo-notifications';
import { AppNavigator } from './src/navigation/AppNavigator';
import { useAuthStore } from './src/stores/authStore';
import socketClient from './src/services/websocket/socketClient';
import notificationService from './src/services/notifications/notificationService';
import notificationGrouping from './src/services/notifications/notificationGrouping';
import deepLinkHandler from './src/services/navigation/deepLinkHandler';

export default function App() {
  const { checkAuthStatus, isAuthenticated } = useAuthStore();
  const navigationRef = useRef<NavigationContainerRef<any>>(null);
  const notificationListener = useRef<any>();
  const responseListener = useRef<any>();

  useEffect(() => {
    // Check auth status on app launch
    checkAuthStatus();
    
    // Initialize services
    initializeServices();

    // Set up notification listeners
    notificationListener.current = Notifications.addNotificationReceivedListener(notification => {
      notificationGrouping.addNotification(notification);
    });

    responseListener.current = Notifications.addNotificationResponseReceivedListener(response => {
      const data = response.notification.request.content.data;
      deepLinkHandler.handleNotificationDeepLink(data);
    });

    return () => {
      // Cleanup
      if (notificationListener.current) {
        Notifications.removeNotificationSubscription(notificationListener.current);
      }
      if (responseListener.current) {
        Notifications.removeNotificationSubscription(responseListener.current);
      }
      notificationService.cleanup();
      deepLinkHandler.cleanup();
    };
  }, []);

  useEffect(() => {
    // Connect socket when authenticated
    if (isAuthenticated) {
      socketClient.connect().catch(console.error);
      // Register for push notifications when authenticated
      notificationService.initialize();
    } else {
      socketClient.disconnect();
    }

    return () => {
      socketClient.disconnect();
    };
  }, [isAuthenticated]);

  const initializeServices = async () => {
    // Initialize deep linking
    await deepLinkHandler.initialize();
    
    // Set navigation ref when ready
    if (navigationRef.current) {
      deepLinkHandler.setNavigationRef(navigationRef.current);
    }
  };

  return (
    <SafeAreaProvider>
      <AppNavigator ref={navigationRef} />
      <StatusBar style="auto" />
    </SafeAreaProvider>
  );
}
