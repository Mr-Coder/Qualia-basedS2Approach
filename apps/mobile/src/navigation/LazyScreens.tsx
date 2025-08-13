import React, { lazy, Suspense } from 'react';
import { View, ActivityIndicator, StyleSheet } from 'react-native';
import { useTheme } from '../hooks/useTheme';

// Lazy load screens for code splitting
export const LazyRoomsScreen = lazy(() => import('../screens/RoomsScreen'));
export const LazyChatScreen = lazy(() => import('../screens/ChatScreen'));
export const LazyWhiteboardScreen = lazy(() => import('../screens/WhiteboardScreen'));
export const LazyProfileScreen = lazy(() => import('../screens/ProfileScreen'));
export const LazyThemeSettingsScreen = lazy(() => import('../screens/settings/ThemeSettingsScreen'));
export const LazyDeviceManagementScreen = lazy(() => import('../screens/settings/DeviceManagementScreen'));
export const LazyNotificationPreferencesScreen = lazy(() => import('../screens/settings/NotificationPreferencesScreen'));

// Loading component
const ScreenLoader: React.FC = () => {
  const { theme } = useTheme();

  return (
    <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <ActivityIndicator size="large" color={theme.colors.primary} />
    </View>
  );
};

// HOC for lazy loaded screens
export const withLazyLoading = <P extends object>(
  LazyComponent: React.LazyExoticComponent<React.ComponentType<P>>
): React.FC<P> => {
  return (props: P) => (
    <Suspense fallback={<ScreenLoader />}>
      <LazyComponent {...props} />
    </Suspense>
  );
};

// Export wrapped screens
export const RoomsScreen = withLazyLoading(LazyRoomsScreen);
export const ChatScreen = withLazyLoading(LazyChatScreen);
export const WhiteboardScreen = withLazyLoading(LazyWhiteboardScreen);
export const ProfileScreen = withLazyLoading(LazyProfileScreen);
export const ThemeSettingsScreen = withLazyLoading(LazyThemeSettingsScreen);
export const DeviceManagementScreen = withLazyLoading(LazyDeviceManagementScreen);
export const NotificationPreferencesScreen = withLazyLoading(LazyNotificationPreferencesScreen);

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
});