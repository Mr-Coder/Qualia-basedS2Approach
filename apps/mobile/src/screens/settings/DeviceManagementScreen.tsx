import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  RefreshControl,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { MaterialIcons, MaterialCommunityIcons } from '@expo/vector-icons';
import { useTheme } from '../../hooks/useTheme';
import { ResponsiveLayout } from '../../components/layout/ResponsiveLayout';
import { spacing, typography, scale } from '../../utils/responsive';
import { StateSyncService, DeviceInfo } from '../../services/sync/stateSyncService';
import { SessionHandoffService, HandoffSession } from '../../services/sync/sessionHandoffService';
import { generateDeviceId } from '../../utils/crypto';

export const DeviceManagementScreen: React.FC = () => {
  const { theme } = useTheme();
  const [devices, setDevices] = useState<DeviceInfo[]>([]);
  const [currentDevice, setCurrentDevice] = useState<DeviceInfo | null>(null);
  const [pendingHandoffs, setPendingHandoffs] = useState<HandoffSession[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const stateSyncService = StateSyncService.getInstance();
  const handoffService = SessionHandoffService.getInstance();

  useEffect(() => {
    loadDevices();
    loadPendingHandoffs();
    setupHandoffCallbacks();
  }, []);

  const loadDevices = async () => {
    try {
      const activeDevices = stateSyncService.getDevices();
      setDevices(activeDevices);

      // Get current device info
      const DeviceInfo = await import('react-native-device-info');
      const { Platform } = await import('react-native');
      
      const current: DeviceInfo = {
        id: await DeviceInfo.default.getUniqueId(),
        name: await DeviceInfo.default.getDeviceName(),
        platform: Platform.OS as 'ios' | 'android',
        lastSeen: Date.now(),
        isActive: true,
        capabilities: ['chat', 'whiteboard', 'video', 'offline'],
      };
      
      setCurrentDevice(current);
    } catch (error) {
      console.error('Failed to load devices:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const loadPendingHandoffs = async () => {
    try {
      const pending = await handoffService.checkPendingHandoffs();
      setPendingHandoffs(pending);
    } catch (error) {
      console.error('Failed to load pending handoffs:', error);
    }
  };

  const setupHandoffCallbacks = () => {
    handoffService.setCallbacks({
      onHandoffInitiated: (session) => {
        setPendingHandoffs(prev => [...prev, session]);
      },
      onHandoffReceived: (session) => {
        Alert.alert(
          'Session Handoff',
          `${session.fromDevice.name} wants to continue their session on this device`,
          [
            {
              text: 'Accept',
              onPress: () => acceptHandoff(session),
            },
            {
              text: 'Reject',
              style: 'cancel',
              onPress: () => rejectHandoff(session.id),
            },
          ]
        );
      },
      onHandoffCompleted: (session) => {
        setPendingHandoffs(prev => prev.filter(h => h.id !== session.id));
        Alert.alert('Success', 'Session handoff completed successfully');
      },
      onHandoffFailed: (error) => {
        Alert.alert('Handoff Failed', error.message);
      },
    });
  };

  const onRefresh = async () => {
    setIsRefreshing(true);
    await Promise.all([loadDevices(), loadPendingHandoffs()]);
    setIsRefreshing(false);
  };

  const initiateHandoff = async (device: DeviceInfo) => {
    try {
      const currentState = stateSyncService.getCurrentState();
      await handoffService.initiateHandoff(device, currentState, {
        requireConfirmation: true,
        immediate: false,
      });
    } catch (error: any) {
      Alert.alert('Error', error.message);
    }
  };

  const acceptHandoff = async (session: HandoffSession) => {
    try {
      await handoffService.acceptHandoff(session.id, session.token);
    } catch (error: any) {
      Alert.alert('Error', error.message);
    }
  };

  const rejectHandoff = async (sessionId: string) => {
    try {
      await handoffService.rejectHandoff(sessionId);
      setPendingHandoffs(prev => prev.filter(h => h.id !== sessionId));
    } catch (error: any) {
      Alert.alert('Error', error.message);
    }
  };

  const getDeviceIcon = (platform: string) => {
    switch (platform) {
      case 'ios':
        return 'phone-iphone';
      case 'android':
        return 'phone-android';
      case 'web':
        return 'computer';
      default:
        return 'devices';
    }
  };

  const formatLastSeen = (timestamp: number) => {
    const now = Date.now();
    const diff = now - timestamp;
    const minutes = Math.floor(diff / 60000);

    if (minutes < 1) return 'Active now';
    if (minutes < 60) return `${minutes}m ago`;
    
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  };

  if (isLoading) {
    return (
      <View style={[styles.loadingContainer, { backgroundColor: theme.colors.background }]}>
        <ActivityIndicator size="large" color={theme.colors.primary} />
      </View>
    );
  }

  return (
    <ScrollView
      style={[styles.container, { backgroundColor: theme.colors.background }]}
      refreshControl={
        <RefreshControl
          refreshing={isRefreshing}
          onRefresh={onRefresh}
          tintColor={theme.colors.primary}
        />
      }
    >
      <ResponsiveLayout padded>
        {/* Current Device */}
        <View style={styles.section}>
          <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>
            This Device
          </Text>
          {currentDevice && (
            <View
              style={[
                styles.deviceCard,
                styles.currentDeviceCard,
                {
                  backgroundColor: theme.colors.surface,
                  borderColor: theme.colors.primary,
                },
              ]}
            >
              <MaterialIcons
                name={getDeviceIcon(currentDevice.platform)}
                size={scale(32)}
                color={theme.colors.primary}
              />
              <View style={styles.deviceInfo}>
                <Text style={[styles.deviceName, { color: theme.colors.text }]}>
                  {currentDevice.name}
                </Text>
                <Text style={[styles.deviceStatus, { color: theme.colors.success }]}>
                  Current Device
                </Text>
              </View>
              <MaterialIcons
                name="check-circle"
                size={scale(24)}
                color={theme.colors.success}
              />
            </View>
          )}
        </View>

        {/* Active Devices */}
        <View style={styles.section}>
          <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>
            Other Devices
          </Text>
          {devices.length === 0 ? (
            <View
              style={[
                styles.emptyState,
                { backgroundColor: theme.colors.surface },
              ]}
            >
              <MaterialCommunityIcons
                name="devices"
                size={scale(48)}
                color={theme.colors.textSecondary}
              />
              <Text style={[styles.emptyText, { color: theme.colors.textSecondary }]}>
                No other devices connected
              </Text>
            </View>
          ) : (
            devices.map((device) => (
              <TouchableOpacity
                key={device.id}
                style={[
                  styles.deviceCard,
                  {
                    backgroundColor: theme.colors.surface,
                    borderColor: theme.colors.border,
                  },
                ]}
                onPress={() => initiateHandoff(device)}
              >
                <MaterialIcons
                  name={getDeviceIcon(device.platform)}
                  size={scale(32)}
                  color={theme.colors.text}
                />
                <View style={styles.deviceInfo}>
                  <Text style={[styles.deviceName, { color: theme.colors.text }]}>
                    {device.name}
                  </Text>
                  <Text style={[styles.deviceStatus, { color: theme.colors.textSecondary }]}>
                    {device.isActive ? formatLastSeen(device.lastSeen) : 'Offline'}
                  </Text>
                </View>
                <MaterialIcons
                  name="arrow-forward"
                  size={scale(20)}
                  color={theme.colors.textSecondary}
                />
              </TouchableOpacity>
            ))
          )}
        </View>

        {/* Pending Handoffs */}
        {pendingHandoffs.length > 0 && (
          <View style={styles.section}>
            <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>
              Pending Handoffs
            </Text>
            {pendingHandoffs.map((handoff) => (
              <View
                key={handoff.id}
                style={[
                  styles.handoffCard,
                  { backgroundColor: theme.colors.warning + '20' },
                ]}
              >
                <MaterialIcons
                  name="swap-horiz"
                  size={scale(24)}
                  color={theme.colors.warning}
                />
                <View style={styles.handoffInfo}>
                  <Text style={[styles.handoffText, { color: theme.colors.text }]}>
                    From {handoff.fromDevice.name} to {handoff.toDevice.name}
                  </Text>
                  <Text style={[styles.handoffExpiry, { color: theme.colors.textSecondary }]}>
                    Expires in {Math.ceil((handoff.expiresAt - Date.now()) / 60000)} minutes
                  </Text>
                </View>
                <TouchableOpacity
                  onPress={() => rejectHandoff(handoff.id)}
                  style={styles.cancelButton}
                >
                  <MaterialIcons
                    name="close"
                    size={scale(20)}
                    color={theme.colors.error}
                  />
                </TouchableOpacity>
              </View>
            ))}
          </View>
        )}

        {/* Info Section */}
        <View
          style={[
            styles.infoSection,
            { backgroundColor: theme.colors.surfaceVariant },
          ]}
        >
          <MaterialIcons
            name="info-outline"
            size={scale(20)}
            color={theme.colors.textSecondary}
          />
          <Text style={[styles.infoText, { color: theme.colors.textSecondary }]}>
            Tap on a device to start a session handoff. Your current session state will be
            transferred securely to the selected device.
          </Text>
        </View>
      </ResponsiveLayout>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  loadingContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  section: {
    marginBottom: spacing.xl,
  },
  sectionTitle: {
    ...typography.h3,
    marginBottom: spacing.md,
  },
  deviceCard: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: spacing.md,
    borderRadius: scale(12),
    borderWidth: 1,
    marginBottom: spacing.sm,
  },
  currentDeviceCard: {
    borderWidth: 2,
  },
  deviceInfo: {
    flex: 1,
    marginLeft: spacing.md,
  },
  deviceName: {
    ...typography.body1,
    fontWeight: '500',
  },
  deviceStatus: {
    ...typography.caption,
    marginTop: spacing.xs,
  },
  emptyState: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing.xl,
    borderRadius: scale(12),
  },
  emptyText: {
    ...typography.body2,
    marginTop: spacing.md,
    textAlign: 'center',
  },
  handoffCard: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: spacing.md,
    borderRadius: scale(8),
    marginBottom: spacing.sm,
  },
  handoffInfo: {
    flex: 1,
    marginLeft: spacing.sm,
  },
  handoffText: {
    ...typography.body2,
    fontWeight: '500',
  },
  handoffExpiry: {
    ...typography.caption,
    marginTop: spacing.xs,
  },
  cancelButton: {
    padding: spacing.xs,
  },
  infoSection: {
    flexDirection: 'row',
    padding: spacing.md,
    borderRadius: scale(8),
    marginTop: spacing.md,
  },
  infoText: {
    ...typography.caption,
    marginLeft: spacing.sm,
    flex: 1,
  },
});