import React, { useEffect, useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Animated,
  TouchableOpacity,
} from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import { useTheme } from '../../hooks/useTheme';
import { scale, spacing, typography } from '../../utils/responsive';
import { StateSyncService } from '../../services/sync/stateSyncService';

interface SyncStatusIndicatorProps {
  showDetails?: boolean;
  position?: 'inline' | 'floating';
  onPress?: () => void;
}

type SyncStatus = 'synced' | 'syncing' | 'pending' | 'error' | 'offline';

export const SyncStatusIndicator: React.FC<SyncStatusIndicatorProps> = ({
  showDetails = true,
  position = 'inline',
  onPress,
}) => {
  const { theme } = useTheme();
  const [syncStatus, setSyncStatus] = useState<SyncStatus>('synced');
  const [pendingChanges, setPendingChanges] = useState(0);
  const [lastSyncTime, setLastSyncTime] = useState<Date | null>(null);
  const [connectedDevices, setConnectedDevices] = useState(0);
  
  const rotateAnimation = useRef(new Animated.Value(0)).current;
  const pulseAnimation = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    const stateSyncService = StateSyncService.getInstance();

    // Set up sync callbacks
    stateSyncService.initialize('', '', {
      onStateUpdate: () => {
        setSyncStatus('synced');
        setLastSyncTime(new Date());
        setPendingChanges(0);
      },
      onDeviceUpdate: (devices) => {
        setConnectedDevices(devices.length);
      },
      onHandoffRequest: () => {},
      onSyncComplete: () => {
        setSyncStatus('synced');
        setLastSyncTime(new Date());
        setPendingChanges(0);
      },
      onSyncError: () => {
        setSyncStatus('error');
      },
      onConnectionChange: (connected) => {
        if (!connected) {
          setSyncStatus('offline');
        } else if (pendingChanges > 0) {
          setSyncStatus('pending');
        } else {
          setSyncStatus('synced');
        }
      },
    });

    return () => {
      // Cleanup
    };
  }, [pendingChanges]);

  useEffect(() => {
    // Animate syncing icon
    if (syncStatus === 'syncing') {
      const rotation = Animated.loop(
        Animated.timing(rotateAnimation, {
          toValue: 1,
          duration: 1000,
          useNativeDriver: true,
        })
      );
      rotation.start();
      return () => rotation.stop();
    } else {
      rotateAnimation.setValue(0);
    }
  }, [syncStatus, rotateAnimation]);

  useEffect(() => {
    // Pulse animation for pending status
    if (syncStatus === 'pending') {
      const pulse = Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnimation, {
            toValue: 1.2,
            duration: 1000,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnimation, {
            toValue: 1,
            duration: 1000,
            useNativeDriver: true,
          }),
        ])
      );
      pulse.start();
      return () => pulse.stop();
    } else {
      pulseAnimation.setValue(1);
    }
  }, [syncStatus, pulseAnimation]);

  const getStatusIcon = () => {
    switch (syncStatus) {
      case 'synced':
        return 'cloud-done';
      case 'syncing':
        return 'sync';
      case 'pending':
        return 'cloud-upload';
      case 'error':
        return 'cloud-off';
      case 'offline':
        return 'cloud-off';
      default:
        return 'cloud';
    }
  };

  const getStatusColor = () => {
    switch (syncStatus) {
      case 'synced':
        return theme.colors.success;
      case 'syncing':
        return theme.colors.primary;
      case 'pending':
        return theme.colors.warning;
      case 'error':
        return theme.colors.error;
      case 'offline':
        return theme.colors.textSecondary;
      default:
        return theme.colors.text;
    }
  };

  const getStatusText = () => {
    switch (syncStatus) {
      case 'synced':
        return 'All changes synced';
      case 'syncing':
        return 'Syncing...';
      case 'pending':
        return `${pendingChanges} changes pending`;
      case 'error':
        return 'Sync error';
      case 'offline':
        return 'Offline';
      default:
        return 'Unknown';
    }
  };

  const formatLastSync = () => {
    if (!lastSyncTime) return 'Never';

    const now = new Date();
    const diff = now.getTime() - lastSyncTime.getTime();
    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (seconds < 60) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    
    return lastSyncTime.toLocaleDateString();
  };

  const spin = rotateAnimation.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '360deg'],
  });

  const content = (
    <View style={[
      styles.container,
      position === 'floating' && styles.floatingContainer,
      position === 'floating' && { backgroundColor: theme.colors.surface },
    ]}>
      <Animated.View
        style={[
          styles.iconContainer,
          syncStatus === 'syncing' && { transform: [{ rotate: spin }] },
          syncStatus === 'pending' && { transform: [{ scale: pulseAnimation }] },
        ]}
      >
        <MaterialIcons
          name={getStatusIcon()}
          size={scale(20)}
          color={getStatusColor()}
        />
      </Animated.View>

      {showDetails && (
        <View style={styles.details}>
          <Text style={[styles.statusText, { color: getStatusColor() }]}>
            {getStatusText()}
          </Text>
          <Text style={[styles.lastSyncText, { color: theme.colors.textSecondary }]}>
            Last sync: {formatLastSync()}
          </Text>
        </View>
      )}

      {connectedDevices > 0 && (
        <View style={[styles.deviceBadge, { backgroundColor: theme.colors.primary }]}>
          <Text style={[styles.deviceBadgeText, { color: theme.colors.textInverse }]}>
            {connectedDevices}
          </Text>
        </View>
      )}
    </View>
  );

  if (onPress) {
    return (
      <TouchableOpacity onPress={onPress} activeOpacity={0.7}>
        {content}
      </TouchableOpacity>
    );
  }

  return content;
};

// Mini sync indicator for toolbar/header
export const MiniSyncIndicator: React.FC = () => {
  const { theme } = useTheme();
  const [syncStatus, setSyncStatus] = useState<SyncStatus>('synced');
  const rotateAnimation = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (syncStatus === 'syncing') {
      const rotation = Animated.loop(
        Animated.timing(rotateAnimation, {
          toValue: 1,
          duration: 1000,
          useNativeDriver: true,
        })
      );
      rotation.start();
      return () => rotation.stop();
    }
  }, [syncStatus, rotateAnimation]);

  const spin = rotateAnimation.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '360deg'],
  });

  const getStatusColor = () => {
    switch (syncStatus) {
      case 'synced':
        return theme.colors.success;
      case 'syncing':
        return theme.colors.primary;
      case 'pending':
        return theme.colors.warning;
      case 'error':
        return theme.colors.error;
      case 'offline':
        return theme.colors.textSecondary;
      default:
        return theme.colors.text;
    }
  };

  return (
    <Animated.View
      style={[
        styles.miniIndicator,
        syncStatus === 'syncing' && { transform: [{ rotate: spin }] },
      ]}
    >
      <MaterialIcons
        name={syncStatus === 'syncing' ? 'sync' : 'cloud-done'}
        size={scale(16)}
        color={getStatusColor()}
      />
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: spacing.sm,
  },
  floatingContainer: {
    borderRadius: scale(20),
    paddingHorizontal: spacing.md,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 4,
  },
  iconContainer: {
    marginRight: spacing.sm,
  },
  details: {
    flex: 1,
  },
  statusText: {
    ...typography.body2,
    fontWeight: '500',
  },
  lastSyncText: {
    ...typography.caption,
    marginTop: 2,
  },
  deviceBadge: {
    width: scale(20),
    height: scale(20),
    borderRadius: scale(10),
    alignItems: 'center',
    justifyContent: 'center',
    marginLeft: spacing.sm,
  },
  deviceBadgeText: {
    ...typography.caption,
    fontSize: scale(10),
    fontWeight: '600',
  },
  miniIndicator: {
    padding: spacing.xs,
  },
});