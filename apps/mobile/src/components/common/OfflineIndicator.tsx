import React, { useEffect, useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Animated,
  TouchableOpacity,
  Platform,
} from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import NetInfo, { NetInfoState } from '@react-native-community/netinfo';
import { MessageQueue } from '../../services/offline/messageQueue';
import { BackgroundSyncManager } from '../../services/offline/backgroundSync';

interface OfflineIndicatorProps {
  position?: 'top' | 'bottom';
  showDetails?: boolean;
  onPress?: () => void;
}

export const OfflineIndicator: React.FC<OfflineIndicatorProps> = ({
  position = 'top',
  showDetails = true,
  onPress,
}) => {
  const [isOnline, setIsOnline] = useState(true);
  const [queueSize, setQueueSize] = useState(0);
  const [isSyncing, setIsSyncing] = useState(false);
  const animatedValue = useRef(new Animated.Value(0)).current;
  const pulseAnimation = useRef<Animated.CompositeAnimation | null>(null);

  useEffect(() => {
    let unsubscribe: (() => void) | null = null;
    const messageQueue = MessageQueue.getInstance();

    const setupNetworkListener = async () => {
      // Get initial state
      const state = await NetInfo.fetch();
      const connected = state.isConnected && state.isInternetReachable !== false;
      setIsOnline(connected);

      // Listen for changes
      unsubscribe = NetInfo.addEventListener((state: NetInfoState) => {
        const connected = state.isConnected && state.isInternetReachable !== false;
        setIsOnline(connected);
        
        if (connected) {
          // Trigger sync when coming back online
          BackgroundSyncManager.getInstance().triggerSync();
        }
      });
    };

    const setupMessageQueueListener = () => {
      messageQueue.setCallbacks({
        onQueueChange: (size) => setQueueSize(size),
        onSyncStart: () => setIsSyncing(true),
        onSyncComplete: () => setIsSyncing(false),
        onSyncError: () => setIsSyncing(false),
      });

      // Get initial queue size
      setQueueSize(messageQueue.getQueueSize());
    };

    setupNetworkListener();
    setupMessageQueueListener();

    return () => {
      if (unsubscribe) {
        unsubscribe();
      }
      if (pulseAnimation.current) {
        pulseAnimation.current.stop();
      }
    };
  }, []);

  useEffect(() => {
    if (!isOnline || queueSize > 0) {
      // Show indicator
      Animated.timing(animatedValue, {
        toValue: 1,
        duration: 300,
        useNativeDriver: true,
      }).start();

      // Start pulse animation for offline state
      if (!isOnline) {
        const pulse = Animated.loop(
          Animated.sequence([
            Animated.timing(animatedValue, {
              toValue: 1.1,
              duration: 1000,
              useNativeDriver: true,
            }),
            Animated.timing(animatedValue, {
              toValue: 1,
              duration: 1000,
              useNativeDriver: true,
            }),
          ])
        );
        pulseAnimation.current = pulse;
        pulse.start();
      }
    } else {
      // Hide indicator
      Animated.timing(animatedValue, {
        toValue: 0,
        duration: 300,
        useNativeDriver: true,
      }).start();

      // Stop pulse animation
      if (pulseAnimation.current) {
        pulseAnimation.current.stop();
        pulseAnimation.current = null;
      }
    }
  }, [isOnline, queueSize, animatedValue]);

  const translateY = animatedValue.interpolate({
    inputRange: [0, 1],
    outputRange: position === 'top' ? [-60, 0] : [60, 0],
  });

  const getStatusText = () => {
    if (!isOnline) {
      return 'Offline';
    } else if (isSyncing) {
      return 'Syncing...';
    } else if (queueSize > 0) {
      return `${queueSize} pending`;
    }
    return 'Online';
  };

  const getStatusColor = () => {
    if (!isOnline) {
      return '#FF5252';
    } else if (isSyncing) {
      return '#FFA726';
    } else if (queueSize > 0) {
      return '#66BB6A';
    }
    return '#4CAF50';
  };

  const getStatusIcon = () => {
    if (!isOnline) {
      return 'cloud-off';
    } else if (isSyncing) {
      return 'sync';
    } else if (queueSize > 0) {
      return 'cloud-upload';
    }
    return 'cloud-done';
  };

  return (
    <Animated.View
      style={[
        styles.container,
        position === 'top' ? styles.containerTop : styles.containerBottom,
        {
          transform: [{ translateY }],
          opacity: animatedValue,
        },
      ]}
      pointerEvents={isOnline && queueSize === 0 ? 'none' : 'auto'}
    >
      <TouchableOpacity
        style={[styles.indicator, { backgroundColor: getStatusColor() }]}
        onPress={onPress}
        activeOpacity={0.8}
      >
        <MaterialIcons
          name={getStatusIcon()}
          size={20}
          color="white"
          style={isSyncing && styles.syncingIcon}
        />
        <Text style={styles.statusText}>{getStatusText()}</Text>
        
        {showDetails && !isOnline && (
          <Text style={styles.detailText}>Changes will sync when reconnected</Text>
        )}
        
        {showDetails && queueSize > 0 && isOnline && !isSyncing && (
          <Text style={styles.detailText}>Tap to sync now</Text>
        )}
      </TouchableOpacity>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    left: 0,
    right: 0,
    zIndex: 1000,
    elevation: 1000,
  },
  containerTop: {
    top: 0,
    paddingTop: Platform.OS === 'ios' ? 44 : 0,
  },
  containerBottom: {
    bottom: 0,
    paddingBottom: Platform.OS === 'ios' ? 34 : 0,
  },
  indicator: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 8,
    paddingHorizontal: 16,
    minHeight: 40,
  },
  statusText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '600',
    marginLeft: 8,
  },
  detailText: {
    color: 'rgba(255, 255, 255, 0.9)',
    fontSize: 12,
    marginLeft: 8,
  },
  syncingIcon: {
    // Animation will be applied via transforms
  },
});

// Export a hook for programmatic access to offline state
export const useOfflineState = () => {
  const [isOnline, setIsOnline] = useState(true);
  const [queueSize, setQueueSize] = useState(0);

  useEffect(() => {
    let unsubscribe: (() => void) | null = null;
    const messageQueue = MessageQueue.getInstance();

    const setup = async () => {
      const state = await NetInfo.fetch();
      setIsOnline(state.isConnected && state.isInternetReachable !== false);

      unsubscribe = NetInfo.addEventListener((state: NetInfoState) => {
        setIsOnline(state.isConnected && state.isInternetReachable !== false);
      });

      messageQueue.setCallbacks({
        onQueueChange: setQueueSize,
        onSyncStart: () => {},
        onSyncComplete: () => {},
        onSyncError: () => {},
      });

      setQueueSize(messageQueue.getQueueSize());
    };

    setup();

    return () => {
      if (unsubscribe) {
        unsubscribe();
      }
    };
  }, []);

  return { isOnline, queueSize };
};