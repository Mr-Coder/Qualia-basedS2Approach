import React, { useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Animated,
  TouchableOpacity,
  PanResponder,
} from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import { useTheme } from '../../hooks/useTheme';
import { scale, spacing, typography } from '../../utils/responsive';

interface Message {
  id: string;
  text: string;
  sender: string;
  timestamp: Date;
  isRead: boolean;
}

interface SwipeableMessageItemProps {
  message: Message;
  onPress: () => void;
  onDelete: () => void;
  onArchive: () => void;
  onReply: () => void;
}

export const SwipeableMessageItem: React.FC<SwipeableMessageItemProps> = ({
  message,
  onPress,
  onDelete,
  onArchive,
  onReply,
}) => {
  const { theme } = useTheme();
  const translateX = useRef(new Animated.Value(0)).current;
  const actionThreshold = scale(75);
  const maxSwipeDistance = scale(200);

  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => false,
      onMoveShouldSetPanResponder: (evt, gestureState) => {
        return Math.abs(gestureState.dx) > 5 && Math.abs(gestureState.dy) < 10;
      },
      onPanResponderMove: (evt, gestureState) => {
        let dx = gestureState.dx;
        
        // Add resistance at edges
        if (dx > maxSwipeDistance) {
          dx = maxSwipeDistance + (dx - maxSwipeDistance) * 0.3;
        } else if (dx < -maxSwipeDistance) {
          dx = -maxSwipeDistance + (dx + maxSwipeDistance) * 0.3;
        }

        translateX.setValue(dx);
      },
      onPanResponderRelease: (evt, gestureState) => {
        const { dx, vx } = gestureState;
        const absDx = Math.abs(dx);
        const shouldTriggerAction = absDx > actionThreshold || Math.abs(vx) > 0.5;

        if (shouldTriggerAction) {
          if (dx > actionThreshold) {
            // Right swipe - Archive
            Animated.timing(translateX, {
              toValue: scale(500),
              duration: 300,
              useNativeDriver: true,
            }).start(() => {
              onArchive();
              translateX.setValue(0);
            });
          } else if (dx < -actionThreshold) {
            // Left swipe - Delete
            Animated.timing(translateX, {
              toValue: scale(-500),
              duration: 300,
              useNativeDriver: true,
            }).start(() => {
              onDelete();
              translateX.setValue(0);
            });
          }
        } else {
          // Snap back
          Animated.spring(translateX, {
            toValue: 0,
            useNativeDriver: true,
            friction: 7,
          }).start();
        }
      },
    })
  ).current;

  const leftActionOpacity = translateX.interpolate({
    inputRange: [-maxSwipeDistance, -actionThreshold, 0],
    outputRange: [1, 0.8, 0],
    extrapolate: 'clamp',
  });

  const rightActionOpacity = translateX.interpolate({
    inputRange: [0, actionThreshold, maxSwipeDistance],
    outputRange: [0, 0.8, 1],
    extrapolate: 'clamp',
  });

  const leftActionScale = translateX.interpolate({
    inputRange: [-maxSwipeDistance, -actionThreshold, 0],
    outputRange: [1, 0.8, 0.5],
    extrapolate: 'clamp',
  });

  const rightActionScale = translateX.interpolate({
    inputRange: [0, actionThreshold, maxSwipeDistance],
    outputRange: [0.5, 0.8, 1],
    extrapolate: 'clamp',
  });

  return (
    <View style={styles.container}>
      {/* Background actions */}
      <View style={[styles.actionsContainer, { backgroundColor: theme.colors.background }]}>
        {/* Right action - Archive */}
        <Animated.View
          style={[
            styles.action,
            styles.rightAction,
            {
              opacity: rightActionOpacity,
              transform: [{ scale: rightActionScale }],
            },
          ]}
        >
          <MaterialIcons name="archive" size={scale(24)} color={theme.colors.primary} />
          <Text style={[styles.actionText, { color: theme.colors.primary }]}>Archive</Text>
        </Animated.View>

        {/* Left action - Delete */}
        <Animated.View
          style={[
            styles.action,
            styles.leftAction,
            {
              opacity: leftActionOpacity,
              transform: [{ scale: leftActionScale }],
            },
          ]}
        >
          <MaterialIcons name="delete" size={scale(24)} color={theme.colors.error} />
          <Text style={[styles.actionText, { color: theme.colors.error }]}>Delete</Text>
        </Animated.View>
      </View>

      {/* Swipeable message content */}
      <Animated.View
        style={[
          styles.messageContainer,
          {
            backgroundColor: theme.colors.surface,
            transform: [{ translateX }],
          },
        ]}
        {...panResponder.panHandlers}
      >
        <TouchableOpacity
          style={styles.messageTouchable}
          onPress={onPress}
          activeOpacity={0.7}
        >
          <View style={styles.messageContent}>
            <View style={styles.messageHeader}>
              <Text
                style={[
                  styles.senderName,
                  { color: theme.colors.text },
                  !message.isRead && styles.unreadText,
                ]}
              >
                {message.sender}
              </Text>
              <Text style={[styles.timestamp, { color: theme.colors.textSecondary }]}>
                {formatTime(message.timestamp)}
              </Text>
            </View>
            <Text
              style={[
                styles.messageText,
                { color: theme.colors.text },
                !message.isRead && styles.unreadText,
              ]}
              numberOfLines={2}
            >
              {message.text}
            </Text>
          </View>

          {!message.isRead && (
            <View style={[styles.unreadIndicator, { backgroundColor: theme.colors.primary }]} />
          )}
        </TouchableOpacity>

        {/* Quick reply button */}
        <TouchableOpacity
          style={[styles.replyButton, { borderLeftColor: theme.colors.border }]}
          onPress={onReply}
        >
          <MaterialIcons name="reply" size={scale(20)} color={theme.colors.textSecondary} />
        </TouchableOpacity>
      </Animated.View>
    </View>
  );
};

const formatTime = (date: Date): string => {
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);

  if (minutes < 1) return 'Just now';
  if (minutes < 60) return `${minutes}m`;
  if (hours < 24) return `${hours}h`;
  if (days < 7) return `${days}d`;
  
  return date.toLocaleDateString();
};

const styles = StyleSheet.create({
  container: {
    marginHorizontal: spacing.md,
    marginVertical: spacing.xs,
  },
  actionsContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: spacing.lg,
  },
  action: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  leftAction: {
    marginLeft: 'auto',
  },
  rightAction: {
    marginRight: 'auto',
  },
  actionText: {
    ...typography.caption,
    marginTop: spacing.xs,
  },
  messageContainer: {
    flexDirection: 'row',
    borderRadius: scale(12),
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
  },
  messageTouchable: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    padding: spacing.md,
  },
  messageContent: {
    flex: 1,
  },
  messageHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.xs,
  },
  senderName: {
    ...typography.body1,
    fontWeight: '500',
  },
  timestamp: {
    ...typography.caption,
  },
  messageText: {
    ...typography.body2,
  },
  unreadText: {
    fontWeight: '600',
  },
  unreadIndicator: {
    width: scale(8),
    height: scale(8),
    borderRadius: scale(4),
    marginLeft: spacing.sm,
  },
  replyButton: {
    paddingHorizontal: spacing.md,
    borderLeftWidth: StyleSheet.hairlineWidth,
    alignItems: 'center',
    justifyContent: 'center',
  },
});