import { useRef } from 'react';
import {
  PanResponder,
  PanResponderInstance,
  GestureResponderEvent,
  PanResponderGestureState,
} from 'react-native';

export type SwipeDirection = 'left' | 'right' | 'up' | 'down';

interface SwipeConfig {
  velocityThreshold?: number;
  directionalOffsetThreshold?: number;
  gestureIsClickThreshold?: number;
}

interface SwipeCallbacks {
  onSwipeLeft?: () => void;
  onSwipeRight?: () => void;
  onSwipeUp?: () => void;
  onSwipeDown?: () => void;
  onSwipe?: (direction: SwipeDirection) => void;
}

const defaultConfig: SwipeConfig = {
  velocityThreshold: 0.3,
  directionalOffsetThreshold: 80,
  gestureIsClickThreshold: 5,
};

export const useSwipeGesture = (
  callbacks: SwipeCallbacks,
  config: SwipeConfig = {}
): PanResponderInstance => {
  const mergedConfig = { ...defaultConfig, ...config };
  
  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => false,
      onMoveShouldSetPanResponder: (evt, gestureState) => {
        const { dx, dy } = gestureState;
        return (
          Math.abs(dx) > mergedConfig.gestureIsClickThreshold! ||
          Math.abs(dy) > mergedConfig.gestureIsClickThreshold!
        );
      },
      onPanResponderRelease: (evt, gestureState) => {
        const { dx, dy, vx, vy } = gestureState;
        const absDx = Math.abs(dx);
        const absDy = Math.abs(dy);
        const absVx = Math.abs(vx);
        const absVy = Math.abs(vy);

        // Determine if it's a horizontal or vertical swipe
        const isHorizontalSwipe = absDx > absDy;
        const isValidVelocity = isHorizontalSwipe
          ? absVx > mergedConfig.velocityThreshold!
          : absVy > mergedConfig.velocityThreshold!;
        const isValidDistance = isHorizontalSwipe
          ? absDx > mergedConfig.directionalOffsetThreshold!
          : absDy > mergedConfig.directionalOffsetThreshold!;

        if (isValidVelocity || isValidDistance) {
          let direction: SwipeDirection;

          if (isHorizontalSwipe) {
            direction = dx > 0 ? 'right' : 'left';
          } else {
            direction = dy > 0 ? 'down' : 'up';
          }

          // Call specific direction callback
          switch (direction) {
            case 'left':
              callbacks.onSwipeLeft?.();
              break;
            case 'right':
              callbacks.onSwipeRight?.();
              break;
            case 'up':
              callbacks.onSwipeUp?.();
              break;
            case 'down':
              callbacks.onSwipeDown?.();
              break;
          }

          // Call general swipe callback
          callbacks.onSwipe?.(direction);
        }
      },
    })
  ).current;

  return panResponder;
};

// Hook for swipeable list items
interface SwipeableItemConfig {
  threshold?: number;
  overshootLeft?: boolean;
  overshootRight?: boolean;
  leftActions?: Array<{ key: string; onPress: () => void }>;
  rightActions?: Array<{ key: string; onPress: () => void }>;
}

export const useSwipeableItem = (config: SwipeableItemConfig) => {
  const translateX = useRef(0);
  const actionThreshold = config.threshold || 75;

  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => false,
      onMoveShouldSetPanResponder: (evt, gestureState) => {
        return Math.abs(gestureState.dx) > 5;
      },
      onPanResponderMove: (evt, gestureState) => {
        let newTranslateX = gestureState.dx;

        // Apply resistance when overshooting is disabled
        if (!config.overshootLeft && newTranslateX > 0) {
          newTranslateX = Math.sqrt(Math.abs(newTranslateX)) * 5;
        }
        if (!config.overshootRight && newTranslateX < 0) {
          newTranslateX = -Math.sqrt(Math.abs(newTranslateX)) * 5;
        }

        translateX.current = newTranslateX;
      },
      onPanResponderRelease: (evt, gestureState) => {
        const { dx, vx } = gestureState;
        const absDx = Math.abs(dx);
        const absVx = Math.abs(vx);

        // Determine if action should be triggered
        if (absDx > actionThreshold || absVx > 0.5) {
          if (dx > 0 && config.rightActions && config.rightActions.length > 0) {
            // Trigger right action
            const actionIndex = Math.min(
              Math.floor(absDx / actionThreshold),
              config.rightActions.length - 1
            );
            config.rightActions[actionIndex].onPress();
          } else if (dx < 0 && config.leftActions && config.leftActions.length > 0) {
            // Trigger left action
            const actionIndex = Math.min(
              Math.floor(absDx / actionThreshold),
              config.leftActions.length - 1
            );
            config.leftActions[actionIndex].onPress();
          }
        }

        // Reset position
        translateX.current = 0;
      },
    })
  ).current;

  return {
    panResponder,
    translateX: translateX.current,
  };
};