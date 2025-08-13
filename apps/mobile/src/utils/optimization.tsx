import React, { memo, useCallback, useMemo } from 'react';
import { InteractionManager } from 'react-native';

// Performance optimization utilities

/**
 * Memoize expensive computations
 */
export const useMemoizedValue = <T,>(
  factory: () => T,
  deps: React.DependencyList
): T => {
  return useMemo(factory, deps);
};

/**
 * Debounce function calls
 */
export const useDebounce = <T extends (...args: any[]) => any>(
  callback: T,
  delay: number,
  deps: React.DependencyList = []
): T => {
  const timeoutRef = React.useRef<NodeJS.Timeout | null>(null);

  const debouncedCallback = useCallback(
    (...args: Parameters<T>) => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }

      timeoutRef.current = setTimeout(() => {
        callback(...args);
      }, delay);
    },
    [callback, delay, ...deps]
  ) as T;

  React.useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return debouncedCallback;
};

/**
 * Throttle function calls
 */
export const useThrottle = <T extends (...args: any[]) => any>(
  callback: T,
  delay: number,
  deps: React.DependencyList = []
): T => {
  const lastRunRef = React.useRef<number>(0);
  const timeoutRef = React.useRef<NodeJS.Timeout | null>(null);

  const throttledCallback = useCallback(
    (...args: Parameters<T>) => {
      const now = Date.now();
      const timeSinceLastRun = now - lastRunRef.current;

      if (timeSinceLastRun >= delay) {
        callback(...args);
        lastRunRef.current = now;
      } else {
        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current);
        }

        timeoutRef.current = setTimeout(() => {
          callback(...args);
          lastRunRef.current = Date.now();
        }, delay - timeSinceLastRun);
      }
    },
    [callback, delay, ...deps]
  ) as T;

  React.useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return throttledCallback;
};

/**
 * Defer heavy operations until after interactions
 */
export const useInteractionCallback = <T extends (...args: any[]) => any>(
  callback: T
): T => {
  return useCallback(
    (...args: Parameters<T>) => {
      InteractionManager.runAfterInteractions(() => {
        callback(...args);
      });
    },
    [callback]
  ) as T;
};

/**
 * Optimize list rendering with memo
 */
export interface OptimizedListItemProps<T> {
  item: T;
  index: number;
  onPress?: (item: T, index: number) => void;
}

export function createOptimizedListItem<T, P extends OptimizedListItemProps<T>>(
  Component: React.ComponentType<P>
): React.MemoExoticComponent<React.ComponentType<P>> {
  return memo(Component, (prevProps, nextProps) => {
    // Custom comparison for better performance
    return (
      prevProps.item === nextProps.item &&
      prevProps.index === nextProps.index &&
      prevProps.onPress === nextProps.onPress
    );
  });
}

/**
 * Lazy load images with placeholder
 */
export interface LazyImageProps {
  source: { uri: string };
  style?: any;
  placeholder?: React.ReactNode;
}

export const LazyImage: React.FC<LazyImageProps> = memo(({ source, style, placeholder }) => {
  const [loaded, setLoaded] = React.useState(false);
  const [error, setError] = React.useState(false);

  const Image = React.useMemo(
    () => require('react-native').Image,
    []
  );

  if (error && placeholder) {
    return <>{placeholder}</>;
  }

  return (
    <>
      {!loaded && placeholder}
      <Image
        source={source}
        style={[style, !loaded && { position: 'absolute', opacity: 0 }]}
        onLoad={() => setLoaded(true)}
        onError={() => setError(true)}
      />
    </>
  );
});

/**
 * Batch state updates for better performance
 */
export const useBatchedState = <T extends Record<string, any>>(
  initialState: T
): [T, (updates: Partial<T>) => void] => {
  const [state, setState] = React.useState(initialState);
  const pendingUpdates = React.useRef<Partial<T>>({});
  const timeoutRef = React.useRef<NodeJS.Timeout | null>(null);

  const batchedSetState = useCallback((updates: Partial<T>) => {
    pendingUpdates.current = { ...pendingUpdates.current, ...updates };

    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    timeoutRef.current = setTimeout(() => {
      setState((prevState) => ({
        ...prevState,
        ...pendingUpdates.current,
      }));
      pendingUpdates.current = {};
    }, 16); // One frame at 60fps
  }, []);

  React.useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return [state, batchedSetState];
};

/**
 * Virtualized list optimization
 */
export const getOptimizedListProps = () => ({
  removeClippedSubviews: true,
  maxToRenderPerBatch: 10,
  updateCellsBatchingPeriod: 50,
  initialNumToRender: 10,
  windowSize: 10,
  getItemLayout: undefined, // Implement if items have fixed height
  keyExtractor: (item: any, index: number) => item.id || index.toString(),
});