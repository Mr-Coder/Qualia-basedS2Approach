import { useEffect, useRef, useCallback } from 'react';
import { performanceMonitor } from '../services/performance/performanceMonitor';

export interface UsePerformanceTrackingOptions {
  screenName?: string;
  trackScreenLoad?: boolean;
  trackMemory?: boolean;
  memoryTrackingInterval?: number;
}

export const usePerformanceTracking = (options: UsePerformanceTrackingOptions = {}) => {
  const {
    screenName,
    trackScreenLoad = true,
    trackMemory = false,
    memoryTrackingInterval = 30000, // 30 seconds
  } = options;

  const screenLoadStartTime = useRef<number>(Date.now());
  const memoryInterval = useRef<NodeJS.Timeout | null>(null);

  // Track screen load time
  useEffect(() => {
    if (trackScreenLoad && screenName) {
      performanceMonitor.trackScreenLoad(screenName, screenLoadStartTime.current);
    }
  }, [screenName, trackScreenLoad]);

  // Track memory usage
  useEffect(() => {
    if (trackMemory) {
      // Initial memory tracking
      performanceMonitor.trackMemoryUsage();

      // Set up interval for periodic memory tracking
      memoryInterval.current = setInterval(() => {
        performanceMonitor.trackMemoryUsage();
      }, memoryTrackingInterval);

      return () => {
        if (memoryInterval.current) {
          clearInterval(memoryInterval.current);
        }
      };
    }
  }, [trackMemory, memoryTrackingInterval]);

  // Track custom metric
  const trackMetric = useCallback((
    name: string,
    value: number,
    unit: string = 'ms',
    context?: Record<string, any>
  ) => {
    performanceMonitor.recordMetric({
      name,
      value,
      unit,
      timestamp: Date.now(),
      context,
    });
  }, []);

  // Track API call
  const trackApiCall = useCallback(async <T,>(
    apiCall: () => Promise<T>,
    endpoint: string,
    method: string = 'GET'
  ): Promise<T> => {
    const startTime = Date.now();
    let status = 0;

    try {
      const result = await apiCall();
      status = 200; // Assume success if no error
      return result;
    } catch (error: any) {
      status = error.response?.status || 500;
      throw error;
    } finally {
      const duration = Date.now() - startTime;
      performanceMonitor.trackApiCall(endpoint, method, duration, status);
    }
  }, []);

  // Track gesture performance
  const trackGesture = useCallback((gestureName: string, frameRate: number) => {
    performanceMonitor.trackGesturePerformance(gestureName, frameRate);
  }, []);

  // Track WebSocket latency
  const trackWebSocketLatency = useCallback((eventType: string, latency: number) => {
    performanceMonitor.trackWebSocketLatency(eventType, latency);
  }, []);

  // Track offline sync
  const trackOfflineSync = useCallback((
    operation: string,
    itemCount: number,
    duration: number
  ) => {
    performanceMonitor.trackOfflineSync(operation, itemCount, duration);
  }, []);

  // Measure operation time
  const measureOperation = useCallback(async <T,>(
    operation: () => Promise<T> | T,
    operationName: string
  ): Promise<T> => {
    const startTime = Date.now();
    
    try {
      const result = await operation();
      const duration = Date.now() - startTime;
      
      trackMetric(`operation_${operationName}`, duration, 'ms', {
        operationName,
        success: true,
      });
      
      return result;
    } catch (error) {
      const duration = Date.now() - startTime;
      
      trackMetric(`operation_${operationName}`, duration, 'ms', {
        operationName,
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      
      throw error;
    }
  }, [trackMetric]);

  // Get performance summary
  const getPerformanceSummary = useCallback(async () => {
    return await performanceMonitor.getPerformanceSummary();
  }, []);

  return {
    trackMetric,
    trackApiCall,
    trackGesture,
    trackWebSocketLatency,
    trackOfflineSync,
    measureOperation,
    getPerformanceSummary,
  };
};