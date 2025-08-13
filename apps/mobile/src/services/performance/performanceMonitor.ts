import { PerformanceObserver } from 'react-native-performance';
import NetInfo from '@react-native-community/netinfo';
import * as Device from 'expo-device';
import AsyncStorage from '@react-native-async-storage/async-storage';

export interface PerformanceMetric {
  name: string;
  value: number;
  unit: string;
  timestamp: number;
  context?: Record<string, any>;
}

export interface PerformanceReport {
  deviceInfo: {
    brand: string | null;
    modelName: string | null;
    osVersion: string | null;
    totalMemory: number | null;
  };
  networkInfo: {
    type: string | null;
    isConnected: boolean | null;
    isInternetReachable: boolean | null;
  };
  metrics: PerformanceMetric[];
  sessionId: string;
  timestamp: number;
}

export class PerformanceMonitor {
  private static instance: PerformanceMonitor;
  private metrics: PerformanceMetric[] = [];
  private sessionId: string;
  private observer: PerformanceObserver | null = null;
  private navigationStartTime: number = Date.now();
  private reportInterval: NodeJS.Timeout | null = null;

  private constructor() {
    this.sessionId = this.generateSessionId();
    this.initializeObserver();
    this.startReporting();
  }

  static getInstance(): PerformanceMonitor {
    if (!PerformanceMonitor.instance) {
      PerformanceMonitor.instance = new PerformanceMonitor();
    }
    return PerformanceMonitor.instance;
  }

  private generateSessionId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  private initializeObserver(): void {
    try {
      this.observer = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry) => {
          this.recordMetric({
            name: entry.name,
            value: entry.duration || entry.startTime,
            unit: 'ms',
            timestamp: Date.now(),
            context: {
              entryType: entry.entryType,
            },
          });
        });
      });

      this.observer.observe({ 
        entryTypes: ['measure', 'navigation', 'mark', 'frame', 'event'] 
      });
    } catch (error) {
      console.error('Failed to initialize performance observer:', error);
    }
  }

  /**
   * Record a custom performance metric
   */
  recordMetric(metric: PerformanceMetric): void {
    this.metrics.push(metric);

    // Keep only last 1000 metrics in memory
    if (this.metrics.length > 1000) {
      this.metrics = this.metrics.slice(-1000);
    }
  }

  /**
   * Measure time between two marks
   */
  measureTime(startMark: string, endMark: string, metricName: string): void {
    try {
      performance.mark(startMark);
      // ... some operation ...
      performance.mark(endMark);
      performance.measure(metricName, startMark, endMark);
    } catch (error) {
      console.error('Failed to measure performance:', error);
    }
  }

  /**
   * Track screen load time
   */
  trackScreenLoad(screenName: string, startTime: number): void {
    const loadTime = Date.now() - startTime;
    this.recordMetric({
      name: `screen_load_${screenName}`,
      value: loadTime,
      unit: 'ms',
      timestamp: Date.now(),
      context: { screenName },
    });
  }

  /**
   * Track API call performance
   */
  trackApiCall(endpoint: string, method: string, duration: number, status: number): void {
    this.recordMetric({
      name: 'api_call',
      value: duration,
      unit: 'ms',
      timestamp: Date.now(),
      context: {
        endpoint,
        method,
        status,
        success: status >= 200 && status < 300,
      },
    });
  }

  /**
   * Track WebSocket message latency
   */
  trackWebSocketLatency(eventType: string, latency: number): void {
    this.recordMetric({
      name: `websocket_${eventType}_latency`,
      value: latency,
      unit: 'ms',
      timestamp: Date.now(),
      context: { eventType },
    });
  }

  /**
   * Track memory usage
   */
  async trackMemoryUsage(): Promise<void> {
    if (Device.totalMemory) {
      const usedMemory = process.memoryUsage?.() || { heapUsed: 0 };
      const memoryUsagePercent = (usedMemory.heapUsed / Device.totalMemory) * 100;
      
      this.recordMetric({
        name: 'memory_usage',
        value: memoryUsagePercent,
        unit: '%',
        timestamp: Date.now(),
        context: {
          heapUsed: usedMemory.heapUsed,
          totalMemory: Device.totalMemory,
        },
      });
    }
  }

  /**
   * Track bundle size metrics
   */
  trackBundleSize(bundleName: string, size: number): void {
    this.recordMetric({
      name: `bundle_size_${bundleName}`,
      value: size,
      unit: 'bytes',
      timestamp: Date.now(),
      context: { bundleName },
    });
  }

  /**
   * Track offline sync performance
   */
  trackOfflineSync(operation: string, itemCount: number, duration: number): void {
    this.recordMetric({
      name: `offline_sync_${operation}`,
      value: duration,
      unit: 'ms',
      timestamp: Date.now(),
      context: {
        operation,
        itemCount,
        itemsPerSecond: itemCount > 0 ? (itemCount / duration) * 1000 : 0,
      },
    });
  }

  /**
   * Track gesture performance
   */
  trackGesturePerformance(gestureName: string, frameRate: number): void {
    this.recordMetric({
      name: `gesture_${gestureName}_fps`,
      value: frameRate,
      unit: 'fps',
      timestamp: Date.now(),
      context: { gestureName },
    });
  }

  /**
   * Get performance summary
   */
  async getPerformanceSummary(): Promise<PerformanceReport> {
    const [deviceInfo, networkInfo] = await Promise.all([
      this.getDeviceInfo(),
      this.getNetworkInfo(),
    ]);

    return {
      deviceInfo,
      networkInfo,
      metrics: [...this.metrics],
      sessionId: this.sessionId,
      timestamp: Date.now(),
    };
  }

  private async getDeviceInfo() {
    return {
      brand: Device.brand,
      modelName: Device.modelName,
      osVersion: Device.osVersion,
      totalMemory: Device.totalMemory,
    };
  }

  private async getNetworkInfo() {
    const netInfo = await NetInfo.fetch();
    return {
      type: netInfo.type,
      isConnected: netInfo.isConnected,
      isInternetReachable: netInfo.isInternetReachable,
    };
  }

  /**
   * Start automatic performance reporting
   */
  private startReporting(): void {
    // Report every 5 minutes
    this.reportInterval = setInterval(async () => {
      await this.sendPerformanceReport();
    }, 5 * 60 * 1000);
  }

  /**
   * Send performance report to server
   */
  private async sendPerformanceReport(): Promise<void> {
    try {
      const report = await this.getPerformanceSummary();
      
      // Store locally first
      await this.storeReportLocally(report);

      // Send to server if connected
      const netInfo = await NetInfo.fetch();
      if (netInfo.isConnected) {
        // TODO: Implement server endpoint
        // await api.post('/performance/report', report);
        
        // Clear local reports after successful send
        await this.clearLocalReports();
      }
    } catch (error) {
      console.error('Failed to send performance report:', error);
    }
  }

  private async storeReportLocally(report: PerformanceReport): Promise<void> {
    try {
      const existingReports = await AsyncStorage.getItem('performance_reports');
      const reports = existingReports ? JSON.parse(existingReports) : [];
      reports.push(report);
      
      // Keep only last 10 reports
      const recentReports = reports.slice(-10);
      await AsyncStorage.setItem('performance_reports', JSON.stringify(recentReports));
    } catch (error) {
      console.error('Failed to store performance report:', error);
    }
  }

  private async clearLocalReports(): Promise<void> {
    try {
      await AsyncStorage.removeItem('performance_reports');
    } catch (error) {
      console.error('Failed to clear local reports:', error);
    }
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    if (this.observer) {
      this.observer.disconnect();
      this.observer = null;
    }

    if (this.reportInterval) {
      clearInterval(this.reportInterval);
      this.reportInterval = null;
    }

    this.metrics = [];
  }
}

// Export singleton instance
export const performanceMonitor = PerformanceMonitor.getInstance();