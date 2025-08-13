import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  Modal,
  Switch,
} from 'react-native';
import { performanceMonitor } from '../../services/performance/performanceMonitor';
import { useTheme } from '../../hooks/useTheme';
import type { PerformanceReport } from '../../services/performance/performanceMonitor';

interface PerformanceDashboardProps {
  visible: boolean;
  onClose: () => void;
}

export const PerformanceDashboard: React.FC<PerformanceDashboardProps> = ({
  visible,
  onClose,
}) => {
  const { theme } = useTheme();
  const [report, setReport] = useState<PerformanceReport | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (visible) {
      loadReport();

      if (autoRefresh) {
        const interval = setInterval(loadReport, 2000);
        setRefreshInterval(interval);
      }
    }

    return () => {
      if (refreshInterval) {
        clearInterval(refreshInterval);
      }
    };
  }, [visible, autoRefresh]);

  const loadReport = async () => {
    const performanceReport = await performanceMonitor.getPerformanceSummary();
    setReport(performanceReport);
  };

  const getMetricsByType = (type: string) => {
    if (!report) return [];
    return report.metrics.filter((metric) => metric.name.startsWith(type));
  };

  const getAverageMetric = (metrics: typeof report.metrics) => {
    if (metrics.length === 0) return 0;
    const sum = metrics.reduce((acc, metric) => acc + metric.value, 0);
    return Math.round(sum / metrics.length);
  };

  const formatValue = (value: number, unit: string): string => {
    if (unit === 'ms') {
      if (value > 1000) {
        return `${(value / 1000).toFixed(2)}s`;
      }
      return `${Math.round(value)}ms`;
    }
    if (unit === '%') {
      return `${value.toFixed(1)}%`;
    }
    if (unit === 'bytes') {
      if (value > 1024 * 1024) {
        return `${(value / (1024 * 1024)).toFixed(2)}MB`;
      }
      if (value > 1024) {
        return `${(value / 1024).toFixed(2)}KB`;
      }
      return `${value}B`;
    }
    return `${Math.round(value)}${unit}`;
  };

  const renderMetricSection = (title: string, metrics: typeof report.metrics) => {
    if (metrics.length === 0) return null;

    return (
      <View style={[styles.section, { backgroundColor: theme.colors.surface }]}>
        <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>
          {title}
        </Text>
        {metrics.slice(-5).map((metric, index) => (
          <View key={index} style={styles.metricRow}>
            <Text style={[styles.metricName, { color: theme.colors.textSecondary }]}>
              {metric.name.replace(/_/g, ' ')}
            </Text>
            <Text style={[styles.metricValue, { color: theme.colors.primary }]}>
              {formatValue(metric.value, metric.unit)}
            </Text>
          </View>
        ))}
        {metrics.length > 1 && (
          <View style={[styles.metricRow, styles.averageRow]}>
            <Text style={[styles.metricName, { color: theme.colors.text }]}>
              Average
            </Text>
            <Text style={[styles.metricValue, { color: theme.colors.accent }]}>
              {formatValue(getAverageMetric(metrics), metrics[0].unit)}
            </Text>
          </View>
        )}
      </View>
    );
  };

  if (!report) return null;

  return (
    <Modal
      visible={visible}
      animationType="slide"
      transparent={true}
      onRequestClose={onClose}
    >
      <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
        <View style={[styles.header, { backgroundColor: theme.colors.surface }]}>
          <Text style={[styles.title, { color: theme.colors.text }]}>
            Performance Dashboard
          </Text>
          <TouchableOpacity onPress={onClose} style={styles.closeButton}>
            <Text style={[styles.closeText, { color: theme.colors.primary }]}>
              Close
            </Text>
          </TouchableOpacity>
        </View>

        <View style={styles.controls}>
          <Text style={[styles.controlLabel, { color: theme.colors.text }]}>
            Auto Refresh
          </Text>
          <Switch
            value={autoRefresh}
            onValueChange={setAutoRefresh}
            trackColor={{
              false: theme.colors.border,
              true: theme.colors.primary,
            }}
          />
        </View>

        <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
          {/* Device Info */}
          <View style={[styles.section, { backgroundColor: theme.colors.surface }]}>
            <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>
              Device Information
            </Text>
            <View style={styles.metricRow}>
              <Text style={[styles.metricName, { color: theme.colors.textSecondary }]}>
                Model
              </Text>
              <Text style={[styles.metricValue, { color: theme.colors.text }]}>
                {report.deviceInfo.brand} {report.deviceInfo.modelName}
              </Text>
            </View>
            <View style={styles.metricRow}>
              <Text style={[styles.metricName, { color: theme.colors.textSecondary }]}>
                OS Version
              </Text>
              <Text style={[styles.metricValue, { color: theme.colors.text }]}>
                {report.deviceInfo.osVersion}
              </Text>
            </View>
            <View style={styles.metricRow}>
              <Text style={[styles.metricName, { color: theme.colors.textSecondary }]}>
                Network
              </Text>
              <Text style={[styles.metricValue, { color: theme.colors.text }]}>
                {report.networkInfo.type} ({report.networkInfo.isConnected ? 'Connected' : 'Offline'})
              </Text>
            </View>
          </View>

          {/* Screen Load Times */}
          {renderMetricSection('Screen Load Times', getMetricsByType('screen_load'))}

          {/* API Performance */}
          {renderMetricSection('API Performance', getMetricsByType('api_call'))}

          {/* WebSocket Latency */}
          {renderMetricSection('WebSocket Latency', getMetricsByType('websocket'))}

          {/* Memory Usage */}
          {renderMetricSection('Memory Usage', getMetricsByType('memory_usage'))}

          {/* Gesture Performance */}
          {renderMetricSection('Gesture Performance', getMetricsByType('gesture'))}

          {/* Offline Sync */}
          {renderMetricSection('Offline Sync', getMetricsByType('offline_sync'))}

          {/* Custom Operations */}
          {renderMetricSection('Operations', getMetricsByType('operation'))}
        </ScrollView>
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    marginTop: 50,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(0, 0, 0, 0.1)',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  closeButton: {
    padding: 8,
  },
  closeText: {
    fontSize: 16,
    fontWeight: '600',
  },
  controls: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
  },
  controlLabel: {
    fontSize: 16,
  },
  content: {
    flex: 1,
    padding: 16,
  },
  section: {
    marginBottom: 16,
    padding: 16,
    borderRadius: 8,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 12,
  },
  metricRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginVertical: 4,
  },
  metricName: {
    fontSize: 14,
    flex: 1,
  },
  metricValue: {
    fontSize: 14,
    fontWeight: '500',
  },
  averageRow: {
    marginTop: 8,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: 'rgba(0, 0, 0, 0.1)',
  },
});