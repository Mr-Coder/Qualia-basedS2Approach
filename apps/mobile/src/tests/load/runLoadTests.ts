import { runLoadTestScenario, loadTestScenarios } from './syncLoadTest';
import { runConcurrentEditingTest, concurrentEditingScenarios } from './concurrentEditingLoadTest';
import { performanceMonitor } from '../../services/performance/performanceMonitor';
import AsyncStorage from '@react-native-async-storage/async-storage';

interface LoadTestSuite {
  name: string;
  serverUrl: string;
  roomId: string;
  scenarios: string[];
}

interface LoadTestReport {
  suiteName: string;
  startTime: Date;
  endTime: Date;
  results: {
    syncTests: Record<string, any>;
    concurrentEditingTests: Record<string, any>;
  };
  summary: {
    totalTests: number;
    passed: number;
    failed: number;
    avgLatency: number;
    maxLatency: number;
  };
}

export class LoadTestRunner {
  private results: LoadTestReport;

  constructor(private suite: LoadTestSuite) {
    this.results = {
      suiteName: suite.name,
      startTime: new Date(),
      endTime: new Date(),
      results: {
        syncTests: {},
        concurrentEditingTests: {},
      },
      summary: {
        totalTests: 0,
        passed: 0,
        failed: 0,
        avgLatency: 0,
        maxLatency: 0,
      },
    };
  }

  async runAllTests(): Promise<LoadTestReport> {
    console.log(`\n🚀 Starting Load Test Suite: ${this.suite.name}\n`);
    
    try {
      // Run sync load tests
      await this.runSyncLoadTests();
      
      // Run concurrent editing tests
      await this.runConcurrentEditingTests();
      
      // Calculate summary
      this.calculateSummary();
      
      // Save results
      await this.saveResults();
      
      this.results.endTime = new Date();
      
      // Print summary
      this.printSummary();
      
      return this.results;
    } catch (error) {
      console.error('Load test suite failed:', error);
      throw error;
    }
  }

  private async runSyncLoadTests(): Promise<void> {
    console.log('\n📊 Running WebSocket Sync Load Tests...\n');
    
    for (const scenario of this.suite.scenarios) {
      if (scenario in loadTestScenarios) {
        console.log(`Running ${scenario} sync test...`);
        
        try {
          const result = await runLoadTestScenario(
            scenario as keyof typeof loadTestScenarios,
            this.suite.serverUrl,
            this.suite.roomId
          );
          
          this.results.results.syncTests[scenario] = result;
          this.results.summary.totalTests++;
          
          if (result.failedMessages === 0) {
            this.results.summary.passed++;
            console.log(`✅ ${scenario}: Success (Avg latency: ${result.averageLatency}ms)`);
          } else {
            this.results.summary.failed++;
            console.log(`❌ ${scenario}: Failed (${result.failedMessages} messages failed)`);
          }
          
          // Update max latency
          if (result.p99Latency > this.results.summary.maxLatency) {
            this.results.summary.maxLatency = result.p99Latency;
          }
        } catch (error) {
          console.error(`❌ ${scenario} sync test failed:`, error);
          this.results.results.syncTests[scenario] = { error: error.message };
          this.results.summary.failed++;
          this.results.summary.totalTests++;
        }
        
        // Pause between tests
        await new Promise(resolve => setTimeout(resolve, 5000));
      }
    }
  }

  private async runConcurrentEditingTests(): Promise<void> {
    console.log('\n✏️ Running Concurrent Editing Load Tests...\n');
    
    for (const scenario of this.suite.scenarios) {
      if (scenario in concurrentEditingScenarios) {
        console.log(`Running ${scenario} concurrent editing test...`);
        
        try {
          const result = await runConcurrentEditingTest(
            scenario as keyof typeof concurrentEditingScenarios
          );
          
          this.results.results.concurrentEditingTests[scenario] = result;
          this.results.summary.totalTests++;
          
          if (result.finalDocumentConsistency && result.failedTransformations === 0) {
            this.results.summary.passed++;
            console.log(`✅ ${scenario}: Success (Avg transform: ${result.averageTransformTime}ms)`);
          } else {
            this.results.summary.failed++;
            console.log(`❌ ${scenario}: Failed (Consistency: ${result.finalDocumentConsistency})`);
          }
        } catch (error) {
          console.error(`❌ ${scenario} concurrent editing test failed:`, error);
          this.results.results.concurrentEditingTests[scenario] = { error: error.message };
          this.results.summary.failed++;
          this.results.summary.totalTests++;
        }
        
        // Pause between tests
        await new Promise(resolve => setTimeout(resolve, 3000));
      }
    }
  }

  private calculateSummary(): void {
    const allLatencies: number[] = [];
    
    // Collect sync test latencies
    Object.values(this.results.results.syncTests).forEach((result: any) => {
      if (result.averageLatency) {
        allLatencies.push(result.averageLatency);
      }
    });
    
    // Collect concurrent editing latencies
    Object.values(this.results.results.concurrentEditingTests).forEach((result: any) => {
      if (result.averageTransformTime) {
        allLatencies.push(result.averageTransformTime);
      }
    });
    
    // Calculate average
    if (allLatencies.length > 0) {
      const sum = allLatencies.reduce((a, b) => a + b, 0);
      this.results.summary.avgLatency = Math.round(sum / allLatencies.length);
    }
  }

  private async saveResults(): Promise<void> {
    try {
      // Save to AsyncStorage
      const key = `load_test_results_${Date.now()}`;
      await AsyncStorage.setItem(key, JSON.stringify(this.results));
      
      // Track in performance monitor
      performanceMonitor.recordMetric({
        name: 'load_test_suite_completed',
        value: this.results.summary.avgLatency,
        unit: 'ms',
        timestamp: Date.now(),
        context: {
          suiteName: this.suite.name,
          passed: this.results.summary.passed,
          failed: this.results.summary.failed,
          totalTests: this.results.summary.totalTests,
        },
      });
    } catch (error) {
      console.error('Failed to save load test results:', error);
    }
  }

  private printSummary(): void {
    const duration = (this.results.endTime.getTime() - this.results.startTime.getTime()) / 1000;
    
    console.log('\n' + '='.repeat(50));
    console.log('📈 LOAD TEST SUMMARY');
    console.log('='.repeat(50));
    console.log(`Suite: ${this.results.suiteName}`);
    console.log(`Duration: ${duration.toFixed(1)}s`);
    console.log(`Total Tests: ${this.results.summary.totalTests}`);
    console.log(`Passed: ${this.results.summary.passed} ✅`);
    console.log(`Failed: ${this.results.summary.failed} ❌`);
    console.log(`Average Latency: ${this.results.summary.avgLatency}ms`);
    console.log(`Max Latency (P99): ${this.results.summary.maxLatency}ms`);
    console.log('='.repeat(50) + '\n');
  }
}

// Predefined test suites
export const testSuites = {
  quick: {
    name: 'Quick Load Test',
    scenarios: ['light', 'small'],
  },
  
  standard: {
    name: 'Standard Load Test',
    scenarios: ['light', 'medium', 'small', 'medium'],
  },
  
  comprehensive: {
    name: 'Comprehensive Load Test',
    scenarios: ['light', 'medium', 'heavy', 'small', 'medium', 'large'],
  },
  
  stress: {
    name: 'Stress Test',
    scenarios: ['heavy', 'stress', 'large', 'stress'],
  },
};

// Helper function to run a test suite
export async function runLoadTestSuite(
  suiteName: keyof typeof testSuites,
  serverUrl: string,
  roomId: string
): Promise<LoadTestReport> {
  const suiteConfig = testSuites[suiteName];
  const suite: LoadTestSuite = {
    ...suiteConfig,
    serverUrl,
    roomId,
  };
  
  const runner = new LoadTestRunner(suite);
  return await runner.runAllTests();
}

// Example usage function
export async function runExampleLoadTests(): Promise<void> {
  const serverUrl = process.env.COMMUNICATION_SERVICE_URL || 'http://localhost:3002';
  const roomId = 'load-test-room';
  
  try {
    // Run quick test suite
    const results = await runLoadTestSuite('quick', serverUrl, roomId);
    
    console.log('\n🎉 Load tests completed successfully!');
    console.log('Results saved to AsyncStorage');
    
    // Get performance summary
    const perfSummary = await performanceMonitor.getPerformanceSummary();
    console.log(`\nTotal metrics collected: ${perfSummary.metrics.length}`);
  } catch (error) {
    console.error('\n❌ Load tests failed:', error);
    throw error;
  }
}