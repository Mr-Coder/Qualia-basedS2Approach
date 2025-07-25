/**
 * React错误边界组件
 * 捕获组件树中的JavaScript错误，记录错误并显示备用UI
 */

import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  resetKeys?: Array<string | number>;
  resetOnPropsChange?: boolean;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  errorId: string;
}

export class ErrorBoundary extends Component<Props, State> {
  private resetTimeoutId: number | null = null;

  constructor(props: Props) {
    super(props);
    
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: ''
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    // 更新state，使下一次渲染显示错误UI
    return {
      hasError: true,
      error,
      errorId: `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // 记录错误信息
    this.setState({ errorInfo });
    
    // 调用外部错误处理函数
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
    
    // 发送错误到监控服务
    this.reportError(error, errorInfo);
  }

  componentDidUpdate(prevProps: Props) {
    const { resetKeys, resetOnPropsChange } = this.props;
    const { hasError } = this.state;
    
    if (hasError && prevProps.resetKeys !== resetKeys) {
      if (resetKeys?.some((key, idx) => prevProps.resetKeys?.[idx] !== key)) {
        this.resetErrorBoundary();
      }
    }
    
    if (hasError && resetOnPropsChange && prevProps.children !== this.props.children) {
      this.resetErrorBoundary();
    }
  }

  resetErrorBoundary = () => {
    if (this.resetTimeoutId) {
      clearTimeout(this.resetTimeoutId);
    }
    
    this.resetTimeoutId = window.setTimeout(() => {
      this.setState({
        hasError: false,
        error: null,
        errorInfo: null,
        errorId: ''
      });
    }, 100);
  };

  private reportError(error: Error, errorInfo: ErrorInfo) {
    try {
      // 发送到错误监控服务
      const errorReport = {
        message: error.message,
        stack: error.stack,
        componentStack: errorInfo.componentStack,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        url: window.location.href,
        errorId: this.state.errorId
      };
      
      // 这里可以集成Sentry、LogRocket等错误监控服务
      console.error('Error Boundary caught an error:', errorReport);
      
      // 示例：发送到API
      // fetch('/api/errors', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify(errorReport)
      // }).catch(console.error);
      
    } catch (reportingError) {
      console.error('Failed to report error:', reportingError);
    }
  }

  render() {
    if (this.state.hasError) {
      // 自定义错误UI
      if (this.props.fallback) {
        return this.props.fallback;
      }
      
      return (
        <ErrorFallbackComponent
          error={this.state.error}
          errorInfo={this.state.errorInfo}
          resetErrorBoundary={this.resetErrorBoundary}
          errorId={this.state.errorId}
        />
      );
    }

    return this.props.children;
  }
}

// 默认错误回退组件
interface ErrorFallbackProps {
  error: Error | null;
  errorInfo: ErrorInfo | null;
  resetErrorBoundary: () => void;
  errorId: string;
}

const ErrorFallbackComponent: React.FC<ErrorFallbackProps> = ({
  error,
  errorInfo,
  resetErrorBoundary,
  errorId
}) => {
  const [showDetails, setShowDetails] = React.useState(false);
  
  return (
    <div className="error-boundary-container" style={{
      padding: '20px',
      margin: '20px',
      border: '1px solid #ff6b6b',
      borderRadius: '8px',
      backgroundColor: '#fff5f5',
      fontFamily: 'Arial, sans-serif'
    }}>
      <div className="error-header" style={{ marginBottom: '15px' }}>
        <h2 style={{ color: '#e03131', margin: '0 0 10px 0' }}>
          ⚠️ 出现了错误
        </h2>
        <p style={{ color: '#666', margin: 0 }}>
          很抱歉，页面遇到了问题。我们已经记录了这个错误并会尽快修复。
        </p>
      </div>
      
      <div className="error-actions" style={{ marginBottom: '15px' }}>
        <button
          onClick={resetErrorBoundary}
          style={{
            padding: '8px 16px',
            backgroundColor: '#228be6',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            marginRight: '10px'
          }}
        >
          重试
        </button>
        
        <button
          onClick={() => window.location.reload()}
          style={{
            padding: '8px 16px',
            backgroundColor: '#868e96',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            marginRight: '10px'
          }}
        >
          刷新页面
        </button>
        
        <button
          onClick={() => setShowDetails(!showDetails)}
          style={{
            padding: '8px 16px',
            backgroundColor: 'transparent',
            color: '#228be6',
            border: '1px solid #228be6',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          {showDetails ? '隐藏' : '显示'}详情
        </button>
      </div>
      
      {showDetails && (
        <div className="error-details" style={{
          backgroundColor: '#f8f9fa',
          padding: '15px',
          borderRadius: '4px',
          fontSize: '14px',
          fontFamily: 'monospace'
        }}>
          <div style={{ marginBottom: '10px' }}>
            <strong>错误ID:</strong> {errorId}
          </div>
          
          {error && (
            <div style={{ marginBottom: '10px' }}>
              <strong>错误信息:</strong>
              <pre style={{ 
                whiteSpace: 'pre-wrap', 
                wordBreak: 'break-word',
                margin: '5px 0 0 0',
                color: '#e03131'
              }}>
                {error.message}
              </pre>
            </div>
          )}
          
          {error?.stack && (
            <div style={{ marginBottom: '10px' }}>
              <strong>错误堆栈:</strong>
              <pre style={{ 
                whiteSpace: 'pre-wrap', 
                wordBreak: 'break-word',
                margin: '5px 0 0 0',
                fontSize: '12px',
                maxHeight: '200px',
                overflow: 'auto'
              }}>
                {error.stack}
              </pre>
            </div>
          )}
          
          {errorInfo?.componentStack && (
            <div>
              <strong>组件堆栈:</strong>
              <pre style={{ 
                whiteSpace: 'pre-wrap', 
                wordBreak: 'break-word',
                margin: '5px 0 0 0',
                fontSize: '12px',
                maxHeight: '200px',
                overflow: 'auto'
              }}>
                {errorInfo.componentStack}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// 高阶组件包装器
export function withErrorBoundary<P extends object>(
  Component: React.ComponentType<P>,
  errorBoundaryProps?: Omit<Props, 'children'>
) {
  const WrappedComponent = (props: P) => (
    <ErrorBoundary {...errorBoundaryProps}>
      <Component {...props} />
    </ErrorBoundary>
  );
  
  WrappedComponent.displayName = `withErrorBoundary(${Component.displayName || Component.name})`;
  
  return WrappedComponent;
}

// Hook版本的错误边界
export function useErrorBoundary() {
  const [error, setError] = React.useState<Error | null>(null);
  
  const resetError = React.useCallback(() => {
    setError(null);
  }, []);
  
  const captureError = React.useCallback((error: Error) => {
    setError(error);
  }, []);
  
  React.useEffect(() => {
    if (error) {
      throw error;
    }
  }, [error]);
  
  return { captureError, resetError };
}
