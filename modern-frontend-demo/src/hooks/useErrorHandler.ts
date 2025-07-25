/**
 * 错误处理Hook
 * 提供统一的错误处理和用户通知机制
 */

import { useCallback, useRef } from 'react';
import { useOptimizedState } from './optimizedState';

interface ErrorState {
  errors: Array<{
    id: string;
    message: string;
    type: 'error' | 'warning' | 'info';
    timestamp: number;
    dismissed: boolean;
  }>;
  globalError: string | null;
}

const initialErrorState: ErrorState = {
  errors: [],
  globalError: null
};

export function useErrorHandler() {
  const { state, updateState, setState } = useOptimizedState(initialErrorState);
  const errorIdCounter = useRef(0);

  // 添加错误
  const addError = useCallback((
    message: string, 
    type: 'error' | 'warning' | 'info' = 'error',
    autoRemove: boolean = true
  ) => {
    const errorId = `error_${++errorIdCounter.current}_${Date.now()}`;
    
    const newError = {
      id: errorId,
      message,
      type,
      timestamp: Date.now(),
      dismissed: false
    };

    updateState({
      errors: [...state.errors, newError]
    });

    // 自动移除错误（可选）
    if (autoRemove) {
      setTimeout(() => {
        removeError(errorId);
      }, type === 'error' ? 8000 : 4000);
    }

    return errorId;
  }, [state.errors, updateState]);

  // 移除错误
  const removeError = useCallback((errorId: string) => {
    updateState({
      errors: state.errors.filter(error => error.id !== errorId)
    });
  }, [state.errors, updateState]);

  // 标记错误为已忽略
  const dismissError = useCallback((errorId: string) => {
    updateState({
      errors: state.errors.map(error =>
        error.id === errorId ? { ...error, dismissed: true } : error
      )
    });
  }, [state.errors, updateState]);

  // 清空所有错误
  const clearErrors = useCallback(() => {
    updateState({ errors: [] });
  }, [updateState]);

  // 设置全局错误
  const setGlobalError = useCallback((message: string | null) => {
    updateState({ globalError: message });
  }, [updateState]);

  // 处理API错误
  const handleApiError = useCallback((error: any, context?: string) => {
    let message = '发生了未知错误';
    
    if (error?.response?.data?.message) {
      message = error.response.data.message;
    } else if (error?.message) {
      message = error.message;
    } else if (typeof error === 'string') {
      message = error;
    }
    
    if (context) {
      message = `${context}: ${message}`;
    }
    
    return addError(message, 'error');
  }, [addError]);

  // 处理异步操作错误
  const withErrorHandling = useCallback(<T extends any[], R>(
    asyncFn: (...args: T) => Promise<R>,
    errorContext?: string
  ) => {
    return async (...args: T): Promise<R | null> => {
      try {
        return await asyncFn(...args);
      } catch (error) {
        handleApiError(error, errorContext);
        return null;
      }
    };
  }, [handleApiError]);

  // 安全执行函数
  const safeExecute = useCallback(<T extends any[], R>(
    fn: (...args: T) => R,
    errorContext?: string,
    defaultValue?: R
  ) => {
    return (...args: T): R | undefined => {
      try {
        return fn(...args);
      } catch (error) {
        handleApiError(error, errorContext);
        return defaultValue;
      }
    };
  }, [handleApiError]);

  // 获取活跃错误（未忽略的）
  const activeErrors = state.errors.filter(error => !error.dismissed);
  
  // 获取最新错误
  const latestError = activeErrors.length > 0 
    ? activeErrors[activeErrors.length - 1] 
    : null;

  return {
    // 状态
    errors: state.errors,
    activeErrors,
    latestError,
    globalError: state.globalError,
    hasErrors: activeErrors.length > 0,
    
    // 操作
    addError,
    removeError,
    dismissError,
    clearErrors,
    setGlobalError,
    handleApiError,
    withErrorHandling,
    safeExecute
  };
}

// 用于组件的错误处理Hook
export function useComponentErrorHandler(componentName: string) {
  const errorHandler = useErrorHandler();
  
  const handleError = useCallback((error: Error | string, action?: string) => {
    const message = typeof error === 'string' ? error : error.message;
    const context = action ? `${componentName} - ${action}` : componentName;
    
    return errorHandler.addError(message, 'error');
  }, [errorHandler, componentName]);
  
  const handleWarning = useCallback((message: string, action?: string) => {
    const context = action ? `${componentName} - ${action}` : componentName;
    return errorHandler.addError(`${context}: ${message}`, 'warning');
  }, [errorHandler, componentName]);
  
  return {
    ...errorHandler,
    handleError,
    handleWarning
  };
}

// 错误边界集成Hook
export function useErrorBoundaryIntegration() {
  const { addError } = useErrorHandler();
  
  const reportError = useCallback((error: Error, errorInfo?: any) => {
    const message = `组件错误: ${error.message}`;
    addError(message, 'error', false); // 不自动移除
    
    // 可以在这里集成错误监控服务
    console.error('Error Boundary Integration:', { error, errorInfo });
  }, [addError]);
  
  return { reportError };
}
