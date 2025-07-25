/**
 * 优化的状态管理Hooks
 * 提供高性能的状态管理方案
 */

import { useCallback, useReducer, useMemo, useRef, useEffect } from 'react';

// 通用状态Action类型
interface StateAction<T> {
  type: 'SET' | 'UPDATE' | 'RESET';
  payload?: Partial<T>;
}

// 优化的状态reducer
function optimizedStateReducer<T>(state: T, action: StateAction<T>): T {
  switch (action.type) {
    case 'SET':
      return action.payload as T;
    case 'UPDATE':
      return { ...state, ...action.payload };
    case 'RESET':
      return action.payload as T;
    default:
      return state;
  }
}

/**
 * 优化的状态管理Hook
 * 使用useReducer和useCallback优化性能
 */
export function useOptimizedState<T>(initialState: T) {
  const [state, dispatch] = useReducer(optimizedStateReducer<T>, initialState);
  
  const setState = useCallback((newState: T | Partial<T>) => {
    dispatch({ type: 'SET', payload: newState });
  }, []);
  
  const updateState = useCallback((updates: Partial<T>) => {
    dispatch({ type: 'UPDATE', payload: updates });
  }, []);
  
  const resetState = useCallback((resetValue?: T) => {
    dispatch({ type: 'RESET', payload: resetValue || initialState });
  }, [initialState]);
  
  return {
    state,
    setState,
    updateState,
    resetState
  };
}

/**
 * 防抖状态Hook
 * 适用于用户输入等频繁更新场景
 */
export function useDebouncedState<T>(initialValue: T, delay: number = 300) {
  const [immediateValue, setImmediateValue] = useOptimizedState(initialValue);
  const [debouncedValue, setDebouncedValue] = useOptimizedState(initialValue);
  const timeoutRef = useRef<NodeJS.Timeout>();

  const setValue = useCallback((value: T | Partial<T>) => {
    setImmediateValue.setState(value as T);
    
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    
    timeoutRef.current = setTimeout(() => {
      setDebouncedValue.setState(value as T);
    }, delay);
  }, [delay, setImmediateValue, setDebouncedValue]);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return {
    immediateValue: immediateValue.state,
    debouncedValue: debouncedValue.state,
    setValue
  };
}

/**
 * 异步状态管理Hook
 * 处理加载状态、错误状态等
 */
interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

export function useAsyncState<T>(initialData: T | null = null) {
  const { state, updateState, resetState } = useOptimizedState<AsyncState<T>>({
    data: initialData,
    loading: false,
    error: null
  });

  const setLoading = useCallback((loading: boolean) => {
    updateState({ loading });
  }, [updateState]);

  const setData = useCallback((data: T) => {
    updateState({ data, loading: false, error: null });
  }, [updateState]);

  const setError = useCallback((error: string) => {
    updateState({ error, loading: false });
  }, [updateState]);

  const execute = useCallback(async (asyncFn: () => Promise<T>) => {
    setLoading(true);
    try {
      const result = await asyncFn();
      setData(result);
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '未知错误';
      setError(errorMessage);
      throw err;
    }
  }, [setLoading, setData, setError]);

  return {
    ...state,
    setLoading,
    setData,
    setError,
    execute,
    reset: resetState
  };
}

/**
 * 选择器Hook - 优化大状态对象的订阅
 */
export function useSelector<T, K>(
  state: T,
  selector: (state: T) => K,
  deps?: React.DependencyList
): K {
  return useMemo(() => selector(state), deps ? [state, ...deps] : [state]);
}

/**
 * 批量状态更新Hook
 * 避免多次渲染
 */
export function useBatchedState<T>(initialState: T) {
  const [state, setState] = useOptimizedState(initialState);
  const batchedUpdates = useRef<Partial<T>[]>([]);
  const batchTimeoutRef = useRef<NodeJS.Timeout>();

  const batchUpdate = useCallback((update: Partial<T>) => {
    batchedUpdates.current.push(update);
    
    if (batchTimeoutRef.current) {
      clearTimeout(batchTimeoutRef.current);
    }
    
    batchTimeoutRef.current = setTimeout(() => {
      const combinedUpdate = batchedUpdates.current.reduce(
        (acc, update) => ({ ...acc, ...update }),
        {}
      );
      setState.updateState(combinedUpdate);
      batchedUpdates.current = [];
    }, 0);
  }, [setState]);

  useEffect(() => {
    return () => {
      if (batchTimeoutRef.current) {
        clearTimeout(batchTimeoutRef.current);
      }
    };
  }, []);

  return {
    state: state.state,
    batchUpdate,
    setState: setState.setState,
    updateState: setState.updateState,
    resetState: setState.resetState
  };
}
