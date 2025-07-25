/**
 * 性能优化相关Hooks
 * 提供React.memo、虚拟滚动、懒加载等性能优化方案
 */

import { 
  useCallback, 
  useMemo, 
  useRef, 
  useEffect, 
  useState,
  RefObject,
  DependencyList
} from 'react';

/**
 * 防抖Hook
 */
export function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
}

/**
 * 节流Hook
 */
export function useThrottle<T>(value: T, interval: number): T {
  const [throttledValue, setThrottledValue] = useState<T>(value);
  const lastExecuted = useRef<number>(Date.now());

  useEffect(() => {
    if (Date.now() >= lastExecuted.current + interval) {
      lastExecuted.current = Date.now();
      setThrottledValue(value);
    } else {
      const timer = setTimeout(() => {
        lastExecuted.current = Date.now();
        setThrottledValue(value);
      }, interval);

      return () => clearTimeout(timer);
    }
  }, [value, interval]);

  return throttledValue;
}

/**
 * 稳定化回调Hook - 避免不必要的重渲染
 */
export function useStableCallback<T extends (...args: any[]) => any>(
  callback: T,
  deps: DependencyList
): T {
  const stableCallback = useCallback(callback, deps);
  return stableCallback;
}

/**
 * 深度比较Memo Hook
 */
export function useDeepMemo<T>(factory: () => T, deps: DependencyList): T {
  const ref = useRef<{ deps: DependencyList; value: T }>();
  
  if (!ref.current || !deepEqual(ref.current.deps, deps)) {
    ref.current = { deps, value: factory() };
  }
  
  return ref.current.value;
}

function deepEqual(a: any, b: any): boolean {
  if (a === b) return true;
  if (a == null || b == null) return false;
  if (typeof a !== typeof b) return false;
  
  if (typeof a === 'object') {
    const keysA = Object.keys(a);
    const keysB = Object.keys(b);
    
    if (keysA.length !== keysB.length) return false;
    
    for (const key of keysA) {
      if (!keysB.includes(key) || !deepEqual(a[key], b[key])) {
        return false;
      }
    }
    
    return true;
  }
  
  return false;
}

/**
 * 懒加载Hook
 */
interface UseLazyLoadOptions {
  threshold?: number;
  rootMargin?: string;
  enabled?: boolean;
}

export function useLazyLoad(
  ref: RefObject<Element>,
  options: UseLazyLoadOptions = {}
) {
  const [isIntersecting, setIsIntersecting] = useState(false);
  const [hasLoaded, setHasLoaded] = useState(false);
  
  const { threshold = 0.1, rootMargin = '0px', enabled = true } = options;

  useEffect(() => {
    if (!enabled || !ref.current) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsIntersecting(true);
          setHasLoaded(true);
          observer.disconnect();
        }
      },
      { threshold, rootMargin }
    );

    observer.observe(ref.current);

    return () => {
      observer.disconnect();
    };
  }, [ref, threshold, rootMargin, enabled]);

  return { isIntersecting, hasLoaded };
}

/**
 * 虚拟滚动Hook
 */
interface UseVirtualScrollOptions {
  itemHeight: number;
  containerHeight: number;
  overscan?: number;
}

export function useVirtualScroll<T>(
  items: T[],
  options: UseVirtualScrollOptions
) {
  const { itemHeight, containerHeight, overscan = 5 } = options;
  const [scrollTop, setScrollTop] = useState(0);
  
  const visibleItems = useMemo(() => {
    const visibleCount = Math.ceil(containerHeight / itemHeight);
    const startIndex = Math.floor(scrollTop / itemHeight);
    const endIndex = Math.min(
      startIndex + visibleCount + overscan,
      items.length
    );
    const adjustedStartIndex = Math.max(0, startIndex - overscan);
    
    return {
      startIndex: adjustedStartIndex,
      endIndex,
      visibleItems: items.slice(adjustedStartIndex, endIndex),
      totalHeight: items.length * itemHeight,
      offsetY: adjustedStartIndex * itemHeight
    };
  }, [items, itemHeight, containerHeight, scrollTop, overscan]);

  const handleScroll = useCallback((event: React.UIEvent<HTMLElement>) => {
    setScrollTop(event.currentTarget.scrollTop);
  }, []);

  return {
    ...visibleItems,
    handleScroll
  };
}

/**
 * 窗口大小Hook
 */
export function useWindowSize() {
  const [windowSize, setWindowSize] = useState({
    width: typeof window !== 'undefined' ? window.innerWidth : 0,
    height: typeof window !== 'undefined' ? window.innerHeight : 0,
  });

  useEffect(() => {
    if (typeof window === 'undefined') return;

    const handleResize = () => {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return windowSize;
}

/**
 * 媒体查询Hook
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => {
    if (typeof window === 'undefined') return false;
    return window.matchMedia(query).matches;
  });

  useEffect(() => {
    if (typeof window === 'undefined') return;

    const mediaQuery = window.matchMedia(query);
    const handler = (event: MediaQueryListEvent) => setMatches(event.matches);
    
    mediaQuery.addListener(handler);
    return () => mediaQuery.removeListener(handler);
  }, [query]);

  return matches;
}

/**
 * 图片懒加载Hook
 */
export function useImageLazyLoad(src: string, placeholder?: string) {
  const [imageSrc, setImageSrc] = useState(placeholder || '');
  const [imageRef, setImageRef] = useState<HTMLImageElement | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);

  const imgCallbackRef = useCallback((node: HTMLImageElement | null) => {
    setImageRef(node);
  }, []);

  const { hasLoaded } = useLazyLoad(
    { current: imageRef },
    { threshold: 0.1 }
  );

  useEffect(() => {
    if (!hasLoaded || !src) return;

    setIsLoading(true);
    setHasError(false);

    const img = new Image();
    img.onload = () => {
      setImageSrc(src);
      setIsLoading(false);
    };
    img.onerror = () => {
      setHasError(true);
      setIsLoading(false);
    };
    img.src = src;
  }, [hasLoaded, src]);

  return {
    imageSrc,
    imgCallbackRef,
    isLoading,
    hasError,
    hasLoaded
  };
}

/**
 * 性能监控Hook
 */
export function usePerformanceMonitor(name: string) {
  const renderStart = useRef<number>(performance.now());
  const renderCount = useRef<number>(0);

  useEffect(() => {
    renderCount.current += 1;
    const renderEnd = performance.now();
    const renderTime = renderEnd - renderStart.current;
    
    // 记录性能指标
    if (renderTime > 16) { // 超过一帧的时间
      console.warn(`${name} render took ${renderTime.toFixed(2)}ms (render #${renderCount.current})`);
    }
    
    renderStart.current = performance.now();
  });

  return {
    renderCount: renderCount.current
  };
}

/**
 * 内存泄漏防护Hook
 */
export function useMemoryLeakProtection() {
  const timeouts = useRef<Set<NodeJS.Timeout>>(new Set());
  const intervals = useRef<Set<NodeJS.Timeout>>(new Set());
  const abortControllers = useRef<Set<AbortController>>(new Set());

  const setTimeout = useCallback((callback: () => void, delay: number) => {
    const timeoutId = globalThis.setTimeout(() => {
      timeouts.current.delete(timeoutId);
      callback();
    }, delay);
    timeouts.current.add(timeoutId);
    return timeoutId;
  }, []);

  const setInterval = useCallback((callback: () => void, delay: number) => {
    const intervalId = globalThis.setInterval(callback, delay);
    intervals.current.add(intervalId);
    return intervalId;
  }, []);

  const createAbortController = useCallback(() => {
    const controller = new AbortController();
    abortControllers.current.add(controller);
    return controller;
  }, []);

  useEffect(() => {
    return () => {
      // 清理所有定时器
      timeouts.current.forEach(id => globalThis.clearTimeout(id));
      intervals.current.forEach(id => globalThis.clearInterval(id));
      abortControllers.current.forEach(controller => controller.abort());
      
      timeouts.current.clear();
      intervals.current.clear();
      abortControllers.current.clear();
    };
  }, []);

  return {
    setTimeout,
    setInterval,
    createAbortController
  };
}
