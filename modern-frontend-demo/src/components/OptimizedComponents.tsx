/**
 * 优化组件示例
 * 展示React.memo、useCallback、useMemo等性能优化技术
 */

import React, { 
  memo, 
  useCallback, 
  useMemo, 
  useState,
  useRef,
  forwardRef,
  useImperativeHandle
} from 'react';
import { ErrorBoundary } from '../shared/ErrorBoundary';
import { VirtualScroll } from './VirtualScroll';
import { useDebounce, useThrottle, useImageLazyLoad } from '../hooks/usePerformance';

// 基础优化组件示例
interface OptimizedButtonProps {
  onClick: () => void;
  children: React.ReactNode;
  disabled?: boolean;
  variant?: 'primary' | 'secondary';
  size?: 'small' | 'medium' | 'large';
}

export const OptimizedButton = memo<OptimizedButtonProps>(({
  onClick,
  children,
  disabled = false,
  variant = 'primary',
  size = 'medium'
}) => {
  // 使用useMemo缓存样式计算
  const buttonStyles = useMemo(() => {
    const baseStyles = {
      padding: size === 'small' ? '4px 8px' : size === 'large' ? '12px 24px' : '8px 16px',
      fontSize: size === 'small' ? '12px' : size === 'large' ? '18px' : '14px',
      borderRadius: '4px',
      border: 'none',
      cursor: disabled ? 'not-allowed' : 'pointer',
      opacity: disabled ? 0.6 : 1,
      transition: 'all 0.2s ease'
    };

    const variantStyles = variant === 'primary' 
      ? { backgroundColor: '#007bff', color: 'white' }
      : { backgroundColor: '#f8f9fa', color: '#333', border: '1px solid #dee2e6' };

    return { ...baseStyles, ...variantStyles };
  }, [variant, size, disabled]);

  // 使用useCallback防止不必要的重渲染
  const handleClick = useCallback(() => {
    if (!disabled) {
      onClick();
    }
  }, [onClick, disabled]);

  return (
    <button style={buttonStyles} onClick={handleClick}>
      {children}
    </button>
  );
});

OptimizedButton.displayName = 'OptimizedButton';

// 搜索输入框优化示例
interface OptimizedSearchProps {
  onSearch: (query: string) => void;
  placeholder?: string;
  debounceMs?: number;
}

export const OptimizedSearch = memo<OptimizedSearchProps>(({
  onSearch,
  placeholder = '搜索...',
  debounceMs = 300
}) => {
  const [query, setQuery] = useState('');
  const debouncedQuery = useDebounce(query, debounceMs);

  // 防抖后执行搜索
  React.useEffect(() => {
    onSearch(debouncedQuery);
  }, [debouncedQuery, onSearch]);

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setQuery(e.target.value);
  }, []);

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <input
        type="text"
        value={query}
        onChange={handleInputChange}
        placeholder={placeholder}
        style={{
          padding: '8px 12px',
          border: '1px solid #ddd',
          borderRadius: '4px',
          fontSize: '14px',
          width: '200px'
        }}
      />
      {query && (
        <button
          onClick={() => setQuery('')}
          style={{
            position: 'absolute',
            right: '8px',
            top: '50%',
            transform: 'translateY(-50%)',
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            fontSize: '18px',
            color: '#999'
          }}
        >
          ×
        </button>
      )}
    </div>
  );
});

OptimizedSearch.displayName = 'OptimizedSearch';

// 图片懒加载组件
interface LazyImageProps {
  src: string;
  alt: string;
  width?: number;
  height?: number;
  placeholder?: string;
  className?: string;
}

export const LazyImage = memo<LazyImageProps>(({
  src,
  alt,
  width,
  height,
  placeholder = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtc2l6ZT0iMTgiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIiBmaWxsPSIjOTk5Ij5Mb2FkaW5nLi4uPC90ZXh0Pjwvc3ZnPg==',
  className = ''
}) => {
  const { imageSrc, imgCallbackRef, isLoading, hasError } = useImageLazyLoad(src, placeholder);

  if (hasError) {
    return (
      <div
        style={{
          width,
          height,
          backgroundColor: '#f5f5f5',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#999'
        }}
        className={className}
      >
        加载失败
      </div>
    );
  }

  return (
    <img
      ref={imgCallbackRef}
      src={imageSrc}
      alt={alt}
      width={width}
      height={height}
      className={className}
      style={{
        opacity: isLoading ? 0.5 : 1,
        transition: 'opacity 0.3s ease'
      }}
    />
  );
});

LazyImage.displayName = 'LazyImage';

// 大列表优化组件
interface OptimizedListProps<T> {
  items: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  itemHeight: number;
  height: number;
  loading?: boolean;
  error?: string;
  onLoadMore?: () => void;
}

export const OptimizedList = memo(<T,>({
  items,
  renderItem,
  itemHeight,
  height,
  loading = false,
  error,
  onLoadMore
}: OptimizedListProps<T>) => {
  const loadMoreRef = useRef<HTMLDivElement>(null);

  // 无限滚动
  React.useEffect(() => {
    if (!onLoadMore || loading || error) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          onLoadMore();
        }
      },
      { threshold: 0.1 }
    );

    if (loadMoreRef.current) {
      observer.observe(loadMoreRef.current);
    }

    return () => observer.disconnect();
  }, [onLoadMore, loading, error]);

  if (error) {
    return (
      <div style={{ 
        height, 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        color: '#e74c3c'
      }}>
        错误: {error}
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <VirtualScroll
        items={items}
        itemHeight={itemHeight}
        height={height}
        renderItem={renderItem}
      />
      {onLoadMore && (
        <div ref={loadMoreRef} style={{ height: '40px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          {loading ? '加载中...' : '加载更多'}
        </div>
      )}
    </ErrorBoundary>
  );
});

OptimizedList.displayName = 'OptimizedList';

// 表单优化组件
interface OptimizedFormProps {
  onSubmit: (data: Record<string, any>) => void;
  initialValues?: Record<string, any>;
  validation?: Record<string, (value: any) => string | null>;
}

export const OptimizedForm = memo<OptimizedFormProps>(({
  onSubmit,
  initialValues = {},
  validation = {}
}) => {
  const [values, setValues] = useState(initialValues);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [touched, setTouched] = useState<Record<string, boolean>>({});

  // 防抖验证
  const debouncedValues = useDebounce(values, 300);

  // 验证表单
  const validateField = useCallback((name: string, value: any) => {
    const validator = validation[name];
    if (validator) {
      const error = validator(value);
      setErrors(prev => ({
        ...prev,
        [name]: error || ''
      }));
      return !error;
    }
    return true;
  }, [validation]);

  // 处理输入变化
  const handleChange = useCallback((name: string, value: any) => {
    setValues(prev => ({ ...prev, [name]: value }));
    if (touched[name]) {
      validateField(name, value);
    }
  }, [touched, validateField]);

  // 处理失焦
  const handleBlur = useCallback((name: string) => {
    setTouched(prev => ({ ...prev, [name]: true }));
    validateField(name, values[name]);
  }, [values, validateField]);

  // 处理提交
  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    
    // 验证所有字段
    const newErrors: Record<string, string> = {};
    let isValid = true;
    
    Object.keys(validation).forEach(name => {
      const error = validation[name](values[name]);
      if (error) {
        newErrors[name] = error;
        isValid = false;
      }
    });
    
    setErrors(newErrors);
    setTouched(Object.keys(validation).reduce((acc, key) => ({ ...acc, [key]: true }), {}));
    
    if (isValid) {
      onSubmit(values);
    }
  }, [values, validation, onSubmit]);

  return (
    <form onSubmit={handleSubmit} style={{ maxWidth: '400px', margin: '0 auto' }}>
      <div style={{ marginBottom: '16px' }}>
        <label style={{ display: 'block', marginBottom: '4px' }}>姓名:</label>
        <input
          type="text"
          value={values.name || ''}
          onChange={(e) => handleChange('name', e.target.value)}
          onBlur={() => handleBlur('name')}
          style={{
            width: '100%',
            padding: '8px',
            border: errors.name ? '1px solid #e74c3c' : '1px solid #ddd',
            borderRadius: '4px'
          }}
        />
        {errors.name && touched.name && (
          <div style={{ color: '#e74c3c', fontSize: '12px', marginTop: '4px' }}>
            {errors.name}
          </div>
        )}
      </div>

      <div style={{ marginBottom: '16px' }}>
        <label style={{ display: 'block', marginBottom: '4px' }}>邮箱:</label>
        <input
          type="email"
          value={values.email || ''}
          onChange={(e) => handleChange('email', e.target.value)}
          onBlur={() => handleBlur('email')}
          style={{
            width: '100%',
            padding: '8px',
            border: errors.email ? '1px solid #e74c3c' : '1px solid #ddd',
            borderRadius: '4px'
          }}
        />
        {errors.email && touched.email && (
          <div style={{ color: '#e74c3c', fontSize: '12px', marginTop: '4px' }}>
            {errors.email}
          </div>
        )}
      </div>

      <OptimizedButton onClick={() => {}}>
        提交
      </OptimizedButton>
    </form>
  );
});

OptimizedForm.displayName = 'OptimizedForm';

// 综合示例组件
export const PerformanceShowcase = memo(() => {
  const [searchQuery, setSearchQuery] = useState('');
  const [listItems, setListItems] = useState(
    Array.from({ length: 1000 }, (_, i) => ({ id: i, name: `项目 ${i + 1}` }))
  );

  const filteredItems = useMemo(() => {
    if (!searchQuery) return listItems;
    return listItems.filter(item => 
      item.name.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [listItems, searchQuery]);

  const handleFormSubmit = useCallback((data: Record<string, any>) => {
    console.log('表单提交:', data);
  }, []);

  const formValidation = useMemo(() => ({
    name: (value: string) => !value ? '姓名不能为空' : null,
    email: (value: string) => !value || !/S+@S+.S+/.test(value) ? '请输入有效邮箱' : null
  }), []);

  return (
    <ErrorBoundary>
      <div style={{ padding: '20px' }}>
        <h2>性能优化组件示例</h2>
        
        <div style={{ marginBottom: '20px' }}>
          <h3>搜索组件</h3>
          <OptimizedSearch onSearch={setSearchQuery} />
        </div>

        <div style={{ marginBottom: '20px' }}>
          <h3>虚拟滚动列表 (显示 {filteredItems.length} 项)</h3>
          <OptimizedList
            items={filteredItems}
            itemHeight={40}
            height={300}
            renderItem={(item) => (
              <div style={{ padding: '8px', borderBottom: '1px solid #eee' }}>
                {item.name}
              </div>
            )}
          />
        </div>

        <div style={{ marginBottom: '20px' }}>
          <h3>懒加载图片</h3>
          <LazyImage
            src="https://picsum.photos/200/200"
            alt="示例图片"
            width={200}
            height={200}
          />
        </div>

        <div>
          <h3>优化表单</h3>
          <OptimizedForm
            onSubmit={handleFormSubmit}
            validation={formValidation}
          />
        </div>
      </div>
    </ErrorBoundary>
  );
});

PerformanceShowcase.displayName = 'PerformanceShowcase';
