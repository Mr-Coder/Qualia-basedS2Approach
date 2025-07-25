/**
 * 虚拟滚动组件
 * 高性能长列表渲染解决方案
 */

import React, { 
  memo, 
  useMemo, 
  useRef, 
  useEffect, 
  useState,
  useCallback,
  CSSProperties
} from 'react';

interface VirtualScrollProps<T> {
  items: T[];
  itemHeight: number;
  height: number;
  renderItem: (item: T, index: number) => React.ReactNode;
  className?: string;
  style?: CSSProperties;
  overscan?: number;
  onScroll?: (scrollTop: number) => void;
  estimatedItemSize?: number;
}

export const VirtualScroll = memo(<T,>({
  items,
  itemHeight,
  height,
  renderItem,
  className = '',
  style = {},
  overscan = 5,
  onScroll,
  estimatedItemSize
}: VirtualScrollProps<T>) => {
  const [scrollTop, setScrollTop] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  
  // 计算可见项目
  const visibleRange = useMemo(() => {
    const containerHeight = height;
    const totalItems = items.length;
    
    if (totalItems === 0) {
      return {
        startIndex: 0,
        endIndex: 0,
        visibleItems: [],
        totalHeight: 0,
        offsetY: 0
      };
    }
    
    const startIndex = Math.floor(scrollTop / itemHeight);
    const visibleCount = Math.ceil(containerHeight / itemHeight);
    const endIndex = Math.min(startIndex + visibleCount, totalItems);
    
    // 添加overscan
    const overscanStart = Math.max(0, startIndex - overscan);
    const overscanEnd = Math.min(totalItems, endIndex + overscan);
    
    const visibleItems = items.slice(overscanStart, overscanEnd);
    const totalHeight = totalItems * itemHeight;
    const offsetY = overscanStart * itemHeight;
    
    return {
      startIndex: overscanStart,
      endIndex: overscanEnd,
      visibleItems,
      totalHeight,
      offsetY
    };
  }, [items, itemHeight, height, scrollTop, overscan]);

  const handleScroll = useCallback((event: React.UIEvent<HTMLDivElement>) => {
    const scrollTop = event.currentTarget.scrollTop;
    setScrollTop(scrollTop);
    onScroll?.(scrollTop);
  }, [onScroll]);

  // 滚动到指定索引
  const scrollToIndex = useCallback((index: number) => {
    if (containerRef.current) {
      const scrollTop = index * itemHeight;
      containerRef.current.scrollTop = scrollTop;
    }
  }, [itemHeight]);

  return (
    <div
      ref={containerRef}
      className={`virtual-scroll ${className}`}
      style={{
        height: height,
        overflow: 'auto',
        ...style
      }}
      onScroll={handleScroll}
    >
      <div
        style={{
          height: visibleRange.totalHeight,
          position: 'relative'
        }}
      >
        <div
          style={{
            transform: `translateY(${visibleRange.offsetY}px)`
          }}
        >
          {visibleRange.visibleItems.map((item, index) => (
            <div
              key={visibleRange.startIndex + index}
              style={{
                height: itemHeight,
                overflow: 'hidden'
              }}
            >
              {renderItem(item, visibleRange.startIndex + index)}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
});

VirtualScroll.displayName = 'VirtualScroll';

// 动态高度虚拟滚动
interface DynamicVirtualScrollProps<T> {
  items: T[];
  height: number;
  estimatedItemSize: number;
  renderItem: (item: T, index: number) => React.ReactNode;
  getItemKey?: (item: T, index: number) => string | number;
  className?: string;
  style?: CSSProperties;
  overscan?: number;
}

export const DynamicVirtualScroll = memo(<T,>({
  items,
  height,
  estimatedItemSize,
  renderItem,
  getItemKey = (_, index) => index,
  className = '',
  style = {},
  overscan = 5
}: DynamicVirtualScrollProps<T>) => {
  const [scrollTop, setScrollTop] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const itemSizes = useRef<Map<number, number>>(new Map());
  const itemRefs = useRef<Map<number, HTMLDivElement>>(new Map());

  // 测量项目高度
  const measureItem = useCallback((index: number, element: HTMLDivElement) => {
    const height = element.getBoundingClientRect().height;
    itemSizes.current.set(index, height);
    itemRefs.current.set(index, element);
  }, []);

  // 计算项目偏移
  const getItemOffset = useCallback((index: number) => {
    let offset = 0;
    for (let i = 0; i < index; i++) {
      offset += itemSizes.current.get(i) || estimatedItemSize;
    }
    return offset;
  }, [estimatedItemSize]);

  // 计算总高度
  const getTotalHeight = useCallback(() => {
    let totalHeight = 0;
    for (let i = 0; i < items.length; i++) {
      totalHeight += itemSizes.current.get(i) || estimatedItemSize;
    }
    return totalHeight;
  }, [items.length, estimatedItemSize]);

  // 查找可见范围
  const findVisibleRange = useCallback(() => {
    const containerHeight = height;
    let startIndex = 0;
    let endIndex = items.length;
    let currentOffset = 0;

    // 找到开始索引
    for (let i = 0; i < items.length; i++) {
      const itemHeight = itemSizes.current.get(i) || estimatedItemSize;
      if (currentOffset + itemHeight > scrollTop) {
        startIndex = i;
        break;
      }
      currentOffset += itemHeight;
    }

    // 找到结束索引
    currentOffset = getItemOffset(startIndex);
    for (let i = startIndex; i < items.length; i++) {
      if (currentOffset > scrollTop + containerHeight) {
        endIndex = i;
        break;
      }
      const itemHeight = itemSizes.current.get(i) || estimatedItemSize;
      currentOffset += itemHeight;
    }

    return {
      startIndex: Math.max(0, startIndex - overscan),
      endIndex: Math.min(items.length, endIndex + overscan)
    };
  }, [items.length, height, scrollTop, estimatedItemSize, overscan, getItemOffset]);

  const { startIndex, endIndex } = findVisibleRange();
  const visibleItems = items.slice(startIndex, endIndex);

  const handleScroll = useCallback((event: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(event.currentTarget.scrollTop);
  }, []);

  return (
    <div
      ref={containerRef}
      className={`dynamic-virtual-scroll ${className}`}
      style={{
        height: height,
        overflow: 'auto',
        ...style
      }}
      onScroll={handleScroll}
    >
      <div style={{ height: getTotalHeight(), position: 'relative' }}>
        {visibleItems.map((item, index) => {
          const itemIndex = startIndex + index;
          const offset = getItemOffset(itemIndex);
          
          return (
            <div
              key={getItemKey(item, itemIndex)}
              ref={(el) => el && measureItem(itemIndex, el)}
              style={{
                position: 'absolute',
                top: offset,
                left: 0,
                right: 0,
                minHeight: estimatedItemSize
              }}
            >
              {renderItem(item, itemIndex)}
            </div>
          );
        })}
      </div>
    </div>
  );
});

DynamicVirtualScroll.displayName = 'DynamicVirtualScroll';

// 网格虚拟滚动
interface VirtualGridProps<T> {
  items: T[];
  itemWidth: number;
  itemHeight: number;
  containerWidth: number;
  containerHeight: number;
  renderItem: (item: T, index: number) => React.ReactNode;
  gap?: number;
  className?: string;
  style?: CSSProperties;
}

export const VirtualGrid = memo(<T,>({
  items,
  itemWidth,
  itemHeight,
  containerWidth,
  containerHeight,
  renderItem,
  gap = 0,
  className = '',
  style = {}
}: VirtualGridProps<T>) => {
  const [scrollTop, setScrollTop] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  const {
    columnsPerRow,
    totalRows,
    visibleStartRow,
    visibleEndRow,
    visibleItems,
    totalHeight,
    offsetY
  } = useMemo(() => {
    const columnsPerRow = Math.floor((containerWidth + gap) / (itemWidth + gap));
    const totalRows = Math.ceil(items.length / columnsPerRow);
    const rowHeight = itemHeight + gap;
    
    const visibleStartRow = Math.floor(scrollTop / rowHeight);
    const visibleRowCount = Math.ceil(containerHeight / rowHeight) + 1;
    const visibleEndRow = Math.min(visibleStartRow + visibleRowCount, totalRows);
    
    const startIndex = visibleStartRow * columnsPerRow;
    const endIndex = Math.min(visibleEndRow * columnsPerRow, items.length);
    
    return {
      columnsPerRow,
      totalRows,
      visibleStartRow,
      visibleEndRow,
      visibleItems: items.slice(startIndex, endIndex),
      totalHeight: totalRows * rowHeight,
      offsetY: visibleStartRow * rowHeight
    };
  }, [items, itemWidth, itemHeight, containerWidth, containerHeight, gap, scrollTop]);

  const handleScroll = useCallback((event: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(event.currentTarget.scrollTop);
  }, []);

  return (
    <div
      ref={containerRef}
      className={`virtual-grid ${className}`}
      style={{
        height: containerHeight,
        overflow: 'auto',
        ...style
      }}
      onScroll={handleScroll}
    >
      <div style={{ height: totalHeight, position: 'relative' }}>
        <div
          style={{
            transform: `translateY(${offsetY}px)`,
            display: 'grid',
            gridTemplateColumns: `repeat(${columnsPerRow}, ${itemWidth}px)`,
            gap: `${gap}px`
          }}
        >
          {visibleItems.map((item, index) => {
            const actualIndex = visibleStartRow * columnsPerRow + index;
            return (
              <div key={actualIndex} style={{ width: itemWidth, height: itemHeight }}>
                {renderItem(item, actualIndex)}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
});

VirtualGrid.displayName = 'VirtualGrid';
