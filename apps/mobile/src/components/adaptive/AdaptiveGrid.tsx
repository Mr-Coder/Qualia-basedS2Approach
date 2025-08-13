import React, { ReactNode } from 'react';
import { View, StyleSheet, ScrollView, ViewStyle } from 'react-native';
import { isTablet, spacing, responsive, breakpoints, Dimensions } from '../../utils/responsive';

interface AdaptiveGridProps {
  children: ReactNode[];
  minItemWidth?: number;
  maxColumns?: number;
  gap?: number;
  scrollable?: boolean;
  style?: ViewStyle;
  itemStyle?: ViewStyle;
}

export const AdaptiveGrid: React.FC<AdaptiveGridProps> = ({
  children,
  minItemWidth = 150,
  maxColumns = isTablet ? 4 : 2,
  gap = spacing.md,
  scrollable = false,
  style,
  itemStyle,
}) => {
  const screenWidth = Dimensions.get('window').width;
  const containerPadding = responsive(spacing.md, spacing.lg, spacing.xl);
  const availableWidth = screenWidth - containerPadding * 2;

  // Calculate optimal number of columns
  const possibleColumns = Math.floor((availableWidth + gap) / (minItemWidth + gap));
  const columns = Math.min(Math.max(1, possibleColumns), maxColumns);
  
  // Calculate item width
  const itemWidth = (availableWidth - gap * (columns - 1)) / columns;

  const renderGrid = () => {
    const rows: ReactNode[][] = [];
    
    React.Children.forEach(children, (child, index) => {
      const rowIndex = Math.floor(index / columns);
      if (!rows[rowIndex]) {
        rows[rowIndex] = [];
      }
      rows[rowIndex].push(child);
    });

    return rows.map((row, rowIndex) => (
      <View key={rowIndex} style={[styles.row, { marginBottom: rowIndex < rows.length - 1 ? gap : 0 }]}>
        {row.map((child, colIndex) => (
          <View
            key={colIndex}
            style={[
              styles.gridItem,
              {
                width: itemWidth,
                marginRight: colIndex < columns - 1 ? gap : 0,
              },
              itemStyle,
            ]}
          >
            {child}
          </View>
        ))}
        {/* Fill empty cells in the last row */}
        {row.length < columns &&
          Array.from({ length: columns - row.length }).map((_, index) => (
            <View
              key={`empty-${index}`}
              style={{
                width: itemWidth,
                marginRight: row.length + index < columns - 1 ? gap : 0,
              }}
            />
          ))}
      </View>
    ));
  };

  const content = <View style={[styles.container, style]}>{renderGrid()}</View>;

  if (scrollable) {
    return (
      <ScrollView
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
      >
        {content}
      </ScrollView>
    );
  }

  return content;
};

// Masonry layout for varied height items
interface AdaptiveMasonryProps {
  children: ReactNode[];
  columns?: number;
  gap?: number;
  style?: ViewStyle;
}

export const AdaptiveMasonry: React.FC<AdaptiveMasonryProps> = ({
  children,
  columns = isTablet ? 3 : 2,
  gap = spacing.md,
  style,
}) => {
  const columnWrappers: ReactNode[][] = Array.from({ length: columns }, () => []);

  // Distribute children across columns
  React.Children.forEach(children, (child, index) => {
    const columnIndex = index % columns;
    columnWrappers[columnIndex].push(child);
  });

  return (
    <View style={[styles.masonryContainer, style]}>
      {columnWrappers.map((column, columnIndex) => (
        <View
          key={columnIndex}
          style={[
            styles.masonryColumn,
            {
              marginRight: columnIndex < columns - 1 ? gap : 0,
            },
          ]}
        >
          {column.map((child, index) => (
            <View
              key={index}
              style={{
                marginBottom: index < column.length - 1 ? gap : 0,
              }}
            >
              {child}
            </View>
          ))}
        </View>
      ))}
    </View>
  );
};

// Carousel for horizontal scrolling on small screens
interface AdaptiveCarouselProps {
  children: ReactNode[];
  itemWidth?: number | 'auto';
  gap?: number;
  showsIndicator?: boolean;
  style?: ViewStyle;
}

export const AdaptiveCarousel: React.FC<AdaptiveCarouselProps> = ({
  children,
  itemWidth = 'auto',
  gap = spacing.md,
  showsIndicator = false,
  style,
}) => {
  const screenWidth = Dimensions.get('window').width;
  const calculatedItemWidth = itemWidth === 'auto' ? screenWidth * 0.8 : itemWidth;

  return (
    <ScrollView
      horizontal
      pagingEnabled={itemWidth !== 'auto'}
      showsHorizontalScrollIndicator={showsIndicator}
      contentContainerStyle={[styles.carouselContent, { paddingHorizontal: gap }]}
      style={[styles.carousel, style]}
      snapToInterval={itemWidth === 'auto' ? undefined : calculatedItemWidth + gap}
      decelerationRate="fast"
    >
      {React.Children.map(children, (child, index) => (
        <View
          style={[
            styles.carouselItem,
            {
              width: calculatedItemWidth,
              marginRight: index < children.length - 1 ? gap : 0,
            },
          ]}
        >
          {child}
        </View>
      ))}
    </ScrollView>
  );
};

// Adaptive container that switches between grid and carousel
interface AdaptiveContainerProps {
  children: ReactNode[];
  mode?: 'auto' | 'grid' | 'carousel' | 'masonry';
  threshold?: number;
  gridProps?: Partial<AdaptiveGridProps>;
  carouselProps?: Partial<AdaptiveCarouselProps>;
  style?: ViewStyle;
}

export const AdaptiveContainer: React.FC<AdaptiveContainerProps> = ({
  children,
  mode = 'auto',
  threshold = breakpoints.medium,
  gridProps = {},
  carouselProps = {},
  style,
}) => {
  const screenWidth = Dimensions.get('window').width;
  
  const shouldUseCarousel = mode === 'carousel' || (mode === 'auto' && screenWidth < threshold);
  const shouldUseMasonry = mode === 'masonry';

  if (shouldUseCarousel) {
    return <AdaptiveCarousel {...carouselProps} style={style}>{children}</AdaptiveCarousel>;
  }

  if (shouldUseMasonry) {
    return <AdaptiveMasonry style={style}>{children}</AdaptiveMasonry>;
  }

  return <AdaptiveGrid {...gridProps} style={style}>{children}</AdaptiveGrid>;
};

const styles = StyleSheet.create({
  container: {
    width: '100%',
  },
  scrollContent: {
    flexGrow: 1,
  },
  row: {
    flexDirection: 'row',
    width: '100%',
  },
  gridItem: {
    flexDirection: 'row',
  },
  masonryContainer: {
    flexDirection: 'row',
    width: '100%',
  },
  masonryColumn: {
    flex: 1,
  },
  carousel: {
    width: '100%',
  },
  carouselContent: {
    alignItems: 'center',
  },
  carouselItem: {
    height: '100%',
  },
});