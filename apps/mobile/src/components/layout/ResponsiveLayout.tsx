import React, { ReactNode } from 'react';
import {
  View,
  ScrollView,
  StyleSheet,
  ViewStyle,
  SafeAreaView,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { isTablet, spacing, getSafeAreaPadding, responsive } from '../../utils/responsive';

interface ResponsiveLayoutProps {
  children: ReactNode;
  scrollable?: boolean;
  padded?: boolean;
  centered?: boolean;
  keyboardAware?: boolean;
  safeArea?: boolean;
  style?: ViewStyle;
  contentContainerStyle?: ViewStyle;
}

export const ResponsiveLayout: React.FC<ResponsiveLayoutProps> = ({
  children,
  scrollable = false,
  padded = true,
  centered = false,
  keyboardAware = true,
  safeArea = true,
  style,
  contentContainerStyle,
}) => {
  const Container = safeArea ? SafeAreaView : View;
  const safeAreaPadding = getSafeAreaPadding();

  const containerStyle: ViewStyle = {
    flex: 1,
    ...style,
  };

  const contentStyle: ViewStyle = {
    ...(padded && {
      paddingHorizontal: responsive(spacing.md, spacing.lg, spacing.xl),
      paddingVertical: spacing.md,
    }),
    ...(centered && {
      alignItems: 'center',
      justifyContent: 'center',
    }),
    ...contentContainerStyle,
  };

  const content = scrollable ? (
    <ScrollView
      style={styles.scrollView}
      contentContainerStyle={[styles.scrollContent, contentStyle]}
      showsVerticalScrollIndicator={false}
      keyboardShouldPersistTaps="handled"
    >
      {children}
    </ScrollView>
  ) : (
    <View style={[styles.content, contentStyle]}>{children}</View>
  );

  if (keyboardAware && !scrollable) {
    return (
      <Container style={containerStyle}>
        <KeyboardAvoidingView
          style={styles.keyboardAvoid}
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
          keyboardVerticalOffset={safeAreaPadding.top}
        >
          {content}
        </KeyboardAvoidingView>
      </Container>
    );
  }

  return <Container style={containerStyle}>{content}</Container>;
};

// Grid layout component
interface GridLayoutProps {
  children: ReactNode[];
  columns?: number;
  gap?: number;
  style?: ViewStyle;
}

export const GridLayout: React.FC<GridLayoutProps> = ({
  children,
  columns = isTablet ? 3 : 2,
  gap = spacing.sm,
  style,
}) => {
  const itemWidth = `${100 / columns}%`;

  return (
    <View style={[styles.gridContainer, style]}>
      {React.Children.map(children, (child, index) => (
        <View
          key={index}
          style={[
            styles.gridItem,
            {
              width: itemWidth,
              paddingRight: (index + 1) % columns === 0 ? 0 : gap,
              paddingBottom: gap,
            },
          ]}
        >
          {child}
        </View>
      ))}
    </View>
  );
};

// Flex row component with responsive wrapping
interface FlexRowProps {
  children: ReactNode;
  wrap?: boolean;
  gap?: number;
  align?: 'flex-start' | 'center' | 'flex-end' | 'stretch';
  justify?: 'flex-start' | 'center' | 'flex-end' | 'space-between' | 'space-around' | 'space-evenly';
  style?: ViewStyle;
}

export const FlexRow: React.FC<FlexRowProps> = ({
  children,
  wrap = true,
  gap = spacing.sm,
  align = 'center',
  justify = 'flex-start',
  style,
}) => {
  return (
    <View
      style={[
        styles.flexRow,
        {
          flexWrap: wrap ? 'wrap' : 'nowrap',
          alignItems: align,
          justifyContent: justify,
          gap,
        },
        style,
      ]}
    >
      {children}
    </View>
  );
};

// Adaptive container that changes layout based on screen size
interface AdaptiveContainerProps {
  children: ReactNode;
  mobileLayout: 'column' | 'row';
  tabletLayout: 'column' | 'row';
  gap?: number;
  style?: ViewStyle;
}

export const AdaptiveContainer: React.FC<AdaptiveContainerProps> = ({
  children,
  mobileLayout,
  tabletLayout,
  gap = spacing.md,
  style,
}) => {
  const flexDirection = isTablet ? tabletLayout : mobileLayout;

  return (
    <View
      style={[
        {
          flexDirection,
          gap,
        },
        style,
      ]}
    >
      {children}
    </View>
  );
};

const styles = StyleSheet.create({
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
  },
  content: {
    flex: 1,
  },
  keyboardAvoid: {
    flex: 1,
  },
  gridContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  gridItem: {
    flexDirection: 'row',
  },
  flexRow: {
    flexDirection: 'row',
  },
});