import { Dimensions, PixelRatio, Platform } from 'react-native';

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

// Device size categories
export const isSmallDevice = screenWidth < 375;
export const isMediumDevice = screenWidth >= 375 && screenWidth < 414;
export const isLargeDevice = screenWidth >= 414;
export const isTablet = screenWidth >= 768;

// Orientation
export const isPortrait = () => {
  const { width, height } = Dimensions.get('window');
  return height > width;
};

export const isLandscape = () => !isPortrait();

// Responsive scaling functions
export const scale = (size: number): number => {
  const baseWidth = 375; // iPhone 11 Pro
  const scaleFactor = screenWidth / baseWidth;
  const newSize = size * scaleFactor;
  
  if (isTablet) {
    // Limit scaling on tablets
    return Math.round(PixelRatio.roundToNearestPixel(newSize * 0.8));
  }
  
  return Math.round(PixelRatio.roundToNearestPixel(newSize));
};

export const verticalScale = (size: number): number => {
  const baseHeight = 812; // iPhone 11 Pro
  const scaleFactor = screenHeight / baseHeight;
  const newSize = size * scaleFactor;
  
  if (isTablet) {
    // Limit scaling on tablets
    return Math.round(PixelRatio.roundToNearestPixel(newSize * 0.8));
  }
  
  return Math.round(PixelRatio.roundToNearestPixel(newSize));
};

export const moderateScale = (size: number, factor = 0.5): number => {
  return size + (scale(size) - size) * factor;
};

// Font scaling with accessibility support
export const fontScale = (size: number): number => {
  const fontScaleFactor = PixelRatio.getFontScale();
  const scaledSize = size * fontScaleFactor;
  
  // Limit maximum font size for better layout stability
  const maxScale = 1.5;
  const minScale = 0.85;
  
  if (fontScaleFactor > maxScale) {
    return Math.round(size * maxScale);
  } else if (fontScaleFactor < minScale) {
    return Math.round(size * minScale);
  }
  
  return Math.round(scaledSize);
};

// Spacing scale
export const spacing = {
  xs: scale(4),
  sm: scale(8),
  md: scale(16),
  lg: scale(24),
  xl: scale(32),
  xxl: scale(48),
};

// Common dimensions
export const dimensions = {
  screenWidth,
  screenHeight,
  navBarHeight: Platform.select({
    ios: isTablet ? 65 : 44,
    android: 56,
    default: 56,
  }),
  tabBarHeight: Platform.select({
    ios: isTablet ? 65 : 49,
    android: 56,
    default: 56,
  }),
  statusBarHeight: Platform.select({
    ios: isPortrait() ? 44 : 0,
    android: 0,
    default: 0,
  }),
  headerHeight: Platform.select({
    ios: 44,
    android: 56,
    default: 56,
  }),
};

// Breakpoints for responsive design
export const breakpoints = {
  small: 375,
  medium: 414,
  large: 768,
  xlarge: 1024,
};

// Helper function to get responsive value based on screen size
export const responsive = <T extends any>(
  small: T,
  medium?: T,
  large?: T,
  xlarge?: T
): T => {
  if (screenWidth < breakpoints.medium) {
    return small;
  } else if (screenWidth < breakpoints.large) {
    return medium ?? small;
  } else if (screenWidth < breakpoints.xlarge) {
    return large ?? medium ?? small;
  } else {
    return xlarge ?? large ?? medium ?? small;
  }
};

// Grid system
export const grid = {
  columns: isTablet ? 12 : 6,
  gutter: spacing.md,
  marginHorizontal: responsive(spacing.md, spacing.lg, spacing.xl),
};

// Safe area helpers
export const getSafeAreaPadding = () => {
  return {
    top: Platform.select({
      ios: isPortrait() ? 44 : 20,
      android: 0,
      default: 0,
    }),
    bottom: Platform.select({
      ios: isPortrait() ? 34 : 0,
      android: 0,
      default: 0,
    }),
    left: Platform.select({
      ios: isLandscape() ? 44 : 0,
      android: 0,
      default: 0,
    }),
    right: Platform.select({
      ios: isLandscape() ? 44 : 0,
      android: 0,
      default: 0,
    }),
  };
};

// Typography scale
export const typography = {
  h1: {
    fontSize: fontScale(responsive(28, 32, 36)),
    lineHeight: responsive(34, 38, 42),
    fontWeight: '700' as const,
  },
  h2: {
    fontSize: fontScale(responsive(24, 26, 28)),
    lineHeight: responsive(30, 32, 34),
    fontWeight: '600' as const,
  },
  h3: {
    fontSize: fontScale(responsive(20, 22, 24)),
    lineHeight: responsive(26, 28, 30),
    fontWeight: '600' as const,
  },
  body1: {
    fontSize: fontScale(responsive(16, 16, 18)),
    lineHeight: responsive(22, 22, 24),
    fontWeight: '400' as const,
  },
  body2: {
    fontSize: fontScale(responsive(14, 14, 16)),
    lineHeight: responsive(20, 20, 22),
    fontWeight: '400' as const,
  },
  caption: {
    fontSize: fontScale(responsive(12, 12, 14)),
    lineHeight: responsive(16, 16, 18),
    fontWeight: '400' as const,
  },
};