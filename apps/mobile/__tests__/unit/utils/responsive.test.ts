import { Dimensions, PixelRatio } from 'react-native';
import {
  scale,
  verticalScale,
  moderateScale,
  fontScale,
  responsive,
  isSmallDevice,
  isMediumDevice,
  isLargeDevice,
  isTablet,
} from '../../../src/utils/responsive';

// Mock React Native modules
jest.mock('react-native', () => ({
  Dimensions: {
    get: jest.fn(() => ({ width: 375, height: 812 })),
  },
  PixelRatio: {
    roundToNearestPixel: jest.fn((size) => Math.round(size)),
    getFontScale: jest.fn(() => 1),
  },
  Platform: {
    select: jest.fn((obj) => obj.ios || obj.default),
  },
}));

describe('Responsive Utilities', () => {
  const mockDimensions = Dimensions as jest.Mocked<typeof Dimensions>;
  const mockPixelRatio = PixelRatio as jest.Mocked<typeof PixelRatio>;

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('scale', () => {
    it('should scale size based on screen width', () => {
      mockDimensions.get.mockReturnValue({ width: 375, height: 812 });
      expect(scale(10)).toBe(10); // Base width, no scaling

      mockDimensions.get.mockReturnValue({ width: 414, height: 896 });
      expect(scale(10)).toBe(11); // Larger screen, scaled up

      mockDimensions.get.mockReturnValue({ width: 320, height: 568 });
      expect(scale(10)).toBe(9); // Smaller screen, scaled down
    });

    it('should limit scaling on tablets', () => {
      mockDimensions.get.mockReturnValue({ width: 768, height: 1024 });
      const scaledValue = scale(10);
      expect(scaledValue).toBeLessThan(20); // Should be limited
    });
  });

  describe('verticalScale', () => {
    it('should scale size based on screen height', () => {
      mockDimensions.get.mockReturnValue({ width: 375, height: 812 });
      expect(verticalScale(10)).toBe(10); // Base height, no scaling

      mockDimensions.get.mockReturnValue({ width: 375, height: 896 });
      expect(verticalScale(10)).toBe(11); // Taller screen, scaled up
    });
  });

  describe('moderateScale', () => {
    it('should apply moderate scaling with custom factor', () => {
      mockDimensions.get.mockReturnValue({ width: 414, height: 896 });
      
      const fullScale = scale(10);
      const halfScale = moderateScale(10, 0.5);
      const quarterScale = moderateScale(10, 0.25);

      expect(halfScale).toBeGreaterThan(10);
      expect(halfScale).toBeLessThan(fullScale);
      expect(quarterScale).toBeLessThan(halfScale);
    });
  });

  describe('fontScale', () => {
    it('should scale font based on system font scale', () => {
      mockPixelRatio.getFontScale.mockReturnValue(1);
      expect(fontScale(16)).toBe(16);

      mockPixelRatio.getFontScale.mockReturnValue(1.5);
      expect(fontScale(16)).toBe(24);
    });

    it('should limit maximum font scaling', () => {
      mockPixelRatio.getFontScale.mockReturnValue(2);
      expect(fontScale(16)).toBe(24); // Limited to 1.5x

      mockPixelRatio.getFontScale.mockReturnValue(0.5);
      expect(fontScale(16)).toBe(14); // Limited to 0.85x minimum
    });
  });

  describe('device size detection', () => {
    it('should detect small devices', () => {
      mockDimensions.get.mockReturnValue({ width: 320, height: 568 });
      expect(isSmallDevice).toBe(true);
      expect(isMediumDevice).toBe(false);
      expect(isLargeDevice).toBe(false);
      expect(isTablet).toBe(false);
    });

    it('should detect medium devices', () => {
      mockDimensions.get.mockReturnValue({ width: 375, height: 812 });
      expect(isSmallDevice).toBe(false);
      expect(isMediumDevice).toBe(true);
      expect(isLargeDevice).toBe(false);
      expect(isTablet).toBe(false);
    });

    it('should detect tablets', () => {
      mockDimensions.get.mockReturnValue({ width: 768, height: 1024 });
      expect(isTablet).toBe(true);
      expect(isLargeDevice).toBe(true);
    });
  });

  describe('responsive helper', () => {
    it('should return correct value based on screen size', () => {
      mockDimensions.get.mockReturnValue({ width: 320, height: 568 });
      expect(responsive('small', 'medium', 'large')).toBe('small');

      mockDimensions.get.mockReturnValue({ width: 414, height: 896 });
      expect(responsive('small', 'medium', 'large')).toBe('medium');

      mockDimensions.get.mockReturnValue({ width: 768, height: 1024 });
      expect(responsive('small', 'medium', 'large')).toBe('large');
    });

    it('should fallback to previous values when not specified', () => {
      mockDimensions.get.mockReturnValue({ width: 768, height: 1024 });
      expect(responsive('small', 'medium')).toBe('medium');
      expect(responsive('small')).toBe('small');
    });
  });
});