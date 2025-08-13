import { PalmRejection } from '../../../src/components/whiteboard/PalmRejection';
import { Point } from '../../../src/types/whiteboard';

describe('PalmRejection', () => {
  let palmRejection: PalmRejection;

  beforeEach(() => {
    palmRejection = new PalmRejection();
  });

  it('should not reject initial touches', () => {
    const point: Point = { x: 100, y: 100, timestamp: Date.now() };
    expect(palmRejection.isPalmTouch(point)).toBe(false);
  });

  it('should reject touches with large area spread', () => {
    const timestamp = Date.now();
    const points: Point[] = [
      { x: 100, y: 100, timestamp },
      { x: 150, y: 150, timestamp: timestamp + 10 },
      { x: 200, y: 200, timestamp: timestamp + 20 },
    ];

    points.forEach(point => palmRejection.isPalmTouch(point));
    
    // Large spread should be detected as palm
    const palmPoint: Point = { x: 300, y: 300, timestamp: timestamp + 30 };
    expect(palmRejection.isPalmTouch(palmPoint)).toBe(true);
  });

  it('should reject touches with high velocity', () => {
    const timestamp = Date.now();
    
    // Normal drawing speed
    palmRejection.isPalmTouch({ x: 100, y: 100, timestamp });
    palmRejection.isPalmTouch({ x: 105, y: 105, timestamp: timestamp + 50 });
    
    // Very fast movement (likely palm)
    expect(
      palmRejection.isPalmTouch({ x: 300, y: 300, timestamp: timestamp + 60 })
    ).toBe(true);
  });

  it('should reject touches with sharp angle changes', () => {
    const timestamp = Date.now();
    
    // Draw a line
    palmRejection.isPalmTouch({ x: 100, y: 100, timestamp });
    palmRejection.isPalmTouch({ x: 150, y: 100, timestamp: timestamp + 50 });
    
    // Sharp angle change (likely palm)
    expect(
      palmRejection.isPalmTouch({ x: 150, y: 200, timestamp: timestamp + 100 })
    ).toBe(false); // Need more points for angle detection
    
    // Add another point for angle calculation
    palmRejection.isPalmTouch({ x: 100, y: 250, timestamp: timestamp + 150 });
    expect(
      palmRejection.isPalmTouch({ x: 200, y: 250, timestamp: timestamp + 200 })
    ).toBe(true);
  });

  it('should reset touch history', () => {
    const timestamp = Date.now();
    
    // Add some points
    palmRejection.isPalmTouch({ x: 100, y: 100, timestamp });
    palmRejection.isPalmTouch({ x: 200, y: 200, timestamp: timestamp + 50 });
    
    // Reset
    palmRejection.reset();
    
    // Should not reject after reset
    expect(
      palmRejection.isPalmTouch({ x: 300, y: 300, timestamp: timestamp + 100 })
    ).toBe(false);
  });

  it('should adjust sensitivity levels', () => {
    const timestamp = Date.now();
    const points: Point[] = [
      { x: 100, y: 100, timestamp },
      { x: 120, y: 120, timestamp: timestamp + 50 },
      { x: 140, y: 140, timestamp: timestamp + 100 },
    ];

    // High sensitivity
    palmRejection.setSensitivity('high');
    points.forEach(point => palmRejection.isPalmTouch(point));
    
    const testPoint: Point = { x: 160, y: 160, timestamp: timestamp + 150 };
    const highSensitivityResult = palmRejection.isPalmTouch(testPoint);

    // Reset and test with low sensitivity
    palmRejection.reset();
    palmRejection.setSensitivity('low');
    points.forEach(point => palmRejection.isPalmTouch(point));
    
    const lowSensitivityResult = palmRejection.isPalmTouch(testPoint);

    // High sensitivity should be more likely to reject
    expect(highSensitivityResult || !lowSensitivityResult).toBe(true);
  });

  it('should maintain a limited history size', () => {
    const timestamp = Date.now();
    
    // Add more than history size limit
    for (let i = 0; i < 20; i++) {
      palmRejection.isPalmTouch({
        x: i * 10,
        y: i * 10,
        timestamp: timestamp + i * 10,
      });
    }

    // Should still function correctly
    expect(
      palmRejection.isPalmTouch({ x: 250, y: 250, timestamp: timestamp + 210 })
    ).toBeDefined();
  });
});