import { Point } from '../../types/whiteboard';

interface TouchMetrics {
  area: number;
  velocity: number;
  angle: number;
  pressure?: number;
}

export class PalmRejection {
  private touchHistory: Point[] = [];
  private readonly historySize = 10;
  private readonly palmThreshold = {
    minArea: 30, // Minimum touch area that might indicate palm
    maxVelocity: 200, // Maximum velocity for legitimate drawing
    angleThreshold: 45, // Maximum angle change for smooth drawing
  };

  /**
   * Analyzes touch characteristics to determine if it's likely a palm touch
   */
  isPalmTouch(point: Point): boolean {
    // Add point to history
    this.touchHistory.push(point);
    if (this.touchHistory.length > this.historySize) {
      this.touchHistory.shift();
    }

    // Need at least 3 points to analyze
    if (this.touchHistory.length < 3) {
      return false;
    }

    const metrics = this.calculateTouchMetrics();

    // Check for palm characteristics
    return (
      metrics.area > this.palmThreshold.minArea ||
      metrics.velocity > this.palmThreshold.maxVelocity ||
      metrics.angle > this.palmThreshold.angleThreshold
    );
  }

  /**
   * Calculates metrics from touch history to identify palm touches
   */
  private calculateTouchMetrics(): TouchMetrics {
    const history = this.touchHistory;
    const len = history.length;

    // Calculate touch area (based on point spread)
    let minX = history[0].x, maxX = history[0].x;
    let minY = history[0].y, maxY = history[0].y;

    for (const point of history) {
      minX = Math.min(minX, point.x);
      maxX = Math.max(maxX, point.x);
      minY = Math.min(minY, point.y);
      maxY = Math.max(maxY, point.y);
    }

    const area = (maxX - minX) * (maxY - minY);

    // Calculate velocity
    const lastPoint = history[len - 1];
    const prevPoint = history[len - 2];
    const timeDiff = lastPoint.timestamp - prevPoint.timestamp;
    const distance = Math.sqrt(
      Math.pow(lastPoint.x - prevPoint.x, 2) +
      Math.pow(lastPoint.y - prevPoint.y, 2)
    );
    const velocity = timeDiff > 0 ? distance / timeDiff * 1000 : 0; // pixels per second

    // Calculate angle change
    let angle = 0;
    if (len >= 3) {
      const p1 = history[len - 3];
      const p2 = history[len - 2];
      const p3 = history[len - 1];

      const angle1 = Math.atan2(p2.y - p1.y, p2.x - p1.x);
      const angle2 = Math.atan2(p3.y - p2.y, p3.x - p2.x);
      
      angle = Math.abs(angle2 - angle1) * (180 / Math.PI);
      // Normalize angle to 0-180 range
      if (angle > 180) angle = 360 - angle;
    }

    return { area, velocity, angle };
  }

  /**
   * Resets the touch history
   */
  reset(): void {
    this.touchHistory = [];
  }

  /**
   * Adjusts palm rejection sensitivity
   */
  setSensitivity(level: 'low' | 'medium' | 'high'): void {
    switch (level) {
      case 'low':
        this.palmThreshold.minArea = 50;
        this.palmThreshold.maxVelocity = 300;
        this.palmThreshold.angleThreshold = 60;
        break;
      case 'medium':
        this.palmThreshold.minArea = 30;
        this.palmThreshold.maxVelocity = 200;
        this.palmThreshold.angleThreshold = 45;
        break;
      case 'high':
        this.palmThreshold.minArea = 20;
        this.palmThreshold.maxVelocity = 150;
        this.palmThreshold.angleThreshold = 30;
        break;
    }
  }
}