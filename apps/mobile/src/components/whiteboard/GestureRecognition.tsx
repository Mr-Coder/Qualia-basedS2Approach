import { Point } from '../../types/whiteboard';

export type GestureType = 'tap' | 'double_tap' | 'long_press' | 'swipe' | 'pinch' | 'two_finger_tap';

export interface Gesture {
  type: GestureType;
  points: Point[];
  timestamp: number;
  data?: any;
}

export class GestureRecognition {
  private touchPoints: Map<number, Point[]> = new Map();
  private gestureTimeout?: NodeJS.Timeout;
  private lastTapTime = 0;
  private readonly doubleTapThreshold = 300; // ms
  private readonly longPressThreshold = 500; // ms
  private readonly swipeThreshold = 50; // pixels

  onGesture?: (gesture: Gesture) => void;

  handleTouchStart(touches: Array<{ identifier: number; x: number; y: number; timestamp: number }>) {
    touches.forEach((touch) => {
      this.touchPoints.set(touch.identifier, [{
        x: touch.x,
        y: touch.y,
        timestamp: touch.timestamp,
      }]);
    });

    // Detect long press
    if (this.touchPoints.size === 1) {
      this.gestureTimeout = setTimeout(() => {
        const points = Array.from(this.touchPoints.values())[0];
        if (points && points.length === 1) {
          this.emitGesture({
            type: 'long_press',
            points,
            timestamp: Date.now(),
          });
        }
      }, this.longPressThreshold);
    }

    // Detect two-finger tap
    if (this.touchPoints.size === 2) {
      this.gestureTimeout = setTimeout(() => {
        if (this.touchPoints.size === 2) {
          const allPoints = Array.from(this.touchPoints.values()).flat();
          this.emitGesture({
            type: 'two_finger_tap',
            points: allPoints,
            timestamp: Date.now(),
          });
        }
      }, 200);
    }
  }

  handleTouchMove(touches: Array<{ identifier: number; x: number; y: number; timestamp: number }>) {
    // Clear any pending gesture detection
    if (this.gestureTimeout) {
      clearTimeout(this.gestureTimeout);
      this.gestureTimeout = undefined;
    }

    touches.forEach((touch) => {
      const points = this.touchPoints.get(touch.identifier);
      if (points) {
        points.push({
          x: touch.x,
          y: touch.y,
          timestamp: touch.timestamp,
        });
      }
    });

    // Detect pinch gesture
    if (this.touchPoints.size === 2) {
      const [points1, points2] = Array.from(this.touchPoints.values());
      if (points1.length > 1 && points2.length > 1) {
        const initialDistance = this.getDistance(points1[0], points2[0]);
        const currentDistance = this.getDistance(
          points1[points1.length - 1],
          points2[points2.length - 1]
        );
        
        const scale = currentDistance / initialDistance;
        if (Math.abs(scale - 1) > 0.1) {
          this.emitGesture({
            type: 'pinch',
            points: [...points1, ...points2],
            timestamp: Date.now(),
            data: { scale },
          });
        }
      }
    }
  }

  handleTouchEnd(touches: Array<{ identifier: number }>) {
    if (this.gestureTimeout) {
      clearTimeout(this.gestureTimeout);
      this.gestureTimeout = undefined;
    }

    // Check for tap or swipe
    if (this.touchPoints.size === 1) {
      const points = Array.from(this.touchPoints.values())[0];
      if (points && points.length > 0) {
        const startPoint = points[0];
        const endPoint = points[points.length - 1];
        const distance = this.getDistance(startPoint, endPoint);

        if (distance < 10 && points.length < 5) {
          // Tap detected
          const now = Date.now();
          if (now - this.lastTapTime < this.doubleTapThreshold) {
            // Double tap
            this.emitGesture({
              type: 'double_tap',
              points: [endPoint],
              timestamp: now,
            });
            this.lastTapTime = 0;
          } else {
            // Single tap
            this.emitGesture({
              type: 'tap',
              points: [endPoint],
              timestamp: now,
            });
            this.lastTapTime = now;
          }
        } else if (distance > this.swipeThreshold) {
          // Swipe detected
          const direction = this.getSwipeDirection(startPoint, endPoint);
          this.emitGesture({
            type: 'swipe',
            points,
            timestamp: Date.now(),
            data: { direction },
          });
        }
      }
    }

    // Remove ended touches
    touches.forEach((touch) => {
      this.touchPoints.delete(touch.identifier);
    });
  }

  private getDistance(p1: Point, p2: Point): number {
    return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
  }

  private getSwipeDirection(start: Point, end: Point): string {
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const absDx = Math.abs(dx);
    const absDy = Math.abs(dy);

    if (absDx > absDy) {
      return dx > 0 ? 'right' : 'left';
    } else {
      return dy > 0 ? 'down' : 'up';
    }
  }

  private emitGesture(gesture: Gesture) {
    if (this.onGesture) {
      this.onGesture(gesture);
    }
  }

  reset() {
    this.touchPoints.clear();
    if (this.gestureTimeout) {
      clearTimeout(this.gestureTimeout);
      this.gestureTimeout = undefined;
    }
  }
}