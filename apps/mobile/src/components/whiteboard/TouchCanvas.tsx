import React, { useRef, useState, useCallback, useEffect } from 'react';
import {
  View,
  StyleSheet,
  PanResponder,
  Dimensions,
  GestureResponderEvent,
  PanResponderGestureState,
} from 'react-native';
import Svg, { Path, G, Defs, ClipPath, Rect } from 'react-native-svg';
import { PalmRejection } from './PalmRejection';
import { DrawingTool, DrawingPath, Point } from '../../types/whiteboard';

interface TouchCanvasProps {
  tool: DrawingTool;
  color: string;
  strokeWidth: number;
  onPathComplete: (path: DrawingPath) => void;
  paths: DrawingPath[];
  isDrawingEnabled: boolean;
}

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

export const TouchCanvas: React.FC<TouchCanvasProps> = ({
  tool,
  color,
  strokeWidth,
  onPathComplete,
  paths,
  isDrawingEnabled,
}) => {
  const [currentPath, setCurrentPath] = useState<Point[]>([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const palmRejection = useRef(new PalmRejection()).current;
  const pathIdRef = useRef(0);

  const generatePathData = (points: Point[]): string => {
    if (points.length < 2) return '';

    let pathData = `M ${points[0].x} ${points[0].y}`;
    
    // Use quadratic bezier curves for smoother lines
    for (let i = 1; i < points.length - 1; i++) {
      const xc = (points[i].x + points[i + 1].x) / 2;
      const yc = (points[i].y + points[i + 1].y) / 2;
      pathData += ` Q ${points[i].x} ${points[i].y} ${xc} ${yc}`;
    }
    
    // Add the last point
    if (points.length > 1) {
      pathData += ` L ${points[points.length - 1].x} ${points[points.length - 1].y}`;
    }
    
    return pathData;
  };

  const handleTouchStart = useCallback(
    (event: GestureResponderEvent) => {
      if (!isDrawingEnabled) return;

      const { locationX, locationY, timestamp } = event.nativeEvent;
      
      // Check if this is likely a palm touch
      if (palmRejection.isPalmTouch({ x: locationX, y: locationY, timestamp })) {
        return;
      }

      setIsDrawing(true);
      setCurrentPath([{ x: locationX, y: locationY, timestamp }]);
    },
    [isDrawingEnabled, palmRejection]
  );

  const handleTouchMove = useCallback(
    (event: GestureResponderEvent) => {
      if (!isDrawing || !isDrawingEnabled) return;

      const { locationX, locationY, timestamp } = event.nativeEvent;
      
      // Filter out palm movements
      if (palmRejection.isPalmTouch({ x: locationX, y: locationY, timestamp })) {
        return;
      }

      setCurrentPath((prevPath) => [
        ...prevPath,
        { x: locationX, y: locationY, timestamp },
      ]);
    },
    [isDrawing, isDrawingEnabled, palmRejection]
  );

  const handleTouchEnd = useCallback(() => {
    if (!isDrawing || currentPath.length < 2) {
      setIsDrawing(false);
      setCurrentPath([]);
      return;
    }

    const newPath: DrawingPath = {
      id: `path_${Date.now()}_${pathIdRef.current++}`,
      points: currentPath,
      tool,
      color,
      strokeWidth,
      pathData: generatePathData(currentPath),
    };

    onPathComplete(newPath);
    setIsDrawing(false);
    setCurrentPath([]);
  }, [isDrawing, currentPath, tool, color, strokeWidth, onPathComplete]);

  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: () => true,
      onPanResponderGrant: handleTouchStart,
      onPanResponderMove: handleTouchMove,
      onPanResponderRelease: handleTouchEnd,
      onPanResponderTerminate: handleTouchEnd,
    })
  ).current;

  // Update pan responder when handlers change
  useEffect(() => {
    panResponder.panHandlers = PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: () => true,
      onPanResponderGrant: handleTouchStart,
      onPanResponderMove: handleTouchMove,
      onPanResponderRelease: handleTouchEnd,
      onPanResponderTerminate: handleTouchEnd,
    }).panHandlers;
  }, [handleTouchStart, handleTouchMove, handleTouchEnd]);

  const renderPath = (path: DrawingPath) => {
    switch (path.tool) {
      case 'pen':
      case 'pencil':
        return (
          <Path
            key={path.id}
            d={path.pathData}
            stroke={path.color}
            strokeWidth={path.strokeWidth}
            fill="none"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeOpacity={path.tool === 'pencil' ? 0.7 : 1}
          />
        );
      case 'highlighter':
        return (
          <Path
            key={path.id}
            d={path.pathData}
            stroke={path.color}
            strokeWidth={path.strokeWidth * 2}
            fill="none"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeOpacity={0.3}
          />
        );
      case 'eraser':
        // Eraser is handled differently - it removes intersecting paths
        return null;
      default:
        return null;
    }
  };

  return (
    <View style={styles.container} {...panResponder.panHandlers}>
      <Svg style={StyleSheet.absoluteFill} width={screenWidth} height={screenHeight}>
        <Defs>
          <ClipPath id="canvasClip">
            <Rect x="0" y="0" width={screenWidth} height={screenHeight} />
          </ClipPath>
        </Defs>
        <G clipPath="url(#canvasClip)">
          {/* Render existing paths */}
          {paths.map((path) => renderPath(path))}
          
          {/* Render current path being drawn */}
          {isDrawing && currentPath.length > 1 && (
            <Path
              d={generatePathData(currentPath)}
              stroke={color}
              strokeWidth={strokeWidth}
              fill="none"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeOpacity={tool === 'pencil' ? 0.7 : tool === 'highlighter' ? 0.3 : 1}
            />
          )}
        </G>
      </Svg>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'white',
  },
});