import React from 'react';
import {
  View,
  TouchableOpacity,
  Text,
  StyleSheet,
  ScrollView,
  Animated,
} from 'react-native';
import { MaterialIcons, MaterialCommunityIcons } from '@expo/vector-icons';
import { DrawingTool } from '../../types/whiteboard';

interface DrawingToolsPaletteProps {
  currentTool: DrawingTool;
  currentColor: string;
  currentStrokeWidth: number;
  onToolChange: (tool: DrawingTool) => void;
  onColorChange: (color: string) => void;
  onStrokeWidthChange: (width: number) => void;
  onUndo: () => void;
  onRedo: () => void;
  onClear: () => void;
  canUndo: boolean;
  canRedo: boolean;
}

const COLORS = [
  '#000000', // Black
  '#FF0000', // Red
  '#00FF00', // Green
  '#0000FF', // Blue
  '#FFFF00', // Yellow
  '#FF00FF', // Magenta
  '#00FFFF', // Cyan
  '#FFA500', // Orange
  '#800080', // Purple
  '#FFC0CB', // Pink
];

const STROKE_WIDTHS = [2, 4, 6, 8, 10];

export const DrawingToolsPalette: React.FC<DrawingToolsPaletteProps> = ({
  currentTool,
  currentColor,
  currentStrokeWidth,
  onToolChange,
  onColorChange,
  onStrokeWidthChange,
  onUndo,
  onRedo,
  onClear,
  canUndo,
  canRedo,
}) => {
  const [isExpanded, setIsExpanded] = React.useState(false);
  const animatedHeight = React.useRef(new Animated.Value(60)).current;

  const toggleExpanded = () => {
    const toValue = isExpanded ? 60 : 200;
    Animated.spring(animatedHeight, {
      toValue,
      useNativeDriver: false,
      friction: 8,
    }).start();
    setIsExpanded(!isExpanded);
  };

  const tools: Array<{ name: DrawingTool; icon: string; library: 'material' | 'material-community' }> = [
    { name: 'pen', icon: 'create', library: 'material' },
    { name: 'pencil', icon: 'pencil', library: 'material-community' },
    { name: 'highlighter', icon: 'highlighter', library: 'material-community' },
    { name: 'eraser', icon: 'eraser', library: 'material-community' },
  ];

  return (
    <Animated.View style={[styles.container, { height: animatedHeight }]}>
      {/* Main toolbar */}
      <View style={styles.mainToolbar}>
        {/* Tools */}
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.toolsContainer}
        >
          {tools.map((tool) => (
            <TouchableOpacity
              key={tool.name}
              style={[
                styles.toolButton,
                currentTool === tool.name && styles.toolButtonActive,
              ]}
              onPress={() => onToolChange(tool.name)}
            >
              {tool.library === 'material' ? (
                <MaterialIcons
                  name={tool.icon as any}
                  size={24}
                  color={currentTool === tool.name ? '#007AFF' : '#333'}
                />
              ) : (
                <MaterialCommunityIcons
                  name={tool.icon as any}
                  size={24}
                  color={currentTool === tool.name ? '#007AFF' : '#333'}
                />
              )}
            </TouchableOpacity>
          ))}
        </ScrollView>

        {/* Actions */}
        <View style={styles.actionsContainer}>
          <TouchableOpacity
            style={[styles.actionButton, !canUndo && styles.actionButtonDisabled]}
            onPress={onUndo}
            disabled={!canUndo}
          >
            <MaterialIcons name="undo" size={24} color={canUndo ? '#333' : '#ccc'} />
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.actionButton, !canRedo && styles.actionButtonDisabled]}
            onPress={onRedo}
            disabled={!canRedo}
          >
            <MaterialIcons name="redo" size={24} color={canRedo ? '#333' : '#ccc'} />
          </TouchableOpacity>
          <TouchableOpacity style={styles.actionButton} onPress={onClear}>
            <MaterialIcons name="clear" size={24} color="#FF3B30" />
          </TouchableOpacity>
          <TouchableOpacity style={styles.expandButton} onPress={toggleExpanded}>
            <MaterialIcons
              name={isExpanded ? 'expand-less' : 'expand-more'}
              size={24}
              color="#333"
            />
          </TouchableOpacity>
        </View>
      </View>

      {/* Expanded options */}
      {isExpanded && (
        <View style={styles.expandedContainer}>
          {/* Color palette */}
          <View style={styles.colorPalette}>
            <Text style={styles.label}>Colors</Text>
            <ScrollView
              horizontal
              showsHorizontalScrollIndicator={false}
              contentContainerStyle={styles.colorsContainer}
            >
              {COLORS.map((color) => (
                <TouchableOpacity
                  key={color}
                  style={[
                    styles.colorButton,
                    { backgroundColor: color },
                    currentColor === color && styles.colorButtonActive,
                  ]}
                  onPress={() => onColorChange(color)}
                />
              ))}
            </ScrollView>
          </View>

          {/* Stroke width */}
          <View style={styles.strokeWidthContainer}>
            <Text style={styles.label}>Stroke Width</Text>
            <ScrollView
              horizontal
              showsHorizontalScrollIndicator={false}
              contentContainerStyle={styles.strokeWidthsContainer}
            >
              {STROKE_WIDTHS.map((width) => (
                <TouchableOpacity
                  key={width}
                  style={[
                    styles.strokeWidthButton,
                    currentStrokeWidth === width && styles.strokeWidthButtonActive,
                  ]}
                  onPress={() => onStrokeWidthChange(width)}
                >
                  <View
                    style={[
                      styles.strokeWidthIndicator,
                      { height: width, backgroundColor: currentColor },
                    ]}
                  />
                </TouchableOpacity>
              ))}
            </ScrollView>
          </View>
        </View>
      )}
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'white',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 5,
  },
  mainToolbar: {
    height: 60,
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
  },
  toolsContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  toolButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#f0f0f0',
    alignItems: 'center',
    justifyContent: 'center',
    marginHorizontal: 5,
  },
  toolButtonActive: {
    backgroundColor: '#E3F2FD',
  },
  actionsContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginLeft: 'auto',
  },
  actionButton: {
    width: 44,
    height: 44,
    alignItems: 'center',
    justifyContent: 'center',
  },
  actionButtonDisabled: {
    opacity: 0.5,
  },
  expandButton: {
    width: 44,
    height: 44,
    alignItems: 'center',
    justifyContent: 'center',
    marginLeft: 10,
  },
  expandedContainer: {
    flex: 1,
    paddingHorizontal: 15,
    paddingTop: 10,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 8,
  },
  colorPalette: {
    marginBottom: 20,
  },
  colorsContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  colorButton: {
    width: 32,
    height: 32,
    borderRadius: 16,
    marginHorizontal: 4,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  colorButtonActive: {
    borderColor: '#007AFF',
  },
  strokeWidthContainer: {
    marginBottom: 10,
  },
  strokeWidthsContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  strokeWidthButton: {
    width: 50,
    height: 40,
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    marginHorizontal: 4,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  strokeWidthButtonActive: {
    borderColor: '#007AFF',
    backgroundColor: '#E3F2FD',
  },
  strokeWidthIndicator: {
    width: 30,
    borderRadius: 2,
  },
});