import React, { useState, useCallback, useEffect, useRef } from 'react';
import { View, StyleSheet, SafeAreaView, ActivityIndicator, Text } from 'react-native';
import { useRoute, RouteProp } from '@react-navigation/native';
import { TouchCanvas } from '../components/whiteboard/TouchCanvas';
import { DrawingToolsPalette } from '../components/whiteboard/DrawingToolsPalette';
import { GestureRecognition, Gesture } from '../components/whiteboard/GestureRecognition';
import { WhiteboardSyncService } from '../services/whiteboard/whiteboardSync';
import { DrawingPath, DrawingTool, WhiteboardState } from '../types/whiteboard';
import { useAuthStore } from '../stores/authStore';

type WhiteboardScreenRouteProp = RouteProp<{ Whiteboard: { roomId: string } }, 'Whiteboard'>;

const WhiteboardScreen: React.FC = () => {
  const route = useRoute<WhiteboardScreenRouteProp>();
  const { roomId } = route.params || { roomId: 'default' };
  const { user, token } = useAuthStore();

  const [state, setState] = useState<WhiteboardState>({
    paths: [],
    currentTool: 'pen',
    currentColor: '#000000',
    currentStrokeWidth: 4,
    isDrawingEnabled: true,
  });

  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [undoStack, setUndoStack] = useState<DrawingPath[]>([]);
  const [redoStack, setRedoStack] = useState<DrawingPath[]>([]);

  const syncServiceRef = useRef<WhiteboardSyncService | null>(null);
  const gestureRecognitionRef = useRef<GestureRecognition>(new GestureRecognition());

  useEffect(() => {
    if (!user || !token) return;

    const syncService = new WhiteboardSyncService();
    syncServiceRef.current = syncService;

    const initializeSync = async () => {
      await syncService.initialize(user.id, {
        onPathAdded: (path) => {
          setState((prev) => ({
            ...prev,
            paths: [...prev.paths, path],
          }));
        },
        onPathRemoved: (pathId) => {
          setState((prev) => ({
            ...prev,
            paths: prev.paths.filter((p) => p.id !== pathId),
          }));
        },
        onClear: () => {
          setState((prev) => ({
            ...prev,
            paths: [],
          }));
          setUndoStack([]);
          setRedoStack([]);
        },
        onSyncComplete: (paths) => {
          setState((prev) => ({
            ...prev,
            paths,
          }));
          setIsLoading(false);
        },
        onConnectionChange: setIsConnected,
      });

      await syncService.joinRoom(roomId, token);
      syncService.requestSync();
    };

    initializeSync();

    // Set up gesture recognition
    const gestureRecognition = gestureRecognitionRef.current;
    gestureRecognition.onGesture = handleGesture;

    return () => {
      syncService.destroy();
    };
  }, [user, token, roomId]);

  const handleGesture = useCallback((gesture: Gesture) => {
    switch (gesture.type) {
      case 'double_tap':
        // Toggle tool between pen and eraser
        setState((prev) => ({
          ...prev,
          currentTool: prev.currentTool === 'pen' ? 'eraser' : 'pen',
        }));
        break;
      case 'two_finger_tap':
        // Undo
        handleUndo();
        break;
      case 'pinch':
        // Could implement zoom functionality here
        console.log('Pinch gesture detected, scale:', gesture.data?.scale);
        break;
    }
  }, []);

  const handlePathComplete = useCallback((path: DrawingPath) => {
    setState((prev) => ({
      ...prev,
      paths: [...prev.paths, path],
    }));

    // Clear redo stack when new path is added
    setRedoStack([]);

    // Sync with server
    syncServiceRef.current?.addPath(path);
  }, []);

  const handleToolChange = useCallback((tool: DrawingTool) => {
    setState((prev) => ({ ...prev, currentTool: tool }));
  }, []);

  const handleColorChange = useCallback((color: string) => {
    setState((prev) => ({ ...prev, currentColor: color }));
  }, []);

  const handleStrokeWidthChange = useCallback((width: number) => {
    setState((prev) => ({ ...prev, currentStrokeWidth: width }));
  }, []);

  const handleUndo = useCallback(() => {
    if (state.paths.length === 0) return;

    const lastPath = state.paths[state.paths.length - 1];
    setState((prev) => ({
      ...prev,
      paths: prev.paths.slice(0, -1),
    }));

    setUndoStack((prev) => [...prev, lastPath]);
    syncServiceRef.current?.removePath(lastPath.id);
  }, [state.paths]);

  const handleRedo = useCallback(() => {
    if (redoStack.length === 0) return;

    const pathToRedo = redoStack[redoStack.length - 1];
    setState((prev) => ({
      ...prev,
      paths: [...prev.paths, pathToRedo],
    }));

    setRedoStack((prev) => prev.slice(0, -1));
    syncServiceRef.current?.addPath(pathToRedo);
  }, [redoStack]);

  const handleClear = useCallback(() => {
    setState((prev) => ({
      ...prev,
      paths: [],
    }));

    setUndoStack([]);
    setRedoStack([]);
    syncServiceRef.current?.clearWhiteboard();
  }, []);

  if (isLoading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={styles.loadingText}>Loading whiteboard...</Text>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.statusBar}>
        <View style={[styles.connectionIndicator, isConnected ? styles.connected : styles.disconnected]} />
        <Text style={styles.statusText}>
          {isConnected ? 'Connected' : 'Offline - changes will sync when reconnected'}
        </Text>
      </View>

      <TouchCanvas
        tool={state.currentTool}
        color={state.currentColor}
        strokeWidth={state.currentStrokeWidth}
        onPathComplete={handlePathComplete}
        paths={state.paths}
        isDrawingEnabled={state.isDrawingEnabled}
      />

      <DrawingToolsPalette
        currentTool={state.currentTool}
        currentColor={state.currentColor}
        currentStrokeWidth={state.currentStrokeWidth}
        onToolChange={handleToolChange}
        onColorChange={handleColorChange}
        onStrokeWidthChange={handleStrokeWidthChange}
        onUndo={handleUndo}
        onRedo={handleRedo}
        onClear={handleClear}
        canUndo={state.paths.length > 0}
        canRedo={redoStack.length > 0}
      />
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  loadingContainer: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#666',
  },
  statusBar: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 8,
    backgroundColor: '#f8f8f8',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  connectionIndicator: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 8,
  },
  connected: {
    backgroundColor: '#4CAF50',
  },
  disconnected: {
    backgroundColor: '#FF5252',
  },
  statusText: {
    fontSize: 14,
    color: '#666',
  },
});

export default WhiteboardScreen;