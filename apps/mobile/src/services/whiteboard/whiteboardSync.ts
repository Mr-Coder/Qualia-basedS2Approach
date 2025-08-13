import { io, Socket } from 'socket.io-client';
import { DrawingPath, WhiteboardSync } from '../../types/whiteboard';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { COMMUNICATION_SERVICE_URL } from '../config';

interface WhiteboardSyncCallbacks {
  onPathAdded: (path: DrawingPath) => void;
  onPathRemoved: (pathId: string) => void;
  onClear: () => void;
  onSyncComplete: (paths: DrawingPath[]) => void;
  onConnectionChange: (connected: boolean) => void;
}

export class WhiteboardSyncService {
  private socket: Socket | null = null;
  private roomId: string | null = null;
  private userId: string | null = null;
  private callbacks: WhiteboardSyncCallbacks | null = null;
  private pendingSync: WhiteboardSync[] = [];
  private isConnected = false;
  private syncTimer: NodeJS.Timeout | null = null;

  async initialize(userId: string, callbacks: WhiteboardSyncCallbacks) {
    this.userId = userId;
    this.callbacks = callbacks;
    
    // Load pending sync from storage
    await this.loadPendingSync();
  }

  async joinRoom(roomId: string, authToken: string) {
    if (this.socket) {
      this.leaveRoom();
    }

    this.roomId = roomId;

    this.socket = io(COMMUNICATION_SERVICE_URL, {
      transports: ['websocket'],
      auth: { token: authToken },
      reconnection: true,
      reconnectionAttempts: Infinity,
      reconnectionDelay: 1000,
    });

    this.setupSocketListeners();
    
    // Join whiteboard room
    this.socket.emit('whiteboard:join', { roomId });
  }

  private setupSocketListeners() {
    if (!this.socket || !this.callbacks) return;

    this.socket.on('connect', () => {
      this.isConnected = true;
      this.callbacks?.onConnectionChange(true);
      
      // Sync any pending operations
      this.syncPendingOperations();
    });

    this.socket.on('disconnect', () => {
      this.isConnected = false;
      this.callbacks?.onConnectionChange(false);
    });

    this.socket.on('whiteboard:path:added', (data: { path: DrawingPath; userId: string }) => {
      if (data.userId !== this.userId) {
        this.callbacks?.onPathAdded(data.path);
      }
    });

    this.socket.on('whiteboard:path:removed', (data: { pathId: string; userId: string }) => {
      if (data.userId !== this.userId) {
        this.callbacks?.onPathRemoved(data.pathId);
      }
    });

    this.socket.on('whiteboard:cleared', (data: { userId: string }) => {
      if (data.userId !== this.userId) {
        this.callbacks?.onClear();
      }
    });

    this.socket.on('whiteboard:sync:complete', (data: { paths: DrawingPath[] }) => {
      this.callbacks?.onSyncComplete(data.paths);
    });

    this.socket.on('whiteboard:error', (error: { message: string }) => {
      console.error('Whiteboard sync error:', error.message);
    });
  }

  addPath(path: DrawingPath) {
    const syncOp: WhiteboardSync = {
      type: 'add_path',
      path,
      timestamp: Date.now(),
      userId: this.userId!,
    };

    if (this.isConnected && this.socket) {
      this.socket.emit('whiteboard:path:add', {
        roomId: this.roomId,
        path,
      });
    } else {
      this.queueSyncOperation(syncOp);
    }
  }

  removePath(pathId: string) {
    const syncOp: WhiteboardSync = {
      type: 'remove_path',
      pathId,
      timestamp: Date.now(),
      userId: this.userId!,
    };

    if (this.isConnected && this.socket) {
      this.socket.emit('whiteboard:path:remove', {
        roomId: this.roomId,
        pathId,
      });
    } else {
      this.queueSyncOperation(syncOp);
    }
  }

  clearWhiteboard() {
    const syncOp: WhiteboardSync = {
      type: 'clear',
      timestamp: Date.now(),
      userId: this.userId!,
    };

    if (this.isConnected && this.socket) {
      this.socket.emit('whiteboard:clear', {
        roomId: this.roomId,
      });
    } else {
      this.queueSyncOperation(syncOp);
    }
  }

  requestSync() {
    if (this.isConnected && this.socket) {
      this.socket.emit('whiteboard:sync:request', {
        roomId: this.roomId,
      });
    }
  }

  private queueSyncOperation(op: WhiteboardSync) {
    this.pendingSync.push(op);
    this.savePendingSync();
  }

  private async syncPendingOperations() {
    if (!this.isConnected || !this.socket || this.pendingSync.length === 0) {
      return;
    }

    const operations = [...this.pendingSync];
    this.pendingSync = [];

    for (const op of operations) {
      switch (op.type) {
        case 'add_path':
          if (op.path) {
            this.socket.emit('whiteboard:path:add', {
              roomId: this.roomId,
              path: op.path,
            });
          }
          break;
        case 'remove_path':
          if (op.pathId) {
            this.socket.emit('whiteboard:path:remove', {
              roomId: this.roomId,
              pathId: op.pathId,
            });
          }
          break;
        case 'clear':
          this.socket.emit('whiteboard:clear', {
            roomId: this.roomId,
          });
          break;
      }
    }

    await this.clearPendingSync();
  }

  private async savePendingSync() {
    try {
      await AsyncStorage.setItem(
        `whiteboard_pending_${this.roomId}`,
        JSON.stringify(this.pendingSync)
      );
    } catch (error) {
      console.error('Failed to save pending sync:', error);
    }
  }

  private async loadPendingSync() {
    if (!this.roomId) return;

    try {
      const data = await AsyncStorage.getItem(`whiteboard_pending_${this.roomId}`);
      if (data) {
        this.pendingSync = JSON.parse(data);
      }
    } catch (error) {
      console.error('Failed to load pending sync:', error);
    }
  }

  private async clearPendingSync() {
    if (!this.roomId) return;

    try {
      await AsyncStorage.removeItem(`whiteboard_pending_${this.roomId}`);
    } catch (error) {
      console.error('Failed to clear pending sync:', error);
    }
  }

  leaveRoom() {
    if (this.socket) {
      this.socket.emit('whiteboard:leave', { roomId: this.roomId });
      this.socket.disconnect();
      this.socket = null;
    }

    if (this.syncTimer) {
      clearInterval(this.syncTimer);
      this.syncTimer = null;
    }

    this.roomId = null;
    this.isConnected = false;
  }

  destroy() {
    this.leaveRoom();
    this.callbacks = null;
    this.userId = null;
  }
}