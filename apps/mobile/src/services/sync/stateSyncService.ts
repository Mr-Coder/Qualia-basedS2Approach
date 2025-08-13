import { io, Socket } from 'socket.io-client';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { COMMUNICATION_SERVICE_URL } from '../config';
import { MessageQueue } from '../offline/messageQueue';
import { SyncConflictResolver } from '../offline/syncConflictResolver';

export interface SyncState {
  roomId: string;
  userId: string;
  deviceId: string;
  lastSync: number;
  state: any;
}

export interface DeviceInfo {
  id: string;
  name: string;
  platform: 'ios' | 'android' | 'web';
  lastSeen: number;
  isActive: boolean;
  capabilities: string[];
}

export interface SyncEvent {
  type: 'state_update' | 'device_join' | 'device_leave' | 'handoff_request' | 'handoff_complete';
  deviceId: string;
  timestamp: number;
  data?: any;
}

interface StateSyncCallbacks {
  onStateUpdate: (state: any, deviceId: string) => void;
  onDeviceUpdate: (devices: DeviceInfo[]) => void;
  onHandoffRequest: (fromDevice: DeviceInfo, state: any) => void;
  onSyncComplete: () => void;
  onSyncError: (error: Error) => void;
  onConnectionChange: (connected: boolean) => void;
}

export class StateSyncService {
  private static instance: StateSyncService;
  private socket: Socket | null = null;
  private roomId: string | null = null;
  private userId: string | null = null;
  private deviceId: string | null = null;
  private callbacks: StateSyncCallbacks | null = null;
  private syncState: Map<string, any> = new Map();
  private devices: Map<string, DeviceInfo> = new Map();
  private messageQueue: MessageQueue;
  private conflictResolver: SyncConflictResolver;
  private syncTimer: NodeJS.Timeout | null = null;
  private isConnected = false;

  private readonly SYNC_INTERVAL = 5000; // 5 seconds
  private readonly STATE_KEY_PREFIX = 'sync_state_';

  private constructor() {
    this.messageQueue = MessageQueue.getInstance();
    this.conflictResolver = SyncConflictResolver.getInstance();
  }

  static getInstance(): StateSyncService {
    if (!StateSyncService.instance) {
      StateSyncService.instance = new StateSyncService();
    }
    return StateSyncService.instance;
  }

  async initialize(userId: string, deviceId: string, callbacks: StateSyncCallbacks) {
    this.userId = userId;
    this.deviceId = deviceId;
    this.callbacks = callbacks;

    // Load cached state
    await this.loadCachedState();
  }

  async joinRoom(roomId: string, authToken: string) {
    if (this.socket) {
      this.leaveRoom();
    }

    this.roomId = roomId;

    this.socket = io(COMMUNICATION_SERVICE_URL, {
      transports: ['websocket'],
      auth: { 
        token: authToken,
        deviceId: this.deviceId,
        platform: 'mobile',
      },
      reconnection: true,
      reconnectionAttempts: Infinity,
      reconnectionDelay: 1000,
    });

    this.setupSocketListeners();
    
    // Join sync room
    this.socket.emit('sync:join', {
      roomId,
      userId: this.userId,
      deviceId: this.deviceId,
      deviceInfo: await this.getDeviceInfo(),
    });

    // Start periodic sync
    this.startPeriodicSync();
  }

  private setupSocketListeners() {
    if (!this.socket || !this.callbacks) return;

    this.socket.on('connect', () => {
      this.isConnected = true;
      this.callbacks?.onConnectionChange(true);
      
      // Re-join room and sync state
      if (this.roomId) {
        this.socket?.emit('sync:join', {
          roomId: this.roomId,
          userId: this.userId,
          deviceId: this.deviceId,
          deviceInfo: this.getDeviceInfo(),
        });
      }
    });

    this.socket.on('disconnect', () => {
      this.isConnected = false;
      this.callbacks?.onConnectionChange(false);
    });

    this.socket.on('sync:state:update', (data: {
      state: any;
      deviceId: string;
      timestamp: number;
    }) => {
      if (data.deviceId !== this.deviceId) {
        this.handleStateUpdate(data);
      }
    });

    this.socket.on('sync:devices:update', (data: {
      devices: DeviceInfo[];
    }) => {
      this.updateDevices(data.devices);
      this.callbacks?.onDeviceUpdate(Array.from(this.devices.values()));
    });

    this.socket.on('sync:handoff:request', (data: {
      fromDevice: DeviceInfo;
      state: any;
    }) => {
      this.callbacks?.onHandoffRequest(data.fromDevice, data.state);
    });

    this.socket.on('sync:handoff:complete', (data: {
      toDeviceId: string;
      success: boolean;
    }) => {
      if (data.toDeviceId === this.deviceId && data.success) {
        this.callbacks?.onSyncComplete();
      }
    });

    this.socket.on('sync:error', (error: {
      message: string;
      code: string;
    }) => {
      console.error('Sync error:', error);
      this.callbacks?.onSyncError(new Error(error.message));
    });
  }

  async updateState(state: any, immediate = false) {
    // Update local state
    this.syncState.set(this.roomId!, state);

    // Save to cache
    await this.saveCachedState();

    if (immediate && this.isConnected && this.socket) {
      // Send immediately
      this.socket.emit('sync:state:update', {
        roomId: this.roomId,
        state,
        deviceId: this.deviceId,
        timestamp: Date.now(),
      });
    }
    // Otherwise, it will be sent in the next periodic sync
  }

  private async handleStateUpdate(data: {
    state: any;
    deviceId: string;
    timestamp: number;
  }) {
    const localState = this.syncState.get(this.roomId!) || {};
    const localTimestamp = localState._lastUpdate || 0;

    if (data.timestamp > localTimestamp) {
      // Remote state is newer, check for conflicts
      const hasConflicts = this.detectConflicts(localState, data.state);

      if (hasConflicts) {
        // Resolve conflicts
        const resolution = this.conflictResolver.resolve({
          type: 'state',
          clientData: localState,
          serverData: data.state,
          clientTimestamp: localTimestamp,
          serverTimestamp: data.timestamp,
        }, 'last_write_wins');

        this.syncState.set(this.roomId!, resolution.resolved);
      } else {
        // No conflicts, accept remote state
        this.syncState.set(this.roomId!, data.state);
      }

      await this.saveCachedState();
      this.callbacks?.onStateUpdate(this.syncState.get(this.roomId!), data.deviceId);
    }
  }

  private detectConflicts(localState: any, remoteState: any): boolean {
    // Simple conflict detection - can be enhanced based on specific needs
    if (!localState._lastUpdate || !remoteState._lastUpdate) {
      return false;
    }

    // Check if both states have been modified since last sync
    const timeDiff = Math.abs(localState._lastUpdate - remoteState._lastUpdate);
    return timeDiff < this.SYNC_INTERVAL;
  }

  async requestHandoff(toDeviceId: string) {
    if (!this.isConnected || !this.socket) {
      throw new Error('Not connected to sync service');
    }

    const currentState = this.syncState.get(this.roomId!) || {};

    this.socket.emit('sync:handoff:request', {
      roomId: this.roomId,
      fromDeviceId: this.deviceId,
      toDeviceId,
      state: currentState,
      timestamp: Date.now(),
    });
  }

  async acceptHandoff(fromDeviceId: string, state: any) {
    if (!this.isConnected || !this.socket) {
      throw new Error('Not connected to sync service');
    }

    // Update local state
    this.syncState.set(this.roomId!, state);
    await this.saveCachedState();

    this.socket.emit('sync:handoff:accept', {
      roomId: this.roomId,
      fromDeviceId,
      toDeviceId: this.deviceId,
      timestamp: Date.now(),
    });

    this.callbacks?.onSyncComplete();
  }

  getDevices(): DeviceInfo[] {
    return Array.from(this.devices.values());
  }

  getActiveDevices(): DeviceInfo[] {
    return Array.from(this.devices.values()).filter(device => device.isActive);
  }

  getCurrentState(): any {
    return this.syncState.get(this.roomId!) || {};
  }

  private updateDevices(devices: DeviceInfo[]) {
    this.devices.clear();
    devices.forEach(device => {
      if (device.id !== this.deviceId) {
        this.devices.set(device.id, device);
      }
    });
  }

  private async getDeviceInfo(): Promise<DeviceInfo> {
    const { Platform } = await import('react-native');
    const DeviceInfo = await import('react-native-device-info');

    return {
      id: this.deviceId!,
      name: await DeviceInfo.default.getDeviceName(),
      platform: Platform.OS as 'ios' | 'android',
      lastSeen: Date.now(),
      isActive: true,
      capabilities: [
        'chat',
        'whiteboard',
        'video',
        'offline',
        'push_notifications',
      ],
    };
  }

  private startPeriodicSync() {
    this.stopPeriodicSync();

    this.syncTimer = setInterval(() => {
      if (this.isConnected && this.socket) {
        const state = this.syncState.get(this.roomId!);
        if (state) {
          this.socket.emit('sync:state:update', {
            roomId: this.roomId,
            state,
            deviceId: this.deviceId,
            timestamp: Date.now(),
          });
        }
      }
    }, this.SYNC_INTERVAL);
  }

  private stopPeriodicSync() {
    if (this.syncTimer) {
      clearInterval(this.syncTimer);
      this.syncTimer = null;
    }
  }

  private async saveCachedState() {
    if (!this.roomId) return;

    try {
      const state = this.syncState.get(this.roomId) || {};
      await AsyncStorage.setItem(
        `${this.STATE_KEY_PREFIX}${this.roomId}`,
        JSON.stringify({
          state,
          timestamp: Date.now(),
        })
      );
    } catch (error) {
      console.error('Failed to save cached state:', error);
    }
  }

  private async loadCachedState() {
    if (!this.roomId) return;

    try {
      const cached = await AsyncStorage.getItem(`${this.STATE_KEY_PREFIX}${this.roomId}`);
      if (cached) {
        const { state, timestamp } = JSON.parse(cached);
        
        // Only use cached state if it's less than 1 hour old
        if (Date.now() - timestamp < 3600000) {
          this.syncState.set(this.roomId, state);
        }
      }
    } catch (error) {
      console.error('Failed to load cached state:', error);
    }
  }

  leaveRoom() {
    if (this.socket) {
      this.socket.emit('sync:leave', {
        roomId: this.roomId,
        deviceId: this.deviceId,
      });
      this.socket.disconnect();
      this.socket = null;
    }

    this.stopPeriodicSync();
    this.roomId = null;
    this.isConnected = false;
    this.devices.clear();
    this.syncState.clear();
  }

  destroy() {
    this.leaveRoom();
    this.callbacks = null;
    this.userId = null;
    this.deviceId = null;
  }
}