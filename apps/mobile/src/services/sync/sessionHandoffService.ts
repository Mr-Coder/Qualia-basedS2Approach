import { Alert, Linking } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { StateSyncService, DeviceInfo } from './stateSyncService';
import { generateSecureToken } from '../../utils/crypto';

export interface HandoffSession {
  id: string;
  fromDevice: DeviceInfo;
  toDevice: DeviceInfo;
  state: any;
  token: string;
  expiresAt: number;
  status: 'pending' | 'accepted' | 'rejected' | 'expired';
}

export interface HandoffOptions {
  immediate?: boolean;
  requireConfirmation?: boolean;
  expirationTime?: number; // in milliseconds
}

interface HandoffCallbacks {
  onHandoffInitiated: (session: HandoffSession) => void;
  onHandoffReceived: (session: HandoffSession) => void;
  onHandoffCompleted: (session: HandoffSession) => void;
  onHandoffFailed: (error: Error) => void;
}

export class SessionHandoffService {
  private static instance: SessionHandoffService;
  private stateSyncService: StateSyncService;
  private pendingSessions: Map<string, HandoffSession> = new Map();
  private callbacks: HandoffCallbacks | null = null;
  private handoffTimer: NodeJS.Timeout | null = null;

  private readonly HANDOFF_KEY_PREFIX = 'handoff_session_';
  private readonly DEFAULT_EXPIRATION = 5 * 60 * 1000; // 5 minutes
  private readonly HANDOFF_CHECK_INTERVAL = 30 * 1000; // 30 seconds

  private constructor() {
    this.stateSyncService = StateSyncService.getInstance();
    this.startHandoffMonitor();
  }

  static getInstance(): SessionHandoffService {
    if (!SessionHandoffService.instance) {
      SessionHandoffService.instance = new SessionHandoffService();
    }
    return SessionHandoffService.instance;
  }

  setCallbacks(callbacks: HandoffCallbacks) {
    this.callbacks = callbacks;
  }

  async initiateHandoff(
    toDevice: DeviceInfo,
    state: any,
    options: HandoffOptions = {}
  ): Promise<HandoffSession> {
    const {
      immediate = false,
      requireConfirmation = true,
      expirationTime = this.DEFAULT_EXPIRATION,
    } = options;

    const devices = this.stateSyncService.getDevices();
    const fromDevice = await this.getCurrentDevice();

    if (!toDevice || !devices.find(d => d.id === toDevice.id)) {
      throw new Error('Target device not found or not active');
    }

    const session: HandoffSession = {
      id: this.generateSessionId(),
      fromDevice,
      toDevice,
      state,
      token: await generateSecureToken(),
      expiresAt: Date.now() + expirationTime,
      status: 'pending',
    };

    // Store session
    this.pendingSessions.set(session.id, session);
    await this.saveSession(session);

    // Generate handoff URL
    const handoffUrl = this.generateHandoffUrl(session);

    if (immediate) {
      // Attempt direct handoff via state sync
      try {
        await this.stateSyncService.requestHandoff(toDevice.id);
        this.callbacks?.onHandoffInitiated(session);
      } catch (error) {
        this.callbacks?.onHandoffFailed(error as Error);
        throw error;
      }
    } else {
      // Show handoff options
      this.showHandoffOptions(session, handoffUrl, requireConfirmation);
    }

    return session;
  }

  async acceptHandoff(sessionId: string, token?: string): Promise<void> {
    const session = this.pendingSessions.get(sessionId);

    if (!session) {
      throw new Error('Handoff session not found');
    }

    if (session.status !== 'pending') {
      throw new Error(`Handoff session is ${session.status}`);
    }

    if (Date.now() > session.expiresAt) {
      session.status = 'expired';
      await this.saveSession(session);
      throw new Error('Handoff session has expired');
    }

    if (token && token !== session.token) {
      throw new Error('Invalid handoff token');
    }

    try {
      // Accept the handoff through state sync
      await this.stateSyncService.acceptHandoff(session.fromDevice.id, session.state);

      session.status = 'accepted';
      await this.saveSession(session);

      this.callbacks?.onHandoffCompleted(session);

      // Clean up
      this.pendingSessions.delete(sessionId);
      await this.removeSession(sessionId);
    } catch (error) {
      this.callbacks?.onHandoffFailed(error as Error);
      throw error;
    }
  }

  async rejectHandoff(sessionId: string): Promise<void> {
    const session = this.pendingSessions.get(sessionId);

    if (!session) {
      throw new Error('Handoff session not found');
    }

    session.status = 'rejected';
    await this.saveSession(session);

    this.pendingSessions.delete(sessionId);
    await this.removeSession(sessionId);
  }

  async checkPendingHandoffs(): Promise<HandoffSession[]> {
    const pendingSessions: HandoffSession[] = [];

    try {
      const keys = await AsyncStorage.getAllKeys();
      const handoffKeys = keys.filter(key => key.startsWith(this.HANDOFF_KEY_PREFIX));

      for (const key of handoffKeys) {
        const sessionData = await AsyncStorage.getItem(key);
        if (sessionData) {
          const session: HandoffSession = JSON.parse(sessionData);
          
          if (session.status === 'pending' && Date.now() < session.expiresAt) {
            pendingSessions.push(session);
            this.pendingSessions.set(session.id, session);
          } else if (Date.now() >= session.expiresAt) {
            // Clean up expired sessions
            await this.removeSession(session.id);
          }
        }
      }
    } catch (error) {
      console.error('Failed to check pending handoffs:', error);
    }

    return pendingSessions;
  }

  private async getCurrentDevice(): Promise<DeviceInfo> {
    const { Platform } = await import('react-native');
    const DeviceInfo = await import('react-native-device-info');

    return {
      id: await DeviceInfo.default.getUniqueId(),
      name: await DeviceInfo.default.getDeviceName(),
      platform: Platform.OS as 'ios' | 'android',
      lastSeen: Date.now(),
      isActive: true,
      capabilities: ['chat', 'whiteboard', 'video', 'offline'],
    };
  }

  private generateSessionId(): string {
    return `handoff_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private generateHandoffUrl(session: HandoffSession): string {
    const baseUrl = 'mathsolver://handoff';
    const params = new URLSearchParams({
      sessionId: session.id,
      token: session.token,
      fromDevice: session.fromDevice.name,
    });

    return `${baseUrl}?${params.toString()}`;
  }

  private showHandoffOptions(
    session: HandoffSession,
    handoffUrl: string,
    requireConfirmation: boolean
  ) {
    const options = [
      {
        text: 'Copy Link',
        onPress: () => {
          // Copy to clipboard
          import('expo-clipboard').then(({ setStringAsync }) => {
            setStringAsync(handoffUrl);
            Alert.alert('Link Copied', 'Handoff link copied to clipboard');
          });
        },
      },
      {
        text: 'Share',
        onPress: () => {
          import('react-native-share').then(({ default: Share }) => {
            Share.open({
              title: 'Continue session on another device',
              message: `Continue your Math Solver session on ${session.toDevice.name}`,
              url: handoffUrl,
            });
          });
        },
      },
    ];

    if (session.toDevice.platform === 'web') {
      options.push({
        text: 'Open in Browser',
        onPress: () => {
          const webUrl = handoffUrl.replace('mathsolver://', 'https://mathsolver.app/');
          Linking.openURL(webUrl);
        },
      });
    }

    if (!requireConfirmation) {
      // Automatically show options
      Alert.alert(
        'Session Handoff',
        `Continue your session on ${session.toDevice.name}?`,
        [
          ...options,
          {
            text: 'Cancel',
            style: 'cancel',
            onPress: () => this.rejectHandoff(session.id),
          },
        ]
      );
    }

    this.callbacks?.onHandoffInitiated(session);
  }

  private async saveSession(session: HandoffSession): Promise<void> {
    try {
      await AsyncStorage.setItem(
        `${this.HANDOFF_KEY_PREFIX}${session.id}`,
        JSON.stringify(session)
      );
    } catch (error) {
      console.error('Failed to save handoff session:', error);
    }
  }

  private async removeSession(sessionId: string): Promise<void> {
    try {
      await AsyncStorage.removeItem(`${this.HANDOFF_KEY_PREFIX}${sessionId}`);
    } catch (error) {
      console.error('Failed to remove handoff session:', error);
    }
  }

  private startHandoffMonitor() {
    this.handoffTimer = setInterval(async () => {
      const sessions = await this.checkPendingHandoffs();
      
      sessions.forEach(session => {
        if (session.toDevice.id === this.stateSyncService.getDevices()[0]?.id) {
          this.callbacks?.onHandoffReceived(session);
        }
      });

      // Clean up expired sessions
      this.pendingSessions.forEach((session, id) => {
        if (Date.now() > session.expiresAt) {
          this.pendingSessions.delete(id);
          this.removeSession(id);
        }
      });
    }, this.HANDOFF_CHECK_INTERVAL);
  }

  destroy() {
    if (this.handoffTimer) {
      clearInterval(this.handoffTimer);
      this.handoffTimer = null;
    }
    this.pendingSessions.clear();
    this.callbacks = null;
  }
}