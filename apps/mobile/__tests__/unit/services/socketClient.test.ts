import { io } from 'socket.io-client';
import * as SecureStore from 'expo-secure-store';
import socketClient from '../../../src/services/websocket/socketClient';

// Mock dependencies
jest.mock('socket.io-client');
jest.mock('expo-secure-store');

const mockIo = io as jest.MockedFunction<typeof io>;
const mockSocket = {
  connected: false,
  connect: jest.fn(),
  disconnect: jest.fn(),
  emit: jest.fn(),
  on: jest.fn(),
  off: jest.fn(),
};

describe('SocketClient', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockIo.mockReturnValue(mockSocket as any);
  });

  describe('connect', () => {
    it('should connect with auth token', async () => {
      const mockToken = 'test-token';
      (SecureStore.getItemAsync as jest.Mock).mockResolvedValue(mockToken);

      await socketClient.connect();

      expect(SecureStore.getItemAsync).toHaveBeenCalledWith('accessToken');
      expect(mockIo).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          transports: ['websocket'],
          auth: { token: mockToken },
          reconnection: true,
        })
      );
    });

    it('should throw error if no auth token', async () => {
      (SecureStore.getItemAsync as jest.Mock).mockResolvedValue(null);

      await expect(socketClient.connect()).rejects.toThrow('No auth token available');
    });

    it('should not reconnect if already connected', async () => {
      mockSocket.connected = true;
      await socketClient.connect();

      expect(mockIo).not.toHaveBeenCalled();
    });
  });

  describe('disconnect', () => {
    it('should disconnect socket', () => {
      socketClient.disconnect();

      expect(mockSocket.disconnect).toHaveBeenCalled();
    });
  });

  describe('emit', () => {
    it('should emit event when connected', async () => {
      mockSocket.connected = true;
      const mockToken = 'test-token';
      (SecureStore.getItemAsync as jest.Mock).mockResolvedValue(mockToken);
      
      await socketClient.connect();
      socketClient.emit('test-event', { data: 'test' });

      expect(mockSocket.emit).toHaveBeenCalledWith('test-event', { data: 'test' });
    });

    it('should warn when not connected', () => {
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();
      mockSocket.connected = false;

      socketClient.emit('test-event', { data: 'test' });

      expect(consoleSpy).toHaveBeenCalledWith(
        'Socket not connected, queuing event:',
        'test-event'
      );
      expect(mockSocket.emit).not.toHaveBeenCalled();

      consoleSpy.mockRestore();
    });
  });

  describe('isConnected', () => {
    it('should return connection status', () => {
      mockSocket.connected = true;
      expect(socketClient.isConnected()).toBe(true);

      mockSocket.connected = false;
      expect(socketClient.isConnected()).toBe(false);
    });
  });
});