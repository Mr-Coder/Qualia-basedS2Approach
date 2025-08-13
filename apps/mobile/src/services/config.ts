// API Configuration
export const API_BASE_URL = process.env.EXPO_PUBLIC_API_URL || 'http://localhost:3000';
export const COMMUNICATION_SERVICE_URL = process.env.EXPO_PUBLIC_COMMUNICATION_SERVICE_URL || 'http://localhost:3002';
export const AI_ENGINE_URL = process.env.EXPO_PUBLIC_AI_ENGINE_URL || 'http://localhost:3001';

// WebSocket Configuration
export const WS_RECONNECT_ATTEMPTS = 5;
export const WS_RECONNECT_DELAY = 1000;

// Audio Configuration
export const AUDIO_RECORDING_OPTIONS = {
  android: {
    extension: '.m4a',
    outputFormat: 2, // MPEG_4
    audioEncoder: 3, // AAC
    sampleRate: 44100,
    numberOfChannels: 2,
    bitRate: 128000,
  },
  ios: {
    extension: '.m4a',
    outputFormat: 'applem4a',
    audioQuality: 127, // MAX
    sampleRate: 44100,
    numberOfChannels: 2,
    bitRate: 128000,
    linearPCMBitDepth: 16,
    linearPCMIsBigEndian: false,
    linearPCMIsFloat: false,
  },
};

// Message Configuration
export const MESSAGE_BATCH_SIZE = 20;
export const TYPING_INDICATOR_TIMEOUT = 3000;

// Offline Configuration
export const OFFLINE_QUEUE_MAX_SIZE = 1000;
export const SYNC_BATCH_SIZE = 50;