import * as SecureStore from 'expo-secure-store';
import { COMMUNICATION_SERVICE_URL } from '../config';

interface TranscriptionResult {
  text: string;
  confidence?: number;
  language?: string;
}

class SpeechToTextService {
  private apiUrl: string;

  constructor() {
    this.apiUrl = `${COMMUNICATION_SERVICE_URL}/api/speech`;
  }

  async transcribeAudio(audioUri: string): Promise<TranscriptionResult> {
    try {
      const token = await SecureStore.getItemAsync('accessToken');
      if (!token) {
        throw new Error('No authentication token available');
      }

      // Create form data with audio file
      const formData = new FormData();
      const audioFile = {
        uri: audioUri,
        type: 'audio/m4a',
        name: 'voice_message.m4a',
      } as any;
      formData.append('audio', audioFile);

      // Send to backend for transcription
      const response = await fetch(`${this.apiUrl}/transcribe`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to transcribe audio');
      }

      const result = await response.json();
      return {
        text: result.text,
        confidence: result.confidence,
        language: result.language,
      };
    } catch (error) {
      console.error('Speech-to-text error:', error);
      throw error;
    }
  }

  async transcribeWithSocket(audioUri: string, socketClient: any): Promise<TranscriptionResult> {
    return new Promise((resolve, reject) => {
      // Create timeout
      const timeout = setTimeout(() => {
        socketClient.off('transcription_result');
        socketClient.off('transcription_error');
        reject(new Error('Transcription timeout'));
      }, 30000);

      // Listen for transcription result
      socketClient.on('transcription_result', (result: TranscriptionResult) => {
        clearTimeout(timeout);
        socketClient.off('transcription_result');
        socketClient.off('transcription_error');
        resolve(result);
      });

      // Listen for errors
      socketClient.on('transcription_error', (error: any) => {
        clearTimeout(timeout);
        socketClient.off('transcription_result');
        socketClient.off('transcription_error');
        reject(new Error(error.message || 'Transcription failed'));
      });

      // Send audio for transcription
      socketClient.emit('transcribe_audio', { audioUri });
    });
  }

  async detectLanguage(text: string): Promise<string> {
    try {
      const token = await SecureStore.getItemAsync('accessToken');
      if (!token) {
        throw new Error('No authentication token available');
      }

      const response = await fetch(`${this.apiUrl}/detect-language`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error('Failed to detect language');
      }

      const result = await response.json();
      return result.language;
    } catch (error) {
      console.error('Language detection error:', error);
      return 'en'; // Default to English
    }
  }
}

export default new SpeechToTextService();