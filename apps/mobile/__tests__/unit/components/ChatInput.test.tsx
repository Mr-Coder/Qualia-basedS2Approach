import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import { Audio } from 'expo-av';
import ChatInput from '../../../src/components/chat/ChatInput';

// Mock expo-av
jest.mock('expo-av', () => ({
  Audio: {
    requestPermissionsAsync: jest.fn(),
    setAudioModeAsync: jest.fn(),
    Recording: {
      createAsync: jest.fn(),
    },
    RecordingOptionsPresets: {
      HIGH_QUALITY: {},
    },
  },
}));

const mockOnSendMessage = jest.fn();
const mockOnTyping = jest.fn();

describe('ChatInput', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders correctly', () => {
    const { getByPlaceholderText, getByTestId } = render(
      <ChatInput onSendMessage={mockOnSendMessage} />
    );

    expect(getByPlaceholderText('Type a message...')).toBeTruthy();
  });

  it('sends text message', () => {
    const { getByPlaceholderText, getByTestId } = render(
      <ChatInput onSendMessage={mockOnSendMessage} />
    );

    const input = getByPlaceholderText('Type a message...');
    fireEvent.changeText(input, 'Hello world');
    
    const sendButton = getByTestId('send-button');
    fireEvent.press(sendButton);

    expect(mockOnSendMessage).toHaveBeenCalledWith('Hello world', 'text');
  });

  it('calls onTyping when text changes', () => {
    const { getByPlaceholderText } = render(
      <ChatInput onSendMessage={mockOnSendMessage} onTyping={mockOnTyping} />
    );

    const input = getByPlaceholderText('Type a message...');
    fireEvent.changeText(input, 'H');

    expect(mockOnTyping).toHaveBeenCalled();
  });

  it('disables input when disabled prop is true', () => {
    const { getByPlaceholderText } = render(
      <ChatInput onSendMessage={mockOnSendMessage} disabled={true} />
    );

    const input = getByPlaceholderText('Type a message...');
    expect(input.props.editable).toBe(false);
  });

  it('starts recording on voice button press', async () => {
    (Audio.requestPermissionsAsync as jest.Mock).mockResolvedValue({
      status: 'granted',
    });
    
    const mockRecording = {
      stopAndUnloadAsync: jest.fn(),
      getURI: jest.fn().mockReturnValue('audio://test.m4a'),
    };
    
    (Audio.Recording.createAsync as jest.Mock).mockResolvedValue({
      recording: mockRecording,
    });

    const { getByTestId } = render(
      <ChatInput onSendMessage={mockOnSendMessage} />
    );

    // Find voice button
    const voiceButton = getByTestId('voice-button');
    
    // Start recording
    fireEvent(voiceButton, 'pressIn');

    await waitFor(() => {
      expect(Audio.requestPermissionsAsync).toHaveBeenCalled();
      expect(Audio.Recording.createAsync).toHaveBeenCalled();
    });

    // Stop recording
    fireEvent(voiceButton, 'pressOut');

    await waitFor(() => {
      expect(mockRecording.stopAndUnloadAsync).toHaveBeenCalled();
      expect(mockOnSendMessage).toHaveBeenCalledWith('', 'voice', 'audio://test.m4a');
    });
  });

  it('shows alert when microphone permission denied', async () => {
    const alertSpy = jest.spyOn(global, 'alert').mockImplementation();
    
    (Audio.requestPermissionsAsync as jest.Mock).mockResolvedValue({
      status: 'denied',
    });

    const { getByTestId } = render(
      <ChatInput onSendMessage={mockOnSendMessage} />
    );

    const voiceButton = getByTestId('voice-button');
    fireEvent(voiceButton, 'pressIn');

    await waitFor(() => {
      expect(alertSpy).toHaveBeenCalledWith(
        'Permission to access microphone is required!'
      );
    });

    alertSpy.mockRestore();
  });

  it('hides voice button when text is entered', () => {
    const { getByPlaceholderText, queryByTestId } = render(
      <ChatInput onSendMessage={mockOnSendMessage} />
    );

    // Voice button should be visible initially
    expect(queryByTestId('voice-button')).toBeTruthy();

    // Enter text
    const input = getByPlaceholderText('Type a message...');
    fireEvent.changeText(input, 'Hello');

    // Voice button should be hidden
    expect(queryByTestId('voice-button')).toBeFalsy();
  });
});