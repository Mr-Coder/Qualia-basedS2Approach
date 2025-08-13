import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  View,
  StyleSheet,
  FlatList,
  KeyboardAvoidingView,
  Platform,
  Alert,
  TouchableOpacity,
  Text,
} from 'react-native';
import { useRoute, useNavigation } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import MessageBubble from '../components/chat/MessageBubble';
import ChatInput from '../components/chat/ChatInput';
import TypingIndicator from '../components/chat/TypingIndicator';
import AudioPlayer from '../components/chat/AudioPlayer';
import { useAuthStore } from '../stores/authStore';
import socketClient from '../services/websocket/socketClient';
import speechToText from '../services/speech/speechToText';
import { MESSAGE_BATCH_SIZE, TYPING_INDICATOR_TIMEOUT } from '../services/config';

interface Message {
  id: string;
  content: string;
  userId: string;
  userName: string;
  timestamp: Date;
  type: 'text' | 'voice';
  audioUrl?: string;
  transcription?: string;
  reactions?: { [emoji: string]: string[] };
}

interface RouteParams {
  roomId: string;
  roomName: string;
}

const ChatScreen: React.FC = () => {
  const route = useRoute();
  const navigation = useNavigation();
  const { roomId, roomName } = route.params as RouteParams;
  const { user } = useAuthStore();
  
  const [messages, setMessages] = useState<Message[]>([]);
  const [typingUsers, setTypingUsers] = useState<string[]>([]);
  const [audioPlayerUrl, setAudioPlayerUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  
  const flatListRef = useRef<FlatList>(null);
  const typingTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    navigation.setOptions({ title: roomName });
    loadMessages();
    setupSocketListeners();

    return () => {
      cleanupSocketListeners();
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
    };
  }, [roomId]);

  const loadMessages = async () => {
    try {
      setIsLoading(true);
      // In a real app, this would fetch messages from the server
      // For now, we'll use mock data
      setMessages([
        {
          id: '1',
          content: 'Welcome to the chat!',
          userId: 'system',
          userName: 'System',
          timestamp: new Date(),
          type: 'text',
        },
      ]);
    } catch (error) {
      console.error('Failed to load messages:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const setupSocketListeners = () => {
    socketClient.on('new_message', handleNewMessage);
    socketClient.on('user_typing', handleUserTyping);
    socketClient.on('reaction_added', handleReactionAdded);
    socketClient.on('voice_transcribed', handleVoiceTranscribed);
    
    // Join room
    socketClient.emit('join_room', { roomId });
  };

  const cleanupSocketListeners = () => {
    socketClient.off('new_message');
    socketClient.off('user_typing');
    socketClient.off('reaction_added');
    socketClient.off('voice_transcribed');
    
    // Leave room
    socketClient.emit('leave_room', { roomId });
  };

  const handleNewMessage = (message: Message) => {
    setMessages(prev => [...prev, message]);
    scrollToBottom();
  };

  const handleUserTyping = ({ userId, userName }: { userId: string; userName: string }) => {
    if (userId !== user?.id) {
      setTypingUsers(prev => {
        if (!prev.includes(userName)) {
          return [...prev, userName];
        }
        return prev;
      });

      // Clear typing indicator after timeout
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
      typingTimeoutRef.current = setTimeout(() => {
        setTypingUsers(prev => prev.filter(u => u !== userName));
      }, TYPING_INDICATOR_TIMEOUT);
    }
  };

  const handleReactionAdded = ({ messageId, emoji, userId }: any) => {
    setMessages(prev =>
      prev.map(msg => {
        if (msg.id === messageId) {
          const reactions = msg.reactions || {};
          if (!reactions[emoji]) {
            reactions[emoji] = [];
          }
          if (!reactions[emoji].includes(userId)) {
            reactions[emoji].push(userId);
          }
          return { ...msg, reactions };
        }
        return msg;
      })
    );
  };

  const handleVoiceTranscribed = ({ messageId, transcription }: any) => {
    setMessages(prev =>
      prev.map(msg => {
        if (msg.id === messageId) {
          return { ...msg, transcription };
        }
        return msg;
      })
    );
  };

  const sendMessage = async (content: string, type: 'text' | 'voice', audioUri?: string) => {
    if (!user) return;

    const message: Message = {
      id: Date.now().toString(),
      content,
      userId: user.id,
      userName: user.name,
      timestamp: new Date(),
      type,
    };

    if (type === 'voice' && audioUri) {
      try {
        // Upload audio and get URL
        message.audioUrl = audioUri; // In real app, upload to server first
        
        // Start transcription
        speechToText.transcribeWithSocket(audioUri, socketClient)
          .then(result => {
            message.transcription = result.text;
            socketClient.emit('update_message', {
              messageId: message.id,
              transcription: result.text,
            });
          })
          .catch(error => {
            console.error('Transcription failed:', error);
          });
      } catch (error) {
        Alert.alert('Error', 'Failed to send voice message');
        return;
      }
    }

    // Send message via socket
    socketClient.emit('send_message', {
      roomId,
      message,
    });

    // Optimistically add to local state
    setMessages(prev => [...prev, message]);
    scrollToBottom();
  };

  const handleTyping = useCallback(() => {
    if (user) {
      socketClient.emit('typing', {
        roomId,
        userId: user.id,
        userName: user.name,
      });
    }
  }, [roomId, user]);

  const handleLongPressMessage = (messageId: string) => {
    // Show reaction picker
    const reactions = ['❤️', '👍', '😂', '😮', '😢', '🔥'];
    Alert.alert(
      'React to message',
      '',
      reactions.map(emoji => ({
        text: emoji,
        onPress: () => handleAddReaction(messageId, emoji),
      })),
      { cancelable: true }
    );
  };

  const handleAddReaction = (messageId: string, emoji: string) => {
    if (!user) return;

    socketClient.emit('add_reaction', {
      roomId,
      messageId,
      emoji,
      userId: user.id,
    });
  };

  const handlePlayAudio = (audioUrl: string) => {
    setAudioPlayerUrl(audioUrl);
  };

  const scrollToBottom = () => {
    setTimeout(() => {
      flatListRef.current?.scrollToEnd({ animated: true });
    }, 100);
  };

  const renderMessage = ({ item }: { item: Message }) => (
    <MessageBubble
      message={item}
      isOwnMessage={item.userId === user?.id}
      onLongPress={handleLongPressMessage}
      onPlayAudio={handlePlayAudio}
      onAddReaction={handleAddReaction}
    />
  );

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
    >
      <FlatList
        ref={flatListRef}
        data={messages}
        renderItem={renderMessage}
        keyExtractor={item => item.id}
        contentContainerStyle={styles.messagesList}
        onContentSizeChange={scrollToBottom}
        ListFooterComponent={<TypingIndicator users={typingUsers} />}
      />
      
      <ChatInput
        onSendMessage={sendMessage}
        onTyping={handleTyping}
        disabled={!socketClient.isConnected()}
      />
      
      {audioPlayerUrl && (
        <AudioPlayer
          audioUrl={audioPlayerUrl}
          onClose={() => setAudioPlayerUrl(null)}
        />
      )}
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  messagesList: {
    paddingVertical: 16,
  },
});

export default ChatScreen;