import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { format } from 'date-fns';

interface MessageBubbleProps {
  message: {
    id: string;
    content: string;
    userId: string;
    userName: string;
    timestamp: Date;
    type: 'text' | 'voice';
    audioUrl?: string;
    transcription?: string;
    reactions?: { [emoji: string]: string[] };
  };
  isOwnMessage: boolean;
  onLongPress?: (messageId: string) => void;
  onPlayAudio?: (audioUrl: string) => void;
  onAddReaction?: (messageId: string, emoji: string) => void;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  isOwnMessage,
  onLongPress,
  onPlayAudio,
  onAddReaction,
}) => {
  const handleLongPress = () => {
    onLongPress?.(message.id);
  };

  const handlePlayAudio = () => {
    if (message.audioUrl && onPlayAudio) {
      onPlayAudio(message.audioUrl);
    }
  };

  return (
    <TouchableOpacity
      style={[
        styles.container,
        isOwnMessage ? styles.ownMessage : styles.otherMessage,
      ]}
      onLongPress={handleLongPress}
      activeOpacity={0.8}
    >
      {!isOwnMessage && (
        <Text style={styles.userName}>{message.userName}</Text>
      )}
      
      {message.type === 'voice' ? (
        <TouchableOpacity
          style={styles.voiceMessage}
          onPress={handlePlayAudio}
        >
          <Ionicons name="mic" size={20} color={isOwnMessage ? '#fff' : '#007AFF'} />
          <View style={styles.voiceContent}>
            <View style={styles.voiceWaveform}>
              {/* Placeholder for waveform visualization */}
              <View style={[styles.voiceBar, { height: 10 }]} />
              <View style={[styles.voiceBar, { height: 15 }]} />
              <View style={[styles.voiceBar, { height: 20 }]} />
              <View style={[styles.voiceBar, { height: 15 }]} />
              <View style={[styles.voiceBar, { height: 10 }]} />
            </View>
            {message.transcription && (
              <Text style={[styles.transcription, isOwnMessage && styles.ownTranscription]}>
                "{message.transcription}"
              </Text>
            )}
          </View>
        </TouchableOpacity>
      ) : (
        <Text style={[styles.content, isOwnMessage && styles.ownContent]}>
          {message.content}
        </Text>
      )}
      
      <Text style={[styles.timestamp, isOwnMessage && styles.ownTimestamp]}>
        {format(new Date(message.timestamp), 'HH:mm')}
      </Text>
      
      {message.reactions && Object.keys(message.reactions).length > 0 && (
        <View style={styles.reactions}>
          {Object.entries(message.reactions).map(([emoji, userIds]) => (
            <TouchableOpacity
              key={emoji}
              style={styles.reaction}
              onPress={() => onAddReaction?.(message.id, emoji)}
            >
              <Text style={styles.reactionEmoji}>{emoji}</Text>
              {userIds.length > 1 && (
                <Text style={styles.reactionCount}>{userIds.length}</Text>
              )}
            </TouchableOpacity>
          ))}
        </View>
      )}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    maxWidth: '80%',
    marginVertical: 4,
    marginHorizontal: 12,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 16,
  },
  ownMessage: {
    alignSelf: 'flex-end',
    backgroundColor: '#007AFF',
  },
  otherMessage: {
    alignSelf: 'flex-start',
    backgroundColor: '#E9E9EB',
  },
  userName: {
    fontSize: 12,
    fontWeight: '600',
    color: '#666',
    marginBottom: 4,
  },
  content: {
    fontSize: 16,
    color: '#000',
  },
  ownContent: {
    color: '#fff',
  },
  timestamp: {
    fontSize: 11,
    color: '#666',
    marginTop: 4,
  },
  ownTimestamp: {
    color: '#E1E1E1',
  },
  voiceMessage: {
    flexDirection: 'row',
    alignItems: 'center',
    minWidth: 200,
  },
  voiceContent: {
    flex: 1,
    marginLeft: 8,
  },
  voiceWaveform: {
    flexDirection: 'row',
    alignItems: 'center',
    height: 30,
    gap: 2,
  },
  voiceBar: {
    width: 3,
    backgroundColor: '#007AFF',
    borderRadius: 2,
  },
  transcription: {
    fontSize: 14,
    color: '#666',
    fontStyle: 'italic',
    marginTop: 4,
  },
  ownTranscription: {
    color: '#E1E1E1',
  },
  reactions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 4,
    gap: 4,
  },
  reaction: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.05)',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 12,
  },
  reactionEmoji: {
    fontSize: 12,
  },
  reactionCount: {
    fontSize: 10,
    marginLeft: 2,
    color: '#666',
  },
});

export default MessageBubble;