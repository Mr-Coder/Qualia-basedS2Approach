import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Switch,
  TouchableOpacity,
  Alert,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import notificationService from '../../services/notifications/notificationService';

interface NotificationPreferences {
  messages: boolean;
  aiResponses: boolean;
  collaborationRequests: boolean;
  gradeUpdates: boolean;
  sound: boolean;
  vibration: boolean;
  groupNotifications: boolean;
}

const NotificationPreferencesScreen: React.FC = () => {
  const [preferences, setPreferences] = useState<NotificationPreferences>(
    notificationService.getDefaultPreferences()
  );
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadPreferences();
  }, []);

  const loadPreferences = async () => {
    try {
      const savedPrefs = await notificationService.getNotificationPreferences();
      setPreferences(savedPrefs);
    } catch (error) {
      console.error('Failed to load preferences:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleToggle = async (key: keyof NotificationPreferences) => {
    const newPreferences = {
      ...preferences,
      [key]: !preferences[key],
    };
    
    setPreferences(newPreferences);
    
    try {
      await notificationService.setNotificationPreferences(newPreferences);
    } catch (error) {
      Alert.alert('Error', 'Failed to save preferences');
      // Revert the change
      setPreferences(preferences);
    }
  };

  const renderToggleItem = (
    icon: string,
    title: string,
    subtitle: string,
    key: keyof NotificationPreferences,
    iconColor: string = '#007AFF'
  ) => (
    <View style={styles.item}>
      <View style={styles.itemLeft}>
        <View style={[styles.iconContainer, { backgroundColor: `${iconColor}15` }]}>
          <Ionicons name={icon as any} size={24} color={iconColor} />
        </View>
        <View style={styles.itemText}>
          <Text style={styles.itemTitle}>{title}</Text>
          <Text style={styles.itemSubtitle}>{subtitle}</Text>
        </View>
      </View>
      <Switch
        value={preferences[key]}
        onValueChange={() => handleToggle(key)}
        trackColor={{ false: '#E5E5EA', true: '#34C759' }}
        thumbColor="#fff"
      />
    </View>
  );

  if (isLoading) {
    return (
      <View style={styles.loadingContainer}>
        <Text>Loading preferences...</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Notification Types</Text>
        
        {renderToggleItem(
          'chatbubbles',
          'Messages',
          'New messages in study rooms',
          'messages',
          '#007AFF'
        )}
        
        {renderToggleItem(
          'bulb',
          'AI Responses',
          'AI assistant responses and suggestions',
          'aiResponses',
          '#FF9500'
        )}
        
        {renderToggleItem(
          'people',
          'Collaboration Requests',
          'Invitations to collaborate',
          'collaborationRequests',
          '#5856D6'
        )}
        
        {renderToggleItem(
          'school',
          'Grade Updates',
          'Assignment grades and feedback',
          'gradeUpdates',
          '#34C759'
        )}
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Notification Settings</Text>
        
        {renderToggleItem(
          'volume-high',
          'Sound',
          'Play sound for notifications',
          'sound',
          '#FF3B30'
        )}
        
        {renderToggleItem(
          'phone-portrait',
          'Vibration',
          'Vibrate for notifications',
          'vibration',
          '#AF52DE'
        )}
        
        {renderToggleItem(
          'albums',
          'Group Notifications',
          'Group similar notifications together',
          'groupNotifications',
          '#007AFF'
        )}
      </View>

      <TouchableOpacity style={styles.testButton} onPress={sendTestNotification}>
        <Text style={styles.testButtonText}>Send Test Notification</Text>
      </TouchableOpacity>
    </ScrollView>
  );
};

const sendTestNotification = async () => {
  await notificationService.scheduleLocalNotification(
    'Test Notification',
    'This is a test notification from COT-DIR Mobile',
    { type: 'test' }
  );
  Alert.alert('Success', 'Test notification sent!');
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F2F2F7',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  section: {
    backgroundColor: '#fff',
    marginVertical: 8,
    paddingVertical: 12,
  },
  sectionTitle: {
    fontSize: 13,
    fontWeight: '600',
    color: '#8E8E93',
    textTransform: 'uppercase',
    marginLeft: 16,
    marginBottom: 8,
  },
  item: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  itemLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  iconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  itemText: {
    flex: 1,
  },
  itemTitle: {
    fontSize: 16,
    fontWeight: '500',
    color: '#000',
    marginBottom: 2,
  },
  itemSubtitle: {
    fontSize: 13,
    color: '#8E8E93',
  },
  testButton: {
    backgroundColor: '#007AFF',
    marginHorizontal: 16,
    marginVertical: 24,
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
  },
  testButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});

export default NotificationPreferencesScreen;