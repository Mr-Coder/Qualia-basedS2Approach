import * as Notifications from 'expo-notifications';

interface GroupedNotification {
  id: string;
  title: string;
  body: string;
  data: any;
  timestamp: Date;
}

interface NotificationGroup {
  type: string;
  roomId?: string;
  notifications: GroupedNotification[];
  summary: string;
}

class NotificationGroupingService {
  private notificationGroups: Map<string, NotificationGroup> = new Map();
  private groupingTimeout: NodeJS.Timeout | null = null;
  private readonly GROUPING_WINDOW = 5000; // 5 seconds

  addNotification(notification: Notifications.Notification) {
    const content = notification.request.content;
    const data = content.data as any;
    
    // Determine group key
    const groupKey = this.getGroupKey(data);
    
    // Create notification object
    const notif: GroupedNotification = {
      id: notification.request.identifier,
      title: content.title || '',
      body: content.body || '',
      data: data,
      timestamp: new Date(),
    };
    
    // Get or create group
    let group = this.notificationGroups.get(groupKey);
    if (!group) {
      group = {
        type: data.type || 'default',
        roomId: data.roomId,
        notifications: [],
        summary: '',
      };
      this.notificationGroups.set(groupKey, group);
    }
    
    // Add notification to group
    group.notifications.push(notif);
    
    // Update summary
    group.summary = this.generateSummary(group);
    
    // Schedule group processing
    this.scheduleGroupProcessing();
  }

  private getGroupKey(data: any): string {
    // Group by type and room
    if (data.roomId) {
      return `${data.type}-${data.roomId}`;
    }
    return data.type || 'default';
  }

  private generateSummary(group: NotificationGroup): string {
    const count = group.notifications.length;
    
    switch (group.type) {
      case 'message':
        if (count === 1) {
          return group.notifications[0].body;
        }
        return `${count} new messages`;
        
      case 'ai_response':
        if (count === 1) {
          return 'AI has responded to your question';
        }
        return `${count} AI responses ready`;
        
      case 'collaboration_request':
        if (count === 1) {
          const sender = group.notifications[0].data.senderName || 'Someone';
          return `${sender} wants to collaborate`;
        }
        return `${count} collaboration requests`;
        
      case 'grade_update':
        if (count === 1) {
          return 'Your assignment has been graded';
        }
        return `${count} assignments graded`;
        
      default:
        return `${count} new notifications`;
    }
  }

  private scheduleGroupProcessing() {
    // Clear existing timeout
    if (this.groupingTimeout) {
      clearTimeout(this.groupingTimeout);
    }
    
    // Schedule new processing
    this.groupingTimeout = setTimeout(() => {
      this.processGroups();
    }, this.GROUPING_WINDOW);
  }

  private async processGroups() {
    for (const [key, group] of this.notificationGroups) {
      if (group.notifications.length > 1) {
        // Cancel individual notifications
        for (const notif of group.notifications) {
          await Notifications.dismissNotificationAsync(notif.id);
        }
        
        // Create grouped notification
        await this.createGroupedNotification(group);
      }
    }
    
    // Clear processed groups
    this.notificationGroups.clear();
  }

  private async createGroupedNotification(group: NotificationGroup) {
    const title = this.getGroupTitle(group);
    const body = group.summary;
    
    // Create inbox style notification for Android
    const androidStyle = {
      type: 'inbox',
      lines: group.notifications.map(n => n.body).slice(0, 5),
      summary: group.notifications.length > 5 
        ? `+${group.notifications.length - 5} more` 
        : undefined,
    };
    
    await Notifications.scheduleNotificationAsync({
      content: {
        title,
        body,
        data: {
          type: group.type,
          roomId: group.roomId,
          grouped: true,
          count: group.notifications.length,
        },
        badge: group.notifications.length,
        sound: 'default',
      },
      trigger: null,
    });
  }

  private getGroupTitle(group: NotificationGroup): string {
    switch (group.type) {
      case 'message':
        return group.roomId ? 'New Messages' : 'Messages';
      case 'ai_response':
        return 'AI Responses';
      case 'collaboration_request':
        return 'Collaboration Requests';
      case 'grade_update':
        return 'Grade Updates';
      default:
        return 'Notifications';
    }
  }

  // Get unread count by type
  getUnreadCount(type?: string): number {
    if (!type) {
      // Total unread count
      let total = 0;
      for (const group of this.notificationGroups.values()) {
        total += group.notifications.length;
      }
      return total;
    }
    
    // Count for specific type
    let count = 0;
    for (const group of this.notificationGroups.values()) {
      if (group.type === type) {
        count += group.notifications.length;
      }
    }
    return count;
  }

  // Clear notifications for a specific room
  clearRoomNotifications(roomId: string) {
    const keysToRemove: string[] = [];
    
    for (const [key, group] of this.notificationGroups) {
      if (group.roomId === roomId) {
        keysToRemove.push(key);
      }
    }
    
    for (const key of keysToRemove) {
      this.notificationGroups.delete(key);
    }
  }
}

export default new NotificationGroupingService();