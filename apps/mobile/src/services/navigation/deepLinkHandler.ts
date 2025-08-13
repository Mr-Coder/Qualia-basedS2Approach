import { Linking } from 'react-native';
import { NavigationContainerRef } from '@react-navigation/native';

class DeepLinkHandler {
  private navigationRef: NavigationContainerRef<any> | null = null;
  private pendingDeepLink: string | null = null;

  setNavigationRef(ref: NavigationContainerRef<any>) {
    this.navigationRef = ref;
    
    // Process pending deep link if any
    if (this.pendingDeepLink) {
      this.handleDeepLink(this.pendingDeepLink);
      this.pendingDeepLink = null;
    }
  }

  async initialize() {
    // Handle deep link when app is opened from a link
    const initialUrl = await Linking.getInitialURL();
    if (initialUrl) {
      this.handleDeepLink(initialUrl);
    }

    // Handle deep links when app is already open
    Linking.addEventListener('url', this.handleUrlEvent);
  }

  cleanup() {
    Linking.removeAllListeners('url');
  }

  private handleUrlEvent = (event: { url: string }) => {
    this.handleDeepLink(event.url);
  };

  handleDeepLink(url: string) {
    if (!this.navigationRef || !this.navigationRef.isReady()) {
      // Save for later processing
      this.pendingDeepLink = url;
      return;
    }

    // Parse the URL
    const route = this.parseDeepLink(url);
    if (route) {
      this.navigate(route);
    }
  }

  handleNotificationDeepLink(data: any) {
    if (!this.navigationRef || !this.navigationRef.isReady()) {
      // Save for later processing
      this.pendingDeepLink = this.createDeepLinkFromData(data);
      return;
    }

    const route = this.parseNotificationData(data);
    if (route) {
      this.navigate(route);
    }
  }

  private parseDeepLink(url: string): any {
    // Remove the scheme and host
    const path = url.replace(/^.*?:\/\//, '').replace(/^[^\/]+/, '');
    const segments = path.split('/').filter(Boolean);

    if (segments.length === 0) {
      return null;
    }

    // Parse different deep link patterns
    switch (segments[0]) {
      case 'room':
        if (segments[1]) {
          return {
            name: 'Chat',
            params: { roomId: segments[1] },
          };
        }
        break;
        
      case 'profile':
        if (segments[1]) {
          return {
            name: 'UserProfile',
            params: { userId: segments[1] },
          };
        }
        break;
        
      case 'assignment':
        if (segments[1]) {
          return {
            name: 'Assignment',
            params: { assignmentId: segments[1] },
          };
        }
        break;
        
      case 'whiteboard':
        if (segments[1]) {
          return {
            name: 'Whiteboard',
            params: { sessionId: segments[1] },
          };
        }
        break;
        
      case 'settings':
        return {
          name: 'Settings',
          params: segments[1] ? { tab: segments[1] } : undefined,
        };
    }

    return null;
  }

  private parseNotificationData(data: any): any {
    switch (data.type) {
      case 'message':
        if (data.roomId) {
          return {
            name: 'Chat',
            params: { 
              roomId: data.roomId,
              messageId: data.messageId,
            },
          };
        }
        break;
        
      case 'ai_response':
        if (data.roomId) {
          return {
            name: 'Chat',
            params: { 
              roomId: data.roomId,
              highlightAI: true,
            },
          };
        }
        break;
        
      case 'collaboration_request':
        if (data.roomId) {
          return {
            name: 'CollaborationRequest',
            params: { 
              roomId: data.roomId,
              requestId: data.requestId,
            },
          };
        }
        break;
        
      case 'grade_update':
        if (data.assignmentId) {
          return {
            name: 'Assignment',
            params: { 
              assignmentId: data.assignmentId,
              showGrade: true,
            },
          };
        }
        break;
    }

    // Fallback to deep link if provided
    if (data.deepLink) {
      return this.parseDeepLink(data.deepLink);
    }

    return null;
  }

  private createDeepLinkFromData(data: any): string {
    const baseUrl = 'cotdir://';
    
    switch (data.type) {
      case 'message':
        return `${baseUrl}room/${data.roomId}`;
      case 'ai_response':
        return `${baseUrl}room/${data.roomId}?ai=true`;
      case 'collaboration_request':
        return `${baseUrl}room/${data.roomId}?request=${data.requestId}`;
      case 'grade_update':
        return `${baseUrl}assignment/${data.assignmentId}`;
      default:
        return data.deepLink || baseUrl;
    }
  }

  private navigate(route: any) {
    if (!this.navigationRef) return;

    // Reset navigation state to ensure we're at the right place
    this.navigationRef.reset({
      index: 1,
      routes: [
        { name: 'Main' }, // Ensure we have the main tab navigator
        route,
      ],
    });
  }

  // Helper method to create shareable deep links
  createShareableLink(type: string, id: string): string {
    const baseUrl = 'https://cotdir.app/';
    
    switch (type) {
      case 'room':
        return `${baseUrl}room/${id}`;
      case 'assignment':
        return `${baseUrl}assignment/${id}`;
      case 'whiteboard':
        return `${baseUrl}whiteboard/${id}`;
      default:
        return baseUrl;
    }
  }
}

export default new DeepLinkHandler();