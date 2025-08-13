# Mobile Integration Strategy

## Communication Service Integration

### WebSocket Endpoints
- **Production**: `wss://api.cot-dir.com/ws`
- **Development**: `ws://localhost:8000/ws`
- **Staging**: `wss://staging-api.cot-dir.com/ws`

### Connection Configuration
```typescript
const COMMUNICATION_SERVICE_URL = __DEV__ 
  ? 'http://localhost:8000' 
  : 'https://api.cot-dir.com';

const socket = io(COMMUNICATION_SERVICE_URL, {
  transports: ['websocket'],
  auth: { token: authToken },
  reconnection: true,
  reconnectionAttempts: Infinity,
  reconnectionDelay: 1000,
  reconnectionDelayMax: 5000
});
```

## Authentication Flow

### Mobile Token Management
1. **Initial Login**
   - User enters credentials
   - API returns JWT token (24hr expiry) + refresh token (30d expiry)
   - Store tokens in Secure Storage (Keychain/Keystore)

2. **Token Refresh**
   - Check token expiry on app resume
   - Auto-refresh if < 1hr remaining
   - Silent refresh in background

3. **Biometric Authentication**
   - Optional after initial login
   - Stores encrypted credentials
   - Uses device biometric APIs

### Authentication Service
```typescript
class AuthService {
  async login(credentials: LoginCredentials): Promise<AuthTokens> {
    const response = await api.post('/auth/login', credentials);
    await SecureStore.setItemAsync('accessToken', response.accessToken);
    await SecureStore.setItemAsync('refreshToken', response.refreshToken);
    return response;
  }

  async refreshToken(): Promise<string> {
    const refreshToken = await SecureStore.getItemAsync('refreshToken');
    const response = await api.post('/auth/refresh', { refreshToken });
    await SecureStore.setItemAsync('accessToken', response.accessToken);
    return response.accessToken;
  }

  async setupBiometric(): Promise<void> {
    const isAvailable = await LocalAuthentication.hasHardwareAsync();
    if (isAvailable) {
      await LocalAuthentication.authenticateAsync();
      // Store encrypted credentials
    }
  }
}
```

## Cross-Platform Sync Protocol

### State Synchronization
1. **Device Registration**
   - Each device gets unique ID
   - Stored in user profile
   - Used for sync conflict resolution

2. **Sync Events**
   ```typescript
   // Sync events emitted via WebSocket
   socket.emit('sync:state', {
     deviceId: deviceId,
     timestamp: Date.now(),
     state: {
       messages: localMessages,
       whiteboardState: localWhiteboard,
       preferences: userPreferences
     }
   });
   ```

3. **Conflict Resolution**
   - Last-write-wins for messages
   - Operational transformation for whiteboard
   - Server authoritative for grades
   - Device preference for UI settings

### Session Handoff
```typescript
interface SessionHandoff {
  sessionId: string;
  deviceId: string;
  timestamp: number;
  currentState: {
    activeRoom?: string;
    activeTab?: string;
    scrollPosition?: number;
    whiteboardViewport?: ViewportState;
  };
}

// Emit handoff when switching devices
socket.emit('session:handoff', handoffData);

// Listen for handoff on new device
socket.on('session:resume', (data: SessionHandoff) => {
  // Restore UI state
  navigation.navigate(data.currentState.activeRoom);
});
```

## API Integration Points

### REST API Endpoints
- `/api/v1/auth/*` - Authentication
- `/api/v1/users/*` - User management
- `/api/v1/rooms/*` - Study rooms
- `/api/v1/assignments/*` - Assignments
- `/api/v1/grades/*` - Grading

### WebSocket Events
- `connection` - Initial connection
- `message:send` - Send chat message
- `message:receive` - Receive chat message
- `whiteboard:draw` - Whiteboard updates
- `presence:update` - User presence
- `sync:state` - State synchronization
- `notification:push` - Push notifications

## Security Considerations

### Network Security
- Certificate pinning for API calls
- TLS 1.3 for all connections
- Request signing with HMAC

### Local Security
- Encrypted storage for sensitive data
- Biometric authentication
- App transport security
- Code obfuscation

### Token Security
- Short-lived access tokens (24hr)
- Secure refresh token storage
- Token rotation on refresh
- Device binding for tokens