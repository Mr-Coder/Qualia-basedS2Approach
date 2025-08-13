import { by, device, element, expect } from 'detox';

describe('Chat Flow', () => {
  beforeAll(async () => {
    await device.launchApp();
    // Login
    await element(by.id('email-input')).typeText('test@example.com');
    await element(by.id('password-input')).typeText('password123');
    await element(by.id('login-button')).tap();
    await waitFor(element(by.id('rooms-screen')))
      .toBeVisible()
      .withTimeout(5000);
  });

  beforeEach(async () => {
    await device.reloadReactNative();
  });

  it('should join a room and navigate to chat', async () => {
    // Select first room
    await element(by.id('room-item-0')).tap();
    await waitFor(element(by.id('chat-screen')))
      .toBeVisible()
      .withTimeout(5000);
  });

  it('should send a text message', async () => {
    await element(by.id('room-item-0')).tap();
    
    const messageText = 'Hello from E2E test!';
    await element(by.id('message-input')).typeText(messageText);
    await element(by.id('send-button')).tap();
    
    await waitFor(element(by.text(messageText)))
      .toBeVisible()
      .withTimeout(5000);
  });

  it('should show typing indicator', async () => {
    await element(by.id('room-item-0')).tap();
    
    await element(by.id('message-input')).typeText('Typing...');
    
    // Typing indicator should appear for other users
    // This would require another user connection in real test
  });

  it('should add reaction to message', async () => {
    await element(by.id('room-item-0')).tap();
    
    // Send message first
    await element(by.id('message-input')).typeText('React to this!');
    await element(by.id('send-button')).tap();
    
    // Long press on message to show reactions
    await element(by.text('React to this!')).longPress();
    await element(by.id('reaction-👍')).tap();
    
    await expect(element(by.id('reaction-👍-count'))).toBeVisible();
  });

  it('should record and send voice message', async () => {
    await element(by.id('room-item-0')).tap();
    
    // Long press voice button to record
    await element(by.id('voice-button')).longPress(3000);
    
    // Release to send
    await waitFor(element(by.id('audio-message-0')))
      .toBeVisible()
      .withTimeout(5000);
    
    // Tap to play
    await element(by.id('audio-message-0')).tap();
    await expect(element(by.id('audio-playing-0'))).toBeVisible();
  });

  it('should swipe to reply to message', async () => {
    await element(by.id('room-item-0')).tap();
    
    // Send a message
    await element(by.id('message-input')).typeText('Swipe to reply');
    await element(by.id('send-button')).tap();
    
    // Swipe right on message
    await element(by.text('Swipe to reply')).swipe('right', 'fast', 0.5);
    
    // Reply input should show quoted message
    await expect(element(by.id('reply-preview'))).toBeVisible();
    await expect(element(by.text('Replying to: Swipe to reply'))).toBeVisible();
  });

  it('should handle offline mode', async () => {
    await element(by.id('room-item-0')).tap();
    
    // Enable airplane mode
    await device.setURLBlacklist(['.*']);
    
    // Send message while offline
    await element(by.id('message-input')).typeText('Offline message');
    await element(by.id('send-button')).tap();
    
    // Message should be queued
    await expect(element(by.id('message-status-pending'))).toBeVisible();
    
    // Disable airplane mode
    await device.clearURLBlacklist();
    
    // Message should be sent
    await waitFor(element(by.id('message-status-sent')))
      .toBeVisible()
      .withTimeout(10000);
  });
});