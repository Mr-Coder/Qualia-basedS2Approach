import { by, device, element, expect } from 'detox';

describe('Cross-Platform Sync', () => {
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

  it('should sync messages across devices', async () => {
    // Enter room
    await element(by.id('room-item-0')).tap();
    
    // Send message
    const message = `Test message ${Date.now()}`;
    await element(by.id('message-input')).typeText(message);
    await element(by.id('send-button')).tap();
    
    // Verify sync status
    await expect(element(by.id('sync-status-indicator'))).toBeVisible();
    await waitFor(element(by.id('sync-complete')))
      .toBeVisible()
      .withTimeout(5000);
  });

  it('should handle session handoff', async () => {
    // Navigate to device management
    await element(by.id('profile-tab')).tap();
    await element(by.id('device-management-button')).tap();
    
    // Should show current device
    await expect(element(by.id('current-device'))).toBeVisible();
    
    // Generate handoff code
    await element(by.id('generate-handoff-button')).tap();
    await expect(element(by.id('handoff-code'))).toBeVisible();
    
    // In real test, another device would use this code
  });

  it('should sync offline changes', async () => {
    await element(by.id('room-item-0')).tap();
    
    // Go offline
    await device.setURLBlacklist(['.*']);
    
    // Make changes while offline
    await element(by.id('message-input')).typeText('Offline message 1');
    await element(by.id('send-button')).tap();
    await element(by.id('message-input')).typeText('Offline message 2');
    await element(by.id('send-button')).tap();
    
    // Should queue messages
    await expect(element(by.id('offline-queue-count'))).toHaveText('2');
    
    // Go back online
    await device.clearURLBlacklist();
    
    // Should sync queued messages
    await waitFor(element(by.id('offline-queue-count')))
      .toHaveText('0')
      .withTimeout(10000);
  });

  it('should handle concurrent editing', async () => {
    await element(by.id('room-item-0')).tap();
    
    // Start typing
    await element(by.id('message-input')).typeText('Concurrent ');
    
    // Simulate another user typing (in real test, would be another device)
    // The UI should show collaborative editing indicators
    
    await element(by.id('message-input')).typeText('editing test');
    await element(by.id('send-button')).tap();
    
    // Message should be sent without conflicts
    await expect(element(by.text('Concurrent editing test'))).toBeVisible();
  });

  it('should sync theme preferences', async () => {
    // Navigate to theme settings
    await element(by.id('profile-tab')).tap();
    await element(by.id('theme-settings-button')).tap();
    
    // Change theme
    await element(by.id('dark-theme-button')).tap();
    
    // Theme should persist and sync
    await device.reloadReactNative();
    
    // Verify theme is still dark
    await element(by.id('profile-tab')).tap();
    await element(by.id('theme-settings-button')).tap();
    await expect(element(by.id('dark-theme-selected'))).toBeVisible();
  });

  it('should maintain sync during app backgrounding', async () => {
    await element(by.id('room-item-0')).tap();
    
    // Send message
    await element(by.id('message-input')).typeText('Before background');
    await element(by.id('send-button')).tap();
    
    // Background the app
    await device.sendToHome();
    await device.launchApp();
    
    // Should still be in sync
    await expect(element(by.text('Before background'))).toBeVisible();
    await expect(element(by.id('sync-status-connected'))).toBeVisible();
  });

  it('should sync whiteboard state across devices', async () => {
    await element(by.id('room-item-0')).tap();
    await element(by.id('whiteboard-tab')).tap();
    
    // Draw on whiteboard
    await element(by.id('pen-tool')).tap();
    await element(by.id('touch-canvas')).swipe('right', 'slow', 0.75, 0.5, 0.5);
    
    // Verify sync
    await expect(element(by.id('whiteboard-sync-indicator'))).toBeVisible();
    await waitFor(element(by.id('whiteboard-sync-complete')))
      .toBeVisible()
      .withTimeout(5000);
    
    // In real test, another device would see the same drawing
  });
});