import { by, device, element, expect } from 'detox';

describe('Whiteboard Flow', () => {
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
    // Navigate to whiteboard
    await element(by.id('room-item-0')).tap();
    await element(by.id('whiteboard-tab')).tap();
  });

  it('should show whiteboard tools', async () => {
    await expect(element(by.id('whiteboard-screen'))).toBeVisible();
    await expect(element(by.id('drawing-tools-palette'))).toBeVisible();
    await expect(element(by.id('pen-tool'))).toBeVisible();
  });

  it('should draw on canvas', async () => {
    // Select pen tool
    await element(by.id('pen-tool')).tap();
    
    // Draw a line
    await element(by.id('touch-canvas')).swipe('right', 'slow', 0.75, 0.5, 0.5);
    
    // Should create a stroke
    await expect(element(by.id('stroke-count'))).toHaveText('1');
  });

  it('should change pen color', async () => {
    // Open color picker
    await element(by.id('color-picker-button')).tap();
    await element(by.id('color-red')).tap();
    
    // Draw with red pen
    await element(by.id('touch-canvas')).swipe('down', 'slow', 0.5, 0.3, 0.5);
    
    // Verify color changed
    await expect(element(by.id('current-color'))).toHaveText('#FF0000');
  });

  it('should use eraser tool', async () => {
    // Draw something first
    await element(by.id('pen-tool')).tap();
    await element(by.id('touch-canvas')).swipe('right', 'slow', 0.75, 0.5, 0.5);
    
    // Switch to eraser
    await element(by.id('eraser-tool')).tap();
    
    // Erase part of the drawing
    await element(by.id('touch-canvas')).swipe('right', 'slow', 0.75, 0.5, 0.5);
    
    // Stroke should be modified
    await expect(element(by.id('stroke-modified'))).toBeVisible();
  });

  it('should handle multi-touch gestures', async () => {
    // Draw something
    await element(by.id('pen-tool')).tap();
    await element(by.id('touch-canvas')).swipe('right', 'slow', 0.75, 0.5, 0.5);
    
    // Pinch to zoom
    await element(by.id('touch-canvas')).pinch(0.5);
    await expect(element(by.id('zoom-level'))).toHaveText('50%');
    
    // Pinch to zoom in
    await element(by.id('touch-canvas')).pinch(2.0);
    await expect(element(by.id('zoom-level'))).toHaveText('100%');
  });

  it('should clear canvas', async () => {
    // Draw multiple strokes
    await element(by.id('pen-tool')).tap();
    await element(by.id('touch-canvas')).swipe('right', 'slow', 0.75, 0.5, 0.5);
    await element(by.id('touch-canvas')).swipe('down', 'slow', 0.5, 0.3, 0.5);
    
    await expect(element(by.id('stroke-count'))).toHaveText('2');
    
    // Clear canvas
    await element(by.id('clear-button')).tap();
    await element(by.text('Clear')).tap(); // Confirm dialog
    
    await expect(element(by.id('stroke-count'))).toHaveText('0');
  });

  it('should sync drawing in real-time', async () => {
    // Draw on canvas
    await element(by.id('pen-tool')).tap();
    await element(by.id('touch-canvas')).swipe('right', 'slow', 0.75, 0.5, 0.5);
    
    // Sync indicator should show
    await expect(element(by.id('sync-indicator'))).toBeVisible();
    
    // Wait for sync to complete
    await waitFor(element(by.id('sync-complete')))
      .toBeVisible()
      .withTimeout(3000);
  });

  it('should handle offline drawing', async () => {
    // Enable airplane mode
    await device.setURLBlacklist(['.*']);
    
    // Draw while offline
    await element(by.id('pen-tool')).tap();
    await element(by.id('touch-canvas')).swipe('right', 'slow', 0.75, 0.5, 0.5);
    
    // Should show offline indicator
    await expect(element(by.id('offline-indicator'))).toBeVisible();
    
    // Disable airplane mode
    await device.clearURLBlacklist();
    
    // Should sync when back online
    await waitFor(element(by.id('sync-complete')))
      .toBeVisible()
      .withTimeout(10000);
  });

  it('should export whiteboard as image', async () => {
    // Draw something
    await element(by.id('pen-tool')).tap();
    await element(by.id('touch-canvas')).swipe('right', 'slow', 0.75, 0.5, 0.5);
    
    // Export
    await element(by.id('export-button')).tap();
    await element(by.id('export-png')).tap();
    
    // Should show success message
    await waitFor(element(by.text('Image saved to gallery')))
      .toBeVisible()
      .withTimeout(5000);
  });
});