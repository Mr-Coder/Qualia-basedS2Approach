// 缓存调试工具
export class CacheDebugger {
  static logCacheStatus() {
    console.group('🔍 Cache Status Debug');
    
    // 检查localStorage
    console.log('📦 LocalStorage:', localStorage.length, 'items');
    Object.keys(localStorage).forEach(key => {
      console.log(`  - ${key}:`, localStorage.getItem(key)?.slice(0, 100) + '...');
    });
    
    // 检查sessionStorage
    console.log('📦 SessionStorage:', sessionStorage.length, 'items');
    Object.keys(sessionStorage).forEach(key => {
      console.log(`  - ${key}:`, sessionStorage.getItem(key)?.slice(0, 100) + '...');
    });
    
    // 检查URL参数
    const url = new URL(window.location.href);
    console.log('🔗 URL Parameters:', url.searchParams.toString());
    
    // 检查页面加载时间
    console.log('⏰ Page Load Time:', new Date().toISOString());
    console.log('🔄 Build Time:', typeof __BUILD_TIME__ !== 'undefined' ? __BUILD_TIME__ : 'Not available');
    
    // 检查网络状态
    console.log('🌐 Network Status:', navigator.onLine ? 'Online' : 'Offline');
    
    console.groupEnd();
  }
  
  static clearAllCaches() {
    console.group('🗑️ Clearing All Caches');
    
    // 清除localStorage
    const localStorageKeys = Object.keys(localStorage);
    localStorage.clear();
    console.log('✅ LocalStorage cleared:', localStorageKeys.length, 'items removed');
    
    // 清除sessionStorage
    const sessionStorageKeys = Object.keys(sessionStorage);
    sessionStorage.clear();
    console.log('✅ SessionStorage cleared:', sessionStorageKeys.length, 'items removed');
    
    // 清除cookie (仅限当前域)
    document.cookie.split(";").forEach(function(c) { 
      document.cookie = c.replace(/^ +/, "").replace(/=.*/, "=;expires=" + new Date().toUTCString() + ";path=/"); 
    });
    console.log('✅ Cookies cleared');
    
    // 清除缓存API (如果支持)
    if ('caches' in window) {
      caches.keys().then(cacheNames => {
        return Promise.all(
          cacheNames.map(cacheName => {
            console.log('✅ Cache cleared:', cacheName);
            return caches.delete(cacheName);
          })
        );
      });
    }
    
    console.log('🔄 Forcing page reload...');
    console.groupEnd();
    
    // 强制刷新
    window.location.href = window.location.href.split('?')[0] + '?_cacheBust=' + Date.now();
  }
  
  static addDebugUI() {
    // 只在开发环境显示
    if (import.meta.env.MODE !== 'development') return;
    
    const debugPanel = document.createElement('div');
    debugPanel.style.cssText = `
      position: fixed;
      top: 10px;
      right: 10px;
      background: rgba(0, 0, 0, 0.8);
      color: white;
      padding: 10px;
      border-radius: 5px;
      font-family: monospace;
      font-size: 12px;
      z-index: 10000;
      display: flex;
      gap: 5px;
    `;
    
    const logButton = document.createElement('button');
    logButton.textContent = '🔍 Debug';
    logButton.style.cssText = 'padding: 5px; border: none; border-radius: 3px; cursor: pointer;';
    logButton.onclick = () => this.logCacheStatus();
    
    const clearButton = document.createElement('button');
    clearButton.textContent = '🗑️ Clear';
    clearButton.style.cssText = 'padding: 5px; border: none; border-radius: 3px; cursor: pointer; background: #ff4444; color: white;';
    clearButton.onclick = () => this.clearAllCaches();
    
    debugPanel.appendChild(logButton);
    debugPanel.appendChild(clearButton);
    
    document.body.appendChild(debugPanel);
  }
}

// 自动初始化
if (import.meta.env.MODE === 'development') {
  // 页面加载时自动记录状态
  window.addEventListener('load', () => {
    CacheDebugger.logCacheStatus();
    CacheDebugger.addDebugUI();
  });
}