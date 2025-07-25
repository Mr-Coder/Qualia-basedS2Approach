/**
 * 优化的React Context实现
 * 避免不必要的重渲染
 */

import React, { 
  createContext, 
  useContext, 
  useMemo, 
  useCallback,
  ReactNode,
  memo
} from 'react';
import { useOptimizedState } from '../hooks/optimizedState';

// 应用状态类型定义
interface AppState {
  user: {
    id: string;
    name: string;
    email: string;
  } | null;
  settings: {
    theme: 'light' | 'dark';
    language: string;
    notifications: boolean;
  };
  ui: {
    sidebarOpen: boolean;
    loading: boolean;
    modal: string | null;
  };
}

// 初始状态
const initialState: AppState = {
  user: null,
  settings: {
    theme: 'light',
    language: 'zh-CN',
    notifications: true
  },
  ui: {
    sidebarOpen: false,
    loading: false,
    modal: null
  }
};

// 分离的Context - 避免单个大Context导致的性能问题
const UserContext = createContext<{
  user: AppState['user'];
  setUser: (user: AppState['user']) => void;
} | null>(null);

const SettingsContext = createContext<{
  settings: AppState['settings'];
  updateSettings: (settings: Partial<AppState['settings']>) => void;
} | null>(null);

const UIContext = createContext<{
  ui: AppState['ui'];
  updateUI: (ui: Partial<AppState['ui']>) => void;
} | null>(null);

// 优化的Provider组件
export const OptimizedProvider = memo(({ children }: { children: ReactNode }) => {
  const userState = useOptimizedState(initialState.user);
  const settingsState = useOptimizedState(initialState.settings);
  const uiState = useOptimizedState(initialState.ui);

  // 用户相关操作
  const userValue = useMemo(() => ({
    user: userState.state,
    setUser: userState.setState
  }), [userState.state, userState.setState]);

  // 设置相关操作
  const settingsValue = useMemo(() => ({
    settings: settingsState.state,
    updateSettings: settingsState.updateState
  }), [settingsState.state, settingsState.updateState]);

  // UI相关操作
  const uiValue = useMemo(() => ({
    ui: uiState.state,
    updateUI: uiState.updateState
  }), [uiState.state, uiState.updateState]);

  return (
    <UserContext.Provider value={userValue}>
      <SettingsContext.Provider value={settingsValue}>
        <UIContext.Provider value={uiValue}>
          {children}
        </UIContext.Provider>
      </SettingsContext.Provider>
    </UserContext.Provider>
  );
});

OptimizedProvider.displayName = 'OptimizedProvider';

// 优化的Hook - 只订阅需要的数据
export function useUser() {
  const context = useContext(UserContext);
  if (!context) {
    throw new Error('useUser must be used within OptimizedProvider');
  }
  return context;
}

export function useSettings() {
  const context = useContext(SettingsContext);
  if (!context) {
    throw new Error('useSettings must be used within OptimizedProvider');
  }
  return context;
}

export function useUI() {
  const context = useContext(UIContext);
  if (!context) {
    throw new Error('useUI must be used within OptimizedProvider');
  }
  return context;
}

// 复合Hook - 组合多个Context
export function useAppState() {
  const user = useUser();
  const settings = useSettings();
  const ui = useUI();

  const actions = useMemo(() => ({
    // 用户操作
    login: (userData: AppState['user']) => {
      user.setUser(userData);
    },
    logout: () => {
      user.setUser(null);
    },
    
    // 设置操作
    toggleTheme: () => {
      settings.updateSettings({
        theme: settings.settings.theme === 'light' ? 'dark' : 'light'
      });
    },
    
    // UI操作
    toggleSidebar: () => {
      ui.updateUI({ sidebarOpen: !ui.ui.sidebarOpen });
    },
    showModal: (modalType: string) => {
      ui.updateUI({ modal: modalType });
    },
    hideModal: () => {
      ui.updateUI({ modal: null });
    },
    setLoading: (loading: boolean) => {
      ui.updateUI({ loading });
    }
  }), [user, settings, ui]);

  return {
    user: user.user,
    settings: settings.settings,
    ui: ui.ui,
    actions
  };
}

// 选择器Hook - 精确订阅
export function useUserSelector<T>(selector: (user: AppState['user']) => T): T {
  const { user } = useUser();
  return useMemo(() => selector(user), [user, selector]);
}

export function useSettingsSelector<T>(selector: (settings: AppState['settings']) => T): T {
  const { settings } = useSettings();
  return useMemo(() => selector(settings), [settings, selector]);
}

export function useUISelector<T>(selector: (ui: AppState['ui']) => T): T {
  const { ui } = useUI();
  return useMemo(() => selector(ui), [ui, selector]);
}

// 使用示例组件
export const ExampleUsage = memo(() => {
  // 只订阅用户名
  const userName = useUserSelector(user => user?.name || '未登录');
  
  // 只订阅主题
  const theme = useSettingsSelector(settings => settings.theme);
  
  // 只订阅加载状态
  const loading = useUISelector(ui => ui.loading);
  
  const { actions } = useAppState();

  return (
    <div className={`app ${theme}`}>
      <div>用户: {userName}</div>
      <div>主题: {theme}</div>
      {loading && <div>加载中...</div>}
      <button onClick={actions.toggleTheme}>切换主题</button>
      <button onClick={actions.toggleSidebar}>切换侧边栏</button>
    </div>
  );
});

ExampleUsage.displayName = 'ExampleUsage';
