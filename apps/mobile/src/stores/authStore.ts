import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as SecureStore from 'expo-secure-store';

interface AuthTokens {
  accessToken: string;
  refreshToken: string;
}

interface User {
  id: string;
  email: string;
  name: string;
  role: 'student' | 'teacher';
}

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  tokens: AuthTokens | null;
  isLoading: boolean;
  biometricEnabled: boolean;
  
  // Actions
  login: (credentials: { email: string; password: string }) => Promise<void>;
  logout: () => Promise<void>;
  refreshToken: () => Promise<void>;
  setBiometric: (enabled: boolean) => void;
  checkAuthStatus: () => Promise<void>;
}

const API_URL = __DEV__ ? 'http://localhost:8000' : 'https://api.cot-dir.com';

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      isAuthenticated: false,
      tokens: null,
      isLoading: false,
      biometricEnabled: false,

      login: async (credentials) => {
        set({ isLoading: true });
        try {
          const response = await fetch(`${API_URL}/api/v1/auth/login`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(credentials),
          });

          if (!response.ok) {
            throw new Error('Login failed');
          }

          const data = await response.json();
          
          // Store tokens securely
          await SecureStore.setItemAsync('accessToken', data.accessToken);
          await SecureStore.setItemAsync('refreshToken', data.refreshToken);
          
          set({
            user: data.user,
            isAuthenticated: true,
            tokens: {
              accessToken: data.accessToken,
              refreshToken: data.refreshToken,
            },
            isLoading: false,
          });
        } catch (error) {
          set({ isLoading: false });
          throw error;
        }
      },

      logout: async () => {
        // Clear secure storage
        await SecureStore.deleteItemAsync('accessToken');
        await SecureStore.deleteItemAsync('refreshToken');
        
        set({
          user: null,
          isAuthenticated: false,
          tokens: null,
        });
      },

      refreshToken: async () => {
        try {
          const refreshToken = await SecureStore.getItemAsync('refreshToken');
          if (!refreshToken) {
            throw new Error('No refresh token');
          }

          const response = await fetch(`${API_URL}/api/v1/auth/refresh`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ refreshToken }),
          });

          if (!response.ok) {
            throw new Error('Token refresh failed');
          }

          const data = await response.json();
          
          await SecureStore.setItemAsync('accessToken', data.accessToken);
          
          set({
            tokens: {
              accessToken: data.accessToken,
              refreshToken: refreshToken,
            },
          });
        } catch (error) {
          // If refresh fails, logout
          get().logout();
          throw error;
        }
      },

      setBiometric: (enabled) => {
        set({ biometricEnabled: enabled });
      },

      checkAuthStatus: async () => {
        try {
          const accessToken = await SecureStore.getItemAsync('accessToken');
          if (accessToken) {
            // Verify token with backend
            const response = await fetch(`${API_URL}/api/v1/auth/me`, {
              headers: {
                Authorization: `Bearer ${accessToken}`,
              },
            });

            if (response.ok) {
              const userData = await response.json();
              set({
                user: userData,
                isAuthenticated: true,
                tokens: {
                  accessToken,
                  refreshToken: await SecureStore.getItemAsync('refreshToken') || '',
                },
              });
            } else {
              // Token invalid, try refresh
              await get().refreshToken();
            }
          }
        } catch (error) {
          // Silent fail - user needs to login
        }
      },
    }),
    {
      name: 'auth-storage',
      storage: createJSONStorage(() => AsyncStorage),
      partialize: (state) => ({
        biometricEnabled: state.biometricEnabled,
      }),
    }
  )
);