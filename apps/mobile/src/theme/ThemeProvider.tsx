import React, { createContext, useState, useEffect, ReactNode } from 'react';
import { useColorScheme } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Theme, lightTheme, darkTheme, lightHighContrastTheme, darkHighContrastTheme } from './themes';

export type ThemeMode = 'light' | 'dark' | 'system';
export type ContrastMode = 'normal' | 'high';

interface ThemeContextType {
  theme: Theme;
  themeMode: ThemeMode;
  contrastMode: ContrastMode;
  setThemeMode: (mode: ThemeMode) => void;
  setContrastMode: (mode: ContrastMode) => void;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

const THEME_STORAGE_KEY = '@mathsolver_theme_mode';
const CONTRAST_STORAGE_KEY = '@mathsolver_contrast_mode';

interface ThemeProviderProps {
  children: ReactNode;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const systemColorScheme = useColorScheme();
  const [themeMode, setThemeMode] = useState<ThemeMode>('system');
  const [contrastMode, setContrastMode] = useState<ContrastMode>('normal');
  const [isLoading, setIsLoading] = useState(true);

  // Load saved preferences
  useEffect(() => {
    const loadPreferences = async () => {
      try {
        const [savedTheme, savedContrast] = await Promise.all([
          AsyncStorage.getItem(THEME_STORAGE_KEY),
          AsyncStorage.getItem(CONTRAST_STORAGE_KEY),
        ]);

        if (savedTheme) {
          setThemeMode(savedTheme as ThemeMode);
        }
        if (savedContrast) {
          setContrastMode(savedContrast as ContrastMode);
        }
      } catch (error) {
        console.error('Failed to load theme preferences:', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadPreferences();
  }, []);

  // Save theme mode
  const handleSetThemeMode = async (mode: ThemeMode) => {
    setThemeMode(mode);
    try {
      await AsyncStorage.setItem(THEME_STORAGE_KEY, mode);
    } catch (error) {
      console.error('Failed to save theme mode:', error);
    }
  };

  // Save contrast mode
  const handleSetContrastMode = async (mode: ContrastMode) => {
    setContrastMode(mode);
    try {
      await AsyncStorage.setItem(CONTRAST_STORAGE_KEY, mode);
    } catch (error) {
      console.error('Failed to save contrast mode:', error);
    }
  };

  // Toggle between light and dark themes
  const toggleTheme = () => {
    const newMode = themeMode === 'light' ? 'dark' : 'light';
    handleSetThemeMode(newMode);
  };

  // Determine which theme to use
  const getTheme = (): Theme => {
    const isDark = themeMode === 'dark' || (themeMode === 'system' && systemColorScheme === 'dark');
    
    if (contrastMode === 'high') {
      return isDark ? darkHighContrastTheme : lightHighContrastTheme;
    }
    
    return isDark ? darkTheme : lightTheme;
  };

  const theme = getTheme();

  const contextValue: ThemeContextType = {
    theme,
    themeMode,
    contrastMode,
    setThemeMode: handleSetThemeMode,
    setContrastMode: handleSetContrastMode,
    toggleTheme,
  };

  // Don't render until preferences are loaded
  if (isLoading) {
    return null;
  }

  return (
    <ThemeContext.Provider value={contextValue}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = (): ThemeContextType => {
  const context = React.useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};