export interface Theme {
  dark: boolean;
  colors: {
    primary: string;
    primaryLight: string;
    primaryDark: string;
    secondary: string;
    background: string;
    surface: string;
    surfaceVariant: string;
    text: string;
    textSecondary: string;
    textInverse: string;
    error: string;
    warning: string;
    success: string;
    info: string;
    border: string;
    divider: string;
    shadow: string;
    overlay: string;
  };
  spacing: {
    xs: number;
    sm: number;
    md: number;
    lg: number;
    xl: number;
    xxl: number;
  };
  borderRadius: {
    sm: number;
    md: number;
    lg: number;
    xl: number;
    full: number;
  };
}

export const lightTheme: Theme = {
  dark: false,
  colors: {
    primary: '#007AFF',
    primaryLight: '#5AC8FA',
    primaryDark: '#0051D5',
    secondary: '#5856D6',
    background: '#FFFFFF',
    surface: '#F9F9F9',
    surfaceVariant: '#F2F2F7',
    text: '#000000',
    textSecondary: '#6B6B6B',
    textInverse: '#FFFFFF',
    error: '#FF3B30',
    warning: '#FFA726',
    success: '#4CAF50',
    info: '#2196F3',
    border: '#E0E0E0',
    divider: '#C6C6C8',
    shadow: '#000000',
    overlay: 'rgba(0, 0, 0, 0.5)',
  },
  spacing: {
    xs: 4,
    sm: 8,
    md: 16,
    lg: 24,
    xl: 32,
    xxl: 48,
  },
  borderRadius: {
    sm: 4,
    md: 8,
    lg: 12,
    xl: 16,
    full: 999,
  },
};

export const darkTheme: Theme = {
  dark: true,
  colors: {
    primary: '#0A84FF',
    primaryLight: '#5AC8FA',
    primaryDark: '#0051D5',
    secondary: '#5E5CE6',
    background: '#000000',
    surface: '#1C1C1E',
    surfaceVariant: '#2C2C2E',
    text: '#FFFFFF',
    textSecondary: '#98989D',
    textInverse: '#000000',
    error: '#FF453A',
    warning: '#FFA726',
    success: '#32D74B',
    info: '#2196F3',
    border: '#38383A',
    divider: '#48484A',
    shadow: '#000000',
    overlay: 'rgba(0, 0, 0, 0.7)',
  },
  spacing: {
    xs: 4,
    sm: 8,
    md: 16,
    lg: 24,
    xl: 32,
    xxl: 48,
  },
  borderRadius: {
    sm: 4,
    md: 8,
    lg: 12,
    xl: 16,
    full: 999,
  },
};

// High contrast themes for accessibility
export const lightHighContrastTheme: Theme = {
  ...lightTheme,
  colors: {
    ...lightTheme.colors,
    primary: '#0055CC',
    text: '#000000',
    textSecondary: '#333333',
    background: '#FFFFFF',
    surface: '#F5F5F5',
    border: '#000000',
  },
};

export const darkHighContrastTheme: Theme = {
  ...darkTheme,
  colors: {
    ...darkTheme.colors,
    primary: '#66B2FF',
    text: '#FFFFFF',
    textSecondary: '#CCCCCC',
    background: '#000000',
    surface: '#1A1A1A',
    border: '#FFFFFF',
  },
};