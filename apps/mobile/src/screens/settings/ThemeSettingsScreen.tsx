import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Switch,
} from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import { useTheme } from '../../hooks/useTheme';
import { ResponsiveLayout } from '../../components/layout/ResponsiveLayout';
import { spacing, typography, scale } from '../../utils/responsive';

interface ThemeOption {
  mode: 'light' | 'dark' | 'system';
  label: string;
  icon: string;
}

const themeOptions: ThemeOption[] = [
  { mode: 'light', label: 'Light', icon: 'brightness-high' },
  { mode: 'dark', label: 'Dark', icon: 'brightness-2' },
  { mode: 'system', label: 'System', icon: 'brightness-auto' },
];

export const ThemeSettingsScreen: React.FC = () => {
  const { theme, themeMode, contrastMode, setThemeMode, setContrastMode } = useTheme();

  return (
    <ResponsiveLayout scrollable padded>
      <View style={styles.section}>
        <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>
          Appearance
        </Text>
        <Text style={[styles.sectionDescription, { color: theme.colors.textSecondary }]}>
          Choose how Math Solver looks to you
        </Text>

        <View style={styles.themeOptions}>
          {themeOptions.map((option) => (
            <TouchableOpacity
              key={option.mode}
              style={[
                styles.themeOption,
                {
                  backgroundColor: theme.colors.surface,
                  borderColor:
                    themeMode === option.mode ? theme.colors.primary : theme.colors.border,
                },
                themeMode === option.mode && styles.selectedOption,
              ]}
              onPress={() => setThemeMode(option.mode)}
              activeOpacity={0.7}
            >
              <MaterialIcons
                name={option.icon as any}
                size={scale(32)}
                color={themeMode === option.mode ? theme.colors.primary : theme.colors.text}
              />
              <Text
                style={[
                  styles.optionLabel,
                  {
                    color:
                      themeMode === option.mode ? theme.colors.primary : theme.colors.text,
                  },
                ]}
              >
                {option.label}
              </Text>
              {themeMode === option.mode && (
                <MaterialIcons
                  name="check-circle"
                  size={scale(20)}
                  color={theme.colors.primary}
                  style={styles.checkIcon}
                />
              )}
            </TouchableOpacity>
          ))}
        </View>
      </View>

      <View style={[styles.divider, { backgroundColor: theme.colors.divider }]} />

      <View style={styles.section}>
        <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>
          Accessibility
        </Text>

        <View
          style={[
            styles.settingRow,
            {
              backgroundColor: theme.colors.surface,
              borderColor: theme.colors.border,
            },
          ]}
        >
          <View style={styles.settingInfo}>
            <Text style={[styles.settingLabel, { color: theme.colors.text }]}>
              High Contrast
            </Text>
            <Text style={[styles.settingDescription, { color: theme.colors.textSecondary }]}>
              Increase contrast for better visibility
            </Text>
          </View>
          <Switch
            value={contrastMode === 'high'}
            onValueChange={(value) => setContrastMode(value ? 'high' : 'normal')}
            trackColor={{
              false: theme.colors.border,
              true: theme.colors.primaryLight,
            }}
            thumbColor={contrastMode === 'high' ? theme.colors.primary : '#f4f3f4'}
          />
        </View>
      </View>

      <View style={[styles.divider, { backgroundColor: theme.colors.divider }]} />

      <View style={styles.section}>
        <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>Preview</Text>
        
        <View
          style={[
            styles.previewContainer,
            {
              backgroundColor: theme.colors.surface,
              borderColor: theme.colors.border,
            },
          ]}
        >
          <View style={styles.previewHeader}>
            <View
              style={[
                styles.previewAvatar,
                { backgroundColor: theme.colors.primary },
              ]}
            >
              <Text style={[styles.previewAvatarText, { color: theme.colors.textInverse }]}>
                S
              </Text>
            </View>
            <View style={styles.previewHeaderText}>
              <Text style={[styles.previewName, { color: theme.colors.text }]}>
                Sample Student
              </Text>
              <Text style={[styles.previewSubtext, { color: theme.colors.textSecondary }]}>
                Active now
              </Text>
            </View>
          </View>

          <View
            style={[
              styles.previewMessage,
              { backgroundColor: theme.colors.background },
            ]}
          >
            <Text style={[styles.previewMessageText, { color: theme.colors.text }]}>
              This is how messages will appear in {theme.dark ? 'dark' : 'light'} mode
            </Text>
          </View>

          <View style={styles.previewActions}>
            <TouchableOpacity
              style={[
                styles.previewButton,
                {
                  backgroundColor: theme.colors.primary,
                },
              ]}
            >
              <Text style={[styles.previewButtonText, { color: theme.colors.textInverse }]}>
                Primary Button
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[
                styles.previewButton,
                styles.secondaryButton,
                {
                  borderColor: theme.colors.primary,
                },
              ]}
            >
              <Text style={[styles.previewButtonText, { color: theme.colors.primary }]}>
                Secondary
              </Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </ResponsiveLayout>
  );
};

const styles = StyleSheet.create({
  section: {
    marginBottom: spacing.xl,
  },
  sectionTitle: {
    ...typography.h3,
    marginBottom: spacing.xs,
  },
  sectionDescription: {
    ...typography.body2,
    marginBottom: spacing.md,
  },
  themeOptions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: spacing.sm,
  },
  themeOption: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.lg,
    borderRadius: scale(12),
    borderWidth: 2,
    position: 'relative',
  },
  selectedOption: {
    borderWidth: 2,
  },
  optionLabel: {
    ...typography.body2,
    marginTop: spacing.sm,
    fontWeight: '500',
  },
  checkIcon: {
    position: 'absolute',
    top: spacing.sm,
    right: spacing.sm,
  },
  divider: {
    height: 1,
    marginVertical: spacing.lg,
  },
  settingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: spacing.md,
    borderRadius: scale(12),
    borderWidth: 1,
  },
  settingInfo: {
    flex: 1,
    marginRight: spacing.md,
  },
  settingLabel: {
    ...typography.body1,
    fontWeight: '500',
    marginBottom: spacing.xs,
  },
  settingDescription: {
    ...typography.caption,
  },
  previewContainer: {
    padding: spacing.md,
    borderRadius: scale(12),
    borderWidth: 1,
  },
  previewHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  previewAvatar: {
    width: scale(40),
    height: scale(40),
    borderRadius: scale(20),
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: spacing.sm,
  },
  previewAvatarText: {
    ...typography.body1,
    fontWeight: '600',
  },
  previewHeaderText: {
    flex: 1,
  },
  previewName: {
    ...typography.body1,
    fontWeight: '500',
  },
  previewSubtext: {
    ...typography.caption,
  },
  previewMessage: {
    padding: spacing.sm,
    borderRadius: scale(8),
    marginBottom: spacing.md,
  },
  previewMessageText: {
    ...typography.body2,
  },
  previewActions: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  previewButton: {
    flex: 1,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    borderRadius: scale(8),
    alignItems: 'center',
  },
  secondaryButton: {
    backgroundColor: 'transparent',
    borderWidth: 1,
  },
  previewButtonText: {
    ...typography.body2,
    fontWeight: '500',
  },
});