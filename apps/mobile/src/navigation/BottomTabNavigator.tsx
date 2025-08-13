import React from 'react';
import { Platform, StyleSheet, View, Text } from 'react-native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { MaterialIcons, MaterialCommunityIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useTheme } from '../hooks/useTheme';
import { scale, spacing, isTablet } from '../utils/responsive';

// Import screens
import RoomsScreen from '../screens/RoomsScreen';
import ChatScreen from '../screens/ChatScreen';
import WhiteboardScreen from '../screens/WhiteboardScreen';
import ProfileScreen from '../screens/ProfileScreen';
import { OfflineIndicator } from '../components/common/OfflineIndicator';

const Tab = createBottomTabNavigator();

interface TabIconProps {
  focused: boolean;
  color: string;
  size: number;
  route: string;
}

const TabIcon: React.FC<TabIconProps> = ({ focused, color, size, route }) => {
  const iconSize = scale(size);

  switch (route) {
    case 'Rooms':
      return <MaterialIcons name="dashboard" size={iconSize} color={color} />;
    case 'Chat':
      return <MaterialIcons name="chat" size={iconSize} color={color} />;
    case 'Whiteboard':
      return <MaterialCommunityIcons name="draw" size={iconSize} color={color} />;
    case 'Profile':
      return <MaterialIcons name="person" size={iconSize} color={color} />;
    default:
      return <MaterialIcons name="help" size={iconSize} color={color} />;
  }
};

interface TabLabelProps {
  focused: boolean;
  color: string;
  children: string;
}

const TabLabel: React.FC<TabLabelProps> = ({ focused, color, children }) => {
  const { theme } = useTheme();
  
  return (
    <Text
      style={[
        styles.tabLabel,
        {
          color,
          fontWeight: focused ? '600' : '400',
          fontSize: scale(isTablet ? 12 : 10),
        },
      ]}
      numberOfLines={1}
    >
      {children}
    </Text>
  );
};

export const BottomTabNavigator: React.FC = () => {
  const insets = useSafeAreaInsets();
  const { theme } = useTheme();

  const tabBarHeight = Platform.select({
    ios: scale(49) + insets.bottom,
    android: scale(56),
    default: scale(56),
  });

  return (
    <>
      <OfflineIndicator position="top" />
      <Tab.Navigator
        screenOptions={({ route }) => ({
          tabBarIcon: ({ focused, color, size }) => (
            <TabIcon focused={focused} color={color} size={size} route={route.name} />
          ),
          tabBarLabel: ({ focused, color, children }) => (
            <TabLabel focused={focused} color={color}>
              {children}
            </TabLabel>
          ),
          tabBarActiveTintColor: theme.colors.primary,
          tabBarInactiveTintColor: theme.colors.textSecondary,
          tabBarStyle: {
            backgroundColor: theme.colors.background,
            borderTopColor: theme.colors.border,
            borderTopWidth: StyleSheet.hairlineWidth,
            height: tabBarHeight,
            paddingBottom: Platform.OS === 'ios' ? insets.bottom : spacing.xs,
            paddingTop: spacing.xs,
            elevation: 8,
            shadowColor: '#000',
            shadowOffset: { width: 0, height: -2 },
            shadowOpacity: 0.1,
            shadowRadius: 4,
          },
          tabBarItemStyle: {
            paddingVertical: Platform.OS === 'ios' ? 0 : spacing.xs,
          },
          headerStyle: {
            backgroundColor: theme.colors.background,
            elevation: 0,
            shadowOpacity: 0,
            borderBottomWidth: StyleSheet.hairlineWidth,
            borderBottomColor: theme.colors.border,
          },
          headerTintColor: theme.colors.text,
          headerTitleStyle: {
            fontSize: scale(18),
            fontWeight: '600',
          },
          headerTitleAlign: 'center',
        })}
      >
        <Tab.Screen
          name="Rooms"
          component={RoomsScreen}
          options={{
            title: 'Study Rooms',
            tabBarLabel: 'Rooms',
          }}
        />
        <Tab.Screen
          name="Chat"
          component={ChatScreen}
          options={{
            title: 'Messages',
            tabBarLabel: 'Chat',
            tabBarBadge: undefined, // Can be set dynamically for unread messages
          }}
        />
        <Tab.Screen
          name="Whiteboard"
          component={WhiteboardScreen}
          options={{
            title: 'Whiteboard',
            tabBarLabel: 'Draw',
          }}
        />
        <Tab.Screen
          name="Profile"
          component={ProfileScreen}
          options={{
            title: 'Profile',
            tabBarLabel: 'Profile',
          }}
        />
      </Tab.Navigator>
    </>
  );
};

// Custom tab bar component for advanced customization
export const CustomTabBar: React.FC<any> = ({ state, descriptors, navigation }) => {
  const { theme } = useTheme();
  const insets = useSafeAreaInsets();

  return (
    <View
      style={[
        styles.customTabBar,
        {
          backgroundColor: theme.colors.background,
          borderTopColor: theme.colors.border,
          paddingBottom: insets.bottom,
        },
      ]}
    >
      {state.routes.map((route: any, index: number) => {
        const { options } = descriptors[route.key];
        const label = options.tabBarLabel ?? options.title ?? route.name;
        const isFocused = state.index === index;

        const onPress = () => {
          const event = navigation.emit({
            type: 'tabPress',
            target: route.key,
            canPreventDefault: true,
          });

          if (!isFocused && !event.defaultPrevented) {
            navigation.navigate(route.name);
          }
        };

        const onLongPress = () => {
          navigation.emit({
            type: 'tabLongPress',
            target: route.key,
          });
        };

        return (
          <View
            key={route.key}
            style={styles.customTabItem}
            accessibilityRole="button"
            accessibilityState={isFocused ? { selected: true } : {}}
            accessibilityLabel={options.tabBarAccessibilityLabel}
            testID={options.tabBarTestID}
          >
            <TabIcon
              focused={isFocused}
              color={isFocused ? theme.colors.primary : theme.colors.textSecondary}
              size={24}
              route={route.name}
            />
            <TabLabel
              focused={isFocused}
              color={isFocused ? theme.colors.primary : theme.colors.textSecondary}
            >
              {label}
            </TabLabel>
          </View>
        );
      })}
    </View>
  );
};

const styles = StyleSheet.create({
  tabLabel: {
    marginTop: spacing.xs,
    textAlign: 'center',
  },
  customTabBar: {
    flexDirection: 'row',
    borderTopWidth: StyleSheet.hairlineWidth,
    elevation: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  customTabItem: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.sm,
  },
});