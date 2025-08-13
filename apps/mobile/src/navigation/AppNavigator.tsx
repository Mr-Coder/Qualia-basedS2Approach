import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { AuthNavigator } from './AuthNavigator';
import { useAuthStore } from '../stores/authStore';

// Import screens (to be implemented)
import ChatScreen from '../screens/ChatScreen';
import RoomsScreen from '../screens/RoomsScreen';
import WhiteboardScreen from '../screens/WhiteboardScreen';
import ProfileScreen from '../screens/ProfileScreen';

const Stack = createStackNavigator();
const Tab = createBottomTabNavigator();

export type RootStackParamList = {
  Auth: undefined;
  Main: undefined;
};

export type MainTabParamList = {
  Rooms: undefined;
  Chat: { roomId: string };
  Whiteboard: { roomId: string };
  Profile: undefined;
};

const MainTabs = () => {
  return (
    <Tab.Navigator
      screenOptions={{
        tabBarStyle: {
          backgroundColor: '#fff',
          borderTopWidth: 1,
          borderTopColor: '#e0e0e0',
        },
        tabBarActiveTintColor: '#2196F3',
        tabBarInactiveTintColor: '#666',
      }}
    >
      <Tab.Screen
        name="Rooms"
        component={RoomsScreen}
        options={{
          title: 'Study Rooms',
          tabBarIcon: ({ color, size }) => (
            // Icon placeholder - will use actual icons later
            <></>
          ),
        }}
      />
      <Tab.Screen
        name="Chat"
        component={ChatScreen}
        options={{
          title: 'Chat',
          tabBarIcon: ({ color, size }) => (
            <></>
          ),
        }}
      />
      <Tab.Screen
        name="Whiteboard"
        component={WhiteboardScreen}
        options={{
          title: 'Whiteboard',
          tabBarIcon: ({ color, size }) => (
            <></>
          ),
        }}
      />
      <Tab.Screen
        name="Profile"
        component={ProfileScreen}
        options={{
          title: 'Profile',
          tabBarIcon: ({ color, size }) => (
            <></>
          ),
        }}
      />
    </Tab.Navigator>
  );
};

export const AppNavigator = React.forwardRef<any, any>((props, ref) => {
  const { isAuthenticated } = useAuthStore();

  return (
    <NavigationContainer ref={ref}>
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        {!isAuthenticated ? (
          <Stack.Screen name="Auth" component={AuthNavigator} />
        ) : (
          <Stack.Screen name="Main" component={MainTabs} />
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
});