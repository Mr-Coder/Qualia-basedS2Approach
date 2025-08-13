import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

const RoomsScreen: React.FC = () => {
  return (
    <View style={styles.container}>
      <Text style={styles.text}>Study Rooms Screen</Text>
      <Text style={styles.subtext}>To be implemented</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  text: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  subtext: {
    fontSize: 16,
    color: '#666',
  },
});

export default RoomsScreen;