const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Configure code splitting
config.transformer = {
  ...config.transformer,
  minifierConfig: {
    keep_fnames: true,
    mangle: {
      keep_fnames: true,
    },
  },
};

// Optimize bundle size
config.resolver = {
  ...config.resolver,
  // Use .native.js extensions for React Native specific code
  sourceExts: ['js', 'jsx', 'json', 'ts', 'tsx', 'cjs', 'mjs'],
  // Prioritize platform-specific files
  platforms: ['ios', 'android'],
};

// Enable RAM bundles for faster startup
config.serializer = {
  ...config.serializer,
  createModuleIdFactory: () => {
    const fileToIdMap = new Map();
    let nextId = 0;
    return (path) => {
      let id = fileToIdMap.get(path);
      if (id == null) {
        id = nextId++;
        fileToIdMap.set(path, id);
      }
      return id;
    };
  },
};

// Configure async imports for code splitting
config.dynamicDepsInPackages = 'throwAtRuntime';

module.exports = config;