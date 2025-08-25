import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [
    react(),
    // 智能缓存策略插件
    {
      name: 'smart-cache-control',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          const url = req.url || '';
          
          // 对不同类型的资源应用不同的缓存策略
          if (url.includes('/src/') || url.endsWith('.tsx') || url.endsWith('.ts') || url.endsWith('.vue')) {
            // 源代码文件：短时间缓存，便于开发
            res.setHeader('Cache-Control', 'no-cache');
            res.setHeader('X-Cache-Type', 'source-code');
          } else if (url.includes('/node_modules/') || url.includes('/.vite/deps/')) {
            // 依赖包：长时间缓存，提升性能
            res.setHeader('Cache-Control', 'public, max-age=86400'); // 24小时
            res.setHeader('X-Cache-Type', 'dependencies');
          } else if (url === '/' || url.includes('.html')) {
            // HTML文件：短时间缓存，确保内容更新
            res.setHeader('Cache-Control', 'no-cache');
            res.setHeader('X-Cache-Type', 'html');
          } else if (url.includes('.css') || url.includes('.js') || url.includes('.png') || url.includes('.svg')) {
            // 静态资源：中等时间缓存
            res.setHeader('Cache-Control', 'public, max-age=3600'); // 1小时
            res.setHeader('X-Cache-Type', 'static-assets');
          } else {
            // 其他请求：默认无缓存
            res.setHeader('Cache-Control', 'no-cache');
            res.setHeader('X-Cache-Type', 'default');
          }
          
          // 开发环境标识
          res.setHeader('X-Dev-Server', 'vite');
          
          next();
        });
      }
    }
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  optimizeDeps: {
    // 移除强制重构建，允许正常的依赖缓存
    // force: true 
  },
  server: {
    port: 3000,
    host: '0.0.0.0', // 允许外部访问
    cors: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5001',
        changeOrigin: true,
        secure: false,
        ws: false, // 禁用WebSocket代理
        configure: (proxy, options) => {
          proxy.on('error', (err, req, res) => {
            console.log('API代理错误:', err.message);
            if (!res.headersSent) {
              res.writeHead(500, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify({ error: 'API代理错误', message: err.message }));
            }
          });
          proxy.on('proxyReq', (proxyReq, req, res) => {
            console.log(`[代理请求] ${req.method} ${req.url} -> ${options.target}${proxyReq.path}`);
          });
          proxy.on('proxyRes', (proxyRes, req, res) => {
            console.log(`[代理响应] ${proxyRes.statusCode} ${req.url}`);
          });
        }
      },
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: undefined,
        // 使用内容哈希而不是时间戳，允许正常缓存
        entryFileNames: `[name]-[hash].js`,
        chunkFileNames: `[name]-[hash].js`,
        assetFileNames: `[name]-[hash].[ext]`
      }
    }
  }
  // 移除构建时间定义，避免每次构建都变化
})