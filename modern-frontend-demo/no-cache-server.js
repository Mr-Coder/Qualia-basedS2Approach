import express from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function createNoCacheServer() {
  const app = express();
  
  // 创建Vite服务器
  const vite = await createViteServer({
    server: { middlewareMode: true },
    appType: 'spa',
    optimizeDeps: {
      force: true
    },
    build: {
      rollupOptions: {
        output: {
          manualChunks: undefined,
        }
      }
    }
  });

  // 强制无缓存中间件
  app.use((req, res, next) => {
    // 设置最强的无缓存头
    res.set({
      'Cache-Control': 'no-cache, no-store, must-revalidate, max-age=0, private',
      'Pragma': 'no-cache',
      'Expires': '0',
      'Last-Modified': new Date().toUTCString(),
      'ETag': Math.random().toString(),
      'Surrogate-Control': 'no-store',
      'Vary': '*',
      'X-Accel-Expires': '0'
    });
    
    // 禁用HTTP/2 server push
    res.set('Link', '');
    
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
    next();
  });

  // 使用Vite的中间件
  app.use(vite.ssrFixStacktrace);
  app.use(vite.middlewares);

  // 错误处理
  app.use((err, req, res, next) => {
    console.error('Server error:', err);
    vite.ssrFixStacktrace(err);
    res.status(500).end(err.message);
  });

  const port = 3001;
  
  app.listen(port, () => {
    console.log(`🚀 No-Cache Server running at:`);
    console.log(`   Local:   http://localhost:${port}/`);
    console.log(`   Network: http://192.168.0.134:${port}/`);
    console.log(`\n💡 This server forces no-cache on ALL requests`);
  });
}

createNoCacheServer().catch(console.error);