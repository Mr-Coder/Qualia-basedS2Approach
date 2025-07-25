const express = require('express');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();
const PORT = 3000;

// 安装依赖
// npm install http-proxy-middleware

// 设置静态文件目录
app.use(express.static(path.join(__dirname, 'dist')));

// 代理所有/api请求到后端
app.use('/api', createProxyMiddleware({
  target: 'http://127.0.0.1:8000',
  changeOrigin: true,
  logLevel: 'debug',
  onProxyReq: (proxyReq, req, res) => {
    console.log(`[PROXY] ${req.method} ${req.url} -> http://127.0.0.1:8000${req.url}`);
  },
  onError: (err, req, res) => {
    console.error('[PROXY ERROR]', err.message);
    res.status(502).json({ 
      error: 'Proxy Error', 
      message: err.message,
      hint: '请确保后端服务器在 http://127.0.0.1:8000 运行'
    });
  }
}));

// 处理所有其他路由（SPA路由）
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`✅ 代理服务器运行在 http://localhost:${PORT}`);
  console.log(`📁 静态文件目录: ${path.join(__dirname, 'dist')}`);
  console.log(`🔗 API代理: /api/* -> http://127.0.0.1:8000/api/*`);
});