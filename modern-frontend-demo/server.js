const express = require('express');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();
const PORT = 3000;

// 先处理API代理
app.use('/api', createProxyMiddleware({
  target: 'http://127.0.0.1:8000',
  changeOrigin: true,
  logLevel: 'debug',
  onProxyReq: (proxyReq, req, res) => {
    console.log(`[PROXY REQUEST] ${req.method} ${req.url}`);
  },
  onProxyRes: (proxyRes, req, res) => {
    console.log(`[PROXY RESPONSE] ${req.url} -> ${proxyRes.statusCode}`);
  },
  onError: (err, req, res) => {
    console.error('[PROXY ERROR]', req.url, err.message);
    if (!res.headersSent) {
      res.status(502).json({
        error: 'Proxy Error',
        message: err.message,
        hint: '请确保后端服务器在 http://127.0.0.1:8000 运行'
      });
    }
  }
}));

// 然后处理静态文件
app.use(express.static(path.join(__dirname, 'dist')));

// 最后处理SPA路由
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`\n✅ 服务器启动成功！`);
  console.log(`📁 访问地址: http://localhost:${PORT}`);
  console.log(`🔗 API代理: /api/* -> http://127.0.0.1:8000/api/*`);
  console.log(`📂 静态文件: ${path.join(__dirname, 'dist')}\n`);
});