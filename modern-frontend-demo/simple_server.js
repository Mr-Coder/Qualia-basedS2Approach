const express = require('express');
const path = require('path');

const app = express();
const PORT = 3000;

// 提供静态文件
app.use(express.static(path.join(__dirname, 'dist')));

// 所有其他路由返回index.html（支持SPA路由）
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`✅ 服务器运行在 http://localhost:${PORT}`);
  console.log(`📁 提供文件目录: ${path.join(__dirname, 'dist')}`);
  console.log(`⚠️  注意：API请求需要直接访问 http://localhost:8000`);
});