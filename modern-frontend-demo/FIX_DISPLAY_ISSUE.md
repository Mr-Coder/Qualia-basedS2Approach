# 🔧 修复物性推理组件显示问题

## 问题诊断

从您的错误日志看，主要有两个问题：

1. **API 端口错误**：一些组件尝试连接端口 5001，但 API 服务器运行在端口 5000
2. **API 服务器未运行**：显示 "ERR_CONNECTION_REFUSED" 错误

## ✅ 解决方案

### 1. 端口问题已修复
我已经将以下文件中的端口从 5001 改为 5000：
- `IRDAlgorithmVisualization.tsx`
- `SimplifiedConstraintVisualization.tsx`
- `EnhancedPhysicsConstraintVisualization.tsx`

### 2. 快速启动方法

#### 方法一：使用快速启动脚本（推荐）
```bash
cd modern-frontend-demo
./quick_start.sh
```

#### 方法二：手动启动
```bash
# 终端 1 - API 服务器
cd modern-frontend-demo
python demo_frontend.py

# 终端 2 - 前端开发服务器
cd modern-frontend-demo
npm run dev
```

### 3. 验证步骤

1. **确认服务器运行**
   - API 服务器：访问 http://localhost:5000/api/test
   - 前端服务器：访问 http://localhost:3000

2. **清除浏览器缓存**
   - 使用 Ctrl/Cmd + Shift + R 硬刷新
   - 或使用无痕/隐私模式

3. **访问物性推理页面**
   - 打开 http://localhost:3000
   - 点击 "🧩 物性推理" 标签页

## 📋 您应该看到的内容

### 1. 算法讲解部分（新增）
- 标题：基于物性关系的推理算法讲解
- 5个可折叠的部分：
  - 核心概念
  - 算法步骤（5步流程）
  - 推理规则（3条核心规则）
  - 算法示例
  - 关键洞察

### 2. 分步演示部分
- 标题：基于物性关系的推理 - 分步演示
- 3个示例问题选择
- "开始分步推理"按钮
- 5步骤的交互式演示

## 🚨 常见问题

### Q: 仍然看不到组件
A: 尝试以下步骤：
1. 完全关闭所有终端
2. 删除缓存：`rm -rf node_modules/.vite`
3. 使用 `./quick_start.sh` 重新启动
4. 使用浏览器的无痕模式访问

### Q: API 连接错误
A: 确保：
1. `demo_frontend.py` 正在运行
2. 没有其他程序占用端口 5000
3. 防火墙没有阻止本地连接

### Q: 控制台其他错误
A: `recordTabs.js` 错误是浏览器扩展引起的，可以忽略，不影响功能。

## 📝 总结

现在物性推理页面应该包含：
1. **算法讲解**：清晰展示算法的5个步骤和3条推理规则
2. **分步演示**：交互式展示每个步骤的图形化过程

这完全符合您的要求：将算法思想整理成清晰的步骤，说明"什么时候怎么做"，并通过实例演示。