# COT-DIR 现代化前端系统

基于React + Radix UI + Tailwind CSS + Framer Motion的现代化数学推理系统前端界面。

## 🌟 特性

- **现代化设计**: 采用玻璃拟态设计和流畅动画
- **智能推理**: 支持COT(思维链)、GOT(思维图)、TOT(思维树)多种推理策略
- **实体关系图**: 自动生成数学问题实体关系可视化
- **响应式布局**: 支持桌面和移动设备
- **实时推理**: 连接后端API进行实时问题求解

## 🚀 快速开始

### 1. 启动后端服务器

```bash
cd modern-frontend-demo
python backend_server.py
```

服务器将在 http://localhost:3001 启动

### 2. 打开前端界面

有三种方式访问前端：

#### 方式一：直接打开HTML文件
```bash
open integrated-demo.html
```

#### 方式二：通过HTTP服务器访问
```bash
# 使用Python内置服务器
python -m http.server 8000
# 访问 http://localhost:8000/integrated-demo.html
```

#### 方式三：React开发服务器（需要Node.js）
```bash
npm install
npm run dev
# 访问 http://localhost:3000
```

## 📊 系统状态

系统支持两种模式：

- **真实算法模式**: 连接实际的COT-DIR推理算法
- **模拟模式**: 使用智能模拟结果，用于演示和测试

状态指示器会显示当前运行模式。

## 🔧 API接口

### 解题接口
```
POST /api/solve
Content-Type: application/json

{
  "problem": "数学问题文本",
  "strategy": "auto|cot|got|tot"
}
```

### 健康检查
```
GET /api/health
```

### 策略列表
```
GET /api/strategies
```

## 🎯 支持的问题类型

- 算术问题（加减乘除）
- 几何问题（面积、周长等）
- 应用题（人数分配、比例计算）
- 百分比和折扣问题
- 复杂数学推理问题

## 🛠️ 技术架构

### 前端技术栈
- **React 18**: 组件化界面
- **TypeScript**: 类型安全
- **Tailwind CSS**: 原子化CSS
- **Framer Motion**: 动画效果
- **Zustand**: 状态管理
- **Radix UI**: 无障碍组件

### 后端技术栈
- **Flask**: Web服务器
- **COT-DIR算法**: 数学推理引擎
- **NLP处理**: 自然语言理解
- **实体抽取**: 关系图生成

## 📁 项目结构

```
modern-frontend-demo/
├── src/                    # React源码
│   ├── components/        # UI组件
│   ├── stores/           # 状态管理
│   ├── services/         # API服务
│   └── utils/            # 工具函数
├── integrated-demo.html   # 完整演示页面
├── backend_server.py      # 后端API服务器
└── README.md             # 项目文档
```

## 🎨 组件库

### UI组件
- Button: 可定制按钮组件
- Card: 卡片容器组件
- Input/Textarea: 输入组件
- Select: 选择器组件

### 功能组件
- ProblemSolver: 问题解决器
- EntityRelationshipDiagram: 实体关系图
- StrategyAnalysis: 策略分析
- LearningGuide: 学习指导

## 🔍 开发调试

### 查看API日志
后端服务器会输出详细的API调用日志，包括：
- 请求参数
- 推理过程
- 错误信息

### 状态监控
前端会实时显示：
- 系统连接状态
- 算法可用性
- 推理策略选择

## 📈 性能特性

- **懒加载**: 组件按需加载
- **状态持久化**: 本地存储用户设置
- **错误边界**: 优雅的错误处理
- **响应式设计**: 适配各种屏幕尺寸

## 🎯 使用示例

### 示例1: 算术问题
```
输入: "小明有10个苹果，他给了小红3个，又买了5个，请问小明现在有多少个苹果？"
输出: 答案12个，包含详细推理步骤和实体关系图
```

### 示例2: 几何问题  
```
输入: "一个长方形的长是12厘米，宽是8厘米，求这个长方形的面积。"
输出: 答案96平方厘米，展示计算过程
```

## 🚨 故障排除

### 连接问题
- 确保后端服务器在3001端口运行
- 检查防火墙设置
- 验证CORS配置

### 推理问题
- 查看系统状态指示器
- 检查后端日志输出
- 尝试不同的推理策略

## 🔗 相关链接

- [COT-DIR算法文档](../README.md)
- [API文档](http://localhost:3001/api/health)
- [React文档](https://react.dev/)
- [Tailwind CSS](https://tailwindcss.com/)

## 📝 许可证

本项目遵循原COT-DIR项目的许可证。