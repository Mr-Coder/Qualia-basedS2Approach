# newfile项目 API演示完整指南
## 🚀 所有可用演示接口总览

---

## 📋 **主要演示API分类**

### **🎯 1. 核心算法演示 - 推荐使用**

#### **1.1 交互式演示系统** ⭐⭐⭐⭐⭐
```bash
python demos/interactive_demo.py
```
**功能**: 
- 最完整的交互式演示界面
- 支持自定义数学问题输入
- 显示完整推理过程
- 实体识别 → 关系发现 → 多层推理 → 答案生成

#### **1.2 详细逐步演示** ⭐⭐⭐⭐
```bash
python demos/detailed_step_by_step_demo.py
```
**功能**:
- 展示算法每个步骤的详细过程
- 包含推理链可视化
- 适合理解算法原理

#### **1.3 快速测试套件** ⭐⭐⭐
```bash
python demos/quick_test.py
```
**功能**:
- 5个预设测试案例
- 快速验证系统功能
- 支持逐个或批量演示

---

### **🔬 2. 高级系统演示**

#### **2.1 COT-DIR + MLR集成演示** ⭐⭐⭐⭐⭐
```bash
python demos/cotdir_mlr_integration_demo.py
```
**功能**:
- Chain of Thought with Directed Implicit Reasoning
- Multi-Layer Reasoning 多层推理
- 最新算法集成展示

#### **2.2 MLR增强演示最终版** ⭐⭐⭐⭐
```bash
python demos/mlr_enhanced_demo_final.py
```
**功能**:
- 多层推理(MLR)系统最终版本
- 支持L0-L3复杂度问题
- 五维验证机制

#### **2.3 MLR系统完整演示** ⭐⭐⭐⭐
```bash
python demos/mlr_demo_final.py
```
**功能**:
- MLR系统完整版演示
- 包含详细性能分析
- 适合技术评估

---

### **📊 3. 实验分析演示**

#### **3.1 高级实验演示** ⭐⭐⭐⭐
```bash
python demos/advanced_experimental_demo.py
```
**功能**:
- 综合性能评估
- 复杂度分析
- 错误分析报告

#### **3.2 AI协作设计演示** ⭐⭐⭐
```bash
python demos/ai_collaborative_demo.py
```
**功能**:
- AI协作推理演示
- 多模块协同工作
- 系统架构展示

---

### **🎨 4. 可视化演示**

#### **4.1 表格可视化演示**
```bash
# 表5：复杂度性能分析
python demos/visualizations/table5_visualization.py
python demos/visualizations/complete_table5_demo.py

# 表6：推理链质量评估  
python demos/visualizations/table6_visualization.py
python demos/visualizations/complete_table6_demo.py

# 表8：消融研究结果
python demos/visualizations/table8_visualization.py
python demos/visualizations/complete_table8_demo.py
```

---

### **📈 5. 性能分析演示**

#### **5.1 性能分析示例**
```bash
python demos/examples/performance_analysis_example.py
```
**功能**:
- 消融研究结果分析
- 组件贡献度计算
- 效率对比分析

#### **5.2 数据集分析示例**
```bash
python demos/examples/dataset_analysis_example.py
```
**功能**:
- 数据集复杂度分析
- 跨数据集性能对比
- 语言差异分析

#### **5.3 评估器使用示例**
```bash
python demos/examples/evaluator_usage_example.py
```
**功能**:
- 多维度评估演示
- 评估指标详解
- 评估结果解释

---

### **🔧 6. 工具与处理演示**

#### **6.1 复杂度分类演示**
```bash
python batch_complexity_classifier.py
```
**功能**:
- L0-L3复杂度自动分类
- 大规模数据集分析
- 分类置信度评估

#### **6.2 重构系统演示**
```bash
python demo_refactored_system.py
```
**功能**:
- 系统重构后功能展示
- 模块化架构演示
- 性能对比

---

### **🧪 7. 专项测试演示**

#### **7.1 GSM8K数据集测试**
```bash
python demos/gsm8k_cotdir_test.py --num_samples 10 --verbose
```
**功能**:
- GSM8K数据集专项测试
- COT-DIR算法验证
- 详细结果分析

#### **7.2 实验能力验证**
```bash
python experimental_validation_demo.py
```
**功能**:
- 实验框架完整性验证
- 系统能力评估
- 论文投稿就绪度检查

---

### **⚙️ 8. 底层模块演示**

#### **8.1 消融研究演示**
```bash
python src/evaluation/ablation_study.py
```
**功能**:
- 自动化消融研究
- 组件贡献度分析
- 统计显著性测试

#### **8.2 失效分析演示**
```bash
python src/evaluation/failure_analysis.py
```
**功能**:
- 错误模式识别
- 失效案例分类
- 鲁棒性评估

#### **8.3 计算复杂度分析**
```bash
python src/evaluation/computational_analysis.py
```
**功能**:
- 算法复杂度验证
- 性能基准测试
- 可扩展性分析

---

## 🎯 **推荐使用顺序**

### **新用户入门** (依次执行):
1. `python demos/quick_test.py` - 快速了解功能
2. `python demos/interactive_demo.py` - 深入体验交互
3. `python demos/detailed_step_by_step_demo.py` - 理解算法细节

### **技术评估** (推荐组合):
1. `python demos/cotdir_mlr_integration_demo.py` - 最新算法
2. `python demos/advanced_experimental_demo.py` - 性能分析
3. `python experimental_validation_demo.py` - 系统验证

### **研究开发** (完整流程):
1. `python batch_complexity_classifier.py` - 数据分类
2. `python demos/mlr_enhanced_demo_final.py` - 算法演示
3. `python demos/examples/performance_analysis_example.py` - 性能分析
4. `python experimental_framework.py` - 完整实验

---

## 📊 **演示结果展示**

### **每个演示都会生成**:
- ✅ **问题解析**: 文字 → 数学结构
- ✅ **实体识别**: 提取数值、对象、关系
- ✅ **关系发现**: 隐含关系识别
- ✅ **推理过程**: 逐步推理链
- ✅ **答案验证**: 多维度验证结果
- ✅ **性能指标**: 时间、准确率、置信度

### **表格解释方法**:
- **复杂度分布表**: L0显性 → L3深层的问题分布
- **性能对比表**: 不同算法在各数据集上的表现
- **消融研究表**: 各组件对整体性能的贡献
- **错误分析表**: 失效案例的分类统计

---

## 🚀 **立即开始演示**

选择您感兴趣的演示，直接运行对应命令即可！

**最推荐新手使用**:
```bash
cd /Users/menghao/Desktop/newfile
python demos/interactive_demo.py
```

**最适合技术展示**:
```bash
cd /Users/menghao/Desktop/newfile  
python demos/cotdir_mlr_integration_demo.py
```

**最完整的实验验证**:
```bash
cd /Users/menghao/Desktop/newfile
python experimental_validation_demo.py
``` 