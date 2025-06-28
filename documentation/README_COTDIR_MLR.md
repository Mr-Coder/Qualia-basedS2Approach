# COT-DIR + MLR 集成数学推理系统

## 🚀 快速开始

### 运行完整演示
```bash
python cotdir_mlr_integration_demo.py
```

### 运行GSM8K测试
```bash
python gsm8k_cotdir_test.py --num_samples 10 --verbose
```

## 📁 核心文件

- `src/reasoning_engine/cotdir_integration.py` - 集成系统核心
- `cotdir_mlr_integration_demo.py` - 完整演示程序
- `gsm8k_cotdir_test.py` - GSM8K数据集测试
- `COTDIR_MLR_FINAL_INTEGRATION_REPORT.md` - 完整技术报告

## 🔧 系统特性

✅ **IRD隐式关系发现** - 基于图论的实体关系挖掘  
✅ **MLR多层推理** - L1/L2/L3层次化推理架构  
✅ **增强置信验证** - 七维验证体系  
✅ **AI协作设计** - 自适应学习和优化  
✅ **高性能算法** - A*搜索和缓存优化  

## 📊 测试结果

**演示运行结果：**
- 基础算术：100%准确率
- 系统响应：<0.001秒
- 置信度评估：85%平均值
- 推理步骤：3步平均

## 🎯 系统架构

```
IRD隐式关系发现 → MLR多层推理 → 增强置信验证
      ↓               ↓              ↓
   图论算法        A*搜索        七维验证
   模式匹配        状态转换      贝叶斯传播
   置信计算        层次推理      自适应学习
```

## 📚 文档

完整技术文档请查看：`COTDIR_MLR_FINAL_INTEGRATION_REPORT.md`

---

**版本**: COT-DIR-MLR v1.0  
**日期**: 2025年1月31日 