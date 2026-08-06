# TRAJDEBUG

> [English](README.md)

**TRAJDEBUG：通过追踪错误生命周期定位长程 Agent 轨迹中的关键失败**，是一个基于证据的关键错误定位框架。本仓库提供 detector、统一数据适配器、评测工具与本地 viewer。

## 方法

TRAJDEBUG 先构建多粒度轨迹视图，再执行三个可审计阶段：

1. **错误触发检测：**识别错误承诺，并要求逐字引用错误内容与被违反的参照。
2. **错误状态分类：**按被违反对象聚合 trigger，判断错误是否修复及其终态影响。
3. **因果归因：**从与终态相关的候选中选择对失败负责的起源步。

![TRAJDEBUG pipeline 总览](assets/%20pipeline.png)

实现细节见 [`detector/README_zh.md`](detector/README_zh.md)，论文见 [`main.pdf`](main.pdf)。

## 结果

![关键错误检测结果](assets/results.png)

相关代码与数据正在进行内部审核和审批，审批完成后将尽快公开。感谢您的关注与理解。
