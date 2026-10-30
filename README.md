# Hermes MobilityBench

本项目包含 Hermes Agent 和 MobilityBench 评估框架。

## 项目结构

```
├── TEMP-FILE-STATION/
│   ├── hermes-agent/          # Hermes Agent 主项目
│   └── MB-with-agent-adapter/ # MobilityBench 评估适配器
```

## 主要组件

### 1. Hermes Agent
- **位置**: `TEMP-FILE-STATION/hermes-agent/`
- **描述**: Hermes Agent 核心实现

### 2. MobilityBench with Agent Adapter
- **位置**: `TEMP-FILE-STATION/MB-with-agent-adapter/`
- **描述**: MobilityBench 评估框架及 Agent 适配器
- **功能**: 
  - 模型评估脚本
  - 指标计算和汇总
  - 结果可视化

## 快速开始

### 启动模型服务

```bash
cd TEMP-FILE-STATION/MB-with-agent-adapter/scripts/hermes_eval
./serve_qwen3.8-27b.sh
```

### 运行评估

```bash
cd TEMP-FILE-STATION/MB-with-agent-adapter/scripts/hermes_eval
./run_eval_qwen3.8-27b.sh
```

## 评估结果

评估结果保存在 `TEMP-FILE-STATION/MB-with-agent-adapter/data/results/` 目录下。

## 注意事项

- 确保 GPU 资源充足（至少需要足够的显存来加载模型）
- 启动新模型服务前，请先停止其他占用 GPU 的服务
- 使用 `nvidia-smi` 检查 GPU 使用情况

## 相关链接

- [Hermes Agent](TEMP-FILE-STATION/hermes-agent/)
- [MobilityBench](TEMP-FILE-STATION/MB-with-agent-adapter/)
