# Hermes Agent × MobilityBench 评估项目

<div align="center">

**使用 Hermes Agent 在 MobilityBench 基准测试上进行路径规划 Agent 评估**

[![MobilityBench Paper](https://img.shields.io/badge/MobilityBench-arXiv-b31b1b)](https://arxiv.org/abs/2602.22638)
[![Hermes Agent](https://img.shields.io/badge/Hermes-Agent-blue)](https://github.com/NousResearch/hermes-agent)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## 📋 目录

- [项目概述](#-项目概述)
- [我的工作](#-我的工作)
- [技术架构](#-技术架构)
- [环境准备](#-环境准备)
- [快速开始](#-快速开始)
- [评估流程](#-评估流程)
- [评估结果](#-评估结果)
- [项目结构](#-项目结构)
- [常见问题](#-常见问题)
- [引用](#-引用)

---

## 📖 项目概述

本项目实现了 **Hermes Agent** 与 **MobilityBench** 基准测试的集成适配，用于在真实世界移动场景中对基于 LLM 的路径规划 Agent 进行系统性评估。

### 核心贡献

✅ **Hermes Agent 适配器开发** - 将 Hermes Agent 集成到 MobilityBench 评估框架  
✅ **完整评估流水线** - 从模型服务启动、并发评估到结果汇总的全流程自动化  
✅ **多模型对比实验** - 支持 Qwen3.5-4B、Qwen3.8-27B、DeepSeek-V4-Flash 等多模型评估  
✅ **大规模并行测试** - 实现 5000 条样本的 32-worker 并发评估，支持断点续跑  

### 背景介绍

- **MobilityBench**: KDD 2026 Oral 接受的基准测试，包含 100K 真实用户查询，覆盖全球 350+ 城市的路径规划场景
- **Hermes Agent**: Nous Research 开发的通用 Agent 框架，支持多种 LLM 和工具调用
- **本项目的目标**: 验证 Hermes Agent 在复杂移动场景中的路径规划能力

---

## 💼 我的工作

### 1. Hermes Agent 与 MobilityBench 集成

开发了完整的适配器层，使 Hermes Agent 能够：
- 解析 MobilityBench 的数据格式（query, context, tool_list, route_ans）
- 调用 MobilityBench 的沙箱工具（路线规划、POI 查询、天气等 12+ 种工具）
- 生成符合评估要求的输出格式

**关键文件**:
- `src/mobility_bench/agent/frameworks/hermes.py` - Hermes Agent 框架适配器
- `configs/agent/default.yaml` - Agent 配置
- `mcp_server/` - MCP 沙箱服务器实现

### 2. 自动化评估流水线

设计了端到端的评估脚本，实现：

```bash
# 一键启动模型服务 → 运行评估 → 停止服务
./serve_qwen3.8-27b.sh    # 启动 vLLM 服务
./run_eval_qwen3.8-27b.sh # 运行 5000 条样本评估
./stop_servers.sh         # 清理资源
```

**核心特性**:
- ✅ 自动检测 GPU 资源，避免显存冲突
- ✅ 后台启动 vLLM 服务，实时显示启动日志
- ✅ 32-worker 并发评估，加速测试进程
- ✅ 断点续跑支持，中断后可继续
- ✅ 自动生成评估报告和指标汇总

**关键文件**:
- `scripts/hermes_eval/serve_*.sh` - 模型服务启动脚本
- `scripts/hermes_eval/run_eval_*.sh` - 评估运行脚本
- `scripts/hermes_eval/lib.sh` - 共享库函数
- `scripts/hermes_eval/env.sh` - 环境变量配置

### 3. 多模型对比实验

完成了三个主流模型的全面评估：

| 模型 | 参数量 | GPU 配置 | 样本数 | 状态 |
|------|--------|----------|--------|------|
| Qwen3.5-4B | 4B | 单卡 H20 (TP=1) | 5000 | ✅ 完成 |
| Qwen3.8-27B | 27B | 双卡 H20 (TP=2) | 5000 | ✅ 完成 |
| DeepSeek-V4-Flash | - | 双卡 H20 (TP=2, FP8) | 5000 | ✅ 完成 |

**评估维度**（5 大类、11 个指标）:
- **Tool Use**: TS_P, TS_R, SC（工具调用精确率、召回率、成功率）
- **Planning**: DEC_P, DEC_R（决策精确率、召回率）
- **Efficiency**: IT, OT（迭代次数、总耗时）
- **Instruction**: ID, IE（指令理解、执行效率）
- **Answer**: DR, FPR（答案准确率、误报率）

### 4. 结果分析与可视化

实现了详细的结果汇总和报告生成：

```
data/results/hermes_qwen3.5-4b_5000/
├── report.md                  # Markdown 格式的详细报告
├── case_summary.csv           # 逐题明细（5000 条）
└── evaluation/
    └── metrics_summary.md     # 指标汇总表
```

**分析功能**:
- 按意图类别聚合统计（Basic Info, Route Planning 等）
- 失败案例分类诊断（工具调用错误、超时、格式错误等）
- 性能瓶颈分析（迭代次数分布、耗时统计）

**关键文件**:
- `scripts/merge_eval_summaries.py` - 多模型结果合并
- `scripts/summarize_hermes_batch.py` - 批量结果汇总
- `scripts/diag_answer_failures.py` - 失败案例诊断

---

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                    我的工作流程                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 环境准备                                                 │
│     ├── 安装依赖 (uv sync)                                   │
│     ├── 下载数据集 (HuggingFace)                             │
│     └── 下载模型 (vLLM compatible)                           │
│                                                              │
│  2. 启动模型服务                                             │
│     ├── serve_qwen3.5-4b.sh  → vLLM @ :8000                 │
│     ├── serve_qwen3.8-27b.sh → vLLM @ :8000 (TP=2)          │
│     └── serve_deepseek-v4-flash.sh → vLLM @ :8000 (FP8)     │
│                                                              │
│  3. 运行评估                                                 │
│     ├── run_eval_*.sh                                        │
│     │   ├── 加载 5000 条样本                                 │
│     │   ├── 32-worker 并发调度                               │
│     │   ├── Hermes Agent 推理                                │
│     │   │   ├── 解析 query + context                         │
│     │   │   ├── 规划工具调用序列                              │
│     │   │   ├── 调用 MCP 沙箱工具                            │
│     │   │   └── 生成最终答案                                 │
│     │   ├── MobilityBench 评估器                             │
│     │   │   ├── Tool Use 评分                                │
│     │   │   ├── Planning 评分                                │
│     │   │   ├── Efficiency 评分                              │
│     │   │   ├── Instruction 评分                             │
│     │   │   └── Answer 评分                                  │
│     │   └── 保存结果 (CSV + JSON)                            │
│     └── 断点续跑支持                                         │
│                                                              │
│  4. 结果汇总                                                 │
│     ├── metrics_summary.md  (指标汇总)                       │
│     ├── report.md           (详细报告)                       │
│     └── case_summary.csv    (逐题明细)                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ 环境准备

### 系统要求

- **操作系统**: Linux (Ubuntu 20.04+) / macOS / Windows (WSL2)
- **Python**: 3.12+
- **GPU**: NVIDIA GPU with CUDA (推荐 H20/A100/V100，至少 24GB 显存)
- **磁盘空间**: 至少 100GB（模型 + 数据集）
- **内存**: 至少 32GB RAM

### 安装步骤

#### 1. 克隆仓库

```bash
git clone https://github.com/LiuJW111/hermes_Mobilitybench.git
cd hermes_Mobilitybench/TEMP-FILE-STATION/MB-with-agent-adapter
```

#### 2. 安装依赖

```bash
# 推荐使用 uv（速度快）
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 或使用 pip
pip install -e .
pip install -e ".[eval]"
```

#### 3. 验证安装

```bash
python -m mobility_bench --help
```

---

## 🚀 快速开始

### 前置条件

1. **下载 MobilityBench 数据集**

```bash
# 方式 1: huggingface-cli
pip install huggingface_hub
huggingface-cli download GD-ML/MobilityBench \
  pool_evaluable_20261028_sample5000.csv \
  --repo-type dataset --local-dir data/datasets

# 方式 2: 手动下载
# 访问 https://huggingface.co/datasets/GD-ML/MobilityBench
# 下载 pool_evaluable_20261028_sample5000.csv
# 放置到 data/datasets/ 目录
```

2. **下载模型权重**

```bash
# 设置模型目录
export HF_HOME=/mnt/data/models

# 下载 Qwen3.5-4B
huggingface-cli download Qwen/Qwen3.5-4B \
  --local-dir /mnt/data/models/Qwen3.5-4B

# 下载 Qwen3.8-27B
huggingface-cli download Qwen/Qwen3.8-27B \
  --local-dir /mnt/data/models/Qwen3.8-27B

# 下载 DeepSeek-V4-Flash
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /mnt/data/models/DeepSeek-V4-Flash-0731
```

3. **下载 Embedding 模型**

```bash
.venv/bin/python scripts/fetch_embedding_weights.py
```

4. **配置环境变量**

```bash
cp .env.h20.example .env.h20
# 编辑 .env.h20，设置模型路径
```

### 运行评估

#### 评估 Qwen3.5-4B

```bash
cd scripts/hermes_eval

# Step 1: 启动模型服务（约 3-8 分钟）
./serve_qwen3.5-4b.sh
# 等待看到 "Uvicorn running on http://127.0.0.1:8000"

# Step 2: 运行评估（5000 条样本，32 worker 并发）
./run_eval_qwen3.5-4b.sh
# 实时显示进度: Progress: 1234/5000 (24.7%) | ETA: 45 min

# Step 3: 停止服务
./stop_servers.sh qwen3.5-4b
```

#### 评估 Qwen3.8-27B

```bash
cd scripts/hermes_eval

./serve_qwen3.8-27b.sh   # 双卡 TP=2，约 5-10 分钟
./run_eval_qwen3.8-27b.sh
./stop_servers.sh qwen3.8-27b
```

#### 评估 DeepSeek-V4-Flash

```bash
cd scripts/hermes_eval

./serve_deepseek-v4-flash.sh  # 双卡 TP=2, FP8，约 15-20 分钟
./run_eval_deepseek-v4-flash.sh
./stop_servers.sh deepseek-v4-flash
```

### 查看结果

```bash
# 查看指标汇总
cat data/results/hermes_qwen3.5-4b_5000/evaluation/metrics_summary.md

# 查看详细报告
cat data/results/hermes_qwen3.5-4b_5000/report.md

# 查看逐题明细
head -n 20 data/results/hermes_qwen3.5-4b_5000/case_summary.csv
```

---

## 📊 评估流程详解

### 完整流程图

```
启动 vLLM 服务
    ↓
检查 GPU 资源 (nvidia-smi)
    ↓
后台启动 vLLM (@ port 8000)
    ↓
等待服务就绪 (health check)
    ↓
加载数据集 (5000 条样本)
    ↓
初始化 32 workers
    ↓
┌──────────────────────────────┐
│  并发处理每个样本             │
│  ├─ Hermes Agent 推理        │
│  │   ├─ 解析 query           │
│  │   ├─ 规划工具调用          │
│  │   ├─ 执行 MCP 工具         │
│  │   └─ 生成答案             │
│  ├─ MobilityBench 评估       │
│  │   ├─ Tool Use 评分        │
│  │   ├─ Planning 评分        │
│  │   ├─ Efficiency 评分      │
│  │   ├─ Instruction 评分     │
│  │   └─ Answer 评分          │
│  └─ 保存结果                 │
└──────────────────────────────┘
    ↓
汇总所有结果
    ↓
生成报告 (metrics_summary.md)
    ↓
生成逐题明细 (case_summary.csv)
    ↓
停止 vLLM 服务
```

### 关键参数说明

**服务启动参数** (`serve_*.sh`):

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--max-model-len` | 最大上下文长度 | 32768 |
| `--tensor-parallel-size` | Tensor 并行度 | 1 (4B), 2 (27B) |
| `--gpu-memory-utilization` | GPU 显存利用率 | 0.7 (4B), 0.95 (27B) |
| `--dtype` | 数据类型 | bf16, fp8 |

**评估参数** (`run_eval_*.sh`):

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `WORKERS` | 并发 worker 数量 | 32 |
| `LIMIT` | 评估样本数限制 | 5000 |
| `DATASET` | 数据集路径 | `data/datasets/pool_evaluable_20261028_sample5000.csv` |

### 断点续跑

如果评估中断，直接重新运行即可，脚本会自动跳过已完成的样本：

```bash
./run_eval_qwen3.5-4b.sh  # 自动从中断处继续
```

---

## 📈 评估结果

### 指标说明

| 维度 | 指标 | 全称 | 含义 | 越...越好 |
|------|------|------|------|-----------|
| **Tool Use**<br>(工具使用) | TS_P | Tool Selection Precision | 工具选择精确率（覆盖率）- 预测的工具中覆盖期望工具的比例 | 高 |
| | TS_R | Tool Selection Recall | 工具选择召回率（非冗余度）- 1 减去冗余率，衡量是否有多余调用 | 高 |
| | SC | Schema Compliance | 模式合规率 - 工具调用参数的完整性和正确性 | 高 |
| **Planning**<br>(规划能力) | DEC_P | Decomposition Precision | 分解精确率 - 预测步骤中匹配标准步骤的比例 | 高 |
| | DEC_R | Decomposition Recall | 分解召回率 - 标准步骤中被预测步骤覆盖的比例 | 高 |
| **Efficiency**<br>(效率) | IT | Input Tokens | 输入 token 总数 - Agent 消耗的上下文 token 数 | 低 |
| | OT | Output Tokens | 输出 token 总数 - Agent 生成的 token 数 | 低 |
| **Instruction**<br>(指令理解) | ID | Intent Detection | 意图检测准确度 - Agent 是否正确识别用户查询意图（语义相似度） | 高 |
| | IE | Information Extraction | 信息抽取准确度 - Agent 是否正确提取所有约束条件和槽位 | 高 |
| **Answer**<br>(答案质量) | DR | Delivery Rate | 交付率 - Agent 是否产生完整、可执行的输出（而非失败消息） | 高 |
| | FPR | Final Pass Rate | 最终通过率 - 解决方案是否满足所有用户约束条件（答案准确率） | 高 |

**重要说明**：

- **TS_P vs TS_R**: TS_P 关注"是否调用了需要的工具"（覆盖率），TS_R 关注"是否有冗余调用"（非冗余度）
- **IT & OT**: 这两个是绝对值（token 数量），不是比率，数值越低表示效率越高
- **DR vs FPR**: DR 检查是否给出了有效回答（而不是"无法提供"等失败消息），FPR 检查回答内容是否正确
- **原生评分器**: 如果 embedding 模型缺失，ID/IE/FPR 会退化为简单的字符相似度匹配，结果可能失真

### 结果示例

查看实际评估结果：

```
# Qwen3.5-4B 结果
cat data/results/hermes_qwen3.5-4b_5000/evaluation/metrics_summary.md

# Qwen3.8-27B 结果
cat data/results/hermes_qwen3.8-27b_5000/evaluation/metrics_summary.md

# DeepSeek-V4-Flash 结果
cat data/results/hermes_deepseek-v4-flash_5000/evaluation/metrics_summary.md
```

### 结果对比

使用合并脚本对比多个模型：

```
python scripts/merge_eval_summaries.py \
  --results-dir data/results \
  --output comparison_report.md
```

---

## 📁 项目结构

```
MB-with-agent-adapter/
├── configs/                    # 配置文件
│   ├── agent/                  # Hermes Agent 配置
│   │   └── default.yaml
│   ├── datasets/               # 数据集配置
│   │   └── mobility_6262.yaml
│   ├── evaluation/             # 评估配置
│   │   └── default.yaml
│   └── models/                 # 模型配置
│       └── default.yaml
├── data/                       # 数据目录
│   ├── datasets/               # MobilityBench 数据集
│   │   └── pool_evaluable_20261028_sample5000.csv
│   └── results/                # 评估结果
│       ├── hermes_qwen3.5-4b_5000/
│       ├── hermes_qwen3.8-27b_5000/
│       └── hermes_deepseek-v4-flash_5000/
├── embedding-model/            # Embedding 模型（用于语义相似度评估）
├── mcp_server/                 # MCP 沙箱服务器（我的实现）
│   ├── sandbox_cli.py          # 沙箱命令行接口
│   ├── sandbox_tools.py        # 工具注册和执行
│   └── shared_sandbox.py       # 共享沙箱管理
├── scripts/                    # 脚本目录（我的核心工作）
│   ├── hermes_eval/            # Hermes 评估流水线
│   │   ├── README.md           # 评估脚本说明
│   │   ├── env.sh              # 环境变量配置
│   │   ├── lib.sh              # 共享库函数
│   │   ├── setup_env.sh        # 环境初始化
│   │   ├── serve_qwen3.5-4b.sh # 启动 Qwen3.5-4B 服务
│   │   ├── serve_qwen3.8-27b.sh# 启动 Qwen3.8-27B 服务
│   │   ├── serve_deepseek-v4-flash.sh # 启动 DeepSeek 服务
│   │   ├── run_eval_qwen3.5-4b.sh     # 运行 Qwen3.5-4B 评估
│   │   ├── run_eval_qwen3.8-27b.sh    # 运行 Qwen3.8-27B 评估
│   │   ├── run_eval_deepseek-v4-flash.sh # 运行 DeepSeek 评估
│   │   └── stop_servers.sh     # 停止所有服务
│   ├── fetch_embedding_weights.py   # 下载 embedding 模型
│   ├── merge_eval_summaries.py      # 合并多模型结果
│   ├── summarize_hermes_batch.py    # 批量结果汇总
│   └── diag_answer_failures.py      # 失败案例诊断
├── src/                        # 源代码
│   └── mobility_bench/         # MobilityBench 框架
│       ├── agent/              # Agent 框架（含 Hermes 适配器）
│       │   └── frameworks/
│       │       └── hermes.py   # ← 我的 Hermes 适配器实现
│       ├── dataset/            # 数据集加载器
│       ├── evaluation/         # 评估模块
│       │   └── metrics/        # 5 大维度指标实现
│       ├── tools/              # 工具实现（12+ 种）
│       │   └── sandbox/        # 沙箱工具
│       └── reporting/          # 报告生成
├── tests/                      # 测试用例
│   └── test_hermes_source_integration.py # Hermes 集成测试
├── .env.h20.example            # H20 服务器配置示例
├── pyproject.toml              # Python 项目依赖
└── README.md                   # 本文档
```

---

## ❓ 常见问题

### Q1: GPU 显存不足怎么办？

**A:** 
```bash
# 检查当前 GPU 使用情况
nvidia-smi

# 降低并发 worker 数量
sed -i 's/WORKERS=32/WORKERS=8/' scripts/hermes_eval/run_eval_qwen3.8-27b.sh

# 或停止其他占用 GPU 的进程
./stop_servers.sh all
```

### Q2: vLLM 服务启动失败？

**A:**
```bash
# 检查端口是否被占用
lsof -i :8000

# 查看服务日志
cat logs/serve_qwen3.8-27b.log

# 强制重启（谨慎使用）
FORCE=1 ./serve_qwen3.8-27b.sh
```

### Q3: 评估速度太慢？

**A:**
- 增加 worker 数量（确保显存足够）
- 使用 `LIMIT=100` 先测试少量样本
- 检查网络延迟和 API 响应时间

### Q4: 如何查看实时进度？

**A:**
评估脚本会实时输出：
```
Progress: 1234/5000 (24.7%) | ETA: 45 min | Current: Case #1235
```

也可以查看日志：
```bash
tail -f logs/eval_qwen3.5-4b.log
```

### Q5: 评估中断后如何恢复？

**A:**
直接重新运行，自动跳过已完成样本：
```bash
./run_eval_qwen3.5-4b.sh
```

### Q6: 如何清理旧结果？

**A:**
```bash
# 删除特定模型结果
rm -rf data/results/hermes_qwen3.5-4b_*

# 删除所有结果
rm -rf data/results/*
```

### Q7: 如何添加新模型？

**A:**
1. 复制现有脚本模板：
```bash
cp scripts/hermes_eval/serve_qwen3.5-4b.sh scripts/hermes_eval/serve_new_model.sh
cp scripts/hermes_eval/run_eval_qwen3.5-4b.sh scripts/hermes_eval/run_eval_new_model.sh
```

2. 修改模型路径和参数
3. 运行评估

---

## 📝 引用

### MobilityBench

如果您使用了 MobilityBench 数据集或评估框架，请引用：

```
@article{song2026mobilitybench,
  title={MobilityBench: A Benchmark for Evaluating Route-Planning Agents in Real-World Mobility Scenarios},
  author={Song, Zhiheng and Zhang, Jingshuai and Qin, Chuan and Wang, Chao and Chen, Chao and Xu, Longfei and Liu, Kaikui and Chu, Xiangxiang and Zhu, Hengshu},
  journal={arXiv preprint arXiv:2602.22638},
  year={2026},
  note={Accepted at KDD 2026 (Oral)}
}
```

### Hermes Agent

如果您使用了 Hermes Agent，请引用：

```
@misc{hermes_agent,
  title={Hermes Agent: A General-Purpose Agent Framework},
  author={Nous Research},
  year={2024},
  howpublished={\url{https://github.com/NousResearch/hermes-agent}}
}
```

---

## 🙏 致谢

- **MobilityBench 团队** - 提供高质量的基准测试数据集和评估框架
- **Hermes Agent 团队** - 提供强大的通用 Agent 框架
- **Qwen & DeepSeek 团队** - 开源高质量的语言模型
- **vLLM 团队** - 提供高效的推理引擎

---

## 📞 联系方式

如有问题或建议，欢迎：
- 提交 [GitHub Issue](https://github.com/LiuJW111/hermes_Mobilitybench/issues)
- 发送邮件至：2750439797@qq.com

---

**Happy Evaluating! 🎉**
