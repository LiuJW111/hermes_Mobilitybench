# MobilityBench: LLM-Based Route-Planning Agent Evaluation Framework

<div align="center">

**MobilityBench** 是一个用于评估真实世界移动场景中基于 LLM 的路径规划 Agent 的可扩展基准测试框架。

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/2602.22638)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/GD-ML/MobilityBench/tree/main)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Accepted at KDD 2026 (Oral)**

</div>

---

## 📋 目录

- [项目介绍](#-项目介绍)
- [核心特性](#-核心特性)
- [系统架构](#-系统架构)
- [环境准备](#-环境准备)
  - [系统要求](#系统要求)
  - [安装依赖](#安装依赖)
- [数据集下载](#-数据集下载)
- [模型下载](#-模型下载)
- [快速开始](#-快速开始)
  - [配置环境变量](#配置环境变量)
  - [启动 vLLM 模型服务](#启动-vllm-模型服务)
  - [运行评估](#运行评估)
  - [查看结果](#查看结果)
- [详细使用指南](#-详细使用指南)
- [评估指标](#-评估指标)
- [项目结构](#-项目结构)
- [常见问题](#-常见问题)
- [引用](#-引用)

---

## 📖 项目介绍

**MobilityBench** 基于高德地图（Amap）收集的大规模、匿名化用户查询构建，涵盖全球多个城市的广泛路径规划意图。它提供了一个确定性的 API 回放沙箱环境，消除了实时服务的环境方差，支持可重现的端到端评估。

### 为什么需要 MobilityBench？

- **真实性**: 基于真实用户查询，覆盖 22 个国家、350+ 城市
- **可扩展性**: 100,000 个评估样本，支持大规模并行测试
- **多维度评估**: 不仅关注结果正确性，还评估指令理解、规划能力、工具使用和效率
- **可重现性**: 确定性沙箱确保每次评估结果一致

### 场景分布

- **36.6%** 基础信息检索（Basic Information Retrieval）
- **9.6%** 路线依赖信息检索（Route-Dependent Information Retrieval）
- **42.5%** 基础路径规划（Basic Route Planning）
- **11.3%** 偏好约束路径规划（Preference-Constrained Route Planning）

---

## ✨ 核心特性

✅ **确定性沙箱环境** - 消除外部 API 波动，确保评估可重现  
✅ **多维度评估协议** - 5 大维度、11 个子指标全面评估 Agent 能力  
✅ **大规模数据集** - 100K 样本，覆盖全球 350+ 城市  
✅ **灵活的工具调用** - 支持 12+ 种移动相关工具（路线规划、POI 查询、天气等）  
✅ **并发评估** - 支持多 worker 并行运行，加速评估过程  
✅ **断点续跑** - 支持中断后从上次位置继续，避免重复计算  
✅ **详细报告生成** - 自动生成可视化报告和逐题明细  

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    MobilityBench                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   Dataset    │    │   Config     │    │  Metrics  │ │
│  │   Loader     │───▶│   Manager    │───▶│Aggregator │ │
│  └──────────────┘    └──────────────┘    └───────────┘ │
│         │                                    ▲          │
│         ▼                                    │          │
│  ┌──────────────────────────────────────────────┐      │
│  │         Agent Framework (Hermes)              │      │
│  │  ┌─────────┐  ┌────────┐  ┌──────────────┐  │      │
│  │  │ Planner │─▶│ Worker │─▶│ Tool Executor│  │      │
│  │  └─────────┘  └────────┘  └──────┬───────┘  │      │
│  └──────────────────────────────────┼──────────┘      │
│                                     │                 │
│                          ┌──────────▼──────────┐      │
│                          │  Deterministic      │      │
│                          │  Sandbox (MCP)      │      │
│                          │  - Route Planning   │      │
│                          │  - POI Query        │      │
│                          │  - Weather/Traffic  │      │
│                          └─────────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ 环境准备

### 系统要求

- **操作系统**: Linux (推荐 Ubuntu 20.04+) / macOS / Windows (WSL2)
- **Python**: 3.12+
- **GPU**: NVIDIA GPU with CUDA support (推荐 H20/A100/V100，至少 24GB 显存)
- **磁盘空间**: 至少 100GB（模型 + 数据集）
- **内存**: 至少 32GB RAM

### 安装依赖

#### 1. 克隆仓库

```bash
git clone https://github.com/LiuJW111/hermes_Mobilitybench.git
cd hermes_Mobilitybench/TEMP-FILE-STATION/MB-with-agent-adapter
```

#### 2. 安装 uv（推荐的 Python 包管理器）

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或者使用 pip
pip install uv
```

#### 3. 创建虚拟环境并安装依赖

```bash
# 使用 uv 同步环境（推荐，速度快）
uv sync

# 如果需要评估相关的额外依赖
uv sync --extra eval

# 或者使用传统的 pip 方式
pip install -e .
pip install -e ".[eval]"
```

#### 4. 验证安装

```bash
# 检查 mobility-bench 是否正确安装
python -m mobility_bench --help

# 应该看到帮助信息
```

---

## 📊 数据集下载

### 官方数据集

MobilityBench 数据集托管在 Hugging Face，包含 100,000 个评估样本。

#### 下载方式 1: 使用 huggingface-cli（推荐）

```bash
# 安装 huggingface-cli
pip install huggingface_hub

# 下载完整数据集
huggingface-cli download GD-ML/MobilityBench --repo-type dataset --local-dir data/datasets

# 或者只下载评估子集（推荐，更快）
huggingface-cli download GD-ML/MobilityBench pool_evaluable_20261028_sample5000.csv \
  --repo-type dataset --local-dir data/datasets
```

#### 下载方式 2: 手动下载

1. 访问 [HuggingFace - MobilityBench](https://huggingface.co/datasets/GD-ML/MobilityBench/tree/main)
2. 下载需要的文件（如 `pool_evaluable_20261028_sample5000.csv`）
3. 将文件放置到 `data/datasets/` 目录

#### 数据集结构

```
data/datasets/
├── pool_evaluable_20261028_sample5000.csv  # 5000 条评估样本
├── mobility_6262.yaml                       # 数据集配置文件
└── ...                                      # 其他数据集
```

#### 数据格式示例

CSV 文件包含以下字段：

| 字段 | 说明 | 示例 |
|------|------|------|
| `query` | 用户查询文本 | "去大石桥不走高速" |
| `context` | 上下文信息（JSON） | `{"current_location": "...", "city": "..."}` |
| `task_scenario` | 细粒度任务类别 | "Option-Constrained Route Planning" |
| `intent_family` | 粗粒度意图类别 | "Preference-Constrained Route Planning" |
| `tool_list` | 期望的工具调用（JSON 数组） | `[{"tool": "driving_route", ...}]` |
| `route_ans` | 标准答案（JSON） | `{"distance": 12345, "duration": 600}` |

---

## 🤖 模型下载

### 支持的模型

本项目支持多种 LLM 模型，通过 vLLM 提供服务：

| 模型 | 参数量 | 推荐 GPU | 显存需求 |
|------|--------|----------|----------|
| Qwen3.5-4B | 4B | 单卡 H20/A100 | ~10GB |
| Qwen3.8-27B | 27B | 双卡 H20/A100 (TP=2) | ~60GB |
| DeepSeek-V4-Flash | - | 双卡 H20 (TP=2, FP8) | ~40GB |

### 下载模型

#### 方式 1: 使用 huggingface-cli

```bash
# 设置缓存目录（可选）
export HF_HOME=/mnt/data/models

# 下载 Qwen3.5-4B
huggingface-cli download Qwen/Qwen3.5-4B --local-dir /mnt/data/models/Qwen3.5-4B

# 下载 Qwen3.8-27B
huggingface-cli download Qwen/Qwen3.8-27B --local-dir /mnt/data/models/Qwen3.8-27B

# 下载 DeepSeek-V4-Flash
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /mnt/data/models/DeepSeek-V4-Flash-0731
```

#### 方式 2: 使用 ModelScope（国内镜像，速度更快）

```bash
# 安装 modelscope
pip install modelscope

# 下载模型
python -c "from modelscope import snapshot_download; \
  snapshot_download('Qwen/Qwen3.5-4B', cache_dir='/mnt/data/models')"
```

#### 方式 3: 手动下载

1. 访问 HuggingFace 或 ModelScope
2. 下载模型文件
3. 放置到指定目录（如 `/mnt/data/models/`）

### Embedding 模型

评估需要使用 embedding 模型来计算语义相似度：

```bash
# 自动下载 embedding 模型
cd /mnt/workspace/TEMP-FILE-STATION/MB-with-agent-adapter
.venv/bin/python scripts/fetch_embedding_weights.py

# 或者手动下载
huggingface-cli download Qwen/Qwen3-Embedding-0.6B \
  --local-dir embedding-model
```

---

## 🚀 快速开始

### 配置环境变量

#### 1. 创建 `.env` 文件

```bash
cp .env.api.example .env.api
cp .env.h20.example .env.h20
```

#### 2. 编辑配置文件

**`.env.api`** - API 配置（如果使用云端 API）:
```bash
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your-api-key-here
```

**`.env.h20`** - 本地 H20 服务器配置:
```bash
# 模型路径
MODEL_PATH_QWEN35_4B=/mnt/data/models/Qwen3.5-4B
MODEL_PATH_QWEN38_27B=/mnt/data/models/Qwen3.8-27B
MODEL_PATH_DEEPSEEK_V4=/mnt/data/models/DeepSeek-V4-Flash-0731

# vLLM 配置
VLLM_PORT=8000
MAX_MODEL_LEN=32768
```

### 启动 vLLM 模型服务

所有服务脚本位于 `scripts/hermes_eval/` 目录。

#### 启动 Qwen3.5-4B（单卡）

```bash
cd scripts/hermes_eval

# 启动服务（后台运行）
./serve_qwen3.5-4b.sh

# 等待服务就绪（约 3-8 分钟）
# 看到 "Uvicorn running on http://127.0.0.1:8000" 表示成功

# 验证服务
curl http://127.0.0.1:8000/v1/models
```

#### 启动 Qwen3.8-27B（双卡 TP=2）

```bash
cd scripts/hermes_eval

# 启动服务
./serve_qwen3.8-27b.sh

# 等待服务就绪（约 5-10 分钟）
```

#### 启动 DeepSeek-V4-Flash（双卡 TP=2, FP8）

```bash
cd scripts/hermes_eval

# 启动服务
./serve_deepseek-v4-flash.sh

# 等待服务就绪（约 15-20 分钟）
```

#### 停止服务

```bash
# 停止特定模型服务
./stop_servers.sh qwen3.5-4b
./stop_servers.sh qwen3.8-27b
./stop_servers.sh deepseek-v4-flash

# 停止所有服务
./stop_servers.sh all
```

### 运行评估

#### 评估 Qwen3.5-4B

```bash
cd scripts/hermes_eval

# 确保服务已启动
./serve_qwen3.5-4b.sh

# 运行评估（5000 条样本，32 个 worker 并发）
./run_eval_qwen3.5-4b.sh

# 评估完成后停止服务
./stop_servers.sh qwen3.5-4b
```

#### 评估 Qwen3.8-27B

```bash
cd scripts/hermes_eval

./serve_qwen3.8-27b.sh
./run_eval_qwen3.8-27b.sh
./stop_servers.sh qwen3.8-27b
```

#### 评估 DeepSeek-V4-Flash

```bash
cd scripts/hermes_eval

./serve_deepseek-v4-flash.sh
./run_eval_deepseek-v4-flash.sh
./stop_servers.sh deepseek-v4-flash
```

### 查看结果

评估结果保存在 `data/results/` 目录：

```bash
# 查看结果目录
ls -lh data/results/

# 查看指标汇总
cat data/results/hermes_qwen3.5-4b_5000/evaluation/metrics_summary.md

# 查看详细报告
cat data/results/hermes_qwen3.5-4b_5000/report.md

# 查看逐题明细
head data/results/hermes_qwen3.5-4b_5000/case_summary.csv
```

#### 指标说明

| 维度 | 指标 | 说明 |
|------|------|------|
| **Tool Use** | TS_P | 工具调用精确率 |
| | TS_R | 工具调用召回率 |
| | SC | 成功率 |
| **Planning** | DEC_P | 决策精确率 |
| | DEC_R | 决策召回率 |
| **Efficiency** | IT | 迭代次数（越少越好） |
| | OT | 总耗时（秒，越少越好） |
| **Instruction** | ID | 指令理解准确度 |
| | IE | 指令执行效率 |
| **Answer** | DR | 答案准确率 |
| | FPR | 误报率（越低越好） |

---

## 📚 详细使用指南

### 自定义评估参数

编辑 `scripts/hermes_eval/run_eval_*.sh` 文件中的参数：

```bash
# 修改 worker 数量（默认 32）
WORKERS=16

# 限制评估样本数（用于快速测试）
LIMIT=100

# 指定数据集路径
DATASET=data/datasets/pool_evaluable_20261028_sample5000.csv
```

### 断点续跑

如果评估中断，可以从中断处继续：

```bash
# 脚本会自动检测已完成的样本
# 只需重新运行相同的命令即可
./run_eval_qwen3.5-4b.sh
```

### 批量评估多个模型

```bash
#!/bin/bash
# batch_eval.sh

models=("qwen3.5-4b" "qwen3.8-27b" "deepseek-v4-flash")

for model in "${models[@]}"; do
  echo "=== Evaluating $model ==="
  
  # 启动服务
  ./serve_${model}.sh
  
  # 等待服务就绪
  sleep 300
  
  # 运行评估
  ./run_eval_${model}.sh
  
  # 停止服务
  ./stop_servers.sh ${model}
  
  echo "=== Completed $model ==="
done
```

### 使用自定义数据集

```bash
# 准备你的数据集
cp your_dataset.csv data/datasets/

# 修改评估脚本中的 DATASET 变量
sed -i 's|pool_evaluable_20261028_sample5000.csv|your_dataset.csv|' \
  scripts/hermes_eval/run_eval_qwen3.5-4b.sh

# 运行评估
./run_eval_qwen3.5-4b.sh
```

### 调整并发度

根据 GPU 显存和 CPU 核心数调整 worker 数量：

```bash
# 编辑 run_eval_*.sh
WORKERS=8   # 降低并发，减少内存压力
# 或
WORKERS=64  # 提高并发，加速评估
```

---

## 📈 评估指标详解

### 1. Tool Use（工具使用）

评估 Agent 选择和调用工具的准确性。

- **TS_P (Tool Selection Precision)**: 预测的工具调用中正确的比例
- **TS_R (Tool Selection Recall)**: 应该调用的工具中被正确预测的比例
- **SC (Success Rate)**: 工具调用成功的比例

### 2. Planning（规划能力）

评估 Agent 的决策质量。

- **DEC_P (Decision Precision)**: 正确决策的比例
- **DEC_R (Decision Recall)**: 应做决策中被正确识别的比例

### 3. Efficiency（效率）

评估 Agent 的执行效率。

- **IT (Iteration Count)**: 完成任务所需的迭代次数（越少越好）
- **OT (Overall Time)**: 总耗时，单位秒（越少越好）

### 4. Instruction（指令理解）

评估 Agent 对用户指令的理解和执行能力。

- **ID (Instruction Understanding)**: 指令理解的准确度（embedding 相似度）
- **IE (Instruction Execution)**: 指令执行的效率

### 5. Answer（答案质量）

评估最终答案的质量。

- **DR (Direct Response Accuracy)**: 直接回答的准确率
- **FPR (False Positive Rate)**: 误报率，越低越好

---

## 📁 项目结构

```
MB-with-agent-adapter/
├── configs/                    # 配置文件
│   ├── agent/                  # Agent 配置
│   ├── datasets/               # 数据集配置
│   ├── evaluation/             # 评估配置
│   ├── harness/                # Harness 配置
│   └── models/                 # 模型配置
├── data/                       # 数据目录
│   ├── datasets/               # 数据集文件
│   └── results/                # 评估结果
├── embedding-model/            # Embedding 模型权重
├── mcp_server/                 # MCP 沙箱服务器
│   ├── sandbox_cli.py
│   ├── sandbox_tools.py
│   └── shared_sandbox.py
├── scripts/                    # 脚本目录
│   ├── hermes_eval/            # Hermes 评估脚本
│   │   ├── serve_*.sh          # 启动模型服务
│   │   ├── run_eval_*.sh       # 运行评估
│   │   └── stop_servers.sh     # 停止服务
│   ├── fetch_embedding_weights.py
│   └── setup_hermes.sh
├── src/                        # 源代码
│   └── mobility_bench/         # MobilityBench 核心代码
│       ├── agent/              # Agent 框架
│       ├── dataset/            # 数据集加载器
│       ├── evaluation/         # 评估模块
│       ├── tools/              # 工具实现
│       └── reporting/          # 报告生成
├── tests/                      # 测试用例
├── .env.api.example            # API 配置示例
├── .env.h20.example            # H20 配置示例
├── pyproject.toml              # 项目依赖
└── README.md                   # 本文档
```

---

## ❓ 常见问题

### Q1: GPU 显存不足怎么办？

**A:** 
- 降低并发 worker 数量（修改 `WORKERS` 变量）
- 使用更小的模型（如 Qwen3.5-4B 而非 Qwen3.8-27B）
- 启用量化（FP8/INT8）

### Q2: 评估速度太慢？

**A:**
- 增加 worker 数量（确保显存足够）
- 使用 `LIMIT` 参数先测试少量样本
- 检查网络连接和 API 响应时间

### Q3: 如何查看实时进度？

**A:**
评估脚本会实时输出进度信息：
```
Progress: 1234/5000 (24.7%) | ETA: 45 min
```

也可以查看日志文件：
```bash
tail -f logs/eval_qwen3.5-4b.log
```

### Q4: 评估中断后如何恢复？

**A:**
直接重新运行评估脚本，会自动从上次中断处继续：
```bash
./run_eval_qwen3.5-4b.sh
```

### Q5: 如何自定义评估指标？

**A:**
编辑 `src/mobility_bench/evaluation/metrics/` 下的对应文件，添加新的指标计算逻辑。

### Q6: vLLM 服务启动失败？

**A:**
- 检查模型路径是否正确
- 确认 GPU 显存是否充足（`nvidia-smi`）
- 查看服务日志：`cat logs/serve_qwen3.5-4b.log`
- 确保端口 8000 未被占用

### Q7: 如何清理旧的评估结果？

**A:**
```bash
# 删除特定模型的评估结果
rm -rf data/results/hermes_qwen3.5-4b_*

# 删除所有评估结果
rm -rf data/results/*
```

---

## 📝 引用

如果您在研究中使用了 MobilityBench，请引用我们的论文：

```bibtex
@article{song2026mobilitybench,
  title={MobilityBench: A Benchmark for Evaluating Route-Planning Agents in Real-World Mobility Scenarios},
  author={Song, Zhiheng and Zhang, Jingshuai and Qin, Chuan and Wang, Chao and Chen, Chao and Xu, Longfei and Liu, Kaikui and Chu, Xiangxiang and Zhu, Hengshu},
  journal={arXiv preprint arXiv:2602.22638},
  year={2026},
  note={Accepted at KDD 2026 (Oral)}
}
```

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- 感谢高德地图（Amap）提供真实用户查询数据
- 感谢 Qwen、DeepSeek 团队开源高质量模型
- 感谢 vLLM 团队提供高效的推理引擎

---

## 📞 联系方式

如有问题或建议，欢迎：
- 提交 [GitHub Issue](https://github.com/LiuJW111/hermes_Mobilitybench/issues)
- 发送邮件至：[联系邮箱]

---

**Happy Evaluating! 🎉**
