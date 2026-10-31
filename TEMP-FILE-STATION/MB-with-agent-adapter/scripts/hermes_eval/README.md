# Hermes × MobilityBench 横评脚本（H20 本机）

本目录只负责 **Hermes Agent** 一侧的评测：拉起三个本地 vLLM 服务，并各自跑完
`data/datasets/pool_evaluable_20261028_sample5000.csv` 的 5000 条题目。

## 0. 上下文窗口：按组内 32K 基线

三个服务严格按提供的启动配置，统一 `--max-model-len 32768`：

| 模型 | 启动配置 |
|---|---|
| Qwen3.5-4B | 单卡 TP=1，dtype bf16，gpu-mem 0.7，`qwen3_xml` |
| Qwen3.8-27B | 双卡 TP=2，dtype bf16，gpu-mem 0.95，`qwen3_xml` |
| DeepSeek-V4-Flash | 双卡 TP=2，gpu-mem 0.95，`deepseek_v4` |

三个服务都关闭思考：`--default-chat-template-kwargs '{"enable_thinking": false}'`。
Hermes 的工具调用步数预算由 `configs/harness/default.yaml` 的 `max_steps` 控制，
当前为 **20 轮**；预算耗尽时 Hermes 会再请求一次总结，但若仍未正常结束，该题会被
记为异常。

> **需要知道的差异**：Hermes 源码硬性要求 `context_length >= 64000`
> （`agent/model_metadata.py` 的 `MINIMUM_CONTEXT_LENGTH`，低于即抛
> `ValueError`），所以 Hermes 每次运行的 `config.yaml` 仍写 65536；但实际服务
> 窗口是 32768。短会话不受影响；如果某个 case 的提示词加输出真的超过 32768，
> vLLM 会返回上下文超限错误，该题会被记录为失败。这是 32K 基线下运行 Hermes
> 的已知取舍，不是脚本错误。要消除它只能让服务真实提供 ≥64000 窗口，并让组内
> 其他 Agent 同步调整。


## 1. 本机实际路径（已修正旧文档/命令）

| 旧写法 | 本机实际 |
|---|---|
| `/opt/vllm-env/bin/vllm` | `/usr/local/bin/vllm`（0.31.0） |
| `/opt/models/DeepSeek-V4-Flash` | `/mnt/data/models/DeepSeek-V4-Flash-0731` |
| `/opt/models/Qwen3.5-4B` | `/mnt/data/models/Qwen3.5-4B` |
| `/opt/models/Qwen3.8-27B` | `/mnt/data/models/Qwen3.8-27B` |
| `D:/work/...`、`/root/mb/...` | `/mnt/workspace/TEMP-FILE-STATION/...` |

所有路径集中在 `env.sh`，可用同名环境变量覆盖，不需要改脚本正文。

## 2. 一次性安装环境

仓库自带的 `.venv` 与 `hermes-agent/venv` 都是 Windows 环境，Linux 上不可用。
`setup_env.sh` 会新建 `*.venv-linux`（复用系统已装的 torch/transformers，避免重复下载
数 GB），并安装 MobilityBench、Hermes 依赖和 `mcp<2`（沙箱代码依赖
`mcp.server.fastmcp`，系统全局是 mcp 2.x）。

```bash
cd /mnt/workspace/TEMP-FILE-STATION/MB-with-agent-adapter
bash scripts/hermes_eval/setup_env.sh
```

Hermes 依赖优先按 `configs/hermes-requirements.lock` 哈希安装；镜像缺版本时自动回退
到 `pip install -e hermes-agent[mcp]`。安装默认走中科大 PyPI 镜像
（`https://mirrors.ustc.edu.cn/pypi/simple`），可用 `PIP_INDEX_URL=...` 覆盖。

可选：装完跑一次真实链路集成测试（不请求模型）：

```bash
cd /mnt/workspace/TEMP-FILE-STATION/MB-with-agent-adapter
HERMES_SRC=/mnt/workspace/TEMP-FILE-STATION/hermes-agent \
HERMES_PY=/mnt/workspace/TEMP-FILE-STATION/hermes-agent/.venv-linux/bin/python \
.venv-linux/bin/python -m pytest tests/test_hermes_source_integration.py -q
```

## 3. 评测用的 embedding 权重

`embedding-model/model.safetensors`（Qwen3-Embedding-0.6B，1.19GB）缺失时，原生评分器
会退化为字符相似度，`instruction` / `answer` 指标失真。补齐：

```bash
cd /mnt/workspace/TEMP-FILE-STATION/MB-with-agent-adapter
.venv-linux/bin/python scripts/fetch_embedding_weights.py
```

脚本会校验固定 SHA-256，优先 HuggingFace、失败转 ModelScope。

## 4. 逐个模型跑

同一时间只能有一个服务占用 8000 端口（2 张 H20）。三个模型**串行**跑。
每个 `run_eval_*.sh` 已内置 `WORKERS=32`、`LIMIT=5000`，服务就绪后直接运行即可，
不需要额外环境变量：

```bash
cd /mnt/workspace/TEMP-FILE-STATION/MB-with-agent-adapter/scripts/hermes_eval

# ① Qwen3.5-4B：单卡，约 3–8 分钟起服务
./serve_qwen3.5-4b.sh
./run_eval_qwen3.5-4b.sh
./stop_servers.sh qwen3.5-4b

# ② Qwen3.8-27B：双卡 TP=2
./serve_qwen3.8-27b.sh
./run_eval_qwen3.8-27b.sh
./stop_servers.sh qwen3.8-27b

# ③ DeepSeek-V4-Flash：双卡 TP=2，FP8，起服务约 15–20 分钟
./serve_deepseek-v4-flash.sh
./run_eval_deepseek-v4-flash.sh
./stop_servers.sh deepseek-v4-flash
```

`serve_*.sh` 会后台启动 vLLM、写 pid 和日志，把**启动日志实时打到终端**，并一直等到
`/v1/models` 出现目标模型（默认**无超时**，大模型启动慢也不会被杀）。准备就绪后脚本
返回，服务继续在后台运行。建议放在 tmux 里执行。

### 冒烟测试（先跑 3 条再全量）

```bash
LIMIT=3 OUT=/mnt/workspace/TEMP-FILE-STATION/MB-with-agent-adapter/data/results/smoke_qwen35_4b \
  ./run_eval_qwen3.5-4b.sh
```

3 条也会触发五类原生评分，用于确认：模型确实调用了沙箱工具、工具返回非空、最终
答复能映射进 `plan_result`。

### 断点续跑

```bash
RESUME=1 ./run_eval_qwen3.5-4b.sh
```

`run_hermes_batch.py` 会校验模型、endpoint、数据集 SHA-256 和
`src/`、`configs/`、`mcp_server/` 的运行时哈希；这些内容变了就拒绝混跑。
只想重算评分、不再请求模型：

```bash
EVALUATE_ONLY=1 ./run_eval_qwen3.5-4b.sh
```

### 常用环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `LIMIT` | `5000` | 取数据集前 N 条 |
| `WORKERS` | `32` | 并发 Hermes 进程，可选 1/2/4/8/16/32/48/64/96 |
| `OUT` | `data/results/hermes_<model>_5000_s<步数>` | 结果目录（当前为 `_s20`） |
| `RESUME` | `0` | `1` 续跑已有目录 |
| `EVALUATE_ONLY` | `0` | `1` 只重算评分 |
| `FORCE` | `0` | `1` 跳过“GPU 已被占用”检查 |
| `WAIT_TIMEOUT` | `0` | 等服务就绪的秒数；0 表示无限等待 |
| `SERVE_PORT` | `8000` | vLLM 端口（`LLM_BASE_URL` 会跟随） |

## 5. 结果目录

```
data/results/hermes_<model>_5000/
├── metadata.json          模型、endpoint、harness 参数、数据集/运行时哈希
├── dataset.csv            本次冻结的子集（各 Agent 对比时用同一份）
├── completion.json        是否跑完 + 评分是否完成
├── progress.json          实时进度
├── selected_results.json  每题最终选用的结果
├── results/<model>/raw_results.json
├── traces/<case_id>/...   Hermes 原生 session、driver stdout/stderr（已去密钥）
├── evaluation/            tool / planning / efficiency / instruction / answer
│   ├── metrics_summary.csv/md  每个子项的均值 + 有效题数（推荐看这个）
│   └── <model>/*_evaluation.csv 逐题子项
├── evaluation_backend.json 是否使用 embedding 权重或字符回退
├── sandbox.log            共享沙箱服务日志
└── terminal-summary.txt   终端汇总
```

评分口径保持仓库原生实现不变。五类指标的子项：

| 指标 | 子项 | 含义（越大越好） |
|---|---|---|
| tool | TS_P / TS_R / SC | 工具选择覆盖 / 非冗余 / schema 正确 |
| planning | DEC_P / DEC_R | 规划分解精确率 / 召回率 |
| efficiency | IT / OT | 输入 / 输出 token 数（不是 0–1 分数） |
| instruction | ID / IE | 意图识别 / 信息槽位抽取 |
| answer | DR / FPR | 交付成功率 / 最终通过率（Final Pass Rate） |

> 原生 `MetricResult.score` 和汇总 `average_score` 在五个 Metric 里都固定写成 `0.0`，
> 是仓库占位值，**不代表任务失败**。脚本额外生成 `evaluation/metrics_summary.csv|md`，
> 把所有子项和 `valid_cases/total_cases` 完整列出；判断效果请用子项。
> `successful_cases` 只表示“该指标算出来了”，不代表任务正确。
>
> 异常题（`error` 非空）同样保留 `reporter_output`、`worker_output` 和 `token_usage`，
> `planner_output` 里另有 `api_calls`、`budget_exhausted`、`turn_exit_reason`。
> `error` 只表示执行状态；原生评估器会按已有数据继续评分，不会把整条结果丢掉。

## 6. 已知注意事项

1. **32K 服务 + Hermes 64K 声明**：Hermes 源码拒绝 `context_length < 64000`，所以
   适配器写的 `config.yaml` 仍是 65536，而服务只给 32768。任务提示词较长时可能
   触发 vLLM 上下文超限，该题记失败。按组内统一 32K 的要求保留此取舍。
2. `Qwen3.5-4B/` 下还有一组未被索引引用的旧权重
   `model.safetensors-0000{1,2}-of-00002`（约 9.3GB）。索引实际指向
   `model-0000x-of-00003`，不影响加载；确认无用后可自行删除。
3. DeepSeek-V4-Flash 在 0.95 显存下若启动 OOM，优先把
   `--gpu-memory-utilization` 降到 0.92，或加 `--max-num-seqs 8`。
4. 并发上限已放开到 96（`run_hermes_batch.py --workers`）。每个 worker 是一个独立
   Hermes 进程（约 300–500MB 常驻），沙箱数据仍只有共享 MCP 服务的一份；本机
   128GB 内存下 96 并发可行，但 vLLM 侧吞吐与延迟会明显变化。建议先 `WORKERS=8`
   测单题耗时，再按模型梯度加（16→32→48→96），并同步观察
   `nvidia-smi` 与 `logs/hermes_eval/serve_*.log`。高并发时 DeepSeek-V4-Flash
   最容易先到瓶颈，必要时给 vLLM 加 `--max-num-seqs` 限制排队。
5. `run_hermes_batch.py` 接受 `WORKERS` 1/2/4/8/16/32/48/64/96；再往上需要同时
   评估进程内存与 vLLM 并发预算，不属于本目录的范围。
