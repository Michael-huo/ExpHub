# ExpHub 系统架构

> 本文回答什么问题：正式实现放在哪里、阶段如何被调度、哪些边界不能再被打破。

更多阶段级输入输出请看 [PIPELINE_CONTRACT.md](./PIPELINE_CONTRACT.md)，日志规则请看 [LOGGING.md](./LOGGING.md)。

## 1. 平台分层

当前正式实现已经收口到 `exphub/`：

| 层级 | 位置 | 主要职责 |
|---|---|---|
| 平台入口层 | `exphub/__main__.py`, `exphub/cli.py` | 解析命令、选择 mode、打印实验摘要 |
| 编排层 | `exphub/pipeline/orchestrator.py` | 串起 `segment -> ... -> stats`，控制 step 生命周期 |
| 运行器层 | `exphub/common/subprocess.py`, `exphub/runner.py` | 解析 phase Python、拉起子进程、收口日志 |
| 契约与路径层 | `exphub/contracts/`, `exphub/common/paths.py` | 统一阶段产物、命名与路径访问 |
| 正式阶段实现层 | `exphub/pipeline/<stage>/service.py` | 执行各阶段正式主链逻辑 |
| 配置层 | `config/platform.yaml`, `config/datasets.json` | 注册 phase Python、模型、外部仓库与数据集 |

`tools/` 只保留非正式主链工具。旧 `scripts/_*` 实现树不再属于当前正式架构。

## 2. 执行入口与调度

统一入口是：

`python -m exphub`

当前标准调用链是：

`exphub/__main__.py -> exphub/cli.py -> exphub/pipeline/orchestrator.py -> exphub/pipeline/<stage>/service.py`

### 2.1 phase Python 解析

所有跨环境解释器都从 `config/platform.yaml` 的 `environments.phases.<phase>.python` 读取。

当前主链常用 phase 包括：

- `segment`
- `prompt`
- `prompt_smol`
- `infer`
- `infer_fun_5b`
- `slam`

选择逻辑保持：

- `prompt_backend=smolvlm2` 时使用 `prompt_smol`
- `prompt_backend=qwen` 时使用 `prompt`
- `infer_backend=wan_fun_5b_inp` 时使用 `infer_fun_5b`
- `infer_backend=wan_fun_a14b_inp` 时使用 `infer`
- `merge` 复用当前 infer phase
- `eval` 使用 `slam` phase
- `stats` 使用 `prompt` phase

### 2.2 跨环境规则

- 顶层调度不拼接 `conda activate`
- `run_env_python()` 直接使用 phase 对应的绝对 Python 路径
- 底层继续拉起子进程时必须继承当前解释器，而不是依赖系统 `PATH`

## 3. 标准主链路

主链路固定为：

`segment -> prompt -> infer -> merge -> slam -> eval -> stats`

| 阶段 | 正式实现 | 系统职责 |
|---|---|---|
| `segment` | `exphub/pipeline/segment/service.py` | 生成标准帧序列、raw keyframes、deploy schedule 与 state 产物 |
| `prompt` | `exphub/pipeline/prompt/service.py` | 生成 `base_prompt`、state prompt manifest 与 runtime prompt plan |
| `infer` | `exphub/pipeline/infer/service.py` | 校验执行边界与 prompt plan，对接 Wan backend |
| `merge` | `exphub/pipeline/merge/service.py` | 按 `runs_plan.json` 的真实边界合并结果 |
| `slam` | `exphub/pipeline/slam/service.py` | 在 `ori` / `gen` 轨道上估计位姿 |
| `eval` | `exphub/pipeline/eval/service.py` | 汇总轨迹、图像与 slam-friendly 评测 |
| `stats` | `exphub/pipeline/stats/service.py` | 汇总阶段报告、压缩统计与实验摘要 |

## 4. 目录与产物边界

每个实验运行在独立 `EXP_DIR`，目录按阶段隔离：

- `segment/`
- `prompt/`
- `infer/`
- `merge/`
- `slam/`
- `eval/`
- `stats/`
- `logs/`

正式边界必须保持：

- 下游阶段只能消费上游结果，不能回写上游目录
- `segment/keyframes/keyframes_meta.json` 仍是 raw keyframe 事实源
- `segment/deploy_schedule.json` 仍是执行投影，不回写 raw schedule
- `prompt/runtime_prompt_plan.json` 是 `infer` 的唯一正式 prompt 输入
- `merge` 只信任 `infer/runs_plan.json`

## 5. 当前不能再回退的架构共识

- 正式实现归 `exphub/common`、`exphub/contracts`、`exphub/pipeline`
- 旧实现树不再与正式主链并存
- 文档口径必须与当前正式结构一致
- 缺失关键输入时必须 fail fast，而不是自动脑补
