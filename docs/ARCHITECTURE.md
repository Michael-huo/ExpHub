# ExpHub 系统架构

> 本文回答什么问题：ExpHub 由哪些层组成、阶段如何被调度、实验目录如何组织、哪些边界不能被打破。

更多阶段级输入输出请看 [PIPELINE_CONTRACT.md](./PIPELINE_CONTRACT.md)，日志规则请看 [LOGGING.md](./LOGGING.md)；若需理解当前工程模块如何映射到论文方法论，请看 [TITS_METHODOLOGY.md](./TITS_METHODOLOGY.md)。

## 1. 平台分层

ExpHub 的核心原则是“平台调度层”和“业务脚本层”分离。

| 层级 | 位置 | 主要职责 |
|---|---|---|
| 平台入口层 | `exphub/__main__.py`, `exphub/cli.py` | 解析命令、选择 mode、组织 `segment -> ... -> stats` 调度 |
| 运行器层 | `exphub/runner.py` | 按 phase 选择解释器，拉起子进程，收口日志 |
| 实验上下文层 | `exphub/context.py`, `exphub/meta.py`, `exphub/cleanup.py` | 计算 `EXP_DIR`、统一命名、记录元数据、清理中间产物 |
| 业务脚本层 | `scripts/*.py`, `scripts/_*/` | 执行各阶段算法逻辑 |
| 配置层 | `config/platform.yaml`, `config/datasets.json` | 注册 phase Python、外部 repo、模型与数据集 |

平台层不直接实现算法；它的职责是把“在哪个环境执行哪段脚本”这件事做稳定。

## 2. 执行入口与 phase 调度

统一入口是：

`python -m exphub`

典型调用链为：

`exphub/__main__.py -> exphub/cli.py -> exphub/runner.py -> scripts/*.py`

### 2.1 phase Python 解析

所有跨环境解释器都从 `config/platform.yaml` 的 `environments.phases.<phase>.python` 读取。

当前主链路相关的 phase 包括：

- `segment`
- `prompt`
- `prompt_smol`
- `infer`
- `infer_fun_5b`
- `slam`

当前默认 phase 选择逻辑是：

- `prompt_backend=smolvlm2` 时使用 `prompt_smol`
- `prompt_backend=qwen` 时使用 `prompt`
- `infer_backend=wan_fun_5b_inp` 时使用 `infer_fun_5b`
- `infer_backend=wan_fun_a14b_inp` 时使用 `infer`
- `merge` 复用当前 infer phase
- `eval` 使用 `slam` phase 解释器调度 `scripts/eval_main.py`，前端壳负责调度 `_eval/` 后端，统一封装轨迹与图像评测
- `stats` 由 `prompt` phase 执行

### 2.2 跨环境执行规则

- 顶层调度不拼接 `conda activate`
- `runner.run_env_python()` 直接使用 phase 对应的绝对 Python 路径
- `segment` 仍通过 ROS 入口执行，但 phase 解析同样来自统一配置
- 底层若继续拉起子进程或多卡 worker，必须继承当前解释器，而不是依赖系统 `PATH`

## 3. 标准主链路

主链路固定为：

`segment -> prompt -> infer -> merge -> slam -> eval -> stats`

各阶段的系统职责如下。

| 阶段 | 主要脚本 | 系统职责 |
|---|---|---|
| `segment` | `scripts/segment_make.py` | 读取原始数据，按正式 `uniform | state` 口径输出标准帧序列、raw keyframes、deploy schedule 与研究产物 |
| `prompt` | `scripts/prompt_gen.py` | 从 `segment/frames/` 抽代表帧，基于 `segment/state_segmentation/state_segments.json` 与 `segment/deploy_schedule.json` 生成 `final_prompt`、state prompt 产物与聚合 `report.json` |
| `infer` | `scripts/infer_i2v.py` | 读取 prompt 与执行计划，路由到具体 Wan backend |
| `merge` | `scripts/merge_seq.py` | 按 `runs_plan.json` 的真实边界合并生成结果 |
| `slam` | `scripts/slam_droid.py` | 在 `ori` 或 `gen` 轨道上估计位姿 |
| `eval` | `scripts/eval_main.py` | 调度 `_eval/` 后端，对 `ori/gen` 轨迹做 APE/RPE 评估，并补充 merge-vs-ori 的图像指标与两视图几何型 SLAM-friendly 指标评估 |
| `stats` | `scripts/stats_collect.py` | 汇总 `segment/merge step_meta.json`、`prompt/infer report.json` 与日志，生成最终统计 |

`segment` 之后默认还会触发一次 analyze 旁路：

- 它是研究旁路，不属于主链路强依赖
- 默认在 `--mode segment` 和 `--mode all` 后执行
- 默认只刷新 `segment/signal_extraction/` 与 `segment/state_segmentation/` 下的聚合研究产物，不再独立扩张 `segment/analysis/`
- 它只刷新旁路研究产物，不回写 `segment/keyframes/keyframes_meta.json` 或 `segment/deploy_schedule.json`
- 失败时只报 `WARN`，不阻断 `prompt / infer / merge / slam`

## 4. 时间计划与 prompt 语义边界

当前系统把“时间计划”和“prompt 文本”明确拆开。

### 4.1 时间计划三层

- raw schedule：`segment/keyframes/keyframes_meta.json`
- deploy schedule：`segment/deploy_schedule.json`
- runtime execution plan：`infer` 前端临时文件，不作为默认落盘产物保留
- execution facts：`infer/runs_plan.json`

当前默认行为是：

- `segment` 产出 raw keyframes，并投影出 `wan_r4` deploy schedule
- `segment/keyframes/keyframes_meta.json` 是 raw keyframe 事实源
- `segment/deploy_schedule.json` 是执行投影，不回写覆盖 raw schedule
- `infer` 优先从 `deploy_schedule.json` 派生 execution segments
- 如果 deploy schedule 缺失，`infer` 才回退到 `legacy_kf_gap` slicing
- `merge` 只信任 `runs_plan.json` 中的真实 `start_idx / end_idx`

### 4.2 prompt 边界

当前 prompt 主链路已经收敛为：

- `prompt/final_prompt.json`
- `prompt/state_prompt_manifest.json`
- `prompt/deploy_to_state_prompt_map.json`
- `prompt/report.json`

`prompt/state_prompt_manifest.json` 的语义单位以 `segment/state_segmentation/state_segments.json` 为准，`prompt/deploy_to_state_prompt_map.json` 只负责把 deploy execution segments 对齐到 state prompt。`infer` 仍以 `final_prompt.json` 里的全局 `prompt / negative_prompt` 作为 base scene prompt，但当前会在前端把它与 `state_prompt_manifest.json` / `deploy_to_state_prompt_map.json` 派生为 `infer/prompt_manifest_resolved.json`。runtime 继续只理解“prompt manifest + per-segment override”，不直接理解 state 文件本身。

专题细节见 [PROMPT_PROFILE_SYSTEM.md](./PROMPT_PROFILE_SYSTEM.md)。

## 5. 实验上下文与目录布局

每个实验运行在独立 `EXP_DIR`：

`{tag}_{w}x{h}_t{start}s_dur{dur}s_fps{fps}_gap{kf_gap}`

目录按阶段隔离，常见布局如下：

- `segment/`：`frames/`、`keyframes/`、`keyframes_meta.json`、`deploy_schedule.json`
- `segment/` 还会收敛正式研究产物 `signal_extraction/` 与 `state_segmentation/`
- `prompt/`：`final_prompt.json`、`state_prompt_manifest.json`、`deploy_to_state_prompt_map.json`、`report.json`
- `infer/`：`prompt_manifest_resolved.json`、`runs/`、`runs_plan.json`、`report.json`
- `merge/`：`frames/`、`timestamps.txt`、`calib.txt`、`step_meta.json`
- `slam/`：`ori/`、`gen/` 轨迹与运行元数据
- `eval/`：`report.json`、`details.csv`、`plots/traj_xy.png`、`plots/metrics_overview.png` 以及必要失败摘要
- `stats/`：`report.json`、`compression.json`
- `logs/`：各阶段完整日志

## 6. 不能被打破的架构边界

- 下游阶段只能读取上游产物，不能回写上游目录
- 当前实验目录允许覆盖式重跑，但产物路径约定不能随意改名
- 缺失关键输入时必须 fail fast，而不是自动脑补
- 文档和代码必须一致；如果默认 backend、phase 或产物变了，文档需要同步

## 7. Doctor 与运行方式

`python -m exphub --mode doctor ...` 只做安全扫描，不创建实验目录。

当前 `doctor` 至少检查：

- `segment`
- `prompt`
- 选中的 infer phase
- `slam`
- 当默认 prompt backend 为 `smolvlm2` 时，额外检查 `prompt_smol`

输出格式为：

`DOCTOR phase=<name> python=<path> exists=<bool>`

任一关键 phase 缺失或不可执行时，结果应为 `FAIL`。
