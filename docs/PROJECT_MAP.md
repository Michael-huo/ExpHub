# ExpHub 目录结构与工程地图 (PROJECT_MAP.md)

> **文档定位**：本文档展示了 ExpHub 项目的物理文件树、各模块的具体职责，以及从顶层外壳到底层脚本的调用映射关系。AI 在进行跨文件重构或寻找特定功能逻辑时，请首选参考此地图。

## 1. 核心调度层 (`exphub/`)
该目录下的文件构成了 ExpHub 的“平台引擎”，负责全局调度、参数解析与环境穿透，**不包含具体的算法业务逻辑**。

| 文件路径 | 核心职责 | 关键数据/产物 |
|---|---|---|
| `exphub/__main__.py` | 框架统一入口，拦截 `python -m exphub` | 转发至 `cli.main()` |
| `exphub/cli.py` | 命令行参数解析、流水线模式 (mode) 分发、依赖校验 (doctor) | 统筹整个实验生命周期；统一通过 phase resolver 取得各阶段解释器 |
| `exphub/context.py` | 实验上下文中心，负责计算所有相对/绝对路径及命名契约 | 输出 `EXP_NAME` 及各阶段落盘路径 |
| `exphub/runner.py` | **跨环境执行器**。封装 `subprocess`，调用目标环境的绝对路径解释器拉起子进程 | 提供 `run_env_python`, `ros_exec`, `conda_exec` 与 phase interpreter 解析 |
| `exphub/config.py` | `datasets.json` 解析与校验 | `DatasetResolved` (含 bag/topic/intrinsics) |
| `exphub/meta.py` | 元信息辅助，实验参数序列化 | 读写 `step_meta.json` |
| `exphub/cleanup.py` | 中间文件清理策略 | 根据 `keep_level` 删除缓存 |

## 2. 算法实现层 (`scripts/`)
该目录下的脚本是具体的“干活者”，它们由 `exphub/` 跨环境拉起，负责具体的算法执行。

| 文件路径 | 对应 Pipeline 阶段 | 核心职责 |
|---|---|---|
| `scripts/_common.py` | **全局基建** (All) | **极其重要**：负责解析 `config/platform.yaml`；提供标准日志门面 (`log_info`, `[BAR]`) |
| `scripts/segment_make.py` | `segment` | `segment` 稳定入口，负责生成标准 segment 产物，并在 raw keyframes 之外额外写出 Wan `deploy_schedule.json` |
| `scripts/segment_analyze.py` | `segment` 研究旁路 | 读取既有 `segment/` 产物，输出正式三策略（`uniform / motion / semantic`）的逐帧 kinematics/allocation/projection 分析图表，并内建 active policy + passive observer 横向对比（summary + compare 图，不改正式 keyframes） |
| `scripts/_segment/` | `segment` (内部实现) | `api/extract/materialize/policies/research` 内聚 `segment` 的主链路与研究旁路逻辑；当前 `policies/` 只承载正式策略 `uniform / motion / semantic`，并由 `uniform.py / motion.py / semantic.py` 三个正式实现入口对外服务；研究侧 `research/` 负责共享信号与可视化支撑 |
| `scripts/prompt_gen.py` | `prompt` | `prompt` 前端入口；负责解析 CLI、按 clip 采样图像、调用具体 backend 生成 prompt，并写回 `clip_prompts.json / manifest.json / step_meta.json` |
| `scripts/_prompt/` | `prompt` (内部实现) | 内聚 prompt backend 抽象、采样策略与具体 VLM 实现；当前包含 `qwen` 与 `smolvlm2` 两个 backend |
| `scripts/infer_i2v.py` | `infer` (外壳) | 读取 execution manifest / deploy schedule，生成逐段运行计划并分发 Wan 推理 |
| `scripts/_infer_i2v_impl.py`| `infer` (内核) | Wan2.2 实际的 float8 量化与张量推理逻辑；按逐段 `start_idx / end_idx / num_frames` 执行 |
| `scripts/merge_seq.py` | `merge` | 基于 `runs_plan.json` 的真实逐段边界做时间轴对齐、冗余去重与合并 |
| `scripts/slam_droid.py` | `slam` | 调用 DROID-SLAM 提取生成轨道或原始轨道的位姿 |
| `scripts/stats_collect.py` | `stats` | 扫描各阶段 `step_meta.json`，汇总全链路性能数据 |

## 3. 配置文件 (`config/`)
平台配置中枢，实现代码与环境的彻底解耦。

| 文件路径 | 核心职责 | 备注 |
|---|---|---|
| `config/platform.yaml` | **外部依赖注册表** | 配置各跨域 Python 解释器（通过 `environments.phases.<phase>.python` 组织；`prompt` 可额外声明 `prompt_smol`）、模型路径、算法源码路径 |
| `config/datasets.json` | **数据源注册表** | 定义实验可用的数据集 (如 ROS bag 路径、内参矩阵) |
| `config/prompt_manifest.json`| **提示词模板库** | 提供基础 prompt 与负向 prompt 的预设配置 |

## 4. 动态调用映射网格 (Invocation Map)
展示 `python -m exphub --mode <X>` 是如何一步步穿透到最底层的算法内核的：

| 模式 (Mode) | 调度链条 (Call Chain) |
|---|---|
| `segment` | `cli.py` -> resolve phase `segment` -> `runner.ros_exec` -> `segment_make.py` |
| `prompt` | `cli.py` -> resolve phase `prompt` or `prompt_smol` -> `runner.run_env_python` -> `prompt_gen.py` -> `_prompt.api` -> selected backend |
| `infer` | `cli.py` -> resolve phase `infer` -> `runner.run_env_python` -> `infer_i2v.py` -> `sys.executable` -> `_infer_i2v_impl.py` |
| `merge` | `cli.py` -> resolve phase `infer` -> `runner.run_env_python` -> `merge_seq.py` |
| `slam` | `cli.py` -> resolve phase `slam` -> `runner.run_env_python` -> `slam_droid.py` |
| `eval` | `cli.py` -> resolve phase `slam` -> `runner.run_env_python` -> `evo_traj` / `evo_ape` (外部二进制工具) |
| `stats` | `cli.py` -> resolve phase `prompt` -> `runner.run_env_python` -> `stats_collect.py` |
| `all` | 依次自动串行执行上述 1~7 步；其中 `segment` 完成后会立即尝试触发 `segment_analyze.py`，各阶段解释器统一来自 phase resolver |
