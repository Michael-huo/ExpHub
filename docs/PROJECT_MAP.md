# ExpHub 目录结构与工程地图 (PROJECT_MAP.md)

> **文档定位**：本文档展示了 ExpHub 项目的物理文件树、各模块的具体职责，以及从顶层外壳到底层脚本的调用映射关系。AI 在进行跨文件重构或寻找特定功能逻辑时，请首选参考此地图。

## 1. 核心调度层 (`exphub/`)
该目录下的文件构成了 ExpHub 的“平台引擎”，负责全局调度、参数解析与环境穿透，**不包含具体的算法业务逻辑**。

| 文件路径 | 核心职责 | 关键数据/产物 |
|---|---|---|
| `exphub/__main__.py` | 框架统一入口，拦截 `python -m exphub` | 转发至 `cli.main()` |
| `exphub/cli.py` | 命令行参数解析、流水线模式 (mode) 分发、依赖校验 (doctor) | 统筹整个实验生命周期 |
| `exphub/context.py` | 实验上下文中心，负责计算所有相对/绝对路径及命名契约 | 输出 `EXP_NAME` 及各阶段落盘路径 |
| `exphub/runner.py` | **跨环境执行器**。封装 `subprocess`，调用目标环境的绝对路径解释器拉起子进程 | 提供 `run_env_python`, `ros_exec`, `conda_exec` |
| `exphub/config.py` | `datasets.json` 解析与校验 | `DatasetResolved` (含 bag/topic/intrinsics) |
| `exphub/meta.py` | 元信息辅助，实验参数序列化 | 读写 `step_meta.json` |
| `exphub/cleanup.py` | 中间文件清理策略 | 根据 `keep_level` 删除缓存 |

## 2. 算法实现层 (`scripts/`)
该目录下的脚本是具体的“干活者”，它们由 `exphub/` 跨环境拉起，负责具体的算法执行。

| 文件路径 | 对应 Pipeline 阶段 | 核心职责 |
|---|---|---|
| `scripts/_common.py` | **全局基建** (All) | **极其重要**：负责解析 `config/platform.yaml`；提供标准日志门面 (`log_info`, `[BAR]`) |
| `scripts/segment_make.py` | `segment` | `segment` 稳定入口，负责生成标准 segment 产物 |
| `scripts/segment_analyze.py` | `segment` 研究旁路 | 读取既有 `segment/` 产物，输出逐帧非语义信号、分数与可视化 |
| `scripts/_segment/` | `segment` (内部实现) | `api/extract/materialize/policies/research` 内聚 `segment` 的主链路与研究旁路逻辑 |
| `scripts/prompt_gen.py` | `prompt` | 调用 VLM 生成 `manifest.json` |
| `scripts/infer_i2v.py` | `infer` (外壳) | i2v 任务规划、多进程分发与合并调度 |
| `scripts/_infer_i2v_impl.py`| `infer` (内核) | Wan2.2 实际的 float8 量化与张量推理逻辑 |
| `scripts/merge_seq.py` | `merge` | 时间轴对齐、冗余去重、合并长视频/图像序列 |
| `scripts/slam_droid.py` | `slam` | 调用 DROID-SLAM 提取生成轨道或原始轨道的位姿 |
| `scripts/stats_collect.py` | `stats` | 扫描各阶段 `step_meta.json`，汇总全链路性能数据 |

## 3. 配置文件 (`config/`)
平台配置中枢，实现代码与环境的彻底解耦。

| 文件路径 | 核心职责 | 备注 |
|---|---|---|
| `config/platform.yaml` | **外部依赖注册表** | 配置各跨域 Python 解释器、模型路径、算法源码路径 |
| `config/datasets.json` | **数据源注册表** | 定义实验可用的数据集 (如 ROS bag 路径、内参矩阵) |
| `config/prompt_manifest.json`| **提示词模板库** | 提供基础 prompt 与负向 prompt 的预设配置 |

## 4. 动态调用映射网格 (Invocation Map)
展示 `python -m exphub --mode <X>` 是如何一步步穿透到最底层的算法内核的：

| 模式 (Mode) | 调度链条 (Call Chain) |
|---|---|
| `segment` | `cli.py` -> `runner.ros_exec` -> `segment_make.py` |
| `prompt` | `cli.py` -> `runner.run_env_python` -> `prompt_gen.py` |
| `infer` | `cli.py` -> `runner.run_env_python` -> `infer_i2v.py` -> `sys.executable` -> `_infer_i2v_impl.py` |
| `merge` | `cli.py` -> `runner.run_env_python` -> `merge_seq.py` |
| `slam` | `cli.py` -> `runner.run_env_python` -> `slam_droid.py` |
| `eval` | `cli.py` -> `runner.run_env_python` -> `evo_traj` / `evo_ape` (外部二进制工具) |
| `stats` | `cli.py` -> `runner.run_env_python` -> `stats_collect.py` |
| `all` | 依次自动串行执行上述 1~7 步 |
