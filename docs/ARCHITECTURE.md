# ExpHub 系统架构说明 (ARCHITECTURE.md)

> **文档定位**：本文档是 ExpHub 平台的核心架构蓝图，定义了系统的顶层调度机制、主链路生命周期以及数据流转规范。关于具体的脚本输入输出参数，请参阅 `SCRIPTS_AUDIT.md`；关于日志 UI 规范，请参阅 `LOGGING.md`。

## 1. 核心调度架构：平台化与跨环境穿透
ExpHub 的核心设计理念是**“调度外壳与算法核心的分离”**。平台自身不绑定任何算法环境，而是作为一个智能路由器，根据配置动态拉起不同的运行环境。

- **唯一入口**：`python -m exphub`
- **执行链路**：`exphub/__main__.py` -> `exphub/cli.py` -> `exphub/runner.py` -> `scripts/*.py`
- **极致解耦 (Platform Decoupling)**：
  - 平台通过 `scripts/_common.py` 读取 `config/platform.yaml` 获取所有外部依赖（包含 Conda 环境的 Python 解释器绝对路径、算法源码 Repos、模型权重 Models）。
  - **跨环境穿透**：`exphub/runner.py` 不使用传统的 `bash -lc "conda activate ..."`。它直接读取目标环境的绝对解释器路径（如 `prompt_python`, `infer_python`），并使用 `subprocess` 拉起对应的底层脚本。
  - **子进程继承**：底层脚本（如 `infer_i2v.py`）若需拉起更底层的多卡任务（如 `torchrun`），必须使用 `sys.executable`，以确保继承当前被激活的绝对路径，彻底免疫系统的 `PATH` 环境变量污染。

## 2. 主链路生命周期 (Main Pipeline)
ExpHub 支持通过 `--mode all` 一键贯穿，或通过具体 mode 单步执行。主链路严格遵循以下 7 大阶段的单向数据流转：

1. **`segment` (切片锚定)**：读取原始数据集，按时间戳抽取关键帧，构建时间网格。
2. **`prompt` (提示词生成)**：调用 VLM (如 Qwen2-VL) 对关键帧进行场景理解，生成基础与增量 Prompt。
3. **`infer` (视频生成推理)**：调用 I2V 模型 (如 VideoX-Fun/Wan) 进行两帧之间的插帧推理，生成视频片段。
4. **`merge` (序列合并)**：将分散的视频片段在时间轴上对齐、去重，并融合成连贯的长序列图像与视频。
5. **`slam` (位姿估计)**：将合成的图像序列送入 VSLAM 算法（如 DROID-SLAM）计算相机轨迹。
6. **`eval` (轨迹评估)**：使用 `evo` 工具比对估计轨迹与真实轨迹 (Ground Truth)，计算 APE/RPE 等指标。
7. **`stats` (统计出图)**：收集上述各阶段的 `step_meta.json` 与运行日志，生成最终的 `EXPERIMENT PERFORMANCE PROFILING` 面板与数据报表。

## 3. 产物与目录规范 (Artifacts & Workspace)
每个实验都会在一个专属的 `EXP_DIR` 下运行，命名契约由 `ExperimentContext` 统一管理（格式：`{tag}_{w}x{h}_t{start}s_dur{dur}s_fps{fps}_gap{kf_gap}`）。

目录结构严格按照执行阶段划分，实现物理隔离：
- `segment/`：存放抽取的关键帧、缩略图预览及分割元数据。
- `prompt/`：存放 `manifest.json` 及提示词相关产物。
- `infer/`：存放各段生成的视频、推理帧文件夹及 `runs_plan.json`。
- `merge/`：存放最终合并的 `frames/`、对齐时间戳 `timestamps.txt` 及完整视频。
- `slam/`：存放轨迹输出（如 `traj_est.tum`）及 SLAM 系统内部缓存。
- `eval/`：存放误差统计结果及 evo 绘制的误差对比图（`.zip` 或图像）。
- `logs/`：收口所有阶段的终端全量输出（如 `infer.log`, `slam.log`），严格遵守前缀规范，不含原地刷新的乱码。

## 4. 模块执行边界与防篡改机制
- **单向依赖**：下游步骤只能**只读**上游步骤的输出（例如 `prompt` 只能读 `segment/frames`），绝不允许反向修改或回写上游数据。
- **无状态重试**：同参数重复执行任意步骤，必须直接覆盖原有输出目录或无缝接着跑（幂等性）。
- **缺失阻断**：若前置数据缺失（如 `infer` 找不到 `prompt/manifest.json`），脚本必须立即抛出清晰的 `[ERR]` 错误并退出，禁止自行“脑补”或越权生成。

## 5. Doctor 体检机制 (`--mode doctor`)
`doctor` 模式用于在正式运行前对环境和配置进行“防呆”检查。它**只检查不落盘**，不会创建任何实验目录。
- **Critical (失败即阻断)**：检查 `datasets.json` 解析、数据源有效性、核心脚本完整性。
- **Optional (缺失仅警告)**：检查 `platform.yaml` 中的外部依赖路径（如算法仓库、模型权重）。若路径为空字符串，作缺失处理触发 `WARN`，防止将空值误解析为当前目录导致灾难性覆写。