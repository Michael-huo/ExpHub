# ExpHub 脚本 I/O 契约与职责说明 (SCRIPTS_AUDIT.md)

> **文档定位**：本文档严格定义了 ExpHub 主链路中各个底层脚本的职责、环境依赖、输入数据 (Inputs) 以及必须产出的数据 (Outputs)。AI 在修改任何底层脚本时，必须严格遵守此处的 I/O 契约，以确保上下游数据流转不会断裂。

## 1. 核心链路与脚本映射总览

| 执行阶段 | 核心脚本 | 环境要求 | 核心产物落盘位置 |
|---|---|---|---|
| `segment` | `scripts/segment_make.py` | 主环境 | `segment/` |
| `prompt` | `scripts/prompt_gen.py` | `prompt_python` | `prompt/` |
| `infer` | `scripts/infer_i2v.py` <br> `_infer_i2v_impl.py` | `infer_python` | `infer/` |
| `merge` | `scripts/merge_seq.py` | 主环境 | `merge/` |
| `slam` | `scripts/slam_droid.py` | `slam_python` | `slam/<track>/` |
| `eval` | `exphub/cli.py` (调用 evo) | `slam_python` | `eval/<track>/` |
| `stats` | `scripts/stats_collect.py` | 主环境 | `stats/` 或根目录报告 |

---

## 2. 脚本 I/O 契约详情

### 2.1 `scripts/segment_make.py` (切片锚定)
- **职责**：`segment` 的稳定命令行入口。内部实现位于 `scripts/_segment/`，负责解析原始数据集、抽帧、写回标准目录结构。
- **配置依赖**：`config/datasets.json`。
- **Inputs (读取)**：外部原始数据集。
- **Outputs (写入)**：
  - `segment/frames/`：按规范命名（如 `000000.png`）的抽帧图像。
  - `segment/keyframes/` 与 `segment/keyframes/keyframes_meta.json`：uniform 锚帧及元信息。
  - `segment/timestamps.txt`：相对时间网格。
  - `segment/calib.txt`：与 `frames/` 尺寸对齐的内参。
  - `segment/preprocess_meta.json`：裁剪、缩放与内参变换元数据。
  - `segment/step_meta.json`：包含基础元信息（如分辨率、fps、帧数）。

### 2.2 `scripts/segment_analyze.py` (研究分析旁路)
- **职责**：不进入主链路，只读取已存在的 `segment/` 结果，为关键帧研究提供逐帧非语义信号、OpenCLIP 图像语义变化信号、candidate role 判别、候选点 rerank 与可视化。
- **配置依赖**：优先复用 `exphub/context.py` 的实验路径规则；不读取 `platform.yaml`。
- **Inputs (读取)**：
  - `segment/frames/`
  - `segment/timestamps.txt`
  - `segment/keyframes/keyframes_meta.json`
  - `segment/step_meta.json`
  - 可选：`segment/preprocess_meta.json`
- **Outputs (写入)**：
  - `segment/analysis/frame_scores.csv`
  - `segment/analysis/frame_scores.json`
  - `segment/analysis/score_curve.png`
  - `segment/analysis/score_curve_with_keyframes.png`
  - `segment/analysis/analysis_meta.json`
  - `segment/analysis/candidate_points.json`
  - `segment/analysis/candidate_roles_summary.json`
  - `segment/analysis/candidate_points_overview.png`
  - `segment/analysis/candidate_roles_overview.png`
  - `segment/analysis/semantic_embeddings.npz`：OpenCLIP image embedding cache，仅供 `segment_analyze.py` 研究旁路复用。
  - `segment/analysis/semantic_curve.png`
  - `segment/analysis/semantic_vs_nonsemantic.png`
  - 可选：`segment/analysis/peaks_preview.png`

### 2.3 `scripts/prompt_gen.py` (提示词生成)
- **职责**：调用 VLM 模型，对 `segment` 提取的关键帧进行理解，生成用于下游视频生成的提示词清单。
- **配置依赖**：从 `platform.yaml` 读取 `models.qwen2_vl.path`。
- **Inputs (读取)**：`segment/frames/`。
- **Outputs (写入)**：
  - `prompt/manifest.json`：标准的提示词清单文件（包含 base_prompt 和各段 delta_prompt）。
  - `prompt/step_meta.json`。

### 2.4 `scripts/infer_i2v.py` & `scripts/_infer_i2v_impl.py` (视频生成推理)
- **职责**：
  - `infer_i2v.py`：外壳编排器，负责规划分段任务，并使用 `sys.executable` 跨环境拉起底层推理引擎（支持多卡 `torchrun`）。
  - `_infer_i2v_impl.py`：底层执行器（Wan2.2 Backend），执行 float8 量化与核心生成逻辑。
- **配置依赖**：从 `platform.yaml` 读取 `repos.videox_fun` 以及 `models.wan2_2` 的路径与配置。
- **Inputs (读取)**：
  - `segment/frames/`（读取首尾关键帧作为生成锚点）。
  - `prompt/manifest.json`。
- **Outputs (写入)**：
  - `infer/runs/`：包含各个分段生成的视频片段。
  - `infer/runs_plan.json`：运行计划与参数记录。
  - `infer/step_meta.json`。

### 2.5 `scripts/merge_seq.py` (序列合并)
- **职责**：将 `infer` 生成的分散视频片段在时间轴上对齐、去重边界关键帧，并融合成最终的长序列图像。
- **配置依赖**：无。
- **Inputs (读取)**：`infer/runs_plan.json` 以及 `infer/runs/` 下的各段帧。
- **Outputs (写入)**：
  - `merge/frames/`：全局对齐后的最终连续图像序列。
  - `merge/timestamps.txt`：供 SLAM 系统使用的时间戳对齐文件。
  - `merge/step_meta.json`。

### 2.6 `scripts/slam_droid.py` (位姿估计)
- **职责**：在特定图像轨道（如 `ori` 原始轨道或 `gen` 生成轨道）上运行 DROID-SLAM 算法提取相机轨迹。
- **配置依赖**：从 `platform.yaml` 读取 `repos.droid_slam` 与 `models.droid.path`。
- **Inputs (读取)**：
  - 若为 `gen` 轨道：读取 `merge/frames/` 与 `merge/timestamps.txt`。
  - 若为 `ori` 轨道：读取 `segment/frames/` 与 `segment/timestamps.txt`（如果存在）。
- **Outputs (写入)**：
  - `slam/<track>/traj_est.tum`：TUM 格式的位姿估计输出。
  - `slam/<track>/step_meta.json`。

### 2.7 `scripts/stats_collect.py` (统计出图)
- **职责**：全局扫描所有阶段的元数据与系统日志，汇总性能数据与耗时拆分。
- **Inputs (读取)**：各阶段产生的 `step_meta.json` 及 `logs/` 下的心跳日志。
- **Outputs (写入)**：最终的 JSON 格式性能数据报表，支撑终端的 `EXPERIMENT PERFORMANCE PROFILING` 面板输出。

### 2.8 工具支撑：`scripts/_common.py`
- **职责**：提供所有脚本共享的核心基建。
- **核心接口契约**：
  - `get_platform_config()`：全局唯一的配置寻址入口。
  - 日志门面：`log_info`, `log_warn`, `log_err`, `log_prog`。严禁在任何子脚本中私自使用原生 `print()` 处理业务逻辑。
