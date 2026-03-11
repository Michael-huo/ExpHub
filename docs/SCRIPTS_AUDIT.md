# ExpHub 脚本 I/O 契约与职责说明 (SCRIPTS_AUDIT.md)

> **文档定位**：本文档严格定义了 ExpHub 主链路中各个底层脚本的职责、环境依赖、输入数据 (Inputs) 以及必须产出的数据 (Outputs)。AI 在修改任何底层脚本时，必须严格遵守此处的 I/O 契约，以确保上下游数据流转不会断裂。

## 1. 核心链路与脚本映射总览

| 执行阶段 | 核心脚本 | 环境要求 | 核心产物落盘位置 |
|---|---|---|---|
| `segment` | `scripts/segment_make.py` | phase `segment` | `segment/` |
| `prompt` | `scripts/prompt_gen.py` | phase `prompt` | `prompt/` |
| `infer` | `scripts/infer_i2v.py` <br> `_infer_i2v_impl.py` | phase `infer` | `infer/` |
| `merge` | `scripts/merge_seq.py` | 主环境 | `merge/` |
| `slam` | `scripts/slam_droid.py` | phase `slam` | `slam/<track>/` |
| `eval` | `exphub/cli.py` (调用 evo) | phase `slam` | `eval/<track>/` |
| `stats` | `scripts/stats_collect.py` | 主环境 | `stats/` 或根目录报告 |

---

## 2. 脚本 I/O 契约详情

### 2.1 `scripts/segment_make.py` (切片锚定)
- **职责**：`segment` 的稳定命令行入口。内部实现位于 `scripts/_segment/`，负责解析原始数据集、抽帧、写回标准目录结构。
- **配置依赖**：`config/datasets.json`；顶层默认解释器来自 `config/platform.yaml -> environments.phases.segment.python`。
- **Inputs (读取)**：外部原始数据集。
- **Outputs (写入)**：
  - `segment/frames/`：按规范命名（如 `000000.png`）的抽帧图像。
  - `segment/keyframes/` 与 `segment/keyframes/keyframes_meta.json`：正式关键帧集合及元信息。默认 `--segment_policy uniform` 仍输出 legacy uniform 锚帧；`--segment_policy sks_v1` 输出 “uniform 骨架 + 固定预算纯语义受限重定位” 的正式硬关键帧；`--segment_policy semantic_guarded_v1` 输出 “uniform 骨架 + boundary/support 修正” 的第一版硬关键帧；`--segment_policy semantic_guarded_v2` 则在 v1 基础上额外启用 support 插点与 `low_prominence` 型 suppressed-high 晋升。
  - `segment/timestamps.txt`：相对时间网格。
  - `segment/calib.txt`：与 `frames/` 尺寸对齐的内参。
  - `segment/preprocess_meta.json`：裁剪、缩放与内参变换元数据。
  - `segment/step_meta.json`：包含基础元信息（如分辨率、fps、帧数）。
- **关键帧策略契约**：
  - `uniform`：保持旧行为，每隔 `kf_gap` 采样一个锚点。
  - `sks_v1`：先生成 uniform 骨架，再在 `segment_make.py` 进程内复用 OpenCLIP image embedding 旁路，计算 `semantic displacement / velocity / acceleration / density / cumulative action`，并在不改变预算的前提下做局部受限重定位：
    - 首尾关键帧固定；
    - 最终关键帧数严格等于 uniform；
    - 中间关键帧仅允许在 `uniform_base_indices` 对应邻域内移动；
    - 通过 local density peak snap 向局部语义高值吸附；
    - 不接入旧的非语义 score 作为主公式。
  - `semantic_guarded_v1`：先生成 uniform 骨架，再在 `segment_make.py` 进程内复用研究旁路同源逻辑，直接计算 `boundary_candidate / support_candidate / semantic_only_candidate / suppressed` 与 `rerank_score`，其中：
    - `boundary_candidate` 可做吸附/替换/插入；
    - `support_candidate` 只做局部补点；
    - `semantic_only_candidate` 第一版仅保留为 soft observation，不进入硬关键帧；
    - `suppressed` 不进入硬关键帧。
  - `semantic_guarded_v2`：保留 v1 的 boundary 优先级与 observe-only 语义策略，并新增：
    - 原生 `support_candidate` 按 `rerank_score / nonsemantic_support / local_prominence` 排序参与插点；
    - 仅对 `low_prominence` 型 `suppressed` 候选开放二次晋升，晋升后统一标记为 `promoted_support_candidate`；
    - 轻量 `burst/window` 规则可在高活动且覆盖偏稀疏的窗口内额外放行 1 个 suppressed-high；
    - `semantic_only_candidate` 仍不进入硬关键帧集合。
- **`keyframes_meta.json` 向后兼容扩展字段**：
  - 保留旧字段：`kf_gap`、`frame_count_total`、`frame_count_used`、`tail_drop`、`keyframe_count`、`keyframe_indices`、`keyframe_bytes_sum`。
  - 新增 `policy_name`、`uniform_base_indices`、`summary`、`keyframes`、`policy_meta`。
  - `summary` 至少包含 `num_uniform_base / num_boundary_selected / num_support_selected / num_boundary_relocated / num_boundary_inserted / num_support_inserted / num_promoted_support_inserted / num_burst_windows_triggered / num_final_keyframes / extra_kf_ratio`；若 `policy_name=sks_v1`，还应包含 `uniform_count / final_keyframe_count / fixed_budget / relocated_count / avg_abs_shift / max_abs_shift / semantic_velocity_mean / semantic_velocity_max / semantic_acceleration_mean / semantic_acceleration_max`。
  - `keyframes[*]` 额外包含 `source_type / source_role / promotion_source / promotion_reason / window_id / rerank_score / semantic_relation / is_inserted / is_relocated / replaced_uniform_index`。

### 2.2 `scripts/segment_analyze.py` (研究分析旁路)
- **职责**：不改变主链路关键帧决策，只读取已存在的 `segment/` 结果，为关键帧研究提供逐帧非语义信号、OpenCLIP 图像语义变化信号、candidate role 判别、候选点 rerank 与收敛后的核心可视化。
- **调度契约**：
  - `python -m exphub --mode segment ...` 在 `segment` 成功后默认自动触发 `segment_analyze.py --exp_dir <EXP_DIR>`。
  - 若显式传入 `--skip_analyze`，则跳过该后处理。
  - `--mode all` 与 `--mode doctor` 不自动触发分析旁路。
  - 由 `cli.py` 自动触发时，默认复用与 `segment_make.py` 相同的 `segment` phase interpreter。
  - analyze 失败只能以 WARN 形式报告，不得让 `segment` 主流程失败。
- **配置依赖**：优先复用 `exphub/context.py` 的实验路径规则；不读取 `platform.yaml`。
- **Inputs (读取)**：
  - `segment/frames/`
  - `segment/timestamps.txt`
  - `segment/keyframes/keyframes_meta.json`
  - `segment/step_meta.json`
  - 可选：`segment/preprocess_meta.json`
- **关键帧集合解释**：
  - 若 `keyframes_meta.json` 中存在 `uniform_base_indices`，研究旁路将其视为 `is_uniform_keyframe` 的基准集合。
  - `keyframe_indices` 仍表示当前 policy 输出的正式硬关键帧集合，可用于可视化最终布局。
  - `keyframes_meta.json` 中的 `keyframes[*].source_role / promotion_source` 可用于区分 `boundary_candidate / support_candidate / promoted_support_candidate` 的最终来源。
- **Outputs (写入)**：
  - `segment/analysis/analysis_summary.json`：唯一核心汇总 json，汇合基础实验信息、关键帧统计、最终来源统计、策略行为统计、候选角色统计、关键索引、语义配置与 boundary/support/promoted 摘要；若当前 policy 为 `sks_v1`，还会附带 fixed-budget relocation 与 semantic velocity / acceleration / density 统计。
  - `segment/analysis/frame_scores.csv`：逐帧瘦身表，仅保留研究需要的核心列。
  - `segment/analysis/score_overview.png`
  - `segment/analysis/roles_overview.png`
  - `segment/analysis/semantic_overview.png`：继续沿用单文件输出，但允许在同一图中增量展示 `semantic displacement / velocity / acceleration / density` 及重定位关键帧标记。
  - `segment/.segment_cache/segment_analyze/semantic_embeddings.npz`：OpenCLIP image embedding cache，仅供 `segment_analyze.py` 旁路复用，不再写入 `analysis/`。

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
