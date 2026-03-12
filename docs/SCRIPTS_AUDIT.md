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
  - `segment/keyframes/` 与 `segment/keyframes/keyframes_meta.json`：正式关键帧集合及元信息。当前正式 analyze 与方法叙事聚焦 `--segment_policy uniform / sks_v1 / motion_energy_v1`：
    - `uniform` 输出 legacy uniform 锚帧；
    - `sks_v1` 输出 “uniform 骨架 + fixed-budget allocation + semantic kinematics” 的正式硬关键帧；
    - `motion_energy_v1` 输出 “uniform 骨架 + fixed-budget allocation + motion energy kinematics” 的正式硬关键帧。
  - `segment/timestamps.txt`：相对时间网格。
  - `segment/calib.txt`：与 `frames/` 尺寸对齐的内参。
  - `segment/preprocess_meta.json`：裁剪、缩放与内参变换元数据。
  - `segment/deploy_schedule.json`：当前第一版 `wan_r4` backend-specific deploy schedule。它保留：
    - `raw_keyframe_indices`：raw schedule 的引用；
    - `deploy_keyframe_indices`：Wan 合法时间网格上的执行边界；
    - `segments[*]`：每段 `raw_start/end_idx`、`deploy_start/end_idx`、`raw_gap`、`deploy_gap`、`num_frames`；
    - `projection_stats`：boundary/gap 误差统计。
  - `segment/step_meta.json`：包含基础元信息（如分辨率、fps、帧数）。
- **raw / deploy 契约**：
  - `keyframes_meta.json` 仍然只表达 raw schedule，保持 canonical/source-of-truth 地位；
  - `deploy_schedule.json` 只服务当前 Wan 执行，不回写覆盖 raw keyframes；
  - 当前 `wan_r4` 规则要求：首尾固定、段数不变、总跨度不变、每段 deploy gap 为 4 的倍数；若无解则 fail-fast。
- **关键帧策略契约**：
  - `uniform`：保持旧行为，每隔 `kf_gap` 采样一个锚点。
  - `sks_v1`：先生成 uniform 骨架，再在 `segment_make.py` 进程内复用 OpenCLIP image embedding 旁路，计算 `semantic displacement / velocity / acceleration / density / cumulative action`，并在不改变预算的前提下做局部受限重定位：
    - 首尾关键帧固定；
    - 最终关键帧数严格等于 uniform；
    - 中间关键帧仅允许在 `uniform_base_indices` 对应邻域内移动；
    - 通过 local density peak snap 向局部语义高值吸附；
    - 不接入旧的非语义 score 作为主公式。
  - `motion_energy_v1`：与 `sks_v1` 共用完全相同的 fixed-budget allocation 骨架，但把输入信号替换为统一尺寸帧序列上的轻量 motion energy kinematics：
    - 先对灰度帧做 `5x5` 高斯模糊；
    - 使用相邻帧差分定义 `motion displacement`；
    - 再计算 `motion velocity / acceleration / density / cumulative action`；
    - 首尾关键帧固定；
    - 最终关键帧数严格等于 uniform；
    - 中间关键帧仅允许在 `uniform_base_indices` 对应邻域内移动。
- **`keyframes_meta.json` 向后兼容扩展字段**：
  - 保留旧字段：`kf_gap`、`frame_count_total`、`frame_count_used`、`tail_drop`、`keyframe_count`、`keyframe_indices`、`keyframe_bytes_sum`。
  - 新增 `policy_name`、`uniform_base_indices`、`summary`、`keyframes`、`policy_meta`。
  - `summary` 至少包含 `num_uniform_base / num_final_keyframes / extra_kf_ratio`；若 `policy_name=sks_v1`，还应包含 `uniform_count / final_keyframe_count / fixed_budget / relocated_count / avg_abs_shift / max_abs_shift / semantic_displacement_* / semantic_velocity_* / semantic_acceleration_* / semantic_density_* / semantic_action_total`；若 `policy_name=motion_energy_v1`，则输出完全平行的 `motion_displacement_* / motion_velocity_* / motion_acceleration_* / motion_density_* / motion_action_total`。
  - `keyframes[*]` 额外包含 `source_type / source_role / rerank_score / semantic_relation / is_inserted / is_relocated / replaced_uniform_index`；历史字段如 `promotion_source / promotion_reason / window_id` 若存在，仅用于被动读取旧实验产物。

### 2.2 `scripts/segment_analyze.py` (研究分析旁路)
- **职责**：不改变主链路关键帧决策，只读取已存在的 `segment/` 结果，为正式三策略 `uniform / sks_v1 / motion_energy_v1` 提供逐帧 kinematics/allocation 分析；同时内建 active policy + passive observer compare 旁路：`sks_v1` 默认观测 `motion_energy_v1`，`motion_energy_v1` 默认观测 `sks_v1`，`uniform` 同时观测两者；observer 只写 analyze 产物，不写正式 keyframes；已移除的早期启发式策略不会再进入 analyze 主路径。
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
  - `keyframes_meta.json` 中的 `keyframes[*]` 会保留最终来源与重定位标记，供 allocation/observer 可视化复用。
- **Outputs (写入)**：
  - `segment/analysis/analysis_summary.json`：唯一核心汇总 json。对 `uniform` 保留基础关键帧摘要，并追加 dual observer compare block；对 `sks_v1 / motion_energy_v1` 主字段明确收敛为 fixed-budget relocation 与 `semantic_*` 或 `motion_*` kinematics 统计，并追加 `comparison.{observer_policy,signal_alignment,allocation_alignment,observer_summary}`。
  - `segment/analysis/frame_scores.csv`：正式离线分析表。通用列至少包含 `frame_idx / is_uniform_anchor / is_selected_keyframe / is_relocated_keyframe`，并按 policy 追加 active density/action、observer density/action 与 observer final keyframe 标记；`uniform` 下允许同时输出 `sks_v1_*` 与 `motion_energy_v1_*` 对比列。
  - `segment/analysis/score_overview.png`：正式 compare overview，主用于表达 active vs observer（或 `uniform` 下 observer vs observer）的 density / action / allocation 横向对比。
  - `segment/analysis/roles_overview.png`：文件名保留，但对正式策略实际表达 allocation overview，不再围绕 legacy role 讲 `boundary/support/promoted` 叙事。
  - `segment/analysis/semantic_overview.png`：文件名保留，作为正式主图按 policy 自适应展示 semantic 或 motion kinematics，并叠加 uniform anchors / relocated keyframes / final keyframes。
  - `segment/.segment_cache/<policy>/semantic_embeddings.npz`：当 active 或 observer 需要 `sks_v1` 语义信号时，analyze 复用/写入对应的 policy cache；不会再写入 `analysis/`。

### 2.3 `scripts/prompt_gen.py` (提示词生成)
- **职责**：调用 VLM 模型，对 `segment` 提取的关键帧进行理解，生成用于下游视频生成的提示词清单。
- **配置依赖**：从 `platform.yaml` 读取 `models.qwen2_vl.path`。
- **Inputs (读取)**：`segment/frames/`。
- **Outputs (写入)**：
  - `prompt/manifest.json`：标准的提示词清单文件（包含 base_prompt 和各段 delta_prompt），并复用 `segments[*]` 作为 downstream execution manifest：
    - `segments[*].start_idx / end_idx / num_frames` 是 `infer / merge` 真正消费的运行计划；
    - 若 `segment/deploy_schedule.json` 存在，则这些边界来自 deploy schedule；
    - 否则为历史实验回退到 legacy `kf_gap` 切段。
  - `prompt/step_meta.json`。

### 2.4 `scripts/infer_i2v.py` & `scripts/_infer_i2v_impl.py` (视频生成推理)
- **职责**：
  - `infer_i2v.py`：外壳编排器，负责规划分段任务，并使用 `sys.executable` 跨环境拉起底层推理引擎（支持多卡 `torchrun`）。
  - `_infer_i2v_impl.py`：底层执行器（Wan2.2 Backend），执行 float8 量化与核心生成逻辑。
- **配置依赖**：从 `platform.yaml` 读取 `repos.videox_fun` 以及 `models.wan2_2` 的路径与配置。
- **Inputs (读取)**：
  - `segment/frames/`（读取首尾关键帧作为生成锚点）。
  - `prompt/manifest.json`（优先读取其中的 execution segments；旧 manifest 则回退到 `segment/deploy_schedule.json` 或 legacy `kf_gap`）。
- **Outputs (写入)**：
  - `infer/runs/`：包含各个分段生成的视频片段。
  - `infer/runs_plan.json`：运行计划与参数记录。`segments[*]` 会显式保存每段真实的 `start_idx / end_idx / raw_* / deploy_* / num_frames`。
  - `infer/step_meta.json`。

### 2.5 `scripts/merge_seq.py` (序列合并)
- **职责**：将 `infer` 生成的分散视频片段在时间轴上对齐、去重边界关键帧，并融合成最终的长序列图像。
- **配置依赖**：无。
- **Inputs (读取)**：`infer/runs_plan.json` 以及 `infer/runs/` 下的各段帧。
- **Outputs (写入)**：
  - `merge/frames/`：全局对齐后的最终连续图像序列。
  - `merge/timestamps.txt`：供 SLAM 系统使用的时间戳对齐文件。
  - `merge/step_meta.json`。
- **合并契约**：
  - `merge_seq.py` 以 `runs_plan.json` 的真实 `start_idx / end_idx` 为唯一时间轴依据；
  - 不再假设所有 segment 共用固定 `kf_gap`/`stride`；
  - 只要相邻 segments 在 plan 中共享 boundary，merge 就按 boundary overlap 去重并恢复原时间轴长度。

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
