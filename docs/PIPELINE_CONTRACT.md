# ExpHub 流水线契约

> 本文回答什么问题：每个阶段读取什么、写出什么、阶段之间靠哪些文件衔接、哪些契约改动时不能随意破坏。

系统分层请看 [ARCHITECTURE.md](./ARCHITECTURE.md)，日志规则请看 [LOGGING.md](./LOGGING.md)；若需理解当前工程模块如何映射到论文方法论，请看 [TITS_METHODOLOGY.md](./TITS_METHODOLOGY.md)。

## 1. 全局强契约

以下内容属于当前主链路的强契约。

- 主链路顺序固定为 `segment -> prompt -> infer -> merge -> slam -> eval -> stats`
- `segment/keyframes/keyframes_meta.json` 是 raw keyframe 事实源
- `segment/deploy_schedule.json` 是当前 Wan 执行投影；它不能回写覆盖 raw schedule
- `segment/state_segmentation/state_segments.json` 是 state 区间事实源；`prompt` 基于它生成 state prompt manifest
- `prompt` 当前默认强承诺输出 `base_prompt.json`、`state_prompt_manifest.json`、`runtime_prompt_plan.json` 与 `report.json`
- `infer` 当前正式消费 `prompt/runtime_prompt_plan.json`，不再自行拼接 `base + local` prompt
- `infer` 当前默认强承诺输出 `runs_plan.json` 与 `report.json`
- `merge` 必须按 `infer/runs_plan.json` 的真实边界合并，不能再假设全局固定 `kf_gap`
- `segment / merge` 应持续写出稳定的 `step_meta.json`；`prompt / infer` 默认元信息已收敛到各自 `report.json`；`slam` 轨道元数据当前写在 `run_meta.json`
- 下游阶段只能消费上游结果，不能回写上游目录

## 2. 阶段依赖总览

| 阶段 | 主要脚本 | 关键输入 | 关键输出 | 下游依赖 |
|---|---|---|---|---|
| `segment` | `scripts/segment_make.py` | 数据集、标定、`platform.yaml` phase | `segment/frames/`, `segment/keyframes/keyframes_meta.json`, `segment/deploy_schedule.json`, `timestamps.txt`, `calib.txt`, `preprocess_meta.json`, `step_meta.json`, `segment/state_segmentation/*` | `prompt`, `infer`, `slam` |
| `prompt` | `scripts/prompt_gen.py` | `segment/frames/`, `segment/state_segmentation/state_segments.json`, `segment/deploy_schedule.json` | `prompt/base_prompt.json`, `prompt/state_prompt_manifest.json`, `prompt/runtime_prompt_plan.json`, `prompt/report.json` | `infer`, `stats` |
| `infer` | `scripts/infer_i2v.py` | `segment/frames/`, `prompt/runtime_prompt_plan.json`, `segment/deploy_schedule.json` | `infer/runs/`, `infer/runs_plan.json`, `infer/report.json` | `merge`, `stats` |
| `merge` | `scripts/merge_seq.py` | `infer/runs_plan.json`, `infer/runs/*`, `segment/calib.txt`, `segment/timestamps.txt` | `merge/frames/`, `merge/timestamps.txt`, `merge/calib.txt`, `merge/merge_meta.json`, `merge/step_meta.json` | `slam`, `stats` |
| `slam` | `scripts/slam_droid.py` | `segment/` 或 `merge/` 轨道数据 | `slam/<track>/traj_est.tum`, `traj_est.npz`, `run_meta.json` | `eval` |
| `eval` | `scripts/eval_main.py` | `slam/ori/traj_est.tum`, `slam/gen/traj_est.tum`, `merge/frames/`, `segment/frames/` | `eval/report.json`, `eval/details.csv`, `eval/plots/traj_xy.png`, `eval/plots/metrics_overview.png` | 人工分析 |
| `stats` | `scripts/stats_collect.py` | `segment/step_meta.json`, `prompt/report.json`, `infer/report.json`, `merge/step_meta.json` 与日志 | `stats/report.json`, `stats/compression.json` | 汇总出口 |

## 3. 各阶段最小契约

### `segment`

必须保证：

- `segment/frames/` 可供 `prompt` 与 `infer` 直接读取
- `segment/keyframes/keyframes_meta.json` 存在，并表达正式 raw keyframe 结果
- `segment/deploy_schedule.json` 存在时，`infer` 会优先使用它；它是执行投影，不回写 raw schedule
- `segment/state_segmentation/*` 属于当前正式主线产物；`segment/signal_extraction/*` 只在显式 sidecar 任务中生成
- `segment/state_segmentation/state_segments.json` 是 state 区间事实源
- `segment/step_meta.json` 至少能支撑 `stats/report.json` 的压缩字段汇总

当前正式 `segment_policy` 口径是：

- `uniform`
- `state`（当前正式研究主线）

当前 `state` 正式主线输入固定为：

- `motion_velocity`
- `semantic_velocity`

补充约束：

- 这两者是当前正式 `state_score` 的唯一输入；主链允许在内部完成轻量预处理，但默认不再要求把 `segment/signal_extraction/signal_timeseries.csv` 作为正式产物落盘
- `blur_score`、`appearance_delta` 等其它已提取信号只允许保留在显式 `signal_extract` / `analyze` 侧路观察中，不能再作为当前正式 state score 输入集合
- 当前正式 detector 目标是高风险区间检测：它在正式 score 上用最小 online / changepoint-style 方法识别 `low_state / high_state` 区间序列，并稳定输出 `state_segments.json`
- 当前 detector 应表现为 regime-shift-sensitive 的状态响应：基于因果 local mean / local spread 与持续证据累积形成 detector score，而不是对每个局部波峰单独强响应
- `segment/state_segmentation/` 默认只保留 `state_overview.png`、`state_report.json`、`state_segments.json` 三件套

### `prompt`

必须保证：

- 输入来自 `segment/frames/`
- 输出至少包含 `base_prompt.json`、`state_prompt_manifest.json`、`runtime_prompt_plan.json` 与 `report.json`
- `base_prompt.json` 只表达全局硬约束，如 first-person continuity、geometry consistency、stable exposure / white balance、no flicker / warping / ghosting
- 若可读到 `segment/state_segmentation/state_segments.json`，应按 state 区间生成 `state_prompt_manifest.json`
- `state_prompt_manifest.json` 的语义单位以 `segment/state_segmentation/state_segments.json` 为准；其中每个区间至少包含 `state_segment_id`、`start_frame`、`end_frame`、`state_label`、`prompt_text`、`negative_prompt_delta`、`prompt_strength`
- 若可读到 `segment/deploy_schedule.json`，应生成 `runtime_prompt_plan.json`，把 deploy execution segments 直接展开为 runtime 可消费的 per-segment prompt plan
- `runtime_prompt_plan.json` 中每个 deploy segment 至少包含 `deploy_segment_id`、`start_frame`、`end_frame`、`state_segment_id`、`state_label`、`base_prompt`、`local_prompt`、`resolved_prompt`、`negative_prompt`、`prompt_strength`
- `report.json` 记录 backend、采样方式、代表帧、profile 摘要、state prompt 统计与输出摘要

默认行为：

- 默认 backend 为 `smolvlm2`
- 默认采样口径为 `even + 5 images`
- 代表帧数量会被收敛到 `3..5`
- `state_prompt_manifest.json` 的语义单位以 `segment/state_segmentation/state_segments.json` 为准，而不是 `deploy_schedule`
- `runtime_prompt_plan.json` 是 `infer` 的唯一正式 prompt 输入文件

### `infer`

必须保证：

- 优先从 `segment/deploy_schedule.json` 构建 execution segments
- 缺失 deploy schedule 时，明确记录回退到 `legacy_kf_gap`
- 直接读取 `prompt/runtime_prompt_plan.json`，而不是在 `infer` 前端重新拼接 global prompt 与 local prompt
- `runs_plan.json` 保存真实的 `start_idx / end_idx / num_frames`
- `report.json` 记录 backend、phase、schedule_source、runtime prompt source 统计、motion trend 统计与 runs plan 摘要

兼容要求：

- 当前默认 prompt 输入是 `prompt/runtime_prompt_plan.json`
- `infer_i2v.py` 只负责校验 execution plan 与 runtime prompt plan 对齐，不再承担 prompt 拼接职责
- `segment/deploy_schedule.json` 继续只负责 execution 边界，不承担 prompt 文本生成

### `merge`

必须保证：

- 只按 `runs_plan.json` 的真实边界去重与拼接
- `merge/frames/` 与 `merge/timestamps.txt` 一一对应
- 结果目录可直接被 `slam` 当作“数据集式”输入读取

### `slam`

必须保证：

- `ori` 轨读取 `segment/`
- `gen` 轨读取 `merge/`
- 每个轨道至少产出 `traj_est.tum` 与 `run_meta.json`

### `eval`

必须保证：

- 在 `slam` phase 环境中调度 `scripts/eval_main.py`；前端壳负责组织输入，后端 `_eval/` 统一执行 traj/image eval
- 结构化输出至少包含聚合后的 `eval/report.json` 与 `eval/details.csv`
- `eval/report.json` 至少聚合 `traj_eval`、`image_eval`、`slam_friendly_eval`、`summary_text`、`warnings`、`eval_status` 与 `artifact_contract`
- 轨迹图像输出收敛到 `eval/plots/`，当前默认只保留 `traj_xy.png` 与 `metrics_overview.png`
- 当 `segment/keyframes/keyframes_meta.json` 可用时，`eval/plots/` 中的主轨迹图与各类曲线图会以轻量标记标出最终关键帧位置；缺失时只降级为不标记，不影响图生成
- `traj_xy.png` 保留历史文件名，但其主图语义已是“主二维投影视图”，不再强绑定真实世界固定 XY 平面
- 图像评价默认统计 `infer -> merge` 生成帧与对应 `ori` 帧的逐帧比较，其聚合指标并入 `eval/report.json`，逐帧明细并入 `eval/details.csv`
- SLAM-friendly 图像评价默认只统计时间上连续的相邻生成帧对，其聚合指标并入 `eval/report.json`，pair 明细并入 `eval/details.csv`
- `eval/details.csv` 使用 `row_type=image_frame|slam_pair` 同时承载 image per-frame 与 slam pair 明细
- `traj_eval` 块至少包含 APE translation、RPE translation、matched pose 数、`ori_path_length_m`、`gen_path_length_m`、`eval_status` 与 `warnings`
- `image_eval` 块至少包含 `psnr`、`ms_ssim`、`lpips` 的聚合统计、`frame_count`、`eval_status` 与 `warnings`
- `slam_friendly_eval` 块至少包含 `inlier_ratio`、`pose_success_rate`、`reference_source`、`uses_proxy_reference`、`valid_pair_count`、`valid_pose_pair_count` 与 `warnings`
- `summary_text` 汇总轨迹、图像与 slam-friendly 评价的人类可读摘要
- 缺少轨迹文件或 `evo` 依赖时不能让主链路崩溃，应写出可理解的失败摘要，而不是只留下零散 txt

### `stats`

必须保证：

- 输出 `stats/report.json`
- 保留 `stats/compression.json` 兼容历史消费
- 对缺失的 `segment/step_meta.json`、`merge/step_meta.json`、`prompt/report.json` 或 `infer/report.json` 给出 `WARN`，而不是直接崩溃
- 实验结束后终端统一打印单块 `EXPERIMENT REPORT`，汇总 time / quality / compression 三类关键指标；其中 quality 区块当前也包含 `ori_path_length_m` / `gen_path_length_m` 的展示

## 4. `segment` 后分析旁路契约

`segment` 后的 analyze 旁路不是主链路阶段；当前内部入口为 `scripts/_segment/analysis/app.py`，但它已降级为显式手动 sidecar，不再由默认正式主链自动触发。

它的边界是：

- 只读取 `segment/` 既有产物
- 只刷新 `segment/signal_extraction/`、`segment/state_segmentation/` 与 `segment/.segment_cache/`
- 失败时只能 warn-only，不能破坏主链路

当前显式 sidecar 分析产物收敛为：

- `segment/signal_extraction/signal_report.json`
- `segment/signal_extraction/signal_timeseries.csv`
- `segment/signal_extraction/signal_overview.png`

其中当前 state 旁路/主线边界应满足：

- `signal_overview.png` 需要明确标出当前正式主线输入是 `motion_velocity`、`semantic_velocity`
- 这些正式输入曲线应反映已经完成轻量预处理后的 processed 结果
- 当前默认不再输出 `state_signal_candidate_compare.png` 等旧 sidecar 对照图

当前默认不再独立生成旧 `segment/analysis/` 目录中的 `segment_summary.json`、`segment_timeseries.csv`、`kinematics_overview.png`、`allocation_overview.png`、`comparison_overview.png`、`projection_overview.png` 以及历史 `risk_*` / `proposed_schedule*` 研究文件。

这些 sidecar 产物都不回写 `segment/keyframes/keyframes_meta.json`，也不回写 `segment/deploy_schedule.json`，更不构成 prompt / infer / schedule 的正式消费契约。

## 5. 最小验收检查点

对主链路与文档描述的一致性，至少要满足以下检查。

- `python -m exphub --mode doctor ...` 返回 `PASS`
- 当前默认命令能读到 `prompt/runtime_prompt_plan.json`
- `infer/runs_plan.json` 中的边界与 `merge` 实际合并边界一致
- `stats/report.json` 能从 `segment/step_meta.json`、`prompt/report.json`、`infer/report.json` 与 `merge/step_meta.json` 汇总出结果
- 日志前缀遵守 [LOGGING.md](./LOGGING.md) 中的约定

快速静态验收命令：

```bash
python -m py_compile exphub/*.py scripts/*.py
python -m exphub --mode doctor --dataset <ds> --sequence <seq> --tag <tag> --w <w> --h <h> --fps <fps> --dur <dur> --kf_gap <gap>
```
