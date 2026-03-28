# ExpHub 流水线契约

> 本文回答什么问题：每个阶段读取什么、写出什么、阶段之间靠哪些文件衔接、哪些契约改动时不能随意破坏。

系统分层请看 [ARCHITECTURE.md](./ARCHITECTURE.md)，日志规则请看 [LOGGING.md](./LOGGING.md)；若需理解当前工程模块如何映射到论文方法论，请看 [TITS_METHODOLOGY.md](./TITS_METHODOLOGY.md)。

## 1. 全局强契约

以下内容属于当前主链路的强契约。

- 主链路顺序固定为 `segment -> prompt -> infer -> merge -> slam -> eval -> stats`
- `segment/keyframes/keyframes_meta.json` 是 raw keyframe 事实源
- `segment/deploy_schedule.json` 是当前 Wan 执行投影；它不能回写覆盖 raw schedule
- `segment/state_segmentation/state_segments.json` 是 state 区间事实源；`prompt` 基于它生成 state prompt manifest
- `prompt` 当前默认强承诺输出 `final_prompt.json`、`state_prompt_manifest.json`、`deploy_to_state_prompt_map.json` 与 `report.json`
- `infer` 当前以 `prompt/final_prompt.json` 作为 base prompt 输入，并在可用时派生 `infer/prompt_manifest_resolved.json` 给 runtime 消费
- `infer` 当前默认强承诺输出 `runs_plan.json`、`prompt_manifest_resolved.json` 与 `report.json`
- `merge` 必须按 `infer/runs_plan.json` 的真实边界合并，不能再假设全局固定 `kf_gap`
- `segment / merge` 应持续写出稳定的 `step_meta.json`；`prompt / infer` 默认元信息已收敛到各自 `report.json`；`slam` 轨道元数据当前写在 `run_meta.json`
- 下游阶段只能消费上游结果，不能回写上游目录

## 2. 阶段依赖总览

| 阶段 | 主要脚本 | 关键输入 | 关键输出 | 下游依赖 |
|---|---|---|---|---|
| `segment` | `scripts/segment_make.py` | 数据集、标定、`platform.yaml` phase | `segment/frames/`, `segment/keyframes/keyframes_meta.json`, `segment/deploy_schedule.json`, `timestamps.txt`, `calib.txt`, `preprocess_meta.json`, `step_meta.json`，以及收敛后的研究产物 `segment/signal_extraction/*`、`segment/state_segmentation/*` | `prompt`, `infer`, `slam`, `segment_analyze` |
| `prompt` | `scripts/prompt_gen.py` | `segment/frames/`, `segment/state_segmentation/state_segments.json`, `segment/deploy_schedule.json`（存在时） | `prompt/final_prompt.json`, `prompt/state_prompt_manifest.json`, `prompt/deploy_to_state_prompt_map.json`, `prompt/report.json` | `infer`, `stats` |
| `infer` | `scripts/infer_i2v.py` | `segment/frames/`, `prompt/final_prompt.json`, `prompt/state_prompt_manifest.json`, `prompt/deploy_to_state_prompt_map.json`, `segment/deploy_schedule.json` | `infer/prompt_manifest_resolved.json`, `infer/runs/`, `infer/runs_plan.json`, `infer/report.json` | `merge`, `stats` |
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
- `segment/signal_extraction/*` 与 `segment/state_segmentation/*` 属于当前正式研究产物
- `segment/state_segmentation/state_segments.json` 是 state 区间事实源
- `segment/step_meta.json` 至少能支撑 `stats/report.json` 的压缩字段汇总

当前正式 `segment_policy` 口径是：

- `uniform`
- `state`（当前正式研究主线）

当前 `state` 正式主线输入固定为：

- `motion_velocity`
- `semantic_velocity`

补充约束：

- 这两者在 `segment/signal_extraction/signal_timeseries.csv` 中除了 observed raw 列，还应提供轻量的 processed formal state input 列，供正式 state score 直接消费
- `blur_score`、`appearance_delta` 等其它已提取信号可继续保留在 `signal_report.json`、`signal_overview.png` 或其它 analysis sidecar 旁路观察中，但不能再作为当前正式 state score 输入集合
- 当前正式 detector 目标是高风险区间检测：它在正式 score 上用最小 online / changepoint-style 方法识别 `low_state / high_state` 区间序列，并稳定输出 `state_segments.json`
- 当前 detector 应表现为 regime-shift-sensitive 的状态响应：基于因果 local mean / local spread 与持续证据累积形成 detector score，而不是对每个局部波峰单独强响应
- `segment/state_segmentation/` 默认只保留 `state_overview.png`、`state_report.json`、`state_segments.json` 三件套

### `prompt`

必须保证：

- 输入来自 `segment/frames/`
- 输出至少包含 `final_prompt.json`、`state_prompt_manifest.json`、`deploy_to_state_prompt_map.json` 与 `report.json`
- 若可读到 `segment/state_segmentation/state_segments.json`，应按 state 区间额外产出 `state_prompt_manifest.json`
- 若可读到 `segment/deploy_schedule.json`，应额外产出 `deploy_to_state_prompt_map.json`，把 execution segment 映射到 state prompt，而不是再生成新的 local prompt
- `final_prompt.json` 中有可直接被 `infer` 消费的 `prompt`
- `report.json` 记录 backend、采样方式、代表帧、profile 摘要、state prompt 统计与输出摘要

默认行为：

- 默认 backend 为 `smolvlm2`
- 默认采样口径为 `even + 5 images`
- 代表帧数量会被收敛到 `3..5`
- `state_prompt_manifest.json` 的语义单位以 `segment/state_segmentation/state_segments.json` 为准，而不是 `deploy_schedule`
- `deploy_to_state_prompt_map.json` 只做映射，不负责生成 prompt 文本

### `infer`

必须保证：

- 优先从 `segment/deploy_schedule.json` 构建 execution segments
- 缺失 deploy schedule 时，明确记录回退到 `legacy_kf_gap`
- 若 `prompt/state_prompt_manifest.json` 与 `prompt/deploy_to_state_prompt_map.json` 存在且可解析，必须把 global prompt 与 state local prompt 派生为 execution-segment 级别的 prompt override
- `runs_plan.json` 保存真实的 `start_idx / end_idx / num_frames`
- `prompt_manifest_resolved.json` 是 runtime 真正消费的 prompt manifest
- `report.json` 记录 backend、phase、schedule_source、prompt manifest mode、prompt source 统计、motion trend 统计与 runs plan 摘要

兼容要求：

- 当前默认 base 输入仍是 `prompt/final_prompt.json`
- `state_prompt_manifest.json` 与 `deploy_to_state_prompt_map.json` 缺失时必须无损回退到 global-only，不应影响现有实验目录运行
- runtime 继续只消费 prompt manifest；state 文件的解析与拼接发生在 `infer_i2v.py` 前端
- 为兼容旧实验，consumer 仍能从旧 prompt 文件读取 `base_prompt / base_neg_prompt`
- 这条兼容路径不应再被写成当前默认机制

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

`segment` 后的 analyze 旁路不是主链路阶段，但当前系统默认会在 `segment` 后触发；当前内部入口为 `scripts/_segment/analysis/app.py`。

它的边界是：

- 只读取 `segment/` 既有产物
- 只刷新 `segment/signal_extraction/`、`segment/state_segmentation/` 与 `segment/.segment_cache/`
- 失败时只能 warn-only，不能破坏主链路

当前正式分析产物收敛为：

- `segment/signal_extraction/signal_report.json`
- `segment/signal_extraction/signal_timeseries.csv`
- `segment/signal_extraction/signal_overview.png`
- `segment/state_segmentation/state_report.json`
- `segment/state_segmentation/state_overview.png`
- `segment/state_segmentation/state_segments.json`（事实源，继续保留）

其中当前 state 旁路/主线边界应满足：

- `signal_overview.png` 与 `state_overview.png` 需要明确标出当前正式主线输入是 `motion_velocity`、`semantic_velocity`
- 这些正式输入曲线应反映已经完成轻量预处理后的 processed 结果
- `state_overview.png` 需要直接服务高风险区间分析：上层展示两信号与正式 score，中层展示 score 与更平稳的 regime-style detector score 及最终 high-risk 区间边界，下层展示最终 `low_state / high_state` 区间序列与关键帧位置
- 当前默认不再输出 `state_signal_candidate_compare.png` 等 state score 候选 sidecar 图

当前默认不再独立生成旧 `segment/analysis/` 目录中的 `segment_summary.json`、`segment_timeseries.csv`、`kinematics_overview.png`、`allocation_overview.png`、`comparison_overview.png`、`projection_overview.png` 以及历史 `risk_*` / `proposed_schedule*` 研究文件。

其中仍有价值的旧摘要会被并入 `state_report.json`，与当前 state 研究强相关的图表信息会并入 `state_overview.png`；它们都不回写 `segment/keyframes/keyframes_meta.json`，也不回写 `segment/deploy_schedule.json`。

## 5. 最小验收检查点

对主链路与文档描述的一致性，至少要满足以下检查。

- `python -m exphub --mode doctor ...` 返回 `PASS`
- 当前默认命令能读到 `prompt/final_prompt.json`，而不是旧 manifest 文档
- `infer/runs_plan.json` 中的边界与 `merge` 实际合并边界一致
- `stats/report.json` 能从 `segment/step_meta.json`、`prompt/report.json`、`infer/report.json` 与 `merge/step_meta.json` 汇总出结果
- 日志前缀遵守 [LOGGING.md](./LOGGING.md) 中的约定

快速静态验收命令：

```bash
python -m py_compile exphub/*.py scripts/*.py
python -m exphub --mode doctor --dataset <ds> --sequence <seq> --tag <tag> --w <w> --h <h> --fps <fps> --dur <dur> --kf_gap <gap>
```
