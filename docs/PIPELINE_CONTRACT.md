# ExpHub 流水线契约

> 本文回答什么问题：每个阶段读取什么、写出什么、阶段之间靠哪些文件衔接、哪些契约改动时不能随意破坏。

系统分层请看 [ARCHITECTURE.md](./ARCHITECTURE.md)，日志规则请看 [LOGGING.md](./LOGGING.md)；若需理解当前工程模块如何映射到论文方法论，请看 [TITS_METHODOLOGY.md](./TITS_METHODOLOGY.md)。

## 1. 全局强契约

以下内容属于当前主链路的强契约。

- 主链路顺序固定为 `segment -> prompt -> infer -> merge -> slam -> eval -> stats`
- `segment/keyframes/keyframes_meta.json` 是 raw keyframe schedule 的事实源
- `segment/deploy_schedule.json` 是当前 Wan 执行投影；它不能回写覆盖 raw schedule
- `prompt` 当前默认只承诺输出 `profile.json` 与 `final_prompt.json`
- `infer` 当前默认从 `prompt/final_prompt.json` 读取 `prompt / negative_prompt`
- `merge` 必须按 `infer/runs_plan.json` 的真实边界合并，不能再假设全局固定 `kf_gap`
- `segment / prompt / infer / merge` 应持续写出稳定的 `step_meta.json`；`slam` 轨道元数据当前写在 `run_meta.json`
- 下游阶段只能消费上游结果，不能回写上游目录

## 2. 阶段依赖总览

| 阶段 | 主要脚本 | 关键输入 | 关键输出 | 下游依赖 |
|---|---|---|---|---|
| `segment` | `scripts/segment_make.py` | 数据集、标定、`platform.yaml` phase | `segment/frames/`, `segment/keyframes/keyframes_meta.json`, `segment/deploy_schedule.json`, `timestamps.txt`, `calib.txt`, `preprocess_meta.json`, `step_meta.json` | `prompt`, `infer`, `slam`, `segment_analyze` |
| `prompt` | `scripts/prompt_gen.py` | `segment/frames/` | `prompt/profile.json`, `prompt/final_prompt.json`, `prompt/step_meta.json` | `infer`, `stats` |
| `infer` | `scripts/infer_i2v.py` | `segment/frames/`, `prompt/final_prompt.json`, `segment/deploy_schedule.json` | `infer/execution_plan.json`, `infer/runs/`, `infer/runs_plan.json`, `infer/step_meta.json` | `merge`, `stats` |
| `merge` | `scripts/merge_seq.py` | `infer/runs_plan.json`, `infer/runs/*`, `segment/calib.txt`, `segment/timestamps.txt` | `merge/frames/`, `merge/timestamps.txt`, `merge/calib.txt`, `merge/merge_meta.json`, `merge/step_meta.json` | `slam`, `stats` |
| `slam` | `scripts/slam_droid.py` | `segment/` 或 `merge/` 轨道数据 | `slam/<track>/traj_est.tum`, `traj_est.npz`, `run_meta.json` | `eval` |
| `eval` | `scripts/eval_main.py` | `slam/ori/traj_est.tum`, `slam/gen/traj_est.tum`, `merge/frames/`, `segment/frames/` | `eval/traj_metrics.json`, `eval/image_metrics.json`, `eval/slam_metrics.json`, `eval/summary.txt`, `eval/image_per_frame.csv`, `eval/slam_pairs.csv`, `eval/plots/*.png` | 人工分析 |
| `stats` | `scripts/stats_collect.py` | 各阶段 `step_meta.json` 与日志 | `stats/report.json`, `stats/compression.json` | 汇总出口 |

## 3. 各阶段最小契约

### `segment`

必须保证：

- `segment/frames/` 可供 `prompt` 与 `infer` 直接读取
- `segment/keyframes/keyframes_meta.json` 存在，并表达正式 raw keyframe 结果
- `segment/deploy_schedule.json` 存在时，`infer` 会优先使用它
- `segment/step_meta.json` 至少能支撑 `stats/report.json` 的压缩字段汇总

当前正式 `segment_policy` 口径是：

- `uniform`
- `motion`
- `semantic`

### `prompt`

必须保证：

- 输入来自 `segment/frames/`
- 输出至少包含 `profile.json` 与 `final_prompt.json`
- `final_prompt.json` 中有可直接被 `infer` 消费的 `prompt`
- `step_meta.json` 记录 backend、采样方式、代表帧与输出摘要

默认行为：

- 默认 backend 为 `smolvlm2`
- 默认采样口径为 `even + 5 images`
- 代表帧数量会被收敛到 `3..5`

### `infer`

必须保证：

- 优先从 `segment/deploy_schedule.json` 构建 execution segments
- 缺失 deploy schedule 时，明确记录回退到 `legacy_kf_gap`
- `runs_plan.json` 保存真实的 `start_idx / end_idx / num_frames`
- `step_meta.json` 记录 backend、phase、schedule_source 与 runs plan 摘要

兼容要求：

- 当前默认输入是 `prompt/final_prompt.json`
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
- 结构化输出至少包含 `eval/traj_metrics.json`、`eval/image_metrics.json`、`eval/slam_metrics.json` 与 `eval/summary.txt`
- 轨迹图像输出收敛到 `eval/plots/`，当前至少包含 `traj_xy.png`、`ape_curve.png`、`rpe_curve.png`
- 当 `segment/keyframes/keyframes_meta.json` 可用时，`eval/plots/` 中的主轨迹图与各类曲线图会以轻量标记标出最终关键帧位置；缺失时只降级为不标记，不影响图生成
- `traj_xy.png` 保留历史文件名，但其主图语义已是“主二维投影视图”，不再强绑定真实世界固定 XY 平面
- 图像评价默认统计 `infer -> merge` 生成帧与对应 `ori` 帧的逐帧比较，新增 `eval/image_metrics.json`、`eval/image_per_frame.csv` 与 `eval/plots/image_metrics_curve.png`
- SLAM-friendly 图像评价默认只统计时间上连续的相邻生成帧对，输出 `eval/slam_metrics.json`、`eval/slam_pairs.csv` 与 `eval/plots/slam_metrics_curve.png`
- `traj_metrics.json` 作为轨迹评价事实源，至少包含 APE translation、RPE translation、matched pose 数、`ori_path_length_m`、`gen_path_length_m`、`eval_status` 与 `warnings`
- `image_metrics.json` 作为图像评价事实源，至少包含 `psnr`、`ms_ssim`、`lpips` 的聚合统计、`frame_count`、`eval_status` 与 `warnings`
- `slam_metrics.json` 作为两视图几何型 SLAM-friendly 评价事实源，至少包含 `inlier_ratio`、`pose_success_rate`、`reference_source`、`uses_proxy_reference`、`valid_pair_count`、`valid_pose_pair_count` 与 `warnings`
- `summary.txt` 汇总轨迹与图像评价的人类可读摘要，不再单独输出 `image_summary.txt`
- 缺少轨迹文件或 `evo` 依赖时不能让主链路崩溃，应写出可理解的失败摘要，而不是只留下零散 txt

### `stats`

必须保证：

- 输出 `stats/report.json`
- 保留 `stats/compression.json` 兼容历史消费
- 对缺失的 `step_meta.json` 给出 `WARN`，而不是直接崩溃
- 实验结束后终端统一打印单块 `EXPERIMENT REPORT`，汇总 time / quality / compression 三类关键指标；其中 quality 区块当前也包含 `ori_path_length_m` / `gen_path_length_m` 的展示

## 4. `segment_analyze` 的旁路契约

`segment_analyze.py` 不是主链路阶段，但当前系统默认会在 `segment` 后触发。

它的边界是：

- 只读取 `segment/` 既有产物
- 只写 `segment/analysis/` 和 `segment/.segment_cache/`
- 失败时只能 warn-only，不能破坏主链路

当前正式分析产物收敛为：

- `segment_summary.json`
- `segment_timeseries.csv`
- `kinematics_overview.png`
- `allocation_overview.png`
- `comparison_overview.png`
- `projection_overview.png`

## 5. 最小验收检查点

对主链路与文档描述的一致性，至少要满足以下检查。

- `python -m exphub --mode doctor ...` 返回 `PASS`
- 当前默认命令能读到 `prompt/final_prompt.json`，而不是旧 manifest 文档
- `infer/runs_plan.json` 中的边界与 `merge` 实际合并边界一致
- `stats/report.json` 能从各阶段 `step_meta.json` 汇总出结果
- 日志前缀遵守 [LOGGING.md](./LOGGING.md) 中的约定

快速静态验收命令：

```bash
python -m py_compile exphub/*.py scripts/*.py
python -m exphub --mode doctor --dataset <ds> --sequence <seq> --tag <tag> --w <w> --h <h> --fps <fps> --dur <dur> --kf_gap <gap>
```
