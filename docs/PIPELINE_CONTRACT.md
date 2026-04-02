# ExpHub 流水线契约

> 本文回答什么问题：每个阶段读取什么、写出什么、阶段之间靠哪些文件衔接、哪些契约不能被随意破坏。

运行入口、配置位置与日志规则请看 [LOGGING.md](./LOGGING.md)。

## 1. 全局强契约

当前主链路的强契约是：

- 主链顺序固定为 `segment -> prompt -> infer -> merge -> slam -> eval -> stats`
- `segment/keyframes/keyframes_meta.json` 是 raw keyframe 事实源
- `segment/deploy_schedule.json` 是执行投影，不回写 raw schedule
- `segment/state_segmentation/state_segments.json` 是 state 区间事实源
- `prompt` 正式输出 `base_prompt.json`、`state_prompt_manifest.json`、`runtime_prompt_plan.json`、`report.json`
- `prompt` 正式 backend 固定为 `smolvlm2`
- `infer` 正式消费 `prompt/runtime_prompt_plan.json`
- `infer` 正式输出 `runs_plan.json` 与 `report.json`
- `merge` 必须按 `infer/runs_plan.json` 的真实边界拼接
- 下游阶段只能消费上游结果，不能回写上游目录

## 2. 阶段依赖总览

| 阶段 | 正式实现 | 关键输入 | 关键输出 | 下游依赖 |
|---|---|---|---|---|
| `segment` | `exphub/pipeline/segment/service.py` | 数据集、标定、phase Python | `segment/frames/`, `segment/keyframes/keyframes_meta.json`, `segment/deploy_schedule.json`, `segment/state_segmentation/*`, `timestamps.txt`, `calib.txt`, `preprocess_meta.json`, `step_meta.json` | `prompt`, `infer`, `slam` |
| `prompt` | `exphub/pipeline/prompt/service.py` | `segment/frames/`, `segment/state_segmentation/state_segments.json`, `segment/deploy_schedule.json` | `prompt/base_prompt.json`, `prompt/state_prompt_manifest.json`, `prompt/runtime_prompt_plan.json`, `prompt/report.json` | `infer`, `stats` |
| `infer` | `exphub/pipeline/infer/service.py` | `segment/frames/`, `prompt/runtime_prompt_plan.json`, `segment/deploy_schedule.json` | `infer/runs/`, `infer/runs_plan.json`, `infer/report.json` | `merge`, `stats` |
| `merge` | `exphub/pipeline/merge/service.py` | `infer/runs_plan.json`, `infer/runs/*`, `segment/calib.txt`, `segment/timestamps.txt` | `merge/frames/`, `merge/timestamps.txt`, `merge/calib.txt`, `merge/merge_meta.json`, `merge/step_meta.json` | `slam`, `stats` |
| `slam` | `exphub/pipeline/slam/service.py` | `segment/` 或 `merge/` 轨道数据 | `slam/<track>/traj_est.tum`, `traj_est.npz`, `run_meta.json` | `eval` |
| `eval` | `exphub/pipeline/eval/service.py` | `slam/ori/traj_est.tum`, `slam/gen/traj_est.tum`, `merge/frames/`, `segment/frames/` | `eval/report.json`, `eval/details.csv`, `eval/plots/traj_xy.png`, `eval/plots/metrics_overview.png` | 人工分析 |
| `stats` | `exphub/pipeline/stats/service.py` | `segment/step_meta.json`, `prompt/report.json`, `infer/report.json`, `merge/step_meta.json` 与日志 | `stats/report.json`, `stats/compression.json` | 汇总出口 |

## 3. 各阶段最小契约

### `segment`

- 必须写出 `segment/frames/`
- 必须写出 `segment/keyframes/keyframes_meta.json`
- 必须写出 `segment/deploy_schedule.json`
- 必须写出 `segment/state_segmentation/state_segments.json`
- `state` 正式输入只保留 `motion_velocity` 与 `semantic_velocity`
- `segment/step_meta.json` 需要能支撑 `stats` 汇总

### `prompt`

- 输入来自 `segment/frames/`
- backend 固定为 `smolvlm2`
- 输出至少包含 `base_prompt.json`、`state_prompt_manifest.json`、`runtime_prompt_plan.json`、`report.json`
- `runtime_prompt_plan.json` 是 `infer` 的唯一正式 prompt 输入文件
- `base_prompt.json` 负责全局约束
- `state_prompt_manifest.json` 负责 state 区间 prompt manifest
- `runtime_prompt_plan.json` 负责按 deploy schedule 展开可直接执行的 prompt plan

### `infer`

- 优先从 `segment/deploy_schedule.json` 构建 execution segments
- deploy schedule 缺失时允许回退到 `fallback_kf_gap`
- 不再在前端重拼 prompt 文本
- `runs_plan.json` 必须保存真实 `start_idx / end_idx / num_frames`

### `merge`

- 只按 `runs_plan.json` 的真实边界去重与拼接
- `merge/frames/` 与 `merge/timestamps.txt` 必须一一对应

### `slam`

- `ori` 轨读取 `segment/`
- `gen` 轨读取 `merge/`
- 每个轨道至少产出 `traj_est.tum` 与 `run_meta.json`

### `eval`

- 输出至少包含聚合后的 `eval/report.json` 与 `eval/details.csv`
- 图像输出收敛到 `eval/plots/`
- 缺少轨迹文件或依赖时不能让主链崩溃，应写出可理解的失败摘要

### `stats`

- 输出 `stats/report.json`
- 保留 `stats/compression.json` 兼容历史消费
- 对缺失上游报告给出 `WARN`，而不是直接崩溃

## 4. 非正式内容边界

任何研究性、一次性或人工分析工具都不应重新成为正式阶段依赖，也不应回写主链事实源。

## 5. 最小验收检查点

- `python -m exphub --mode doctor ...` 返回 `PASS`
- 当前默认命令能读到 `prompt/runtime_prompt_plan.json`
- `infer/runs_plan.json` 中的边界与 `merge` 实际边界一致
- `stats/report.json` 能从各阶段报告汇总结果
- 日志前缀遵守 [LOGGING.md](./LOGGING.md)

快速静态检查：

```bash
python -m py_compile $(find exphub -name "*.py")
python -m exphub --mode doctor --dataset <ds> --sequence <seq> --tag <tag> --w <w> --h <h> --fps <fps> --dur <dur> --start_sec <start>
```
