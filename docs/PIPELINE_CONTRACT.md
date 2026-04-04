# ExpHub 流水线契约

> 本文回答什么问题：每个阶段读取什么、写出什么、阶段之间靠哪些文件衔接、哪些契约不能被随意破坏。

运行入口、配置位置与日志规则请看 [LOGGING.md](./LOGGING.md)。

## 1. 全局强契约

当前主链路的强契约是：

- 主链顺序固定为 `segment -> prompt -> infer -> merge -> slam -> eval -> stats`
- `segment/segment_manifest.json` 是 `segment` 正式事实源，并内嵌 `deploy_schedule`、`state_segments` 与 `state_report`
- `segment/keyframes/` 与 `segment_manifest.json.keyframes` 共同定义正式 keyframe 集合
- `segment/visuals/state_overview.png` 是唯一正式 state 总览图
- `prompt` 对下游唯一正式 prompt 契约是 `prompt/runtime_prompt_plan.json`
- `prompt` 会同时写出 `base_prompt.json` 与 `state_prompt_manifest.json` 作为阶段内部支撑与追溯产物
- `prompt` 正式 backend 固定为 `smolvlm2`
- `infer` 正式消费 `prompt/runtime_prompt_plan.json`
- `infer` 正式输出 `runs_plan.json` 与 `report.json`
- `merge` 必须按 `infer/runs_plan.json` 的真实边界拼接
- 下游阶段只能消费上游结果，不能回写上游目录

## 2. 阶段依赖总览

| 阶段 | 正式实现 | 关键输入 | 关键输出 | 下游依赖 |
|---|---|---|---|---|
| `segment` | `exphub/pipeline/segment/service.py` | 数据集、标定、phase Python | `segment/frames/`, `segment/keyframes/`, `segment/segment_manifest.json`, `segment/report.json`, `segment/visuals/state_overview.png`, `timestamps.txt`, `calib.txt` | `prompt`, `infer`, `slam` |
| `prompt` | `exphub/pipeline/prompt/service.py` | `segment/frames/`, `segment/segment_manifest.json` | `prompt/runtime_prompt_plan.json`, `prompt/report.json`, 以及内部支撑产物 `prompt/base_prompt.json`、`prompt/state_prompt_manifest.json` | `infer`, `stats` |
| `infer` | `exphub/pipeline/infer/service.py` | `segment/frames/`, `prompt/runtime_prompt_plan.json` | `infer/runs/`, `infer/runs_plan.json`, `infer/report.json` | `merge`, `stats` |
| `merge` | `exphub/pipeline/merge/service.py` | `infer/runs_plan.json`, `infer/runs/*`, `segment/calib.txt`, `segment/timestamps.txt` | `merge/frames/`, `merge/timestamps.txt`, `merge/calib.txt`, `merge/merge_meta.json`, `merge/step_meta.json` | `slam`, `stats` |
| `slam` | `exphub/pipeline/slam/service.py` | `segment/` 或 `merge/` 轨道数据 | `slam/report.json`, `slam/traj_est.txt`, 以及 `slam/<track>/traj_est.tum`, `slam/<track>/traj_est.npz`, `slam/<track>/run_meta.json` | `eval` |
| `eval` | `exphub/pipeline/eval/service.py` | `slam/report.json` 中声明的正式轨迹路径、`merge/frames/`、`segment/frames/` | `eval/report.json`, `eval/details.csv`, `eval/plots/traj_xy.png`, `eval/plots/metrics_overview.png` | 人工分析 |
| `stats` | `exphub/pipeline/stats/service.py` | `segment/report.json`, `prompt/report.json`, `infer/report.json`, `merge/step_meta.json` 与日志 | `stats/final_report.json`, `stats/compression.json` | 汇总出口 |

## 3. 各阶段最小契约

### `segment`

- 必须写出 `segment/frames/`
- 必须写出 `segment/keyframes/`
- 必须写出 `segment/segment_manifest.json`
- 必须写出 `segment/report.json`
- 必须写出 `segment/visuals/state_overview.png`
- 当前 `state` 正式内部实现链固定为 `service.py -> state/detector.py -> state/policies/state.py -> state/signal_extraction/extract.py -> state/state_segmentation/formal.py`
- `state/observed_signals/` 只承载正式链内部使用的原始观测后端，不是独立研究入口
- `state` 正式输入只保留 `motion_velocity` 与 `semantic_velocity`
- `segment_manifest.json` 内嵌 `deploy_schedule`、`state_segments` 与 `state_report`
- `state_segments` 与 `deploy_schedule` 的正式下游读取链收敛到 `segment_manifest.json`

### `prompt`

- 输入来自 `segment/frames/`
- backend 固定为 `smolvlm2`
- 对下游唯一正式 prompt 契约文件是 `runtime_prompt_plan.json`
- `report.json` 是阶段报告，不参与下游 prompt 契约
- `base_prompt.json` 与 `state_prompt_manifest.json` 只作为 prompt 阶段内部支撑与追溯产物保留
- `runtime_prompt_plan.json` 负责按 `segment_manifest.json` 内嵌 deploy schedule 展开可直接执行的 prompt plan

### `infer`

- 正式 prompt 输入固定为 `prompt/runtime_prompt_plan.json`
- execution segments 来自 `prompt/runtime_prompt_plan.json`
- deploy schedule 缺失时允许回退到 `fallback_kf_gap`
- 不再在前端重拼 prompt 文本
- `runs_plan.json` 必须保存真实 `start_idx / end_idx / num_frames`

### `merge`

- 只按 `runs_plan.json` 的真实边界去重与拼接
- `merge/frames/` 与 `merge/timestamps.txt` 必须一一对应

### `slam`

- `ori` 轨读取 `segment/`
- `gen` 轨读取 `merge/`
- 正式聚合输出保留 `slam/report.json` 与 `slam/traj_est.txt`
- `eval` 仍依赖每个轨道的 `traj_est.tum`、`traj_est.npz` 与 `run_meta.json`

### `eval`

- 输出至少包含聚合后的 `eval/report.json` 与 `eval/details.csv`
- 图像输出收敛到 `eval/plots/`
- 缺少轨迹文件或依赖时不能让主链崩溃，应写出可理解的失败摘要

### `stats`

- 正式输出为 `stats/final_report.json`
- 保留 `stats/compression.json`，因为 `exphub/cli.py` 的实验摘要仍会读取它
- `segment` 压缩统计来自 `segment/report.json`
- 对缺失上游报告给出 `WARN`，而不是直接崩溃

## 4. 非正式内容边界

任何研究性、一次性或人工分析工具都不应重新成为正式阶段依赖，也不应回写主链事实源。

对 `segment` 而言，这也包括：

- 不要把 `state/signal_extraction/` 当成独立阶段入口理解；当前正式主链只消费其内联提取结果
- 不要把 `state/state_segmentation/` 当成多实现试验场；当前正式主链固定使用 `formal.py`

## 5. 最小验收检查点

- `python -m exphub --mode doctor ...` 返回 `PASS`
- 当前默认命令能读到 `prompt/runtime_prompt_plan.json`
- `infer/runs_plan.json` 中的边界与 `merge` 实际边界一致
- `stats/final_report.json` 能从各阶段报告汇总结果
- 日志前缀遵守 [LOGGING.md](./LOGGING.md)

快速静态检查：

```bash
python -m py_compile $(find exphub -name "*.py")
python -m exphub --mode doctor --dataset <ds> --sequence <seq> --tag <tag> --w <w> --h <h> --fps <fps> --dur <dur> --start_sec <start>
```
