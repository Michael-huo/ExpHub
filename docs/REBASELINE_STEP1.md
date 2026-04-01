# Re-Baseline Step 1

本说明只覆盖 Step 1 的正式 `segment` re-baseline，不涉及 `prompt / infer / merge / slam / eval / stats` 的内部去 legacy 化。

## 正式入口

- 正式 `segment` 唯一入口是 `exphub.pipeline.segment.service.run(runtime)`
- orchestrator 现在只通过该 service 调度 `segment`
- 正式 workflow 已不再通过 `scripts/segment_make.py` 作为入口

## 正式产物

- 唯一正式上游契约：`segment/segment_manifest.json`
- 正式阶段报告：`segment/report.json`
- 正式可视化目录：`segment/visuals/`
- 兼容性产物仍保留：
  - `segment/keyframes/keyframes_meta.json`
  - `segment/deploy_schedule.json`
  - `segment/state_segmentation/state_segments.json`
  - `segment/state_segmentation/state_report.json`

## 已退出正式 Workflow 的旧路径

- `scripts/segment_make.py` 不再是正式入口
- `scripts/_segment/app.py`、`scripts/_segment/api.py`、`scripts/_segment/make.py`、`scripts/_segment/materialize.py` 不再承担正式入口职责
- `research / analysis / uniform` 不再作为正式 `segment` workflow 的公开口径
- `segment/signal_extraction/` 不再是正式下游契约目录

## Step 2 接续方式

- Step 2 的 `prompt` 重构应以 `segment/segment_manifest.json` 为唯一上游入口
- `segment_manifest.json` 内已嵌入：
  - state 区间契约
  - deploy schedule 契约
  - 正式产物路径索引
- 后续可以在不依赖旧 `state_segmentation/state_segments.json` 直读的前提下继续推进 prompt 主线收口
