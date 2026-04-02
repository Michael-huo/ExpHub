# ExpHub Re-Baseline Step 3

本文只说明 Step 3 收口后的正式主链闭环，不回补旧文档。

## 正式 workflow

当前正式 workflow 固定为：

`segment -> prompt -> infer -> merge -> slam -> eval -> stats`

正式唯一入口链路为：

`python -m exphub -> exphub/cli.py -> exphub/pipeline/orchestrator.py -> exphub/pipeline/*/service.py`

## 每个 stage 的唯一正式入口

- `segment` -> `exphub.pipeline.segment.service.run(...)`
- `prompt` -> `exphub.pipeline.prompt.service.run(...)`
- `infer` -> `exphub.pipeline.infer.service.run(...)`
- `merge` -> `exphub.pipeline.merge.service.run(...)`
- `slam` -> `exphub.pipeline.slam.service.run(...)`
- `eval` -> `exphub.pipeline.eval.service.run(...)`
- `stats` -> `exphub.pipeline.stats.service.run(...)`

## 关键正式契约

- `segment` -> `exphub/contracts/segment.py`
- `prompt` -> `exphub/contracts/prompt.py`
- `infer` -> `exphub/contracts/infer.py`
- `merge` -> `exphub/contracts/merge.py`
  - 正式上游：`infer/runs_plan.json`
  - 正式输出：`merge/merge_manifest.json`、`merge/report.json`、`merge/frames/`
- `slam` -> `exphub/contracts/slam.py`
  - 正式上游：`merge/merge_manifest.json`
  - 正式输出：`slam/report.json`、`slam/traj_est.txt`、`slam/ori|gen/*`
- `eval` -> `exphub/contracts/eval.py`
  - 正式输入：`slam/report.json`、`merge/merge_manifest.json`、`infer/runs_plan.json`
  - 正式输出：`eval/report.json`、`eval/metrics/*.json`、`eval/plots/*`
- `stats` -> `exphub/contracts/stats.py`
  - 正式输入：各 stage `report.json`
  - 正式输出：`stats/final_report.json`

## 已退出正式 workflow 的旧脚本入口

以下旧入口已经退出正式 workflow，并已从主链移除：

- `scripts/merge_seq.py`
- `scripts/slam_droid.py`
- `scripts/eval_main.py`
- `scripts/eval_traj.py`
- `scripts/stats_collect.py`
- `scripts/_eval/`

## 仍保留但已退出正式主链的旧目录

以下目录仍可能保留用于 sidecar / research / archive，但默认正式 workflow 不再触达：

- `scripts/_segment/`
- `scripts/_prompt/`
- `scripts/_infer/`
- `segment/signal_extraction/`
- `segment/analysis/`
- `docs/archive/`
