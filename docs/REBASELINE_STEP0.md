# ExpHub Repo Re-Baseline Step 0

本文只说明 repo re-baseline refactor 的 Step 0 落地结果，不覆盖后续深迁移设计。

## 新正式入口

当前正式主链入口已经收口为：

`python -m exphub -> exphub/cli.py -> exphub/pipeline/orchestrator.py -> exphub/pipeline/<stage>/service.py`

各 stage 的唯一正式入口壳位于：

- `exphub/pipeline/segment/service.py`
- `exphub/pipeline/prompt/service.py`
- `exphub/pipeline/infer/service.py`
- `exphub/pipeline/merge/service.py`
- `exphub/pipeline/slam/service.py`
- `exphub/pipeline/eval/service.py`
- `exphub/pipeline/stats/service.py`

## Step 0 做了什么

- 建立了新的正式骨架：`exphub/common/`、`exphub/contracts/`、`exphub/pipeline/`、`exphub/tools/`
- 将 CLI 的 workflow 组织职责下沉到 `exphub/pipeline/orchestrator.py`
- 为每个 stage 建立了唯一正式 `service.py` 入口壳
- 将配置、路径、基础 IO、日志门面、子进程调度、共享类型抽到 `exphub/common/`
- 将阶段输入输出契约位置固定到 `exphub/contracts/`
- 保留旧 `scripts/*.py` / `scripts/_*/` 实现作为 Step 0 的临时桥接目标，避免提前进入 Step 1/2/3 的深迁移

## 后续 Step 1 / 2 / 3

Step 1 会继续推进 `segment` 正式主线迁移，把更多正式逻辑从旧脚本层收口到 `exphub/pipeline/segment/` 与对应 contracts/common 能力。

Step 2 会处理 `prompt / infer` 的 legacy 去耦，把正式 prompt plan / infer plan 生成与消费逻辑进一步从旧脚本桥接迁到新结构。

Step 3 会继续整理 `merge / slam / eval / stats` 尾部阶段，并处理旧入口与文档的进一步清仓，但不属于本步交付范围。
