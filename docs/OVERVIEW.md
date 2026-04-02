# ExpHub 概览

> 本文回答什么问题：ExpHub 当前正式主链是什么、代码放在哪里、最关键的输入输出和运行入口有哪些。

ExpHub 是一个面向视频流与 VSLAM 实验的正式流水线壳。当前正式实现集中在：

- `exphub/common/`
- `exphub/contracts/`
- `exphub/pipeline/`

统一入口是：

`python -m exphub`

顶层调用链固定为：

`exphub/__main__.py -> exphub/cli.py -> exphub/pipeline/orchestrator.py -> exphub/pipeline/<stage>/service.py`

阶段顺序固定为：

`segment -> prompt -> infer -> merge -> slam -> eval -> stats`

## 当前主链口径

当前默认行为按下面理解即可：

- `segment` 负责产出标准帧序列、关键帧事实源、deploy schedule 与 state 区间
- `prompt` 固定使用 `smolvlm2`，对下游输出 `runtime_prompt_plan.json`，并保留 `base_prompt.json`、`state_prompt_manifest.json`、`report.json` 作为阶段内部支撑与追溯产物
- `infer` 直接消费 `prompt/runtime_prompt_plan.json`，产出 `runs_plan.json` 与 `report.json`
- `merge` 只按 `infer/runs_plan.json` 的真实边界拼接
- `slam` 在 `ori` 与 `gen` 两条轨道上估计位姿，对外正式聚合出口是 `slam/report.json` 与 `slam/traj_est.txt`
- `eval` 汇总轨迹、图像与 slam-friendly 指标
- `stats` 汇总阶段报告、压缩统计与实验摘要，正式输出为 `stats/final_report.json`

如果只记最关键的事实源，请先记住：

- `segment/keyframes/keyframes_meta.json` 是 raw keyframe 事实源
- `segment/deploy_schedule.json` 是执行投影
- `segment/state_segmentation/state_segments.json` 是 state 区间事实源
- `prompt/runtime_prompt_plan.json` 是 `infer` 唯一正式 prompt 输入
- `prompt/base_prompt.json` 与 `prompt/state_prompt_manifest.json` 不是下游正式契约
- `infer/runs_plan.json` 是 `merge` 的真实执行边界来源

## 代码与配置边界

- `exphub/common` 放通用能力、路径、日志和子进程调度
- `exphub/contracts` 放阶段契约与产物约定
- `exphub/pipeline` 放正式阶段实现
- `config/platform.yaml` 配置 phase Python、模型和外部仓库路径
- `config/datasets.json` 配置数据集与序列入口

跨环境解释器从 `config/platform.yaml` 的 `environments.phases.<phase>.python` 读取。当前正式主链常见 phase 包括：

- `segment`
- `prompt_smol`
- `infer`
- `infer_fun_5b`
- `slam`

## 常见运行与排障入口

- 统一入口：`python -m exphub`
- 主链自检：`python -m exphub --mode doctor ...`
- 完整日志目录：`EXP_DIR/logs/`
- 阶段输出目录：`segment/`、`prompt/`、`infer/`、`merge/`、`slam/`、`eval/`、`stats/`

## 文档导航

- [PIPELINE_CONTRACT.md](./PIPELINE_CONTRACT.md)：各阶段输入输出、正式产物与强契约
- [LOGGING.md](./LOGGING.md)：运行入口、配置位置、日志前缀与落盘规则

## 推荐阅读顺序

1. 先读本文
2. 再读 [PIPELINE_CONTRACT.md](./PIPELINE_CONTRACT.md)
3. 需要运行、排障或看日志时再读 [LOGGING.md](./LOGGING.md)
