# ExpHub 概览

> 本文回答什么问题：ExpHub 是什么、当前默认主链路是什么、现在哪些目录才是正式实现。

ExpHub 是一个面向视频流与 VSLAM 实验的平台化调度壳。当前仓库已经完成主链收口，正式实现集中在：

- `exphub/common/`
- `exphub/contracts/`
- `exphub/pipeline/`

统一执行链路固定为：

`cli -> orchestrator -> exphub/pipeline/<stage>/service.py`

阶段顺序固定为：

`segment -> prompt -> infer -> merge -> slam -> eval -> stats`

## 当前默认主链路

当前默认口径是：

- `segment` 正式策略只保留 `uniform` 与 `state`，其中 `state` 是当前正式研究主线
- `state` 正式输入只保留 `motion_velocity` 与 `semantic_velocity`
- `prompt` 默认输出 `base_prompt.json`、`state_prompt_manifest.json`、`runtime_prompt_plan.json`、`report.json`
- `infer` 默认消费 `prompt/runtime_prompt_plan.json`，默认输出 `runs_plan.json` 与 `report.json`
- `merge` 只按 `infer/runs_plan.json` 的真实边界拼接
- `eval` 默认收敛到 `eval/report.json`、`eval/details.csv` 与 `eval/plots/`
- `stats` 汇总主链阶段报告与压缩统计

如果你只需要先抓住当前系统，记住这几件事就够了：

- `segment/keyframes/keyframes_meta.json` 是 raw keyframe 事实源
- `segment/deploy_schedule.json` 是执行投影，不回写 raw schedule
- `segment/state_segmentation/state_segments.json` 是 state 区间事实源
- `prompt/runtime_prompt_plan.json` 是 `infer` 唯一正式 prompt 输入
- `infer/runs_plan.json` 是 `merge` 的真实执行边界来源

## 目录口径

当前仓库目录约束是：

- `exphub/common` 放通用能力
- `exphub/contracts` 放阶段契约与路径约定
- `exphub/pipeline` 放正式阶段实现
- `tools` 只应放非正式主链工具
- 不再保留并行的旧 `scripts/_prompt`、`scripts/_infer`、`scripts/_segment` 正式实现树

## 文档导航

- [ARCHITECTURE.md](./ARCHITECTURE.md)：分层、调度、目录边界
- [PIPELINE_CONTRACT.md](./PIPELINE_CONTRACT.md)：各阶段输入输出与强契约
- [LOGGING.md](./LOGGING.md)：日志前缀、终端/落盘分流、性能心跳
- [PROMPT_PROFILE_SYSTEM.md](./PROMPT_PROFILE_SYSTEM.md)：PromptProfile 与 runtime prompt plan
- [RESEARCH_DEV_GUIDE.md](./RESEARCH_DEV_GUIDE.md)：研究目标、开发协作与实验口径
- [TITS_METHODOLOGY.md](./TITS_METHODOLOGY.md)：论文目标与方法映射
- [LEGACY_PURGE.md](./LEGACY_PURGE.md)：本轮 legacy purge / repo slim-down 摘要

## 推荐阅读顺序

1. 先读本文
2. 再读 [ARCHITECTURE.md](./ARCHITECTURE.md)
3. 再读 [PIPELINE_CONTRACT.md](./PIPELINE_CONTRACT.md)
4. 按任务需要补读 [LOGGING.md](./LOGGING.md)、[PROMPT_PROFILE_SYSTEM.md](./PROMPT_PROFILE_SYSTEM.md)、[RESEARCH_DEV_GUIDE.md](./RESEARCH_DEV_GUIDE.md)
5. 如果任务涉及方法设计或论文贡献映射，再读 [TITS_METHODOLOGY.md](./TITS_METHODOLOGY.md)
