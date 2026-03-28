# ExpHub 概览

> 本文回答什么问题：ExpHub 是什么、当前默认主链路是什么、文档应该从哪里开始读。

ExpHub 是一个面向视频流与 VSLAM 实验的平台化调度外壳。它把关键帧抽取、语义提示生成、视频恢复、SLAM 与评测串成统一实验闭环，同时把不同阶段的 Python 解释器、模型路径和外部仓库配置收口到 `config/platform.yaml`。

## 当前默认主链路

标准执行链路固定为：

`segment -> prompt -> infer -> merge -> slam -> eval -> stats`

当前主链路的默认口径是：

- `segment` 当前正式 policy 只保留 `uniform` 与 `state`，其中 `state` 是当前正式研究主线
- `state` 当前正式主线只使用两类输入信号：`motion_velocity`、`semantic_velocity`；`blur_score`、`appearance_delta` 等其它已提取信号只保留为 analysis / validation sidecar 观察，不进入正式 state score。当前正式 `state_score` 由这两条 processed signal 按固定权重生成，并只做一层轻量平滑，然后交给正式 high-risk interval detector 识别 `low_state / high_state` 区间序列
- `prompt` 默认使用 `smolvlm2`，默认收敛产物到 `prompt/final_prompt.json`、`prompt/state_prompt_manifest.json`、`prompt/deploy_to_state_prompt_map.json` 与 `prompt/report.json`
- `infer` 默认使用 `wan_fun_5b_inp`，优先消费 `segment/deploy_schedule.json`，并默认收敛产物到 `infer/runs_plan.json`、`infer/prompt_manifest_resolved.json` 与 `infer/report.json`
- `segment` 当前正式主线默认只保留 `segment/state_segmentation/` 三件套：`state_report.json`、`state_overview.png`、`state_segments.json`。正式 `state` policy 仍会内部提取 `motion_velocity` 与 `semantic_velocity` 这两条 formal inputs，但不再把 `segment/signal_extraction/` 作为默认正式产物目录
- `segment/state_segmentation/` 当前正式目标是直接做高风险区间检测：正式 score 只基于 `motion_velocity` 与 `semantic_velocity`，再用最小 online / changepoint-style detector 识别 `low_state / high_state` 区间序列；当前 detector 采用 regime-shift 风格的因果 local-mean / local-spread 状态响应，而不是逐波峰事件响应；`state_segments.json` 继续作为 state 区间事实源
- `eval` 会默认收敛评测产物到 `eval/report.json`、`eval/details.csv`、`eval/plots/traj_xy.png` 与 `eval/plots/metrics_overview.png`，不再默认散落多份独立 metrics json/csv/curve png
- `signal_extract` / `analyze` 仍保留为显式 research sidecar 入口，但默认正式主链不再自动触发它们，也不再依赖它们写出正式 segment 产物

如果你只需要把握当前系统，先记住三个核心事实：

- `segment/keyframes/keyframes_meta.json` 是 raw keyframe 事实源
- `segment/deploy_schedule.json` 是当前 Wan 执行投影；`infer` 优先消费它，但它不能回写覆盖 raw schedule
- `segment/state_segmentation/state_segments.json` 是 state 区间事实源；`prompt` 基于它生成 `state_prompt_manifest.json`
- 当前默认 `state_segments.json` 语义是兼容下游的 `low_state / high_state` 区间序列；当前样例中某个 `high_state` 可能对应 turning high-risk interval，但正式主线不再绑定单一模板
- `prompt/final_prompt.json` 是 infer prompt 的 base scene 输入
- `prompt/state_prompt_manifest.json` 是按 state 区间生成的局部 motion prompt
- `prompt/deploy_to_state_prompt_map.json` 只负责把 execution segment 映射到 state prompt，不直接生成新 prompt
- `prompt/report.json` 是 prompt 默认元信息与摘要的聚合出口
- `infer/prompt_manifest_resolved.json` 是 infer 运行时真正消费的派生 prompt manifest；它会把 global prompt 与 state local prompt 对齐到 execution segments
- `infer/report.json` 是 infer 默认元信息与执行摘要的聚合出口
- `segment/signal_extraction/`、`segment/analysis/` 与其它 research sidecar 产物只属于显式旁路分析，不回写主链 schedule 或 raw keyframe 事实源，也不属于当前默认正式 segment 输出

## 核心文档导航

- [ARCHITECTURE.md](./ARCHITECTURE.md)：系统分层、phase 调度、实验目录与模块边界
- [PIPELINE_CONTRACT.md](./PIPELINE_CONTRACT.md)：每个阶段读什么、写什么、哪些文件属于强契约
- [RESEARCH_DEV_GUIDE.md](./RESEARCH_DEV_GUIDE.md)：研究目标、实验组织方式、推荐开发流程
- [TITS_METHODOLOGY.md](./TITS_METHODOLOGY.md)：当前 T-ITS 论文的方法论目标、学术叙事与模块贡献映射
- [LOGGING.md](./LOGGING.md)：当前仍有效的日志前缀、进度条和心跳规范
- [PROMPT_PROFILE_SYSTEM.md](./PROMPT_PROFILE_SYSTEM.md)：当前 PromptProfile / final_prompt 专题说明

根目录的 `AGENTS.md` 仍然是仓库最高约束。

## 推荐阅读顺序

1. 先读本文
2. 再读 [ARCHITECTURE.md](./ARCHITECTURE.md)
3. 再读 [PIPELINE_CONTRACT.md](./PIPELINE_CONTRACT.md)
4. 按任务需要补读 [LOGGING.md](./LOGGING.md)、[PROMPT_PROFILE_SYSTEM.md](./PROMPT_PROFILE_SYSTEM.md)、[RESEARCH_DEV_GUIDE.md](./RESEARCH_DEV_GUIDE.md)
5. 如果任务涉及方法设计、实验方向或论文贡献映射，再读 [TITS_METHODOLOGY.md](./TITS_METHODOLOGY.md)

## 关于 `archive/`

[archive/](./archive/) 只保存历史方案、旧术语和旧实验设计说明。

- 它们不代表当前默认主链路
- 新开发与新文档应以上面的核心文档为准
