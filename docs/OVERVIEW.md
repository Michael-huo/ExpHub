# ExpHub 概览

> 本文回答什么问题：ExpHub 是什么、当前默认主链路是什么、文档应该从哪里开始读。

ExpHub 是一个面向视频流与 VSLAM 实验的平台化调度外壳。它把关键帧抽取、语义提示生成、视频恢复、SLAM 与评测串成统一实验闭环，同时把不同阶段的 Python 解释器、模型路径和外部仓库配置收口到 `config/platform.yaml`。

## 当前默认主链路

标准执行链路固定为：

`segment -> prompt -> infer -> merge -> slam -> eval -> stats`

当前主链路的默认口径是：

- `prompt` 默认使用 `smolvlm2`，稳定产出 `prompt/profile.json` 与 `prompt/final_prompt.json`，并可附加产出 `prompt/state_prompt_manifest.json` 与 `prompt/deploy_to_state_prompt_map.json`
- `infer` 默认使用 `wan_fun_5b_inp`，优先消费 `segment/deploy_schedule.json`
- `segment` 成功后会默认触发一次 `segment_analyze.py` 旁路分析；它只写 `segment/analysis/`，不改写主链路产物

如果你只需要把握当前系统，先记住三个核心事实：

- `segment/keyframes/keyframes_meta.json` 是 raw keyframe schedule 的事实源
- `segment/deploy_schedule.json` 是当前 Wan 执行投影
- `prompt/final_prompt.json` 是 `infer` 的默认 prompt 输入
- `prompt/state_prompt_manifest.json` 是按 state 区间生成的局部 motion prompt
- `prompt/deploy_to_state_prompt_map.json` 只负责把 execution segment 映射到 state prompt，不直接生成新 prompt

## 核心文档导航

- [ARCHITECTURE.md](./ARCHITECTURE.md)：系统分层、phase 调度、实验目录与模块边界
- [PIPELINE_CONTRACT.md](./PIPELINE_CONTRACT.md)：每个阶段读什么、写什么、哪些文件属于强契约
- [RESEARCH_DEV_GUIDE.md](./RESEARCH_DEV_GUIDE.md)：研究目标、实验组织方式、推荐开发流程
- [TITS_METHODOLOGY.md](./TITS_METHODOLOGY.md)：当前 T-ITS 论文的方法论目标、学术叙事与模块贡献映射
- [LOGGING.md](./LOGGING.md)：当前仍有效的日志前缀、进度条和心跳规范
- [PROMPT_PROFILE_SYSTEM.md](./PROMPT_PROFILE_SYSTEM.md)：当前 PromptProfile / final_prompt 专题说明

根目录的 `AGENTS.md` 仍然是仓库最高约束。

## 推荐阅读顺序

1. 先读 [ARCHITECTURE.md](./ARCHITECTURE.md)
2. 再读 [PIPELINE_CONTRACT.md](./PIPELINE_CONTRACT.md)
3. 按任务需要补读 [RESEARCH_DEV_GUIDE.md](./RESEARCH_DEV_GUIDE.md) 或 [LOGGING.md](./LOGGING.md)
4. 如果任务涉及 prompt 语义编码，再读 [PROMPT_PROFILE_SYSTEM.md](./PROMPT_PROFILE_SYSTEM.md)
5. 如果任务涉及方法设计、实验方向或论文贡献映射，再读 [TITS_METHODOLOGY.md](./TITS_METHODOLOGY.md)

## 关于 `archive/`

[archive/](./archive/) 只保存历史方案、旧术语和旧实验设计说明。

- 它们不代表当前默认主链路
- 新开发与新文档应以上面的核心文档为准
