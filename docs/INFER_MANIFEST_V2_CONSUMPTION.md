# 历史文档说明

本文档描述的 `prompt_manifest_v2 / structured / base_only` infer 消费逻辑已经退出当前主链路。

当前默认实现改为：

- `prompt` 只产出 `prompt/profile.json` 与 `prompt/final_prompt.json`
- `infer` 只消费 `prompt/final_prompt.json` 中的 `prompt / negative_prompt`
- execution schedule 由 `segment/deploy_schedule.json` 或 `infer/execution_plan.json` 单独承载

请以 [PROMPT_PROFILE_SYSTEM.md](/data/hx/ExpHub/docs/PROMPT_PROFILE_SYSTEM.md) 与 [ARCHITECTURE.md](/data/hx/ExpHub/docs/ARCHITECTURE.md) 为准。
