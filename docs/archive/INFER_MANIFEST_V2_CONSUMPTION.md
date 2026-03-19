# INFER_MANIFEST_V2_CONSUMPTION

> 历史方案归档文档
> - 不代表当前默认主链路
> - 仅供追溯旧实验设计或旧消费逻辑时参考

本文档描述的 `prompt_manifest_v2 / structured / base_only` infer 消费逻辑已经退出当前主链路。

当前默认实现改为：

- `prompt` 只产出 `prompt/profile.json` 与 `prompt/final_prompt.json`
- `infer` 只消费 `prompt/final_prompt.json` 中的 `prompt / negative_prompt`
- execution schedule 由 `segment/deploy_schedule.json` 或 `infer/execution_plan.json` 单独承载

当前事实源请看：

- [../PROMPT_PROFILE_SYSTEM.md](../PROMPT_PROFILE_SYSTEM.md)
- [../ARCHITECTURE.md](../ARCHITECTURE.md)
- [../PIPELINE_CONTRACT.md](../PIPELINE_CONTRACT.md)
