# INFER_BASE_ONLY_BASELINE

> 历史方案归档文档
> - 不代表当前默认主链路
> - 仅供追溯旧 baseline 设计时参考

`base_only` 已不再属于当前 infer 主链路概念。

当前主链路没有 `structured / base_only` prompt policy 分支，`infer` 统一读取 `prompt/final_prompt.json` 中的全局 `prompt / negative_prompt`。

当前事实源请看：

- [../PROMPT_PROFILE_SYSTEM.md](../PROMPT_PROFILE_SYSTEM.md)
- [../PIPELINE_CONTRACT.md](../PIPELINE_CONTRACT.md)
