# Infer Base-Only Baseline

## 为什么做这个基线
当前 infer v1 已支持消费 `prompt_manifest_v2` 的结构化字段，但实验中观察到 structured prompt/policy 可能让生成结果更容易崩坏。`base_only` 模式提供一个最简对照组，用来严格验证“复杂 prompt 是否反而有害”。

## 模式行为
显式传入 `--prompt_policy base_only` 后，infer 对所有 segment 统一使用：
- `final_prompt = manifest.base_prompt`
- `final_neg_prompt = manifest.base_neg_prompt`
- `num_inference_steps = backend/profile 默认值`
- `guidance_scale = backend/profile 默认值`

## 会被忽略的字段
- 顶层 `global_invariants`
- `segments[*].intent_card`
- `segments[*].control_hints`
- `segments[*].legacy.delta_prompt / delta_neg_prompt`
- `segments[*].compiled.final_prompt_preview / final_neg_prompt_preview`
- 所有 structured policy 映射

## 用途
这个模式只用于 baseline 验证：当 `base_only` 比 `structured` 更稳时，说明复杂 segment-level prompt 或轻量 policy 映射本身可能在当前模型/数据口径下带来副作用。
