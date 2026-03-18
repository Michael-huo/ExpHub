# Infer Manifest V2 Consumption

## 1. 为什么做 infer v1
`prompt_manifest_v2` 已经把语义从“直接写 delta prompt”升级成了 `global_invariants + intent_card + control_hints`。本轮 infer v1 的目标不是重写 infer 主链路，而是在不破坏既有 `segment -> prompt -> infer -> merge` 契约的前提下，让 infer 开始显式消费这些结构化字段，并把结果落到可观测产物里。

这轮仍然坚持保守策略：
- 只新增最小 `--prompt_policy` 开关；
- 不要求 backend 只能吃 v2；
- 不做黑箱式自动调参；
- 只做可解释、可回退的薄策略层。

## 2. infer 当前消费哪些 manifest v2 字段
当 `prompt/manifest.json` 同时满足：
- `version == 2`
- `schema == "prompt_manifest_v2"`

infer 会进入 `manifest_v2_structured` 模式，显式消费：
- 顶层 `base_prompt / base_neg_prompt`
- 顶层 `global_invariants`
- `segments[*].intent_card`
- `segments[*].control_hints`

逐段流程如下：
1. 先用 `base_prompt / base_neg_prompt` 作为序列级底座。
2. 用 `global_invariants + intent_card + control_hints` 重新编译 segment 级 `delta_prompt / delta_neg_prompt`。
3. 生成 infer 实际执行的 `final_prompt / final_neg_prompt`。
4. 生成每段的 runtime override：`num_inference_steps / guidance_scale`。

若 manifest 不是 v2，或某段缺少可用结构化字段，则 infer 会回退：
- 优先回退到 legacy `delta_prompt / delta_neg_prompt`
- 再不行则回退到纯 `base_prompt / base_neg_prompt`

若显式传入 `--prompt_policy base_only`，infer 会跳过所有 segment-level/structured 文本，只对所有 segment 统一使用：
- `final_prompt = manifest.base_prompt`
- `final_neg_prompt = manifest.base_neg_prompt`
- `num_inference_steps = backend/profile 默认值`
- `guidance_scale = backend/profile 默认值`

此时会忽略：
- `global_invariants`
- `intent_card`
- `control_hints`
- `legacy.delta_prompt / legacy.delta_neg_prompt`
- `compiled.final_prompt_preview / compiled.final_neg_prompt_preview`
- 所有 structured policy 映射

对应 `policy_source` 现在有四种：
- `manifest_v2_structured`
- `base_only`
- `legacy`
- `fallback`

## 3. control_hints 到 runtime policy 的保守映射
本轮只映射三个字段。

### 3.1 `motion_intensity -> num_inference_steps`
- `low`: `default_steps - 10`，下限 32
- `medium`: `default_steps`
- `high`: `default_steps + 10`

默认 Wan profile 为 50 steps 时，对应就是 `40 / 50 / 60`。

### 3.2 `geometry_priority -> guidance_scale`
- `low`: `default_guidance - 0.5`
- `medium`: `default_guidance`
- `high`: `default_guidance + 0.5`

默认 Wan profile 为 6.0 时，对应就是 `5.5 / 6.0 / 6.5`。

### 3.3 `risk_level -> negative prompt 强度`
- `low`: 不额外补强
- `medium`: 追加 `temporal instability, viewpoint drift`
- `high`: 再追加 `geometry collapse, abrupt motion spikes`

这轮没有直接改 `teacache` 策略，保持运行时保守稳定。

## 4. 运行时落点与可观测性
segment-level override 目前已经接到 Wan runtime 执行层，实际生效字段为：
- `prompt`
- `negative_prompt`
- `num_inference_steps`
- `guidance_scale`

可观测性落点：
- `infer/runs_plan.json`
  - 每段都会写入 `prompt / negative_prompt / delta_* / num_inference_steps / guidance_scale / prompt_source / policy_source / control_hints`
  - 顶层会额外写入 `prompt_policy`
- `infer/policy_debug.json`
  - 保存完整的 consumer 输出、`prompt_policy` 与默认 profile 值，便于对照策略来源
- `infer/runs/*/params.json`
  - 保存该段最终执行时真正使用的 prompt、neg prompt、steps、guidance、source 与 `prompt_policy`
- `infer/step_meta.json`
  - 汇总 `manifest_consumer_mode / prompt_policy / prompt_source_counts / policy_source_counts`

## 5. 为什么这轮仍然保守
本轮目标是先把结构化 prompt 真正接入 infer，而不是让策略层过度“聪明”。

因此当前实现仍然刻意保守：
- policy 只映射到少量离散参数；
- 优先保持 backend profile 默认值，只做轻度偏移；
- legacy manifest 完整兼容；
- 下游 `merge / slam / eval` 的接口与产物路径不变。
