# Prompt Manifest V2

## 1. 设计目的
`prompt/manifest.json` 现在升级为 `prompt_manifest_v2`。目标不是改掉 `prompt -> infer` 主链路，而是在保留旧 `delta_prompt / delta_neg_prompt` 消费方式的前提下，把 prompt 阶段的中间语义从“直接写自然语言 delta”改成“先抽结构化 intent，再编译回 legacy prompt”。

本轮只做三件事：
- 给 `prompt/manifest.json` 增加稳定的结构化语义接口；
- 提供从 `intent_card` 到 legacy `delta_prompt` 的 compiler；
- 保证现有 infer 仍直接读取同一路径、同一组 legacy 字段继续运行。

本轮**不**修改 infer 超参数策略，也**不**改变生成 backend 的推理策略。

## 2. 顶层字段
- `version = 2`
- `schema = "prompt_manifest_v2"`
- `base_prompt`：序列级正向基础 prompt。
- `base_neg_prompt`：序列级负向基础 prompt。
- `sequence_meta`：本次 prompt 生成的序列上下文，如 `fps / kf_gap / backend / sample_mode / num_images / schedule_source`。
- `global_invariants`：由 base prompt 拆出的全局不变量，供 compiler 去重与约束。
- `compiler`：legacy compiler 的名称、版本、parse 策略与 fallback 统计。
- `segments`：下游执行段清单，仍复用为 execution manifest。

## 3. `segments[*]` 结构
每个 segment 现在至少包含：
- 执行边界兼容字段：`seg / seg_id / segment_id / start_idx / end_idx / num_frames / deploy_gap`
- 可选执行上下文：`rep_indices / boundary_meta / schedule_source / execution_backend / raw_* / deploy_*`
- `intent_card`
- `control_hints`
- `legacy`
- `compiled`
- infer 兼容字段：`delta_prompt / delta_neg_prompt`

其中：
- `intent_card.scene_anchor`：稳定场景锚点。
- `intent_card.motion_intent`：运动意图或局部动态趋势。
- `intent_card.geometry_constraints`：几何、布局、透视、路径边界约束。
- `intent_card.appearance_constraints`：光照、材质、天气、显著物体约束。
- `intent_card.suppressions`：应抑制的伪影或内容。
- `control_hints.motion_intensity / geometry_priority / risk_level`：保守控制信号，当前主要用于 manifest 描述与 legacy compiler。

## 4. Legacy 兼容策略
`prompt_gen.py` 不再直接把 VLM 文本当最终 delta prompt，而是执行：

1. 先要求 VLM 输出结构化 `intent_card`；
2. 解析失败时，先尝试半结构化键值清洗；
3. 仍失败时，退回 raw text fallback，至少生成最小可用的 legacy delta；
4. 用 `legacy_prompt_compiler` 生成：
   - `segments[*].legacy.delta_prompt`
   - `segments[*].legacy.delta_neg_prompt`
   - `segments[*].compiled.final_prompt_preview`
   - `segments[*].compiled.final_neg_prompt_preview`
5. 再把：
   - `segments[*].delta_prompt = segments[*].legacy.delta_prompt`
   - `segments[*].delta_neg_prompt = segments[*].legacy.delta_neg_prompt`

因此，旧 infer 仍然只需要读取：
- 顶层 `base_prompt / base_neg_prompt`
- 段内 `delta_prompt / delta_neg_prompt`

## 5. infer v1 的消费方式
当前 infer 已进入第一阶段接入：

- 若 manifest 满足 `version=2` 且 `schema=prompt_manifest_v2`，infer 会优先消费 `global_invariants / intent_card / control_hints`，在 infer 侧重新编译每段 `final_prompt / final_neg_prompt`；
- 同时会把 `motion_intensity / geometry_priority / risk_level` 保守映射到 segment-level `num_inference_steps / guidance_scale / negative prompt boost`；
- 若不是 v2，或某段缺少可用结构化字段，则 infer 会退回 legacy `delta_prompt / delta_neg_prompt`；
- 再不行时，仍可回退到纯 `base_prompt / base_neg_prompt`。

也就是说，manifest v2 现在不再只是 prompt 阶段的中间语义接口，而是 infer 侧实际会消费的稳定结构化输入；但整条主链路、CLI 与产物路径都保持不变。
