# PromptProfile 系统说明

> 本文回答什么问题：当前 prompt 子系统到底输出什么、`PromptProfile v1` 包含什么、`infer` 如何消费它。

## 1. 当前默认设计

当前主链路的 prompt 系统已经收敛为：

`PromptProfile v1 -> prompt/final_prompt.json`

设计目标只有四个：

- 语义表示足够小
- 主链路足够稳
- `infer` 消费方式足够简单
- 输出尽量贴近当前有效的手工 prompt 风格

默认 prompt backend 是 `smolvlm2`，`qwen` 保留为显式回退或对照。

## 2. 默认生成流程

`scripts/prompt_gen.py` 的当前默认流程是：

1. 从 `segment/frames/` 选代表帧
2. 默认按 `even` 采样，目标为 5 张，实际会被收敛到 `3..5`
3. 对每张图做闭集 `PromptProfile` 分类
4. 多帧投票聚合为单个 clip-level profile
5. 用固定模板生成全局 `prompt / negative_prompt`
6. 只写 `profile.json`、`final_prompt.json`、`step_meta.json`

它不再把旧的 `manifest_v2 / structured / base_only / delta_prompt` 当作当前默认输出。

## 3. `PromptProfile v1` 字段

`prompt/profile.json` 当前使用的闭集字段是：

- `version`
- `scene_type`
- `surface_type`
- `side_structures`
- `lighting_type`
- `dynamic_risk`
- `repetition_risk`
- `profile_confidence`

字段值被限制在固定词表内，目的是减少自由文本漂移，让 prompt 生成更可控。

## 4. `final_prompt.json` 结构

`infer` 默认消费的是 `prompt/final_prompt.json`，核心结构如下：

```json
{
  "version": 1,
  "prompt": "...",
  "negative_prompt": "...",
  "profile": {
    "scene_type": "...",
    "surface_type": "...",
    "side_structures": ["..."],
    "lighting_type": "...",
    "dynamic_risk": "...",
    "repetition_risk": "...",
    "profile_confidence": "..."
  },
  "source": "prompt_profile_v1",
  "rules_hit": ["..."]
}
```

`source` 当前可能是：

- `prompt_profile_v1`
- `prompt_profile_v1_canonical_best`
- `prompt_profile_v1_fallback_best`

## 5. 模板保证

当前模板的原则是：

- 正向 prompt 保守、短小、强调几何稳定与纹理稳定
- 负向 prompt 强调模糊、闪烁、几何错误、重复纹理等风险
- 避免把过细的对象细节、方位细节或噪声细节写进正向文本

这套模板的目的不是最大化描述丰富度，而是尽量减少对几何稳定性不利的文本干扰。

## 6. 与 `infer` 的接口关系

`infer_i2v.py` 当前只需要：

- `prompt`
- `negative_prompt`

它会把同一组文本广播到所有 execution segments。

执行边界来自：

- `segment/deploy_schedule.json`
- 或 `infer/execution_plan.json`

而不是来自 prompt 文件本身。

## 7. 已退出当前默认主链路的旧概念

以下概念只适合出现在历史归档或旧实验追溯里：

- `prompt_manifest_v2`
- `structured`
- `base_only`
- `delta_prompt`
- `delta_neg_prompt`
- `intent_card`
- `control_hints`
- `compiled`

如果需要追溯旧方案，请看 [archive/](./archive/)；如果需要看当前 prompt 在系统里的位置，请回到 [ARCHITECTURE.md](./ARCHITECTURE.md) 和 [PIPELINE_CONTRACT.md](./PIPELINE_CONTRACT.md)。
