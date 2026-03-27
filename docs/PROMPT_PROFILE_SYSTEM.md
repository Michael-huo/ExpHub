# PromptProfile 系统说明

> 本文回答什么问题：当前 prompt 子系统到底输出什么、`PromptProfile v1` 包含什么、`infer` 如何消费它。

## 1. 当前默认设计

当前主链路的 prompt 系统已经收敛为：

`PromptProfile v1 -> prompt/final_prompt.json + prompt/report.json`

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
6. 写出 `final_prompt.json`
7. 若 `segment/state_segmentation/state_segments.json` 可用，则按 state 区间附加生成 `state_prompt_manifest.json`
8. 若 `segment/deploy_schedule.json` 可用，则附加生成 `deploy_to_state_prompt_map.json`
9. `report.json` 汇总 profile 摘要、backend/sampling、state prompt 统计与 deploy 映射统计

它不再把旧的 `manifest_v2 / structured / base_only / delta_prompt` 当作当前默认输出。

## 3. `PromptProfile v1` 字段

`final_prompt.json.profile` 与 `prompt/report.json.profile` 当前使用的闭集字段是：

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

## 5. `state_prompt_manifest.json` 与映射文件

当前 prompt 子系统新增了两类附加产物：

- `prompt/state_prompt_manifest.json`
- `prompt/deploy_to_state_prompt_map.json`

它们的职责拆分是：

- `final_prompt.json` 继续负责 clip-level 场景身份 prompt
- `state_prompt_manifest.json` 负责按 state 区间描述 local motion trend
- `deploy_to_state_prompt_map.json` 只把 execution segments 映射到 state prompt，不再为每个 deploy segment 再生成一条新 prompt

第一版 `motion_trend` 规则保持闭集：

- `low_state -> straight`
- `high_state -> turn`
- 其它或无法识别 -> `unknown`

`state_prompt_manifest.json` 的示意结构：

```json
{
  "version": "v1",
  "source_policy": "state",
  "alignment_source": "state_segments",
  "global_prompt_path": "prompt/final_prompt.json",
  "state_segment_count": 3,
  "state_segments": [
    {
      "state_segment_id": 0,
      "start_frame": 0,
      "end_frame": 484,
      "state_label": "low_state",
      "motion_trend": "straight",
      "prompt_text": "The camera continues moving straight ahead with stable forward motion and consistent perspective."
    }
  ]
}
```

`deploy_to_state_prompt_map.json` 的示意结构：

```json
{
  "version": "v1",
  "alignment_source": "deploy_schedule_to_state_segments",
  "state_prompt_manifest_path": "prompt/state_prompt_manifest.json",
  "deploy_segment_count": 29,
  "deploy_segments": [
    {
      "deploy_segment_id": 0,
      "start_frame": 0,
      "end_frame": 60,
      "state_segment_id": 0
    }
  ]
}
```

## 6. 模板保证

当前模板的原则是：

- 正向 prompt 保守、短小、强调几何稳定与纹理稳定
- 负向 prompt 强调模糊、闪烁、几何错误、重复纹理等风险
- 避免把过细的对象细节、方位细节或噪声细节写进正向文本

这套模板的目的不是最大化描述丰富度，而是尽量减少对几何稳定性不利的文本干扰。

## 7. 与 `infer` 的接口关系

`infer_i2v.py` 当前仍只向 runtime 提供：

- `prompt`
- `negative_prompt`

但现在会先派生 `infer/prompt_manifest_resolved.json`：

- 全局 `prompt / negative_prompt` 继续作为 base scene prompt
- 若存在 `state_prompt_manifest.json` 与 `deploy_to_state_prompt_map.json`，则把 local motion prompt 拼到对应 execution segment 的正向 prompt 上
- runtime 继续只消费 prompt manifest 和 per-segment override，不直接解析 state 文件
- `negative_prompt` 默认沿用全局配置，不对 state prompt 做重复拼接

因此这两类附加产物现在不再只是旁路接口：

- 它们驱动 infer 前端生成 execution-segment 级别的 prompt override
- 缺失或格式异常时，infer 会清晰告警并无损回退到 global-only
- `final_prompt.json` 的兼容主线仍然保留，作为所有拼接的 base prompt 源

执行边界来自：

- `segment/deploy_schedule.json`
- 或 `infer` 前端生成的临时 runtime execution plan

而不是来自 prompt 文件本身。

## 8. 已退出当前默认主链路的旧概念

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
