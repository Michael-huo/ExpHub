# PromptProfile 系统说明

> 本文回答什么问题：当前 prompt 子系统到底输出什么、`PromptProfile v1` 还承担什么职责、`infer` 如何消费 prompt plan。

## 1. 当前默认设计

当前主链路的 prompt 系统收敛为：

`PromptProfile v1 -> base_prompt.json + state_prompt_manifest.json + runtime_prompt_plan.json + report.json`

默认 prompt backend 是 `smolvlm2`，`qwen` 保留为显式回退或对照。

这里的职责拆分是：

- `base_prompt.json` 只负责全局硬约束，不再主导场景名词
- `state_prompt_manifest.json` 负责把 `state_segments.json` 变成区间 prompt manifest
- `runtime_prompt_plan.json` 负责按 deploy segments 展开为 infer 可直接消费的 runtime prompt plan
- `report.json` 汇总 backend、采样、profile 摘要与产物统计

## 2. 默认生成流程

`scripts/prompt_gen.py` 的当前默认流程是：

1. 从 `segment/frames/` 选代表帧
2. 对代表帧做闭集 `PromptProfile` 分类，并聚合成 clip-level profile
3. 基于聚合 profile 生成一个收缩语义后的 `base_prompt.json`
4. 基于 `segment/state_segmentation/state_segments.json` 生成 `state_prompt_manifest.json`
5. 基于 `segment/deploy_schedule.json` 把 deploy segments 展开为 `runtime_prompt_plan.json`
6. 写出 `report.json`

它不再默认生成 `final_prompt.json` 或 `deploy_to_state_prompt_map.json`。

## 3. `PromptProfile v1` 当前作用

`PromptProfile v1` 仍然保留闭集 profile，用于采样摘要与 prompt report，但它不再直接驱动一个“全局场景主导 final prompt”。

当前 `base_prompt.json.profile` 与 `prompt/report.json.profile` 仍使用以下字段：

- `version`
- `scene_type`
- `surface_type`
- `side_structures`
- `lighting_type`
- `dynamic_risk`
- `repetition_risk`
- `profile_confidence`

## 4. `base_prompt.json`

`base_prompt.json` 只负责全局 hard constraints。当前正向语义收敛到：

- first-person continuity
- geometry consistency
- stable exposure / white balance
- no flicker / warping / ghosting

示意结构如下：

```json
{
  "version": 1,
  "schema": "base_prompt.v1",
  "base_prompt": "...",
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
  "source": "prompt_runtime_base_v1",
  "rules_hit": ["global_invariants_only", "..."]
}
```

## 5. `state_prompt_manifest.json`

`state_prompt_manifest.json` 的语义单位仍以 `segment/state_segmentation/state_segments.json` 为准，但字段语义已切到“区间 prompt manifest”。

每个区间至少包含：

- `state_segment_id`
- `start_frame`
- `end_frame`
- `state_label`
- `prompt_text`
- `negative_prompt_delta`
- `prompt_strength`

当前正式 `state_label` 仍保留 `low_state / high_state` 两类；`motion_trend` 如果存在，只是辅助字段。

## 6. `runtime_prompt_plan.json`

`runtime_prompt_plan.json` 是 `infer` 唯一正式消费的 prompt 文件。

它按 deploy segments 展开，每个 segment 至少包含：

- `deploy_segment_id`
- `start_frame`
- `end_frame`
- `state_segment_id`
- `state_label`
- `base_prompt`
- `local_prompt`
- `resolved_prompt`
- `negative_prompt`
- `prompt_strength`

示意结构如下：

```json
{
  "version": 1,
  "schema": "runtime_prompt_plan.v1",
  "source": "runtime_prompt_plan_v1",
  "base_prompt": "...",
  "negative_prompt": "...",
  "segments": [
    {
      "deploy_segment_id": 0,
      "start_frame": 0,
      "end_frame": 60,
      "state_segment_id": 0,
      "state_label": "low_state",
      "base_prompt": "...",
      "local_prompt": "...",
      "resolved_prompt": "...",
      "negative_prompt": "...",
      "prompt_strength": 0.35
    }
  ]
}
```

## 7. 与 `infer` 的接口关系

`infer_i2v.py` 当前行为是：

- execution 边界继续来自 `segment/deploy_schedule.json`
- prompt 文本不再在 `infer` 前端重拼
- `infer` 直接读取 `prompt/runtime_prompt_plan.json`
- runtime 继续只消费 prompt plan，不直接解析 state 文件

因此当前主链路不再存在：

- `final_prompt.json` 统领一切
- `deploy_to_state_prompt_map.json` 单独作为正式产物
- `infer/prompt_manifest_resolved.json` 作为正式 runtime prompt 输入

## 8. 已退出当前默认主链路的旧概念

以下概念只适合出现在历史归档或旧实验追溯里：

- `final_prompt`
- `deploy_to_state_prompt_map`
- `prompt_manifest_resolved`
- `prompt_manifest_v2`
- `structured`
- `base_only`
- `delta_prompt`
