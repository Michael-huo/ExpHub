# PromptProfile 最小系统说明

## 1. 为什么废弃旧复杂 prompt 架构
旧的 `prompt_manifest_v2 / structured / base_only / delta_prompt / intent_card / control_hints / compiled` 体系虽然提供了更多中间层，但在当前数据口径和 Wan 推理链路下，实际效果是：

- prompt 侧概念过多，验收成本高；
- infer 侧需要理解额外策略层，链路更脆弱；
- segment-level prompt/policy 更容易把局部噪声放大成几何不稳定；
- 与当前最优 hand-crafted prompt 相比，复杂 prompt 并没有稳定带来更好的 SLAM 精度。

因此主链路收敛为一个目标：`PromptProfile -> prompt / negative_prompt`。

## 2. 新系统目标
- prompt 端内部只保留一个极简结构：`PromptProfile v1`
- prompt 端最终只产出两条文本：`prompt` 与 `negative_prompt`
- infer 端只消费这两条文本
- 默认行为保守、短小、可复现，并尽量贴近当前最佳手工 prompt

## 3. PromptProfile v1 字段
`prompt/profile.json` 只包含以下字段：

- `version`
- `scene_type`: `park_walkway | campus_path | orchard_row | field_path | road_edge | corridor | indoor_walkway | unknown`
- `surface_type`: `pavement | concrete | brick | dirt | mixed | unknown`
- `side_structures`: 最多 2 个，词表为 `grass / trees / shrubs / walls / columns / fence / buildings / crops / soil_edges / hedges`
- `lighting_type`: `daylight | overcast | indoor_even | unknown`
- `dynamic_risk`: `low | people | vehicles | animals | mixed`
- `repetition_risk`: `low | medium | high`
- `profile_confidence`: `low | medium | high`

## 4. 生成流程
`scripts/prompt_gen.py` 的默认流程为：

1. 从 `segment/frames/` 中抽取 3 到 5 张代表帧
2. 对每张帧调用 VLM 做闭集分类
3. 对候选 profile 做多帧投票聚合
4. 规则清洗，确保字段严格落在闭集词表内
5. 用固定模板生成最终 `prompt / negative_prompt`
6. 只写 `prompt/profile.json`、`prompt/final_prompt.json`、`prompt/step_meta.json`

不会再默认生成 `manifest.json`、`clip_prompts.json`、segment-level delta prompt 或 policy debug。

## 5. 正向模板
正向 prompt 固定为 6 句，只有 `scene_phrase` 和 `texture_phrase` 可变：

1. `First-person camera moving forward along an {scene_phrase}.`
2. `Photorealistic.`
3. `Stable exposure and white balance.`
4. `Consistent perspective and geometry, level horizon.`
5. `Sharp, stable textures on {texture_phrase}.`
6. `No flicker, no warping, no artifacts.`

模板会严格避免以下正向内容：

- `person / people / car / bike / animal`
- `bench / lamp / sign`
- `background / right side / left side / near / far`
- 颜色细节与局部装饰细节

## 6. 反向模板
负向 prompt 由三部分拼接：

1. 固定底板：
   `blurry, flickering, warping, wobble, rolling shutter artifacts, ghosting, double edges, inconsistent geometry, wrong perspective, texture swimming, repeating patterns, oversharpening halos, heavy motion blur, text, watermark, jpeg artifacts, excessive noise, color shift, low quality`
2. `dynamic_risk` 追加项
3. `repetition_risk=high` 时追加 `duplicate structures, repeated textures`

## 7. Infer 消费方式
infer 现在只读取 `prompt/final_prompt.json`：

```json
{
  "version": 1,
  "prompt": "...",
  "negative_prompt": "...",
  "profile": {...},
  "source": "prompt_profile_v1"
}
```

`infer_i2v.py` 会把这两条文本广播到所有 execution segments；执行边界单独来自 `segment/deploy_schedule.json` 或 `infer/execution_plan.json`，不再由 prompt 文件承载。

## 8. 已退出主链路的旧概念
以下概念已不再属于默认执行链路：

- `prompt_manifest_v2`
- `intent_card`
- `control_hints`
- `legacy`
- `compiled`
- `delta_prompt`
- `delta_neg_prompt`
- `structured`
- `base_only`
- `manifest_v2_structured`
