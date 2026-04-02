# PromptProfile 系统说明

> 本文回答什么问题：当前 prompt 子系统到底输出什么、`PromptProfile v1` 现在承担什么职责、`infer` 如何消费 prompt plan。

## 1. 当前默认设计

当前主链路的 prompt 系统收敛为：

`PromptProfile v1 -> base_prompt.json + state_prompt_manifest.json + runtime_prompt_plan.json + report.json`

默认 prompt backend 是 `smolvlm2`，`qwen` 保留为显式回退或对照。

职责拆分如下：

- `base_prompt.json` 负责全局硬约束
- `state_prompt_manifest.json` 负责把 state 区间变成区间 prompt manifest
- `runtime_prompt_plan.json` 负责把 deploy segments 展开为 infer 可直接消费的 runtime plan
- `report.json` 汇总 backend、采样、profile 摘要与产物统计

## 2. 默认生成流程

`exphub/pipeline/prompt/service.py` 的当前默认流程是：

1. 从 `segment/frames/` 选代表帧
2. 对代表帧做闭集 `PromptProfile` 分类，并聚合成 clip-level profile
3. 基于聚合 profile 生成 `base_prompt.json`
4. 基于 `segment/state_segmentation/state_segments.json` 生成 `state_prompt_manifest.json`
5. 基于 `segment/deploy_schedule.json` 展开 `runtime_prompt_plan.json`
6. 写出 `report.json`

## 3. `PromptProfile v1` 当前作用

`PromptProfile v1` 当前主要用于采样摘要、约束生成与报告聚合，不再承担独立“旧式全局 prompt 产物”的职责。

当前 profile 字段保持：

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

## 5. `state_prompt_manifest.json`

`state_prompt_manifest.json` 的语义单位以 `segment/state_segmentation/state_segments.json` 为准。

每个区间至少包含：

- `state_segment_id`
- `start_frame`
- `end_frame`
- `state_label`
- `prompt_text`
- `negative_prompt_delta`
- `prompt_strength`

## 6. `runtime_prompt_plan.json`

`runtime_prompt_plan.json` 是 `infer` 唯一正式消费的 prompt 文件。

每个 deploy segment 至少包含：

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

## 7. 与 `infer` 的接口关系

当前 `infer` 接口边界是：

- execution 边界继续来自 `segment/deploy_schedule.json`
- prompt 文本不再在 `infer` 前端重拼
- `infer` 直接读取 `prompt/runtime_prompt_plan.json`
- runtime 不直接解析 state 文件

## 8. 已退出正式主链的旧 prompt 残留

已经退出正式主链的旧 prompt 桥接产物与旧 schema 只允许存在于兼容清理逻辑或历史实验记录中，不再属于当前默认工作流、默认文档或默认帮助文案。
