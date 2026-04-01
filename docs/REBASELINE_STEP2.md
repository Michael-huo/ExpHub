# Re-baseline Step 2

Step 2 正式把 `prompt` 和 `infer` 两个阶段收口到新的标准 stage 结构，不再让旧脚本入口主导正式 workflow。

## 正式入口

- `prompt` 唯一正式入口：`exphub.pipeline.prompt.service.run(...)`
- `infer` 唯一正式入口：`exphub.pipeline.infer.service.run(...)`

## 唯一正式契约

- `prompt` 的唯一正式下游契约：`prompt/runtime_prompt_plan.json`
- `infer` 的唯一正式上游契约：`prompt/runtime_prompt_plan.json`

`prompt` 仍正式写出四个产物：

- `prompt/base_prompt.json`
- `prompt/state_prompt_manifest.json`
- `prompt/runtime_prompt_plan.json`
- `prompt/report.json`

`infer` 仍正式写出三个产物族：

- `infer/runs_plan.json`
- `infer/report.json`
- `infer/runs/*`

## 已退出正式 Workflow 的旧路径

以下路径不再是正式入口，也不再由 `cli/orchestrator` 直接调用：

- `scripts/prompt_gen.py`
- `scripts/infer_i2v.py`

以下旧概念也不再主导正式链路：

- `final_prompt` 作为 prompt 主产物
- `deploy_to_state_prompt_map` 作为 prompt -> infer 桥梁
- `infer` 侧旧 `prompt_resolver` 主导的 prompt 拼接
- `legacy_kf_gap` / legacy execution segments 作为正式 schedule 双轨

## Step 3 预告

Step 3 会继续沿着这条新主线，把 `merge / slam / eval / stats` 的正式消费口径收口到：

- `merge` 只信任 `infer/runs_plan.json`
- `slam` 只围绕 `segment/` 与 `merge/` 的正式数据轨道
- `eval` 只围绕 `slam` 与 `merge` 的正式产物
- `stats` 只围绕各 stage 的正式 `report.json` / `step_meta.json`
