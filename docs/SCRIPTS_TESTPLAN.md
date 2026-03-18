# ExpHub 验收与测试计划 (SCRIPTS_TESTPLAN.md)

> **文档定位**：本文档定义了 ExpHub 的测试方法论与实操命令。AI 在提交代码前需严格比对“防呆体检”标准；开发者在验证新 Feature 时，可直接复制此处的冒烟测试与全链路回归命令。

## 1. 测试分工准则
- **AI 的测试边界**：AI 不负责执行耗时超过 1 分钟的动态测试。AI 交付代码前，必须确保代码通过静态语法检查 (`python -m py_compile ...`)，并逻辑上满足 `doctor` 模式的检验。
- **人类的测试边界**：核心代码合并前，由人类开发者在真实 GPU 环境下手动执行具体的 `--mode <X>` 或全链路 `--mode all` 测试。

## 2. 第一道防线：环境与依赖体检 (`--mode doctor`)
`doctor` 模式是执行任何真实计算前的“安全网”，它只读取配置，不创建任何实验目录。

**执行命令：**
```bash
python -m exphub --mode doctor --dataset scand --sequence A_Jackal_AHG_Library_Thu_Oct_28_2 --tag test --w 480 --h 320 --fps 24 --dur 4 --kf_gap 24
```

**最小验收：**
- 输出中能看到 `DOCTOR phase=segment python=... exists=...`。
- 输出中能看到 `DOCTOR phase=prompt python=... exists=...`、`DOCTOR phase=infer ...`、`DOCTOR phase=slam ...`。
- 若执行 `python -m exphub ... --mode doctor --infer_backend wan_fun_5b_inp`，输出中的 infer phase 应切换为 `DOCTOR phase=infer_fun_5b ...`。
- 若 `environments.phases.<phase>.python` 未配置或路径不存在，doctor 应直接报 `FAIL`。

## 3. `segment` 正式关键帧策略冒烟测试

当前正式测试与 analyze 收敛对象为：
- `uniform`
- `semantic`
- `motion`

### 3.1 `semantic` 单步验证
```bash
python -m exphub \
  --dataset ncd \
  --sequence rooster_2020-03-10-10-36-30_0 \
  --tag debug_sks \
  --w 480 --h 320 \
  --start_sec 30 --fps 12 --kf_gap 24 --dur 48 \
  --mode segment \
  --segment_policy semantic
```

**最小验收：**
- `segment/keyframes/keyframes_meta.json` 已生成，且 `policy_name="semantic"`。
- `segment/deploy_schedule.json` 已生成，且 `backend="wan_r4"`。
- `deploy_schedule.json` 中 `raw_keyframe_indices` 与 `keyframes_meta.json.keyframe_indices` 对齐；`deploy_keyframe_indices[0/-1]` 与 raw 首尾一致。
- `summary.uniform_count == summary.final_keyframe_count == summary.num_uniform_base == summary.num_final_keyframes`。
- `summary.fixed_budget=true`。
- `keyframe_indices[0]` 与 `uniform_base_indices[0]` 一致，最后一个索引也必须与 uniform 尾锚一致。
- 若场景中存在明显语义变化区，`summary.relocated_count > 0`，且 `summary.avg_abs_shift / summary.max_abs_shift` 可解释。
- `keyframes[*].source_type` 只应出现 `uniform / semantic`，其中 `semantic` 项必须同时满足 `is_relocated=true` 与 `replaced_uniform_index != null`。
- 默认会继续自动触发 post analyze；若环境已安装 `torch + open_clip`，日志中应出现 `post analyze start` 与 `post analyze done`。

### 3.2 `motion` 单步验证
```bash
python -m exphub \
  --dataset ncd \
  --sequence rooster_2020-03-10-10-36-30_0 \
  --tag debug_motion \
  --w 480 --h 320 \
  --start_sec 30 --fps 12 --kf_gap 24 --dur 48 \
  --mode segment \
  --segment_policy motion
```

**最小验收：**
- `segment/keyframes/keyframes_meta.json` 已生成，且 `policy_name="motion"`。
- `segment/deploy_schedule.json` 已生成，且 `segments[*].deploy_gap % 4 == 0`。
- 默认会继续自动触发 post analyze，日志中应出现 `post analyze start` 与 `post analyze done`。
- `segment/analysis/` 最终只保留 `segment_summary.json / segment_timeseries.csv / comparison_overview.png / allocation_overview.png / kinematics_overview.png / projection_overview.png`。
- `segment/analysis/` 中不应再出现 `analysis_meta.json / candidate_points.json / candidate_roles_summary.json / frame_scores.json / peaks_preview.png / score_curve.png / score_curve_with_keyframes.png / candidate_points_overview.png / candidate_roles_overview.png / semantic_curve.png / semantic_vs_nonsemantic.png`。
- `semantic_embeddings.npz` 不应写入 `segment/analysis/`，而应位于 `segment/.segment_cache/segment_analyze/`。
- `summary.uniform_count == summary.final_keyframe_count == summary.num_uniform_base == summary.num_final_keyframes`。
- `summary.fixed_budget=true`。
- `summary.relocated_count > 0` 时，`keyframes[*].is_relocated=true` 的项必须满足 `replaced_uniform_index != null`。
- `summary.motion_displacement_mean / motion_velocity_mean / motion_acceleration_mean / motion_density_mean / motion_action_total` 已存在。
- `keyframes[*].source_type` 只应出现 `uniform / motion`，其中 `motion` 项必须表示预算内重定位而非加帧。

### 3.3 与 uniform 对比
```bash
python -m exphub \
  --dataset ncd \
  --sequence rooster_2020-03-10-10-36-30_0 \
  --tag debug_uniform \
  --w 480 --h 320 \
  --start_sec 30 --fps 12 --kf_gap 24 --dur 48 \
  --mode segment \
  --segment_policy uniform
```

**对比项：**
- `num_uniform_base`
- `num_final_keyframes`
- `extra_kf_ratio`
- `comparison.observers.semantic.observer_summary`
- `comparison.observers.motion.observer_summary`
- `comparison.observer_pair_alignment`

### 3.4 `uniform / semantic / motion` 对比
```bash
python -m exphub \
  --dataset ncd \
  --sequence rooster_2020-03-10-10-36-30_0 \
  --tag debug_motion \
  --w 480 --h 320 \
  --start_sec 30 --fps 12 --kf_gap 24 --dur 48 \
  --mode segment \
  --segment_policy motion
```

**重点观察：**
- `uniform / semantic / motion` 三者的 `final_keyframe_count` 是否一致。
- `semantic` 是否在不改变预算的前提下，把部分中间关键帧向语义高变化区重定位。
- `motion` 是否在不改变预算的前提下，把部分中间关键帧向运动高变化区重定位。
- `segment_summary.json` 是否分别输出 `semantic_*` 与 `motion_*` 平行统计，并包含 `projection` block。

### 3.5 主链路安全验证
```bash
python -m exphub \
  --dataset ncd \
  --sequence rooster_2020-03-10-10-36-30_0 \
  --tag debug_motion_all \
  --w 480 --h 320 \
  --start_sec 30 --fps 12 --kf_gap 24 --dur 48 \
  --mode all \
  --segment_policy motion
```

**最小验收：**
- `segment` step 不因新 policy 崩溃。
- `segment` 完成后、进入 `prompt` 前，日志中应出现 `post analyze start` 与 `post analyze done`；若 analyze 失败，也只应 WARN，不应阻断后续主链路。
- `segment/frames/`、`segment/keyframes/`、`segment/keyframes/keyframes_meta.json`、`segment/deploy_schedule.json`、`segment/timestamps.txt`、`segment/calib.txt`、`segment/preprocess_meta.json`、`segment/step_meta.json` 全部存在。
- `prompt/manifest.json` 应为 `version=2`、`schema=prompt_manifest_v2`，且 `segments[*]` 至少包含 `start_idx / end_idx / num_frames / deploy_gap / intent_card / control_hints / legacy / compiled / delta_prompt / delta_neg_prompt`。
- `segment/clip_prompts.json` 与 `prompt/manifest.json` 的字段结构不应因 prompt backend 切换而变化。
- `prompt/step_meta.json` 应记录 `backend / attn_impl / sample_mode / num_images / backend_python_phase / prompt_gen_total_sec / manifest_version / manifest_schema / fallback_segments`。
- `infer.log` 不应再只出现固定 stride 的 `0->24 / 24->48 / ...`；应体现真实 deploy 边界，例如 `0->28 / 28->60 / 60->92`。
- `infer/runs_plan.json` 应保存真实执行边界；`segments[*]` 还应能看到 `prompt / negative_prompt / num_inference_steps / guidance_scale / prompt_source / policy_source / control_hints`；`merge/frames/` 数量必须等于 `runs_plan` 推导出的 `merged_end_idx - merged_start_idx + 1`。
- `infer/step_meta.json` 应显示 `infer_backend / backend_python_phase / backend_entry_type / runs_plan_sha1 / manifest_consumer_mode / policy_source_counts` 等编排字段；默认命令下 `infer_backend=wan_fun_5b_inp`，多卡口径下 `backend_entry_type` 应为 `torchrun_backend_worker`。
- `infer/policy_debug.json` 应存在，并保存 segment-level compiled prompt 与 runtime policy，便于核对 `manifest_v2_structured / legacy / fallback` 三种来源。
- 若 `all` 全链路耗时过长，至少应人工确认新链路对旧实验产物仍保留 legacy fallback（manifest/deploy schedule 缺失时退回旧 `kf_gap` 切段）。

### 3.6 Prompt backend 切换冒烟测试
```bash
python -m exphub --mode prompt --dataset <ds> --sequence <seq> --tag <tag> --w <w> --h <h> --fps <fps> --dur <dur> --start_sec <start_sec> --prompt_backend smolvlm2 --prompt_num_images 5
python -m exphub --mode prompt --dataset <ds> --sequence <seq> --tag <tag> --w <w> --h <h> --fps <fps> --dur <dur> --start_sec <start_sec> --prompt_backend smolvlm2 --prompt_num_images 5 --prompt_sample_mode even
python -m exphub --mode prompt --dataset <ds> --sequence <seq> --tag <tag> --w <w> --h <h> --fps <fps> --dur <dur> --start_sec <start_sec> --prompt_backend qwen --prompt_num_images 5
```

**重点观察：**
- 默认命令现在以 `smolvlm2 + sdpa + 5 图` 为主口径；不传 `--prompt_backend` 时也应走 `prompt_smol` phase。
- `--prompt_backend smolvlm2` 时，日志与 `prompt/step_meta.json` 应显示 `backend_python_phase=prompt_smol`。
- `--prompt_backend qwen` 时，应回退到原有 `prompt` phase，并继续兼容 `--qwen_model_dir`。
- `prompt/manifest.json` 的 infer 兼容字段必须保持可读：顶层 `base_prompt / base_neg_prompt` 与段内 `delta_prompt / delta_neg_prompt` 不能缺失；`segment/clip_prompts.json` 仍应保留可直接检查的 `prompt` 字段。

### 3.7 Infer backend 切换冒烟测试
```bash
python -m exphub --mode infer --dataset <ds> --sequence <seq> --tag <tag> --w <w> --h <h> --fps <fps> --dur <dur> --start_sec <start_sec>
python -m exphub --mode infer --dataset <ds> --sequence <seq> --tag <tag> --w <w> --h <h> --fps <fps> --dur <dur> --start_sec <start_sec> --infer_backend wan_fun_5b_inp --infer_extra "-- --num_inference_steps 20"
python -m exphub --mode infer --dataset <ds> --sequence <seq> --tag <tag> --w <w> --h <h> --fps <fps> --dur <dur> --start_sec <start_sec> --infer_backend wan_fun_a14b_inp
python -m exphub --mode infer --dataset <ds> --sequence <seq> --tag <tag> --w <w> --h <h> --fps <fps> --dur <dur> --start_sec <start_sec> --prompt_policy base_only
```

**重点观察：**
- 不传 `--infer_backend` 时，应默认走 `wan_fun_5b_inp`，并继续使用既有 `infer/` 目录结构与 `runs_plan.json` schema。
- `--infer_backend wan_fun_a14b_inp` 时，应回退到 A14B 路线，并显示 `backend_python_phase=infer`。
- `--infer_backend wan_fun_5b_inp` 或默认命令时，日志与 `infer/step_meta.json` 应显示 `backend_python_phase=infer_fun_5b`。
- `infer/runs_plan.json`、`infer/step_meta.json`、`merge/frames/` 的产物路径不应因 backend 切换而变化。
- `--infer_model_dir` 为空时，应从 `config/platform.yaml` 自动解析对应 backend 的默认模型条目。
- 若 `prompt/manifest.json` 为 `prompt_manifest_v2`，则至少抽查一个 segment，确认 `runs_plan.json` 中的 `prompt / negative_prompt / num_inference_steps / guidance_scale` 已非纯全局固定值，且 `policy_source=manifest_v2_structured`。
- 若显式传 `--prompt_policy base_only`，则至少抽查一个 segment，确认 `runs_plan.json` 中 `prompt / negative_prompt` 直接等于顶层 `base_prompt / base_neg_prompt`，`num_inference_steps / guidance_scale` 等于 backend 默认值，且 `prompt_source=policy_source=base_only`。
- 若替换为旧 manifest 或去掉结构化字段，则 infer 仍应完成执行，并在 `runs_plan.json` / `policy_debug.json` 中显示 `policy_source=legacy` 或 `fallback`。

## 4. `segment` 研究旁路冒烟测试
`segment_analyze.py` 现在既会在 `--mode segment` 成功后默认自动触发，也会在 `--mode all` 中于 `segment` 后、`prompt` 前自动触发；也可在已有 `segment/` 产物基础上单独运行。

**按 `exp_dir` 直接分析：**
```bash
python scripts/segment_analyze.py --exp_dir <EXP_DIR>
```

**按实验参数解析 `EXP_DIR`：**
```bash
python scripts/segment_analyze.py \
  --dataset ncd \
  --sequence rooster_2020-03-10-10-36-30_0 \
  --tag debug_motion \
  --w 480 --h 320 \
  --start_sec 60 --fps 12 --kf_gap 24 --dur 4
```

**最小验收：**
- `segment/analysis/` 只保留 6 个核心产物：`segment_summary.json / segment_timeseries.csv / comparison_overview.png / allocation_overview.png / kinematics_overview.png / projection_overview.png`。
- `segment/analysis/segment_timeseries.csv` 已生成，且数据行数与 `segment/frames/*.png` 数量一致。
- `segment/analysis/segment_summary.json` 已生成；对 `uniform` 至少包含 `policy_name / allocation / projection / comparison`，并带有 `comparison.observers.{semantic,motion}` 与 `observer_pair_alignment`；对 `semantic / motion` 还应包含 `allocation / alignment / projection / signals` 四类 block。
- `segment/analysis/` 中不应再出现旧产物：`analysis_meta.json / candidate_points.json / candidate_roles_summary.json / frame_scores.json / peaks_preview.png / score_curve.png / score_curve_with_keyframes.png / candidate_points_overview.png / candidate_roles_overview.png / semantic_curve.png / semantic_vs_nonsemantic.png / semantic_embeddings.npz`。
- 若 `segment_policy=semantic`，`alignment.observer_policy` 应为 `motion`；若 `segment_policy=motion`，则应反向为 `semantic`；若 `segment_policy=uniform`，则应同时生成两条 observer 分支。
- 若正式 analyze 需要 `semantic` 语义信号，`segment/.segment_cache/semantic/semantic_embeddings.npz` 应生成并在后续 analyze 中被复用。
- `comparison_overview.png` 应体现 compare 角色，`allocation_overview.png` 应体现 fixed-budget relocation，`projection_overview.png` 应清楚表达 raw -> deploy 的投影关系。
- 若 `segment_policy=semantic / motion`，`segment_summary.json` 中的 `allocation.final_keyframe_count` 应与 uniform 一致，且 `segment_timeseries.csv` 中 `is_relocated_keyframe=True` 的帧应与 `keyframes_meta.json` 对齐。
