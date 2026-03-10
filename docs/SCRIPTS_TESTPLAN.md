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

## 3. `segment` 正式关键帧策略冒烟测试

### 3.1 `semantic_guarded_v2` 单步验证
```bash
python -m exphub \
  --dataset ncd \
  --sequence rooster_2020-03-10-10-36-30_0 \
  --tag debug_v2 \
  --w 480 --h 320 \
  --start_sec 30 --fps 12 --kf_gap 24 --dur 48 \
  --mode segment \
  --segment_policy semantic_guarded_v2 \
  --sys_py <segmentclip_python>
```

**最小验收：**
- `segment/keyframes/keyframes_meta.json` 已生成，且 `policy_name="semantic_guarded_v2"`。
- `summary.num_uniform_base`、`summary.num_final_keyframes`、`summary.extra_kf_ratio` 已存在。
- `summary.num_boundary_relocated + summary.num_boundary_inserted == summary.num_boundary_selected` 可不强制严格相等，但应能解释 boundary 的去向。
- `summary.num_support_inserted == summary.num_support_selected`。
- `summary.num_promoted_support_inserted` 与 `summary.num_burst_windows_triggered` 已存在。
- `keyframes[*].source_type` 仅应出现 `uniform / boundary / support`。
- `keyframes[*].source_role` 应可区分 `boundary_candidate / support_candidate / promoted_support_candidate`。
- `semantic_only_candidate` 与 `suppressed` 不应作为硬关键帧进入 `keyframe_indices`。

### 3.2 与 uniform 对比
```bash
python -m exphub \
  --dataset ncd \
  --sequence rooster_2020-03-10-10-36-30_0 \
  --tag debug_uniform \
  --w 480 --h 320 \
  --start_sec 30 --fps 12 --kf_gap 24 --dur 48 \
  --mode segment \
  --segment_policy uniform \
  --sys_py <segmentclip_python>
```

**对比项：**
- `num_uniform_base`
- `num_final_keyframes`
- `extra_kf_ratio`
- `num_boundary_relocated`
- `num_boundary_inserted`
- `num_support_inserted`
- `num_promoted_support_inserted`

### 3.3 与 `semantic_guarded_v1` 对比
```bash
python -m exphub \
  --dataset ncd \
  --sequence rooster_2020-03-10-10-36-30_0 \
  --tag debug_v1 \
  --w 480 --h 320 \
  --start_sec 30 --fps 12 --kf_gap 24 --dur 48 \
  --mode segment \
  --segment_policy semantic_guarded_v1 \
  --sys_py <segmentclip_python>
```

**重点观察：**
- v2 是否开始出现 `num_support_inserted > 0`
- v2 是否开始出现 `num_promoted_support_inserted > 0`
- 若仍为 0，需检查是否被 `support_trigger_gap / promoted_min_distance / rerank_score / burst_window` 规则挡住

### 3.4 主链路安全验证
```bash
python -m exphub \
  --dataset ncd \
  --sequence rooster_2020-03-10-10-36-30_0 \
  --tag debug_v2 \
  --w 480 --h 320 \
  --start_sec 30 --fps 12 --kf_gap 24 --dur 48 \
  --mode all \
  --segment_policy semantic_guarded_v2 \
  --sys_py <segmentclip_python>
```

**最小验收：**
- `segment` step 不因新 policy 崩溃。
- `segment/frames/`、`segment/keyframes/`、`segment/keyframes/keyframes_meta.json`、`segment/timestamps.txt`、`segment/calib.txt`、`segment/preprocess_meta.json`、`segment/step_meta.json` 全部存在。
- 若 `all` 全链路耗时过长，至少应人工确认 `segment` 产物格式未破坏，且 `prompt / infer / merge / slam / stats` 未新增对 `keyframes_meta.json` 旧字段的破坏性依赖。

## 4. `segment` 研究旁路冒烟测试
`segment_analyze.py` 不接入 `--mode all`，应在已有 `segment/` 产物基础上单独运行。

**按 `exp_dir` 直接分析：**
```bash
python scripts/segment_analyze.py --exp_dir <EXP_DIR>
```

**按实验参数解析 `EXP_DIR`：**
```bash
python scripts/segment_analyze.py \
  --dataset ncd \
  --sequence rooster_2020-03-10-10-36-30_0 \
  --tag debug_v2 \
  --w 480 --h 320 \
  --start_sec 60 --fps 12 --kf_gap 24 --dur 4
```

**最小验收：**
- `segment/analysis/frame_scores.csv` 已生成，且数据行数与 `segment/frames/*.png` 数量一致。
- `segment/analysis/frame_scores.json` 已生成。
- `segment/analysis/score_curve.png` 与 `score_curve_with_keyframes.png` 已生成。
- `segment/analysis/candidate_points.json` 与 `candidate_points_overview.png` 已生成。
- `segment/analysis/candidate_roles_summary.json` 与 `candidate_roles_overview.png` 已生成。
- `segment/analysis/semantic_embeddings.npz` 已生成，第二次运行时应可复用。
- `segment/analysis/semantic_curve.png` 与 `semantic_vs_nonsemantic.png` 已生成。
- `segment/analysis/analysis_meta.json` 已生成，且 `semantic_enabled=true`、`semantic_backend="openclip"`、`use_semantic_in_score=false`、`candidate_role_enabled=true`，并包含 `observed_signals` / `scored_signals`、`role_rules`、`rerank_weights`、`role_thresholds`。
- 若 `segment_policy=semantic_guarded_v1` 或 `semantic_guarded_v2`，`frame_scores.csv` 中 `is_uniform_keyframe=True` 的数量应与 `uniform_base_indices` 对齐，而不是与最终 `keyframe_indices` 强制相等。
- `analysis_meta.json` 中应能看到 `final_keyframe_source_counts / final_keyframe_source_roles / final_keyframe_promotion_sources`，从而区分 boundary / support / promoted_support 的最终来源。
