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
- 若 `environments.phases.<phase>.python` 未配置或路径不存在，doctor 应直接报 `FAIL`。

## 3. `segment` 正式关键帧策略冒烟测试

当前正式测试与 analyze 收敛对象为：
- `uniform`
- `sks_v1`
- `motion_energy_v1`

### 3.1 `sks_v1` 单步验证
```bash
python -m exphub \
  --dataset ncd \
  --sequence rooster_2020-03-10-10-36-30_0 \
  --tag debug_sks \
  --w 480 --h 320 \
  --start_sec 30 --fps 12 --kf_gap 24 --dur 48 \
  --mode segment \
  --segment_policy sks_v1
```

**最小验收：**
- `segment/keyframes/keyframes_meta.json` 已生成，且 `policy_name="sks_v1"`。
- `summary.uniform_count == summary.final_keyframe_count == summary.num_uniform_base == summary.num_final_keyframes`。
- `summary.fixed_budget=true`。
- `keyframe_indices[0]` 与 `uniform_base_indices[0]` 一致，最后一个索引也必须与 uniform 尾锚一致。
- 若场景中存在明显语义变化区，`summary.relocated_count > 0`，且 `summary.avg_abs_shift / summary.max_abs_shift` 可解释。
- `keyframes[*].source_type` 只应出现 `uniform / semantic`，其中 `semantic` 项必须同时满足 `is_relocated=true` 与 `replaced_uniform_index != null`。
- 默认会继续自动触发 post analyze；若环境已安装 `torch + open_clip`，日志中应出现 `post analyze start` 与 `post analyze done`。

### 3.2 `motion_energy_v1` 单步验证
```bash
python -m exphub \
  --dataset ncd \
  --sequence rooster_2020-03-10-10-36-30_0 \
  --tag debug_motion \
  --w 480 --h 320 \
  --start_sec 30 --fps 12 --kf_gap 24 --dur 48 \
  --mode segment \
  --segment_policy motion_energy_v1
```

**最小验收：**
- `segment/keyframes/keyframes_meta.json` 已生成，且 `policy_name="motion_energy_v1"`。
- 默认会继续自动触发 post analyze，日志中应出现 `post analyze start` 与 `post analyze done`。
- `segment/analysis/` 最终只保留 `analysis_summary.json / frame_scores.csv / score_overview.png / roles_overview.png / semantic_overview.png`。
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
- `comparison.observers.sks_v1.observer_summary`
- `comparison.observers.motion_energy_v1.observer_summary`
- `comparison.observer_pair_alignment`

### 3.4 `uniform / sks_v1 / motion_energy_v1` 对比
```bash
python -m exphub \
  --dataset ncd \
  --sequence rooster_2020-03-10-10-36-30_0 \
  --tag debug_motion \
  --w 480 --h 320 \
  --start_sec 30 --fps 12 --kf_gap 24 --dur 48 \
  --mode segment \
  --segment_policy motion_energy_v1
```

**重点观察：**
- `uniform / sks_v1 / motion_energy_v1` 三者的 `final_keyframe_count` 是否一致。
- `sks_v1` 是否在不改变预算的前提下，把部分中间关键帧向语义高变化区重定位。
- `motion_energy_v1` 是否在不改变预算的前提下，把部分中间关键帧向运动高变化区重定位。
- `analysis_summary.json` 是否分别输出 `semantic_*` 与 `motion_*` 平行字段。

### 3.5 主链路安全验证
```bash
python -m exphub \
  --dataset ncd \
  --sequence rooster_2020-03-10-10-36-30_0 \
  --tag debug_motion_all \
  --w 480 --h 320 \
  --start_sec 30 --fps 12 --kf_gap 24 --dur 48 \
  --mode all \
  --segment_policy motion_energy_v1
```

**最小验收：**
- `segment` step 不因新 policy 崩溃。
- `segment/frames/`、`segment/keyframes/`、`segment/keyframes/keyframes_meta.json`、`segment/timestamps.txt`、`segment/calib.txt`、`segment/preprocess_meta.json`、`segment/step_meta.json` 全部存在。
- 若 `all` 全链路耗时过长，至少应人工确认 `segment` 产物格式未破坏，且 `prompt / infer / merge / slam / stats` 未新增对 `keyframes_meta.json` 旧字段的破坏性依赖。

## 4. `segment` 研究旁路冒烟测试
`segment_analyze.py` 不接入 `--mode all`，但会在 `--mode segment` 成功后默认自动触发；也可在已有 `segment/` 产物基础上单独运行。

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
- `segment/analysis/` 只保留 5 个核心产物：`analysis_summary.json / frame_scores.csv / score_overview.png / roles_overview.png / semantic_overview.png`。
- `segment/analysis/frame_scores.csv` 已生成，且数据行数与 `segment/frames/*.png` 数量一致。
- `segment/analysis/analysis_summary.json` 已生成；对 `uniform` 至少包含 `policy_name / uniform_base_count / final_keyframe_count / keyframe_bytes_sum / extra_kf_ratio`，并带有 `comparison.observers.{sks_v1,motion_energy_v1}` 与 `observer_pair_alignment`；对 `sks_v1 / motion_energy_v1` 还应包含 `fixed_budget / relocated_count / avg_abs_shift / max_abs_shift` 与成套 `semantic_*` 或 `motion_*` kinematics 统计，以及 `comparison.{observer_policy,signal_alignment,allocation_alignment,observer_summary}`。
- `segment/analysis/` 中不应再出现旧产物：`analysis_meta.json / candidate_points.json / candidate_roles_summary.json / frame_scores.json / peaks_preview.png / score_curve.png / score_curve_with_keyframes.png / candidate_points_overview.png / candidate_roles_overview.png / semantic_curve.png / semantic_vs_nonsemantic.png / semantic_embeddings.npz`。
- 若 `segment_policy=sks_v1`，`comparison.observer_policy` 应为 `motion_energy_v1`；若 `segment_policy=motion_energy_v1`，则应反向为 `sks_v1`；若 `segment_policy=uniform`，则应同时生成两条 observer 分支。
- 若正式 analyze 需要 `sks_v1` 语义信号，`segment/.segment_cache/sks_v1/semantic_embeddings.npz` 应生成并在 analyze 阶段复用；不会写回 `segment/analysis/`。
- `score_overview.png` 应体现 compare 角色，不再回到 legacy candidate/role 主叙事。
- 若 `segment_policy=sks_v1 / motion_energy_v1`，`analysis_summary.json` 中的 `final_keyframe_count` 应与 uniform 一致，且 `frame_scores.csv` 中 `is_relocated_keyframe=True` 的帧应与 `keyframes_meta.json` 对齐。
