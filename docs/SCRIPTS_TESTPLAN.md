# ExpHub scripts 最小验收计划（PR1）

## 0. 约定
- 入口统一使用：`python -m exphub`
- mode 统一为：`all, segment, prompt, stats, infer, merge, slam, eval, doctor`
- 示例参数（可替换为本地真实序列）：
  - `--dataset scand`
  - `--sequence A_Jackal_AHG_Library_Thu_Oct_28_2`
  - `--tag baseline --w 480 --h 320 --fps 24 --dur 4 --start_sec 30 --kf_gap 24 --auto_conda`

## 1) segment（调用 `segment_make.py`）

### 最小命令
```bash
python -m exphub --mode segment --dataset scand --sequence A_Jackal_AHG_Library_Thu_Oct_28_2 --tag baseline --w 480 --h 320 --fps 24 --dur 4 --start_sec 30 --kf_gap 24 --auto_conda
```

### 成功判据
- `segment/frames` 存在且非空。
- `segment/calib.txt`、`segment/timestamps.txt`、`segment/preprocess_meta.json` 存在。

### 常见失败
- bag/topic 配置错误：检查 `config/datasets.json`。
- ROS 环境未就绪：检查 `ROS_SETUP`。

## 2) prompt（调用 `prompt_gen.py`）

### 最小命令
```bash
python -m exphub --mode prompt --dataset scand --sequence A_Jackal_AHG_Library_Thu_Oct_28_2 --tag baseline --w 480 --h 320 --fps 24 --dur 4 --start_sec 30 --kf_gap 24 --auto_conda
```

### 最小输入准备
- 已有 `segment/frames`。
- `QWEN_MODEL_DIR` 指向可用模型目录。

### 成功判据
- `prompt/manifest.json` 存在。
- `segment/clip_prompts.json` 存在（调试产物）。

### 常见失败
- `segment/frames` 不存在：先跑 `--mode segment`。
- `qwen_model_dir` 不存在或不可读：修正模型目录。

## 3) infer（调用 `infer_i2v.py`，内部调用 `_infer_i2v_impl.py`）

### 最小命令
```bash
python -m exphub --mode infer --dataset scand --sequence A_Jackal_AHG_Library_Thu_Oct_28_2 --tag baseline --w 480 --h 320 --fps 24 --dur 4 --start_sec 30 --kf_gap 24 --auto_conda
```

### 最小输入准备
- 已有 `prompt/manifest.json`。
- `VIDEOX_ROOT` 可用。

### 成功判据
- `infer/runs/` 存在。
- `infer/runs_plan.json` 存在。
- `infer/step_meta.json` 存在（若脚本版本支持）。

### 常见失败
- 缺失 `prompt/manifest.json`：先跑 `--mode prompt`。
- VideoX 环境/仓库路径错误：检查 `videox` conda env 与 `VIDEOX_ROOT`。

## 4) merge（调用 `merge_seq.py`）

### 最小命令
```bash
python -m exphub --mode merge --dataset scand --sequence A_Jackal_AHG_Library_Thu_Oct_28_2 --tag baseline --w 480 --h 320 --fps 24 --dur 4 --start_sec 30 --kf_gap 24 --auto_conda
```

### 最小输入准备
- 已有 `infer/runs/` 和 `infer/runs_plan.json`。
- `segment/calib.txt`、`segment/timestamps.txt` 存在。

### 成功判据
- `merge/frames/` 存在且非空。
- `merge/calib.txt`、`merge/timestamps.txt`、`merge/merge_meta.json` 存在。
- `merge/step_meta.json` 存在（若脚本版本支持）。

### 常见失败
- runs plan 缺失：先跑 `--mode infer`。
- 计划与帧不一致：清理后重跑 infer + merge。

## 5) slam（调用 `slam_droid.py`）

### 最小命令
```bash
python -m exphub --mode slam --dataset scand --sequence A_Jackal_AHG_Library_Thu_Oct_28_2 --tag baseline --w 480 --h 320 --fps 24 --dur 4 --start_sec 30 --kf_gap 24 --auto_conda
```

### 最小输入准备
- `ori` 轨：`segment/frames` 可用。
- `gen` 轨：`merge/frames` 可用（默认 `droid_seq=both` 会同时跑）。

### 成功判据
- `slam/ori/traj_est.tum` 存在。
- `slam/gen/traj_est.tum` 存在（`both` 或 `gen` 时）。
- `slam/ori/run_meta.json` 与 `slam/gen/run_meta.json` 存在。
- `run_meta.json` 中 `tum_path/npz_path` 必须指向对应 `slam/<track>/traj_est.*` 最终路径。

### 常见失败
- `droid_repo` 或权重不可用：检查 `DROID_REPO`、`DROID_WEIGHTS`。
- 无图形环境：使用 `--no_viz`。

## 6) eval（调用 `evo_traj/evo_ape`）

### 最小命令
```bash
python -m exphub --mode eval --dataset scand --sequence A_Jackal_AHG_Library_Thu_Oct_28_2 --tag baseline --w 480 --h 320 --fps 24 --dur 4 --start_sec 30 --kf_gap 24 --auto_conda
```

### 成功判据
- `eval/evo_traj_ori.txt`、`eval/evo_traj_gen.txt`（若两条轨迹都存在）。
- `eval/evo_ape_gen_vs_ori.txt`（若 `evo_ape` 可用）。

### 常见失败
- 未安装 `evo_*`：应 `WARN` 并跳过，不崩溃。

## 7) stats（调用 `stats_collect.py`）

### 最小命令
```bash
python -m exphub --mode stats --dataset scand --sequence A_Jackal_AHG_Library_Thu_Oct_28_2 --tag baseline --w 480 --h 320 --fps 24 --dur 4 --start_sec 30 --kf_gap 24 --auto_conda
```

### 成功判据
- `stats/report.json` 存在（统一统计入口）。
- `stats/compression.json` 存在。
- 内容包含 `ori/compressed/ratios` 三段。

### 常见失败
- 缺少 `segment/step_meta.json` 或 `prompt/step_meta.json`：先跑 `segment/prompt`，确认上游 step_meta 落盘成功。
- 旧版 step_meta 缺少 `bytes_sum`：`stats` 会回退 `null/0` 并给出 `WARN`，建议升级上游脚本后重跑。

## 8) doctor（只读）

### 最小命令
```bash
python -m exphub --mode doctor --dataset scand --sequence A_Jackal_AHG_Library_Thu_Oct_28_2 --tag baseline --w 480 --h 320 --fps 24 --dur 4 --start_sec 60 --kf_gap 24 --auto_conda
```

### 成功判据
- 输出新 mode 列表与新目录契约摘要。
- 缺失项以 `WARN` 提示。
- 不创建 `experiments/.../<EXP_NAME>/` 目录。

## 9) 黄金回归建议（`--mode all`）

### 建议命令
```bash
python -m exphub --mode all --dataset scand --sequence A_Jackal_AHG_Library_Thu_Oct_28_2 --tag baseline --w 480 --h 320 --start_sec 30 --fps 24 --kf_gap 24 --dur 4 --auto_conda
```

### 必查产物清单
1. `segment/frames` 数量与 `segment/timestamps.txt` 行数一致（例如 `4s@24Hz` 通常为 `97`）。
2. `prompt/manifest.json` 存在且 `segments` 非空。
3. `infer/runs_plan.json` 存在，且 `infer/runs/` 非空。
4. `merge/frames` 数量合理；`merge/timestamps.txt` 与合并帧数一致。
5. `slam/ori/traj_est.tum` 与 `slam/gen/traj_est.tum` 存在。
6. `slam/ori/run_meta.json`、`slam/gen/run_meta.json` 的 `tum_path`/`npz_path` 指向各自 track 目录。
7. `eval/` 目录生成对比文本（若 `evo_*` 可用）。
8. `stats/report.json` 与 `stats/compression.json` 存在。
