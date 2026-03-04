# ExpHub `scripts/` 只读审计（PR2c 命名清理版）

## 审计范围
- 脚本：`scripts/*.py`
- 调用点：`exphub/cli.py`
- 口径：不评价算法效果，只审计职责、I/O 契约、风险与可维护性。

## 1. 调用关系总览

| 脚本 | 对应 mode/step | 一句话职责 | 编排层最终落盘 |
|---|---|---|---|
| `segment_make.py` | `segment` | 从 bag 生成标准化片段序列 | `segment/` |
| `prompt_gen.py` | `prompt` | 从片段帧生成 prompt manifest | `prompt/` + `segment/clip_prompts.json` |
| `infer_i2v.py` | `infer` | 批量规划并执行 i2v 推理入口 | `infer/` |
| `merge_seq.py` | `merge` | 合并推理分段并补齐时序/标定 | `merge/` |
| `slam_droid.py` | `slam` | 在 `ori/gen` 轨道执行 DROID-SLAM | `slam/<track>/` |
| `_infer_i2v_impl.py` | `infer` 间接调用 | i2v 核心推理内部实现（Wan2.2 backend） | 由 `infer_i2v.py` 编排承接 |
| `stats_collect.py` | `stats` | 收集统计并输出统一 report | `stats/report.json` |
| `_common.py` | 多脚本复用 | 路径校验、帧排序、JSON 原子写、日志 | 无直接产物 |

## 2. 脚本逐项审计

### 2.1 `scripts/segment_make.py`
- 职责：从 ROS bag 切片并生成 `frames/keyframes/calib/timestamps/meta`。
- 关键参数：`--bag`、`--topic`、`--duration`、`--fps`、`--start_sec|--start_idx`、`--kf_gap`、`--width/height`、内参参数。
- 输入依赖：ROS1 Python 运行时、bag 文件、相机内参。
- 输出产物：
  - 必需：`segment/frames`、`segment/calib.txt`、`segment/timestamps.txt`、`segment/preprocess_meta.json`
  - 调试/复现：`segment/step_meta.json`、`segment/keyframes`、`segment/keyframes/keyframes_meta.json`
- 被调用：`--mode segment`。
- 风险点：
  - ROS 环境依赖强，运行前未 `source` 会直接失败。
  - `duration/fps` 组合导致帧数边界问题时，用户容易误解预期帧数。
- 建议：
  - 低风险：增强开始日志，显式打印预计帧数与关键帧数。
  - 中风险：抽离采样网格计算函数，减少跨脚本重复逻辑。
  - 高风险：重采样策略变更需轨迹对照验证。

### 2.2 `scripts/prompt_gen.py`
- 职责：读取 `segment/frames`，调用 Qwen2-VL 生成 clip prompt 与 manifest。
- 关键参数：`--frames_dir|--segment_dir`、`--exp_dir`、`--fps`、`--kf_gap`、`--base_idx`、`--model_dir`、`--out_manifest`。
- 输入依赖：Qwen 模型目录、可读帧目录、`transformers/torch`。
- 输出产物：
  - 必需：`prompt/manifest.json`
  - 调试：`segment/clip_prompts.json`、`prompt/step_meta.json`（若启用）
- 被调用：`--mode prompt`。
- 风险点：
  - 模型加载耗时和显存占用高。
  - 帧序列稀疏或损坏时，可能出现部分 clip 无有效 prompt。
- 建议：
  - 低风险：细化失败提示（缺目录/无权限/坏图像）。
  - 中风险：增加可选批处理以降低多 clip 推理开销。
  - 高风险：代表帧选择策略变化会影响生成分布。

### 2.3 `scripts/infer_i2v.py`
- 职责：根据 `segment/frames` 生成分段计划并触发 i2v 批量推理。
- 关键参数：`--segment_dir`、`--exp_dir`、`--videox_root`、`--fps`、`--kf_gap`、`--base_idx`、`--num_segments`、`--gpus`。
- 输入依赖：VideoX-Fun 仓库与环境、推理脚本、可用 GPU。
- 输出产物：
  - 必需：`infer/runs/`、`infer/runs_plan.json`
  - 调试/审计：`infer/step_meta.json`
- 被调用：`--mode infer`。
- 风险点：
  - 子进程日志过滤后，复杂异常定位成本偏高。
  - 多卡场景对环境变量与启动方式敏感。
- 建议：
  - 低风险：增加“完整日志模式”开关。
  - 中风险：推理计划生成逻辑抽公共函数，与 prompt/merge 保持一致。
  - 高风险：分段规则改动会改变 merge 与 SLAM 结果。

### 2.4 `scripts/merge_seq.py`
- 职责：按 `runs_plan` 合并分段结果并生成可用于 SLAM 的连续序列。
- 关键参数：`--segment_dir`、`--exp_dir`、`--runs_root`、`--plan`、`--out_dir`、`--fps`。
- 输入依赖：`infer/runs`、`infer/runs_plan.json`、`segment/calib.txt`、`segment/timestamps.txt`。
- 输出产物：
  - 必需：`merge/frames`、`merge/calib.txt`、`merge/timestamps.txt`、`merge/merge_meta.json`
  - 调试：`merge/preview.mp4`
  - 审计：`merge/step_meta.json`
- 被调用：`--mode merge`。
- 风险点：
  - 计划与分段产物不一致时会硬失败。
  - 涉及覆盖写目录，需依赖安全护栏避免误删。
- 建议：
  - 低风险：删除前输出更明确的目标目录摘要。
  - 中风险：抽象拷贝策略与元数据生成函数。
  - 高风险：重叠去重逻辑改动会直接影响时间对齐。

### 2.5 `scripts/slam_droid.py`
- 职责：在输入序列上执行 DROID-SLAM，输出 `traj_est.tum/npz` 与 `run_meta.json`。
- 关键参数：`--segment_dir`、`--out_dir`、`--slam_out_dir`、`--droid_repo`、`--weights`、`--fps`、`--undistort_mode`。
- 输入依赖：DROID 仓库、权重、`frames+calib(+timestamps)`。
- 输出产物：
  - 必需：`slam/<track>/traj_est.tum`
  - 调试：`slam/<track>/traj_est.npz`
  - 必需元信息：`slam/<track>/run_meta.json`
- 被调用：`--mode slam`。
- 路径一致性：`run_meta.json` 中 `tum_path/npz_path` 指向最终 `slam/<track>/traj_est.*`。
- 风险点：
  - 权重/仓库路径错误会在运行时失败。
  - 可视化窗口在无 GUI 环境中可能导致问题。
- 建议：
  - 低风险：增强前置检查和提示。
  - 中风险：抽取数据流预处理以减少重复代码。
  - 高风险：内参与去畸变策略变化会影响轨迹结果。

### 2.6 `scripts/_infer_i2v_impl.py`
- 职责：i2v 核心引擎（Wan2.2 backend），按分段计划生成各 run 并写计划元数据。
- 关键参数：`--batch`、`--frames_dir`、`--kf_gap`、`--dataset_fps`、`--fps`、`--runs_parent`、`--exp_name`。
- 输入依赖：VideoX 模型、配置、GPU 环境。
- 输出产物（由 infer 编排承接到 `infer/`）：run 目录、计划文件、参数元数据。
- 被调用：`infer_i2v.py` 间接调用。
- 风险点：
  - 全局状态较多，单独调试门槛高。
  - 冷启动成本高，易受环境差异影响。
- 建议：
  - 低风险：保持元数据字段稳定，减少歧义。
  - 中风险：拆分参数解析/执行/落盘逻辑，便于单测。
  - 高风险：默认推理超参数改动会改变生成结果。

### 2.7 `scripts/stats_collect.py`
- 职责：收集当前实验可得统计并输出统一 `stats/report.json`，兼容产出 `stats/compression.json`。
- 关键参数：`--exp_dir`。
- 输入依赖：`segment/step_meta.json`、`prompt/step_meta.json`、`infer/step_meta.json`、`merge/step_meta.json`（统计字段严格来自 step metadata，不再扫描 `frames/*.png` 或读取 `manifest.json` 文件大小）。
- 输出产物：
  - 必需：`stats/report.json`
  - 兼容：`stats/compression.json`
- 被调用：`--mode stats`。
- 容错行为：缺少/损坏 step_meta 或字段缺失时，相关统计字段置 `null`（部分字段可回退 `0`），并输出 `WARN`，不崩溃。
- 风险点：
  - 统计字段依赖上游 step metadata 的完整性，跨实验可比性需结合 warnings 判断。
- 建议：
  - 低风险：统一各 step 的 `outputs.frame_count/outputs.bytes_sum` 字段命名，减少兼容分支。

## 3. 优先优化清单（低风险优先）
1. 增强 infer 子进程日志可观测性（提供完整日志开关）。
2. 在 merge 删除前追加更直观的目标路径与保护提示。
3. 统一脚本“输入摘要/输出摘要/关键计数”日志模板。
4. 细化 prompt 前置校验错误文案，降低排障时间。
5. 抽离并统一 `fps/kf_gap/num_segments` 计算函数，减少重复实现。
