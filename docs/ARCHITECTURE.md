# ExpHub 架构说明（PR2c：接口词清理）

## 1. 入口与分发
- 统一入口：`python -m exphub`
- Python 入口链路：`exphub/__main__.py` -> `exphub/cli.py:main()`
- `--mode` 仅支持：`all`、`segment`、`prompt`、`stats`、`infer`、`merge`、`slam`、`eval`、`doctor`
- 旧 mode `droid` 已移除，不保留 alias。
- 默认保留策略：`keep_level=max`
- conda 自动切换默认开启；可用 `--no_auto_conda` 关闭。
- `--gpus` 默认值为 `2`（用于 infer 阶段传参）。

## 2. 命名契约（固定不变）
- `EXP_NAME`：`{tag}_{w}x{h}_t{start}s_dur{dur}s_fps{fps}_gap{kf_gap}`
- 由 `exphub/context.py::ExperimentContext` 统一生成与管理。
- `kf_gap=0` 时由 `ExperimentContext.resolve_kf_gap()` 自动推导，再写入最终 `EXP_NAME`。
- `segment/prompt/infer/merge/slam/eval/stats` 路径由 `ExperimentContext` 统一派生，CLI 不再手工拼接实验子目录。
- `segment` 数量计算由 `ExperimentContext.compute_segment_count()` 统一，和 `prompt/infer` 脚本中的公式保持一致。

## 3. 编排上下文（ExperimentContext）
- `cli.main()` 在参数 sanitize 后创建一个 `ExperimentContext`。
- 该上下文集中提供：
  - `exp_name` / `exp_dir`
  - `segment_dir`、`prompt_dir`、`infer_dir`、`merge_dir`、`slam_dir`、`eval_dir`、`stats_dir`
  - 关键产物路径（如 `prompt/manifest.json`、`slam/<track>/traj_est.tum`、`stats/report.json`）
- 各 `step_*` 只消费上下文路径属性，避免重复路径拼接逻辑与漂移风险。

## 3.1 执行封装（StepRunner）
- `cli.main()` 创建 `StepRunner(logs_dir, log_level, runner_cfg, ...)`。
- `step_*` 内子进程调用统一通过：
  - `StepRunner.run_ros(...)`（ROS 相关命令）
  - `StepRunner.run_conda(...)`（conda 环境命令）
- `StepRunner` 统一封装日志参数与路由（`log_path/log_level/pass_prefixes/fail_tail_lines`），并维护日志写入状态（同名日志首次 `w`，后续 `a`）。
- `StepRunner` 显式以 `stderr=subprocess.STDOUT` 执行子进程，保证 stdout/stderr 汇总到同一 step 日志文件。
- 底层执行仍由 `conda_exec/ros_exec` 实现，不改变实验语义与产物契约。

## 4. 新 mode 职责
- `segment`：只跑数据切段（`scripts/segment_make.py`）
- `prompt`：只跑提示词生成（`scripts/prompt_gen.py`）
- `infer`：只跑 i2v 推理（`scripts/infer_i2v.py`，内部调用 `scripts/_infer_i2v_impl.py`，模型后端为 Wan2.2）
- `merge`：只跑序列合并（`scripts/merge_seq.py`）
- `slam`：只跑 DROID（`scripts/slam_droid.py`），最终落盘为 `slam/<track>/`
- `eval`：只跑 `evo_traj/evo_ape`
- `stats`：只跑统计汇总（`scripts/stats_collect.py` -> `stats/report.json`，并兼容写 `compression.json`）
- `doctor`：只读检查，无副作用
- `all`：严格顺序 `segment -> prompt -> infer -> merge -> slam -> eval -> stats`

## 5. 实验目录契约（EXP_DIR）
- `segment/`：切段产物（`frames`、`keyframes`、`calib.txt`、`timestamps.txt`、`preprocess_meta.json` 等）
- `prompt/`：提示词产物（至少 `manifest.json`）
- `infer/`：i2v 推理产物（`runs/`、`runs_plan.json`、`step_meta.json`）
- `merge/`：合并后序列（`frames/`、`calib.txt`、`timestamps.txt`、`merge_meta.json`、`step_meta.json`）
- `slam/`：SLAM 结果根目录，至少包含：
  - `slam/ori/{traj_est.tum,traj_est.npz,run_meta.json}`
  - `slam/gen/{traj_est.tum,traj_est.npz,run_meta.json}`
- `eval/`：evo 输出
- `stats/`：统计输出（`report.json`、`compression.json`）

说明：SLAM 步骤不再使用“临时目录再 move”模式，而是直接写入 `slam/<track>/` 最终目录。

## 6. 关键行为约束
- `prompt` 依赖 `segment/frames`，缺失时提前给出清晰错误。
- `infer` 不会自动生成 prompt，必须已有 `prompt/manifest.json`。
- `stats` 仅从 `segment/prompt/infer/merge/step_meta.json` 读取统计来源；不再扫描 `frames/*.png` 与 `prompt/manifest.json` 文件大小。
- `doctor` 只检查不落盘：不会创建 `EXP_DIR` 或任何实验产物。
- `eval` 在 `droid` conda env 内检测与执行 `evo_traj/evo_ape`；缺失时 `WARN`，不崩溃。
- 日志收口：子进程完整输出写入 `EXP_DIR/logs/*.log`，终端按 `--log_level` 透传（详见 `docs/LOGGING.md`）。

## 7. `doctor` 检查矩阵
- Critical（失败即非 0）：
  - `datasets.json` 存在且可解析
  - `dataset/sequence` 可解析
  - `bag` 存在
  - 关键脚本存在（按当前真实调用脚本清单）
- Optional（缺失仅 WARN）：
  - 外部路径：`videox_root`、`droid_repo`、`qwen_model_dir`
  - 自动 conda 开启时（默认；可用 `--no_auto_conda` 关闭）env 工具检查（`python/evo_traj/evo_ape`）
- Informational：
  - 打印新 mode 列表与目录契约摘要
  - 打印推导出的 `EXP_NAME` 与 `EXP_DIR`

## 8. Cleanup Strategy（`keep_level`）
- `keep_level` 仅支持两档：`max`、`min`（默认 `max`）。
- 历史 aliases 已移除；旧值 `all` 已重命名为 `max`，避免与 `--mode all` 语义冲突。
- 非 `min` 输入统一回退到 `max`。
- 清理逻辑仅基于当前目录契约（`segment/prompt/infer/merge/slam/eval/stats/logs`），不再对历史命名做特判。

- `max`（默认）：
  - 不做清理，完整保留所有产物与日志。

- `min`（批跑优化）：
  - 保留轻量关键文件：
    - `logs/*`
    - 各阶段 `step_meta.json`（若存在）
    - `segment|merge` 的 `calib.txt` 与 `timestamps.txt`
    - `prompt/manifest.json`
  - 保留最终输出目录：`slam/`、`eval/`、`stats/`（目录与内容保留）。
  - 强清理重型中间目录：
    - `segment/frames/`
    - `segment/keyframes/`
    - `infer/runs/`
    - `merge/frames/`
