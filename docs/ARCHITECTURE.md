# ExpHub 架构说明（PR2c：接口词清理）

## 1. 入口与分发
- 统一入口：`python -m exphub`
- Python 入口链路：`exphub/__main__.py` -> `exphub/cli.py:main()`
- `--mode` 仅支持：`all`、`segment`、`prompt`、`stats`、`infer`、`merge`、`slam`、`eval`、`doctor`
- 旧 mode `droid` 已移除，不保留 alias。
- 默认保留策略：`keep_level=repro`

## 2. 命名契约（固定不变）
- `EXP_NAME`：`{tag}_{w}x{h}_t{start}s_dur{dur}s_fps{fps}_gap{kf_gap}`
- 由 `exphub/meta.py::build_exp_name()` 统一生成。
- `kf_gap=0` 时会先自动推导，再写入最终 `EXP_NAME`。

## 3. 新 mode 职责
- `segment`：只跑数据切段（`scripts/segment_make.py`）
- `prompt`：只跑提示词生成（`scripts/prompt_gen.py`）
- `infer`：只跑 i2v 推理（`scripts/infer_i2v.py`，内部调用 `scripts/_infer_i2v_impl.py`，模型后端为 Wan2.2）
- `merge`：只跑序列合并（`scripts/merge_seq.py`）
- `slam`：只跑 DROID（`scripts/slam_droid.py`），最终落盘为 `slam/<track>/`
- `eval`：只跑 `evo_traj/evo_ape`
- `stats`：只跑统计汇总（`scripts/stats_collect.py` -> `stats/report.json`，并兼容写 `compression.json`）
- `doctor`：只读检查，无副作用
- `all`：严格顺序 `segment -> prompt -> infer -> merge -> slam -> eval -> stats`

## 4. 实验目录契约（EXP_DIR）
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

## 5. 关键行为约束
- `prompt` 依赖 `segment/frames`，缺失时提前给出清晰错误。
- `infer` 不会自动生成 prompt，必须已有 `prompt/manifest.json`。
- `doctor` 只检查不落盘：不会创建 `EXP_DIR` 或任何实验产物。
- `eval` 在 `droid` conda env 内检测与执行 `evo_traj/evo_ape`；缺失时 `WARN`，不崩溃。
- 日志收口：子进程完整输出写入 `EXP_DIR/logs/*.log`，终端按 `--log_level` 透传（详见 `docs/LOGGING.md`）。

## 6. `doctor` 检查矩阵
- Critical（失败即非 0）：
  - `datasets.json` 存在且可解析
  - `dataset/sequence` 可解析
  - `bag` 存在
  - 关键脚本存在（按当前真实调用脚本清单）
- Optional（缺失仅 WARN）：
  - 外部路径：`videox_root`、`droid_repo`、`qwen_model_dir`
  - `--auto_conda` 下 env 工具检查（`python/evo_traj/evo_ape`）
- Informational：
  - 打印新 mode 列表与目录契约摘要
  - 打印推导出的 `EXP_NAME` 与 `EXP_DIR`
