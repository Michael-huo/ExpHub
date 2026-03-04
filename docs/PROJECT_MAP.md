# ExpHub 工程地图（PR2c）

## 1. 核心模块（`exphub/`）

| 路径 | 职责 | 关键输入 | 关键输出 |
|---|---|---|---|
| `exphub/__main__.py` | 模块入口转发 | CLI 参数 | 调用 `cli.main()` |
| `exphub/cli.py` | 流水线编排与 mode 分发 | 用户参数、数据配置、脚本路径 | 统一目录契约下的实验产物 |
| `exphub/config.py` | `datasets.json` 解析 | `config/datasets.json` | `DatasetResolved`（bag/topic/intrinsics） |
| `exphub/meta.py` | 命名与元信息辅助 | 运行参数 | `EXP_NAME`、`exp_meta.json` |
| `exphub/runner.py` | 执行器封装 | cmd/env/cwd | `ros_exec`、`conda_exec` |
| `exphub/cleanup.py` | `keep_level` 清理策略 | `exp_dir`、`keep_level` | 删除非必要调试文件 |

## 2. 编排脚本（`scripts/`）

| 路径 | 被哪个 mode 调用 | 主要职责 | 在新目录中的落盘 |
|---|---|---|---|
| `scripts/segment_make.py` | `segment` | 生成标准化片段数据 | `segment/*` |
| `scripts/prompt_gen.py` | `prompt` | 生成 prompt manifest | `prompt/manifest.json`（及 `segment/clip_prompts.json`） |
| `scripts/infer_i2v.py` | `infer` | i2v 批量推理入口（内部调用 `_infer_i2v_impl.py`） | `infer/runs/*`、`infer/runs_plan.json`、`infer/step_meta.json` |
| `scripts/merge_seq.py` | `merge` | 合并推理片段 | `merge/frames/*`、`merge/{calib.txt,timestamps.txt,merge_meta.json,step_meta.json}` |
| `scripts/slam_droid.py` | `slam` | 运行 DROID-SLAM | `slam/<track>/{traj_est.tum,traj_est.npz,run_meta.json}`（至少 `ori/gen`） |
| `scripts/stats_collect.py` | `stats` | 统计收集与汇总 | `stats/report.json`、`stats/compression.json` |

## 3. mode 调用链

| mode | 调用链 |
|---|---|
| `segment` | `cli.main -> step_segment -> ros_exec(segment_make.py)` |
| `prompt` | `cli.main -> step_prompt -> conda_exec(prompt_gen.py)` |
| `infer` | `cli.main -> step_infer -> conda_exec(infer_i2v.py -> _infer_i2v_impl.py)` |
| `merge` | `cli.main -> step_merge -> conda_exec(merge_seq.py)` |
| `slam` | `cli.main -> step_slam -> conda_exec(slam_droid.py)` |
| `eval` | `cli.main -> step_eval -> conda_exec(evo_traj/evo_ape)` |
| `stats` | `cli.main -> step_stats -> conda_exec(stats_collect.py)` |
| `doctor` | `cli.main -> step_doctor`（只读） |
| `all` | `segment -> prompt -> infer -> merge -> slam -> eval -> stats` |

## 4. 目录与依赖关系
- 实验根目录：`experiments/<dataset>/<sequence>/<EXP_NAME>/`
- 关键依赖链：
  - `prompt` 依赖 `segment/frames`
  - `infer` 依赖 `prompt/manifest.json` 与 `segment/*`
  - `merge` 依赖 `infer/runs` + `infer/runs_plan.json`
  - `slam(gen)` 依赖 `merge/frames`
  - `eval` 依赖 `slam/ori/traj_est.tum` 与 `slam/gen/traj_est.tum`
- `slam` 步骤直接写最终目录 `slam/<track>/`，不再通过临时 `slam` 目录重命名。
- 清理时机：非 `doctor` 模式在步骤完成后执行 `apply_keep_level`。

## 5. `doctor` 检查要点
- 输出新 mode 列表与目录契约摘要。
- 脚本存在性清单按“当前真实调用脚本”检查（即当前 `scripts/*.py` 文件名）。
- conda 工具检查统一捕获异常，失败只记 `WARN`，继续后续检查。
- 保证无副作用：不创建实验目录与产物。

## 6. 日志机制
- 编排层统一输出 `[RUN]/[STEP]` 摘要。
- 子进程完整输出写入 `EXP_DIR/logs/<step>.log`。
- 终端展示受 `--log_level` 控制（`info|debug|quiet`），失败时会打印日志尾部若干行。
