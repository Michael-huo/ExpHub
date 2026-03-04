# ExpHub 日志规范（PR-A）

## 1. 目标
- 编排层终端输出聚焦关键信息，避免子进程原始输出刷屏。
- 子进程完整输出可追溯，统一落盘到 `EXP_DIR/logs/`。

## 2. 终端摘要格式
- 运行摘要：`[RUN] mode=... dataset=... seq=... exp_dir=... log_level=...`
- 步骤开始：`[STEP] <name> start`
- 步骤完成：`[STEP] <name> done sec=... out=...`
- 步骤失败：`[STEP] <name> FAIL sec=... rc=... log=...`
- 失败时会额外打印 log 最后 N 行：`[TAIL] ...`

## 3. 日志目录与文件命名
- 目录：`EXP_DIR/logs/`
- 文件（按 step）：
  - `segment.log`
  - `prompt.log`
  - `infer.log`
  - `merge.log`
  - `slam_ori.log`
  - `slam_gen.log`
  - `eval.log`
  - `stats.log`

说明：同一步骤内若包含多次子命令（如 `eval`），会追加写入同一个 step 日志文件。

## 4. `--log_level` 行为
- `info`（默认）：
  - 终端仅透传前缀匹配行：`[INFO] [WARN] [ERR] [PROG] [STEP]`
  - 其余输出只写入 `logs/*.log`
- `debug`：
  - 终端透传全部子进程输出（接近旧行为）
  - 同时完整写入 `logs/*.log`
- `quiet`：
  - 终端仅显示编排层摘要
  - 子进程输出仅写入 `logs/*.log`
  - 失败时仍打印最后 N 行

## 5. 设计边界
- 本规范只改变“输出展示方式”，不改变任何 step 的算法/语义/产物契约。
- `doctor` 为只读模式，不创建 `EXP_DIR`，因此不落地 `logs/`。
