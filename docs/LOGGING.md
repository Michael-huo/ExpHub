# ExpHub 日志规范（PR-A）

## 1. 目标
- 编排层终端输出聚焦关键信息，避免子进程原始输出刷屏。
- 子进程完整输出可追溯，统一落盘到 `EXP_DIR/logs/`。

## 2. 终端摘要格式
- 运行摘要：启动时输出 `EXPERIMENT SUMMARY` 区块，所有行使用 `[INFO]` 前缀，首尾使用 `======================================================================` 分隔。
  - 固定键：`Mode`、`Dataset`、`Sequence`、`Tag`、`Resolution`、`FPS`、`Duration`、`GPUs`、`Keep Level`、`Exp Dir`
- 步骤开始：`[STEP] <name> start`
- 步骤完成：`[STEP] <name> done sec=... out=...`
- 步骤失败：`[STEP] <name> FAIL sec=... rc=... log=...`
- 失败时会额外打印 log 最后 N 行：`[TAIL] ...`
- `cli.py::_step` 会在步骤开始时打印上下分隔线，并在步骤完成/失败后打印收尾分隔线。

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
实现说明：`exphub/runner.py::StepRunner` 统一维护每个 `log_name` 的打开状态（首次 `w`，后续 `a`），并负责将 step 子命令路由到对应日志文件。  
补充：子进程执行统一使用 `stderr=subprocess.STDOUT`，`stderr` 与 `stdout` 会合流写入同一 `logs/<step>.log`。  
补充：`eval` 步骤中 `evo_traj/evo_ape` 输出会同时写入 `eval/*.txt` 与 `logs/eval.log`。  
补充：`[BAR]` 行不会写入任何 `logs/*.log`，仅用于终端原地刷新。

## 4. 子进程输出路由矩阵（run_cmd）
- `[BAR]`（新增高频进度条前缀）：
  - 判定条件：行内包含 `[BAR]`（含前导空格场景）。
  - 路由：只输出到终端，不写入日志文件。
  - 呈现：使用 `\r` 回车原地刷新。
- `[STEP]` `[INFO]` `[WARN]` `[ERR]` `[PROG]`：
  - 路由：写入日志文件，并按 log level 规则透传到终端。
- 其他未匹配行：
  - 路由：写入日志文件。
  - 终端：`info/quiet` 隐藏，`debug` 显示。

## 5. `--log_level` 行为
- `info`（默认）：
  - 终端透传前缀匹配行：`[INFO] [WARN] [ERR] [PROG] [STEP]`
  - 终端显示 `[BAR]` 原地刷新行
  - 其余输出只写入 `logs/*.log`
- `debug`：
  - 终端透传全部子进程输出
  - `[BAR]` 仍使用 `\r` 原地刷新
  - 同时完整写入 `logs/*.log`（`[BAR]` 例外）
- `quiet`：
  - 终端仅显示编排层摘要
  - 子进程输出仅写入 `logs/*.log`（`[BAR]` 也不显示）
  - 失败时仍打印最后 N 行

## 6. ANSI 终端样式
- ANSI 颜色仅用于终端展示，不写入日志文件。
- 子进程透传颜色：
  - `[STEP]`：加粗青色
  - `[ERR]`：红色
  - `[WARN]`：黄色
- 编排层 `cli.py::_step` 同步使用加粗/颜色，保持父子进程视觉一致。

## 7. 换行与进度条边界
- 当上一条终端可见输出为 `[BAR]` 且随后出现普通换行输出时，路由器会先补一个换行，再打印普通行，避免文本粘连。
- 子进程结束时若最后一条终端可见输出为 `[BAR]`，路由器会补一个收尾换行，避免后续提示符或 `[STEP]` 贴在同一行。

## 8. 设计边界
- 本规范只改变“输出展示方式”，不改变任何 step 的算法/语义/产物契约。
- `doctor` 为只读模式，不创建 `EXP_DIR`，因此不落地 `logs/`。
