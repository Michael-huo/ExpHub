# ExpHub 运行、配置与日志说明

> 本文回答什么问题：从哪里运行当前主链、关键配置放在哪里、哪些日志前缀是当前有效的、终端与落盘如何分流。

## 1. 运行入口与配置位置

- 统一入口：`python -m exphub`
- 顶层调度：`exphub/cli.py` -> `exphub/pipeline/orchestrator.py`
- 阶段实现：`exphub/pipeline/<stage>/service.py`
- phase Python、模型和外部仓库配置：`config/platform.yaml`
- 数据集与序列配置：`config/datasets.json`

## 2. 目标

当前日志口径是：

- 默认终端只保留阶段状态、关键警告、进度条、少量阶段心跳和最终结果
- `EXP_DIR/logs/*.log` 保留完整排障信息
- 业务实现不要直接散落 `print()`；应优先使用 `exphub/common/logging.py` 的日志门面

顶层 `exphub/cli.py` 的 orchestration 输出属于框架层例外。

## 3. 当前有效前缀

| 前缀 | 用途 | 当前行为 |
|---|---|---|
| `[STEP]` | 顶层阶段开始、结束、失败 | 终端高亮，落盘保留 |
| `[INFO]` | 正常状态变化、耗时心跳、输出摘要 | `info` 下白名单透传；详细内容默认只落盘 |
| `[PROG]` | 阶段进度播报 | `info` 下只透传摘要；详细内容默认只落盘 |
| `[WARN]` | 非致命问题、兼容回退 | 终端透传，落盘保留 |
| `[ERR]` | 致命错误 | 终端透传，落盘保留 |
| `[BAR]` | 进度条原地刷新专用前缀 | 终端原地刷新，不写入 `.log` |
| `[PROMPT]` | prompt 追溯日志 | 默认不进入顶层终端透传，主要保留在日志文件中 |

## 4. `[BAR]` 规则

凡是使用 `tqdm` 等进度条，输出必须以 `[BAR]` 开头，例如：

```text
[BAR] {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]
```

这样 runner 才会把它当作终端原地刷新，而不是污染日志文件。

## 5. 性能心跳

长耗时步骤至少应在以下位置输出 `[INFO]`：

- 模型或 processor 加载完成后
- 关键初始化完成后
- 整个阶段结束后
- 必要时在单段或批次处理后输出汇总

推荐心跳内容：

- 子阶段耗时
- 已处理段数或帧数
- 总耗时与平均耗时

## 6. 终端等级

`exphub/cli.py` 当前沿用 `--log_level` 做终端收口：

- `info`：默认模式
- `debug`：放开更多细节，适合排障
- `quiet`：尽量只保留 `[STEP]`、`[WARN]/[ERR]` 与最终报告

逐帧 prompt 明细、冗长 infer 配置、重复指标与大段绝对路径应默认下沉到 phase 日志。

## 7. Phase 日志落盘

典型文件包括：

- `logs/segment.log`
- `logs/prompt.log`
- `logs/infer.log`
- `logs/merge.log`
- `logs/eval.log`
- `logs/stats.log`
- `logs/slam_ori.log`
- `logs/slam_gen.log`

默认终端被抑制的工程细节，应优先去这些文件排查。

## 8. Prompt 术语口径

当前主链对外优先使用以下 prompt 术语：

- `runtime_prompt_plan`
- `resolved_prompt`
- `negative_prompt`
- `prompt_source`
- `scene_prompt`
- `state_control`

`base_prompt` 与 `state_prompt_manifest` 仍可出现在 prompt 阶段内部追溯日志中，但不再作为下游正式 prompt 契约概念使用。当前正式 prompt 结构固定为 base + scene + control。
