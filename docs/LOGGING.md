# ExpHub 日志规范

> 本文回答什么问题：哪些日志前缀是当前有效的、终端和落盘如何分流、如何打心跳方便性能分析。

## 1. 目标

当前日志口径是：

- 默认终端采用白名单展示，只保留阶段状态、关键警告、进度条、少量阶段心跳和最终结果
- `EXP_DIR/logs/*.log` 保留完整排障信息
- 业务脚本不要直接用 `print()` 输出业务日志，应优先使用 `scripts/_common.py` 的日志门面

顶层 `exphub/cli.py` 的阶段框架输出属于 orchestration 例外，不属于业务脚本自由打印的许可。

## 2. 当前有效前缀

| 前缀 | 用途 | 当前行为 |
|---|---|---|
| `[STEP]` | 顶层阶段开始、结束、失败 | 终端高亮，落盘保留 |
| `[INFO]` | 正常状态变化、耗时心跳、输出路径摘要 | `info` 仅白名单透传关键心跳；详细内容默认只落盘；`debug` 透传更多细节 |
| `[PROG]` | 阶段进度播报 | `info` 仅白名单透传阶段摘要；详细内容默认只落盘；`debug` 透传更多细节 |
| `[WARN]` | 非致命问题、兼容回退、可继续执行的异常 | 终端透传，落盘保留 |
| `[ERR]` | 致命错误 | 终端透传，落盘保留，并在失败时触发尾日志追溯 |
| `[BAR]` | 进度条原地刷新专用前缀 | 终端原地刷新，不写入 `.log` |
| `[PROMPT]` | prompt/negative_prompt 的追溯日志 | 默认不进入顶层终端透传，主要保留在日志文件中 |

`[PROMPT]` 当前记录的是最终 prompt 注入信息，不再特指旧的 `base/delta` 文本体系。

## 3. `[BAR]` 规则

凡是使用 `tqdm` 等进度条，必须让输出以 `[BAR]` 开头，例如：

```text
[BAR] {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]
```

这样 `runner.py` 才会把它当作终端原地刷新，而不是污染日志文件。

## 4. 性能心跳

ExpHub 的性能面板依赖显式心跳，而不是只看顶层计时。

长耗时步骤至少应在以下位置输出 `[INFO]`：

- 模型或 processor 加载完成后
- 初始化或量化完成后
- 整个阶段结束后
- 必要时在单段或批次处理后输出汇总

推荐心跳内容：

- 关键子阶段耗时
- 已处理段数或帧数
- 总耗时与平均耗时

如果心跳格式被随意改坏，`stats` 和人工排障都会受影响。

## 5. 终端等级

`exphub/cli.py` 当前沿用 `--log_level` 做终端收口：

- `info`：默认模式。终端只保留 `EXPERIMENT SUMMARY`、`[STEP]`、`[BAR]`、少量白名单 `[INFO]/[PROG]` 心跳、`[WARN]/[ERR]` 和最终 `EXPERIMENT REPORT`
- `debug`：放开更多 `[INFO]/[PROG]` 细节，适合现场排障
- `quiet`：尽量只保留 `[STEP]`、`[WARN]/[ERR]` 与最终报告；默认不显示 `EXPERIMENT SUMMARY`

白名单之外的路径、写盘回执、cache、policy materialize、schedule 构建、模型路径、命令行拼接等细节，默认都应下沉到 phase 日志，而不是继续刷终端。

默认终端还会进一步压缩逐帧 prompt 明细、冗长 infer 配置、重复 eval 指标与绝对路径输出。

当前 `infer` 的逐段摘要在 `info` 终端下应保持简洁，但建议保留执行边界信息，例如：

`seg 1/5: idx 0->23 infer=12.34s save=0.21s elapsed=12.6s eta=50.4s`

## 6. Phase 日志落盘

顶层 runner 会继续把子脚本完整 stdout/stderr 收口到 `EXP_DIR/logs/*.log`。

典型文件包括：

- `logs/segment.log`
- `logs/prompt.log`
- `logs/infer.log`
- `logs/merge.log`
- `logs/eval.log`
- `logs/stats.log`
- `logs/slam_ori.log` / `logs/slam_gen.log`
- `logs/segment_analyze.log`

默认终端被抑制的详细工程信息，应优先去这些文件排查。

## 7. 外部库防污染

调用 `evo`、`ffmpeg`、SLAM 或其他第三方程序时，应避免让它们直接把杂乱输出刷到终端。

当前规则是：

- 能过滤时优先过滤
- 不能过滤时至少收口到日志文件
- 只有符合前缀契约的内容才应该稳定出现在终端

## 8. 当前默认日志口径与 prompt 系统的关系

主链路 prompt 现在收敛为 `PromptProfile -> final_prompt.json`。

因此日志侧应使用的术语是：

- `profile`
- `final_prompt`
- `prompt / negative_prompt`
- `prompt_source`

不应继续把 `manifest_v2 / base_only / delta_prompt` 写成当前默认日志语言。
