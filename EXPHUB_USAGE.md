# ExpHub 使用与实验复现指南

## 1. 使用前说明

所有命令都应在 ExpHub 仓库根目录执行，主入口是：

```bash
python3 -m exphub
```

久未使用、迁移机器或切换环境后，先确认 `python3` 指向当前可用的 ExpHub 运行环境，并查看当前最终 CLI：

```bash
python3 -m exphub -h
```

`infer all` 或单独补跑 `decode` 前，需要确认 ComfyUI pool 已经启动。单独 `eval` 不需要 ComfyUI，也不会重新打开 ROS bag，但需要已有 prepare/decode/GT 产物。

实验目录由 `dataset`、`sequence`、`tag`、`fps`、`start`、`dur` 共同决定。同一组身份参数会定位到同一个 artifact root；再次执行 `--step all` 会重新运行主链路，并可能覆盖该实验目录中的当前结果。想保留旧结果并启动新实验时，请使用新的 `--tag`。

`--experiments` 只能随 `--mode infer --step all` 使用。它不是“仅补跑支线”的模式，而是先执行完整主链路，再按命令中给出的顺序执行支线实验。

## 2. ComfyUI Pool 启动与停止

主线 `decode` 和 `infer all` 会访问 ComfyUI。运行前启动 pool：

```bash
exphub/comfyui_pool.sh start
```

实验结束后如需停止 pool：

```bash
exphub/comfyui_pool.sh stop
```

脚本还支持重启、状态查看和查看某个实例日志：

```bash
exphub/comfyui_pool.sh restart
```

```bash
exphub/comfyui_pool.sh status
```

```bash
exphub/comfyui_pool.sh logs gpu0
```

`logs` 支持 `gpu0`、`gpu1`、`gpu2`。这些命令来自当前 `exphub/comfyui_pool.sh` 的实际 case 分支。

`prepare`、`encode`、`eval`、`motion-benchmark` 不需要 ComfyUI。`image-quality` 读取主线 decode 结果；`compression-benchmark` 会使用已有主线结果并为压缩方法运行对应评测。

## 3. 最终 CLI 参数速查

| 参数 | 用途 | 示例 |
|---|---|---|
| `--mode` | 执行模式，取值 `infer` 或 `train` | `--mode infer` |
| `--step` | 阶段，取值 `prepare/encode/decode/eval/lora/all` | `--step all` |
| `--dataset` | 数据集名称 | `--dataset gdut` |
| `--sequence` | 推理序列；train 可省略，省略时处理该 dataset 的训练序列集合 | `--sequence 2026-04-14-16-20-42` |
| `--tag` | 实验标签，参与 artifact root 定位 | `--tag gdut_test` |
| `--fps` | 目标 FPS | `--fps 24` |
| `--start` | infer 起始秒；train 不接受 | `--start 40` |
| `--dur` | infer 持续秒数；train 不接受 | `--dur 40` |
| `--seed` | decode 随机种子，默认 `12345` | `--seed 12345` |
| `--decode-profile` | ComfyUI decode workflow/profile | `--decode-profile lora_gdut` |
| `--experiments` | infer all 后执行的支线实验 | `--experiments motion-benchmark image-quality` |
| `--log-level` | 子进程终端日志量 | `--log-level quiet` |

`--mode infer --step all` 的主线顺序是：

```text
prepare -> encode -> decode -> eval
```

`--mode train --step all` 的主线顺序是：

```text
prepare -> encode -> lora
```

显式运行：

```bash
python3 -m exphub --mode infer --step eval --dataset gdut --sequence 2026-04-14-16-20-42 --tag gdut_test --start 40 --dur 40 --fps 24
```

会启用 DROID 实时 viewer。`--step all` 不启动实时 viewer，但仍生成离线轨迹 PNG 和 HTML。

## 4. GDUT Demo：完整主线实验

下面命令只跑主线，不跑支线实验：

```bash
python3 -m exphub --mode infer --step all --dataset gdut --sequence 2026-04-14-16-20-42 --tag gdut_test --start 40 --dur 40 --fps 24 --decode-profile lora_gdut
```

`--tag gdut_test` 可替换为新的实验标签。`--start 40 --dur 40` 表示从 40 秒开始、持续 40 秒。`--decode-profile lora_gdut` 使用 GDUT LoRA profile。实验结束后，终端会显示主线耗时、payload、APE 和 artifact root。

再次用同一组身份参数运行 `--step all` 会定位到同一个实验目录并重新生成当前结果；如需保留旧结果，请换一个新的 `--tag`。

## 5. GDUT Demo：完整主线 + 三项支线实验

下面命令用于完整横评与图像质量复现：

```bash
python3 -m exphub --mode infer --step all --dataset gdut --sequence 2026-04-14-16-20-42 --tag gdut_test --start 40 --dur 40 --fps 24 --experiments motion-benchmark compression-benchmark image-quality --decode-profile lora_gdut
```

主线执行顺序：

```text
prepare -> encode -> decode -> eval
```

支线执行顺序与 `--experiments` 中给出的顺序一致：

```text
motion-benchmark -> compression-benchmark -> image-quality
```

支线只在主链路成功后执行。终端会打印主线耗时、每项支线耗时、`optional total time` 和 `full command wall time`。

支线产物分别写入：

```text
encode/motion_benchmark/
eval/compression_benchmark/
decode/image_quality/
```

## 6. Decode Profile 选择

当前 `config/platform.yaml` 中可用的最终 decode profile：

```text
base
lora_gdut
lora_ncd
lora_tum
```

使用基础 workflow：

```bash
python3 -m exphub --mode infer --step all --dataset gdut --sequence 2026-04-14-16-20-42 --tag gdut_base --start 40 --dur 40 --fps 24 --decode-profile base
```

使用 GDUT LoRA：

```bash
python3 -m exphub --mode infer --step all --dataset gdut --sequence 2026-04-14-16-20-42 --tag gdut_lora --start 40 --dur 40 --fps 24 --decode-profile lora_gdut
```

cross-domain LoRA 不需要特殊实验模式。直接使用常规 infer 命令，替换 `--decode-profile`、`--dataset`、`--sequence` 和 `--tag` 即可。例如用 NCD LoRA 跑 GDUT 时，将 profile 改为 `lora_ncd`，并使用新的 `--tag` 保存为另一轮实验。

## 7. 阶段补跑命令

阶段补跑必须使用与原实验相同的 `dataset`、`sequence`、`tag`、`fps`、`start`、`dur`，这样才能定位到同一个 artifact root。

补跑 prepare：

```bash
python3 -m exphub --mode infer --step prepare --dataset gdut --sequence 2026-04-14-16-20-42 --tag gdut_test --start 40 --dur 40 --fps 24
```

补跑 encode，依赖已有 prepare 产物：

```bash
python3 -m exphub --mode infer --step encode --dataset gdut --sequence 2026-04-14-16-20-42 --tag gdut_test --start 40 --dur 40 --fps 24
```

补跑 decode，依赖已有 encode 产物，并应使用与目标实验一致的 decode profile：

```bash
python3 -m exphub --mode infer --step decode --dataset gdut --sequence 2026-04-14-16-20-42 --tag gdut_test --start 40 --dur 40 --fps 24 --decode-profile lora_gdut
```

补跑 eval，依赖已有 prepare、decode 和 `prepare/gt_traj.tum`：

```bash
python3 -m exphub --mode infer --step eval --dataset gdut --sequence 2026-04-14-16-20-42 --tag gdut_test --start 40 --dur 40 --fps 24
```

单独 eval 会打开 DROID 实时 viewer。standalone eval 不需要 ComfyUI，也不会重新打开 ROS bag。

## 8. 训练与 LoRA 命令

train 支持 `prepare`、`encode`、`lora`、`all`。train 模式不接受 `--start` 和 `--dur`，它们不会参与训练数据构建。

完整 train 最小命令，省略 `--sequence` 时处理该 dataset 的训练序列集合：

```bash
python3 -m exphub --mode train --step all --dataset gdut --tag gdut_train --fps 24
```

只训练指定序列：

```bash
python3 -m exphub --mode train --step all --dataset gdut --sequence 2026-04-14-16-20-42 --tag gdut_train_seq --fps 24
```

单独补跑 LoRA，依赖已有 train encode/trainset 产物：

```bash
python3 -m exphub --mode train --step lora --dataset gdut --tag gdut_train --fps 24
```

train 主线产物位于 `artifacts/train/<dataset>/<tag...>/` 下，主要阶段目录包括 `prepare/`、`encode/`、`trainset/` 和 `lora/`。LoRA 训练使用 `config/lora_profiles.json` 中的默认 profile，不需要 ComfyUI 或 DROID。

## 9. 最终实验产物位置

未来新 run 的高价值 artifact 入口如下：

```text
<run-root>/
  run_meta.json
  effective_config.yaml
  command.txt
  git_state.json

  prepare/
    prepare_result.json
    gt_traj.tum
    raw_frames/

  encode/
    encode_result.json
    motion_segments.json
    motion_overview.png
    hvm_payload/
    motion_benchmark/

  decode/
    decode_report.json
    calib.txt
    timestamps.txt
    preview.mp4
    reconstructed_frames/
    image_quality/

  eval/
    summary.json
    summary.csv
    trajectory_overlay_auto2d.png
    trajectory_overlay_interactive.html
    ori/
    rec/
    compression_benchmark/
```

主线结果直接位于 `prepare/`、`encode/`、`decode/`、`eval/` 根目录；支线实验结果位于对应阶段的二级目录。

`eval/summary.json` 和 `eval/summary.csv` 是主实验表格数值的 canonical summary。`encode/encode_result.json` 是 payload、transmitted frames、generation units 的唯一事实来源。

compression benchmark 中，`raw` 使用 ORI APE，`vlmem` 使用 REC APE。`preview.mp4` 用于快速预览重建结果。离线轨迹图为：

```text
eval/trajectory_overlay_auto2d.png
eval/trajectory_overlay_interactive.html
```

## 10. 终端结果如何解读

最终终端报告的主要区域：

```text
[Main Pipeline Times]
[Optional Experiment Times]
[Full Command]
[Payload]
[VSLAM]
[Motion Benchmark]
[Compression Benchmark]
[Image Quality]
```

`main pipeline wall time` 表示主线 `prepare -> encode -> decode -> eval` 的实际耗时。`optional total time` 表示本次执行的支线实验总耗时。`wall time` 表示整条命令从启动到结束的完整耗时。

`APE RMSE ORI` 是原始图像运行 VSLAM 的结果。`APE RMSE REC` 是重建图像运行 VSLAM 的结果。`APE REC-ORI` 是重建相对于原始的变化。

`Raw` / `VLMem` 的 payload 与 APE 口径来自 canonical summary 和 `encode/encode_result.json`，不要手动修改这些数值。

## 11. 快速回归测试

日常修改或未来补实验前，先运行无服务测试：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests
```

该命令不会运行真实 ROS、DROID、ComfyUI、GPU 或完整实验。它用于快速检查 CLI、阶段计划、路径合同、支线调度、standalone 边界、payload/APE 口径与配置解析是否回退。

提交或交接前可检查空白错误：

```bash
git diff --check
```

真实补实验前，建议先跑上述测试，再运行一条短序列命令验证环境。

## 12. 常见使用原则

- 不使用已删除的历史 CLI 参数；不确定参数时先运行 `python3 -m exphub -h`。
- 不手动修改 `eval/summary.json`、`eval/summary.csv` 或 `encode/encode_result.json` 中的 canonical 数值。
- 主线补跑必须保持同一实验身份参数。
- 新 tag 用于创建新实验，避免覆盖旧结果。
- 主线产物放在阶段根目录，支线产物放在二级目录。
- `--experiments` 不是仅补跑支线；它会先跑完整 infer all 主链路。
