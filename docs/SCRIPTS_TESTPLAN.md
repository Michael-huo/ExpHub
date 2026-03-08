# ExpHub 验收与测试计划 (SCRIPTS_TESTPLAN.md)

> **文档定位**：本文档定义了 ExpHub 的测试方法论与实操命令。AI 在提交代码前需严格比对“防呆体检”标准；开发者在验证新 Feature 时，可直接复制此处的冒烟测试与全链路回归命令。

## 1. 测试分工准则
- **AI 的测试边界**：AI 不负责执行耗时超过 1 分钟的动态测试。AI 交付代码前，必须确保代码通过静态语法检查 (`python -m py_compile ...`)，并逻辑上满足 `doctor` 模式的检验。
- **人类的测试边界**：核心代码合并前，由人类开发者在真实 GPU 环境下手动执行具体的 `--mode <X>` 或全链路 `--mode all` 测试。

## 2. 第一道防线：环境与依赖体检 (`--mode doctor`)
`doctor` 模式是执行任何真实计算前的“安全网”，它只读取配置，不创建任何实验目录。

**执行命令：**
```bash
python -m exphub --mode doctor --dataset scand --sequence A_Jackal_AHG_Library_Thu_Oct_28_2 --tag test --w 480 --h 320 --fps 24 --dur 4 --kf_gap 24
```

## 3. `segment` 研究旁路冒烟测试
`segment_analyze.py` 不接入 `--mode all`，应在已有 `segment/` 产物基础上单独运行。

**按 `exp_dir` 直接分析：**
```bash
python scripts/segment_analyze.py --exp_dir <EXP_DIR>
```

**按实验参数解析 `EXP_DIR`：**
```bash
python scripts/segment_analyze.py \
  --dataset ncd \
  --sequence rooster_2020-03-10-10-36-30_0 \
  --tag debug \
  --w 480 --h 320 \
  --start_sec 60 --fps 12 --kf_gap 24 --dur 4
```

**最小验收：**
- `segment/analysis/frame_scores.csv` 已生成，且数据行数与 `segment/frames/*.png` 数量一致。
- `segment/analysis/frame_scores.json` 已生成。
- `segment/analysis/score_curve.png` 与 `score_curve_with_keyframes.png` 已生成。
- `segment/analysis/analysis_meta.json` 已生成，且 `semantic_enabled=false`。
- `frame_scores.csv` 中 `is_uniform_keyframe=True` 的数量应与 `segment/keyframes/keyframes_meta.json` 对齐。
