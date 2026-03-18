# ExpHub 项目概览 (OVERVIEW.md)

> **文档定位**：本文档是 ExpHub 项目的最高层级说明书，概述了项目的科研愿景、工作流主链路以及 AI 辅助开发文档库的导航目录。无论是人类开发者还是 AI Agent，在首次接触本项目时请先阅读此页。

## 1. 项目愿景与平台定位
ExpHub 是一个**高度平台化、配置驱动**的视频流与 VSLAM（视觉同步定位与建图）实验自动化中枢。

它通过极致的环境解耦架构（读取 `platform.yaml` 穿透 Conda 环境隔离），将上游可切换的视觉大模型（VLM）、视频生成模型（I2V）与下游的 C++ SLAM 算法无缝串联。

**当前主攻的科研课题：**
1. **Prompt 消融实验**：探索“更短、更稳、更保守”的全局 prompt 设计对下游 SLAM 轨迹精度的影响。
2. **时空压缩率 vs 精度 Trade-off**：研究通过抽帧与生成式插帧带来的极高视频压缩率下，SLAM 系统的鲁棒性与精度损失边界。
3. **工作流时间效率优化**：攻坚模型加载、Float8 量化与推理阶段的性能瓶颈，极致缩短实验全链路耗时。

## 2. 核心主链路 (Main Pipeline)
ExpHub 严格定义了不可逆的 7 大单向数据流转阶段：
`segment` → `prompt` → `infer` → `merge` → `slam` → `eval` → `stats`

## 3. 实验目录与产物契约
每次实验都在独立的 `EXP_DIR` 下运行，严格物理隔离。
- **命名规则**：`{tag}_{w}x{h}_t{start}s_dur{dur}s_fps{fps}_gap{kf_gap}`
- **不可妥协的核心产物（必须存在）**：
  - `segment/frames/` (抽取的基础关键帧)
  - `segment/keyframes/keyframes_meta.json` (raw schedule，研究层 canonical source-of-truth)
  - `segment/deploy_schedule.json` (Wan r4 deploy schedule，backend-specific 时间网投影)
  - `prompt/profile.json` (`PromptProfile v1` 闭集画像)
  - `prompt/final_prompt.json` (infer 直接消费的最终提示词)
  - `merge/frames/` (最终用于 SLAM 的长序列图像)
  - `slam/<track>/traj_est.tum` (输出的位姿估计轨迹)
  - `stats/report.json` (最终的全局性能评估与耗时统计)

当前 `segment -> prompt -> infer -> merge` 的时间计划已区分为三层：
- `raw schedule`：仅由 `segment/keyframes/keyframes_meta.json` 表达，保留正式关键帧的研究事实。
- `deploy schedule`：由 `segment/deploy_schedule.json` 表达，当前第一版只实现 `wan_r4`，要求首尾固定、段数不变、每段 gap 为 4 的倍数、总跨度守恒。
- `execution plan`：由 `segment/deploy_schedule.json` 或 `infer/execution_plan.json` 表达，下游 `infer / merge` 直接消费其中的 `start_idx / end_idx / num_frames`，不再让 `prompt` 承载执行边界。

## 4. 常用调度指令
ExpHub 将所有的执行与调度收口于 `cli.py`。
- **环境与依赖防呆体检 (Doctor)**：
  `python -m exphub --mode doctor --dataset <ds> --sequence <seq> --tag <tag> ...`
- **正式 `segment` 策略**：
  当前正式 analyze 与方法叙事聚焦 `uniform / motion / semantic`；其中 `semantic` 与 `motion` 共用 fixed-budget allocation 骨架，只在输入信号上分别使用 semantic kinematics 与 motion energy kinematics。CLI、summary、图标题与图例现在都只接受并展示这三个正式名称。
  每次正式 `--mode segment` 成功后都会默认立即触发 analyze；`--mode all` 现在也会在 `segment` 结束后、进入 `prompt` 之前自动触发 analyze；如需关闭可显式传入 `--skip_analyze`。observer-based comparison 仍保持 `semantic` 默认旁路观测 `motion`、`motion` 默认旁路观测 `semantic`、`uniform` 同时观测两者；observer 结果只写 `segment/analysis/segment_summary.json`、`comparison_overview.png` 等 analyze 产物，不会改写正式 `keyframes_meta.json`。
- **常用 `segment_policy` 示例**：
  `python -m exphub --mode segment ... --segment_policy uniform`
  `python -m exphub --mode segment ... --segment_policy semantic`
  `python -m exphub --mode segment ... --segment_policy motion`
- **一键贯穿全流程 (All)**：
  `python -m exphub --mode all --dataset <ds> --sequence <seq> --tag <tag> ...`
  `all` 会在 `segment` 完成后默认尝试生成 6 个 analyze 正式产物：`segment_summary.json / segment_timeseries.csv / kinematics_overview.png / allocation_overview.png / comparison_overview.png / projection_overview.png`。
- **统一 phase 环境配置**：
  `segment / prompt / infer / slam` 的解释器现在统一从 `config/platform.yaml -> environments.phases.<phase>.python` 读取；当前默认 prompt backend 已收敛为 `smolvlm2`，因此不传 `--prompt_backend` 时会默认走 `prompt_smol` phase；infer 侧默认 backend 现为 `wan_fun_5b_inp`，默认会切到 `infer_fun_5b` phase；如需回退到 14B，可显式传 `--infer_backend wan_fun_a14b_inp`，如需回退 prompt，可显式传 `--prompt_backend qwen`。
- **Doctor 观测点**：
  `--mode doctor` 现在展示各个 core phase 的 python 路径与 `exists=True/False` 检查结果；默认口径下会把 `prompt_smol` 一并纳入检查；若显式切到 `--infer_backend wan_fun_5b_inp`，doctor 也会同步检查 `infer_fun_5b` phase。
- **Infer backend 示例**：
  `python -m exphub ... --mode infer`
  `python -m exphub ... --mode infer --infer_backend wan_fun_5b_inp --infer_extra "-- --num_inference_steps 20"`
  `python -m exphub ... --mode infer --infer_backend wan_fun_a14b_inp`
- **Infer 入口约束**：
  `scripts/infer_i2v.py` 现在是唯一外部 infer 入口；旧 `scripts/_infer_i2v_impl.py` 仅保留为兼容壳，不再承载主要实现。
- **多卡执行约束**：
  infer backend 不再让主进程直接触发多卡 `dist.init_process_group(env://)`；当 `gpus > 1` 时，由 backend 内部自行拉起 `torchrun` worker。

---

## 5. AI 唤醒文档库导航 (AI Initialization Kit)
本项目采用 **Docs-as-Code** 模式进行 AI 辅助开发。AI 在进行任何代码修改或架构分析前，**必须**查阅以下对应领域的规范文件：

- ⚖️ **最高宪法与绝对红线** → 参考根目录 `AGENTS.md`
- 🏗️ **跨环境调度与流水线机制** → 参考 `docs/ARCHITECTURE.md`
- 🗺️ **文件树与脚本职责定位** → 参考 `docs/PROJECT_MAP.md`
- 🔌 **底层脚本的输入输出契约** → 参考 `docs/SCRIPTS_AUDIT.md`
- 🖥️ **终端 UI 路由与心跳日志规范** → 参考 `docs/LOGGING.md`
- ✅ **最小核心功能验证判据** → 参考 `docs/SCRIPTS_TESTPLAN.md`
- 🧭 **研究目标、方法主线与开发协作语境** → 参考 `docs/RESEARCH_DEV_GUIDE.md`

修改任何系统级代码后，开发者或 AI 必须同步检查并更新上述相关的说明文件，以保证“系统大脑”的绝对同步。
