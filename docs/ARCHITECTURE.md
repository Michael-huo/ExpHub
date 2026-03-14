# ExpHub 系统架构说明 (ARCHITECTURE.md)

> **文档定位**：本文档是 ExpHub 平台的核心架构蓝图，定义了系统的顶层调度机制、主链路生命周期以及数据流转规范。关于具体的脚本输入输出参数，请参阅 `SCRIPTS_AUDIT.md`；关于日志 UI 规范，请参阅 `LOGGING.md`。

## 1. 核心调度架构：平台化与跨环境穿透
ExpHub 的核心设计理念是**“调度外壳与算法核心的分离”**。平台自身不绑定任何算法环境，而是作为一个智能路由器，根据配置动态拉起不同的运行环境。

- **唯一入口**：`python -m exphub`
- **执行链路**：`exphub/__main__.py` -> `exphub/cli.py` -> `exphub/runner.py` -> `scripts/*.py`
- **极致解耦 (Platform Decoupling)**：
  - 平台通过 `scripts/_common.py` 读取 `config/platform.yaml` 获取所有外部依赖（包含 Conda 环境的 Python 解释器绝对路径、算法源码 Repos、模型权重 Models）。
  - **统一 phase 配置**：各解释器统一定义在 `config/platform.yaml -> environments.phases.<phase>.python`，例如 `segment / prompt / prompt_smol / infer / infer_fun_5b / slam`。
  - **配置形态**：
    ```yaml
    environments:
      phases:
        segment:
          python: /path/to/segment/python
        prompt:
          python: /path/to/prompt/python
        prompt_smol:
          python: /path/to/prompt_smol/python
        infer:
          python: /path/to/infer/python
        infer_fun_5b:
          python: /path/to/infer_fun_5b/python
        slam:
          python: /path/to/slam/python
    ```
  - **跨环境穿透**：`exphub/runner.py` 不使用传统的 `bash -lc "conda activate ..."`。它按 phase 名读取对应解释器路径，并使用 `subprocess` 拉起底层脚本。
  - **`segment` 的接入方式**：`segment` 仍保持 `runner.ros_exec(...)` 先 source ROS 的调用方式，但解释器解析也已收敛到统一的 phase resolver，不再使用 CLI override 或系统环境变量兼容层。
  - **子进程继承**：底层脚本（如 `infer_i2v.py`）若需拉起更底层的多卡任务（如 `torchrun`），必须使用 `sys.executable`，以确保继承当前被激活的绝对路径，彻底免疫系统的 `PATH` 环境变量污染。

## 2. 主链路生命周期 (Main Pipeline)
ExpHub 支持通过 `--mode all` 一键贯穿，或通过具体 mode 单步执行。`segment` 阶段完成后，系统会在主链路外默认立即尝试一次 warn-only 的 `segment_analyze.py` 后处理；这一步不会改写 raw schedule，也不会阻断后续 `prompt / infer / merge / slam`。主链路本身仍严格遵循以下 7 大阶段的单向数据流转：

1. **`segment` (切片锚定)**：读取原始数据集，按时间戳抽取关键帧，构建时间网格。
2. **`prompt` (提示词生成)**：由 `prompt_gen.py` 前端完成 clip 采样与产物写回，再按 backend 路由到 VLM（当前支持 Qwen2-VL 与 SmolVLM2）生成基础与增量 Prompt。
3. **`infer` (视频生成推理)**：由 `infer_i2v.py` 先完成 request 归一与 schedule 解析，再按 infer backend 调用 I2V 模型 (如 VideoX-Fun/Wan) 进行两帧之间的插帧推理，生成视频片段。
4. **`merge` (序列合并)**：将分散的视频片段在时间轴上对齐、去重，并融合成连贯的长序列图像与视频。
5. **`slam` (位姿估计)**：将合成的图像序列送入 VSLAM 算法（如 DROID-SLAM）计算相机轨迹。
6. **`eval` (轨迹评估)**：使用 `evo` 工具比对估计轨迹与真实轨迹 (Ground Truth)，计算 APE/RPE 等指标。
7. **`stats` (统计出图)**：收集上述各阶段的 `step_meta.json` 与运行日志，生成最终的 `EXPERIMENT PERFORMANCE PROFILING` 面板与数据报表。

### 2.1 时间计划三层语义
当前主链路中，关键帧相关时间计划已明确拆分为三层：

- `raw schedule`：只保存在 `segment/keyframes/keyframes_meta.json`，表达研究层正式关键帧序列，保持 canonical/source-of-truth 地位。
- `deploy schedule`：保存在 `segment/deploy_schedule.json`，当前第一版只实现 `wan_r4` 投影，用于把 raw keyframes 投影到 Wan 可执行的时间网格。
- `execution manifest`：复用 `prompt/manifest.json` 中的 `segments[*]`，由 deploy schedule 派生并被 `prompt / infer / merge` 直接消费。

因此，`prompt / infer / merge` 不再自己根据全局固定 `kf_gap` 重建段边界；Wan 第一版要求 deploy 首尾固定、段数不变、总跨度不变，且每段 deploy gap 为 4 的倍数。

### 2.2 Prompt backend 解耦
- `scripts/prompt_gen.py` 只负责 CLI 参数归一、按 `sample_mode` 选图、调用 backend、写回 `segment/clip_prompts.json` 与 `prompt/manifest.json`。
- `scripts/_prompt/sampling.py` 只负责自然排序与采样策略；当前默认外部行为收敛为 `even + 5 图`，并仅保留其他模式作为兼容入口，不绑定具体模型。
- `scripts/_prompt/backends/base.py` 定义统一 backend 接口：`load() / generate(image_paths, instruction) / meta()`。
- `scripts/_prompt/backends/qwen_backend.py` 保留旧 Qwen 主路径。
- `scripts/_prompt/backends/smolvlm2_backend.py` 接入 `HuggingFaceTB/SmolVLM2-2.2B-Instruct`，并允许通过 `prompt_smol` phase 使用独立 Conda Python。

### 2.3 Infer backend 解耦
- `scripts/infer_i2v.py` 现在只负责读取 `segment/frames`、`prompt/manifest.json`、`deploy_schedule.json`，并构造统一 infer request。
- `scripts/_infer/backends/base.py` 定义统一 backend 接口：`load() / run(request) / meta()`。
- `scripts/_infer/api.py` 负责根据 `--infer_backend` 创建 backend；当前支持：
  - `wan_fun_a14b_inp`：默认 backend，继续复用当前 Wan2.2-Fun-A14B-InP 路线；
  - `wan_fun_5b_inp`：首个替代 backend，用于验证更小的 Wan2.2-Fun-5B-InP 在首尾帧任务上的可行性。
- 当前 A14B backend 已成为真实实现承载体，5B backend 复用同一套 backend 运行框架；单卡可 direct run，多卡则由 backend 内部自行拉起 `torchrun` worker，从而保持产物目录与 `runs_plan.json` schema 不变。
- 当前 Wan infer 的多卡执行模式仍是“同一段多卡协同推理、段与段串行”；所有 rank 走同一套 segment 推理路径，只在副作用出口保留最小角色差异，即仅 main process / rank0 负责保存视频、逐帧图片与结果元信息，并在 worker `finally` 中清理 `destroy_process_group()`。
- `scripts/_infer_i2v_impl.py` 现仅保留为 deprecated compatibility shim，内部直接转发到新的 A14B backend。
- CLI 会根据 backend 切换 phase：默认 A14B 走 `infer`，`wan_fun_5b_inp` 走 `infer_fun_5b`；`infer_i2v.py` 仍是唯一对外 infer 入口。

## 3. 产物与目录规范 (Artifacts & Workspace)
每个实验都会在一个专属的 `EXP_DIR` 下运行，命名契约由 `ExperimentContext` 统一管理（格式：`{tag}_{w}x{h}_t{start}s_dur{dur}s_fps{fps}_gap{kf_gap}`）。

目录结构严格按照执行阶段划分，实现物理隔离：
- `segment/`：存放抽取的关键帧、缩略图预览及分割元数据。
- `prompt/`：存放 `manifest.json` 及提示词相关产物。
- `infer/`：存放各段生成的视频、推理帧文件夹及 `runs_plan.json`。
- `merge/`：存放最终合并的 `frames/`、对齐时间戳 `timestamps.txt` 及完整视频。
- `slam/`：存放轨迹输出（如 `traj_est.tum`）及 SLAM 系统内部缓存。
- `eval/`：存放误差统计结果及 evo 绘制的误差对比图（`.zip` 或图像）。
- `logs/`：收口所有阶段的终端全量输出（如 `infer.log`, `slam.log`），严格遵守前缀规范，不含原地刷新的乱码。

## 4. 模块执行边界与防篡改机制
- **单向依赖**：下游步骤只能**只读**上游步骤的输出（例如 `prompt` 只能读 `segment/frames`），绝不允许反向修改或回写上游数据。
- **无状态重试**：同参数重复执行任意步骤，必须直接覆盖原有输出目录或无缝接着跑（幂等性）。
- **缺失阻断**：若前置数据缺失（如 `infer` 找不到 `prompt/manifest.json`），脚本必须立即抛出清晰的 `[ERR]` 错误并退出，禁止自行“脑补”或越权生成。

## 5. Doctor 体检机制 (`--mode doctor`)
`doctor` 模式用于在正式运行前对环境和配置进行“防呆”检查。它**只检查不落盘**，不会创建任何实验目录。
- **检查范围**：doctor 默认读取 core phase 的解释器配置，即 `segment / prompt / <selected infer phase> / slam`；若默认 prompt backend 为 `smolvlm2`，还会额外检查 `prompt_smol`；若 `--infer_backend wan_fun_5b_inp`，则 infer 侧检查项会切换为 `infer_fun_5b`。
- **输出内容**：逐项输出 `phase=<name> python=<path> exists=<bool>`。
- **失败条件**：若任一 core phase 缺失配置，或其 python 路径不存在/不可执行，则返回 `FAIL`。
