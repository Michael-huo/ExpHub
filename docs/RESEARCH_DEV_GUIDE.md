# ExpHub 研究与开发主说明书（RESEARCH_DEV_GUIDE.md）

> **文档定位**：本文档是 ExpHub 项目在当前科研课题中的“研究与开发统一事实源”，用于沉淀课题背景、方法主线、评测维度、开发协作流程与阶段性进展。  
> 本文档补充而不替代 `docs/OVERVIEW.md`：
> - `docs/OVERVIEW.md` 负责项目入口、系统总览与文档导航；
> - `docs/RESEARCH_DEV_GUIDE.md` 负责研究语境、方法映射、评测逻辑与 AI+人工开发流程。  
> 在后续讨论、方案设计、功能开发、文档维护中，若问题涉及本课题的研究目标、实验主线或阶段进展，应优先参考本文档。

---

## 1. 课题背景与问题定义

ExpHub 服务于当前科研课题：

**基础模型驱动的边云 VSLAM 图像传输优化**

随着视觉 SLAM 技术持续发展，系统性能不断提升，但对算力的需求也越来越高。在许多机器人场景中，边端设备受到体积、负载、功耗与散热条件约束，难以直接搭载高性能计算平台。因此，边云协同 VSLAM 成为具有现实意义的研究方向。

在现有边云 VSLAM 框架中，课题组已有工作已经对 **云→边** 的数据下传过程进行了优化，尤其是在地图传输方面，通过隐式表征等思路降低了传输负担。然而，**边→云** 的上传过程仍缺乏充分优化。当前边端上传的核心数据是图像序列，这既是云端 VSLAM 最重要的输入之一，也是通信负担最大的部分。

因此，本课题的核心目标不是仅做传统工程式图像压缩，而是探索一种**面向 VSLAM 任务的语义化编解码范式**：

- 在边端只保留少量关键帧；
- 将非关键帧转化为轻量化语义表示，如文本或类似文本的嵌入向量；
- 将关键帧与语义信息上传到云端；
- 利用云端强大的基础模型恢复完整图像序列；
- 并验证恢复结果是否仍然适用于下游 VSLAM。

从这个角度看，边→云图像上传过程被重新表述为一个面向 VSLAM 的语义编解码流程：

- **编码端**：关键帧选择 + 图像语义压缩；
- **解码端**：基于关键帧和语义条件的图像/视频恢复；
- **验证端**：检验恢复结果在几何与语义层面对 VSLAM 是否仍然可用。

这是一项探索性研究。当前目标首先是验证该范式是否成立，而不是直接面向工程部署。

---

## 2. 研究目标

本项目的研究目标可以概括为以下四点：

1. **降低边→云图像传输负担**  
   相比直接上传原始图像序列，显著减少上传数据量。

2. **建立关键帧 + 轻量语义表征的上传机制**  
   仅显式保留少量关键帧，并将冗余图像转化为文本或其他轻量语义形式。

3. **在云端恢复完整视觉序列**  
   使用图像/视频生成模型作为“世界模型式”解码器，从关键帧与语义条件恢复完整序列。

4. **保证恢复结果对下游 VSLAM 仍然可用**  
   恢复图像不仅要“看起来合理”，更要尽可能保持几何一致性、语义一致性与 SLAM 可用性。

长期来看，本课题关注的不是“生成一段好看视频”，而是验证：

**语义压缩 + 生成式恢复**  
是否有可能成为未来边云 VSLAM 中一种成立的图像传输策略。

---

## 3. 方法主线与 ExpHub 工作流映射

ExpHub 当前的核心工作流为：

```text
segment → prompt → infer → merge → slam → eval → stats
```

它与本课题的方法主线一一对应如下。

### 3.1 `segment`：关键帧选择

**科研角色**：编码端的关键帧提取模块。

该阶段决定哪些帧被保留为关键帧，哪些帧被视为冗余帧。当前实现仍以等距采样为主，功能上属于“基础版关键帧选择器”。从研究目标看，该阶段未来应升级为**语义驱动或信息增量驱动的关键帧选择模块**，并与既有 IROS 工作中的关键帧方法接轨。

当前主链路已进一步把 `segment` 的时间计划拆成两层：
- `keyframes_meta.json` 继续只表达 raw schedule，作为研究层事实源；
- `deploy_schedule.json` 则是面向当前 Wan r4 后端的执行投影，不覆盖 raw schedule 本身。

### 3.2 `prompt`：图像到语义表征的编码

**科研角色**：编码端的 I2T（Image-to-Text）或更广义语义编码模块。

该阶段当前使用 Qwen 对图像进行理解并生成文本描述。其本质不是单纯生成 caption，而是将图像压缩为更轻量、可传输、可用于后续恢复的语义条件。

从长期研究角度看，该阶段未来未必局限于自然语言文本，也可能扩展为更结构化、更适合生成模型利用的语义表示。

### 3.3 `infer`：关键帧与语义条件驱动的视频恢复

**科研角色**：解码端的 I&T2V（Image-and-Text-to-Video）模块。

该阶段当前基于 Wan2.2，利用关键帧与文本条件恢复视频或图像序列。它是当前工作流中最接近“世界模型”角色的部分，也是整个语义传输框架中的解码核心。

当前第一版 Wan 接入已不再使用单一全局 `kf_gap` 等距调度，而是改为消费 deploy-derived execution segments：每段都有自己的 `start_idx / end_idx / deploy_gap / num_frames`。这使得 `infer` 日志、`runs_plan.json` 与 `merge` 的时间对齐都能反映真实的非等距关键帧布局。当前 infer v1 还会在运行时显式消费 `prompt_manifest_v2` 的 `global_invariants / intent_card / control_hints`，把结构化语义真正传到每段 prompt 与轻量 runtime policy 上，而不是只依赖 prompt 阶段预编译出的 legacy delta 文本。

### 3.4 `merge`：序列整理与产物收口

**工程角色**：将分段生成结果整理成统一的长序列产物，供下游步骤使用。

该阶段不直接定义科研创新点，但对保持全链路闭环至关重要。

### 3.5 `slam`：几何一致性验证

**评测角色**：将恢复后的序列送入下游 SLAM 算法，验证其几何层面的可用性。

### 3.6 `eval`：核心指标计算

**评测角色**：计算压缩相关、轨迹相关及其他实验指标。

### 3.7 `stats`：耗时与实验统计汇总

**评测与维护角色**：汇总各阶段耗时、关键元数据与实验结果，形成全流程的性能画像。

### 3.8 核心科研步骤与支撑步骤

从课题最终目标来看，当前最直接决定研究贡献的步骤是：

- `segment`
- `prompt`
- `infer`

其余步骤虽然不是核心创新点本体，但承担了以下关键职责：

- 整理产物；
- 验证下游可用性；
- 计算指标；
- 保证实验闭环；
- 支撑长期维护与对比分析。

---

## 4. 当前实现进展

### 4.1 已完成部分

目前 ExpHub 已具备较稳定的实验平台能力：

- 已搭建可运行的完整主链路：
  ```text
  segment → prompt → infer → merge → slam → eval → stats
  ```
- `segment` 正式关键帧层当前聚焦三种策略：`uniform / motion / semantic`；其中 `semantic` 与 `motion` 共用“uniform 骨架 + fixed-budget allocation”的正式主框架，只在输入信号上分别使用 semantic kinematics 与 motion energy kinematics；
- `segment / prompt / infer / slam` 的解释器现已统一由 `config/platform.yaml -> environments.phases.<phase>.python` 管理，日常 `--mode segment / --mode all` 不再依赖 CLI override；doctor 仅暴露 phase python 路径与 exists 状态；
- `prompt` 已接入可切换的图像到文本流程：当前默认收敛为 SmolVLM2（`prompt_smol` phase、`sdpa`、`even` 采样、5 图），同时保留 Qwen 作为显式回退/对照 backend；
- `infer` 已接入基于 Wan2.2 的图像与文本到视频流程，并完成“前端入口 + 可切换 backend”重构：当前默认已切换为 `wan_fun_5b_inp`，`wan_fun_a14b_inp` 保留为显式回退/对照 backend；两者通过 `wan_fun_runtime.py` 共享运行时，但在 backend 结构上保持平级；
- `infer` 现已进入 manifest v2 消费第一阶段：runtime 内新增薄策略层，会把 `motion_intensity / geometry_priority / risk_level` 映射到保守的 `num_inference_steps / guidance_scale / negative prompt boost`，并把每段最终策略写回 `runs_plan.json` 与 `policy_debug.json`；
- 当前默认选择 5B 的原因是：在现有实验口径下，它具备更高推理效率，且已观察到优于 14B 的 RMSE 表现；因此主线默认值切到 5B，而 A14B 保留为显式回退与对照路线；
- A14B/5B backend 的默认模型目录与 yaml 已统一迁移到 ExpHub 自身平台配置管理：模型通过 `platform.yaml -> models.wan2_2_fun_a14b_inp.path` 与 `models.wan2_2_fun_5b_inp.path` 指向 `/data/hx/models/exphub/infer` 下的统一模型仓，配置通过各自同项 `config` 指向仓内 `config/models/infer/Wan2.2-Fun-A14B-InP.yaml` 与 `config/models/infer/Wan2.2-Fun-5B-InP.yaml`；infer 已与 VideoX-Fun 的模型目录和配置目录解耦，但仍复用 `videox` conda 中的 Python 包；
- 5B 默认运行策略已不再继承 14B 的 qfloat8 路线：当前 profile 默认走更稳的非量化 `model_cpu_offload`，A14B 则继续保留现有量化路径，以便在同一前端下做稳定对照；
- 下游 `slam / eval / stats` 已可用于验证几何一致性、统计压缩率与汇总实验信息；
- 已新增 `segment_analyze.py` 研究旁路，可对既有 `segment/` 产物输出正式三策略的逐帧 kinematics / allocation / projection 分析；当前 `--mode segment` 与 `--mode all` 都会在 `segment` 成功后默认立即自动触发该分析旁路（可用 `--skip_analyze` 关闭），并将研究输出收敛为 `segment_summary.json / segment_timeseries.csv / kinematics_overview.png / allocation_overview.png / comparison_overview.png / projection_overview.png` 六个正式产物；其中 `projection_overview.png` 专门回答 raw schedule -> deploy schedule 的投影偏移。正式 analyze 仍内建 active policy + passive observer 横向对比：`semantic ↔ motion` 默认互为 observer，`uniform` 同时观测两者；对比结果仅写 analyze 产物，不进入 prompt / infer / merge / slam 主链路；
- 各阶段与全流程耗时统计已纳入实验平台。

### 4.2 尚未完成或尚未正式接入部分

以下关键模块仍待补强：

- `segment` 默认策略仍为 `uniform`；`semantic` 与 `motion` 已作为当前正式收敛对象接入，其中 `semantic` 代表第一版固定预算语义采样方法，`motion` 则提供同预算骨架下的轻量 motion baseline；两者都不代表最终最优策略；
- `segment` 研究旁路当前正式只服务 `uniform / motion / semantic`，并围绕 shared allocator、kinematics 密度、observer alignment、关键帧重定位与 raw->deploy projection 进行分析；`semantic_guarded_v1 / semantic_guarded_v2` 已作为早期 rule-based 启发式策略移除，不再作为当前系统的正式方法或 analyze 主叙事；
- `prompt` 当前使用的语义表示仍相对基础，后续仍有较大优化空间；
- 语义一致性评测尚未纳入标准工作流；
- 当前生成结果是否真正“SLAM-friendly”仍需更系统的定量分析。

### 4.2.1 已移除的早期 guarded 策略

`semantic_guarded_v1 / semantic_guarded_v2` 曾是围绕 boundary/support 类启发式候选回接的早期探索版本。当前代码库已移除其正式策略身份、CLI 入口与 analyze 主路径，只保留与正式三策略仍有复用价值的通用信号/可视化支撑。

### 4.2.2 `semantic` 的当前定位

`semantic` 当前由 `semantic.py` 承载。该方法不再沿用 `boundary / support / suppressed` 的候选回接思路，而是引入一条更简洁的纯语义主方法：

- 仍以 uniform 关键帧布局作为骨架；
- 关键帧总数严格保持与 uniform 一致，不额外加帧，也不删帧；
- 首尾关键帧固定；
- 中间关键帧只允许在 uniform 邻域内做受限重定位；
- 语义密度由 OpenCLIP image embedding 的 `semantic displacement / velocity / acceleration` 构成；
- 通过固定预算的 cumulative action sampling，把更多关键帧分配到高语义变化区。

从研究角度看，`semantic` 的意义是把“语义变化驱动关键帧密度重分配”第一次以兼容主链路的方式接入正式 `segment` 输出，同时保留 uniform 作为安全兜底骨架。

### 4.2.3 `motion` 的当前定位

`motion` 当前由 `motion.py` 承载。它是当前阶段新增的正式 baseline，用来回答“如果保留与 `semantic` 完全相同的 fixed-budget allocation 骨架，只把输入信号替换成轻量 motion energy kinematics，会得到怎样的关键帧分配”：

- 仍以 uniform 关键帧布局作为骨架；
- 关键帧总数严格保持与 uniform 一致；
- 首尾关键帧固定；
- 中间关键帧仅允许在 uniform 邻域内做受限重定位；
- 输入信号由统一尺寸帧序列的灰度高斯模糊后相邻帧差分构成，并进一步计算 `motion displacement / velocity / acceleration / density / cumulative action`；
- `relocate_radius / min_gap / snap_radius / density` 默认参数与 `semantic` 保持一致。

从研究角度看，`motion` 的价值在于提供一个不依赖 OpenCLIP 的轻量对照组，使后续实验可以直接比较“同一 allocator 框架下，semantic kinematics 与 motion energy kinematics 谁更能为 VSLAM 关键帧预算重分配提供有效信号”。

### 4.3 当前系统定位

当前 ExpHub 更准确地说是一个**面向探索性研究的实验平台**，而不是一个可以直接部署到真实机器人系统中的完整产品原型。

它的首要价值在于：

- 快速验证研究想法；
- 支撑可重复实验；
- 建立编码、解码、SLAM 验证与指标分析的统一闭环；
- 为后续算法替换与论文写作提供工程基础。

---

## 5. 评测维度

本项目当前及后续应围绕四个维度建立评测体系。

### 5.1 Transmission Efficiency：传输效率

该维度关注“上传了多少数据”。

可包括但不限于：

- 压缩率；
- 上传数据量；
- 关键帧占比；
- prompt / 语义表示体积；
- 总体边→云传输负担。

该维度回答的问题是：

**所提出的语义编解码流程，是否真正降低了通信负载？**

### 5.2 Geometric Consistency：几何一致性

该维度关注“恢复结果还能否被 SLAM 使用”。

可包括但不限于：

- SLAM 成功率；
- ATE；
- RPE；
- 轨迹连续性；
- 重定位表现；
- 建图稳定性。

该维度回答的问题是：

**恢复图像是否仍然保留了足够的几何信息来支撑下游 VSLAM？**

### 5.3 Semantic Consistency：语义一致性

该维度关注“恢复结果是否保留了原始场景的语义内容”。

可包括但不限于：

- 图文一致性；
- 基于 VLM 的语义相似度；
- 场景内容保持程度；
- 跨帧语义连续性；
- prompt-following 能力。

该维度回答的问题是：

**编码端表达的语义信息，是否在解码后被正确保留并连续地体现在图像序列中？**

目前该维度仍属于平台中的薄弱环节，后续需正式纳入标准实验流程。

### 5.4 Runtime and System Cost：运行耗时与系统代价

该维度关注“整个方案要花多久、代价多大”。

可包括但不限于：

- 各 step 耗时；
- 全流程总耗时；
- `infer` 阶段耗时；
- 关键阶段耗时占比；
- 云端生成带来的计算负担。

这一维度非常重要，因为当前 `infer` 阶段依然明显偏慢，往往需要十几分钟甚至更久。

#### 工程视角

从工程部署角度看，当前时延仍然偏高：

- 一次完整实验通常需要较长时间；
- `infer` 是当前最主要的性能瓶颈；
- 这一时延水平目前并不适合直接面向真实机器人系统落地。

#### 学术视角

从学术探索角度看，当前阶段允许在“云端算力充足”的前提下先验证方法可行性：

- 本课题首先关注该范式是否成立；
- 当前服务器性能有限，与 H100 级大规模集群差距明显；
- 当前极致压缩在一定程度上以生成时间为代价；
- 随着更强算力与更强基础模型出现，耗时问题有望持续改善。

#### 长期优化意义

耗时问题不仅影响工程可部署性，也直接影响实验效率：

- `infer` 越慢，实验迭代越慢；
- 实验迭代越慢，方法对比与改进节奏越受限；
- 因此，降低运行时间应被视为长期独立优化支线，而不是可忽略的问题。

#### 核心权衡

本课题当前本质上是在做如下权衡：

- **更低的传输代价**
- 交换
- **更高的云端计算代价**

这一 trade-off 不应被回避，而应被持续记录、量化与分析。

---

## 6. 开发协作流程

本项目采用“AI 辅助 + 人工主导”的开发闭环。

### 6.1 思考与讨论

在编码前，先围绕研究问题、实现思路、改动范围和风险点进行讨论，确保问题定义清晰。

### 6.2 指令生成

当方案明确后，由 ChatGPT 输出结构化、可验收、可约束的实现指令，交给 Codex 等编程模型执行。

### 6.3 按项目规则实施开发

Codex 的开发行为必须严格遵循根目录 `AGENTS.md`。  
`AGENTS.md` 是项目级最高约束。

### 6.4 本地开发与分支管理

实际开发过程中，交替使用 VS Code 与 PyCharm 进行工程实践与能力训练；以 GitHub 进行分支式代码管理；阶段性任务应尽量在独立分支完成。

### 6.5 自动检查与轻量测试

在每次阶段性改动后，应优先完成：

- 语法检查；
- 快速 smoke test；
- doctor 体检；
- 必要的无重推理轻量验证。

### 6.6 人工长链路测试

对于 `infer` 等耗时较长的步骤，不要求每次由 Codex 自动全量执行。此类步骤允许由人工手动测试，并以人工结果作为最终行为确认依据。

### 6.7 文档同步

代码改动后，`docs/` 下相关说明必须同步更新，保证代码、文档与研究语境三者保持一致。

### 6.8 反馈与下一轮迭代

改动完成并测试后，应将：

- 代码变化；
- 测试结果；
- 文档变化；
- 已发现问题；

反馈到下一轮讨论中，形成持续闭环。

### 6.9 commit / merge

只有在实现、测试与文档同步都完成后，才允许进入提交与合并阶段。

---

## 7. 工程约束

### 7.1 `AGENTS.md` 为最高优先级约束

所有 AI 编码任务必须服从 `AGENTS.md`。

### 7.2 一个分支只做一类事情

一个分支应尽量只承载一类主任务，例如：

- feature；
- fix；
- refactor；
- docs update。

### 7.3 代码与文档同步维护

任何行为变化、接口变化、主链路变化、评测方式变化，都应同步体现在 `docs/` 中。

### 7.4 轻量自动化 + 重测试人工兜底

静态检查与轻量测试尽可能自动化；长耗时推理验证允许人工执行。

### 7.5 实验可复现

实验命令、关键参数与核心输出应可追踪、可复现、可对比。

### 7.6 先讨论，后实现

对非平凡任务，应先完成问题框定与方案讨论，再交由 AI 编码工具实施。

### 7.7 不仅要“能跑”，还要服务研究目标

一个改动成功与否，不应只看是否跑通，还应看它对以下目标的影响：

- 传输效率；
- 几何一致性；
- 语义一致性；
- 耗时表现；
- 长期可维护性。

---

## 8. 核心术语

### Edge
边端，指受资源约束的机器人侧或设备侧计算节点。

### Cloud
云端，指具备更强算力的远程计算节点。

### Keyframe
关键帧，指被显式保留并上传的高价值视觉锚点。

### Non-keyframe
非关键帧，指不以原始图像形式上传，而以轻量语义形式表达的帧。

### I2T
Image-to-Text。广义上指图像到语义表征的编码过程。

### I&T2V
Image-and-Text-to-Video。广义上指基于关键帧与语义条件恢复视频或图像序列的过程。

### VLM
视觉语言模型。主要用于图像理解、语义描述与潜在的语义关键帧选择。

### World Model
在本项目语境中，指作为云端解码核心的图像/视频生成模型。

### Geometric Consistency
恢复序列在几何层面保持可用于下游 SLAM 的程度。

### Semantic Consistency
恢复序列在语义内容与时序语义连续性上保持原始场景的程度。

### SLAM-friendly Generation
不仅视觉上合理，而且对下游 SLAM 实际可用的生成结果。

### Transmission Cost
边→云上传过程中的通信代价。

### Computation Cost
云端生成恢复带来的计算代价。

---

## 9. 下一阶段里程碑

### 9.1 将 IROS 关键帧方法接入 `segment`

使关键帧选择从等距采样升级为语义/信息增量驱动。

### 9.2 优化 `prompt` 的语义表示形式

探索自然语言之外更适合生成模型利用的中间表示。

### 9.3 将语义一致性纳入标准评测链路

避免只看几何、不看语义。

### 9.4 深化面向 SLAM 的解码结果分析

进一步识别哪些生成结果对 SLAM 有帮助，哪些会造成干扰。

### 9.5 持续跟踪并优化 `infer` 耗时

将耗时作为长期支线任务持续改进。

### 9.6 建立更强的基线与对比实验

逐步补齐与原始传输、简单抽帧、传统压缩方法及后续语义传输变体之间的对比。

---

## 10. 维护说明

本文档应随项目持续更新。

当以下任一情况发生时，应主动检查并更新本文档：

- 研究目标发生变化；
- 某个 workflow step 的语义角色发生变化；
- 新的关键实现里程碑达成；
- 新的评测维度或核心指标被引入；
- AI+人工开发流程发生调整。

若本文档与实际代码行为不一致，应将其视为文档债务，并在后续迭代中尽快修复。

本文档的长期目标是同时承担以下角色：

- 研究语境锚点；
- 新聊天快速对齐材料；
- 项目长期维护说明书；
- 连接科研思路与工程实现的桥梁。
