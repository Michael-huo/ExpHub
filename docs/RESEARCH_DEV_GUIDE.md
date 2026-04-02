# ExpHub 研究与开发指南

> 本文回答什么问题：ExpHub 这条主链路服务什么研究问题、当前实验重点放在哪里、开发协作应该如何围绕研究目标展开。

系统事实请以 [ARCHITECTURE.md](./ARCHITECTURE.md) 和 [PIPELINE_CONTRACT.md](./PIPELINE_CONTRACT.md) 为准；本文只讨论研究语境与开发方法。

## 1. 研究问题

ExpHub 服务的核心问题是：

**能否把边到云的图像上传过程，重写为“关键帧 + 轻量语义表示”的语义编解码链路，并仍然支撑下游 VSLAM。**

从研究视角看，这条链路分为三段：

- 编码端：关键帧选择 + 语义压缩
- 解码端：基于关键帧与语义条件恢复视觉序列
- 验证端：检查恢复结果是否仍然对 SLAM 有用

## 2. 工作流与研究映射

当前标准工作流是：

`segment -> prompt -> infer -> merge -> slam -> eval -> stats`

研究意义最强的三个阶段是：

- `segment`：关键帧预算如何分配
- `prompt`：图像如何被压缩成可传输的语义条件
- `infer`：关键帧与语义条件能否恢复出足够好的序列

## 3. 当前实验口径

当前系统的研究口径应按下面理解：

- `segment` 正式策略只保留 `uniform` 与 `state`
- `state` 是当前正式研究主线
- raw keyframe 事实源在 `segment/keyframes/keyframes_meta.json`
- 执行投影在 `segment/deploy_schedule.json`
- `segment/state_segmentation/state_segments.json` 是 state 区间事实源
- `prompt` 主链路收敛到 `PromptProfile v1 -> base_prompt.json + state_prompt_manifest.json + runtime_prompt_plan.json`
- `infer` 默认 backend 是 `wan_fun_5b_inp`
- `wan_fun_a14b_inp` 保留为显式回退/对照路线

旧 prompt schema、旧桥接产物和旧双轨说明不再是当前研究主叙事。

## 4. 重点评测维度

当前实验至少围绕四个维度看结果：

- 传输效率
- 几何一致性
- 语义一致性
- 运行代价

当前最需要持续补强的是：

- 语义一致性评测标准化
- 面向 SLAM 的生成结果分析
- `infer` 时延优化

## 5. 当前系统定位

ExpHub 当前更接近：

- 一个研究平台
- 一个实验编排器
- 一个可复现实验闭环

判断一个改动是否值得做，不能只看能不能跑通，还要看它是否帮助研究比较、实验复现和结果解释。

## 6. 推荐开发流程

1. 先明确研究问题和变更边界
2. 再确认主链路、契约和日志口径
3. 再做最小实现或最小文档改动
4. 做轻量检查与人工长链路验证分工
5. 同步更新相关文档

默认分工：

- AI 负责实现、静态检查、文档同步与轻量验证
- 人工负责耗时较长的推理与全链路实测

## 7. 工程协作约束

- 一个分支尽量只做一类事情
- 代码、文档、研究口径必须同步
- 长耗时测试允许人工兜底，但默认行为说明不能依赖口头记忆
- 正式实现应继续集中在 `exphub/common`、`exphub/contracts`、`exphub/pipeline`

## 8. 近期里程碑

近期更值得持续推进的方向是：

- 把关键帧策略从安全骨架继续推进到更强的研究方法
- 优化 prompt 语义表示，而不是重新引入旧中间层
- 将语义一致性正式纳入标准评测
- 持续压缩 `infer` 的时间成本
