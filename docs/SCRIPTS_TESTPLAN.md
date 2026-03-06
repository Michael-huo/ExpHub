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