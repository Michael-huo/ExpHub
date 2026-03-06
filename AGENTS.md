# ExpHub AI 辅助开发宪法 (AGENTS.md)

> **[全局使命]**
> ExpHub 是一个高度模块化的视频流与 VSLAM 实验平台。任何参与本项目开发的 AI Agent，在生成代码、分析架构或进行底层优化时，**必须且仅能**遵循本文件中的核心规范。

## 1. 核心开发原则：安全与优化并重
- **核心功能兜底 (Safe Refactoring)**：在进行任何底层的深度优化（如提高量化效率、重构数据管道）时，必须保证核心主链路（`segment` -> `prompt` -> `infer` -> `merge` -> `slam` -> `eval` -> `stats`）的数据流转与功能完整性不受破坏。
- **无损后向兼容**：修改现有函数的签名或数据结构（如 JSON meta 文件）时，必须考虑到现存历史实验数据的兼容性，或提供平滑的迁移/处理逻辑。

## 2. 绝对编码红线 (Absolute Red Lines)
- **Python 3.7 兼容性**：目标环境可能运行在较老的 Conda 环境中。绝对禁止使用 Python 3.8+ 语法（如海象操作符 `:=`、`math.prod`），类型提示必须兼容 `typing` 模块（如使用 `List[str]` 而非 `list[str]`）。
- **极致环境解耦 (Decoupling)**：严禁在代码中硬编码任何绝对路径（如 `/data/hx/...`）。所有第三方算法库路径 (Repos)、模型资产 (Models) 以及跨环境 Python 解释器路径 (Environments)，**必须**通过 `scripts/_common.py` 中的 `get_platform_config()` 从 `config/platform.yaml` 动态读取。

## 3. 跨环境与子进程调度 (Cross-Environment)
- **Conda 破壁法则**：严禁在代码里拼接 `bash -lc "conda activate ..."`。
  - 顶层调度：必须在 `exphub/runner.py` 或 `cli.py` 中读取 `platform.yaml` 中目标环境的绝对解释器路径，并使用 `run_env_python()` 拉起子进程。
  - 底层继承：底层脚本若需在同环境内继续拉起子模块（如使用 `torchrun` 或多进程），必须使用 `sys.executable` 确保解释器一致性，禁止直接依赖系统 `PATH`。

## 4. 日志与可观测性契约 (Logging Facade)
- **禁用原生 Print**：严禁在业务逻辑中直接使用 `print()`。所有输出必须通过 `scripts/_common.py` 导入的日志门面（`log_info`, `log_warn`, `log_err`, `log_prog`）。
- **进度条与 UI 纯净**：
  - 凡使用 `tqdm`，必须配置特定的 `bar_format`，并加上 `[BAR]` 前缀以触发顶层路由的原地 `\r` 刷新，严禁污染落盘的 `.log` 文件。
  - 外部 C++ 库（如 SLAM 底层）的日志必须在 Python 层被静音（`disable=True`）或通过 `subprocess.PIPE` 严格过滤。
- **高精度心跳打点**：在长耗时任务（如模型加载、量化、推理）的前后，必须输出明确的耗时打点（`[INFO]` 心跳），以支撑 `EXPERIMENT PERFORMANCE PROFILING` 面板的性能分析。

## 5. 文档强制同步 (Mandatory Docs Sync)
- **Docs-as-Code (文档即代码)**：代码与文档具有同等地位。AI 必须根据修改内容，将更新精准路由到以下职责对应的文档中：
  - **项目概览** → `docs/OVERVIEW.md`（高层级愿景与整体介绍）
  - **系统架构** → `docs/ARCHITECTURE.md`（调用链路、平台化调度机制）
  - **目录结构** → `docs/PROJECT_MAP.md`（文件树与各目录作用）
  - **日志规则** → `docs/LOGGING.md`（终端 UI、前缀规范与落盘机制）
  - **测试流程** → `docs/SCRIPTS_TESTPLAN.md`（如何运行跑通各模块）
  - **脚本IO契约** → `docs/SCRIPTS_AUDIT.md`（各阶段输入输出产物、参数清单）
- **AI 强制自我审查**：AI 在完成一个阶段性的 Feature 开发后，必须主动审视 `docs/` 目录下的相关文档，并提供相应的更新文案。

## 6. 交付与验收标准 (Acceptance Criteria)
任何新功能或重构提交前，AI 需要确保代码能够通过以下快速验收（耗时的全链路测试由人类手动执行）：
1. **零硬编码扫描**：全局搜索必须确认已清理所有遗漏的本地绝对路径。
2. **零污染扫描**：终端输出的日志格式必须严格遵守前缀契约（`[INFO]`, `[BAR]` 等）。
3. **静态语法检查**：能够通过命令 `python -m py_compile exphub/*.py scripts/*.py` 且无报错。
4. **环境与配置体检**：必须能够通过 Doctor 模式的安全扫描：`python -m exphub --mode doctor ...` 必须提示 PASS。