# AGENTS.md — ExpHub 项目规则（Codex / 编程 AI）

## 0. 项目入口与环境
- 主入口：`python -m exphub ...`
- 默认：`keep_level=repro`
- Python：必须兼容 **Python 3.7**
- 禁止使用 3.8+ 特性（如 typing.Literal / Path.unlink(missing_ok=True) 等）
- 外部仓库（只读引用，禁止修改）：`/data/hx/*`

---

## 1. 硬约束（必须遵守）
1) EXP_NAME 命名规则绝对不变：
   `{tag}_{w}x{h}_t{start}s_dur{dur}s_fps{fps}_gap{kf_gap}`

2) 不改变实验语义与产物契约（除非任务明确要求）：
   目录结构必须保持：
   segment/ prompt/ infer/ merge/ slam/<track>/ eval/ stats/

3) 关键文件格式不可随意变更：
   - prompt/manifest.json
   - slam/<track>/traj_est.tum
   - slam/<track>/run_meta.json
   - stats/report.json

4) `--auto_conda` 必须严格执行环境切换逻辑。

---

## 2. Docs 同步强制规则（新增）

任何影响以下内容的代码修改，都必须同步更新 docs：

- step 行为逻辑变化
- 新增/删除 mode
- 目录结构变化
- 新增/删除关键产物文件
- 日志结构或输出规范变化
- run_meta / step_meta 字段变化

必须更新的对应文档：

- 架构变化 → `docs/ARCHITECTURE.md`
- 目录结构变化 → `docs/PROJECT_MAP.md`
- 日志变化 → `docs/LOGGING.md`
- 测试流程变化 → `docs/SCRIPTS_TESTPLAN.md`
- 脚本输入输出契约变化 → `docs/SCRIPTS_AUDIT.md`

如果未更新 docs，视为任务未完成。

---

## 3. SAFE MODE（默认）

允许修改：
- exphub/**
- docs/**
- scripts/**（仅当任务明确涉及）

禁止修改：
- /data/hx/*

优先修改原逻辑，禁止通过叠加分支制造冗余。

---

## 4. REFACTOR 模式（必须显式声明 REFACTOR_OK=1）

允许：
- 删除死代码
- 合并重复逻辑
- 重命名变量 / 函数
- 重排内部模块结构
- 提取公共函数

要求：
- 不改变实验语义
- 不改变目录契约
- 必须同步更新 docs
- 解释为什么采用重构而不是补丁

---

## 5. 验收要求（提交前必须完成）

1) 语法检查：
   python -m py_compile exphub/*.py scripts/*.py

2) doctor 模式必须 PASS：
   python -m exphub --mode doctor ...

3) 至少一个轻量 step 运行成功（segment 或 stats）

4) 若跳过 GPU 重型步骤，必须说明原因。

---

## 6. 优先级原则

代码整洁性 > 冗余补丁  
语义稳定性 > 日志美观  
实验可复现性 > 结构花哨