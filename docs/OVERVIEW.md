# ExpHub 项目概览

## 1. 项目目标

ExpHub 是一个实验平台：
segment → prompt → infer → merge → slam → eval → stats

用于研究：
- prompt 结构设计对 SLAM 精度的影响
- 压缩率 vs 轨迹精度 trade-off

---

## 2. 目录契约（不变量）

EXP_DIR 下必须包含：

segment/
prompt/
infer/
merge/
slam/<track>/
eval/
stats/

命名规则：
{tag}_{w}x{h}_t{start}s_dur{dur}s_fps{fps}_gap{kf_gap}

---

## 3. Docs 文件说明（非常重要）

- ARCHITECTURE.md  
  描述系统整体结构与调度逻辑

- PROJECT_MAP.md  
  描述目录结构与文件分布

- SCRIPTS_AUDIT.md  
  各脚本输入输出契约

- SCRIPTS_TESTPLAN.md  
  回归测试与验证流程

- LOGGING.md  
  日志结构与输出规范

修改代码后，必须检查是否需要更新上述文件。

---

## 4. 核心产物（验收标准）

必须存在：
- segment/frames/
- prompt/manifest.json
- merge/frames/
- slam/<track>/traj_est.tum
- stats/report.json

---

## 5. 常用命令

doctor：
python -m exphub --mode doctor ...

全流程：
python -m exphub --mode all ...

---

## 6. 当前优先研究方向

- prompt 消融实验
- 压缩率与精度分析
- 轨迹对比扩展（ori/gen/kf 等）

---

## 7. 开发原则

- 不改研究语义
- 保持可复现
- 减少代码冗余
- docs 与代码必须同步