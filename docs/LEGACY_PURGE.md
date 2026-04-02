# Legacy Purge

> 记录 2026-04 这轮 legacy purge / repo slim-down / standardization 清理的目标与结果。

## 本轮做了什么

- 删除退出正式 workflow 的旧 `scripts/_prompt`、`scripts/_infer`、`scripts/_segment` 实现树
- 删除旧 `scripts/_schedule.py`、`scripts/_common.py`、`scripts/_infer_i2v_impl.py`
- 删除失效顶层规则文件与旧过程文档
- 把正式主链仍需的实现收回 `exphub/pipeline/`
- 清理高频可见旧术语与旧入口说明

## 当前正式实现归属

当前正式实现集中在：

- `exphub/common/`
- `exphub/contracts/`
- `exphub/pipeline/`

## 清理后的仓库约束

- 旧实现树不再与正式主链并存
- `tools/` 只应用于非正式主链工具
- 文档只保留当前有效结构说明
- 兼容性残留如果仍存在，只允许留在局部内部字段或清理逻辑中，不再作为正式工作流叙述
