# Changelog

## 2026-08-05

- 新增固定 global55 的 pyramidal friction `efc_aref` 单因素反事实诊断工具：从同一 pre-state staged forward，按 row geometry 拟合 normal/tangent reference-acceleration 分量，仅缩放切向分量，并输出 solver-excess、invariant、baseline/restore、step closure 与无条件 ZIP artifact。
- 新增对应单元测试，覆盖 staged pipeline、四行几何分解、非均匀摩擦系数、scale=0 激活、shared physical demand、分类与 ZIP 校验；未运行训练或 sweep。

## 2026-08-04

- 新增固定 global55 pre-state 的 MuJoCo friction-cone 单因素反事实工具，对同一完整 `mjData` snapshot 依次验证 pyramidal、elliptic 与恢复后的 pyramidal 条件。
- 新增 cone constraint parameterization、physical impulse、shared global rigid demand、solver-excess vector/norm、baseline/restore reproduction 和无条件 ZIP 打包证据。
- 修正 MuJoCo global55 contact-demand oracle 的证据口径：显式记录 generalized DOF 顺序，并以 pre-integration `mjData` 克隆读取一致的质量矩阵、接触约束和 Jacobian。
- 新增 Euler implicit-damping integration matrix、generalized velocity closure、constraint closure、physical-contact impulse mapping 与旧 artifact 数值回归。
- 更新正式服务器脚本，使单次 120-substep replay 无论成功或失败都自动生成、校验并哈希 ZIP artifact。
