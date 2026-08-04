# Changelog

## 2026-08-04

- 新增固定 global55 pre-state 的 MuJoCo friction-cone 单因素反事实工具，对同一完整 `mjData` snapshot 依次验证 pyramidal、elliptic 与恢复后的 pyramidal 条件。
- 新增 cone constraint parameterization、physical impulse、shared global rigid demand、solver-excess vector/norm、baseline/restore reproduction 和无条件 ZIP 打包证据。
- 修正 MuJoCo global55 contact-demand oracle 的证据口径：显式记录 generalized DOF 顺序，并以 pre-integration `mjData` 克隆读取一致的质量矩阵、接触约束和 Jacobian。
- 新增 Euler implicit-damping integration matrix、generalized velocity closure、constraint closure、physical-contact impulse mapping 与旧 artifact 数值回归。
- 更新正式服务器脚本，使单次 120-substep replay 无论成功或失败都自动生成、校验并哈希 ZIP artifact。
