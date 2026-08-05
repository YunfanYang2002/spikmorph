# Changelog

## 2026-08-05

- Fixed the regularization consumption audit to replay to populated global55 before runtime probes; added `--mode consumption-audit-only`, canonical fail-closed artifacts, and inner traceback/audit-phase packaging.

- 修正 contact regularization consumption audit：移除不存在的 `iefc_AR` 假设，按 `map_efc2iefc`/`map_iefc2efc` 同步并验证 MuJoCo 的 `iefc_R`/`iefc_D` island mirrors；独立 PGS probe 不再因可证明的 mirror 路径误 fail closed。

- 新增固定 global55 production-pyramidal contact-row regularization `R` 单因素反事实诊断：先审计 MuJoCo 的 `mj_makeConstraint`、`mj_projectConstraint`、`mj_fwdConstraint` 对 `efc_R`、`efc_D`、`efc_AR` 的实际消费路径；无法证明或检测到未覆盖的 island mirror 时 fail closed，并始终生成经过 `testzip()` 与 SHA256 校验的 ZIP artifact。
- 诊断只对 active floor-contact edge rows 使用 `R_SCALE_1_BEFORE`、`R_SCALE_0P1`、`R_SCALE_1_AFTER_RESTORE`，保持 pre-state、M/J/W、`efc_J`/`efc_vel`/`efc_aref` 与 production solver 选项不变；明确该干预是 pyramidal contact-edge R，而非纯切向 R。
- 当 MuJoCo wheel 未随附 C 源码时，新增隔离的 in-memory wheel behavioral probe：验证 `mj_projectConstraint` 对 R/D/AR 的一致更新以及 `mj_fwdConstraint` 对更新状态的消费；probe 失败仍保持 fail closed。

- 新增固定 global55 的 pyramidal friction `efc_aref` 单因素反事实诊断工具：从同一 pre-state staged forward，按 row geometry 拟合 normal/tangent reference-acceleration 分量，仅缩放切向分量，并输出 solver-excess、invariant、baseline/restore、step closure 与无条件 ZIP artifact。
- 新增对应单元测试，覆盖 staged pipeline、四行几何分解、非均匀摩擦系数、scale=0 激活、shared physical demand、分类与 ZIP 校验；未运行训练或 sweep。
- 修正 pre-constraint decomposition 不应读取尚未完成 solver 的 physical wrench sign；现在先用 contact metadata/J-row 完成几何分解，仅在 `mj_fwdConstraint` 后读取 `mj_contactForce`。

## 2026-08-04

- 新增固定 global55 pre-state 的 MuJoCo friction-cone 单因素反事实工具，对同一完整 `mjData` snapshot 依次验证 pyramidal、elliptic 与恢复后的 pyramidal 条件。
- 新增 cone constraint parameterization、physical impulse、shared global rigid demand、solver-excess vector/norm、baseline/restore reproduction 和无条件 ZIP 打包证据。
- 修正 MuJoCo global55 contact-demand oracle 的证据口径：显式记录 generalized DOF 顺序，并以 pre-integration `mjData` 克隆读取一致的质量矩阵、接触约束和 Jacobian。
- 新增 Euler implicit-damping integration matrix、generalized velocity closure、constraint closure、physical-contact impulse mapping 与旧 artifact 数值回归。
- 更新正式服务器脚本，使单次 120-substep replay 无论成功或失败都自动生成、校验并哈希 ZIP artifact。
