import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "analyze_mujoco_global55_contact_demand.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_mujoco_global55_contact_demand", MODULE_PATH
)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


class MujocoGlobal55ContactDemandTests(unittest.TestCase):
    def test_generalized_manifest_explicit_19_dofs(self):
        class Joint:
            mjJNT_FREE = 0
            mjJNT_BALL = 1
            mjJNT_SLIDE = 2
            mjJNT_HINGE = 3

        mujoco = SimpleNamespace(mjtJoint=Joint)
        model = SimpleNamespace(
            nv=19,
            njnt=14,
            nbody=15,
            dof_jntid=np.asarray([0] * 6 + list(range(1, 14))),
            jnt_type=np.asarray([0] + [3] * 13),
            jnt_dofadr=np.asarray([0] + list(range(6, 19))),
            jnt_qposadr=np.asarray([0] + list(range(7, 20))),
            jnt_bodyid=np.arange(14),
            jnt_axis=np.tile([0.0, 1.0, 0.0], (14, 1)),
        )
        names = {
            "joint": ["root"] + [f"limby/{index}" for index in range(13)],
            "body": ["torso"] + [f"limb/{index}" for index in range(14)],
        }
        with mock.patch.object(
            ORACLE.evaluator,
            "_runtime_object_names",
            side_effect=lambda _model, kind, count: names[kind][:count],
        ):
            manifest = ORACLE.build_generalized_dof_order(mujoco, model)
        self.assertEqual(manifest["GENERALIZED_DOF_ORDER"], "EXPLICIT")
        self.assertEqual(manifest["GENERALIZED_DOF_COUNT"], 19)
        self.assertEqual(manifest["free_joint_dof_count"], 6)
        self.assertEqual(manifest["scalar_joint_dof_count"], 13)
        self.assertEqual(
            [item["coordinate_label"] for item in manifest["dofs"][:6]],
            [
                "root_translation_x", "root_translation_y", "root_translation_z",
                "root_rotation_x", "root_rotation_y", "root_rotation_z",
            ],
        )
        self.assertIn("global-frame", manifest["dofs"][0]["coordinate_semantics"])
        self.assertIn("local body-frame", manifest["dofs"][3]["coordinate_semantics"])

    def test_euler_integration_matrix_with_and_without_implicit_damping(self):
        mass = np.asarray([[2.0, 0.25], [0.25, 3.0]])
        damping = np.asarray([1.0, 4.0])
        enabled = {
            "model_opt_integrator_enum_name": "mjINT_EULER",
            "euler_implicit_joint_damping_effective": True,
        }
        matrix, report = ORACLE.build_integration_matrix(
            mass, 0.005, damping, enabled
        )
        np.testing.assert_allclose(matrix, mass + 0.005 * np.diag(damping))
        np.testing.assert_allclose(report["delta"] - np.diag(np.diag(report["delta"])), 0.0)
        disabled = dict(enabled, euler_implicit_joint_damping_effective=False)
        matrix, report = ORACLE.build_integration_matrix(
            mass, 0.005, damping, disabled
        )
        np.testing.assert_allclose(matrix, mass)
        self.assertEqual(report["INTEGRATION_MATRIX_CONSTRUCTION"], "VALIDATED")

    def test_unsupported_integrator_fails_closed(self):
        matrix, report = ORACLE.build_integration_matrix(
            np.eye(2), 0.005, np.ones(2),
            {"model_opt_integrator_enum_name": "mjINT_RK4"},
        )
        self.assertIsNone(matrix)
        self.assertEqual(report["INTEGRATION_MATRIX_CONSTRUCTION"], "UNSUPPORTED")

    def test_velocity_closures_use_pre_jacobian_and_effective_matrix(self):
        capture = {
            "nv": 3,
            "pre_state": {"qvel": np.zeros(3)},
            "post_state": {"qvel": np.asarray([0.1, 0.2, 0.3])},
            "pre_simulation_time": 0.0,
            "post_simulation_time": 0.1,
            "J_phys": np.eye(3),
            "solver_phase_state": {
                "qfrc_smooth": np.asarray([1.0, 0.0, 0.0]),
                "qfrc_constraint": np.asarray([0.0, 2.0, 3.0]),
                "qfrc_smooth_source": "synthetic",
            },
            "contacts": [{
                "normal_impulse": 0.0,
                "tangential_impulse": np.asarray([0.2, 0.3]),
                "formal_physical_projection": {
                    "qfrc_total": [0.0, 2.0, 3.0],
                    "qfrc_constraint_rows_contact": [0.0, 2.0, 3.0],
                },
            }],
        }
        closures = ORACLE.velocity_closures(capture, np.eye(3))
        self.assertEqual(closures["generalized"]["GENERALIZED_VELOCITY_CLOSURE"], "PASS")
        self.assertEqual(closures["constraint"]["CONTACT_CONSTRAINT_GENERALIZED_CLOSURE"], "PASS")
        self.assertEqual(closures["physical"]["PHYSICAL_CONTACT_IMPULSE_MAPPING"], "PASS")
        np.testing.assert_allclose(
            closures["constraint"]["predicted_contact_velocity_delta_from_all_constraints"],
            [0.0, 0.2, 0.3],
        )

    def test_physical_mapping_is_partial_with_nonfloor_constraints(self):
        capture = {
            "nv": 3,
            "pre_state": {"qvel": np.zeros(3)},
            "post_state": {"qvel": np.asarray([0.1, 0.2, 0.4])},
            "pre_simulation_time": 0.0,
            "post_simulation_time": 0.1,
            "J_phys": np.eye(3),
            "solver_phase_state": {
                "qfrc_smooth": np.asarray([1.0, 0.0, 0.0]),
                "qfrc_constraint": np.asarray([0.0, 2.0, 4.0]),
                "qfrc_smooth_source": "synthetic",
            },
            "contacts": [{
                "normal_impulse": 0.0,
                "tangential_impulse": np.asarray([0.2, 0.3]),
                "formal_physical_projection": {
                    "qfrc_total": [0.0, 2.0, 3.0],
                    "qfrc_constraint_rows_contact": [0.0, 2.0, 3.0],
                },
            }],
        }
        closures = ORACLE.velocity_closures(capture, np.eye(3))
        self.assertEqual(closures["physical"]["PHYSICAL_CONTACT_IMPULSE_MAPPING"], "PARTIAL")

    def test_clone_forward_does_not_mutate_live_data(self):
        class Data:
            def __init__(self, model=None):
                self.time = 0.0
                for name in (
                    "qpos", "qvel", "act", "ctrl", "qfrc_applied",
                    "xfrc_applied", "mocap_pos", "mocap_quat", "qacc_warmstart",
                ):
                    setattr(self, name, np.asarray([1.0, 2.0]))

        class Mujoco:
            MjData = Data
            copy_count = 0
            forward_count = 0

            @classmethod
            def mj_copyData(cls, destination, model, source):
                cls.copy_count += 1
                destination.time = source.time
                for name in ORACLE.live_data_fingerprint(source):
                    if name not in ("time", "ncon", "nefc") and hasattr(source, name):
                        setattr(destination, name, getattr(source, name).copy())

            @classmethod
            def mj_forward(cls, model, data):
                cls.forward_count += 1
                data.qpos[0] = 99.0

        live = Data()
        before = ORACLE.live_data_fingerprint(live)
        _, evidence = ORACLE.clone_and_forward_preintegration_data(
            Mujoco, SimpleNamespace(), live
        )
        self.assertTrue(ORACLE.fingerprints_equal(before, ORACLE.live_data_fingerprint(live)))
        self.assertTrue(evidence["live_data_unchanged_by_probe"])
        self.assertEqual(Mujoco.copy_count, 1)
        self.assertEqual(Mujoco.forward_count, 1)

    def test_physical_basis_is_right_handed_and_orthonormal(self):
        frame = np.asarray(
            [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
        )
        for side in (-1, 1):
            with self.subTest(side=side):
                basis = ORACLE.physical_basis(frame, side)
                np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-15)
                self.assertAlmostEqual(np.linalg.det(basis), 1.0)
                np.testing.assert_allclose(basis[0], side * frame[0])

    def test_point_jacobian_matches_independent_rigid_velocity(self):
        class Mujoco:
            @staticmethod
            def mj_jac(model, data, jacp, jacr, point, body_id):
                jacp[:] = np.asarray([[1.0], [0.0], [0.0]])

            @staticmethod
            def mj_jacBody(model, data, jacp, jacr, body_id):
                jacp[:] = np.asarray([[2.0], [0.0], [0.0]])
                jacr[:] = np.asarray([[0.0], [0.0], [1.0]])

        model = SimpleNamespace(nv=1)
        data = SimpleNamespace(
            xpos=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        )
        jacobian, check = ORACLE.point_jacobian_and_velocity(
            Mujoco,
            model,
            data,
            1,
            np.asarray([0.0, 1.0, 0.0]),
            np.asarray([2.0]),
        )
        np.testing.assert_allclose(jacobian, [[1.0], [0.0], [0.0]])
        self.assertLess(check["mapping_max_abs_error"], 1e-15)
        self.assertEqual(
            ORACLE.PHYSICAL_JACOBIAN_CONTRACT["PHYSICAL_JACOBIAN_CONVENTION"],
            "WORLD_POINT_AT_CONTACT",
        )
        self.assertEqual(
            ORACLE.PHYSICAL_JACOBIAN_CONTRACT["PHYSICAL_JACOBIAN_POINT"],
            "exact mjContact.pos",
        )

    def test_mj_fullM_shape_symmetry_and_linear_solve(self):
        expected = np.asarray([[4.0, 1.0], [1.0, 3.0]])

        class Mujoco:
            @staticmethod
            def mj_fullM(model, target, packed):
                target[:] = expected

        model = SimpleNamespace(nv=2)
        data = SimpleNamespace(
            qM=np.asarray([4.0, 1.0, 3.0]),
            qfrc_constraint=np.asarray([1.0, 2.0]),
        )
        mass, stats = ORACLE.expanded_mass_matrix(Mujoco, model, data)
        np.testing.assert_allclose(mass, expected)
        self.assertEqual(stats["mass_matrix_shape"], [2, 2])
        self.assertEqual(stats["symmetry_max_abs_error"], 0.0)
        self.assertGreater(stats["minimum_eigenvalue"], 0.0)
        self.assertLess(
            stats["linear_solve_check"]["residual_max_abs"], 1e-15
        )

    def test_delassus_construction_is_symmetric_and_solve_closes(self):
        mass = np.asarray([[3.0, 0.2], [0.2, 2.0]])
        jacobian = np.asarray([[1.0, 2.0], [-0.5, 1.0]])
        solution, solve = ORACLE.stable_solve(mass, jacobian.T)
        delassus = jacobian @ solution
        np.testing.assert_allclose(delassus, delassus.T, atol=1e-15)
        self.assertLess(solve["residual_max_abs"], 1e-15)

    def test_force_to_impulse_uses_one_physics_substep(self):
        impulse = ORACLE.force_to_impulse(
            [1269.0480557934907, 664.9472720133347], 0.005
        )
        np.testing.assert_allclose(
            impulse,
            [6.345240278967453, 3.3247363600666735],
            rtol=0.0,
            atol=1e-15,
        )

    def test_contact_row_mapping_uses_runtime_address_and_dimension(self):
        rows = ORACLE.evaluator.contact_efc_rows(
            efc_address=7, dim=3, pyramidal=True, nefc=20
        )
        self.assertEqual(rows, [7, 8, 9, 10])

    def test_global55_regression_requires_distal_values_and_mirror(self):
        def contact(geom):
            return {
                "geom1_name": geom,
                "geom2_name": "floor/0",
                "formal_physical_projection": {
                    "Fn": ORACLE.REGRESSION["Fn"],
                    "friction_force_norm": ORACLE.REGRESSION["Ft_norm"],
                    "selected_joints": {
                        joint: {
                            "normal": ORACLE.REGRESSION["normal_generalized"],
                            "friction": ORACLE.REGRESSION["friction_generalized"],
                            "total": ORACLE.REGRESSION["total_generalized"],
                        }
                        for _, joint in ORACLE.SELECTED
                    },
                },
            }

        capture = {"contacts": [contact("limb/12"), contact("limb/11")]}
        regression = ORACLE.regression_check(capture)
        self.assertEqual(regression["MUJOCO_GLOBAL55_REGRESSION"], "PASS")
        capture["contacts"][0]["formal_physical_projection"]["Fn"] += 1.0
        regression = ORACLE.regression_check(capture)
        self.assertEqual(regression["MUJOCO_GLOBAL55_REGRESSION"], "FAIL")

    def test_old_artifact_contact_demand_regression(self):
        metric = {
            "directional_effective_mass_kg": 4.2,
            "uncoupled_sticking_impulse_norm": 2.3,
            "global_normal_conditioned_sticking_impulse_norm": 2.5,
            "actual_normal_impulse": 6.3,
            "actual_tangential_impulse_norm": 3.3,
            "pre_tangential_speed": 0.55,
        }
        contacts = [
            {
                "geom1_name": geom,
                "geom2_name": "floor/0",
                "post_tangential_speed": 0.17,
            }
            for geom, _ in ORACLE.SELECTED
        ]
        capture = {"contacts": [dict(item) for item in contacts]}
        budget = {"selected": {geom: dict(metric) for geom, _ in ORACLE.SELECTED}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "global55_effective_mass_budget.json").write_text(
                json.dumps({"selected": budget["selected"]}), encoding="utf-8"
            )
            (path / "global55_contacts.json").write_text(
                json.dumps({"all_active_robot_floor_contacts": contacts}), encoding="utf-8"
            )
            report = ORACLE.load_old_artifact_regression(path, budget, capture)
            self.assertEqual(report["OLD_ARTIFACT_REGRESSION"], "PASS")
            budget["selected"]["limb/12"]["pre_tangential_speed"] += 0.1
            report = ORACLE.load_old_artifact_regression(path, budget, capture)
            self.assertEqual(report["OLD_ARTIFACT_REGRESSION"], "FAIL")

    def test_hash_readback_does_not_modify_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.xml"
            path.write_bytes(b"<mujoco/>")
            before = path.read_bytes()
            first = ORACLE.sha256(path)
            second = ORACLE.sha256(path)
            self.assertEqual(first, second)
            self.assertEqual(path.read_bytes(), before)

    def test_artifact_packaging_verifies_and_writes_sha_sidecar(self):
        for status in ("success", "failure"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / f"artifact_{status}"
                output.mkdir()
                (output / "summary.json").write_text(
                    '{"status":"' + status + '"}\n', encoding="utf-8"
                )
                archive = root / f"artifact_{status}.zip"
                report = ORACLE.package_artifact(output, archive)
                self.assertEqual(report["ZIP_VERIFY"], "PASS")
                self.assertEqual(report["UPLOAD_THIS_ZIP"], str(archive.resolve()))
                self.assertTrue(Path(report["SHA256_SIDECAR"]).is_file())
                with zipfile.ZipFile(archive) as bundle:
                    self.assertIsNone(bundle.testzip())

    def test_server_wrapper_preserves_rc_and_prints_upload_path(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "tools"
            / "run_mujoco_global55_contact_demand_oracle.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("probe_rc=${PIPESTATUS[0]}", script)
        self.assertIn("package_artifact", script)
        self.assertIn("UPLOAD_THIS_ZIP=", script)
        self.assertNotIn("set -e", script)
        self.assertNotIn("\nexit ", script)
        self.assertNotIn("\nexec ", script)

    def test_formal_contract_is_120_substeps_with_only_clone_forward(self):
        self.assertEqual(ORACLE.CONTROL_STEPS, 30)
        self.assertEqual(ORACLE.EXPECTED_SUBSTEPS, 120)
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        calls = []
        for function in [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            for node in ast.walk(function):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr in {"mj_step", "mj_step1", "mj_step2", "mj_forward"}:
                        calls.append((function.name, node.func.attr))
        self.assertEqual(calls, [("clone_and_forward_preintegration_data", "mj_forward")])


if __name__ == "__main__":
    unittest.main()
