import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

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

    def test_hash_readback_does_not_modify_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.xml"
            path.write_bytes(b"<mujoco/>")
            before = path.read_bytes()
            first = ORACLE.sha256(path)
            second = ORACLE.sha256(path)
            self.assertEqual(first, second)
            self.assertEqual(path.read_bytes(), before)

    def test_formal_contract_is_120_substeps_without_direct_step_calls(self):
        self.assertEqual(ORACLE.CONTROL_STEPS, 30)
        self.assertEqual(ORACLE.EXPECTED_SUBSTEPS, 120)
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        forbidden = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {
                    "mj_step",
                    "mj_step1",
                    "mj_step2",
                    "mj_forward",
                }:
                    forbidden.append(node.func.attr)
        self.assertEqual(forbidden, [])


if __name__ == "__main__":
    unittest.main()
