import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "audit_mujoco_global55_friction_aref_counterfactual.py"
SPEC = importlib.util.spec_from_file_location("audit_mujoco_global55_friction_aref_counterfactual", MODULE_PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class ArefCounterfactualTests(unittest.TestCase):
    def test_staged_pipeline_order_and_no_condition_full_forward(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        stage = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "stage_to_constraint")
        source_calls = [node.value for node in ast.walk(stage) if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("mj_fwd")]
        self.assertEqual(source_calls, ["mj_fwdPosition", "mj_fwdVelocity", "mj_fwdActuation", "mj_fwdAcceleration"])
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_condition")
        self.assertNotIn("mj_forward", {n.func.attr for n in ast.walk(run) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)})
        self.assertNotIn("mj_step", {n.func.attr for n in ast.walk(run) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)})

    def test_four_row_geometry_rank_and_aref_fit(self):
        basis = np.eye(3)
        rows = np.asarray([[1, .7, 0], [1, -.7, 0], [1, 0, .2], [1, 0, -.2]], dtype=float)
        components = np.asarray([2.0, 3.0, -4.0])
        aref = rows @ components
        report = AUDIT.decompose_pyramidal_contact_aref([7, 2, 9, 4], rows, aref, basis, [0.8, .7, .2])
        self.assertEqual(report["rank"], 3)
        np.testing.assert_allclose(report["fitted_components"], components)
        np.testing.assert_allclose(report["reconstructed_efc_aref"], aref)

    def test_nonuniform_friction_and_row_order_are_not_assumed(self):
        basis = np.eye(3)
        rows = np.asarray([[1, 0, -.2], [1, .8, 0], [1, 0, .2], [1, -.8, 0]], dtype=float)
        components = np.asarray([1.2, -.4, .9])
        report = AUDIT.decompose_pyramidal_contact_aref([31, 11, 42, 8], rows, rows @ components, basis, [0.8, .8, .2])
        self.assertEqual(report["row_ids"], [31, 11, 42, 8])
        np.testing.assert_allclose(report["basis_coefficients"], rows)

    def test_scale_zero_keeps_normal_and_removes_tangents(self):
        decomposition = AUDIT.decompose_pyramidal_contact_aref([0, 1, 2, 3], np.asarray([[1, .5, 0], [1, -.5, 0], [1, 0, .25], [1, 0, -.25]]), np.asarray([3.5, .5, 3.0, 1.0]), np.eye(3), [0.5, .25, 0.0])
        zero = AUDIT.rebuild_aref(decomposition, 0)
        np.testing.assert_allclose(zero, decomposition["basis_coefficients"][:, 0] * decomposition["a_normal"])
        self.assertAlmostEqual(decomposition["a_t1"], 3.0)
        self.assertAlmostEqual(decomposition["a_t2"], 4.0)

    def test_activation_requires_before_zero_after_restore(self):
        decomposition = AUDIT.decompose_pyramidal_contact_aref([0, 1, 2, 3], np.asarray([[1, .5, 0], [1, -.5, 0], [1, 0, .25], [1, 0, -.25]]), np.asarray([3.5, .5, 3.0, 1.0]), np.eye(3), [0.5, .25, 0.0])
        rows = {name: {0: AUDIT.rebuild_aref(decomposition, scale)} for name, _, scale in AUDIT.CONDITIONS}
        report = AUDIT.aref_activation({0: decomposition}, rows)
        self.assertEqual(report["FRICTION_AREF_COUNTERFACTUAL_ACTIVATION"], "VALIDATED")

    def test_effect_classification_and_next_action(self):
        base = {"solver_excess_norm": 1.0, "solver_excess_vector_norm": 2.0}
        zero = {"solver_excess_norm": .2, "solver_excess_vector_norm": .5}
        report = AUDIT.classify_effect(base, zero, True)
        self.assertEqual(report["FRICTION_AREF_SOLVER_EXCESS_EFFECT"], "STRONG_REDUCTION")
        self.assertEqual(report["MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER"], "FRICTION_REFERENCE_ACCELERATION_DOMINANT")
        self.assertEqual(report["NEXT_ACTION"], "NO_ADDITIONAL_SOLVER_COUNTERFACTUAL_REQUIRED")

    def test_solver_excess_schema_has_canonical_and_compatibility_names(self):
        capture = {
            "contacts": [{
                "tangential_impulse": [3.0, 4.0],
                "normal_impulse": 6.0,
                "friction": [0.5, 0.5, 0.5],
                "pre_tangential_velocity": [1.0, 0.0],
                "post_tangential_velocity": [0.5, 0.0],
            }],
        }
        demand = {"limb_12_contact_index": 0, "limb_12_tangent_impulse_2d": [0.0, 4.0]}
        excess = AUDIT.compute_solver_excess(capture, demand)
        self.assertTrue(AUDIT.validate_excess_schema(excess)["valid"])
        np.testing.assert_allclose(excess["actual_tangent_impulse"], excess["actual_tangent_impulse_vector"])
        np.testing.assert_allclose(excess["rigid_demand_impulse"], excess["rigid_demand_vector"])

    def test_restore_regression_uses_canonical_schema_without_keyerror(self):
        row = {"efc_row": 0, "efc_type": 6, "efc_id": 1, "efc_aref": 2.0, "efc_R": 0.1, "efc_D": 0.2, "efc_diagApprox": 0.3, "efc_vel": 0.4, "efc_force": 1.0}
        contact = {"geom1_name": "limb/12", "geom2_name": "floor/0", "point_world": [0.0, 0.0, 0.0], "physical_basis_world_rows": np.eye(3), "normal_impulse": 6.0, "solver_rows": [row]}
        def condition():
            return {
                "capture": {"mass_matrix": np.eye(2), "J_phys": np.ones((3, 2)), "W_phys": np.eye(3), "contacts": [contact.copy()]},
                "excess": {"actual_tangent_impulse_vector": [3.0, 4.0], "actual_tangent_impulse_norm": 5.0, "rigid_demand_vector": [0.0, 4.0], "rigid_demand_norm": 4.0, "solver_excess_norm": 1.0, "solver_excess_vector": [3.0, 0.0], "solver_excess_vector_norm": 3.0, "normal_impulse": 6.0, "friction_cap": 3.0, "friction_cap_utilisation": 5.0 / 3.0, "pre_slip": [1.0, 0.0], "post_slip": [0.1, 0.0]},
            }
        report = AUDIT.restore_regression(condition(), condition())
        self.assertEqual(report["AREF_RESTORE_REPRODUCTION"], "PASS")
        changed = condition()
        changed["excess"]["actual_tangent_impulse_vector"] = [3.1, 4.0]
        self.assertEqual(AUDIT.restore_regression(condition(), changed)["AREF_RESTORE_REPRODUCTION"], "FAIL")

    def test_incomplete_restore_schema_fails_closed_without_keyerror(self):
        report = AUDIT.restore_regression({"capture": {}, "excess": {}}, {"capture": {}, "excess": {}})
        self.assertEqual(report["AREF_RESTORE_REPRODUCTION"], "FAIL")
        self.assertFalse(report["checks"]["excess_schema"])

    def test_isolation_reports_only_allowed_aref_effects(self):
        self.assertIn("efc_R", AUDIT.invariant_validation.__code__.co_consts)
        self.assertIn("friction tangent component of efc_aref", MODULE_PATH.read_text(encoding="utf-8"))

    def test_shared_demand_formula_and_physical_readback_are_named(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn("mj_contactForce", source)
        self.assertIn("SHARED_PHYSICAL_GLOBAL_COUPLED_DEMAND", source)
        self.assertIn("rigid_demand_tangent_impulse_6d", source)

    def test_counts_and_formal_replay_contract(self):
        self.assertEqual(AUDIT.EXPECTED_SUBSTEPS, 120)
        self.assertEqual(len(AUDIT.CONDITIONS), 3)
        self.assertEqual([item[2] for item in AUDIT.CONDITIONS], [1.0, 0.0, 1.0])
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn("cone_helper.replay_once(args, paths)", source)
        self.assertIn("formal_replay_additional_steps", source)

    def test_failure_payload_is_fail_closed(self):
        report = AUDIT.failure_payload(RuntimeError("boom"))
        self.assertEqual(report["AREF_COUNTERFACTUAL_ISOLATION"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(report["NEXT_ACTION"], "COUNTERFACTUAL_IMPLEMENTATION_FIX_REQUIRED")
        self.assertEqual(report["UNCONDITIONAL_ZIP_PACKAGING"], "ENABLED")

    def test_success_failure_zip_testzip_and_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "artifact"
            output.mkdir()
            (output / "summary.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
            archive = root / "artifact.zip"
            report = AUDIT.oracle.package_artifact(output, archive)
            self.assertEqual(report["ZIP_VERIFY"], "PASS")
            self.assertTrue(Path(report["SHA256_SIDECAR"]).is_file())
            with zipfile.ZipFile(archive) as bundle:
                self.assertIsNone(bundle.testzip())

    def test_stdout_upload_contract_is_source_present(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn('print(f"ZIP_VERIFY=', source)
        self.assertIn('print(f"ZIP_SHA256=', source)
        self.assertIn('print(f"UPLOAD_THIS_ZIP=', source)

    def test_only_aref_is_written_in_condition_and_production_options_are_read_only(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn("data.efc_aref", source)
        self.assertNotIn("model.opt.solref =", source)
        self.assertNotIn("model.opt.solimp =", source)
        self.assertNotIn("model.opt.impratio =", source)
        self.assertNotIn("model.opt.solver =", source)

    def test_optional_empty_constraint_arrays_are_skipped(self):
        empty = type("Data", (), {"efc_b": np.zeros(0), "efc_AR": np.zeros(0)})()
        self.assertIsNone(AUDIT._optional_constraint_row_value(empty, "efc_b", 2))
        self.assertIsNone(AUDIT._optional_constraint_row_value(empty, "efc_AR", 2))

    def test_optional_constraint_row_arrays_are_read_when_indexable(self):
        data = type("Data", (), {"efc_b": np.asarray([1.0, 2.0, 3.0]), "efc_AR": np.asarray([[4.0], [5.0], [6.0]])})()
        self.assertEqual(AUDIT._optional_constraint_row_value(data, "efc_b", 2), 3.0)
        np.testing.assert_allclose(AUDIT._optional_constraint_row_value(data, "efc_AR", 1), [5.0])


if __name__ == "__main__":
    unittest.main()
