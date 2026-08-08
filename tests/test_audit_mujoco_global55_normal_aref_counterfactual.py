import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import zipfile

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "audit_mujoco_global55_normal_aref_counterfactual.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_mujoco_global55_normal_aref_counterfactual", MODULE_PATH
)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUDIT)


def _model():
    opt = SimpleNamespace(
        iterations=100,
        tolerance=1.0e-8,
        cone=0,
        solver=2,
    )
    return SimpleNamespace(opt=opt, nv=6)


def _decomposition():
    coefficients = np.asarray(
        [[1.0, 0.5, 0.0], [1.0, -0.5, 0.0],
         [1.0, 0.0, 0.5], [1.0, 0.0, -0.5]],
        dtype=np.float64,
    )
    components = np.asarray([3.0, 4.0, 5.0])
    original = coefficients @ components
    return {
        "contact_index": 0,
        "robot_body_name": "limb/12",
        "row_ids": [7, 3, 9, 1],
        "basis_coefficients": coefficients,
        "a_normal": components[0],
        "a_t1": components[1],
        "a_t2": components[2],
        "original_efc_aref": original,
    }


def _excess(actual=3.0, rigid=2.0, vector_actual=None, vector_rigid=None):
    actual_vector = np.asarray(
        [actual, 0.0] if vector_actual is None else vector_actual,
        dtype=np.float64,
    )
    rigid_vector = np.asarray(
        [rigid, 0.0] if vector_rigid is None else vector_rigid,
        dtype=np.float64,
    )
    residual = actual_vector - rigid_vector
    return {
        "actual_tangent_impulse_vector": actual_vector,
        "actual_tangent_impulse_norm": float(np.linalg.norm(actual_vector)),
        "rigid_demand_vector": rigid_vector,
        "rigid_demand_norm": float(np.linalg.norm(rigid_vector)),
        "solver_excess_norm": float(np.linalg.norm(actual_vector) - np.linalg.norm(rigid_vector)),
        "solver_excess_vector": residual,
        "solver_excess_vector_norm": float(np.linalg.norm(residual)),
        "normal_impulse": 1.0,
        "post_slip": np.asarray([0.1, 0.0]),
    }


class NormalArefTests(unittest.TestCase):
    def test_normal_rebuild_zero_removes_only_normal_component(self):
        decomposition = _decomposition()
        rebuilt = AUDIT.rebuild_normal_aref(decomposition, 0.0)
        expected = decomposition["basis_coefficients"][:, 1] * decomposition["a_t1"]
        expected += decomposition["basis_coefficients"][:, 2] * decomposition["a_t2"]
        np.testing.assert_allclose(rebuilt, expected)
        np.testing.assert_allclose(
            AUDIT.rebuild_normal_aref(decomposition, 1.0),
            decomposition["original_efc_aref"],
        )

    def test_activation_reconstructs_scale_one_and_zero(self):
        decomposition = _decomposition()
        decompositions = {0: decomposition}
        conditions = {
            "normal_aref_scale_1_before": {0: AUDIT.rebuild_normal_aref(decomposition, 1.0)},
            "normal_aref_scale_0": {0: AUDIT.rebuild_normal_aref(decomposition, 0.0)},
            "normal_aref_scale_1_after_restore": {0: AUDIT.rebuild_normal_aref(decomposition, 1.0)},
        }
        result = AUDIT.normal_aref_activation(decompositions, conditions)
        self.assertEqual(result["NORMAL_AREF_COUNTERFACTUAL_ACTIVATION"], "VALIDATED")
        zero = result["contacts"]["0"]["conditions"]["normal_aref_scale_0"]
        self.assertTrue(zero["normal_component_zero"])
        self.assertTrue(zero["tangent_components_unchanged"])

    def test_decomposition_does_not_assume_row_order(self):
        decomposition = _decomposition()
        permutation = [2, 0, 3, 1]
        shuffled = dict(decomposition)
        shuffled["basis_coefficients"] = decomposition["basis_coefficients"][permutation]
        shuffled["original_efc_aref"] = decomposition["original_efc_aref"][permutation]
        np.testing.assert_allclose(
            AUDIT.rebuild_normal_aref(shuffled, 1.0), shuffled["original_efc_aref"]
        )

    def test_activation_requires_rank_three(self):
        decomposition = _decomposition()
        decomposition["basis_coefficients"] = np.ones((4, 3))
        conditions = {
            name: {0: np.zeros(4)}
            for name, _, _ in AUDIT.CONDITIONS
        }
        result = AUDIT.normal_aref_activation({0: decomposition}, conditions)
        self.assertEqual(result["NORMAL_AREF_COUNTERFACTUAL_ACTIVATION"], "FAILED")

    def test_normal_contact_regime_retained_and_target_collapse(self):
        def condition(normal):
            return {"capture": {"contacts": [{
                "geom1_name": "floor",
                "geom2_name": "limb/12",
                "normal_impulse": normal,
            }]}}

        retained = {
            name: condition(1.0)
            for name, _, _ in AUDIT.CONDITIONS
        }
        self.assertEqual(
            AUDIT._normal_contact_status(retained)["NORMAL_CONTACT_COUNTERFACTUAL_STATUS"],
            "CONTACT_REGIME_RETAINED",
        )
        collapsed = dict(retained)
        collapsed["normal_aref_scale_0"] = condition(0.0)
        self.assertEqual(
            AUDIT._normal_contact_status(collapsed)["NORMAL_CONTACT_COUNTERFACTUAL_STATUS"],
            "TARGET_NORMAL_FORCE_COLLAPSED",
        )

    def test_friction_cap_gate_thresholds(self):
        def condition(utilization):
            return {"capture": {"contacts": [{
                "geom1_name": "floor",
                "geom2_name": "limb/12",
                "normal_impulse": 1.0,
                "tangential_impulse_norm": utilization,
                "friction": np.asarray([1.0, 0.0, 0.0]),
            }]}}

        near = {name: condition(0.96) for name, _, _ in AUDIT.CONDITIONS}
        self.assertEqual(
            AUDIT._friction_cap_status(near)["NORMAL_AREF_FRICTION_CAP_STATUS"],
            "NEAR_CAP",
        )
        limited = dict(near)
        limited["normal_aref_scale_0"] = condition(1.0)
        self.assertEqual(
            AUDIT._friction_cap_status(limited)["NORMAL_AREF_FRICTION_CAP_STATUS"],
            "CAP_LIMITED",
        )

    def test_effect_classification_thresholds(self):
        baseline = _excess(actual=3.0, rigid=2.0)
        strong = _excess(actual=2.3, rigid=2.0)
        result = AUDIT.classify_effect(
            baseline, strong, True, "CONTACT_REGIME_RETAINED", "NOT_CAP_LIMITED"
        )
        self.assertEqual(result["NORMAL_AREF_SOLVER_EXCESS_EFFECT"], "STRONG_REDUCTION")
        self.assertEqual(result["MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER"], "NORMAL_REFERENCE_ACCELERATION_DOMINANT")

        no_change = AUDIT.classify_effect(
            baseline, baseline, True, "CONTACT_REGIME_RETAINED", "NOT_CAP_LIMITED"
        )
        self.assertEqual(no_change["NORMAL_AREF_SOLVER_EXCESS_EFFECT"], "LITTLE_OR_NO_REDUCTION")

    def test_effect_fails_closed_for_contact_collapse_and_cap(self):
        baseline = _excess()
        candidate = _excess(actual=1.0, rigid=0.5)
        collapse = AUDIT.classify_effect(
            baseline, candidate, True, "TARGET_NORMAL_FORCE_COLLAPSED", "NOT_CAP_LIMITED"
        )
        self.assertEqual(collapse["NORMAL_AREF_SOLVER_EXCESS_EFFECT"], "NONCANONICAL_CONTACT_REGIME")
        limited = AUDIT.classify_effect(
            baseline, candidate, True, "CONTACT_REGIME_RETAINED", "CAP_LIMITED"
        )
        self.assertEqual(limited["NORMAL_AREF_SOLVER_EXCESS_EFFECT"], "NONCANONICAL_CAP_LIMITED")

    def test_per_island_solver_numerics_are_valid(self):
        stat = SimpleNamespace(
            improvement=1.0,
            gradient=1.0e-14,
            lineslope=1.0,
            nactive=3,
            nchange=0,
            neval=2,
            nupdate=1,
        )
        data = SimpleNamespace(
            solver_niter=np.asarray([2] + [0] * 19),
            solver_nnz=np.asarray([361] + [0] * 19),
            nisland=1,
            solver=[stat, stat] + [stat for _ in range(18)],
            solver_fwdinv=np.zeros(2),
        )
        trace, numerics = AUDIT._solver_numerics(data, _model(), np.zeros(3))
        self.assertEqual(trace["active_solver_island_count"], 1)
        self.assertEqual(numerics["NORMAL_AREF_COUNTERFACTUAL_NUMERICS"], "VALID")
        self.assertEqual(numerics["active_solver_niter"], [2])

    def test_json_normalization_handles_numpy_and_nonfinite(self):
        value = AUDIT._json_normalize({
            "bool": np.bool_(True),
            "scalar": np.asarray(3),
            "nan": np.float64(np.nan),
        })
        self.assertEqual(value, {"bool": True, "scalar": 3, "nan": None})

    def test_source_does_not_change_forbidden_production_options(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("model.opt.cone =", source)
        self.assertNotIn("model.opt.solver =", source)
        self.assertNotIn("model.opt.tolerance =", source)
        self.assertNotIn("model.opt.iterations =", source)
        self.assertNotIn("data.efc_R[", source)
        self.assertNotIn("data.efc_D[", source)

    def test_source_uses_normal_component_assignment_and_no_zero_aref_shortcut(self):
        source = inspect.getsource(AUDIT._normal_aref_rows)
        self.assertIn("_rebuild_normal_aref", source)
        self.assertNotIn("= 0", source)

    def test_failure_placeholders_and_zip_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact"
            artifact.mkdir()
            AUDIT._write_failure_placeholders(artifact, RuntimeError("boom"))
            self.assertTrue((artifact / "failure_context.json").is_file())
            archive = root / "artifact.zip"
            result = AUDIT._package(artifact, archive)
            self.assertEqual(result["ZIP_VERIFY"], "PASS")
            with zipfile.ZipFile(archive) as zipped:
                self.assertIsNone(zipped.testzip())
            self.assertTrue(Path(result["SHA256_SIDECAR"]).is_file())

    def test_stdout_contract_and_artifact_names_are_present(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        for token in (
            "ZIP_VERIFY=",
            "ZIP_SHA256=",
            "UPLOAD_THIS_ZIP=",
            "normal_aref_decomposition.json",
            "normal_contact_counterfactual_status.json",
        ):
            self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
