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
    / "audit_mujoco_global55_solver_optimization.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_mujoco_global55_solver_optimization", MODULE_PATH
)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUDIT)


def _model_options():
    return SimpleNamespace(
        cone=0,
        integrator=0,
        solver=2,
        iterations=2,
        ls_iterations=20,
        tolerance=1.0e-8,
        timestep=0.005,
        disableflags=0,
        jacobian=0,
        impratio=1.0,
        ls_tolerance=1.0e-8,
        noslip_tolerance=1.0e-8,
        ccd_tolerance=1.0e-8,
        noslip_iterations=0,
        ccd_iterations=50,
        sdf_initpoints=1,
        enableflags=0,
        o_solref=np.zeros((2,)),
        o_solimp=np.zeros((5,)),
    )


def _model():
    return SimpleNamespace(
        opt=_model_options(),
        geom_friction=np.zeros((1, 3)),
        pair_friction=None,
        geom_solref=np.zeros((1, 2)),
        geom_solimp=np.zeros((1, 5)),
        pair_solref=None,
        pair_solimp=None,
        jnt_solref=np.zeros((1, 2)),
        jnt_solimp=np.zeros((1, 5)),
        dof_damping=np.zeros(1),
    )


def _condition():
    contact = {
        "normal_impulse": 1.0,
        "tangential_impulse": np.asarray([0.2, 0.3]),
        "tangential_impulse_norm": np.sqrt(0.13),
        "pre_tangential_speed": 0.1,
        "contact_index": 0,
        "geom1_name": "floor",
        "geom2_name": "limb/12",
        "post_tangential_velocity": np.asarray([0.01, 0.02]),
        "solver_rows": [],
    }
    return {
        "capture": {"contacts": [contact]},
        "excess": {
            "actual_tangent_impulse_vector": np.asarray([0.2, 0.3]),
            "actual_tangent_impulse_norm": np.sqrt(0.13),
            "normal_impulse": 1.0,
            "rigid_demand_vector": np.asarray([0.1, 0.2]),
            "rigid_demand_norm": np.sqrt(0.05),
            "solver_excess_vector": np.asarray([0.1, 0.1]),
            "solver_excess_norm": np.sqrt(0.02),
            "post_slip": np.asarray([0.01, 0.02]),
        },
        "post_constraint": {"qacc": np.zeros(2)},
        "solver_iteration_trace": {"trace": []},
        "solver_numerics": {"solver_niter": 2},
    }


class SolverOptimizationTests(unittest.TestCase):
    def test_four_conditions_are_fixed(self):
        self.assertEqual(
            [item[0] for item in AUDIT.CONDITIONS],
            [
                "production_warmstart_production_tolerance",
                "zero_warmstart_production_tolerance",
                "production_warmstart_tight_tolerance",
                "zero_warmstart_tight_tolerance",
            ],
        )
        self.assertEqual([item[2:] for item in AUDIT.CONDITIONS], [
            (False, False), (True, False), (False, True), (True, True)
        ])

    def test_tight_tolerance_is_strict_and_iterations_are_raised(self):
        value, rationale = AUDIT._tight_tolerance(1.0e-8)
        self.assertLess(value, 1.0e-8)
        self.assertIn("1e-12", rationale)
        value2, _ = AUDIT._tight_tolerance(1.0e-12)
        self.assertLess(value2, 1.0e-12)

        model = _model()
        production = AUDIT._model_option_snapshot(model)
        configured = AUDIT._configure_model(model, production, True)
        self.assertLess(model.opt.tolerance, production["opt.tolerance"])
        self.assertGreaterEqual(model.opt.iterations, 100)
        self.assertTrue(configured["tight"])

    def test_production_configuration_keeps_options_unchanged(self):
        model = _model()
        production = AUDIT._model_option_snapshot(model)
        AUDIT._configure_model(model, production, False)
        self.assertTrue(AUDIT._option_difference(production, AUDIT._model_option_snapshot(model))["only_allowed"])

    def test_mj_copy_model_is_preferred_and_smoked(self):
        source = _model()

        class FakeMujoco:
            @staticmethod
            def mj_copyModel(destination, model):
                self.assertIsNone(destination)
                return _model()

        clone, method, smoke = AUDIT._copy_model(
            FakeMujoco, source, "MJ_COPY_MODEL"
        )
        self.assertIsNot(clone, source)
        self.assertEqual(method, "MJ_COPY_MODEL")
        self.assertEqual(smoke["CLONE_SMOKE"], "PASS")

    def test_mjb_roundtrip_preserves_compiled_runtime_model(self):
        class FakeModel:
            @staticmethod
            def from_binary_path(path):
                self.assertTrue(Path(path).is_file())
                return _model()

        def save_model(model, path, _vfs):
            Path(path).write_bytes(b"compiled-mjb")

        with tempfile.TemporaryDirectory() as directory:
            clone, method, smoke = AUDIT._copy_model(
                SimpleNamespace(mj_saveModel=save_model, MjModel=FakeModel),
                _model(),
                "MJB_ROUNDTRIP",
                Path(directory) / "live_model.mjb",
            )
        self.assertIsNotNone(clone)
        self.assertEqual(method, "MJB_ROUNDTRIP")
        self.assertEqual(smoke["CLONE_SMOKE"], "PASS")

    def test_transactional_fallback_restores_solver_options(self):
        model = _model()
        before = AUDIT._model_option_snapshot(model)
        report = AUDIT._transactional_smoke(SimpleNamespace(), model)
        self.assertEqual(report["TRANSACTIONAL_SMOKE"], "PASS")
        self.assertTrue(AUDIT._option_difference(before, AUDIT._model_option_snapshot(model))["only_allowed"])

    def test_formal_copy_function_has_no_xml_clone_fallback(self):
        source = inspect.getsource(AUDIT._copy_model)
        self.assertNotIn("from_xml_path", source)
        self.assertNotIn("from_xml_string", source)

    def test_runtime_source_inventory_reports_live_augmentation(self):
        def geom_model(count):
            model = _model()
            model.ngeom = count
            model.nbody = 1
            model.geom_type = np.zeros(count, dtype=np.int32)
            model.geom_bodyid = np.zeros(count, dtype=np.int32)
            model.geom_contype = np.ones(count, dtype=np.int32)
            model.geom_conaffinity = np.ones(count, dtype=np.int32)
            model.geom_friction = np.ones((count, 3))
            return model

        live, source = geom_model(2), geom_model(1)

        class FakeModel:
            @staticmethod
            def from_xml_path(_path):
                return source

        class Obj:
            mjOBJ_GEOM = 0
            mjOBJ_BODY = 1

        def name(_model, object_type, index):
            if object_type == Obj.mjOBJ_GEOM:
                return ["floor", "runtime_augmented"][index]
            return "world"

        report = AUDIT._runtime_source_geom_inventory(
            SimpleNamespace(MjModel=FakeModel, mjtObj=Obj, mj_id2name=name),
            live,
            Path("source.xml"),
        )
        self.assertEqual(report["RUNTIME_MODEL_STRUCTURE"], "HAS_RUNTIME_AUGMENTATION")
        self.assertEqual(report["live_only_geom_names"], ["runtime_augmented"])

    def test_capability_discovery_records_symbols_without_replay(self):
        def copy_model(_destination, _source):
            return None

        class FakeModel:
            from_binary_path = staticmethod(lambda _path: None)

        report = AUDIT._capability_discovery(
            SimpleNamespace(
                __version__="3.8.1",
                mj_copyModel=copy_model,
                mj_saveModel=lambda _model, _path, _vfs: None,
                MjModel=FakeModel,
            )
        )
        self.assertTrue(report["symbols"]["mj_copyModel"]["available"])
        self.assertTrue(report["symbols"]["mj_saveModel"]["available"])
        self.assertTrue(report["symbols"]["MjModel.from_binary_path"]["available"])

    def test_solver_iteration_trace_reads_all_required_statistics(self):
        stats = [SimpleNamespace(**{
            "improvement": 1.0,
            "gradient": 2.0,
            "lineslope": 3.0,
            "nactive": 4,
            "nchange": 5,
            "neval": 6,
            "nupdate": 7,
        })]
        data = SimpleNamespace(solver_niter=1, solver=stats)
        result = AUDIT._solver_iteration_trace(data, _model())
        self.assertEqual(result["solver_niter"], 1)
        self.assertTrue(result["statistics_available"])
        self.assertTrue(result["all_statistics_finite"])
        self.assertEqual(result["trace"][0]["iteration_index"], 0)

    def test_solver_iteration_trace_fails_closed_for_missing_statistics(self):
        data = SimpleNamespace(solver_niter=1, solver=[])
        result = AUDIT._solver_iteration_trace(data, _model())
        self.assertFalse(result["statistics_available"])
        self.assertEqual(result["status"], "INSUFFICIENT_EVIDENCE")

    def test_zero_warmstart_state_is_exact_zero(self):
        snapshot = {
            "time": 1.0,
            "qpos": np.ones(2),
            "qacc_warmstart": np.zeros(3),
        }
        candidate = dict(snapshot)
        report = AUDIT._state_equal_except_warmstart(snapshot, candidate)
        self.assertTrue(report["valid"])

    def test_pair_sensitivity_does_not_use_iteration_count_alone(self):
        left = _condition()
        right = _condition()
        right["solver_numerics"]["solver_niter"] = 100
        report = AUDIT._pair_sensitivity(left, right)
        self.assertEqual(report["classification"], "INSENSITIVE")
        right["excess"]["normal_impulse"] = 1.1
        report = AUDIT._pair_sensitivity(left, right)
        self.assertEqual(report["classification"], "SENSITIVE")

    def test_final_classification_robust_and_sensitive_paths(self):
        invariant = {"SOLVER_OPTIMIZATION_DIAGNOSTIC_ISOLATION": "VALIDATED"}
        warm = {"SOLVER_WARMSTART_SENSITIVITY": "INSENSITIVE"}
        tolerance = {"SOLVER_TOLERANCE_SENSITIVITY": "INSENSITIVE"}
        convergence = {"PRODUCTION_NEWTON_CONVERGENCE": "VALIDATED"}
        baseline = {"OPTIMIZATION_BASELINE_REPRODUCTION": "PASS"}
        result = AUDIT._classify_final(invariant, warm, tolerance, convergence, baseline)
        self.assertEqual(result["MUJOCO_SOLVER_EXCESS_OPTIMIZATION_STATUS"], "ROBUST_CONVERGED_SOLUTION")
        self.assertEqual(result["NEXT_ACTION"], "NORMAL_REFERENCE_ACCELERATION_COUNTERFACTUAL")

        warm["SOLVER_WARMSTART_SENSITIVITY"] = "SENSITIVE"
        result = AUDIT._classify_final(invariant, warm, tolerance, convergence, baseline)
        self.assertEqual(result["MUJOCO_SOLVER_EXCESS_OPTIMIZATION_STATUS"], "WARMSTART_SENSITIVE")

    def test_json_normalization_handles_numpy_scalars_and_nonfinite(self):
        value = AUDIT._json_normalize({
            "bool": np.bool_(True),
            "scalar": np.asarray(3),
            "nan": np.float64(np.nan),
        })
        self.assertEqual(value, {"bool": True, "scalar": 3, "nan": None})

    def test_failure_payload_is_fail_closed(self):
        payload = AUDIT._failure_payload(RuntimeError("boom"))
        self.assertEqual(payload["LOCAL_IMPLEMENTATION"], "INCOMPLETE")
        self.assertEqual(payload["UNCONDITIONAL_ZIP_PACKAGING"], "ENABLED")

    def test_failure_placeholders_include_clone_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact"
            artifact.mkdir()
            AUDIT._write_failure_placeholders(artifact, RuntimeError("boom"))
            for filename in (
                "model_clone_api_discovery.json",
                "runtime_vs_source_geom_inventory.json",
                "model_clone_fidelity.json",
                "clone_data_state_fidelity.json",
                "failure_context.json",
            ):
                self.assertTrue((artifact / filename).is_file())

    def test_zip_helper_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact"
            artifact.mkdir()
            (artifact / "summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
            archive = root / "artifact.zip"
            result = AUDIT.oracle.package_artifact(artifact, archive)
            self.assertEqual(result["ZIP_VERIFY"], "PASS")
            with zipfile.ZipFile(archive) as zipped:
                self.assertIsNone(zipped.testzip())
            self.assertTrue(Path(result["SHA256_SIDECAR"]).is_file())

    def test_source_does_not_assign_forbidden_physics_fields(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("model.opt.cone =", source)
        self.assertNotIn("model.opt.solver =", source)
        self.assertNotIn("data.efc_aref[", source)
        self.assertNotIn("data.efc_R[", source)
        self.assertNotIn("data.efc_D[", source)

    def test_stdout_contract_is_present(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn("ZIP_VERIFY=", source)
        self.assertIn("ZIP_SHA256=", source)
        self.assertIn("UPLOAD_THIS_ZIP=", source)


if __name__ == "__main__":
    unittest.main()
