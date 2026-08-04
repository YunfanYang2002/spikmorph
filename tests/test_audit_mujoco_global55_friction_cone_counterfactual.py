import ast
import importlib.util
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
    / "audit_mujoco_global55_friction_cone_counterfactual.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_mujoco_global55_friction_cone_counterfactual", MODULE_PATH
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class FakeData:
    def __init__(self):
        self.time = 0.27
        for index, field in enumerate(AUDIT.STATE_COPY_FIELDS):
            setattr(self, field, np.asarray([index + 0.25, index + 0.75]))


def fake_model(cone=0):
    return SimpleNamespace(
        opt=SimpleNamespace(
            cone=cone,
            integrator=0,
            solver=2,
            iterations=100,
            ls_iterations=50,
            tolerance=1e-8,
            timestep=0.005,
            disableflags=0,
        ),
        geom_friction=np.asarray([[0.7, 0.1, 0.1]]),
        pair_friction=np.asarray([[0.7, 0.7, 0.1, 0.1, 0.1]]),
        geom_solref=np.asarray([[0.02, 1.0]]),
        geom_solimp=np.asarray([[0.9, 0.95, 0.001, 0.5, 2.0]]),
        pair_solref=np.asarray([[0.02, 1.0]]),
        pair_solimp=np.asarray([[0.9, 0.95, 0.001, 0.5, 2.0]]),
        jnt_solref=np.asarray([[0.02, 1.0]]),
        jnt_solimp=np.asarray([[0.0, 0.99, 0.01, 0.5, 2.0]]),
        dof_damping=np.ones(19),
    )


def parameterization(row_count, cone):
    return {
        "cone_numeric": cone,
        "capture": {"ncon": 3, "nefc": 12 if row_count == 4 else 9},
        "contact_parameterization": [
            {
                "robot_body_name": body,
                "row_count": row_count,
                "efc_rows": list(range(row_count)),
                "efc_types": [6] * row_count,
                "dim": 3,
            }
            for body in ("limb/11", "limb/12")
        ],
    }


def effect(excess, vector_norm=None):
    return {
        "solver_excess_norm": excess,
        "solver_excess_vector_norm": abs(excess) if vector_norm is None else vector_norm,
    }


def baseline_fixture():
    actual = [0.2817184097958504, -3.312779286570642]
    rigid = [0.0, -2.540619084288334]
    contact = {
        "geom1_name": "limb/12",
        "geom2_name": "floor/0",
        "point_world": [1.0, 2.0, 3.0],
        "physical_basis_world_rows": np.eye(3).tolist(),
        "pre_tangential_velocity": [-0.06709699822797885, -0.5520978817696981],
        "post_tangential_velocity": [-0.03657355914172104, -0.16740203417847144],
        "normal_impulse": 6.345240278967453,
        "tangential_impulse": actual,
        "solver_rows": [
            {"efc_type": 6, "efc_id": 1, "efc_force": value}
            for value in (121.0, 201.0, 0.0, 946.0)
        ],
    }
    target = {
        "actual_tangential_impulse": actual,
        "actual_tangential_impulse_norm": 3.3247363600666735,
        "actual_normal_impulse": 6.345240278967453,
        "global_normal_conditioned_sticking_impulse": rigid,
        "global_normal_conditioned_sticking_impulse_norm": 2.540619084288334,
        "pre_tangential_speed": 0.5561601192334749,
        "friction_cap_mu_pn": 4.441668195277217,
    }
    capture = {
        "contacts": [contact],
        "mass_matrix": np.eye(2),
        "J_phys": np.ones((3, 2)),
        "W_phys": np.eye(3),
    }
    condition = {
        "capture": capture,
        "budget": {"selected": {"limb/12": target}},
        "excess": {
            "solver_excess_norm": 0.7841172757783395,
            "actual_tangent_impulse": actual,
            "rigid_demand_impulse": rigid,
            "post_slip": contact["post_tangential_velocity"],
        },
    }
    reference = {
        "contacts": {"all_active_robot_floor_contacts": [json.loads(json.dumps(contact))]},
        "budget": {"selected": {"limb/12": json.loads(json.dumps(target))}},
        "mass": {"mass_matrix": np.eye(2).tolist()},
        "jacobian": {"J_phys": np.ones((3, 2)).tolist()},
        "delassus": {"W_phys": np.eye(3).tolist()},
    }
    return condition, reference


class MujocoFrictionConeCounterfactualTests(unittest.TestCase):
    def test_state_copy_includes_qacc_warmstart_and_all_required_inputs(self):
        data = FakeData()
        manifest = AUDIT.state_copy_manifest(data)
        self.assertTrue(manifest["full_mjData_copy"])
        self.assertTrue(manifest["fields"]["qacc_warmstart"]["available"])
        self.assertEqual(set(AUDIT.STATE_COPY_FIELDS) | {"time"}, set(manifest["fields"]))

    def test_three_clones_compare_equal_to_same_pre_state(self):
        reference = FakeData()
        for _ in AUDIT.CONDITIONS:
            candidate = FakeData()
            self.assertTrue(AUDIT.state_equality(reference, candidate)["STATE_COPY_EQUAL"])
        candidate = FakeData()
        candidate.qacc_warmstart[0] += 1.0
        report = AUDIT.state_equality(reference, candidate)
        self.assertFalse(report["STATE_COPY_EQUAL"])
        self.assertFalse(report["fields"]["qacc_warmstart"]["equal"])

    def test_snapshot_equality_detects_no_mutation(self):
        left = AUDIT.state_input_snapshot(FakeData())
        right = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in left.items()}
        self.assertTrue(AUDIT.state_snapshots_equal(left, right))
        right["qpos"][0] += 1.0
        self.assertFalse(AUDIT.state_snapshots_equal(left, right))

    def test_only_cone_changes_and_restore_is_exact(self):
        model = fake_model(0)
        original = AUDIT.model_option_snapshot(model)
        model.opt.cone = 1
        elliptic = AUDIT.model_option_snapshot(model)
        diff = AUDIT.model_option_difference(original, elliptic)
        self.assertEqual(diff["changed_fields"], ["opt.cone"])
        self.assertEqual(diff["only_changed_field"], "opt.cone")
        model.opt.cone = 0
        restored = AUDIT.model_option_snapshot(model)
        self.assertEqual(AUDIT.model_option_difference(original, restored)["changed_fields"], [])

    def test_constraint_parameterization_activation(self):
        conditions = {
            "pyramidal_before": parameterization(4, 0),
            "elliptic": parameterization(3, 1),
            "pyramidal_after_restore": parameterization(4, 0),
        }
        report = AUDIT.constraint_activation(conditions, 0, 1)
        self.assertEqual(report["CONE_COUNTERFACTUAL_ACTIVATION"], "VALIDATED")
        conditions["elliptic"]["contact_parameterization"][0]["row_count"] = 4
        report = AUDIT.constraint_activation(conditions, 0, 1)
        self.assertEqual(report["CONE_COUNTERFACTUAL_ACTIVATION"], "NOT_ACTIVATED")

    def test_solver_excess_records_vector_norm_angle_and_cap(self):
        contact = {
            "pre_tangential_velocity": [1.0, 0.0],
            "post_tangential_velocity": [0.1, 0.0],
        }
        budget = {
            "actual_tangential_impulse": [3.0, 4.0],
            "global_normal_conditioned_sticking_impulse": [0.0, 4.0],
            "actual_normal_impulse": 6.0,
            "friction_cap_mu_pn": 7.0,
        }
        report = AUDIT.compute_solver_excess(contact, budget)
        self.assertAlmostEqual(report["solver_excess_norm"], 1.0)
        np.testing.assert_allclose(report["solver_excess_vector"], [3.0, 0.0])
        self.assertAlmostEqual(report["solver_excess_vector_norm"], 3.0)
        self.assertAlmostEqual(report["friction_cap_utilisation"], 5.0 / 7.0)
        self.assertIsNotNone(report["actual_rigid_angle_degrees"])

    def test_effect_thresholds_and_next_actions(self):
        cases = (
            (0.2, "STRONG_REDUCTION", "PYRAMIDAL_CONE_PARAMETERIZATION_DOMINANT", "NO_ADDITIONAL_SOLVER_COUNTERFACTUAL_REQUIRED"),
            (0.6, "PARTIAL_REDUCTION", "PYRAMIDAL_CONE_PARAMETERIZATION_CONTRIBUTING", "FRICTION_AREF_REGULARIZATION_COUNTERFACTUAL"),
            (0.9, "LITTLE_OR_NO_REDUCTION", "PYRAMIDAL_CONE_PARAMETERIZATION_NOT_DOMINANT", "FRICTION_AREF_REGULARIZATION_COUNTERFACTUAL"),
            (1.2, "INCREASED", "PYRAMIDAL_CONE_PARAMETERIZATION_NOT_DOMINANT", "FRICTION_AREF_REGULARIZATION_COUNTERFACTUAL"),
        )
        for elliptic, expected, driver, next_action in cases:
            with self.subTest(elliptic=elliptic):
                report = AUDIT.classify_effect(effect(1.0), effect(elliptic), True)
                self.assertEqual(report["FRICTION_CONE_SOLVER_EXCESS_EFFECT"], expected)
                self.assertEqual(report["MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER"], driver)
                self.assertEqual(report["NEXT_ACTION"], next_action)

    def test_reduction_formula_includes_norm_and_vector_residual(self):
        report = AUDIT.classify_effect(effect(2.0, 4.0), effect(1.0, 1.0), True)
        self.assertAlmostEqual(report["absolute_excess_reduction"], 1.0)
        self.assertAlmostEqual(report["relative_excess_reduction"], 0.5)
        self.assertAlmostEqual(report["relative_vector_residual_reduction"], 0.75)

    def test_failed_gate_and_noncanonical_fail_closed(self):
        failed = AUDIT.classify_effect(effect(1.0), effect(0.0), False)
        self.assertEqual(failed["FRICTION_CONE_SOLVER_EXCESS_EFFECT"], "INSUFFICIENT_EVIDENCE")
        noncanonical = AUDIT.classify_effect(effect(1.0), effect(0.0), False, noncanonical=True)
        self.assertEqual(noncanonical["FRICTION_CONE_SOLVER_EXCESS_EFFECT"], "NONCANONICAL")

    def test_baseline_regression_passes_and_fails_closed(self):
        condition, reference = baseline_fixture()
        report = AUDIT.baseline_regression(condition, reference)
        self.assertEqual(report["PYRAMIDAL_BASELINE_REPRODUCTION"], "PASS")
        condition["capture"]["contacts"][0]["point_world"][0] += 0.1
        report = AUDIT.baseline_regression(condition, reference)
        self.assertEqual(report["PYRAMIDAL_BASELINE_REPRODUCTION"], "FAIL")

    def test_restore_regression_checks_mass_j_w_impulses_and_rows(self):
        before, _ = baseline_fixture()
        after, _ = baseline_fixture()
        report = AUDIT.restore_regression(before, after)
        self.assertEqual(report["PYRAMIDAL_RESTORE_REPRODUCTION"], "PASS")
        after["capture"]["W_phys"][0, 0] += 0.1
        report = AUDIT.restore_regression(before, after)
        self.assertEqual(report["PYRAMIDAL_RESTORE_REPRODUCTION"], "FAIL")

    def test_shared_demand_output_names_the_reused_physical_method(self):
        condition = {
            "condition_name": "elliptic",
            "cone_numeric": 1,
            "state_validation": {},
            "capture": {
                "ncon": 0, "nefc": 0, "contacts": [],
                "mass_matrix": np.eye(1), "J_phys": np.zeros((0, 1)),
                "W_phys": np.zeros((0, 0)),
                "post_state": {"qpos": [0.0], "qvel": [0.0]},
            },
            "physical_impulses": {"api": "mujoco.mj_contactForce"},
            "budget": {"selected": {}},
            "excess": {"post_slip": [0.0, 0.0]},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            AUDIT.write_condition(output, condition)
            report = json.loads(
                (output / "conditions/elliptic/shared_physical_global_demand.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["method"], "SHARED_PHYSICAL_GLOBAL_COUPLED_DEMAND")

    def test_physical_readback_is_parameterization_independent(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn('"api": "mujoco.mj_contactForce"', source)
        self.assertIn('"parameterization_independent_readback": True', source)

    def test_formal_contract_has_one_replay_and_three_probe_pairs(self):
        self.assertEqual(AUDIT.EXPECTED_SUBSTEPS, 120)
        self.assertEqual(len(AUDIT.CONDITIONS), 3)
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        run_condition = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_condition"
        )
        calls = [
            node.func.attr for node in ast.walk(run_condition)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"mj_forward", "mj_step"}
        ]
        self.assertEqual(sorted(calls), ["mj_forward", "mj_step"])
        self.assertIn("oracle.replay(args, replay_paths)", MODULE_PATH.read_text(encoding="utf-8"))

    def test_source_hash_readback_is_non_mutating(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.xml"
            source.write_bytes(b"<mujoco/>")
            before = source.read_bytes()
            self.assertEqual(AUDIT.oracle.sha256(source), AUDIT.oracle.sha256(source))
            self.assertEqual(source.read_bytes(), before)

    def test_success_and_failure_artifacts_package_and_verify(self):
        for status in ("success", "failure"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / f"artifact_{status}"
                output.mkdir()
                (output / "summary.json").write_text(
                    json.dumps({"status": status}), encoding="utf-8"
                )
                archive = root / f"artifact_{status}.zip"
                report = AUDIT.oracle.package_artifact(output, archive)
                self.assertEqual(report["ZIP_VERIFY"], "PASS")
                self.assertIn("UPLOAD_THIS_ZIP", report)
                self.assertTrue(Path(report["SHA256_SIDECAR"]).is_file())
                with zipfile.ZipFile(archive) as bundle:
                    self.assertIsNone(bundle.testzip())
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn('print(f"ZIP_VERIFY=', source)
        self.assertIn('print(f"ZIP_SHA256=', source)
        self.assertIn('print(f"UPLOAD_THIS_ZIP=', source)

    def test_failure_payload_is_fail_closed(self):
        report = AUDIT.failure_payload(RuntimeError("boom"))
        self.assertFalse(report["COUNTERFACTUAL_VALID"])
        self.assertEqual(report["FRICTION_CONE_SOLVER_EXCESS_EFFECT"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(report["NEXT_ACTION"], "COUNTERFACTUAL_IMPLEMENTATION_FIX_REQUIRED")


if __name__ == "__main__":
    unittest.main()
