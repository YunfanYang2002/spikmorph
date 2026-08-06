import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "audit_mujoco_global55_contact_regularization_counterfactual.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_mujoco_global55_contact_regularization_counterfactual", MODULE_PATH
)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUDIT)


class RegularizationCounterfactualTests(unittest.TestCase):
    def test_numpy_bool_compatibility_alias_is_installed_before_replay_helpers(self):
        self.assertIs(np.bool, bool)

    def test_solver_storage_semantics_are_read_from_live_model_options(self):
        solver_type = SimpleNamespace(mjSOL_PGS=0, mjSOL_CG=1, mjSOL_NEWTON=2)
        jacobian_type = SimpleNamespace(mjJAC_DENSE=0, mjJAC_SPARSE=1)
        cone_type = SimpleNamespace(mjCONE_PYRAMIDAL=0)
        mujoco = SimpleNamespace(
            mjtSolver=solver_type,
            mjtJacobian=jacobian_type,
            mjtCone=cone_type,
            mj_isDual=lambda model: False,
            mj_isSparse=lambda model: True,
        )
        model = SimpleNamespace(
            opt=SimpleNamespace(solver=1, jacobian=1, cone=0)
        )
        semantics = AUDIT.solver_storage_semantics(mujoco, model)
        self.assertEqual(semantics["SOLVER_FORMULATION"], "CG_PRIMAL")
        self.assertFalse(semantics["efc_AR_required_for_solver"])
        self.assertEqual(semantics["model_opt_solver_name"], "mjSOL_CG")
        self.assertTrue(semantics["mj_isSparse"])

    def test_primal_populated_gate_accepts_unallocated_efc_AR(self):
        contact = SimpleNamespace(geom1=0, geom2=1, efc_address=0, dim=4)
        data = SimpleNamespace(
            ncon=1,
            nefc=4,
            nisland=0,
            contact=[contact],
            efc_R=np.ones(4),
            efc_D=np.ones(4),
            efc_AR=np.zeros(0),
            efc_AR_rownnz=np.ones(4, dtype=int),
            efc_AR_rowadr=np.arange(4, dtype=int),
            efc_AR_colind=np.arange(4, dtype=int),
            efc_force=np.zeros(4),
            qfrc_constraint=np.zeros(2),
            qacc=np.zeros(2),
            efc_type=np.full(4, 6, dtype=int),
            efc_id=np.zeros(4, dtype=int),
        )
        model = SimpleNamespace(nv=2)
        mujoco = SimpleNamespace(
            mjtObj=SimpleNamespace(mjOBJ_GEOM=0),
            mj_id2name=lambda *args: "geom",
        )
        selected = [{"row_id": 0}]
        semantics = {"SOLVER_FORMULATION": "NEWTON_PRIMAL"}
        populated = AUDIT._populated_constraint_state(
            mujoco, model, data, ["mj_fwdAcceleration"], selected, semantics
        )
        self.assertEqual(populated["POPULATED_CONSTRAINT_STATE"], "PASS")
        self.assertEqual(
            populated["efc_AR_status"],
            "EXPECTED_UNALLOCATED_FOR_PRIMAL_SOLVER",
        )
        self.assertTrue(populated["checks"]["efc_AR_gate"])

    def test_primal_constraint_path_does_not_require_projected_AR(self):
        calls = []
        mujoco = SimpleNamespace(
            mj_projectConstraint=lambda *args: calls.append("project"),
            mj_fwdConstraint=lambda *args: calls.append("fwd"),
        )
        self.assertEqual(
            AUDIT._solver_constraint_calls(
                mujoco, object(), object(), {"SOLVER_FORMULATION": "NEWTON_PRIMAL"}
            ),
            ["mj_fwdConstraint"],
        )
        self.assertEqual(calls, ["fwd"])

    def test_unknown_solver_formulation_fails_populated_gate(self):
        contact = SimpleNamespace(geom1=0, geom2=1, efc_address=0, dim=1)
        data = SimpleNamespace(
            ncon=1, nefc=1, nisland=0, contact=[contact],
            efc_R=np.ones(1), efc_D=np.ones(1), efc_AR=np.zeros(0),
            efc_AR_rownnz=np.zeros(1, dtype=int), efc_AR_rowadr=np.zeros(1, dtype=int),
            efc_AR_colind=np.zeros(0, dtype=int), efc_type=np.ones(1, dtype=int),
            efc_id=np.zeros(1, dtype=int),
        )
        populated = AUDIT._populated_constraint_state(
            SimpleNamespace(), SimpleNamespace(), data, [], [{"row_id": 0}],
            {"SOLVER_FORMULATION": "UNKNOWN"},
        )
        self.assertEqual(populated["POPULATED_CONSTRAINT_STATE"], "FAIL")
        self.assertFalse(populated["checks"]["solver_formulation_known"])

    def test_probe_classification_distinguishes_global_and_island_consumption(self):
        def probe(effect):
            return {"FIELD_PROBE_EFFECT": effect}

        probes = {
            "global_r_only": probe("NO_SOLVER_EFFECT"),
            "global_d_only": probe("NO_SOLVER_EFFECT"),
            "island_r_only": probe("OBSERVED"),
            "island_d_only": probe("OBSERVED"),
            "global_and_island_r": probe("OBSERVED"),
            "global_and_island_d": probe("OBSERVED"),
        }
        report = AUDIT._classify_populated_consumption(
            {"POPULATED_CONSTRAINT_STATE": "PASS", "nisland": 1},
            {"REGULARIZATION_AUDIT_PIPELINE_REPRODUCTION": "PASS"},
            probes,
            {"SOLVER_FORMULATION": "NEWTON_PRIMAL"},
        )
        self.assertEqual(report["R_CONSUMPTION_PATH"], "ISLAND_IEFC_R_DIRECT")
        self.assertEqual(report["D_CONSUMPTION_PATH"], "ISLAND_IEFC_D_DIRECT")
        self.assertEqual(report["AR_CONSUMPTION_PATH"], "NOT_APPLICABLE_PRIMAL_SOLVER")
        self.assertEqual(report["ISLAND_MIRROR_REQUIRED"], "YES")
        self.assertEqual(report["CONTACT_R_COUNTERFACTUAL_READY"], "YES")

    def test_source_hash_status_reports_true_only_when_before_and_after_are_complete(self):
        complete = {"a": "1", "b": "2"}
        self.assertTrue(AUDIT._source_hash_status(complete, dict(complete)))
        self.assertIsNone(AUDIT._source_hash_status({"a": None}, {"a": None}))

    def test_source_consumption_audit_requires_function_and_field_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = """
            void mj_makeConstraint(Model* m, Data* d) {
              d->efc_R[0] = 1.0;
              d->efc_D[0] = 1.0;
            }
            void mj_projectConstraint(Model* m, Data* d) {
              d->efc_R[0] = d->efc_R[0];
              d->efc_D[0] = d->efc_D[0];
              d->efc_AR[0] = d->efc_R[0];
            }
            void mj_fwdConstraint(Model* m, Data* d) {
              consume(d->efc_AR);
            }
            """
            (root / "engine_constraint.c").write_text(source, encoding="utf-8")
            report = AUDIT.audit_source_consumption(
                root,
                function_symbols={
                    "mj_makeConstraint": True,
                    "mj_projectConstraint": True,
                    "mj_fwdConstraint": True,
                },
                exposed_fields={
                    "efc_R": {"available": True},
                    "efc_D": {"available": True},
                    "efc_AR": {"available": True},
                },
            )
        self.assertEqual(report["audit_status"], "PASS")
        self.assertTrue(report["counterfactual_ready"])
        self.assertNotEqual(report["R_CONSUMPTION_PATH"], "UNDETERMINED")
        self.assertNotEqual(report["D_CONSUMPTION_PATH"], "UNDETERMINED")
        self.assertNotEqual(report["AR_CONSUMPTION_PATH"], "UNDETERMINED")
        self.assertEqual(report["ISLAND_MIRROR_REQUIRED"], "NO")

    def test_source_consumption_audit_fails_closed_without_source(self):
        with tempfile.TemporaryDirectory() as directory:
            report = AUDIT.audit_source_consumption(
                Path(directory),
                function_symbols={
                    "mj_makeConstraint": True,
                    "mj_projectConstraint": True,
                    "mj_fwdConstraint": True,
                },
                exposed_fields={
                    "efc_R": {"available": True},
                    "efc_D": {"available": True},
                    "efc_AR": {"available": True},
                },
            )
        self.assertEqual(report["audit_status"], "INSUFFICIENT_EVIDENCE")
        self.assertFalse(report["counterfactual_ready"])
        self.assertEqual(report["R_CONSUMPTION_PATH"], "UNDETERMINED")
        self.assertEqual(report["ISLAND_MIRROR_REQUIRED"], "YES")

    def test_island_mirror_evidence_blocks_unproven_intervention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "island_constraint.c").write_text(
                """
                void mj_makeConstraint(Model* m, Data* d) {
                  d->efc_R[0] = 1.0; d->efc_D[0] = 1.0; d->iefc_R[0] = d->efc_R[0];
                }
                void mj_projectConstraint(Model* m, Data* d) {
                  d->efc_R[0] = d->efc_R[0]; d->efc_D[0] = d->efc_D[0];
                  d->efc_AR[0] = d->efc_R[0]; d->iefc_D[0] = d->efc_D[0];
                }
                void mj_fwdConstraint(Model* m, Data* d) { consume(d->efc_AR); }
                """,
                encoding="utf-8",
            )
            report = AUDIT.audit_source_consumption(
                root,
                function_symbols={
                    "mj_makeConstraint": True,
                    "mj_projectConstraint": True,
                    "mj_fwdConstraint": True,
                },
                exposed_fields={
                    "efc_R": {"available": True},
                    "efc_D": {"available": True},
                    "efc_AR": {"available": True},
                },
            )
        self.assertEqual(report["ISLAND_MIRROR_REQUIRED"], "YES")
        self.assertFalse(report["counterfactual_ready"])

    def test_island_mirror_source_path_is_ready_only_with_reciprocal_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "island_constraint.c").write_text(
                """
                void mj_makeConstraint(Model* m, Data* d) {
                  d->efc_R[0] = 1.0; d->efc_D[0] = 1.0;
                  d->iefc_R[0] = d->efc_R[0]; d->iefc_D[0] = d->efc_D[0];
                }
                void mj_projectConstraint(Model* m, Data* d) {
                  d->efc_R[0] = d->efc_R[0]; d->efc_D[0] = d->efc_D[0];
                  d->efc_AR[0] = d->efc_R[0];
                  d->map_efc2iefc[0] = 0;
                  d->map_iefc2efc[0] = 0;
                }
                void mj_fwdConstraint(Model* m, Data* d) {
                  consume(d->iefc_R); consume(d->iefc_D); consume(d->efc_AR);
                }
                """,
                encoding="utf-8",
            )
            report = AUDIT.audit_source_consumption(
                root,
                function_symbols={
                    "mj_makeConstraint": True,
                    "mj_projectConstraint": True,
                    "mj_fwdConstraint": True,
                },
                exposed_fields={
                    "efc_R": {"available": True},
                    "efc_D": {"available": True},
                    "efc_AR": {"available": True},
                },
            )
        self.assertEqual(report["ISLAND_MIRROR_REQUIRED"], "YES")
        self.assertEqual(report["island_update_path"], "PROVEN_BY_SOURCE")
        self.assertTrue(report["counterfactual_ready"])

    def test_island_mirror_mapping_updates_only_selected_rows(self):
        data = SimpleNamespace(
            nefc=3,
            nisland=1,
            map_efc2iefc=np.asarray([2, 0, 1]),
            map_iefc2efc=np.asarray([1, 2, 0]),
            iefc_R=np.asarray([10.0, 20.0, 30.0]),
            iefc_D=np.asarray([0.1, 0.05, 1.0 / 30.0]),
        )
        state = AUDIT._island_regularization_state(data, [1])
        self.assertEqual(state["selected_iefc_rows"], [0])
        update = AUDIT._sync_island_regularization(data, [1], [2.0], [0.5])
        self.assertTrue(update["updated"])
        np.testing.assert_allclose(data.iefc_R, [2.0, 20.0, 30.0])
        np.testing.assert_allclose(data.iefc_D, [0.5, 0.05, 1.0 / 30.0])

    def test_baseline_reciprocal_gate_passes_and_rejects_bad_D(self):
        snapshot = {"efc_R": np.asarray([0.5, 0.25]), "efc_D": np.asarray([2.0, 4.0])}
        report = AUDIT._rd_gate(snapshot, [0, 1])
        self.assertTrue(report["valid"])
        bad = {"efc_R": np.asarray([0.5]), "efc_D": np.asarray([3.0])}
        self.assertFalse(AUDIT._rd_gate(bad, [0])["valid"])

    def test_dense_and_sparse_AR_layouts_are_decoded_without_row_order_assumptions(self):
        dense = type("Data", (), {"efc_AR": np.eye(3)})()
        matrix, layout = AUDIT._ar_matrix(dense, 3)
        self.assertEqual(layout["representation"], "dense")
        np.testing.assert_allclose(matrix, np.eye(3))

        sparse = type(
            "Data",
            (),
            {
                "efc_AR": np.asarray([2.0, 0.5, 3.0, -0.25]),
                "efc_AR_rownnz": np.asarray([2, 1, 1]),
                "efc_AR_rowadr": np.asarray([0, 2, 3]),
                "efc_AR_colind": np.asarray([2, 0, 1, 2]),
            },
        )()
        matrix, layout = AUDIT._ar_matrix(sparse, 3)
        self.assertEqual(layout["representation"], "sparse")
        np.testing.assert_allclose(matrix, [[0.5, 0.0, 2.0], [0.0, 3.0, 0.0], [0.0, 0.0, -0.25]])

    def test_preconstraint_snapshot_never_reads_physical_wrench(self):
        data = SimpleNamespace(nefc=0, efc_AR=np.zeros(0))
        model = SimpleNamespace(nv=0)
        with patch.object(
            AUDIT.aref_audit,
            "contact_force_readback",
            side_effect=RuntimeError("physical wrench must not be read pre-constraint"),
        ):
            staged = AUDIT._constraint_snapshot(data, object(), model)
            self.assertEqual(staged["physical_contact_forces"], [])
            with self.assertRaises(RuntimeError):
                AUDIT._constraint_snapshot(data, object(), model, read_physical_contact_forces=True)

    def test_regularization_activation_checks_selected_R_D_and_AR_diagonal(self):
        base_r = np.asarray([1.0, 2.0, 4.0])
        base_d = 1.0 / base_r
        base_ar = np.diag(base_r)
        altered_r = base_r.copy()
        altered_r[1] *= 0.1
        altered_d = 1.0 / altered_r
        altered_ar = base_ar.copy()
        altered_ar[1, 1] = altered_r[1]

        def condition(r, d, ar):
            return {
                "regularization": {
                    "before": {"efc_R": base_r, "efc_D": base_d, "ar_matrix": base_ar, "ar_layout": {"representation": "dense"}},
                    "after": {"efc_R": r, "efc_D": d, "ar_matrix": ar, "ar_layout": {"representation": "dense"}},
                    "core_unchanged_after_project": {"efc_J": True, "efc_vel": True, "efc_aref": True},
                }
            }

        before = condition(base_r, base_d, base_ar)
        zero = condition(altered_r, altered_d, altered_ar)
        report = AUDIT.regularization_activation_report(before, zero, [1])
        self.assertEqual(report["CONTACT_R_COUNTERFACTUAL_ACTIVATION"], "VALIDATED")
        self.assertTrue(report["checks"]["AR_delta_only_selected_diagonal"])

        bad_ar = altered_ar.copy()
        bad_ar[0, 2] = 1.0
        failed = AUDIT.regularization_activation_report(before, condition(altered_r, altered_d, bad_ar), [1])
        self.assertEqual(failed["CONTACT_R_COUNTERFACTUAL_ACTIVATION"], "FAILED")

    def test_floor_row_manifest_and_restore_regression_use_canonical_fields(self):
        data = SimpleNamespace(
            efc_type=np.asarray([1, 6, 6, 6]),
            efc_id=np.asarray([0, 2, 2, 2]),
        )
        snapshot = {
            "efc_R": np.asarray([9.0, 0.5, 0.4, 0.3]),
            "efc_D": np.asarray([1.0 / 9.0, 2.0, 2.5, 10.0 / 3.0]),
            "ar_matrix": np.diag([9.0, 0.5, 0.4, 0.3]),
        }
        decomposition = {4: {"row_ids": [3, 1, 2, 1]}}
        manifest = AUDIT.selected_floor_contact_rows(data, decomposition, snapshot)
        self.assertEqual([item["row_id"] for item in manifest], [3, 1, 2])
        self.assertEqual([item["efc_type"] for item in manifest], [6, 6, 6])
        self.assertTrue(all(item["baseline_AR_diagonal"] is not None for item in manifest))

        def condition():
            return {
                "regularization": {"after": {"efc_R": [0.5], "efc_D": [2.0], "ar_matrix": [[0.5]]}},
                "post_constraint_snapshot": {"efc_force": [3.0]},
                "capture": {"contacts": [{"tangential_impulse": [1.0, 2.0], "normal_impulse": 4.0}]},
                "excess": {
                    "rigid_demand_vector": [0.0, 1.0],
                    "solver_excess_vector": [1.0, 1.0],
                    "post_slip": [0.1, 0.2],
                },
            }
        self.assertEqual(
            AUDIT._restore_regression(condition(), condition())["R_RESTORE_REPRODUCTION"],
            "PASS",
        )

    def test_classification_thresholds_and_next_actions(self):
        baseline = {"solver_excess_norm": 1.0, "solver_excess_vector_norm": 1.0}
        strong = AUDIT.classify_effect(baseline, {"solver_excess_norm": 0.2, "solver_excess_vector_norm": 0.3}, True)
        self.assertEqual(strong["CONTACT_R_SOLVER_EXCESS_EFFECT"], "STRONG_REDUCTION")
        self.assertEqual(strong["NEXT_ACTION"], "NO_ADDITIONAL_SOLVER_COUNTERFACTUAL_REQUIRED")
        partial = AUDIT.classify_effect(baseline, {"solver_excess_norm": 0.6, "solver_excess_vector_norm": 0.6}, True)
        self.assertEqual(partial["CONTACT_R_SOLVER_EXCESS_EFFECT"], "PARTIAL_REDUCTION")
        self.assertEqual(partial["NEXT_ACTION"], "TARGET_SOLIMP_COUNTERFACTUAL")
        no_effect = AUDIT.classify_effect(baseline, {"solver_excess_norm": 1.2, "solver_excess_vector_norm": 1.2}, True)
        self.assertEqual(no_effect["CONTACT_R_SOLVER_EXCESS_EFFECT"], "INCREASED")
        self.assertEqual(no_effect["NEXT_ACTION"], "SOLVER_OPTIMIZATION_DIAGNOSTIC")
        blocked = AUDIT.classify_effect(baseline, baseline, False)
        self.assertEqual(blocked["CONTACT_R_SOLVER_EXCESS_EFFECT"], "INSUFFICIENT_EVIDENCE")

    def test_required_condition_and_replay_contracts_are_fixed(self):
        self.assertEqual(AUDIT.EXPECTED_SUBSTEPS, 120)
        self.assertEqual([item[2] for item in AUDIT.CONDITIONS], [1.0, 0.1, 1.0])
        self.assertEqual([item[1] for item in AUDIT.CONDITIONS], [
            "R_SCALE_1_BEFORE", "R_SCALE_0P1", "R_SCALE_1_AFTER_RESTORE"
        ])
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn("cone_helper.replay_once(args, paths)", source)
        self.assertIn('"formal_replay_additional_steps": 0', source)
        self.assertIn("mj_contactForce", source)
        self.assertIn("SHARED_PHYSICAL_GLOBAL_COUPLED_DEMAND", source)
        self.assertIn("not a pure tangent-only R intervention", source)

    def test_consumption_audit_only_has_a_dedicated_cli_mode(self):
        audit_args = AUDIT.parser().parse_args(["--mode", "consumption-audit-only"])
        self.assertEqual(audit_args.mode, "consumption-audit-only")
        self.assertEqual(
            AUDIT._default_output_paths(audit_args.mode)[0].name.split("_")[4],
            "consumption",
        )

    def test_consumption_audit_dispatches_before_legacy_empty_state_audit(self):
        args = SimpleNamespace(mode="consumption-audit-only")
        paths = {"output_dir": Path("unused")}
        expected = {"CONTACT_R_COUNTERFACTUAL_READY": "NO"}
        with patch.object(
            AUDIT,
            "execute_consumption_audit_only",
            return_value=expected,
        ) as populated_audit, patch.object(
            AUDIT,
            "run_solver_consumption_audit",
            side_effect=AssertionError("legacy empty-state audit must not run"),
        ):
            self.assertEqual(AUDIT.execute(args, paths), expected)
        populated_audit.assert_called_once_with(args, paths)

    def test_consumption_audit_only_never_invokes_formal_R_conditions(self):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "execute_consumption_audit_only"
        )
        called = {
            node.func.id
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("run_condition", called)
        self.assertNotIn("apply_regularization_scale", called)
        self.assertNotIn("custom_pipeline_one_step_regression", called)

    def test_audit_only_validation_never_claims_solver_excess_result(self):
        populated = {"POPULATED_CONSTRAINT_STATE": "FAIL"}
        validation = AUDIT._audit_only_validation(
            populated, None, None, None, 120, True
        )
        self.assertEqual(
            validation["CONTACT_R_SOLVER_EXCESS_EFFECT"],
            "NOT_RUN_AUDIT_ONLY",
        )
        self.assertEqual(
            validation["MUJOCO_SOLVER_EXCESS_CAUSAL_DRIVER"],
            "NOT_RUN_AUDIT_ONLY",
        )
        self.assertFalse(validation["formal_R_counterfactual_was_run"])
        self.assertEqual(validation["formal_replay_physics_substeps"], 120)

    def test_audit_only_results_write_canonical_failure_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            validation = AUDIT._audit_only_validation(
                {"POPULATED_CONSTRAINT_STATE": "FAIL"},
                None,
                None,
                None,
                120,
                True,
            )
            AUDIT._write_audit_only_results(
                output,
                validation,
                {"source.py": None},
                [Path("source.py")],
                {"source.py": None},
                None,
                None,
            )
            audit = json.loads(
                (output / "constraint_regularization_consumption_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["ISLAND_MIRROR_REQUIRED"], "UNDETERMINED")
            self.assertEqual(
                audit["CONTACT_R_SOLVER_EXCESS_EFFECT"],
                "NOT_RUN_AUDIT_ONLY",
            )
            self.assertEqual(
                summary["SUMMARY_CLASSIFICATION_CONSISTENCY"],
                "PASS",
            )
            self.assertTrue(
                (output / "regularization_audit_pipeline_reproduction.json").is_file()
            )

    def test_audit_only_failure_bundle_keeps_unknown_source_purity_and_inner_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            AUDIT.write_failure_bundle(
                output,
                RuntimeError("populated audit failed"),
                "Traceback: populated audit failed\n",
                mode="consumption-audit-only",
            )
            validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
            purity = json.loads((output / "source_purity.json").read_text(encoding="utf-8"))
            self.assertEqual(validation["ISLAND_MIRROR_REQUIRED"], "UNDETERMINED")
            self.assertEqual(
                validation["CONTACT_R_SOLVER_EXCESS_EFFECT"],
                "NOT_RUN_AUDIT_ONLY",
            )
            self.assertIsNone(purity["source_hashes_unchanged"])
            self.assertTrue((output / "inner_exception_traceback.txt").is_file())
            self.assertTrue((output / "audit_phase.json").is_file())

    def test_only_allowed_solver_arrays_are_written_and_prohibited_options_are_not(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn("data.efc_R", source)
        self.assertIn("data.efc_D", source)
        self.assertNotIn("data.efc_aref[", source)
        self.assertNotIn("model.opt.solref =", source)
        self.assertNotIn("model.opt.solimp =", source)
        self.assertNotIn("model.opt.impratio =", source)
        self.assertNotIn("model.opt.cone =", source)
        self.assertNotIn("model.opt.solver =", source)

    def test_failure_payload_is_fail_closed_and_json_normalizes_numpy_values(self):
        report = AUDIT.failure_payload(RuntimeError("boom"))
        self.assertEqual(report["LOCAL_IMPLEMENTATION"], "INCOMPLETE")
        self.assertEqual(report["CONTACT_R_SOLVER_EXCESS_EFFECT"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(report["UNCONDITIONAL_ZIP_PACKAGING"], "ENABLED")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.json"
            AUDIT.write_json(path, {"check": np.bool_(True), "value": np.float64(1.25)})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"check": True, "value": 1.25})

    def test_success_failure_zip_testzip_sidecar_and_stdout_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact"
            artifact.mkdir()
            (artifact / "summary.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
            archive = root / "artifact.zip"
            packaged = AUDIT.oracle.package_artifact(artifact, archive)
            self.assertEqual(packaged["ZIP_VERIFY"], "PASS")
            self.assertTrue(Path(packaged["SHA256_SIDECAR"]).is_file())
            with zipfile.ZipFile(archive) as bundle:
                self.assertIsNone(bundle.testzip())
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn('print(f"ZIP_VERIFY=', source)
        self.assertIn('print(f"ZIP_SHA256=', source)
        self.assertIn('print(f"UPLOAD_THIS_ZIP=', source)

    def test_invariant_and_failure_artifacts_are_named(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        for filename in (
            "constraint_regularization_consumption_audit.json",
            "solver_storage_semantics.json",
            "probe_global_r_only.json",
            "probe_global_d_only.json",
            "probe_island_r_only.json",
            "probe_island_d_only.json",
            "probe_global_and_island_r.json",
            "probe_global_and_island_d.json",
            "probe_effect_summary.json",
            "selected_floor_contact_rows.json",
            "regularization_counterfactual_activation.json",
            "regularization_invariant_validation.json",
            "regularization_pipeline_baseline_regression.json",
            "baseline_regression.json",
            "restore_regression.json",
            "regularization_counterfactual_comparison.json",
            "failure_context.json",
            "traceback.txt",
        ):
            if filename.startswith("probe_") and filename.endswith(".json"):
                self.assertIn("probe_{probe_name}.json", source)
            else:
                self.assertIn(filename, source)
        tree = ast.parse(source)
        run_condition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_condition")
        calls = {node.func.attr for node in ast.walk(run_condition) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertNotIn("mj_forward", calls)
        self.assertNotIn("mj_step", calls)


if __name__ == "__main__":
    unittest.main()
