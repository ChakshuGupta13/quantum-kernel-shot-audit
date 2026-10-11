import copy
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from sklearn.exceptions import ConvergenceWarning

import prepare_data
import evaluate_models
import diagnose_training


FIXTURE_SEED = 2026
TRAIN_ROWS = 8
TEST_ROWS = 3
FEATURES = 6


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        random = np.random.default_rng(FIXTURE_SEED)
        self.training = random.normal(size=(TRAIN_ROWS, FEATURES))
        self.testing = random.normal(size=(TEST_ROWS, FEATURES))

    def test_live_settings_match_the_frozen_protocol(self):
        protocol = prepare_data.read_protocol()
        self.assertEqual(protocol["preprocessing"]["steps"][1]["n_components"], prepare_data.SETTINGS.n_components)

    def test_test_poisoning_cannot_change_training_fit(self):
        original = prepare_data.fit_angles(self.training, self.testing)
        poisoned = prepare_data.fit_angles(self.training, self.testing + 1000)
        np.testing.assert_array_equal(original[0], poisoned[0])
        for name in ("standard", "pca", "angles", "settings", "training_rows_fitted"):
            self.assertEqual(original[2][name], poisoned[2][name])
        self.assertFalse(np.allclose(original[1], poisoned[1]))
        self.assertTrue(poisoned[2]["clipped_test_coordinates"])

    def test_fitted_parameters_replay_angles_and_clipping(self):
        training, testing, audit = prepare_data.fit_angles(self.training, self.testing * 100)
        scaler, pca, angles = audit["standard"], audit["pca"], audit["angles"]
        for values, expected in ((self.training, training), (self.testing * 100, testing)):
            standardized = (values - scaler["mean_"]) / scaler["scale_"]
            projected = (standardized - pca["mean_"]) @ np.asarray(pca["components_"]).T
            unbounded = projected * angles["scale_"] + angles["min_"]
            replayed = np.clip(unbounded, prepare_data.SETTINGS.angle_lower, prepare_data.SETTINGS.angle_upper)
            np.testing.assert_allclose(replayed, expected, atol=1e-12, rtol=0)
        self.assertEqual(scaler["n_samples_seen_"], TRAIN_ROWS)
        self.assertEqual(angles["n_samples_seen_"], TRAIN_ROWS)
        self.assertEqual(pca["n_samples_"], TRAIN_ROWS)
        self.assertEqual(audit["pipeline_fit_calls"], 1)
        json.dumps(audit, allow_nan=False)
        for coordinate in audit["clipped_test_coordinates"]:
            self.assertEqual(coordinate["angle"], testing[coordinate["test_position"], coordinate["component"]])
            self.assertTrue(coordinate["unclipped_angle"] < 0 or coordinate["unclipped_angle"] > np.pi)

    def test_invalid_inputs_and_protocol_mutation_fail(self):
        for training, testing in (([], self.testing), (self.training, self.testing[:, :-1]), (self.training, np.full_like(self.testing, np.nan))):
            with self.assertRaises(ValueError):
                prepare_data.fit_angles(training, testing)
        payload = (prepare_data.ROOT / "configuration.json").read_bytes() + b"\n"
        with patch.object(type(prepare_data.ROOT), "read_bytes", return_value=payload):
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                prepare_data.read_protocol()

    def test_shared_preprocessor_is_fitted_once_on_training_only(self):
        pipeline = prepare_data.make_preprocessing()
        original_fit = pipeline.fit
        with patch.object(prepare_data, "make_preprocessing", return_value=pipeline), patch.object(pipeline, "fit", wraps=original_fit) as fit:
            prepare_data.fit_angles(self.training, self.testing)
        self.assertEqual(fit.call_count, 1)
        np.testing.assert_array_equal(fit.call_args.args[0], self.training)


class PreparationCaptureTests(unittest.TestCase):
    def setUp(self):
        self.protocol = copy.deepcopy(prepare_data.read_protocol())
        random = np.random.default_rng(FIXTURE_SEED)
        self.source = SimpleNamespace(
            data=random.normal(size=(569, 30)), target=np.zeros(569, dtype=np.uint8),
            feature_names=np.asarray([f"feature_{index}" for index in range(30)]),
            target_names=np.asarray(["malignant", "benign"]),
        )
        self.protocol["dataset"]["sha256"] = prepare_data.dataset_digest(self.source)

    def test_changed_dataset_is_rejected_before_preprocessing(self):
        protocol = prepare_data.read_protocol()
        with patch.object(prepare_data, "load_breast_cancer", return_value=self.source), patch.object(prepare_data, "fit_angles") as fit:
            with self.assertRaisesRegex(ValueError, "Dataset identity differs"):
                prepare_data.admit_dataset(protocol)
        fit.assert_not_called()

    def test_bundle_records_one_fit_and_blocks_repeat_before_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prepared"
            with patch.object(prepare_data, "read_protocol", return_value=self.protocol), patch.object(prepare_data, "admit_dataset", return_value=self.source), patch.object(prepare_data, "fit_angles", wraps=prepare_data.fit_angles) as fit:
                report = prepare_data.prepare(output)
                self.assertEqual(fit.call_count, 1)
                self.assertEqual(report["pair_count"], 1008)
                self.assertEqual(report["quantum_jobs"], 0)
                self.assertFalse(report["classifiers_fitted"])
                self.assertFalse(report["full_sampling_authorized"])
                self.assertEqual(prepare_data.verify_preparation(output), report)
                with self.assertRaises(FileExistsError):
                    prepare_data.prepare(output)
                self.assertEqual(fit.call_count, 1)
                receipt = json.loads((output / "complete.json").read_text())
                self.assertEqual(set(receipt["files"]), set(prepare_data.PREPARATION_ARTIFACTS))

    def test_preprocessing_failure_is_retained_and_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed"
            with patch.object(prepare_data, "read_protocol", return_value=self.protocol), patch.object(prepare_data, "admit_dataset", return_value=self.source), patch.object(prepare_data, "fit_angles", side_effect=RuntimeError("synthetic preprocessing failure")) as fit:
                with self.assertRaisesRegex(RuntimeError, "synthetic preprocessing failure"):
                    prepare_data.prepare(output)
                with self.assertRaises(FileExistsError):
                    prepare_data.prepare(output)
                self.assertEqual(fit.call_count, 1)
            failure = json.loads((output / "failure.json").read_text())
            self.assertEqual(failure["phase"], "preprocessing")
            self.assertFalse((output / "complete.json").exists())
            self.assertTrue((output / "implementation-configuration.json").exists())

    def test_rehashed_noncanonical_row_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tampered"
            with patch.object(prepare_data, "read_protocol", return_value=self.protocol), patch.object(prepare_data, "admit_dataset", return_value=self.source):
                prepare_data.prepare(output)
                angles_path = output / "angles.json"
                angles = json.loads(angles_path.read_text())
                angles["test_row_ids"] = list(reversed(angles["test_row_ids"]))
                angles_path.write_text(json.dumps(angles))
                receipt_path = output / "complete.json"
                receipt = json.loads(receipt_path.read_text())
                receipt["files"]["angles.json"] = prepare_data.file_record(angles_path)
                receipt_path.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, "row IDs differ"):
                    prepare_data.verify_preparation(output)


class ClassifierEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.protocol = prepare_data.read_protocol()
        self.training = np.array([[0.1, 0.2, 0.1, 0.2], [0.3, 0.4, 0.3, 0.4],
                                  [2.6, 2.7, 2.6, 2.7], [2.9, 3.0, 2.9, 3.0]])
        self.testing = np.array([[0.2, 0.3, 0.2, 0.3], [2.8, 2.9, 2.8, 2.9]])
        self.train_labels = np.array([0, 0, 1, 1])
        self.test_labels = np.array([0, 1])
        self.angles = {"training_angles": self.training.tolist(), "testing_angles": self.testing.tolist(),
                       "test_row_ids": [101, 202]}
        training_kernel = self.training @ self.training.T
        cross_kernel = self.testing @ self.training.T
        self.matrices = {"ideal_training": training_kernel, "ideal_test_training": cross_kernel,
                         "sampled_training": training_kernel.copy(), "sampled_test_training": cross_kernel.copy()}

    def test_metrics_match_independent_binary_toy(self):
        metrics = evaluate_models.prediction_metrics([0, 0, 1, 1], [0, 1, 1, 1], [-0.5, 0.8, 0.3, 0.9])
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["balanced_accuracy"], 0.75)
        self.assertEqual(metrics["roc_auc_from_decision_function"], 0.75)
        self.assertEqual(metrics["confusion_matrix_0_1"], [[1, 1], [0, 2]])
        with self.assertRaises(ValueError):
            evaluate_models.prediction_metrics([0, 1], [0], [0, 1])
        with self.assertRaises(ValueError):
            evaluate_models.prediction_metrics([0, 1], [0, 1], [0, np.nan])

    def test_classifier_parameters_match_the_contract(self):
        self.assertEqual(evaluate_models.shared_parameters(), self.protocol["classifiers"]["shared"])
        for row in self.protocol["classifiers"]["rows"]:
            model = evaluate_models.make_classifier(row)
            self.assertEqual(model.kernel, row["kernel"])
            self.assertFalse(hasattr(model, "classes_"))

    def test_four_toy_classifiers_retain_complete_prediction_records(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = evaluate_models.evaluate_rows(self.protocol, self.angles, self.matrices,
                                                self.train_labels, self.test_labels, Path(directory) / "rows")
        self.assertEqual(set(rows), {"quantum_ideal", "quantum_shot", "linear", "rbf"})
        for record in rows.values():
            self.assertEqual(record["calls"], {"fit": 1, "predict": 1, "decision_function": 1})
            self.assertEqual([item["row_id"] for item in record["records"]], [101, 202])
            self.assertEqual(record["metrics"]["accuracy"], 1.0)
        self.assertEqual(rows["quantum_ideal"]["training_shape"], [4, 4])
        self.assertEqual(rows["linear"]["training_shape"], [4, 4])

    def test_estimator_calls_use_only_training_for_fit(self):
        models = []

        class CountingModel:
            def __init__(self):
                self.calls = []
                self.classes_ = np.array([0, 1])
                self._gamma = 0.25

            def fit(self, inputs, labels):
                self.calls.append(("fit", np.array(inputs), np.array(labels)))
                return self

            def predict(self, inputs):
                self.calls.append(("predict", np.array(inputs)))
                return np.array([0, 1])

            def decision_function(self, inputs):
                self.calls.append(("decision_function", np.array(inputs)))
                return np.array([-1.0, 1.0])

            def get_params(self, deep=False):
                return {}

        def factory(row):
            model = CountingModel()
            models.append((row, model))
            return model

        with tempfile.TemporaryDirectory() as directory, patch.object(evaluate_models, "make_classifier", side_effect=factory):
            evaluate_models.evaluate_rows(self.protocol, self.angles, self.matrices,
                                         self.train_labels, self.test_labels, Path(directory) / "rows")
        self.assertEqual(len(models), 4)
        for row, model in models:
            self.assertEqual([call[0] for call in model.calls], ["fit", "predict", "decision_function"])
            expected_train, expected_test = evaluate_models.row_inputs(row, self.angles, self.matrices)
            np.testing.assert_array_equal(model.calls[0][1], expected_train)
            np.testing.assert_array_equal(model.calls[0][2], self.train_labels)
            np.testing.assert_array_equal(model.calls[1][1], expected_test)
            np.testing.assert_array_equal(model.calls[2][1], expected_test)

    def test_convergence_warning_stops_the_row_without_substitution(self):
        class FailedModel:
            def fit(self, inputs, labels):
                warnings.warn("synthetic nonconvergence", ConvergenceWarning)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rows"
            with patch.object(evaluate_models, "make_classifier", return_value=FailedModel()) as factory:
                with self.assertRaises(ConvergenceWarning):
                    evaluate_models.evaluate_rows(self.protocol, self.angles, self.matrices,
                                                 self.train_labels, self.test_labels, output)
                self.assertEqual(factory.call_count, 1)
            failure = json.loads((output / "quantum_ideal-failure.json").read_text())
            self.assertEqual(failure["phase"], "fit")
            self.assertEqual(failure["calls_attempted"]["fit"], 1)
            self.assertFalse((output / "quantum_ideal.json").exists())

    def test_real_plan_json_roundtrip_without_fitting_or_loading_labels(self):
        with patch.object(evaluate_models.SVC, "fit", side_effect=AssertionError("Plan must not fit")) as fit, patch.object(evaluate_models, "admit_dataset", side_effect=AssertionError("Plan must not load labels")) as dataset:
            plan = evaluate_models.evaluation_plan()
        self.assertEqual(plan, json.loads(json.dumps(plan, allow_nan=False)))
        self.assertEqual(len(plan["rows"]), 4)
        self.assertEqual(len(plan["test_row_ids"]), 16)
        self.assertFalse(plan["evaluation_authorized"])
        fit.assert_not_called()
        dataset.assert_not_called()

    def test_unapproved_evaluation_stops_before_input_or_model_access(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unapproved"
            with patch.object(evaluate_models, "evaluation_plan") as plan, patch.object(evaluate_models, "evaluate_rows") as rows:
                with self.assertRaisesRegex(ValueError, "explicit approval"):
                    evaluate_models.run_evaluation(Path(directory) / "absent.json", output)
            plan.assert_not_called()
            rows.assert_not_called()
            self.assertFalse(output.exists())

    def test_sealed_toy_run_completes_once_and_blocks_repeat(self):
        plan = {"status": "synthetic_evaluation_plan"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation"
            plan_file = Path(directory) / "plan.json"
            prepare_data.write_record(plan_file, plan)
            with patch.object(evaluate_models, "evaluation_plan", return_value=plan), patch.object(evaluate_models, "load_evaluation_inputs", return_value=(self.angles, self.matrices, self.train_labels, self.test_labels)), patch.object(evaluate_models, "evaluate_rows", wraps=evaluate_models.evaluate_rows) as rows:
                report = evaluate_models.run_evaluation(plan_file, output, approved=True)
                with self.assertRaises(FileExistsError):
                    evaluate_models.run_evaluation(plan_file, output, approved=True)
                self.assertEqual(rows.call_count, 1)
            self.assertEqual(report["calls"], {"fit": 4, "predict": 4, "decision_function": 4})
            self.assertEqual(report["quantum_jobs"], 0)
            self.assertEqual(report["preprocessing_fits"], 0)
            self.assertEqual(report["shot_minus_ideal_balanced_accuracy"], 0)
            receipt = json.loads((output / "complete.json").read_text())
            self.assertTrue(receipt["comparison_complete"])
            for name, record in receipt["files"].items():
                self.assertEqual(prepare_data.file_record(output / name), record)

    def test_changed_evaluation_plan_blocks_fitting(self):
        expected = {"regularization": 1.0}
        changed = {"regularization": 2.0}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "changed"
            plan_file = Path(directory) / "plan.json"
            prepare_data.write_record(plan_file, changed)
            with patch.object(evaluate_models, "evaluation_plan", return_value=expected), patch.object(evaluate_models, "evaluate_rows") as rows:
                with self.assertRaisesRegex(ValueError, "plan or frozen inputs changed"):
                    evaluate_models.run_evaluation(plan_file, output, approved=True)
            rows.assert_not_called()
            self.assertFalse((output / "complete.json").exists())

    def test_post_fit_binding_change_blocks_completion(self):
        expected = {"status": "synthetic_evaluation_plan"}
        changed = {**expected, "source_changed": True}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "changed-after-fit"
            plan_file = Path(directory) / "plan.json"
            prepare_data.write_record(plan_file, expected)
            with patch.object(evaluate_models, "evaluation_plan", side_effect=[expected, changed]), patch.object(evaluate_models, "load_evaluation_inputs", return_value=(self.angles, self.matrices, self.train_labels, self.test_labels)):
                with self.assertRaisesRegex(ValueError, "bindings changed"):
                    evaluate_models.run_evaluation(plan_file, output, approved=True)
            self.assertFalse((output / "complete.json").exists())
            self.assertTrue((output / "rows/quantum_ideal.json").exists())
            self.assertEqual(json.loads((output / "failure.json").read_text())["status"], "integrity_failed")


class TrainingDiagnosticTests(unittest.TestCase):
    def test_label_kernel_and_identity_reference(self):
        labels = np.array([0, 0, 1, 1])
        signed = 2 * labels - 1
        record = diagnose_training.kernel_summary(np.outer(signed, signed), labels, 1.0)
        self.assertAlmostEqual(record["centered_label_alignment"], 1)
        self.assertEqual(record["within_class"]["count"], 2)
        self.assertEqual(record["between_class"]["count"], 4)
        self.assertEqual(record["within_class"]["mean"], 1)
        self.assertEqual(record["between_class"]["mean"], -1)
        identity = diagnose_training.kernel_summary(np.eye(4), labels, 1.0)
        self.assertEqual(identity["within_class"]["mean"], 0)
        self.assertEqual(identity["between_class"]["mean"], 0)
        self.assertAlmostEqual(identity["centered_label_alignment"], 1 / np.sqrt(3))

    def test_pooled_mean_weights_pairs_not_class_means(self):
        labels = np.array([0, 0, 0, 1, 1])
        matrix = np.full((5, 5), 4.0)
        matrix[:3, :3] = 2.0
        matrix[3:, 3:] = 10.0
        np.fill_diagonal(matrix, 1)
        record = diagnose_training.kernel_summary(matrix, labels, 1.0)
        self.assertEqual(record["within_benign"]["count"], 3)
        self.assertEqual(record["within_malignant"]["count"], 1)
        self.assertEqual(record["within_class"]["mean"], 4)
        self.assertNotEqual(record["within_class"]["mean"], 6)
        self.assertEqual(record["between_class"]["count"], 6)

    def test_alignment_is_scale_invariant_but_matrix_scale_changes(self):
        labels = np.array([0, 0, 1, 1])
        signed = 2 * labels - 1
        matrix = np.outer(signed, signed) + np.eye(4)
        original = diagnose_training.kernel_summary(matrix, labels, 1.0)
        scaled = diagnose_training.kernel_summary(7 * matrix, labels, 1.0)
        self.assertAlmostEqual(original["centered_label_alignment"], scaled["centered_label_alignment"])
        self.assertAlmostEqual(scaled["frobenius_norm"], 7 * original["frobenius_norm"])
        self.assertAlmostEqual(scaled["maximum_eigenvalue"], 7 * original["maximum_eigenvalue"])

    def test_undefined_alignment_and_invalid_inputs(self):
        labels = [0, 0, 1, 1]
        record = diagnose_training.kernel_summary(np.ones((4, 4)), labels, 1.0)
        self.assertIsNone(record["centered_label_alignment"])
        self.assertFalse(record["alignment_defined"])
        for matrix, bad_labels in ((np.eye(3), labels), (np.full((4, 4), np.nan), labels), (np.eye(4), [0, 0, 0, 0])):
            with self.assertRaises(ValueError):
                diagnose_training.kernel_summary(matrix, bad_labels, 1.0)

    def test_classical_kernels_reproduce_direct_training_formulas(self):
        coordinates = np.array([[0, 1], [1, 0], [2, 1]], dtype=float)
        matrices, parameters = diagnose_training.classical_kernels(coordinates)
        expected_gamma = 1 / (coordinates.shape[1] * coordinates.var())
        np.testing.assert_allclose(matrices["linear"], coordinates @ coordinates.T)
        distances = ((coordinates[:, None, :] - coordinates[None, :, :]) ** 2).sum(axis=2)
        np.testing.assert_allclose(matrices["rbf"], np.exp(-expected_gamma * distances))
        self.assertEqual(parameters["rbf_gamma"], expected_gamma)
        constant, settings = diagnose_training.classical_kernels(np.ones((3, 2)))
        self.assertEqual(settings["rbf_gamma"], 1)
        np.testing.assert_array_equal(constant["rbf"], np.ones((3, 3)))

    def test_runner_uses_training_fields_and_never_fits(self):
        coordinates = np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5],
                                [2.5, 2.6, 2.7, 2.8], [2.6, 2.7, 2.8, 2.9]])
        source = SimpleNamespace(data=coordinates, target=np.array([1, 1, 0, 0]),
                                 target_names=np.array(["malignant", "benign"]),
                                 feature_names=np.array(["one", "two", "three", "four"]))
        protocol = {"dataset": {"raw_target_names": source.target_names.tolist(), "sha256": prepare_data.dataset_digest(source)},
                    "partition": {"train_row_ids": [0, 1, 2, 3], "train_class_counts_0_1": [2, 2]},
                    "quantum": {"feature_dimension": 4}, "classifiers": {"shared": {"C": 1.0}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "preparation-v1").mkdir()
            (root / "sampling-v1").mkdir()
            prepare_data.write_record(root / "preparation-v1/angles.json", {
                "train_row_ids": [0, 1, 2, 3], "training_angles": coordinates.tolist(),
                "testing_angles": "This field must not enter the numerical calculation",
            })
            np.savez_compressed(root / "sampling-v1/repaired-kernels.npz", ideal_training=np.eye(4),
                                sampled_training=np.eye(4), unused_test_data=np.array(["not numeric"]))
            with patch.object(diagnose_training, "ROOT", root), patch.object(diagnose_training, "read_protocol", return_value=protocol), patch.object(diagnose_training, "input_bindings", return_value={}), patch.object(diagnose_training, "load_breast_cancer", return_value=source), patch.object(evaluate_models.SVC, "fit", side_effect=AssertionError("No classifier fitting")) as model_fit, patch.object(prepare_data.StandardScaler, "fit", side_effect=AssertionError("No preprocessing fitting")) as preprocessing_fit:
                report = diagnose_training.run_diagnostics(root / "diagnostic")
            model_fit.assert_not_called()
            preprocessing_fit.assert_not_called()
            self.assertFalse(report["test_labels_used"])
            self.assertFalse(report["test_predictions_used"])
            self.assertEqual(report["classifiers_fitted"], 0)
            self.assertEqual(report["quantum_jobs"], 0)
            self.assertEqual(report["training_row_ids"], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()