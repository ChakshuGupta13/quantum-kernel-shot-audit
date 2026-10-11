import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.result import Result

import kernels
import calibrate
import sample_kernels


TRAINING = [kernels.SETTINGS.first_angles, kernels.SETTINGS.second_angles]
TESTING = [(0.3, 0.5, 0.7, 0.9)]
TRAIN_IDS = [101, 202]
TEST_IDS = [303]


class KernelBuilderTests(unittest.TestCase):
    def test_live_settings_match_frozen_contract(self):
        record = kernels.implementation_configuration()
        self.assertEqual(record["protocol_sha256"], kernels.PROTOCOL_SHA256)
        self.assertEqual(record["settings"]["shots_per_entry"], 256)

    def test_compute_uncompute_matches_independent_overlap(self):
        first = kernels.feature_circuit(TRAINING[0])
        second = kernels.feature_circuit(TRAINING[1])
        first_state = Statevector.from_instruction(first)
        for other in (first, second):
            expected = abs(np.vdot(Statevector.from_instruction(other).data, first_state.data)) ** 2
            circuit = kernels.overlap_circuit(first, other, measured=False)
            observed = Statevector.from_instruction(circuit).probabilities()[0]
            self.assertAlmostEqual(observed, expected)
        wrong = first.copy()
        wrong.compose(first, inplace=True)
        self.assertFalse(math.isclose(Statevector.from_instruction(wrong).probabilities()[0], 1,
                                     rel_tol=0, abs_tol=kernels.SETTINGS.numeric_tolerance))

    def test_ideal_matrix_layout_and_positive_semidefiniteness(self):
        training, cross = kernels.ideal_kernels(TRAINING, TESTING)
        self.assertEqual(training.shape, (2, 2))
        self.assertEqual(cross.shape, (1, 2))
        np.testing.assert_allclose(training, training.T)
        np.testing.assert_array_equal(np.diag(training), [1, 1])
        self.assertGreaterEqual(np.linalg.eigvalsh(training).min(), -kernels.SETTINGS.numeric_tolerance)
        test_state = Statevector.from_instruction(kernels.feature_circuit(TESTING[0]))
        for column, angles in enumerate(TRAINING):
            train_state = Statevector.from_instruction(kernels.feature_circuit(angles))
            self.assertAlmostEqual(cross[0, column], abs(np.vdot(train_state.data, test_state.data)) ** 2)

    def test_pair_order_does_not_skip_duplicate_angles(self):
        circuits, pairs = kernels.build_overlap_batch([TRAINING[0], TRAINING[0]], TESTING, TRAIN_IDS, TEST_IDS)
        self.assertEqual([(pair["block"], pair["row"], pair["column"]) for pair in pairs],
                         [("train", 0, 1), ("test_train", 0, 0), ("test_train", 0, 1)])
        self.assertEqual(len(circuits), 3)
        self.assertTrue(all(circuit.num_qubits == 4 and circuit.num_clbits == 4 for circuit in circuits))
        self.assertEqual(circuits[2].metadata["left_row_id"], TEST_IDS[0])

    def test_count_reconstruction_retains_denominators_and_symmetry(self):
        pairs = kernels.ordered_pairs(TRAIN_IDS, TEST_IDS)
        counts = [{"0000": 64, "1111": 192}, {"0000": 128, "1111": 128}, {"1111": 256}]
        training, cross = kernels.kernels_from_counts(counts, pairs, TRAIN_IDS, TEST_IDS)
        np.testing.assert_array_equal(training, [[1, 0.25], [0.25, 1]])
        np.testing.assert_array_equal(cross, [[0.5, 0]])
        for bad_counts in ({"0000": 255}, {"000": 256}, {"0000": -256}, {"0000": True}):
            with self.assertRaises(ValueError):
                kernels.count_probability(bad_counts)
        with self.assertRaises(ValueError):
            kernels.kernels_from_counts(counts[:-1], pairs, TRAIN_IDS, TEST_IDS)
        with self.assertRaises(ValueError):
            kernels.kernels_from_counts(counts, list(reversed(pairs)), TRAIN_IDS, TEST_IDS)

    def test_repair_uses_consistent_out_of_sample_features(self):
        raw = np.array([[1, 230 / 256, 230 / 256], [230 / 256, 1, 0], [230 / 256, 0, 1]])
        cross = np.array([[0.2, 0.8, 0.5]])
        repaired, repaired_cross, diagnostic = kernels.repair_kernels(raw, cross)
        values, vectors = np.linalg.eigh(raw)
        positive = values > kernels.SETTINGS.eigenvalue_threshold
        train_features = vectors[:, positive] * np.sqrt(values[positive])
        test_features = (cross @ vectors[:, positive]) / np.sqrt(values[positive])
        np.testing.assert_allclose(repaired, train_features @ train_features.T)
        np.testing.assert_allclose(repaired_cross, test_features @ train_features.T)
        self.assertLess(diagnostic["minimum_raw_eigenvalue"], 0)
        self.assertFalse(np.allclose(cross, repaired_cross))
        self.assertFalse(np.allclose(np.diag(repaired), 1))
        with self.assertRaises(ValueError):
            kernels.repair_kernels(raw, np.ones((1, 2)))

    def test_calibration_gate_rejects_failed_self_overlap(self):
        first = Statevector.from_instruction(kernels.feature_circuit(TRAINING[0]))
        second = Statevector.from_instruction(kernels.feature_circuit(TRAINING[1]))
        hits = round(abs(np.vdot(second.data, first.data)) ** 2 * kernels.SETTINGS.shots_per_entry)
        count_sets = [{"0000": 256}, {"0000": hits, "1111": 256 - hits}]
        self.assertTrue(kernels.calibration_checks(count_sets)["passed"])
        failed = copy.deepcopy(count_sets)
        failed[0] = {"0000": 255, "1111": 1}
        self.assertEqual(kernels.calibration_checks(failed)["status"], "calibration_failed")
        with self.assertRaises(ValueError):
            kernels.calibration_checks(count_sets[:1])

    def test_numpy_integer_counts_produce_json_safe_calibration(self):
        counts = [{"0000": np.int64(256)}, {"1111": np.int64(256)}]
        record = kernels.calibration_checks(counts)
        self.assertIs(type(record["cases"][0]["passed"]), bool)
        self.assertIs(type(record["cases"][1]["passed"]), bool)
        self.assertEqual(json.loads(json.dumps(record))["status"], record["status"])

    def test_bad_angles_row_overlap_and_protocol_mutation_fail(self):
        for angles in ([], [[0, 1]], [[math.nan] * 4], [[-0.1] * 4]):
            with self.assertRaises(ValueError):
                kernels.validate_angles(angles)
        with self.assertRaises(ValueError):
            kernels.ordered_pairs(TRAIN_IDS, [TRAIN_IDS[0]])
        payload = kernels.PROTOCOL_FILE.read_bytes() + b"\n"
        with patch.object(type(kernels.PROTOCOL_FILE), "read_bytes", return_value=payload):
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                kernels.load_protocol()


class CalibrationRunnerTests(unittest.TestCase):
    def fixture_counts(self):
        first = Statevector.from_instruction(kernels.feature_circuit(TRAINING[0]))
        second = Statevector.from_instruction(kernels.feature_circuit(TRAINING[1]))
        hits = round(abs(np.vdot(second.data, first.data)) ** 2 * kernels.SETTINGS.shots_per_entry)
        return [{"0000": 256}, {"0000": hits, "1111": 256 - hits}]

    def fake_job(self, counts):
        class Job:
            def result(self):
                return self

            def get_counts(self, index):
                return counts[index]
        return Job()

    def test_runner_calls_backend_once_and_blocks_repeat_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "attempt"
            with patch.object(calibrate.BasicSimulator, "run", return_value=self.fake_job(self.fixture_counts())) as run:
                report = calibrate.run_calibration(output)
                self.assertTrue(report["passed"])
                self.assertEqual(run.call_count, 1)
                self.assertEqual(len(run.call_args.args[0]), 2)
                self.assertEqual(run.call_args.kwargs["shots"], 256)
                self.assertEqual(run.call_args.kwargs["seed_simulator"], kernels.SETTINGS.calibration_seed)
                with self.assertRaises(FileExistsError):
                    calibrate.run_calibration(output)
                self.assertEqual(run.call_count, 1)
            self.assertFalse(report["dataset_used"])
            self.assertFalse(report["full_sampling_authorized"])
            self.assertEqual(json.loads((output / "counts.json").read_text())["counts"], self.fixture_counts())
            self.assertEqual(json.loads((output / "implementation-configuration.json").read_text())["settings"]["alpha"], 2.0)

    def test_failed_calibration_retains_counts_without_retry(self):
        counts = self.fixture_counts()
        counts[0] = {"0000": 255, "1111": 1}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed"
            with patch.object(calibrate.BasicSimulator, "run", return_value=self.fake_job(counts)) as run:
                report = calibrate.run_calibration(output)
                self.assertEqual(run.call_count, 1)
            self.assertEqual(report["status"], "calibration_failed")
            self.assertEqual(json.loads((output / "counts.json").read_text())["counts"], counts)
            self.assertFalse(json.loads((output / "report.json").read_text())["passed"])

    def test_sampler_exception_is_recorded_and_propagated(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "exception"
            with patch.object(calibrate.BasicSimulator, "run", side_effect=RuntimeError("synthetic sampler failure")) as run:
                with self.assertRaisesRegex(RuntimeError, "synthetic sampler failure"):
                    calibrate.run_calibration(output)
                self.assertEqual(run.call_count, 1)
            failure = json.loads((output / "failure.json").read_text())
            self.assertEqual(failure["phase"], "sampling")
            self.assertEqual(failure["backend_calls_attempted"], 1)
            self.assertFalse((output / "report.json").exists())


class SamplingCaptureTests(unittest.TestCase):
    def setUp(self):
        self.angles = {"training_angles": TRAINING, "testing_angles": TESTING,
                       "train_row_ids": TRAIN_IDS, "test_row_ids": TEST_IDS}
        self.circuits, self.pairs = kernels.build_overlap_batch(TRAINING, TESTING, TRAIN_IDS, TEST_IDS)
        self.counts = [{"0000": 64, "0001": 192}, {"0000": 128, "0001": 128}, {"0001": 256}]

    def result(self, circuits, counts=None):
        counts = self.counts if counts is None else counts
        return Result.from_dict({
            "backend_name": "basic_simulator", "backend_version": "test", "job_id": "synthetic",
            "success": True,
            "results": [{
                "success": True, "shots": 256,
                "header": {"name": circuit.name, "metadata": circuit.metadata, "memory_slots": 4},
                "data": {"counts": {hex(int(word, 2)): count for word, count in row.items()}},
            } for circuit, row in zip(circuits, counts)],
        })

    def job(self, result):
        class Job:
            def result(self):
                return result
        return Job()

    def test_result_identity_shots_and_complete_coverage_are_required(self):
        result = self.result(self.circuits)
        self.assertEqual(sample_kernels.aligned_counts(result, self.circuits), self.counts)
        missing = copy.deepcopy(result)
        missing.results.pop()
        swapped = copy.deepcopy(result)
        swapped.results.reverse()
        failed = copy.deepcopy(result)
        failed.results[0].success = False
        wrong_shots = copy.deepcopy(result)
        wrong_shots.results[0].shots = 255
        wrong_metadata = copy.deepcopy(result)
        wrong_metadata.results[0].header["metadata"] = {}
        for mutated in (missing, swapped, failed, wrong_shots, wrong_metadata):
            with self.assertRaises(ValueError):
                sample_kernels.aligned_counts(mutated, self.circuits)

    def test_capture_calls_sampler_once_and_retains_all_kernel_forms(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(sample_kernels.BasicSimulator, "run", side_effect=lambda circuits, **options: self.job(self.result(circuits))) as run:
                report = sample_kernels.capture_batch(self.angles, self.pairs, output)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.kwargs, {"shots": 256, "seed_simulator": kernels.SETTINGS.matrix_seed})
            self.assertEqual(report["total_shots"], 768)
            self.assertEqual(report["classifier_fits"], 0)
            self.assertFalse(report["comparison_complete"])
            with np.load(output / "sampled-kernels.npz", allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays["training"], [[1, 0.25], [0.25, 1]])
                np.testing.assert_array_equal(arrays["test_training"], [[0.5, 0]])
            self.assertTrue((output / "ideal-kernels.npz").exists())
            self.assertTrue((output / "repaired-kernels.npz").exists())
            self.assertEqual(json.loads((output / "counts.json").read_text())["counts"], self.counts)

    def test_incomplete_counts_retain_raw_capture_without_completion(self):
        counts = copy.deepcopy(self.counts)
        counts[1]["0000"] -= 1
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(sample_kernels.BasicSimulator, "run", side_effect=lambda circuits, **options: self.job(self.result(circuits, counts))) as run:
                with self.assertRaisesRegex(ValueError, "shot budget"):
                    sample_kernels.capture_batch(self.angles, self.pairs, output)
            self.assertEqual(run.call_count, 1)
            self.assertTrue((output / "raw-result.json").exists())
            self.assertTrue((output / "counts.json").exists())
            self.assertFalse((output / "repaired-kernels.npz").exists())
            failure = json.loads((output / "failure.json").read_text())
            self.assertEqual(failure["phase"], "capture_validation")

    def test_changed_pair_manifest_fails_before_sampler(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sample_kernels.BasicSimulator, "run") as run:
                with self.assertRaisesRegex(ValueError, "pair manifest differs"):
                    sample_kernels.capture_batch(self.angles, list(reversed(self.pairs)), Path(directory))
            run.assert_not_called()

    def test_sampler_exception_is_retained_and_propagated(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(sample_kernels.BasicSimulator, "run", side_effect=RuntimeError("synthetic sampling failure")) as run:
                with self.assertRaisesRegex(RuntimeError, "synthetic sampling failure"):
                    sample_kernels.capture_batch(self.angles, self.pairs, output)
            self.assertEqual(run.call_count, 1)
            failure = json.loads((output / "failure.json").read_text())
            self.assertEqual(failure["backend_calls_attempted"], 1)
            self.assertEqual(failure["phase"], "sampling")

    def test_zero_reference_norm_is_explicitly_undefined(self):
        record = sample_kernels.kernel_error(np.ones((1, 1)), np.zeros((1, 1)))
        self.assertIsNone(record["relative_frobenius_error"])
        self.assertTrue(record["relative_error_undefined"])
        self.assertEqual(record["absolute_frobenius_error"], 1)

    def test_unapproved_run_is_refused_before_input_or_sampler_access(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unapproved"
            with patch.object(sample_kernels, "sampling_plan") as plan, patch.object(sample_kernels.BasicSimulator, "run") as run:
                with self.assertRaisesRegex(ValueError, "explicit launch approval"):
                    sample_kernels.run_sampling(Path(directory) / "absent-plan.json", output)
            plan.assert_not_called()
            run.assert_not_called()
            self.assertFalse(output.exists())

    def test_real_plan_json_roundtrip_without_sampling(self):
        with patch.object(sample_kernels.BasicSimulator, "run", side_effect=AssertionError("Plan must not sample")) as run:
            plan = sample_kernels.sampling_plan()
        self.assertEqual(plan, json.loads(json.dumps(plan, allow_nan=False)))
        self.assertEqual(plan["budget"]["circuits"], 1008)
        self.assertEqual(plan["budget"]["total_shots"], 258048)
        self.assertFalse(plan["full_sampling_authorized"])
        run.assert_not_called()

    def fixture_plan(self):
        return {"prepared_directory": str(sample_kernels.PREPARED_DIRECTORY),
                "budget": {"circuits": 3, "total_shots": 768, "backend_calls": 1}}

    def test_sealed_runner_completes_once_and_refuses_repeat(self):
        plan = self.fixture_plan()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capture"
            plan_file = Path(directory) / "plan.json"
            sample_kernels.write_record(plan_file, plan)
            with patch.object(sample_kernels, "sampling_plan", return_value=plan), patch.object(sample_kernels, "read_inputs", return_value=(self.angles, self.pairs)), patch.object(sample_kernels.BasicSimulator, "run", side_effect=lambda circuits, **options: self.job(self.result(circuits))) as run:
                report = sample_kernels.run_sampling(plan_file, output, approved=True)
                with self.assertRaises(FileExistsError):
                    sample_kernels.run_sampling(plan_file, output, approved=True)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(report["status"], "kernel_capture_complete")
            self.assertFalse(report["comparison_complete"])
            receipt = json.loads((output / "complete.json").read_text())
            self.assertEqual(set(receipt["files"]), set(sample_kernels.CAPTURE_ARTIFACTS))
            for name, expected in receipt["files"].items():
                self.assertEqual(sample_kernels.file_record(output / name), expected)

    def test_changed_bindings_after_capture_block_completion(self):
        plan = self.fixture_plan()
        changed = {**plan, "unexpected_change": True}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "changed"
            plan_file = Path(directory) / "plan.json"
            sample_kernels.write_record(plan_file, plan)
            with patch.object(sample_kernels, "sampling_plan", side_effect=[plan, changed]), patch.object(sample_kernels, "read_inputs", return_value=(self.angles, self.pairs)), patch.object(sample_kernels.BasicSimulator, "run", side_effect=lambda circuits, **options: self.job(self.result(circuits))) as run:
                with self.assertRaisesRegex(ValueError, "bindings changed"):
                    sample_kernels.run_sampling(plan_file, output, approved=True)
            self.assertEqual(run.call_count, 1)
            self.assertFalse((output / "complete.json").exists())
            self.assertTrue((output / "raw-result.json").exists())
            self.assertEqual(json.loads((output / "failure.json").read_text())["status"], "integrity_failed")


if __name__ == "__main__":
    unittest.main()