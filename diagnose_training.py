import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel

from prepare_data import ROOT, dataset_digest, file_record, read_protocol, write_record


NUMERICAL_TOLERANCE = 1e-10
PAIR_STD_DDOF = 0
INPUT_FILES = (
    "configuration.json", "preparation-v1/complete.json", "preparation-v1/angles.json",
    "preparation-v1/implementation-configuration.json", "sampling-v1/complete.json",
    "sampling-v1/repaired-kernels.npz", "sampling-v1/execution-plan.json",
)


def centered(matrix: np.ndarray) -> np.ndarray:
    return matrix - matrix.mean(axis=0, keepdims=True) - matrix.mean(axis=1, keepdims=True) + matrix.mean()


def pair_statistics(values: np.ndarray) -> dict:
    if not len(values):
        return {"count": 0, "defined": False, "mean": None, "population_std": None}
    return {"count": len(values), "defined": True, "mean": float(values.mean()),
            "population_std": float(values.std(ddof=PAIR_STD_DDOF))}


def kernel_summary(matrix, labels, regularization: float) -> dict:
    matrix = np.asarray(matrix, dtype=float)
    labels = np.asarray(labels)
    if labels.ndim != 1 or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Expected both binary training classes")
    if matrix.shape != (len(labels), len(labels)) or not np.isfinite(matrix).all():
        raise ValueError("Training kernel dimensions or finite values differ")
    if not np.allclose(matrix, matrix.T, rtol=0, atol=NUMERICAL_TOLERANCE):
        raise ValueError("Training kernel is not symmetric")
    first, second = np.triu_indices(len(labels), k=1)
    pair_values = matrix[first, second]
    same = labels[first] == labels[second]
    within = pair_statistics(pair_values[same])
    between = pair_statistics(pair_values[~same])
    centered_kernel = centered(matrix)
    signed_labels = 2 * labels.astype(float) - 1
    centered_labels = signed_labels - signed_labels.mean()
    target = np.outer(centered_labels, centered_labels)
    denominator = float(np.linalg.norm(centered_kernel) * np.linalg.norm(target))
    alignment = float(np.sum(centered_kernel * target) / denominator) if denominator else None
    eigenvalues = np.linalg.eigvalsh(matrix)
    return {
        "within_class": within, "between_class": between,
        "within_benign": pair_statistics(pair_values[same & (labels[first] == 0)]),
        "within_malignant": pair_statistics(pair_values[same & (labels[first] == 1)]),
        "within_minus_between": within["mean"] - between["mean"] if within["defined"] else None,
        "centered_label_alignment": alignment, "alignment_defined": denominator > 0,
        "mean_diagonal": float(np.diag(matrix).mean()),
        "frobenius_norm": float(np.linalg.norm(matrix)),
        "centered_frobenius_norm": float(np.linalg.norm(centered_kernel)),
        "minimum_eigenvalue": float(eigenvalues[0]), "maximum_eigenvalue": float(eigenvalues[-1]),
        "regularization_C": regularization,
        "C_times_maximum_eigenvalue": float(regularization * eigenvalues[-1]),
    }


def classical_kernels(training) -> tuple[dict, dict]:
    training = np.asarray(training, dtype=float)
    if training.ndim != 2 or not len(training) or not training.shape[1] or not np.isfinite(training).all():
        raise ValueError("Expected finite shared training coordinates")
    variance = float(training.var())
    gamma = 1.0 / (training.shape[1] * variance) if variance else 1.0
    return {"linear": linear_kernel(training), "rbf": rbf_kernel(training, gamma=gamma)}, {
        "rbf_gamma": gamma, "training_coordinate_variance": variance,
        "gamma_rule": "SVC scale: 1/(feature_count * training_variance); one when variance is zero",
    }


def input_bindings() -> dict:
    protocol = read_protocol()
    if NUMERICAL_TOLERANCE != protocol["calibration"]["numeric_statevector_tolerance"]:
        raise ValueError("Diagnostic numerical tolerance differs from the fixed contract")
    preparation = json.loads((ROOT / "preparation-v1/complete.json").read_text())
    sampling = json.loads((ROOT / "sampling-v1/complete.json").read_text())
    if preparation["status"] != "preparation_complete" or sampling["status"] != "kernel_capture_complete":
        raise ValueError("Completed training input receipts are required")
    if file_record(ROOT / "preparation-v1/angles.json") != preparation["files"]["angles.json"]:
        raise ValueError("Frozen training coordinates changed")
    if file_record(ROOT / "sampling-v1/repaired-kernels.npz") != sampling["files"]["repaired-kernels.npz"]:
        raise ValueError("Frozen training kernels changed")
    preparation_code = json.loads((ROOT / "preparation-v1/implementation-configuration.json").read_text())
    sampling_code = json.loads((ROOT / "sampling-v1/execution-plan.json").read_text())["implementation"]
    for record in (preparation_code, sampling_code):
        for name, expected in record["source_sha256"].items():
            if file_record(ROOT / name)["sha256"] != expected:
                raise ValueError(f"Frozen input source changed: {name}")
    if protocol["classifiers"]["rows"][-1]["gamma"] != "scale":
        raise ValueError("Classical RBF diagnostic requires the frozen scale rule")
    return {name: file_record(ROOT / name) for name in INPUT_FILES}


def run_diagnostics(output: Path) -> dict:
    output.mkdir(parents=True)
    try:
        protocol = read_protocol()
        inputs = input_bindings()
        source_hash = file_record(Path(__file__))["sha256"]
        source = load_breast_cancer()
        if source.target_names.tolist() != protocol["dataset"]["raw_target_names"] or dataset_digest(source) != protocol["dataset"]["sha256"]:
            raise ValueError("Original dataset identity changed")
        train_ids = protocol["partition"]["train_row_ids"]
        labels = (source.target[train_ids] == 0).astype(np.uint8)
        if np.bincount(labels).tolist() != protocol["partition"]["train_class_counts_0_1"]:
            raise ValueError("Training label counts differ from the protocol")
        prepared = json.loads((ROOT / "preparation-v1/angles.json").read_text())
        if prepared["train_row_ids"] != train_ids:
            raise ValueError("Training coordinate order changed")
        coordinates = np.asarray(prepared["training_angles"], dtype=float)
        if coordinates.shape != (len(labels), protocol["quantum"]["feature_dimension"]):
            raise ValueError("Training coordinate dimensions changed")
        matrices, classical = classical_kernels(coordinates)
        with np.load(ROOT / "sampling-v1/repaired-kernels.npz", allow_pickle=False) as archive:
            matrices["quantum_ideal"] = archive["ideal_training"]
            matrices["quantum_shot"] = archive["sampled_training"]
        summaries = {name: kernel_summary(matrix, labels, protocol["classifiers"]["shared"]["C"])
                     for name, matrix in matrices.items()}
        identity = kernel_summary(np.eye(len(labels)), labels, protocol["classifiers"]["shared"]["C"])
        if input_bindings() != inputs or file_record(Path(__file__))["sha256"] != source_hash:
            raise ValueError("Diagnostic source or input bindings changed")
        report = {
            "status": "training_diagnostics_complete", "training_row_ids": train_ids,
            "training_class_counts_0_1": np.bincount(labels).tolist(), "kernels": summaries,
            "identity_kernel_alignment_reference": identity["centered_label_alignment"],
            "classical_kernel_parameters": classical,
            "definitions": {
                "pairs": "Each unordered off-diagonal pair once; pooled within-class mean is pair-weighted",
                "pair_std_ddof": PAIR_STD_DDOF,
                "alignment": "Frobenius cosine of double-centered kernel and centered signed-label outer product",
                "alignment_reference": "Identity self-similarity has nonzero centered alignment even without off-diagonal class structure",
                "scale": "C times largest eigenvalue is a descriptive scale indicator, not a causal or capacity-match test",
                "numerical_tolerance": NUMERICAL_TOLERANCE,
            },
            "source_sha256": source_hash, "input_files": inputs,
            "dataset_sha256": protocol["dataset"]["sha256"],
            "classifiers_fitted": 0, "preprocessing_refits": 0, "quantum_jobs": 0,
            "test_labels_used": False, "test_predictions_used": False,
            "scope": "Post-evaluation exploratory diagnosis; numeric inputs are training coordinates, training kernels and training labels only",
            "integrity_reads": "Complete source and mixed train/test artifact files are hashed; only training fields enter numerical diagnostics",
        }
        write_record(output / "report.json", report)
        write_record(output / "complete.json", {
            "status": "training_diagnostics_complete", "report": file_record(output / "report.json"),
            "classifier_fits": 0, "quantum_jobs": 0,
        })
        return report
    except Exception as error:
        write_record(output / "failure.json", {"status": "training_diagnostics_failed",
                                               "error_type": type(error).__name__, "error": str(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = run_diagnostics(arguments.output)
    print(json.dumps({"status": report["status"], "training_rows": len(report["training_row_ids"]),
                      "identity_alignment_reference": report["identity_kernel_alignment_reference"],
                      "kernels": report["kernels"], "classical_kernel_parameters": report["classical_kernel_parameters"],
                      "classifier_fits": 0, "quantum_jobs": 0, "test_labels_used": False}, indent=2))


if __name__ == "__main__":
    main()