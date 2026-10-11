import argparse
import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.svm import SVC

from prepare_data import ROOT, admit_dataset, configuration_record, file_record, read_protocol, verify_preparation, write_record


PREPARED_DIRECTORY = ROOT / "preparation-v1"
SAMPLING_DIRECTORY = ROOT / "sampling-v1"
SAMPLING_ARTIFACTS = (
    "execution-plan.json", "started.json", "pairs.json", "ideal-kernels.npz",
    "raw-result.json", "counts.json", "sampled-kernels.npz", "repaired-kernels.npz",
    "diagnostics.json", "report.json",
)


@dataclass(frozen=True)
class ClassifierSettings:
    regularization: float = 1.0
    probability: bool = False
    tolerance: float = 0.001
    iteration_cap: int = 10000
    class_weight: None = None
    shrinking: bool = True
    cache_mb: float = 128.0
    rbf_gamma: str = "scale"


SETTINGS = ClassifierSettings()


def shared_parameters(settings: ClassifierSettings = SETTINGS) -> dict:
    return {
        "C": settings.regularization, "probability": settings.probability,
        "tol": settings.tolerance, "max_iter": settings.iteration_cap,
        "class_weight": settings.class_weight, "shrinking": settings.shrinking,
        "cache_size": settings.cache_mb,
    }


def make_classifier(row: dict) -> SVC:
    return SVC(kernel=row["kernel"], gamma=row.get("gamma", SETTINGS.rbf_gamma), **shared_parameters())


def prediction_metrics(labels, predictions, scores) -> dict:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or not len(labels) or predictions.shape != labels.shape or scores.shape != labels.shape:
        raise ValueError("Prediction, label and decision-score coverage differs")
    if set(np.unique(labels)) != {0, 1} or not np.isin(predictions, [0, 1]).all():
        raise ValueError("Expected the fixed binary classes, with both present in test labels")
    if not np.isfinite(scores).all():
        raise ValueError("Decision scores must be finite")
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "roc_auc_from_decision_function": float(roc_auc_score(labels, scores)),
        "confusion_matrix_0_1": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def row_inputs(row: dict, angles: dict, matrices: dict) -> tuple[np.ndarray, np.ndarray]:
    if row["id"] == "quantum_ideal":
        return matrices["ideal_training"], matrices["ideal_test_training"]
    if row["id"] == "quantum_shot":
        return matrices["sampled_training"], matrices["sampled_test_training"]
    return np.asarray(angles["training_angles"], dtype=float), np.asarray(angles["testing_angles"], dtype=float)


def evaluate_rows(protocol: dict, angles: dict, matrices: dict, train_labels, test_labels, output: Path) -> dict:
    rows = {}
    output.mkdir()
    for row in protocol["classifiers"]["rows"]:
        phase = "input_validation"
        calls = {"fit": 0, "predict": 0, "decision_function": 0}
        warning_records = []
        try:
            training, testing = row_inputs(row, angles, matrices)
            if training.ndim != 2 or testing.ndim != 2 or training.shape[1] != testing.shape[1]:
                raise ValueError("Classifier input dimensions differ")
            if len(training) != len(train_labels) or len(testing) != len(test_labels) or len(testing) != len(angles["test_row_ids"]):
                raise ValueError("Classifier inputs do not cover the declared labels and row IDs")
            if not np.isfinite(training).all() or not np.isfinite(testing).all():
                raise ValueError("Classifier inputs must be finite")
            model = make_classifier(row)
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as warning_records:
                warnings.simplefilter("error", ConvergenceWarning)
                phase = "fit"
                calls["fit"] += 1
                model.fit(training, train_labels)
                if model.classes_.tolist() != [0, 1]:
                    raise ValueError("Classifier decision-score class ordering changed")
                phase = "predict"
                calls["predict"] += 1
                predictions = model.predict(testing)
                phase = "decision_function"
                calls["decision_function"] += 1
                scores = model.decision_function(testing)
            elapsed = time.perf_counter() - started
            phase = "metrics"
            metrics = prediction_metrics(test_labels, predictions, scores)
            record = {
                "status": "row_complete", "id": row["id"], "calls": calls,
                "parameters": model.get_params(deep=False), "classes": model.classes_.tolist(),
                "warnings": [{"category": item.category.__name__, "message": str(item.message)} for item in warning_records],
                "training_shape": list(training.shape), "test_shape": list(testing.shape),
                "fit_and_inference_seconds": elapsed, "metrics": metrics,
                "records": [{"row_id": int(row_id), "label": int(label), "prediction": int(prediction),
                             "decision_score": float(score)}
                            for row_id, label, prediction, score in zip(angles["test_row_ids"], test_labels, predictions, scores)],
            }
            if row["id"] == "rbf":
                record["fitted_rbf_gamma"] = float(model._gamma)
            write_record(output / f"{row['id']}.json", record)
            rows[row["id"]] = record
        except Exception as error:
            write_record(output / f"{row['id']}-failure.json", {
                "status": "row_failed", "id": row["id"], "phase": phase,
                "calls_attempted": calls, "error_type": type(error).__name__, "error": str(error),
                "warnings": [{"category": item.category.__name__, "message": str(item.message)} for item in warning_records],
            })
            raise
    return rows


def evaluation_plan() -> dict:
    protocol = read_protocol()
    runtime = configuration_record()
    if shared_parameters() != protocol["classifiers"]["shared"]:
        raise ValueError("Live classifier parameters differ from the fixed contract")
    for row in protocol["classifiers"]["rows"]:
        if row["id"] == "rbf" and row["gamma"] != SETTINGS.rbf_gamma:
            raise ValueError("RBF gamma rule differs from the fixed contract")
    verify_preparation(PREPARED_DIRECTORY)
    receipt = json.loads((SAMPLING_DIRECTORY / "complete.json").read_text())
    if receipt["status"] != "kernel_capture_complete" or receipt["comparison_complete"] is not False:
        raise ValueError("Expected a completed kernel-only capture")
    if set(receipt["files"]) != set(SAMPLING_ARTIFACTS):
        raise ValueError("Sampling receipt coverage changed")
    for name, expected in receipt["files"].items():
        if file_record(SAMPLING_DIRECTORY / name) != expected:
            raise ValueError(f"Sampling artifact changed: {name}")
    sampling_plan_path = ROOT / "sampling-plan-v1.json"
    sampling_plan = json.loads(sampling_plan_path.read_text())
    if json.loads((SAMPLING_DIRECTORY / "execution-plan.json").read_text()) != sampling_plan:
        raise ValueError("Sampling execution plan changed")
    sampling_report = json.loads((SAMPLING_DIRECTORY / "report.json").read_text())
    if sampling_report["execution_plan_sha256"] != file_record(sampling_plan_path)["sha256"]:
        raise ValueError("Sampling plan digest changed")
    quantum = protocol["quantum"]
    if (sampling_report["circuits"], sampling_report["total_shots"], sampling_report["backend_calls"]) != (
        quantum["matrix_circuits"], quantum["matrix_shots"], quantum["matrix_backend_calls"],
    ):
        raise ValueError("Sampling coverage differs from the protocol")
    if sampling_report["classifier_fits"] != 0 or sampling_report["comparison_complete"] is not False:
        raise ValueError("The input capture already contains classifier evaluation")
    for name, expected in sampling_plan["implementation"]["source_sha256"].items():
        if file_record(ROOT / name)["sha256"] != expected:
            raise ValueError(f"Sampling source changed: {name}")
    runtime["source_sha256"][Path(__file__).name] = file_record(Path(__file__))["sha256"]
    return {
        "status": "evaluation_plan_prepared", "evaluation_authorized": False,
        "runtime": runtime, "classifier_settings": asdict(SETTINGS),
        "classifier_parameters": [{"id": row["id"], "parameters": make_classifier(row).get_params(deep=False)}
                                  for row in protocol["classifiers"]["rows"]],
        "sampling_plan": file_record(sampling_plan_path),
        "sampling_receipt": file_record(SAMPLING_DIRECTORY / "complete.json"),
        "sampling_files": receipt["files"],
        "preparation_receipt": file_record(PREPARED_DIRECTORY / "complete.json"),
        "rows": protocol["classifiers"]["rows"], "metrics": protocol["metrics"],
        "train_row_ids": protocol["partition"]["train_row_ids"],
        "test_row_ids": protocol["partition"]["test_row_ids"],
        "calls_per_row": {"fit": 1, "predict": 1, "decision_function": 1},
        "quantum_jobs": 0, "preprocessing_fits": 0,
        "resource_cost": "Four local 32-record SVC fits; no service or hardware charges; expected to take seconds",
    }


def load_evaluation_inputs() -> tuple[dict, dict, np.ndarray, np.ndarray]:
    protocol = read_protocol()
    source = admit_dataset(protocol)
    partition = protocol["partition"]
    labels = (source.target == 0).astype(np.uint8)
    train_labels = labels[partition["train_row_ids"]]
    test_labels = labels[partition["test_row_ids"]]
    angles = json.loads((PREPARED_DIRECTORY / "angles.json").read_text())
    if angles["train_row_ids"] != partition["train_row_ids"] or angles["test_row_ids"] != partition["test_row_ids"]:
        raise ValueError("Classifier input row order differs from the protocol")
    with np.load(SAMPLING_DIRECTORY / "repaired-kernels.npz", allow_pickle=False) as arrays:
        matrices = {name: arrays[name] for name in arrays.files}
    expected_shapes = {
        "ideal_training": (len(train_labels), len(train_labels)),
        "sampled_training": (len(train_labels), len(train_labels)),
        "ideal_test_training": (len(test_labels), len(train_labels)),
        "sampled_test_training": (len(test_labels), len(train_labels)),
    }
    if set(matrices) != set(expected_shapes):
        raise ValueError("Repaired kernel coverage differs")
    for name, shape in expected_shapes.items():
        if matrices[name].shape != shape or not np.isfinite(matrices[name]).all():
            raise ValueError(f"Invalid repaired kernel: {name}")
    for name, rows in (("training_angles", len(train_labels)), ("testing_angles", len(test_labels))):
        values = np.asarray(angles[name], dtype=float)
        if values.shape != (rows, protocol["quantum"]["feature_dimension"]) or not np.isfinite(values).all():
            raise ValueError("Invalid shared angle features")
    return angles, matrices, train_labels, test_labels


def run_evaluation(plan_file: Path, output: Path, approved: bool = False) -> dict:
    if approved is not True:
        raise ValueError("Real test-set evaluation requires explicit approval")
    output.mkdir(parents=True)
    phase = "input_admission"
    try:
        plan = json.loads(plan_file.read_text())
        if plan != evaluation_plan():
            raise ValueError("Evaluation plan or frozen inputs changed")
        plan_record = file_record(plan_file)
        write_record(output / "execution-plan.json", plan)
        write_record(output / "started.json", {
            "status": "evaluation_started", "explicit_evaluation_approval": True,
            "plan_sha256": plan_record["sha256"], "quantum_jobs": 0, "preprocessing_fits": 0,
        })
        protocol = read_protocol()
        angles, matrices, train_labels, test_labels = load_evaluation_inputs()
        phase = "row_evaluation"
        rows = evaluate_rows(protocol, angles, matrices, train_labels, test_labels, output / "rows")
        if set(rows) != {row["id"] for row in protocol["classifiers"]["rows"]}:
            raise ValueError("Classifier comparison coverage differs")
        phase = "integrity"
        if evaluation_plan() != plan or file_record(plan_file) != plan_record:
            raise ValueError("Evaluation bindings changed during fitting or scoring")
        report = {
            "status": "classifier_comparison_complete", "comparison_complete": True,
            "classifiers_fitted": len(rows), "shared_held_out_records": len(test_labels),
            "calls": {name: sum(row["calls"][name] for row in rows.values()) for name in ("fit", "predict", "decision_function")},
            "quantum_jobs": 0, "preprocessing_fits": 0,
            "rows": {name: {"metrics": row["metrics"], "fit_and_inference_seconds": row["fit_and_inference_seconds"],
                            "evidence": {"path": f"rows/{name}.json", **file_record(output / "rows" / f"{name}.json")}}
                     for name, row in rows.items()},
            "shot_minus_ideal_balanced_accuracy": rows["quantum_shot"]["metrics"]["balanced_accuracy"] - rows["quantum_ideal"]["metrics"]["balanced_accuracy"],
            "execution_plan_sha256": plan_record["sha256"], "frozen_inputs_unchanged": True,
            "confidence": "Single-partition, single-seed point estimates; not a statistically established ranking",
            "quantum_advantage_claim": False, "clinical_validation": False,
        }
        write_record(output / "report.json", report)
        names = ["execution-plan.json", "started.json", "report.json"] + [f"rows/{name}.json" for name in rows]
        write_record(output / "complete.json", {
            "status": "classifier_comparison_complete", "comparison_complete": True,
            "files": {name: file_record(output / name) for name in names},
        })
        return report
    except Exception as error:
        write_record(output / "failure.json", {
            "status": "row_failed" if phase == "row_evaluation" else "integrity_failed",
            "phase": phase, "error_type": type(error).__name__, "error": str(error),
            "comparison_complete": False,
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approve-test-evaluation", action="store_true")
    arguments = parser.parse_args()
    if arguments.action == "plan":
        plan = evaluation_plan()
        write_record(arguments.output, plan)
        print(json.dumps({"status": plan["status"], "output": str(arguments.output), "classifiers": len(plan["rows"]),
                          "test_records": len(plan["test_row_ids"]), "calls_per_row": plan["calls_per_row"],
                          "evaluation_authorized": False, "quantum_jobs": 0, "preprocessing_fits": 0}, indent=2))
    else:
        if arguments.plan is None:
            parser.error("Evaluation requires --plan and explicit test-evaluation approval")
        print(json.dumps(run_evaluation(arguments.plan, arguments.output, arguments.approve_test_evaluation), indent=2))


if __name__ == "__main__":
    main()