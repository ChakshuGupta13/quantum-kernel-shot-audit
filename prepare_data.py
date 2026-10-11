import argparse
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


ROOT = Path(__file__).resolve().parent
PROTOCOL_SHA256 = "113f4c42b865406dc24b82f79092c81097ccfd6e13978a764066a24187f27a0f"
CALIBRATION_DIRECTORY = ROOT / "calibration-v1"
CALIBRATION_ARTIFACTS = ("implementation-configuration.json", "started.json", "counts.json", "report.json")
PREPARATION_ARTIFACTS = ("started.json", "implementation-configuration.json", "angles.json", "preprocessing.json", "pairs.json", "report.json")
PAIR_WORKER = """import json, sys
from kernels import ordered_pairs
partition = json.load(sys.stdin)
print(json.dumps(ordered_pairs(partition['train_row_ids'], partition['test_row_ids'])))
"""


@dataclass(frozen=True)
class PreparationSettings:
    with_mean: bool = True
    with_std: bool = True
    n_components: int = 4
    svd_solver: str = "full"
    whiten: bool = False
    angle_lower: float = 0.0
    angle_upper: float = math.pi
    clip: bool = True


SETTINGS = PreparationSettings()


def read_protocol() -> dict:
    payload = (ROOT / "configuration.json").read_bytes()
    if sha256(payload).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen protocol bytes changed")
    protocol = json.loads(payload)
    steps = protocol["preprocessing"]["steps"]
    if [step["class"] for step in steps] != ["StandardScaler", "PCA", "MinMaxScaler"]:
        raise ValueError("Preprocessing step order changed")
    declared = PreparationSettings(
        with_mean=steps[0]["with_mean"], with_std=steps[0]["with_std"],
        n_components=steps[1]["n_components"], svd_solver=steps[1]["svd_solver"],
        whiten=steps[1]["whiten"], angle_lower=steps[2]["feature_range"][0],
        angle_upper=steps[2]["feature_range"][1], clip=steps[2]["clip"],
    )
    if declared != SETTINGS or SETTINGS.n_components != protocol["quantum"]["feature_dimension"]:
        raise ValueError("Live preprocessing settings differ from the protocol")
    return protocol


def dataset_digest(source) -> str:
    payload = source.data.astype("<f8").tobytes(order="C")
    payload += source.target.astype("u1").tobytes()
    payload += json.dumps(source.feature_names.tolist(), separators=(",", ":")).encode()
    return sha256(payload).hexdigest()


def admit_dataset(protocol: dict):
    source = load_breast_cancer()
    dataset = protocol["dataset"]
    if source.data.shape != (dataset["rows"], dataset["features"]) or not np.isfinite(source.data).all():
        raise ValueError("Dataset dimensions or finite values differ")
    if source.target_names.tolist() != dataset["raw_target_names"] or dataset_digest(source) != dataset["sha256"]:
        raise ValueError("Dataset identity differs from the frozen source")
    labels = (source.target == 0).astype(np.uint8)
    partition = protocol["partition"]
    admitted, omitted = train_test_split(
        np.arange(len(labels)), train_size=partition["admitted_rows"], stratify=labels,
        random_state=partition["admission_seed"],
    )
    training, testing = train_test_split(
        admitted, train_size=partition["train_rows"], test_size=partition["test_rows"],
        stratify=labels[admitted], random_state=partition["split_seed"],
    )
    if training.tolist() != partition["train_row_ids"] or testing.tolist() != partition["test_row_ids"]:
        raise ValueError("Regenerated partition differs from frozen row IDs")
    if len(omitted) != partition["omitted_rows"] or set(training) & set(testing):
        raise ValueError("Partition omission or overlap differs")
    for indices, name in ((training, "train_class_counts_0_1"), (testing, "test_class_counts_0_1")):
        if np.bincount(labels[indices]).tolist() != partition[name]:
            raise ValueError("Partition class counts differ")
    return source


def make_preprocessing(settings: PreparationSettings = SETTINGS) -> Pipeline:
    return Pipeline([
        ("standard", StandardScaler(with_mean=settings.with_mean, with_std=settings.with_std)),
        ("pca", PCA(n_components=settings.n_components, svd_solver=settings.svd_solver, whiten=settings.whiten)),
        ("angles", MinMaxScaler(feature_range=(settings.angle_lower, settings.angle_upper), clip=settings.clip)),
    ])


def fitted_values(estimator, names) -> dict:
    return {name: np.asarray(getattr(estimator, name)).tolist() for name in names}


def fit_angles(training, testing, settings: PreparationSettings = SETTINGS) -> tuple[np.ndarray, np.ndarray, dict]:
    training = np.asarray(training, dtype=float)
    testing = np.asarray(testing, dtype=float)
    if training.ndim != 2 or testing.ndim != 2 or not len(training) or not len(testing):
        raise ValueError("Expected nonempty two-dimensional input arrays")
    if training.shape[1] != testing.shape[1] or not np.isfinite(training).all() or not np.isfinite(testing).all():
        raise ValueError("Feature columns or finite values differ")
    pipeline = make_preprocessing(settings)
    pipeline.fit(training)
    training_angles = pipeline.transform(training)
    testing_projection = pipeline[:-1].transform(testing)
    angle_scaler = pipeline.named_steps["angles"]
    unbounded = testing_projection * angle_scaler.scale_ + angle_scaler.min_
    testing_angles = angle_scaler.transform(testing_projection)
    clipped = np.argwhere((unbounded < settings.angle_lower) | (unbounded > settings.angle_upper))
    audit = {
        "settings": asdict(settings),
        "training_rows_fitted": len(training),
        "pipeline_fit_calls": 1,
        "standard": fitted_values(pipeline.named_steps["standard"], ("mean_", "var_", "scale_", "n_samples_seen_")),
        "pca": fitted_values(pipeline.named_steps["pca"], (
            "mean_", "components_", "explained_variance_", "explained_variance_ratio_",
            "singular_values_", "noise_variance_", "n_samples_", "n_components_",
        )),
        "angles": fitted_values(angle_scaler, ("data_min_", "data_max_", "data_range_", "scale_", "min_", "n_samples_seen_")),
        "clipped_test_coordinates": [
            {"test_position": int(row), "component": int(column),
             "unclipped_angle": float(unbounded[row, column]), "angle": float(testing_angles[row, column])}
            for row, column in clipped
        ],
    }
    return training_angles, testing_angles, audit


def pair_manifest(partition: dict) -> list[dict]:
    result = subprocess.run(
        [sys.executable, "-c", PAIR_WORKER], cwd=ROOT,
        input=json.dumps(partition), capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def file_record(source: Path) -> dict:
    payload = source.read_bytes()
    return {"bytes": len(payload), "sha256": sha256(payload).hexdigest()}


def configuration_record() -> dict:
    protocol = read_protocol()
    packages = {name: version(name) for name in protocol["versions"] if name != "python"}
    expected = {name: value for name, value in protocol["versions"].items() if name != "python"}
    if packages != expected or platform.python_version() != protocol["versions"]["python"]:
        raise ValueError("Preparation runtime differs from the frozen protocol")
    calibration_files = {name: file_record(CALIBRATION_DIRECTORY / name) for name in CALIBRATION_ARTIFACTS}
    calibration = json.loads((CALIBRATION_DIRECTORY / "report.json").read_text())
    configuration = json.loads((CALIBRATION_DIRECTORY / "implementation-configuration.json").read_text())
    if configuration["protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("Calibration was bound to a different protocol")
    if calibration["configuration_sha256"] != calibration_files["implementation-configuration.json"]["sha256"]:
        raise ValueError("Calibration configuration changed")
    if calibration["counts_sha256"] != calibration_files["counts.json"]["sha256"]:
        raise ValueError("Calibration counts changed")
    for name, expected_hash in configuration["source_sha256"].items():
        if file_record(ROOT / name)["sha256"] != expected_hash:
            raise ValueError(f"Calibration source changed: {name}")
    if calibration["status"] != "calibration_passed" or calibration["passed"] is not True:
        raise ValueError("Calibration did not pass")
    if calibration["circuits"] != protocol["calibration"]["circuits"] or calibration["total_shots"] != protocol["calibration"]["shots"]:
        raise ValueError("Calibration coverage differs")
    if calibration["backend_calls"] != protocol["calibration"]["backend_calls"] or calibration["full_sampling_authorized"] is not False:
        raise ValueError("Calibration execution boundary differs")
    return {
        "protocol_sha256": PROTOCOL_SHA256,
        "dataset_sha256": protocol["dataset"]["sha256"],
        "settings": asdict(SETTINGS),
        "source_sha256": {name: file_record(ROOT / name)["sha256"] for name in ("prepare_data.py", "kernels.py", "calibrate.py")},
        "calibration_directory": CALIBRATION_DIRECTORY.name,
        "calibration_files": calibration_files,
        "packages": packages,
        "python": platform.python_version(),
        "parameter_origin": "live PreparationSettings dataclass, checked against the fixed contract",
    }


def write_record(destination: Path, record) -> None:
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def prepare(output: Path) -> dict:
    output.mkdir(parents=True)
    phase = "admission"
    try:
        protocol = read_protocol()
        binding = configuration_record()
        source = admit_dataset(protocol)
        if dataset_digest(source) != binding["dataset_sha256"]:
            raise ValueError("Dataset differs before preparation")
        partition = protocol["partition"]
        write_record(output / "implementation-configuration.json", binding)
        write_record(output / "started.json", {
            "status": "preparation_started", "preprocessing_fit_started": False,
            "classifiers_fitted": False, "full_sampling_authorized": False,
            "train_row_ids": partition["train_row_ids"], "test_row_ids": partition["test_row_ids"],
        })
        phase = "preprocessing"
        started = time.perf_counter()
        training, testing, audit = fit_angles(
            source.data[partition["train_row_ids"]], source.data[partition["test_row_ids"]],
        )
        preprocessing_seconds = time.perf_counter() - started
        for values, rows in ((training, partition["train_rows"]), (testing, partition["test_rows"])):
            if values.shape != (rows, SETTINGS.n_components) or not np.isfinite(values).all():
                raise ValueError("Prepared angle dimensions or finite values differ")
            if np.any(values < SETTINGS.angle_lower) or np.any(values > SETTINGS.angle_upper):
                raise ValueError("Prepared angles exceed the frozen range")
        for coordinate in audit["clipped_test_coordinates"]:
            coordinate["test_row_id"] = partition["test_row_ids"][coordinate["test_position"]]
        write_record(output / "angles.json", {
            "train_row_ids": partition["train_row_ids"], "test_row_ids": partition["test_row_ids"],
            "training_angles": training.tolist(), "testing_angles": testing.tolist(),
            "feature_names": source.feature_names.tolist(), "dataset_sha256": binding["dataset_sha256"],
        })
        write_record(output / "preprocessing.json", audit)
        phase = "pair_manifest"
        started = time.perf_counter()
        pairs = pair_manifest(partition)
        pair_generation_seconds = time.perf_counter() - started
        quantum = protocol["quantum"]
        if len(pairs) != quantum["matrix_circuits"]:
            raise ValueError("Prepared pair coverage differs from the protocol")
        if sum(pair["block"] == "train" for pair in pairs) != quantum["train_off_diagonal_circuits"]:
            raise ValueError("Prepared training-pair coverage differs")
        if sum(pair["block"] == "test_train" for pair in pairs) != quantum["test_train_circuits"]:
            raise ValueError("Prepared test-training coverage differs")
        write_record(output / "pairs.json", pairs)
        phase = "integrity"
        if configuration_record() != binding or dataset_digest(source) != binding["dataset_sha256"]:
            raise ValueError("Preparation bindings changed during the run")
        report = {
            "status": "preparation_complete", "train_rows": len(training), "test_rows": len(testing),
            "components": SETTINGS.n_components, "shared_pipeline_fit_calls": 1,
            "classifiers_fitted": False, "test_predictions_generated": False,
            "quantum_jobs": 0, "kernels_evaluated": False, "full_sampling_authorized": False,
            "pair_count": len(pairs), "clipped_test_coordinates": len(audit["clipped_test_coordinates"]),
            "preprocessing_seconds": preprocessing_seconds, "pair_generation_seconds": pair_generation_seconds,
            "binding_sha256": file_record(output / "implementation-configuration.json")["sha256"],
        }
        write_record(output / "report.json", report)
        write_record(output / "complete.json", {
            "status": "preparation_complete", "full_sampling_authorized": False,
            "files": {name: file_record(output / name) for name in PREPARATION_ARTIFACTS},
        })
        return report
    except Exception as error:
        write_record(output / "failure.json", {
            "status": "preparation_failed", "phase": phase,
            "error_type": type(error).__name__, "error": str(error), "full_sampling_authorized": False,
        })
        raise


def verify_preparation(output: Path) -> dict:
    receipt = json.loads((output / "complete.json").read_text())
    if receipt["status"] != "preparation_complete" or receipt["full_sampling_authorized"] is not False:
        raise ValueError("Preparation receipt is not an unapproved completed bundle")
    if set(receipt["files"]) != set(PREPARATION_ARTIFACTS):
        raise ValueError("Preparation receipt coverage changed")
    for name, expected in receipt["files"].items():
        if file_record(output / name) != expected:
            raise ValueError(f"Prepared artifact changed: {name}")
    binding = json.loads((output / "implementation-configuration.json").read_text())
    if binding != configuration_record():
        raise ValueError("Preparation source, protocol, calibration or runtime changed")
    protocol = read_protocol()
    partition = protocol["partition"]
    angles = json.loads((output / "angles.json").read_text())
    if angles["train_row_ids"] != partition["train_row_ids"] or angles["test_row_ids"] != partition["test_row_ids"]:
        raise ValueError("Prepared row IDs differ from the protocol")
    if angles["dataset_sha256"] != protocol["dataset"]["sha256"]:
        raise ValueError("Prepared dataset identity differs")
    for name, rows in (("training_angles", partition["train_rows"]), ("testing_angles", partition["test_rows"])):
        values = np.asarray(angles[name], dtype=float)
        if values.shape != (rows, SETTINGS.n_components) or not np.isfinite(values).all():
            raise ValueError("Prepared angles have invalid shape or values")
        if np.any(values < SETTINGS.angle_lower) or np.any(values > SETTINGS.angle_upper):
            raise ValueError("Prepared angles exceed the frozen range")
    pairs = json.loads((output / "pairs.json").read_text())
    if pairs != pair_manifest(partition):
        raise ValueError("Prepared ordered pairs differ from the kernel builder")
    report = json.loads((output / "report.json").read_text())
    if report["status"] != "preparation_complete" or report["full_sampling_authorized"] is not False:
        raise ValueError("Preparation report status changed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = prepare(arguments.output) if arguments.action == "prepare" else verify_preparation(arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()