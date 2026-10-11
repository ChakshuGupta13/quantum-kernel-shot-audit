import argparse
import json
import math
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit.providers.basic_provider import BasicSimulator

from calibrate import write_record
from kernels import SETTINGS, build_overlap_batch, ideal_kernels, implementation_configuration, kernels_from_counts, load_protocol, repair_kernels


ROOT = Path(__file__).resolve().parent
PREPARED_DIRECTORY = ROOT / "preparation-v1"
ESTIMATE_SAFETY_FACTOR = 4.0
ESTIMATE_STARTUP_ALLOWANCE_SECONDS = 10.0
CAPTURE_ARTIFACTS = (
    "execution-plan.json", "started.json", "pairs.json", "ideal-kernels.npz",
    "raw-result.json", "counts.json", "sampled-kernels.npz", "repaired-kernels.npz",
    "diagnostics.json", "report.json",
)


def aligned_counts(result, circuits) -> list[dict]:
    if result.success is not True or len(result.results) != len(circuits):
        raise ValueError("Incomplete or unsuccessful circuit batch")
    counts = []
    for index, (experiment, circuit) in enumerate(zip(result.results, circuits)):
        if experiment.success is not True or experiment.shots != SETTINGS.shots_per_entry:
            raise ValueError(f"Circuit result failed or its shot budget changed: {index}")
        if experiment.header.get("name") != circuit.name or experiment.header.get("metadata") != circuit.metadata:
            raise ValueError(f"Circuit result identity or order changed: {index}")
        counts.append(dict(result.get_counts(index)))
    return counts


def write_arrays(destination: Path, **arrays) -> None:
    with destination.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def kernel_error(observed, reference) -> dict:
    error = float(np.linalg.norm(observed - reference))
    norm = float(np.linalg.norm(reference))
    return {
        "absolute_frobenius_error": error,
        "reference_frobenius_norm": norm,
        "relative_frobenius_error": error / norm if norm else None,
        "relative_error_undefined": norm == 0,
    }


def capture_batch(angles: dict, manifest: list, output: Path) -> dict:
    phase = "construction"
    backend_calls = 0
    timings = {}
    try:
        started = time.perf_counter()
        circuits, pairs = build_overlap_batch(
            angles["training_angles"], angles["testing_angles"],
            angles["train_row_ids"], angles["test_row_ids"],
        )
        if pairs != manifest:
            raise ValueError("Submitted pair manifest differs from the circuit builder")
        timings["construction_seconds"] = time.perf_counter() - started
        phase = "ideal_kernel"
        started = time.perf_counter()
        ideal_training, ideal_cross = ideal_kernels(angles["training_angles"], angles["testing_angles"])
        timings["ideal_kernel_seconds"] = time.perf_counter() - started
        write_arrays(output / "ideal-kernels.npz", training=ideal_training, test_training=ideal_cross)
        phase = "transpilation"
        started = time.perf_counter()
        backend = BasicSimulator()
        executable = transpile(
            circuits, backend, optimization_level=SETTINGS.transpiler_optimization_level,
            seed_transpiler=SETTINGS.transpiler_seed,
        )
        timings["transpilation_seconds"] = time.perf_counter() - started
        phase = "sampling"
        started = time.perf_counter()
        backend_calls += 1
        result = backend.run(
            executable, shots=SETTINGS.shots_per_entry, seed_simulator=SETTINGS.matrix_seed,
        ).result()
        timings["sampling_seconds"] = time.perf_counter() - started
        write_record(output / "raw-result.json", result.to_dict())
        phase = "capture_validation"
        counts = aligned_counts(result, executable)
        write_record(output / "counts.json", {"counts": counts})
        sampled_training, sampled_cross = kernels_from_counts(
            counts, pairs, angles["train_row_ids"], angles["test_row_ids"],
        )
        write_arrays(output / "sampled-kernels.npz", training=sampled_training, test_training=sampled_cross)
        phase = "spectral_repair"
        started = time.perf_counter()
        ideal_repaired, ideal_cross_repaired, ideal_diagnostic = repair_kernels(ideal_training, ideal_cross)
        sampled_repaired, sampled_cross_repaired, sampled_diagnostic = repair_kernels(sampled_training, sampled_cross)
        timings["repair_seconds"] = time.perf_counter() - started
        write_arrays(
            output / "repaired-kernels.npz", ideal_training=ideal_repaired,
            ideal_test_training=ideal_cross_repaired, sampled_training=sampled_repaired,
            sampled_test_training=sampled_cross_repaired,
        )
        write_record(output / "diagnostics.json", {
            "training_kernel_error": kernel_error(sampled_training, ideal_training),
            "test_training_kernel_error": kernel_error(sampled_cross, ideal_cross),
            "ideal_repair": ideal_diagnostic, "sampled_repair": sampled_diagnostic,
        })
        return {
            "status": "kernel_capture_complete", "backend": backend.name,
            "circuits": len(executable), "backend_calls": backend_calls,
            "shots_per_entry": SETTINGS.shots_per_entry,
            "total_shots": sum(sum(row.values()) for row in counts), "timings": timings,
            "classifier_fits": 0, "test_predictions_generated": False,
            "comparison_complete": False, "hardware_run": False,
        }
    except Exception as error:
        write_record(output / "failure.json", {
            "status": "capture_incomplete", "phase": phase,
            "backend_calls_attempted": backend_calls, "error_type": type(error).__name__,
            "error": str(error), "comparison_complete": False,
        })
        raise


def file_record(source: Path) -> dict:
    payload = source.read_bytes()
    return {"bytes": len(payload), "sha256": sha256(payload).hexdigest()}


def read_inputs(prepared: Path) -> tuple[dict, list]:
    return json.loads((prepared / "angles.json").read_text()), json.loads((prepared / "pairs.json").read_text())


def sampling_plan(prepared: Path = PREPARED_DIRECTORY) -> dict:
    prepared = prepared.resolve()
    verification = subprocess.run(
        [sys.executable, str(ROOT / "prepare_data.py"), "verify", "--output", str(prepared)],
        capture_output=True, text=True, check=True,
    )
    prepared_report = json.loads(verification.stdout)
    protocol = load_protocol()
    if prepared_report["status"] != "preparation_complete" or prepared_report["pair_count"] != protocol["quantum"]["matrix_circuits"]:
        raise ValueError("Prepared input coverage differs from the protocol")
    receipt = json.loads((prepared / "complete.json").read_text())
    configuration = json.loads(json.dumps(implementation_configuration(), allow_nan=False))
    configuration["source_sha256"].update({
        name: file_record(ROOT / name)["sha256"] for name in ("sample_kernels.py", "calibrate.py", "prepare_data.py")
    })
    calibration = json.loads((ROOT / "calibration-v1/report.json").read_text())
    quantum = protocol["quantum"]
    if quantum["matrix_circuits"] != quantum["train_off_diagonal_circuits"] + quantum["test_train_circuits"]:
        raise ValueError("Kernel budget arithmetic differs")
    if quantum["matrix_shots"] != quantum["matrix_circuits"] * SETTINGS.shots_per_entry or quantum["matrix_backend_calls"] != 1:
        raise ValueError("Shot or backend-call budget differs")
    reference_seconds = calibration["transpilation_seconds"] + calibration["sampling_seconds"]
    extrapolation = reference_seconds * quantum["matrix_circuits"] / calibration["circuits"]
    return {
        "status": "sampling_plan_prepared", "full_sampling_authorized": False,
        "prepared_directory": str(prepared), "prepared_receipt": file_record(prepared / "complete.json"),
        "prepared_files": receipt["files"], "implementation": configuration,
        "budget": {"circuits": quantum["matrix_circuits"], "shots_per_entry": SETTINGS.shots_per_entry,
                   "total_shots": quantum["matrix_shots"], "backend_calls": quantum["matrix_backend_calls"]},
        "backend": "local Qiskit BasicSimulator", "hardware_run": False,
        "runtime_estimate": {
            "reference_calibration_seconds": reference_seconds,
            "linear_extrapolation_seconds": extrapolation,
            "estimated_seconds": [math.ceil(extrapolation), math.ceil(extrapolation * ESTIMATE_SAFETY_FACTOR + ESTIMATE_STARTUP_ALLOWANCE_SECONDS)],
            "safety_factor": ESTIMATE_SAFETY_FACTOR,
            "startup_allowance_seconds": ESTIMATE_STARTUP_ALLOWANCE_SECONDS,
            "caveat": "Provisional extrapolation from two synthetic circuits, not a measured full-pilot runtime",
        },
        "resource_cost": "Local CPU and small four-qubit statevectors; no external service or hardware charges",
        "classifier_fitting_authorized": False, "test_scoring_authorized": False,
    }


def run_sampling(plan_file: Path, output: Path, approved: bool = False) -> dict:
    if approved is not True:
        raise ValueError("Full sampling requires explicit launch approval")
    output.mkdir(parents=True)
    phase = "input_admission"
    report = {}
    try:
        plan = json.loads(plan_file.read_text())
        prepared = Path(plan["prepared_directory"])
        if plan != sampling_plan(prepared):
            raise ValueError("Execution plan or sealed inputs changed")
        plan_record = file_record(plan_file)
        write_record(output / "execution-plan.json", plan)
        write_record(output / "started.json", {
            "status": "sampling_started", "explicit_launch_approval": True,
            "plan_sha256": plan_record["sha256"], "classifier_fitting_authorized": False,
            "test_scoring_authorized": False,
        })
        angles, pairs = read_inputs(prepared)
        write_record(output / "pairs.json", pairs)
        phase = "kernel_capture"
        report = capture_batch(angles, pairs, output)
        phase = "integrity"
        if sampling_plan(prepared) != plan or file_record(plan_file) != plan_record:
            raise ValueError("Execution bindings changed during sampling")
        if (report["circuits"], report["total_shots"], report["backend_calls"]) != (
            plan["budget"]["circuits"], plan["budget"]["total_shots"], plan["budget"]["backend_calls"],
        ):
            raise ValueError("Completed capture differs from the sealed budget")
        report["execution_plan_sha256"] = plan_record["sha256"]
        report["sealed_inputs_unchanged"] = True
        write_record(output / "report.json", report)
        write_record(output / "complete.json", {
            "status": "kernel_capture_complete", "comparison_complete": False,
            "files": {name: file_record(output / name) for name in CAPTURE_ARTIFACTS},
        })
        return report
    except Exception as error:
        if not (output / "failure.json").exists():
            write_record(output / "failure.json", {
                "status": "integrity_failed" if phase == "integrity" else "capture_incomplete",
                "phase": phase, "backend_calls_attempted": report.get("backend_calls", 0),
                "error_type": type(error).__name__, "error": str(error), "comparison_complete": False,
            })
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--prepared", type=Path, default=PREPARED_DIRECTORY)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approve-full-sampling", action="store_true")
    arguments = parser.parse_args()
    if arguments.action == "plan":
        plan = sampling_plan(arguments.prepared)
        write_record(arguments.output, plan)
        print(json.dumps({"status": plan["status"], "output": str(arguments.output),
                          "budget": plan["budget"], "runtime_estimate": plan["runtime_estimate"],
                          "full_sampling_authorized": False}, indent=2))
    else:
        if arguments.plan is None:
            parser.error("Sampling requires --plan and explicit launch approval")
        print(json.dumps(run_sampling(arguments.plan, arguments.output, arguments.approve_full_sampling), indent=2))


if __name__ == "__main__":
    main()