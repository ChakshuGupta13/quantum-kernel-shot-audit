import argparse
import json
import time
from hashlib import sha256
from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit.providers.basic_provider import BasicSimulator

from kernels import SETTINGS, calibration_checks, feature_circuit, implementation_configuration, overlap_circuit


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported capture value: {type(value).__name__}")


def write_record(destination: Path, record: dict) -> None:
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False, default=json_scalar)
        stream.write("\n")


def configuration_record() -> dict:
    record = implementation_configuration()
    record["source_sha256"][Path(__file__).name] = sha256(Path(__file__).read_bytes()).hexdigest()
    record["scope"] = "synthetic calibration only; no dataset fit or full matrix execution"
    return record


def run_calibration(output: Path) -> dict:
    configuration = configuration_record()
    output.mkdir(parents=True)
    write_record(output / "implementation-configuration.json", configuration)
    write_record(output / "started.json", {"status": "calibration_started", "full_sampling_authorized": False})
    phase = "construction"
    backend_calls = 0
    started = time.perf_counter()
    try:
        first = feature_circuit(SETTINGS.first_angles)
        second = feature_circuit(SETTINGS.second_angles)
        circuits = [overlap_circuit(first, first), overlap_circuit(first, second)]
        circuits[0].name = "calibration_self"
        circuits[1].name = "calibration_distinct"
        backend = BasicSimulator()
        phase = "transpilation"
        transpilation_started = time.perf_counter()
        executable = transpile(circuits, backend, optimization_level=SETTINGS.transpiler_optimization_level,
                               seed_transpiler=SETTINGS.transpiler_seed)
        transpilation_seconds = time.perf_counter() - transpilation_started
        phase = "sampling"
        sampling_started = time.perf_counter()
        backend_calls += 1
        result = backend.run(executable, shots=SETTINGS.shots_per_entry,
                             seed_simulator=SETTINGS.calibration_seed).result()
        sampling_seconds = time.perf_counter() - sampling_started
        counts = [dict(result.get_counts(index)) for index in range(len(executable))]
        write_record(output / "counts.json", {"counts": counts})
        phase = "validation"
        report = calibration_checks(counts)
        if configuration_record() != configuration:
            raise ValueError("Calibration source, protocol or environment changed during the run")
        report.update({
            "backend": backend.name,
            "backend_calls": backend_calls,
            "circuits": len(executable),
            "shots_per_circuit": SETTINGS.shots_per_entry,
            "total_shots": sum(sum(row.values()) for row in counts),
            "transpilation_seconds": transpilation_seconds,
            "sampling_seconds": sampling_seconds,
            "total_seconds": time.perf_counter() - started,
            "circuit_operations": [dict(circuit.count_ops()) for circuit in executable],
            "configuration_sha256": sha256((output / "implementation-configuration.json").read_bytes()).hexdigest(),
            "counts_sha256": sha256((output / "counts.json").read_bytes()).hexdigest(),
            "dataset_used": False,
            "preprocessing_fitted": False,
            "classifiers_fitted": False,
            "full_sampling_authorized": False,
        })
        write_record(output / "report.json", report)
        return report
    except Exception as error:
        write_record(output / "failure.json", {"status": "calibration_failed", "phase": phase,
                                               "backend_calls_attempted": backend_calls,
                                               "error_type": type(error).__name__, "error": str(error)})
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("export", "calibrate"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.action == "export":
        write_record(arguments.output, configuration_record())
        print(json.dumps({"status": "configuration_exported", "output": str(arguments.output), "backend_calls": 0}))
        return 0
    report = run_calibration(arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())