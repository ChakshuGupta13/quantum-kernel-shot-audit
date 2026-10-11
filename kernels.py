import json
import math
import platform
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version
from numbers import Integral
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector


PROTOCOL_FILE = Path(__file__).with_name("configuration.json")
PROTOCOL_SHA256 = "113f4c42b865406dc24b82f79092c81097ccfd6e13978a764066a24187f27a0f"


@dataclass(frozen=True)
class KernelSettings:
    feature_dimension: int = 4
    reps: int = 1
    entanglement: str = "linear"
    alpha: float = 2.0
    parameter_prefix: str = "x"
    insert_barriers: bool = False
    shots_per_entry: int = 256
    calibration_seed: int = 2029
    matrix_seed: int = 2028
    transpiler_seed: int = 2030
    transpiler_optimization_level: int = 0
    numeric_tolerance: float = 1e-10
    eigenvalue_threshold: float = 1e-10
    distinct_case_delta: float = 0.01
    angle_upper: float = math.pi
    first_angles: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
    second_angles: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4)


SETTINGS = KernelSettings()


def load_protocol() -> dict:
    payload = PROTOCOL_FILE.read_bytes()
    if sha256(payload).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen protocol bytes changed")
    protocol = json.loads(payload)
    quantum = protocol["quantum"]
    calibration = protocol["calibration"]
    declared = KernelSettings(
        **{name: quantum[name] for name in (
            "feature_dimension", "reps", "entanglement", "alpha", "parameter_prefix",
            "insert_barriers", "shots_per_entry", "calibration_seed", "matrix_seed",
            "transpiler_seed", "transpiler_optimization_level",
        )},
        numeric_tolerance=calibration["numeric_statevector_tolerance"],
        eigenvalue_threshold=protocol["spectral_repair"]["eigenvalue_threshold"],
        distinct_case_delta=calibration["distinct_case_delta"],
        angle_upper=protocol["preprocessing"]["steps"][-1]["feature_range"][1],
        first_angles=tuple(calibration["first_angles"]),
        second_angles=tuple(calibration["second_angles"]),
    )
    if declared != SETTINGS:
        raise ValueError("Live kernel settings differ from the frozen protocol")
    return protocol


def implementation_configuration() -> dict:
    protocol = load_protocol()
    packages = {name: version(name) for name in protocol["versions"] if name != "python"}
    if packages != {name: expected for name, expected in protocol["versions"].items() if name != "python"}:
        raise ValueError("Installed package versions differ from the protocol")
    if platform.python_version() != protocol["versions"]["python"]:
        raise ValueError("Python version differs from the protocol")
    return {
        "protocol_sha256": PROTOCOL_SHA256,
        "settings": asdict(SETTINGS),
        "source_sha256": {Path(__file__).name: sha256(Path(__file__).read_bytes()).hexdigest()},
        "packages": packages,
        "python": platform.python_version(),
        "parameter_origin": "live KernelSettings dataclass; checked against the design contract",
        "scope": "Hackathon kernel builder and synthetic calibration; no paper registry or full experiment",
    }


def validate_angles(values, settings: KernelSettings = SETTINGS) -> np.ndarray:
    angles = np.asarray(values, dtype=float)
    if angles.ndim != 2 or not angles.shape[0] or angles.shape[1] != settings.feature_dimension:
        raise ValueError("Expected nonempty rows of feature-map angles")
    if not np.isfinite(angles).all():
        raise ValueError("Angles must be finite")
    if np.any(angles < 0) or np.any(angles > settings.angle_upper):
        raise ValueError("Angles lie outside the declared range")
    return angles


def feature_circuit(values, settings: KernelSettings = SETTINGS) -> QuantumCircuit:
    angles = validate_angles(np.asarray(values)[None, :], settings)[0]
    template = zz_feature_map(
        feature_dimension=settings.feature_dimension,
        reps=settings.reps,
        entanglement=settings.entanglement,
        alpha=settings.alpha,
        parameter_prefix=settings.parameter_prefix,
        insert_barriers=settings.insert_barriers,
    )
    return template.assign_parameters(angles.tolist())


def overlap_circuit(first: QuantumCircuit, second: QuantumCircuit, measured: bool = True) -> QuantumCircuit:
    if first.num_qubits != second.num_qubits or first.num_clbits or second.num_clbits:
        raise ValueError("Expected equally sized unmeasured feature circuits")
    circuit = first.copy()
    circuit.compose(second.inverse(), inplace=True)
    if measured:
        circuit.measure_all()
    return circuit


def ideal_kernels(training, testing, settings: KernelSettings = SETTINGS) -> tuple[np.ndarray, np.ndarray]:
    training = validate_angles(training, settings)
    testing = validate_angles(testing, settings)
    train_states = np.asarray([Statevector.from_instruction(feature_circuit(row, settings)).data for row in training])
    test_states = np.asarray([Statevector.from_instruction(feature_circuit(row, settings)).data for row in testing])
    train_gram = np.abs(train_states @ train_states.conj().T) ** 2
    cross_gram = np.abs(test_states @ train_states.conj().T) ** 2
    np.fill_diagonal(train_gram, 1)
    return train_gram, cross_gram


def ordered_pairs(train_ids, test_ids) -> list[dict]:
    if not train_ids or not test_ids or len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids):
        raise ValueError("Row identifiers must be nonempty and unique")
    if set(train_ids) & set(test_ids):
        raise ValueError("Training and test row identifiers overlap")
    pairs = [{"block": "train", "row": row, "column": column,
              "left_row_id": train_ids[row], "right_row_id": train_ids[column]}
             for row in range(len(train_ids)) for column in range(row + 1, len(train_ids))]
    pairs.extend({"block": "test_train", "row": row, "column": column,
                  "left_row_id": test_ids[row], "right_row_id": train_ids[column]}
                 for row in range(len(test_ids)) for column in range(len(train_ids)))
    return pairs


def build_overlap_batch(training, testing, train_ids, test_ids, settings: KernelSettings = SETTINGS) -> tuple[list, list]:
    training = validate_angles(training, settings)
    testing = validate_angles(testing, settings)
    if len(training) != len(train_ids) or len(testing) != len(test_ids):
        raise ValueError("Angle rows differ from the identifier manifest")
    manifest = ordered_pairs(train_ids, test_ids)
    train_circuits = [feature_circuit(row, settings) for row in training]
    test_circuits = [feature_circuit(row, settings) for row in testing]
    circuits = []
    for index, pair in enumerate(manifest):
        first = train_circuits[pair["row"]] if pair["block"] == "train" else test_circuits[pair["row"]]
        second = train_circuits[pair["column"]]
        circuit = overlap_circuit(first, second)
        circuit.name = f"kernel_{index}"
        circuit.metadata = {"pair_index": index, **pair}
        circuits.append(circuit)
    return circuits, manifest


def count_probability(counts: dict, settings: KernelSettings = SETTINGS) -> float:
    for word, count in counts.items():
        if not isinstance(word, str) or len(word) != settings.feature_dimension or set(word) - {"0", "1"}:
            raise ValueError("Expected one measured binary word per outcome")
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError("Outcome counts must be nonnegative integers")
    if sum(counts.values()) != settings.shots_per_entry:
        raise ValueError("Counts do not cover the declared shot budget")
    return float(counts.get("0" * settings.feature_dimension, 0) / settings.shots_per_entry)


def kernels_from_counts(count_sets, manifest, train_ids, test_ids, settings: KernelSettings = SETTINGS) -> tuple[np.ndarray, np.ndarray]:
    expected = ordered_pairs(train_ids, test_ids)
    if manifest != expected or len(count_sets) != len(expected):
        raise ValueError("Incomplete or changed ordered kernel-pair capture")
    train_gram = np.eye(len(train_ids))
    cross_gram = np.empty((len(test_ids), len(train_ids)))
    for counts, pair in zip(count_sets, expected):
        probability = count_probability(counts, settings)
        if pair["block"] == "train":
            train_gram[pair["row"], pair["column"]] = probability
            train_gram[pair["column"], pair["row"]] = probability
        else:
            cross_gram[pair["row"], pair["column"]] = probability
    return train_gram, cross_gram


def repair_kernels(training, cross, settings: KernelSettings = SETTINGS) -> tuple[np.ndarray, np.ndarray, dict]:
    training = np.asarray(training, dtype=float)
    cross = np.asarray(cross, dtype=float)
    if training.ndim != 2 or not len(training) or training.shape[0] != training.shape[1]:
        raise ValueError("Training Gram matrix must be nonempty and square")
    if cross.ndim != 2 or not len(cross) or cross.shape[1] != len(training):
        raise ValueError("Cross-kernel columns must match the training rows")
    if not np.isfinite(training).all() or not np.isfinite(cross).all():
        raise ValueError("Kernel arrays must be finite")
    if not np.allclose(training, training.T, rtol=0, atol=settings.numeric_tolerance):
        raise ValueError("Training Gram matrix is not symmetric")
    if not np.allclose(np.diag(training), 1, rtol=0, atol=settings.numeric_tolerance):
        raise ValueError("Raw training diagonal differs from analytic one")
    eigenvalues, vectors = np.linalg.eigh(training)
    retained = eigenvalues > settings.eigenvalue_threshold
    if not retained.any():
        raise ValueError("No positive training modes remain")
    positive_vectors = vectors[:, retained]
    repaired = (positive_vectors * eigenvalues[retained]) @ positive_vectors.T
    repaired_cross = (cross @ positive_vectors) @ positive_vectors.T
    diagnostics = {
        "minimum_raw_eigenvalue": float(eigenvalues[0]),
        "retained_rank": int(retained.sum()),
        "training_repair_frobenius_norm": float(np.linalg.norm(repaired - training)),
    }
    return repaired, repaired_cross, diagnostics


def calibration_checks(count_sets, settings: KernelSettings = SETTINGS) -> dict:
    if len(count_sets) != 2:
        raise ValueError("Both declared calibration cases are required")
    first = Statevector.from_instruction(feature_circuit(settings.first_angles, settings))
    second = Statevector.from_instruction(feature_circuit(settings.second_angles, settings))
    reference = float(abs(np.vdot(second.data, first.data)) ** 2)
    observed = [count_probability(counts, settings) for counts in count_sets]
    tolerance = math.sqrt(math.log(2 / settings.distinct_case_delta) / (2 * settings.shots_per_entry))
    cases = [
        {"case": "self_overlap", "reference": 1.0, "observed": observed[0], "passed": observed[0] == 1.0},
        {"case": "distinct_overlap", "reference": reference, "observed": observed[1],
         "absolute_error": abs(reference - observed[1]), "passed": abs(reference - observed[1]) <= tolerance},
    ]
    passed = all(case["passed"] for case in cases)
    return {"status": "calibration_passed" if passed else "calibration_failed", "passed": passed,
            "hoeffding_tolerance": tolerance, "cases": cases}