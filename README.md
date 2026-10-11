# Quantum Kernel Shot Audit

Georgia Tech Qiskit Fall Fest 2026, Prompt A. This repository preserves a small
comparison of ideal and finite-shot quantum kernels with linear and RBF SVMs on
Wisconsin Diagnostic Breast Cancer. It reports the observed negative result,
not a new kernel algorithm or quantum advantage.

## Open the Recorded Demo

Open [demo.html](demo.html) in a desktop or mobile browser. The package works
offline without a server, account or API key. It displays saved metrics, kernel
matrices and per-record predictions, not new model fitting or quantum execution.

The [six-scene presentation kit](video/production-plan.json) includes the
[narration transcript](video/narration.txt), [voice directions](video/voice-style.txt)
and prepared slides. Dedicated audio, video encoding and a public YouTube link
remain pending. This project requires its own QCA form submission; the Prompt B
video is not its presentation.

The kernel builder, guarded sampling and classifier runners, synthetic calibration
and training-only input preparation are implemented. Twenty-two quantum/capture
tests and nineteen preparation/evaluator tests pass. The shared preprocessing was fitted once on the 32 frozen training
records; the 16 test records were transformed without fitting. One real local
BasicSimulator job ran the two synthetic four-qubit controls at 256 shots each.
A separately approved local job completed all 1,008 dataset kernel circuits at
256 shots each. Ideal, sampled and repaired kernels are retained and independently
checked. The four fixed classifiers have now been fitted once on the 32 training
records and evaluated once on the same 16 test records. The comparison and raw
prediction records are complete and independently checked.
Code and recorded evidence are provided here. A dedicated Prompt A demo video
and final form submission are separate tasks; this repository is not a submission
confirmation.

The [design protocol](protocol.md) and [fixed contract](configuration.json) remain
the comparison specification. Runtime parameters are exported from the live
`KernelSettings` dataclass and checked against that contract; they are not copied
into a result by hand. Preprocessing settings likewise come from the live
`PreparationSettings` dataclass. This is a small hackathon pilot, not a clinical study.

## Implemented Scope

- [kernels.py](kernels.py) builds ideal fidelity kernels and ordered measured
  compute-uncompute circuits, reconstructs complete count batches, and applies
  the same training-derived spectral repair to training and test-training kernels.
- [calibrate.py](calibrate.py) exports code-derived settings and runs only the two
  fixed synthetic calibration cases. It uses a local BasicSimulator, reserves an
  exclusive output directory, retains raw counts and propagates recorded failures.
  It has no full-dataset execution command.
- [test_kernels.py](test_kernels.py) tests overlap identities, a wrong-uncompute
  negative control, pair order including coincident angles, incomplete/malformed
  counts, spectral extension, JSON-safe NumPy counts, one-call capture, refusal
  of repeats and failed/exceptional calibration handling.
- [prepare_data.py](prepare_data.py) admits the fixed dataset and row partition,
  fits preprocessing on training rows only, and freezes the angles, fitted
  parameters and ordered pair manifest with source and calibration bindings.
- [test_preparation.py](test_preparation.py) checks held-out-value poisoning,
  parameter replay, clipping, one-fit accounting, altered-source rejection,
  exclusive capture, retained failures and rehashed noncanonical row IDs.
- [sample_kernels.py](sample_kernels.py) consumes the sealed input bundle and
  samples the matrix in one local job after explicit approval. It checks result
  names, pair metadata, successful completion and every shot denominator; it
  retains raw results, raw/repaired kernels and failures without retries.
- [evaluate_models.py](evaluate_models.py) uses the saved repaired kernels and
  shared angles for the four fixed SVC rows. It requires explicit test-evaluation
  approval, fits each row once and retains predictions, decision scores, metrics
  and warnings. It never refits preprocessing or launches a quantum job.

## Calibration Evidence

The [real calibration report](calibration-v1/report.json) and
[raw counts](calibration-v1/counts.json) retain the one 512-shot job. Self-overlap
yielded 256 all-zero outcomes out of 256. The distinct pair's ideal fidelity was
0.057260307721200825 and its measured frequency was 19/256, or 0.07421875.
The absolute difference, 0.016958442278799175, is below the declared Hoeffding
tolerance 0.10172647691992058 at delta 0.01.

[Generated settings and source hashes](calibration-v1/implementation-configuration.json)
bind the capture to the current code and frozen protocol. Counts, reference
overlap, tolerance, source hashes and capture hashes were independently checked.
The recorded calibration-phase time was about 0.05 seconds, excluding interpreter
startup. It is not a timing measurement of the full pilot or a quantum speedup.

This establishes the two synthetic control checks only. It does not establish
predictive performance, clinical validity, research novelty or quantum advantage.
TLC/SANY remains unrun; the protocol model has only shape/reference checks.

## Prepared Inputs

The [preparation receipt](preparation-v1/complete.json) binds all six preparation
artifacts, including [angles](preparation-v1/angles.json),
[fitted parameters](preparation-v1/preprocessing.json),
[ordered pairs](preparation-v1/pairs.json) and the
[preparation report](preparation-v1/report.json). The four-component angles cover
32 training and 16 test rows, and the manifest contains 496 training pairs plus
512 test-training pairs. No diagonal or test-test sampling pairs are included.

Seven test coordinates were clipped to the training-fitted angle range. Their
row IDs, component indices and before/after values are retained. Independent
replay from the original data and saved parameters reproduced every angle and
clipping record without another fit. The test-poisoning check leaves all fitted
parameters and training angles unchanged.

[Generated preparation bindings](preparation-v1/implementation-configuration.json)
retain the protocol, dataset, runtime, source and calibration digests. Pair
generation reuses the existing kernel code in a separate process; Qiskit was not
imported into the scikit-learn preprocessing process. Preparation performed zero
quantum jobs, classifier fits or test predictions. The original calibration and
recorded calibration artifacts remain unchanged.

## Captured Kernels

The [sampling receipt](sampling-v1/complete.json) binds the complete capture,
[raw counts](sampling-v1/counts.json), [kernel diagnostics](sampling-v1/diagnostics.json)
and [execution report](sampling-v1/report.json). Every declared pair has one
successful result with matching circuit name, metadata and 256 accounted shots.
The job used one local BasicSimulator call and 258,048 shots. Including the
earlier synthetic calibration, this pilot has used two local sampler calls and
258,560 shots. No hardware or external-service execution was used.

Confidence: share with the single-partition, single-seed pilot limitations.
The relative Frobenius error is the norm of sampled minus ideal divided by the
ideal matrix norm. These are matrix errors, not classification error rates.

| Kernel block | Relative Frobenius error |
|---|---:|
| Training | 0.06870678656334796 |
| Test-to-training | 0.1184668835160604 |

Both training kernels retained all 32 modes under the fixed spectral rule.
The raw minimum eigenvalues were approximately 0.147712 for the ideal kernel and
0.141510 for the sampled kernel. The matrices and consistent test-to-training
repair were independently reconstructed from the frozen angles and raw counts.
All capture hashes and sealed-input bindings match.

The sampling phase took about 0.348 seconds, excluding construction, transpilation,
ideal-kernel computation and interpreter/input-verification overhead. This is not
an end-to-end speedup benchmark. The provisional pre-launch estimate below remains
a historical estimate; the actual phase timings are in the report. Classifier
performance, significance and quantum advantage have not been established.

## Sampling Plan

[sampling-plan-v1.json](sampling-plan-v1.json) binds the current sampling source,
runtime and prepared-input hashes. The plan's JSON export/reload check passed.
Preparing the plan did not evaluate dataset kernels or dispatch a sampler job.

The requested job contains 1,008 circuits at 256 shots per entry, totaling 258,048
shots in one local BasicSimulator call. The provisional runtime estimate is
24-103 seconds, extrapolated from the two synthetic calibration circuits with
a safety factor and startup allowance. This is not a measured full-pilot runtime
or a speedup claim. It uses local CPU resources without external-service or
hardware charges.

Completed launch command, run after separate approval:

```sh
.venv/bin/python sample_kernels.py run --plan sampling-plan-v1.json --output sampling-v1 --approve-full-sampling
```

The approval flag must only be supplied after the separate launch approval.
Without it the runner stops before reading inputs or touching a backend.
The `sampling-v1` directory is now reserved; do not repeat or replace this capture.
Output directories are exclusive. Incomplete captures, changed bindings or failed
experiments cannot receive a completion receipt, and no retry is automatic.
Successful sampling produces kernel artifacts only, not a completed classifier
comparison. The separately approved classifier evaluation below is now complete.

## Classifier Evaluation Plan

[evaluation-plan-v1.json](evaluation-plan-v1.json) binds the classifier settings,
current source, completed sampling receipt and frozen preprocessing inputs.
The real-plan JSON round-trip and no-fit/no-label-loading checks passed. Tests on
synthetic data verify metric formulas, one fit/predict/decision call per row,
repeat refusal, convergence failures and post-fit integrity changes.

Completed command, run after separate approval:

```sh
.venv/bin/python evaluate_models.py run --plan evaluation-plan-v1.json --output evaluation-v1 --approve-test-evaluation
```

Supply the flag only after explicit approval for the real test evaluation. The
job fits four fixed classifiers on 32 training rows, then predicts and obtains
decision scores on the same 16 test rows. It uses local CPU resources, should
take seconds, and has no external-service or hardware charges. There is no
preprocessing refit, kernel resampling, parameter search or retry.

The `evaluation-v1` directory is now reserved; do not repeat or replace this
evaluation. The plan retains its pre-approval status, while the execution record
and completed receipt describe the approved run.

The pinned scikit-learn release warns that the declared `probability=False`
parameter is deprecated. That setting is retained; ordinary warnings are captured
in each row record, while a convergence warning fails the row and stops evaluation.
Only complete valid rows with unchanged bindings receive a comparison receipt.
The completed comparison is recorded below; no general ranking or advantage is claimed.

## Classifier Results

Confidence: share with noted caveats. The [completed comparison](evaluation-v1/report.json)
and [receipt](evaluation-v1/complete.json) retain four fits, four prediction calls
and four decision-function calls. Each row uses the same 16 test records, comprising
10 benign and 6 malignant labels. These are classical SVC models using either a
quantum-derived kernel or the same four shared angle coordinates.

| Classifier input | Balanced accuracy | Accuracy | ROC-AUC |
|---|---:|---:|---:|
| Ideal quantum kernel | 40.00% | 50.00% | 0.3500 |
| 256-shot quantum kernel | 45.00% | 56.25% | 0.3167 |
| Linear kernel | 95.00% | 93.75% | 0.9833 |
| RBF kernel | 91.67% | 93.75% | 0.9667 |

The sampled-minus-ideal balanced-accuracy change is +5 percentage points. Exactly
one prediction changed: a benign record incorrectly classified by the ideal
kernel became correctly classified by the sampled kernel. Both quantum rows
missed all six malignant records. The sampled kernel's ROC-AUC was lower than
the ideal kernel's despite its higher balanced accuracy.

The fixed quantum-kernel configuration performed worse than both classical
baselines in this pilot. Sampling noise alone does not explain that gap, since
the ideal quantum kernel also performed poorly. The cause of the poor ideal-kernel
performance has not been established. No parameter adjustment or additional fit
was made after seeing these results.

Per-record evidence is retained in [quantum_ideal.json](evaluation-v1/rows/quantum_ideal.json),
[quantum_shot.json](evaluation-v1/rows/quantum_shot.json),
[linear.json](evaluation-v1/rows/linear.json) and [rbf.json](evaluation-v1/rows/rbf.json).
Labels, score orientation, confusion matrices, accuracy, balanced accuracy and
pairwise ROC-AUC were independently recomputed without fitting or predicting again.
Source and input hashes match. The four prediction sets are not 64 independent
test records; they share the same sixteen records.

This is one partition and one sampling seed, with fixed model choices and a very
small test set. No significance test, clinical validation or quantum-advantage
claim is made. The test set is now consumed and remains locked against tuning.

## Training-Only Diagnosis

The [diagnostic report](training-diagnostics-v1/report.json) and
[receipt](training-diagnostics-v1/complete.json) record an exploratory diagnosis
after the fixed evaluation. [diagnose_training.py](diagnose_training.py) uses
the saved training kernels and four training angle coordinates, with only the
32 training labels entering the numerical calculations. Six focused synthetic
tests and independent numerical reconstruction passed. No new classifier fits,
preprocessing fits, quantum jobs or test predictions were generated.

| Training kernel | Within-class mean | Between-class mean | Centered label alignment | Largest eigenvalue |
|---|---:|---:|---:|---:|
| Ideal quantum | 0.060633 | 0.060393 | 0.165429 | 3.023872 |
| 256-shot quantum | 0.059875 | 0.058350 | 0.168120 | 2.986085 |
| Linear | 6.498847 | 6.121837 | 0.424409 | 224.651570 |
| RBF | 0.356113 | 0.179091 | 0.401549 | 11.490153 |

Each unordered off-diagonal pair is counted once. There are 256 within-class
pairs and 240 between-class pairs; the pooled within-class mean weights all
190 benign pairs and 66 malignant pairs, not the two class means equally.
Kernel scales differ, so compare within versus between within each row; centered
alignment provides the scale-normalized comparison across rows.

The identity training matrix has centered alignment 0.179605 from self-similarity
alone. It is a diagnostic reference, not a classifier-performance baseline.
The quantum rows have nearly equal pooled within/between similarities and lower
alignment than both classical kernels in this training sample. All classifiers
used C=1, but the scale differences remain; these summaries do not isolate the
cause of the predictive failure or provide matched-capacity evidence.

Recommendation: park this fixed four-qubit, one-repetition feature map. The
diagnostics are consistent with weak training-label structure, not a demonstrated
shot-count shortage. They do not rule out other feature maps or establish a
general quantum-kernel limit. Preserve the pilot and its consumed test set;
no further runs or test-based tuning are part of this recommendation.

Original mixed training/test artifacts and the full source dataset were hashed
for integrity. Only training fields were extracted for numerical diagnostics;
test labels and prediction records were not inputs to the diagnosis.

## Commands

From this repository root, create a dedicated environment using Python 3.14.7
and the [pinned requirements](requirements.txt). Keep quantum and scikit-learn
tests in separate processes.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s . -p test_kernels.py -v
.venv/bin/python -m unittest discover -s . -p test_preparation.py -v
.venv/bin/python prepare_data.py verify --output preparation-v1
```

Keep the quantum and preprocessing suites in separate processes. The preparation
directory is already reserved; `prepare` refuses to repeat it. Do not delete the
bundle to refit or replace its inputs. The `verify` command checks it without fitting.

The completed calibration command was:

```sh
.venv/bin/python calibrate.py calibrate --output calibration-v1
```

That output directory is already reserved and the command now refuses to repeat
the capture. Do not delete it to rerun the fixed attempt. Parameters can also be
exported with the `export` action to a new output file, without sampler execution.

## Next Gate

Preserve the completed comparison, training diagnosis and locked test set. The
fixed configuration is recommended for parking rather than further sampling or
tuning. Any packaging or new experiment is a separate user decision; the existing
preprocessing, kernels and classifier evaluation must not be repeated or tuned
against these test records.

## Attribution and Publication

Authored code and documentation use [MIT](LICENSE). UCI licenses the source
dataset under CC BY 4.0. Credit follows its [dataset record](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic):
William Wolberg, Olvi Mangasarian, Nick Street and W. Street, DOI
[10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B).

GitHub Copilot assisted with implementation, tests and documentation. Qiskit,
NumPy, SciPy and scikit-learn provide the standard algorithms and numerical
libraries. The organizer supplies the task; no ownership or invention of those
methods is claimed. Historical execution metadata is exported with portable
paths and publication receipts; see [PUBLICATION.md](PUBLICATION.md).