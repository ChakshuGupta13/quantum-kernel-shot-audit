# Prompt A Comparison Protocol

Status: fixed design contract. [configuration.json](configuration.json) is the
complete parameter specification. The [implementation status](README.md) records
the completed synthetic calibration and frozen training-only preprocessing.
The separately approved dataset kernel capture is complete. Classifier fitting
and test scoring are also complete after separate approval. The
[completed comparison](evaluation-v1/report.json) records the fixed four rows;
the held-out records are consumed and remain locked against tuning.

## Question and Scope

Measure how one finite-shot estimate changes a fixed quantum-kernel classifier
relative to its ideal reference, with linear and RBF SVM baselines. This is a
48-record pilot, not a new kernel algorithm, clinical validation or a test of
quantum advantage. Negative and positive observed differences are equally valid
outcomes; neither is a statistically established ranking from sixteen test cases.

The [organizer's Prompt A](https://docs.google.com/presentation/d/1O5WEDQost04m8rjneB8nCuAGwgyVhckoyc87dRa8j0g/edit)
names [Wisconsin Diagnostic Breast Cancer](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
The audited scikit-learn loader has 569 records and 30 numeric features. UCI
licenses the dataset CC BY 4.0, DOI 10.24432/C5DW2B, and credits William Wolberg,
Olvi Mangasarian, Nick Street and W. Street. Its features are derived from images
of fine-needle aspirates; they are not a proposed screening-test panel.

## Admission and Shared Inputs

The configuration binds the source byte digest, feature order and explicit row
IDs. A seeded stratified admission selects 48 records; a second seeded split
reserves 32 for training and 16 for testing. The other 521 records are omitted
from this pilot. Training has 20 benign and 12 malignant records; testing has
10 benign and 6 malignant records. The positive normalized label is malignant.
Labels may be used for stratification and admission counts, not kernel choices.

Fit StandardScaler on the training records, then PCA with four components and
full SVD, then MinMaxScaler to angles from zero to pi. Fit this shared pipeline
once. Transform test inputs without fitting; clip them to the declared angle
range and record the clipping. Every classifier receives the same four angle
coordinates. The comparison matches inputs and preprocessing, not model capacity.

## Kernels and Spectral Repair

Use Qiskit's `zz_feature_map` with four features, one repetition, linear
entanglement and alpha 2. For each input, the feature map prepares a state.
The ideal kernel is the squared inner-product magnitude of two numerical
statevectors. The sampled kernel applies one feature map followed by the inverse
of the other and estimates the all-zero probability from 256 shots.

Set training diagonal entries analytically to one. Estimate each unordered
off-diagonal training pair once and mirror that estimate. Sample every declared
test-training pair. Do not skip coincident transformed inputs or choose pairs
after inspecting estimates. All raw counts and pair identities are retained.

Finite-shot training matrices need not be positive semidefinite. Use the same
fixed spectral treatment for both quantum classifiers. Write a raw training
Gram matrix as $B=V\Lambda V^T$. Retain columns in $V_+$ whose eigenvalues exceed
$10^{-10}$ and collect those positive eigenvalues in $\Lambda_+$. For a raw
test-training matrix $C_{\mathrm{raw}}$, use

$$
K_{\mathrm{train}}=V_+\Lambda_+V_+^T,
\qquad
K_{\mathrm{test,train}}=C_{\mathrm{raw}}V_+V_+^T.
$$

This uses one consistent training-derived feature representation. Training
features are $V_+\Lambda_+^{1/2}$ and test features are
$C_{\mathrm{raw}}V_+\Lambda_+^{-1/2}$; their cross-product is the displayed
test-training matrix. The inverse exists because only positive modes are retained.
Do not renormalize the diagonal or use test labels in the repair. Repaired
entries are classifier-kernel values, not measured fidelity probabilities.
Retain raw errors, minimum eigenvalues, retained ranks and repair norms.

## Rows and Reporting

| Row | Classifier | Input |
|---|---|---|
| quantum_ideal | Precomputed-kernel SVC | Repaired ideal quantum kernel |
| quantum_shot | Precomputed-kernel SVC | Repaired 256-shot quantum kernel |
| linear | Linear SVC | Shared four-dimensional angles |
| rbf | RBF SVC with training-derived scale gamma | Shared four-dimensional angles |

Fix C at 1, no class weighting or probability calibration, and the solver settings
in the configuration. Fit each classifier once, then call `predict` once and
`decision_function` once on its sixteen prepared test inputs. Reuse those outputs
for every metric; neither call constructs another quantum kernel. Convergence warnings fail the row.
No model substitution, hyperparameter tuning, seed search or test-based selection
is permitted. A failed row remains a failure, not a dropped comparison row.

Balanced accuracy is primary. Also retain accuracy, decision-score ROC-AUC,
the confusion matrix and source-aligned predictions. Report the paired sampled
minus ideal balanced-accuracy difference on the same records, without significance,
clinical or advantage claims. Keep data admission, preprocessing and kernel
diagnostics separate from predictive scores. Kernel diagnostics are available;
the [fixed predictive results](evaluation-v1/report.json) are now available.

## Calibration and Budget

Before full sampling, run one local 256-shot calibration batch containing two
fixed circuits. Self-overlap must yield only the all-zero word. The distinct
pair must agree with numerical statevector fidelity within the declared
Hoeffding tolerance at delta 0.01. Failed calibration stops the experiment.
Calibration also supplies the first measured timing; it does not authorize the
larger job. Present its command, estimated runtime and cost for separate approval.

The full matrix requires 496 training pairs and 512 test-training pairs, totaling
1,008 circuits and 258,048 shots. Send those circuits in one local BasicSimulator
job. Including calibration, the ceiling is two backend calls, 1,010 circuits and
258,560 shots. There are no remote or hardware jobs. Record all phase timings;
do not imply that fewer quantum queries establish an end-to-end speedup.

Freeze configuration, dataset and implementation hashes before any fit. Preserve
the pair manifest, raw counts, fitted preprocessing, raw and repaired matrices,
predictions and failure reasons. Complete reporting requires every pair and all
four valid rows with unchanged inputs. No retries or partial-comparison conclusion
are admitted by this protocol.

## Protocol Model and Boundaries

[KernelProtocol.tla](tla/KernelProtocol.tla) models four classifier rows and three
abstract required pairs, with calibration, capture, fit, score, integrity failure
and publication transitions. It does not model kernel mathematics, simulation,
classifier convergence, predictive validity or performance. The model applies
before publication; later consumers must recheck artifact digests.

No `tlc` executable was found on the checked PATH, no matching jar was returned
by the workspace file search, and the two standard local jar paths checked were
absent. Only delimiter-shape and configuration-reference checks were performed;
TLC/SANY and formal verification are not claimed.

Design-stage checks verified the dataset digest, exact partition, circuit/shot
arithmetic and unfitted library APIs. The spectral extension identity and an
omitted-projection negative control passed on a synthetic matrix. The later
[prepared input bundle](preparation-v1/complete.json) records one training-only
preprocessing fit and has passed independent parameter replay. Neither those
checks nor the synthetic calibration establish predictive performance.

This protocol and its recorded results do not constitute an event submission.
The [matrix runner](sample_kernels.py) and
[sealed launch plan](sampling-plan-v1.json) produced the
[complete kernel capture](sampling-v1/complete.json) in one separately approved
local job. Raw counts, kernel matrices, spectral repair and frozen bindings were
independently checked. The separately approved classifier comparison is complete.

The [four-classifier evaluator](evaluate_models.py) and
[evaluation plan](evaluation-plan-v1.json) are implemented and tested on synthetic
inputs, including a no-fit check of the real plan. The separately approved
[real pilot evaluation](evaluation-v1/complete.json) fitted and evaluated each
declared row once. Its predictions, metrics, label orientation and integrity
bindings were independently checked. No preprocessing or kernel regeneration
occurred during evaluation. No significance, clinical or quantum-advantage claim
is made from this single-partition pilot.