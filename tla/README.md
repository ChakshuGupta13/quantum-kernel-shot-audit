# KernelProtocol Model

Source contract: [protocol.md](../protocol.md) and
[configuration.json](../configuration.json). This is a pre-publication protocol
model, not a model of the dataset, fidelity circuit, spectral algebra or SVM.

The bounds are four classifier rows, three abstract required kernel pairs,
two backend-call slots and one fit/score slot per row. The actual experiment
declares 1,008 matrix pairs; three pairs exercise incomplete versus complete
capture in the model but are not an exhaustive check of that larger domain.

Actions distinguish preparation, calibration, sampling, capture, fitting,
scoring and completion, with explicit preparation/calibration/capture/row/integrity
failure reasons. A completed comparison requires calibration, every pair, and
every valid row. Failed or incomplete work cannot become a completed comparison.

The intended invariant checks are type and bounded-count safety, freezing before
work, fit-before-score, complete capture before scoring, score-backed valid labels,
integrity-stop behavior, calibration-before-matrix and complete publication coverage.
No liveness or eventual scientific success is claimed.

From this directory, the intended command is:

```sh
java -cp /path/to/tla2tools.jar tlc2.TLC -workers 1 KernelProtocol
```

TLC was not on the checked PATH and no jar was found in the workspace search or
two standard local paths checked. TLC/SANY was not run. Configuration references
and delimiter shape are the only executable checks available here; those do not
prove the model parses or its invariants hold. A counterexample from a future TLC
run must be reported before changing the contract or spec.