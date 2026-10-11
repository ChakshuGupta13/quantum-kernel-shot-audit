------------------------- MODULE KernelProtocol -------------------------
EXTENDS Naturals, FiniteSets

CONSTANTS Rows, Pairs

VARIABLES phase, integrity, frozen, calibrated, jobs, captured,
          fits, scores, labels, published, failureReason

vars == <<phase, integrity, frozen, calibrated, jobs, captured,
          fits, scores, labels, published, failureReason>>

Phases == {"draft", "frozen", "prepared", "kernelReady", "capturing",
           "evaluating", "complete", "failed"}
Reasons == {"none", "preprocessing", "calibration", "capture", "row", "integrity"}

Init ==
    /\ phase = "draft"
    /\ integrity = TRUE
    /\ frozen = FALSE
    /\ calibrated = FALSE
    /\ jobs = 0
    /\ captured = {}
    /\ fits = [row \in Rows |-> 0]
    /\ scores = [row \in Rows |-> 0]
    /\ labels = [row \in Rows |-> "pending"]
    /\ published = FALSE
    /\ failureReason = "none"

Freeze ==
    /\ phase = "draft"
    /\ integrity
    /\ phase' = "frozen"
    /\ frozen' = TRUE
    /\ UNCHANGED <<integrity, calibrated, jobs, captured, fits, scores,
                   labels, published, failureReason>>

Prepare ==
    /\ phase = "frozen"
    /\ integrity
    /\ phase' = "prepared"
    /\ UNCHANGED <<integrity, frozen, calibrated, jobs, captured, fits,
                   scores, labels, published, failureReason>>

FailPreparation ==
    /\ phase = "frozen"
    /\ phase' = "failed"
    /\ failureReason' = "preprocessing"
    /\ UNCHANGED <<integrity, frozen, calibrated, jobs, captured, fits,
                   scores, labels, published>>

Calibrate ==
    /\ phase = "prepared"
    /\ integrity
    /\ jobs = 0
    /\ phase' = "kernelReady"
    /\ calibrated' = TRUE
    /\ jobs' = 1
    /\ UNCHANGED <<integrity, frozen, captured, fits, scores, labels,
                   published, failureReason>>

FailCalibration ==
    /\ phase = "prepared"
    /\ jobs = 0
    /\ phase' = "failed"
    /\ jobs' = 1
    /\ failureReason' = "calibration"
    /\ UNCHANGED <<integrity, frozen, calibrated, captured, fits, scores,
                   labels, published>>

StartSampling ==
    /\ phase = "kernelReady"
    /\ integrity /\ calibrated /\ jobs = 1
    /\ phase' = "capturing"
    /\ jobs' = 2
    /\ UNCHANGED <<integrity, frozen, calibrated, captured, fits, scores,
                   labels, published, failureReason>>

CapturePair(pair) ==
    /\ phase = "capturing"
    /\ integrity
    /\ pair \in Pairs \ captured
    /\ captured' = captured \cup {pair}
    /\ UNCHANGED <<phase, integrity, frozen, calibrated, jobs, fits,
                   scores, labels, published, failureReason>>

SealKernels ==
    /\ phase = "capturing"
    /\ integrity /\ captured = Pairs
    /\ phase' = "evaluating"
    /\ UNCHANGED <<integrity, frozen, calibrated, jobs, captured, fits,
                   scores, labels, published, failureReason>>

FailCapture ==
    /\ phase = "capturing"
    /\ phase' = "failed"
    /\ failureReason' = "capture"
    /\ UNCHANGED <<integrity, frozen, calibrated, jobs, captured, fits,
                   scores, labels, published>>

FitRow(row) ==
    /\ phase = "evaluating"
    /\ integrity /\ fits[row] = 0
    /\ fits' = [fits EXCEPT ![row] = 1]
    /\ UNCHANGED <<phase, integrity, frozen, calibrated, jobs, captured,
                   scores, labels, published, failureReason>>

ScoreRow(row) ==
    /\ phase = "evaluating"
    /\ integrity /\ fits[row] = 1 /\ scores[row] = 0
    /\ scores' = [scores EXCEPT ![row] = 1]
    /\ labels' = [labels EXCEPT ![row] = "valid"]
    /\ UNCHANGED <<phase, integrity, frozen, calibrated, jobs, captured,
                   fits, published, failureReason>>

FailRow(row) ==
    /\ phase = "evaluating"
    /\ labels[row] = "pending"
    /\ phase' = "failed"
    /\ labels' = [labels EXCEPT ![row] = "failed"]
    /\ failureReason' = "row"
    /\ UNCHANGED <<integrity, frozen, calibrated, jobs, captured, fits,
                   scores, published>>

Invalidate ==
    /\ phase \notin {"complete", "failed"}
    /\ phase' = "failed"
    /\ integrity' = FALSE
    /\ failureReason' = "integrity"
    /\ UNCHANGED <<frozen, calibrated, jobs, captured, fits, scores,
                   labels, published>>

Publish ==
    /\ phase = "evaluating"
    /\ integrity /\ frozen /\ calibrated /\ jobs = 2 /\ captured = Pairs
    /\ \A row \in Rows : fits[row] = 1 /\ scores[row] = 1 /\ labels[row] = "valid"
    /\ phase' = "complete"
    /\ published' = TRUE
    /\ UNCHANGED <<integrity, frozen, calibrated, jobs, captured, fits,
                   scores, labels, failureReason>>

Next == Freeze \/ Prepare \/ FailPreparation \/ Calibrate \/ FailCalibration
        \/ StartSampling \/ SealKernels \/ FailCapture \/ Invalidate \/ Publish
        \/ (\E pair \in Pairs : CapturePair(pair))
        \/ (\E row \in Rows : FitRow(row) \/ ScoreRow(row) \/ FailRow(row))

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ phase \in Phases
    /\ integrity \in BOOLEAN /\ frozen \in BOOLEAN /\ calibrated \in BOOLEAN
    /\ jobs \in 0..2 /\ captured \subseteq Pairs
    /\ fits \in [Rows -> {0, 1}] /\ scores \in [Rows -> {0, 1}]
    /\ labels \in [Rows -> {"pending", "valid", "failed"}]
    /\ published \in BOOLEAN /\ failureReason \in Reasons

FrozenBeforeWork == (jobs > 0 \/ (\E row \in Rows : fits[row] > 0)) => frozen
ScoreRequiresFit == \A row \in Rows : scores[row] <= fits[row]
ScoreRequiresCapture == (\E row \in Rows : scores[row] > 0) => captured = Pairs
ValidLabelRequiresScore == \A row \in Rows : (labels[row] = "valid") => scores[row] = 1
IntegrityFailureStops == ~integrity => phase = "failed" /\ ~published
CalibrationBeforeMatrix == jobs = 2 => calibrated
CompleteCoverage == published =>
    /\ phase = "complete" /\ frozen /\ integrity /\ calibrated
    /\ jobs = 2 /\ captured = Pairs
    /\ \A row \in Rows : fits[row] = 1 /\ scores[row] = 1 /\ labels[row] = "valid"
FailureCannotPublish == phase = "failed" => ~published

=============================================================================