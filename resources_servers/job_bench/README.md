# JobBench resources server

Runs JobBench tasks in a sandbox and scores deliverables with an upstream-derived
weighted-rubric judge. Input preparation uses the modified protocol below.

## Bounded grading inputs

`job-bounded-utf8-v1` retains the upstream rubric prompts and weighted scoring,
but changes input preparation. It is not exact published-leaderboard reproduction.
The original output archive remains intact. The grading view excludes directory
components named `venv`, `.venv`, `node_modules`, `.git`, `.cache`, or `__pycache__`,
including their images. All other deliverables remain eligible.

After upstream conversion and its existing per-file truncation, the combined text
(including relative filenames and omission markers) is capped at 120,000 UTF-8
bytes. Short files remain complete; the remaining budget is shared equally among
longer files. Long files retain equal-sized beginning/end excerpts with a visible
middle-omission marker. Files appear in sorted relative-path order. SQLite text
also participates in this total budget. Too many filenames to fit is an explicit
grading error, never a fabricated zero. The upstream eight-image cap is unchanged.

Use this same version for every compared harness and trial. When migrating an
existing run, regrade all archived Job outputs into a separate results set; retain
the original grades and never rerun policies to implement the grading change.

With artifact retention enabled, `judge-receipt-*.json` preserves returned
per-rubric verdicts and diagnostic metadata, including on grading errors. Receipts
do not infer unreported attempts or turn failed grading into a zero. Preserve the
original receipts and declare separate provenance for any later judge-only run.

The request and frozen rubrics are saved before output collection, preserving
trial identity and the original agent error when collection fails. Without a
complete output archive, these diagnostic files do not establish a valid grade.

Report native execution status separately from artifact grading: a budget-ended
policy can still leave judgeable deliverables. Missing grades are not policy
zeros. The pinned upstream evaluator defines weighted scores per task, but not a
cross-task avg@4 aggregate. Label task-macro and rubric-weight-pooled summaries
explicitly; do not claim either reproduces a published score without matching
the aggregation and evaluation protocol. Reconcile all planned task/repeat keys
before presenting a full-suite result.
