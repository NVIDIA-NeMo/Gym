# DeepAgents missing-grade recovery, November 2, 2026

This record closes the first authorized recovery of missing grades. It preserves
the [October 30 first-attempt report](evaluation-2026-10-30.md) and does not
establish a complete TB2.1 or Observability qualification. A separate collection
of the three remaining ungraded identities is in progress and is not included
in the closed results below.

## Source and selection

Runtime remains `052b2cfda2a5128a18352f1d1df54f95e7c5ff78`, DeepAgents 0.6.12,
and task revision `7131e4375048a0e408a8fb404b5f499d726b695b`. All 8024 frozen source
files were reverified. Original prompts, input rows, verifier, resource and time
budgets, model checkpoint, sampling and per-group concurrency were retained.
The real cache loader selected exactly 160 identities without expanding the
eight repeats again or overlapping any of the 552 existing grades.

On November 1 the evaluation owner authorized new attempts only for identities
without grades. Every original grade, including a valid zero, was retained.
The first recovery made one new attempt per eligible identity, producing 157
new grades (93 passes, 64 zeros) and three failures without grades. This is
recovery from unavailable results, not best-of-eight or a retry of valid zeros.
New allocations, endpoints and execution dates are comparison differences.

Prior-attempt provenance retains the original OOM, four interrupted episodes,
33 continuation seed failures and 122 old dispatch-unknown identities. New
results do not prove what was dispatched in the old continuation. Both original
controller-finished flags remain false; no original completion marker is created.

## Results and historical comparison

| Repeat-0 run | Passes / 89 | Valid zeros | Graded / 89 | Unavailable |
| --- | ---: | ---: | ---: | ---: |
| Original DeepAgents first attempt | 38 | 30 | 68 | 21 |
| DeepAgents after first recovery | 51 | 38 | 89 | 0 |
| Historical Copilot | 53 | 35 | 88 | 1 |
| Historical OpenCode | 50 | 39 | 89 | 0 |
| Historical OpenClaw | 49 | 37 | 86 | 3 |
| Historical Codex | 48 | 36 | 84 | 5 |
| Historical Hermes | 47 | 40 | 87 | 2 |
| Historical Pi | 40 | 48 | 88 | 1 |

Recovered repeat 0 is complete: **51/89 = 57.30%**. These historical runs share the
checkpoint and task revision but differ in harness/source/runtime, infrastructure
and scheduling. They are not controlled causal comparisons. OpenCode and
OpenClaw each include one known reference-exposed pass. On the 88 tasks graded
by both recovered DeepAgents and Copilot, 43 pass both, seven pass only DeepAgents,
ten pass only Copilot and 28 pass neither.

The separate eight-repeat aggregate has **399 passes, 310 valid zeros and three
unavailable grades**, with **709/712 = 99.58%** grade coverage. Repeat pass counts
are `[51, 50, 49, 47, 52, 49, 49, 52]`; grade counts are
`[89, 89, 89, 88, 89, 88, 88, 89]`. Mean observed passes are 49.875 per 89-task
repeat. The all-712 observed-pass fraction, **56.04%**, is a lower bound while
three grades are missing; its all-missing-pass upper bound is **56.46%**. These
are coverage bounds, not confidence intervals. The graded-only rate is
**399/709 = 56.28%**, conditional on the available subset. Missing grades are
never converted into valid zeros.

The three remaining identities are 56/5, 67/6 (`qemu-alpine-ssh`) and 68/3
(`qemu-startup`), all agent-stage HTTP 410 failures. Sandbox OOM/exit 137 was
observed; compact identity-to-sandbox associations for 67/6 and 68/3 are inferred
from timing, not directly proven by a saved mapping. A separate recovery may
attempt only these still-ungraded identities, excluding all 709 current grades.

## Strict checks and source-scoped qualification

The aggregate audit wrote its real final receipt and exited 1, preserving the
following unchanged strict commands:

- Health exited 0 with 709 records/captures: 603 healthy and 106 unhealthy.
  Findings include 78 generation-limit outcomes and 28 hollow turns; four
  ended-on-model-error records overlap these categories. Three tasks were
  consistently unhealthy. All rollout-level checks were observed, while the
  run-level excessive-failure check was unobserved because thresholds were unset.
  Command success does not certify every rollout healthy.
- Strict evidence exited 1: 634 records fulfilled the checks and 75 did not.
  Sixteen new failures include passed-reward identities 67/1 and 84/0; the four
  original positive-reward evidence failures remain. Missing tool records,
  tool status/output and reference/ownership checks are retained as failures.
- Strict profiling exited 1 because exactly three grades were missing. A separate
  partial-profile diagnostic exited 0 and does not replace the strict result.

Recovery changed no adapter behavior. Prior unit tests, controlled conformance,
real smoke and configuration-only second-pairing results remain scoped to their
original tested revisions; none was rerun by this documentation change.
ReasoningGym completion-field, Partner IF judge-ownership and typed-trajectory
identity gaps remain. No score on 052 qualifies the separate observer fix
`6d5e3186d05836fb80b5ee18f6f769a8f881aa32`, whose fresh real-model smoke failed at
sandbox seed before inference. Full-evaluation, saved-evidence and owner-review
gates remain open; no pairing catalog or `allowed_agents` claim is added.

## Receipts and cleanup

The original 552-grade summary and comparison remain immutable. The first
recovery wrote a separately named comparison only after its real final receipt
was available. Raw rollouts and captures remain with the evaluation owner;
compact receipts identify the report:

- Original `segmented-final-summary.json`: SHA-256
  `8b4da6254bc5746c4fc55d026972441796e2aacb63d63d689e5991c9df7a6770`.
- First recovery `recovery-final-summary.json`: SHA-256
  `05d2ac7a31bba97fcbfd7a67e4c03212c5853a0997c24b0f1cddca0086315a62`.
- First recovery comparison summary: SHA-256 `0ddaea701d559655d7f0019e421e0e7f07931704b8dd33b851d69a6e1b5256f1`.

Independent cleanup verified all three exact first-recovery run/user scopes
empty, cancelled only its owned model allocation and removed private run keys.
All first-recovery jobs ended. A subsequent metadata check preserved the
workstation original credential with unchanged size, mode and mtime; its contents
were not read for that check. These are point-in-time observations. The separate
remaining-three recovery has its own cleanup and credential lifecycle.

No synthetic fixture is benchmark evidence. Strict gate failures, missing grades
and source differences remain explicit; this documentation changes no runtime,
benchmark or conformance checks.
