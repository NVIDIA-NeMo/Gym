# DeepAgents completed grade recovery, November 2, 2026

The authorized missing-grade recovery now has **712/712 valid, unmasked grades**:
**400 passes, 312 verifier zeros, no unavailable grades**. This completes grade
coverage for 89 tasks with eight independent repeat identities. The result combines
retained original grades with separately authorized recovery attempts; it is not
a completed original first-attempt run or best-of-eight selection. Saved-evidence
and other qualification failures remain explicit.

## Source and attempt provenance

Runtime is frozen at `052b2cfda2a5128a18352f1d1df54f95e7c5ff78`, DeepAgents 0.6.12,
and task revision `7131e4375048a0e408a8fb404b5f499d726b695b`. Model checkpoint,
sampling, task prompts, verifier, resource limits and time budgets are unchanged.
Original input identities and all 8024 exported source files were checked before
the final recovery launch. Allocation, endpoint and execution date changed.

| Selected attempt | Valid grades | Passes | Zeros |
| --- | ---: | ---: | ---: |
| Original first attempt, retained unchanged | 552 | 306 | 246 |
| First missing-grade recovery, retained unchanged | 157 | 93 | 64 |
| Remaining-three recovery | 3 | 1 | 2 |
| Final selected aggregate | 712 | 400 | 312 |

The [original report](evaluation-2026-10-30.md) and
[first-recovery report](evaluation-recovery-2026-11-02.md) retain their own scores,
failures and coverage. The last recovery excluded all 709 grades, including every
zero, and attempted only 56/5, 67/6 and 68/3. Outcomes are `password-recovery` zero,
`qemu-alpine-ssh` zero, and `qemu-startup` pass. The qemu-alpine-ssh agent response
is incomplete, but verification completed with an unmasked reward of zero; its
agent/evidence failure is retained separately from the valid verifier grade.

All original interruptions, OOM and seed failures remain in prior-attempt
provenance. The 122 old dispatch-unknown identities are not retroactively labeled
never started. Both original controller-finished flags remain false; no original
completion marker was fabricated. Only missing grades were eligible for recovery,
and the first valid grade for each identity was retained thereafter.

## Repeat 0 and historical comparison

| Repeat-0 run | Passes / 89 | Valid zeros | Graded / 89 | Unavailable |
| --- | ---: | ---: | ---: | ---: |
| Original DeepAgents first attempt | 38 | 30 | 68 | 21 |
| DeepAgents after first recovery | 51 | 38 | 89 | 0 |
| DeepAgents after final recovery | 51 | 38 | 89 | 0 |
| Historical Copilot | 53 | 35 | 88 | 1 |
| Historical OpenCode | 50 | 39 | 89 | 0 |
| Historical OpenClaw | 49 | 37 | 86 | 3 |
| Historical Codex | 48 | 36 | 84 | 5 |
| Historical Hermes | 47 | 40 | 87 | 2 |
| Historical Pi | 40 | 48 | 88 | 1 |

Final recovered repeat 0 is **51/89 = 57.30%**, unchanged by the last three attempts,
which belong to repeats 3, 5 and 6. Historical runs share the checkpoint and task
revision but differ in harness/source/runtime, infrastructure and scheduling;
this is not a controlled causal ranking. Historical OpenCode and OpenClaw each
include one known reference-exposed pass. On the 88 tasks graded by both recovered
DeepAgents and Copilot, 43 pass both, seven pass only DeepAgents, ten pass only
Copilot and 28 pass neither.

## Eight-repeat aggregate

Every repeat now has 89 grades. Pass counts for repeats 0 through 7 are
`[51, 50, 49, 48, 52, 49, 49, 52]`. Their mean is **50/89 = 400/712 = 56.18%**,
with **100% grade coverage**. This mean is separate from repeat 0. The sample
standard deviation across repeat pass rates is **1.70 percentage points**; it is
not a confidence interval. No missing-grade bound is needed for this final
selected aggregate, and no unavailable grade was converted to zero.

## Qualification checks and limitations

The final read-only audit retained all 709 earlier grade records exactly, verified
source-input and prior-rollout hashes, and checked all 712 identities and nested
attempt provenance. Unchanged strict commands returned:

- Health exit 0: 712 records and captures, 605 healthy and 107 unhealthy. Issues
  include 78 generation-limit outcomes and 29 hollow turns; four ended-on-model-
  error records overlap those categories. Three tasks remain consistently
  unhealthy. All rollout-level checks were observed. The run-level excessive-
  failure check remains unobserved because its thresholds are unset. Command
  success does not mean every rollout is healthy.
- Strict evidence exit 1: **636 fulfilled, 76 not fulfilled**. The final recovery
  adds one failure on qemu-alpine-ssh 67/6: an unresolved turn/model-call reference
  blocks call-target, ownership and attempt-accounting checks. Its zero remains
  a valid verifier grade. Six older positive-reward evidence failures also remain.
- Strict full profile exit 0: **all 712 grades present**. No partial-profile flag
  was used for this successful full profile.

The overall audit exited 1 because saved evidence still fails; it wrote its final
receipt before exiting. This is a retained failed gate, not an audit crash to be
repaired. Unit/conformance results from earlier revisions were not rerun by this
docs-only update. ReasoningGym completion-field, Partner IF judge-ownership and
typed-trajectory identity gaps remain. No score on 052 qualifies observer fix
`6d5e3186d05836fb80b5ee18f6f769a8f881aa32`, whose real-model smoke failed at sandbox
seed before inference. Complete grade coverage does not establish complete
Observability qualification, benchmark-owner approval or merge readiness. No
supported-pairing catalog or `allowed_agents` claim is added.

## Receipts and cleanup

The original and first-recovery summaries/comparisons are unchanged. The actual
final receipt was copied before running `compare_remaining.py`, which validates
retained grades and writes a separate comparison directory exclusively.

- Final `recovery-final-summary.json`: SHA-256 `e2fff5965cebdb808e671c4ce9ecf999c3d9520dd20f623777c550581aad7354`.
- Final comparison summary: SHA-256 `1b109b99bdcfc5e7515708f435371a3a6b7bd74175ac0b672b7f4f93fff0bae5`.
- First-recovery summary remains SHA-256
  `05d2ac7a31bba97fcbfd7a67e4c03212c5853a0997c24b0f1cddca0086315a62`.
- Original summary remains SHA-256
  `8b4da6254bc5746c4fc55d026972441796e2aacb63d63d689e5991c9df7a6770`.

Independent cleanup verified only the final recovery's exact preflight and batch
run/user scopes empty and cancelled only its owned model. A later read-only check
found all task-owned jobs absent from the queue and all private model-key/sandbox-
key files absent across original, continuation, observer-smoke and both recovery
roots. The workstation original credential remains nonempty with unchanged size,
mode and mtime; its contents were not read for verification. Other owners' jobs,
scopes and branches were preserved. Raw rollouts, captures and all failed checks
remain with the evaluation owner; no synthetic fixture is benchmark evidence.
