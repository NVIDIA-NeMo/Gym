# DeepAgents sandbox evaluation, October 30, 2026

This record describes an interrupted evaluation and its qualification limits.
It is not a completed TB2.1 benchmark or a claim that the adapter is ready to merge.

## Revisions and protocol

- Benchmark runtime: `052b2cfda2a5128a18352f1d1df54f95e7c5ff78`.
- Separate observer fix: `6d5e3186d05836fb80b5ee18f6f769a8f881aa32`.
- DeepAgents: 0.6.12; task revision: `7131e4375048a0e408a8fb404b5f499d726b695b`.
- Planned coverage: 89 tasks, eight independent repeats, 712 identities.
- Sampling: temperature 1, top-p 0.95, context 262144, maximum output 32768 per call.
- Original task prompts, verifier and per-task budgets were retained. No failed,
  graded or interrupted episode was retried. No synthetic fixture is benchmark evidence.

## Collection and accounting

The original model allocation was preempted on October 28. Its collector exited
130 after interruption. A separate allocation was restricted to 155 identities
proven never started in the original run; that allocation was also preempted on
October 30 and its collector exited 130.

| Identity category | Count | Grade availability |
| --- | ---: | --- |
| Original pass | 306 | Valid, unmasked grade |
| Original verifier zero | 246 | Valid, unmasked grade |
| Original OOM | 1 | Unavailable |
| Original started and interrupted | 4 | Unavailable |
| Continuation sandbox-seed failure | 33 | Unavailable; failure records are nonterminal |
| Continuation missing record | 122 | Unavailable; dispatch status unknown |
| Total | 712 | 552 graded, 160 unavailable |

Both controller-finished flags remain false. Collection closure and complete
identity accounting do not imply complete grade coverage. The 122 missing
continuation records must not be relabeled never-started or eligible for retry.
The continuation produced no graded rollout or captured model call.

## Results with grade coverage

The final accounting receipt covers all 712 identities. The comparison uses
repeat 0 against historical single-repeat runs; eight-repeat results are separate.

| Harness | Passed / 89 | Observed pass yield | Graded / 89 | Unavailable |
| --- | ---: | ---: | ---: | ---: |
| DeepAgents repeat 0 | 38/89 | 42.70% | 68/89 | 21 |
| Copilot | 53/89 | 59.55% | 88/89 | 1 |
| OpenCode | 50/89 | 56.18% | 89/89 | 0 |
| OpenClaw | 49/89 | 55.06% | 86/89 | 3 |
| Codex | 48/89 | 53.93% | 84/89 | 5 |
| Hermes | 47/89 | 52.81% | 87/89 | 2 |
| Pi | 40/89 | 44.94% | 88/89 | 1 |

Across eight repeats, the mean observed pass yield is **306/712 = 42.98%**,
with **552/712 = 77.53%** grade coverage. The graded-only rate is **306/552 =
55.43%**, conditional on the subset that completed; it is not an unbiased full
benchmark estimate. Missing grades are not valid zeros. The possible full-grid
pass fraction lies between 42.98% and 65.45% if all missing grades respectively
fail or pass; this is a missing-data bound, not a confidence interval.

Repeat 0 has 38 passes, 30 valid zeros and 21 unavailable grades. On the 68 tasks
graded by both DeepAgents and Copilot, DeepAgents passes 38 and Copilot 42:
34 shared passes, four DeepAgents-only passes, eight Copilot-only passes and
22 shared zeros. This selected subset does not resolve missing-data bias.

The historical runs share the model checkpoint and benchmark revision but differ
in source, infrastructure, harness behavior and scheduling. They are not a
controlled causal ablation or a defensible ranking from these incomplete results.
Historical OpenCode and OpenClaw each contain one known reference-exposed pass;
neither historical run is certified contamination-free.

## Qualification limits

The original source passed 71 adapter tests, seven controlled conformance
scenarios, a real `regex-log` smoke, and a configuration-only SWE-bench Pro NodeBB
pairing. The latter received a valid zero and passed health, evidence and profile
checks. Its brief overlap on the original model allocation is a scheduling
confound for comparisons. Original graded-row health is 467 healthy and 85
unhealthy (63 generation-limit outcomes and 22 hollow turns); three task-budget
model cancellations are a subset of those hollow turns. Strict saved evidence
fulfilled 493 rows and failed 59. Positive grades with failed evidence are retained.

The observer fix passed 74 adapter tests and the same seven controlled scenarios.
Its fresh real-model smoke failed at sandbox seed: zero rollouts and captures,
collector exit 1, evidence exit 2 and strict profile exit 1. Health exit 0 checked
zero rows. It therefore has no passing fresh real-model qualification, and the
original source's scores do not transfer to it.

ReasoningGym's missing completion field, Partner IF's verifier-judge ownership
failure and the shared typed-trajectory identity mismatch remain documented gaps.
The full-evaluation, saved-evidence and benchmark-owner review gates remain open;
no supported-pairing catalog or `allowed_agents` declaration is added by this record.

## Evidence handling and cleanup

Failed completion assertions and their original output directories are retained.
The later accounting audit changes analysis validation only; strict health,
evidence and profile commands remain unchanged. It must retain all unavailable
categories and source provenance rather than fabricate completion markers.

Task-owned model jobs have ended. Cleanup receipts record 51 continuation and two
observer-smoke sandbox deletions, and successful follow-up cleanup of the original
run. A read-only provider inventory at 18:19 UTC on October 30 found no remaining
sandboxes in this task's 25 exact run/user scopes. Private task credential files
were removed; the workstation original remains in place. Other owners' jobs,
branches and scopes were preserved.

CPU-only accounting audit 8478019 finished at 18:23:30 UTC on October 30 with
exit 1. It wrote the final accounting receipt, then retained the failed strict
evidence check (exit 1) and full profile (exit 1: 160 missing results). Health exit
0 reports findings rather than certifying a healthy run. A separately labeled
partial profile passed with 552/712 coverage; 68 tasks have eight results, two
have partial results and 19 have no graded results. Run-level failure fractions
are unobserved because their thresholds are null, and the task-consistency check
has one unobserved task. No completion marker was fabricated.

The comparison ran on that actual final receipt and checked all task/repeat
identities, source provenance and grade totals. Its output directory was created
exclusively. Original audit failures and synthetic script fixtures remain separate.
The evaluation owner retains raw rollouts, captures and all command logs; the
following hashes identify compact receipts, not a rerun of the benchmark:

- `segmented-final-summary.json`: SHA-256 `8b4da6254bc5746c4fc55d026972441796e2aacb63d63d689e5991c9df7a6770`.
- `comparison-summary.json`: SHA-256 `c85222d4cfa4ff2c8c54674c12edc497015e13208c2d3feab771446bd088968b`.
- Accounting script: SHA-256 `09d072aa93c187e615dfd38a3745761db4817068f5a0f7dd5166873cef1956d7`.
- Comparison script: SHA-256 `c37c186aabdf5803a6d84220b4b14637ec7473e44a6781f9e2a12246dc73ebc9`.

This documentation changes no runtime, benchmark or conformance checks. Runtime
snapshots remain frozen; current-main compatibility and fresh qualification are
future integration gates, not results established by this documentation change.
