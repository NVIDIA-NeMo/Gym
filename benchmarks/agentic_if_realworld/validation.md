# Validation and privacy

The implementation has been exercised with independently constructed synthetic
tasks. Real traces, per-item criteria, original policy documents, historical
outputs, registry access instructions and raw receipts remain in approved private
systems. The public repository contains no real benchmark examples.

## Initial public export checks

A clean installation for the initial public export passed 370 scoped tests:
125 Resources Server/scorer tests, two model-adapter tests and 243 focused core
checks. Both exact public component CI commands also pass with private registry
and inference credentials removed. Private replay verifies unchanged inputs,
original prompt bytes and zero saved-grade mismatches; it performs no new inference.
Final CLI and privacy receipts are recorded in the PR separately from platform CI.

## Prior implementation evidence

The following checks were recorded before preparing this public export. They are
historical evidence for the implementation, rather than results from the final
public commit. Final export checks must be recorded separately in the PR.

| Check | Recorded result |
| --- | --- |
| Component tests | 123: 121 Resources Server/scorer and 2 model-adapter cases |
| Focused Gym core tests | 243; 366 scoped tests total with the components |
| Manifest verifier | Nine independent fixture cases |
| Preparation, validation and local publication | Passed with synthetic data |
| Public CI component/data validation | Five distinct synthetic tasks, recorded rollouts and metrics satisfy the legacy gate |
| Synthetic task accounting | Five complete tasks: 3 pass / 2 fail; health and P0 checks passed |
| Adaptive recovery | First complete answer retained with its actual seed/token cap; incomplete attempts stay in private receipts |
| Cache integrity and resume | Changed policy/judge bytes, request, payload, schema or grade are rejected; missing successful cache cannot dispatch |
| Private judge destinations | Native and adaptive scoring reject destinations inside Git |
| Resolved configuration | Adaptive config saved atomically, redacted and private; rejected resume preserves original evidence |
| Private preparation and historical replay | Exact inputs preserved; saved grades replayed with zero mismatches |
| Private registry | Original artifact hashes and denial of anonymous reads verified |
| Pre-commit | All-files checks passed; private guardword catalog unavailable |

The public CI examples were generated through the standard agent, production
verifier, local synthetic policy/judge HTTP endpoint and Gym collation. They
contain independent literal tasks. Tests check their exact inputs, unique
identities, pass/fail rewards, provider metadata, P0 evidence and portable metrics
paths. See the [regeneration instructions](../../resources_servers/agentic_if_realworld/README.md#public-ci-examples).

Recovery tests cover later delivery attempts, exhaustion, failed `/run`, masked
outcomes, stable resume and corruption rejection without new model calls. Live
synthetic checks previously exercised an approved inference provider's policy and
judge endpoints, metadata, capture, health, P0 and resume. These transport checks
do not calibrate judge quality or establish a benchmark reference score.

## Historical real-task reward profile

The October 26 review follow-up replayed all **1,938 completed historical results**
against the current frozen validator and Resources Server aggregation hook. It
found **zero grade mismatches**, zero masked results and identical view/subtype
metrics. These are saved historical rewards, not newly generated Gym rollouts or
a reference score for the current GPT-6.1 Sol judge.

The cohort is the frozen, failure-conditioned release: 912 original cases and
57 supplemental cases, all development-exposed and evaluation-only. Each frozen
task contributes exactly one saved, scored result per model. Coverage is 100% in
each view; masked and unattempted counts are zero. Reward is binary, so mean
reward equals complete accuracy here. Subtype denominators overlap.

| Historical policy | Scoring view | Declared / scored | Reward 1 (pass) | Reward 0 (fail) | Mean reward |
| --- | --- | ---: | ---: | ---: | ---: |
| GLM-5.3 | Original failure focused | 912 / 912 | 495 | 417 | 0.542763 |
| GLM-5.3 | Supplemental active single turn | 57 / 57 | 9 | 48 | 0.157895 |
| GLM-5.3 | Labeled composite | 969 / 969 | 504 | 465 | 0.520124 |
| Nemotron-3.5-Super-Text-Preview | Original failure focused | 912 / 912 | 397 | 515 | 0.435307 |
| Nemotron-3.5-Super-Text-Preview | Supplemental active single turn | 57 / 57 | 7 | 50 | 0.122807 |
| Nemotron-3.5-Super-Text-Preview | Labeled composite | 969 / 969 | 404 | 565 | 0.416925 |

The original policies were `nvidia/zai-org/glm-5.3` and
`nvidia/nvidia/nemotron-3.5-super-text-preview`, temperature 0.7, initially seed 17
with a 32,768-token cap. The historical judge was **GPT-6 Astra, medium reasoning**,
using `agentic-if-generated-response-judge-v2.2-strict`. Original receipts and
manifest bindings were retained; there were no new generation or judge calls.

Historical recovery used more waves than either current adaptive preset. The
saved GLM answers used 960 / 8 / 1 deliveries at 32,768 / 65,536 / 131,072 tokens,
all seed 17. Super used 891 / 52 / 26 at those caps; its 131,072-token deliveries
used seeds 17 (13), 29 (7), 43 (3), 71 (1), 401 (1) and 509 (1). This historical
profile is neither `adaptive_first_complete_v1` nor
`adaptive_first_complete_max_effort_v1`, and is not the stock native profile.

The replay checks exact actor inputs and criteria, original manifest and response
bindings, completed delivery/SSE reconstruction, saved-review schema/evidence
validation (including four bound transport schemas), and unchanged strict-v2.2
grades. Passing those validated grades through the current Resources Server
additionally checks masking, full-cohort accounting, both views and overlapping
subtype metrics. Mechanical checks inspected these contracts; no individual
real example informed a code or rubric change. The receipt-index SHA-256 is
`ff50f5b152a654b0481b98d5081560a7375664ee9589d43bc8d9a8723858e177`.

This replay does **not** exercise new live policy/judge calls, the adaptive
generation scheduler, provider reasoning effort or cluster deployment. Independent
synthetic rollouts and recovery tests cover those local integration paths. A
current-model platform reference score and judge calibration remain separate.

## Frozen code provenance

The validator and companion test are NVIDIA-authored Agentic IF code released
under **Apache-2.0**, not third-party vendored code. Their authoring-workspace
files were uncommitted; no original authoring Git revision is claimed. Their
first tracked package revision is `df90e68292130bb21ffe8a023fc5d0a6e1be0878`.
The original filenames map to the packaged files as follows:

| Original filename | Packaged file |
| --- | --- |
| `run_scripts/validate_agentic_if_response_judge_v2_2.py` | [validate_response_judge.py](../../resources_servers/agentic_if_realworld/benchmark/policy/validate_response_judge.py) |
| `tests/test_agentic_if_response_judge_v2_2.py` | [test_contract.py](../../resources_servers/agentic_if_realworld/benchmark/policy/test_contract.py) |

Both headers carry that origin, first tracked revision, original filename and
license. No fork archive is needed to establish authorship. NVIDIA copyright and
Apache notices are retained. Original
body SHA-256 values remain
`67e67f1ebef6594e6ff1fbbe2f84af51b0f0a5f57d5bf1d22d0e3467dded598a`
and `312f7b61d0a8a1328e9dc180c25da1b196484c939671a9dcf625a7f26fd9dc8f`,
respectively. `frozen_policy_bytes()` accepts only the explicitly defined
packaging header before checking those unchanged hashes. No private source URL,
tasks, criteria or policy documents are included in this provenance record.

## Public export boundaries

Real preparation downloads from the approved authenticated private registry.
The endpoint is supplied externally and checked against a fixed URL-hash pin
before any request. Original artifact hashes remain unchanged. The portable
validator has source provenance separate from public packaging, and real runs
load the unchanged private judge prompt. A separately identified public synthetic
prompt supports checks without registry credentials or paid inference.

Prepared inputs, captures, run outputs and judge receipts stay outside Git.
Public CI only needs generic code and the three independently synthetic example
artifacts; it does not need private data or inference credentials.

## Remaining platform gates

Run the [documented component CI and local mock smoke](README.md#synthetic-smoke-without-registry-credentials-or-paid-inference)
against the final public commit. Standard GitHub CI may require qualified
maintainer approval for that exact commit; see [Gym's CI instructions](../../CONTRIBUTING.md#continuous-integration-ci).

Gym maintainers must confirm the recorded streaming adapter, adaptive driver,
metadata contract and applicable conformance checks. Platform owners must then
provide the execution environment, run a bounded smoke, measure runtime and
agree a reference score, tolerance, coverage and health gates. Deployment
configuration should pin the approved merged Gym revision.

Public synthetic checks do not establish cluster readiness, a complete paid
benchmark result, training authorization or certification.
