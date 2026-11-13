# Validation and privacy

The implementation has been exercised with independently constructed synthetic
tasks and the bounded real-task run below. Real traces, per-item criteria, original policy documents, historical
outputs, registry access instructions and raw receipts remain in approved private
systems. The public repository contains no real benchmark examples.

## Observed resume and connection recovery

The HTTP integration regressions reproduce both completed-resume rejection from
fresh verifier observations and capture reuse after a connection failure. With
the fixes, all **263 component tests pass**. Coverage includes observability on
and off, one or two connection failures, later schedule phases, pass/fail answers,
unchanged earlier receipts/captures, one-row inventory accounting, P0 ownership
checks, and rejection of changed grades, responses and cache bindings.

A one-task synthetic `gym eval run` used GLM-5.3/max and GPT-6.1 Sol/medium with
observability enabled. Both requests returned HTTP 200 on their first attempt.
The answer omitted required punctuation and was correctly graded fail, with no
masked result. Preparation, environment validation, health and no-tools P0 checks
passed. Completed resume made zero model calls and preserved the saved result,
receipts, captures and verifier observations. This is integration evidence, not
a benchmark score or judge calibration; no private benchmark tasks were sent.
All-files pre-commit and the secrets scan passed; the private guardword catalog
was unavailable locally. Hosted CI remains a separate gate for the new commit.

## October 28 review validation

Runtime `4d5e18b55ffccd0444dd6cbdb03369be9bd90d03` passed 248 focused
tests (238 Resources Server/scorer, 5 model adapter and 5 shared relay tests).
The Resources Server CI entrypoint passed its 238 tests and synthetic data
validation; model-component CI passed 5 tests. All-files pre-commit passed,
except the unavailable private guardword catalog; a separate private/public
content audit passed. Shared relay behavior and global lint configuration are
unchanged.

The mock `gym eval run` workflow passed preparation, environment validation,
incomplete-delivery recovery, judging, resume, health and no-tools P0 checks.
Standalone `gym eval aggregate` through `native.yaml` reproduced the complete
aggregation artifact without model calls. A synthetic quote/newline request to
the actual judge returned HTTP 200 and a valid pass.

### Provider compatibility and saved-answer recovery

An initial bounded run at `b58db4abc33dcd02e38d9648575b8d3113c69d47`
stopped after four complete policy answers: two scored pass, two judge requests
returned HTTP 400, and 46 selected tasks were not admitted. The provider rejected
quote/newline literals in `support_quote.enum`. All original receipts remain
unchanged and are separate from the new run.

Transport v2 omits only that enum from the provider request. The full bounded
schema still enforces the exact support quote locally, alongside the unchanged
frozen strict-v2.2 validator and rubric. Both schemas are saved and checked in
a separately versioned cache. Recovery of the two saved answers made zero new
policy calls: one scored fail and one remained masked with
`judge_delivery_or_validation_error` after two reviews. All three judge HTTP
attempts returned 200; original answer hashes and receipts were unchanged.

### Bounded current-model run

The deterministic first 50 prepared task IDs ran with GLM-5.3 (`max` policy
reasoning), GPT-6.1 Sol (`medium` judging), temperature 0.7 and
`adaptive_first_complete_max_effort_v1`, at concurrency 4. Selection did not use
outcomes. All 50 retained the first complete answer at seed 17 / 32,768 tokens.
The frozen input, criteria and policy hashes were unchanged.

**50 scored: 33 pass / 17 fail; 0 masked; 0 selected tasks unsettled.**
Mask reasons: none. Physical judge HTTP attempts: **51 × 200 and 1 × 429**;
the rate-limited attempt recovered through the bounded retry path. One answer
needed a second delivered review before scoring. Policy HTTP attempts were
50 × 200 and 1 × 429, also recovered. These transport retries do not change the
seed/token-limit stage.

| View | Declared | Scored | Pass | Fail | Masked | Unattempted | Scoring coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original failure focused | 912 | 47 | 33 | 14 | 0 | 865 | 5.1535% |
| Supplemental active single turn | 57 | 3 | 0 | 3 | 0 | 54 | 5.2632% |
| Labeled composite | 969 | 50 | 33 | 17 | 0 | 919 | 5.1600% |

`complete_accuracy` remains null for both views and the composite. Subtype
results below count subtype verdicts; their task denominators overlap.

| Subtype | Declared | Scored | Pass | Fail | Masked | Unattempted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IF-CITE | 43 | 2 | 2 | 0 | 0 | 41 |
| IF-COVERAGE | 119 | 4 | 4 | 0 | 0 | 115 |
| IF-FORMAT | 237 | 15 | 13 | 2 | 0 | 222 |
| IF-LANG | 98 | 3 | 3 | 0 | 0 | 95 |
| IF-LENGTH | 282 | 13 | 4 | 9 | 0 | 269 |
| IF-NEWINSTR | 108 | 17 | 16 | 1 | 0 | 91 |
| IF-PLAN | 17 | 1 | 0 | 1 | 0 | 16 |
| IF-SCOPE | 351 | 12 | 5 | 7 | 0 | 339 |
| IF-TOOLPREF | 155 | 0 | 0 | 0 | 0 | 155 |

Health reported 50 healthy, 0 unhealthy and 0 unobserved tasks. The retained-artifact
P0 checker completed with `fulfilled` for all 50 no-tools records; this is not
behavioral qualification. A completed-run `--resume` made zero new policy or judge calls; all 555 saved
receipt/capture files and all selected-response hashes were unchanged.

The private receipt inventory contains 504 artifacts with SHA-256
`2d35c6e6ed51ca1ddc589c12325e7fb5ec302447fadc98025458c476916e72ff`.
The runtime uses Python 3.13.14, nemo-gym 0.7.0rc0, aiohttp 3.14.4,
jsonschema 4.23.0, openai 2.44.0 and pydantic 2.13.4.


### Privacy and interpretation

All 969 tasks remain development-exposed and evaluation-only. Mechanical checks
verified unchanged wrapper text, embedded tool registries, criteria bindings and
private artifact permissions. Real traces, criteria, model outputs and raw
receipts remain in approved private storage; no real example or review rationale
was inspected to change the rubric or implementation. The public history audit
found no private task IDs, wrappers, long source spans or complete artifacts, and
anonymous access to all 28 private registry artifacts was denied.

This bounded integration check does not calibrate judge quality, reproduce the
historical recovery waves, establish a full reference score or certify cluster
readiness. Policy `max` and judge `medium` effort were verified in request bytes;
provider-internal reasoning behavior was not independently measured.

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

## Evidence-ID transport

The `evidence_ids_v1` transport uses instruction quote IDs and exact response
span IDs, which are resolved before unchanged strict-v2.2 validation. The
full rubric and evaluation payload remain present. Historical quote transport
and its cached grades remain available under `quotes`; transport variants use
different request identities and must retain their provenance.

- 258 synthetic component/model tests pass, including canonical round-trip,
  cached replay, wrong-field/foreign-ID rejection, contradictory verdict rejection,
  Unicode span coverage and null omission evidence.
- A live GPT-6.1 Sol/medium judge scored an independent literal-answer mismatch
  correctly as fail. Cached replay returned the identical grade without dispatch.
- All-files pre-commit passes; the private guardword catalog was unavailable.
- The separate full CLI GLM policy-generation smoke hit HTTP 429. This does not
  establish end-to-end live generation coverage for the new transport.
- Frozen policy, source manifest, private inputs and prior grades are unchanged.
  No real benchmark examples were sent by these checks. Synthetic plumbing
  evidence does not establish agreement with Astra or a benchmark accuracy gain.

### Interim default restoration

At `f35546de`, the shipped default was restored to `quotes`; `evidence_ids_v1` remained opt-in.
[Yu's request](https://github.com/NVIDIA-NeMo/Gym/pull/5013#discussion_r4407413874)
requires bounded real-task evidence through the judge request being shipped.
The earlier 50-task run validates the quote transport, not the new evidence-ID
request. Promotion was deferred pending a bounded adaptive run with the approved
policy and judge, including revision, cohort, scored/pass/fail/masked counts, mask
reasons and judge HTTP status counts. The new evidence below is available for reviewer confirmation.
Earlier validation receipts remain bound to their original revisions and modes.
The default-restoration check passes all 258 component/model tests. No model
calls or real benchmark rows were needed to restore this reviewed transport.

## Bounded evidence-ID validation

A bounded real-task run validated `evidence_ids_v1` before default activation.
`gym eval run --benchmark agentic_if_realworld --split benchmark --limit 50
--num-repeats 1 --concurrency 4` completed at
`f35546de22be354e757a5781a950967c81885794`, explicitly selecting that transport.
The default-selection change uses the same runtime implementation and effective
judge request. The policy was `nvidia/zai-org/glm-5.3` with maximum effort; the
judge was `openai/openai/gpt-6.1-sol` with medium effort, both through the approved
Inference API. The profile was `adaptive_first_complete_max_effort_v1`.

The cohort was the first 50 task IDs in frozen deterministic preparation order,
selected without inspecting outcomes. Full inputs, criteria, split/exposure roles
and frozen strict-v2.2 policy remained unchanged. No rubric tuning or individual
outcome inspection was performed.

| View | Scored / full cohort | Pass | Fail | Attempted masked | Unattempted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original failure focused | 47 / 912 | 33 | 14 | 0 | 865 |
| Supplemental active single turn | 3 / 57 | 0 | 3 | 0 | 54 |
| Composite | 50 / 969 | 33 | 17 | 0 | 919 |

- Judge: **50 HTTP 200**, one delivered attempt per task; **zero validation
  failures, HTTP errors, retries or unresolved outcomes**. Mask reasons are empty.
- Policy: **244 physical HTTP attempts: 50 HTTP 200 and 194 HTTP 429**.
  There were 98 logical delivery receipts and 146 transport retries within them.
  The schedule entered phases 1/2/3/4/5 for 50/31/13/4/0 tasks respectively.
  Capacity errors recovered under the existing bounded protocol; all 50 first
  complete answers were retained, including failing answers.
- Independent offline replay verified exact instruction/response evidence,
  request/raw-body hashes, unchanged canonical validation, first-complete
  selection, the two-delivery judge budget and agreement with published grades.
- Completed resume made **zero new model calls**. All **940 immutable receipt,
  capture, configuration and materialized-input files** retained their hashes;
  saved answers and per-task results were unchanged.
- Preparation and environment validation passed. Health reported **50 healthy,
  zero unhealthy or unobserved**; P0 evidence was fulfilled before and after resume.
- Judge usage reported 2,610,026 prompt tokens and 50,806 completion tokens
  (2,660,832 total). This is usage accounting, not a controlled cost comparison.

These checks support enabling evidence IDs in the shipped public profile and its
EFB bindings. Review of the updated default, hosted CI, canonical merge/pin and
platform certification remain separate gates. The older internal Gym alternative
was not used in this run and is not validated by these results.

Full-cohort coverage is **50/969 (5.16%)**; `complete_accuracy` remains null.
Subtype denominators overlap and are preserved in the private aggregate report.
The equal overall pass/fail counts of an earlier quote run are not evidence of
per-task agreement: this run generated new candidate answers. This check does not
establish better judgment accuracy, causal reduction in unknowns, total cost
savings or a new full-benchmark reference score. Real traces and receipts remain
private; only aggregate evidence is published here.
