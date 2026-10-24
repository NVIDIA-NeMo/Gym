# Validation and privacy

The implementation has been exercised with independently constructed synthetic
tasks. Real traces, per-item criteria, original policy documents, historical
outputs, registry access instructions and raw receipts remain in approved private
systems. The public repository contains no real benchmark examples.

## Public export checks

A clean installation on the public upstream base passes 370 scoped tests:
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
