# Recorded-context ordering evaluation

The `partner_if` replay profile continues supplied history against frozen tool data.
The public runner contains no customer source, prompt, service address or credential.
The authenticated registry supplies actor inputs, recorded responses and a validated
`runtime.json` declaring paths, aliases, asset hashes and probe identity.

The existing `single_agent_turn` Environment Server closes each fresh agent
session before shared verification. Its optional `verifier_continuation` loop follows
the shared verifier's next request and enforces a six-activation ceiling. The verifier stops
on the first decisive ordering result. A failed ordering requirement scores zero;
a returned identity-verified prerequisite check can establish a pass. A cutoff,
unsupported operation or infrastructure/judge error is masked. Supplied history is
context; no final answer or successful tool result is invented.

There is no separate replay Environment Server. Dataset-specific ordering decisions,
receipt checks and inconclusive-run exclusions remain in `partner_if`. The generic
environment owns sessions, bounded iteration, usage and trajectory aggregation.
Repeated-activation episodes restart after interruption; exported checkpoint restoration
for the existing single-activation mode is unchanged.

The dispatcher wraps the prepared flat row in a typed episode request for
`single_agent_turn` and converts its typed reply back to the collector's result
format. The published task rows, actor input and rubric remain the pinned data;
the request conversion carries the same task and attempt identities.

## Assets and configuration

Version 2 of the replay asset envelope adds `runtime.json`. Its other members keep
exactly the actor and tool bytes of the prior package. Preparation checks the outer
release digest, every member digest, and the contract's identities before writing
runtime files. Data files are never imported or executed.

The published `partner_if` 2.1.7 envelope is version 2 and bundles the runtime
manifest. Its task rows and seven prior asset members are byte-identical to
2.1.6. Normal registry preparation requires no external contract override.

The older `partner_if` 2.1.6 envelope is version 1 and remains unchanged. A run
using that envelope must supply an externally pinned runtime contract through
`replay.runtime_contract_path` and `replay.runtime_contract_sha256`, or select a
separately prepared version-2 asset package through `replay.package_dir`.
Do not replace a published release artifact in place.

Registry credentials belong in the ignored `env.yaml`. To select a prepared
local package, set `replay_package_dir` in a private override configuration.
To pair the unchanged version-1 assets with an external contract, set
`replay_runtime_contract_path` and `replay_runtime_contract_sha256` there instead.
The same public runtime supports both; no private source checkout is required.

```bash
bash benchmarks/partner_if/run_ngc_replay.sh
```

Every run creates a fresh result directory. Model and judge settings, limits,
code/asset identities and technical exclusions remain in its receipts.

## Public tests

`benchmarks/partner_if/tests/fixtures/replay` contains a small invented project,
fictional registry/service endpoints and synthetic workflow data. The public
suite runs without registry credentials or access to the private data repository.
Private case-specific tests and real evidence belong to the data repository.

See `recorded-context-provenance.md` for code provenance and the adapter README
for lifecycle support and limits.
