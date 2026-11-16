# Recorded-context ordering evaluation

The `partner_if` replay profile continues supplied history against frozen tool data.
The public runner contains no customer source, prompt, service address or credential.
The authenticated registry supplies actor inputs, recorded responses and a validated
`runtime.json` declaring paths, aliases, asset hashes and probe identity.

Gym's existing `single_agent_turn` Environment Server is used without changes.
It seeds Resources and one replay-agent session, invokes the agent once, closes
that session, and performs its normal final verification and cleanup.

The replay agent owns the bounded prefix loop inside that one activation. Each
step settles its policy and tool calls and freezes its capture before asking the
PartnerIF Resources Server's `/replay/prefix-review` endpoint whether to continue.
The outer agent session stays open during these intermediate reviews. The
verifier stops at the first decisive ordering result: a failed ordering
requirement scores zero, while a returned identity-verified prerequisite check
can establish a pass. Cutoffs, unsupported operations, and infrastructure or
judge faults are masked. Supplied history remains context.

There is no separate replay Environment Server and no extension to Gym's server
runtime or episode types. The replay agent aggregates fresh usage, observations,
and trajectories; task-specific ordering decisions and receipt validation stay
in `resources_servers/partner_if`. Internal prefixes reuse saved policy replies
without sending those calls again. The final agent close exports all fresh work
and the intermediate verifier records.

Resources durably binds its typed seed to the original task, attempt, task data
and actor parameters, then retains each review under the configured artifact
root. A terminal receipt binds the exact final response and cumulative usage to the original
input, task, attempt, rubric and capture. After the agent closes, normal `/verify`
validates that receipt and reuses its verdict without another judge request.
Retries reuse completed review receipts. An interrupted review with uncertain
judge completion is masked rather than silently judged again. Live replay does
not support cross-process agent-session continuation or offline regrading.

The dispatcher converts the prepared flat row to an ordinary typed episode and
converts its reply back to the collector's format. Host-selected task metadata
travels through the existing `agent_context` interface, outside the frozen model
input. Published task rows, actor input, rubrics, and task/attempt identities are
preserved. The agent uses the existing direct-HTTP Resources grant for reviews;
that control endpoint is never advertised as a policy tool.

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
