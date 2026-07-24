# Synthetic Tool-Use Generation

The generation workflow runs as four Gym resource servers in one configured server graph. The pipeline server
orchestrates three ordered generation stages and owns validation and dataset materialization. Stage servers own their
prompts, parsing, quality gates, retries, and durable artifact writes.

## Server Graph

```text
synthetic_tool_use_pipeline
  |-- synthetic_tool_use_domain_generation ------> domain model server
  |-- synthetic_tool_use_policy_tool_generation -> policy model server
  |                                             -> judge model server
  `-- synthetic_tool_use_scenario_generation ----> scenario model server
```

Resource-to-resource and resource-to-model requests use Gym `ServerClient` references from the selected YAML. The
model server layer owns endpoint URLs, API keys, and provider adaptation.

## Domain Generation

The domain server sends the checked-in brainstorming prompt, parses JSON candidates, sends a follow-up request that
excludes names from the first response, and deduplicates normalized names. It writes:

- `domains.raw.jsonl`
- `domains.accepted.jsonl`
- `domains/<index>/domain.json`
- `attempts/domains/*.json`

## Policy And Tool Generation

For each accepted domain, the policy/tool server generates a policy and tool schemas, optionally refines both,
performs deterministic validation, and optionally runs cohesion and golden-reference judgments. It writes
`policy.md`, `tools.jsonl`, `quality_report.json`, and per-attempt provider responses.

## Scenario Generation

The scenario server creates a deterministic schedule of inside-policy and outside-policy requests, renders the
checked-in system and user prompts, validates the structured response, removes duplicate scenarios, and writes
scenario JSONL shards under `scenarios/<model>/`.

## Resume And Partitioning

All servers use the same `run_manifest.json` and output directory. With `resume: true`, a completed stage is skipped
only when its expected artifacts still load and pass validation. Policy/tool and scenario calls accept an inclusive
`domain_start` and exclusive `domain_end` for partitioned work.

Provider retries cover transient model-server failures. Semantic attempts cover responses that arrive but cannot be
parsed or accepted. Both are recorded in artifact diagnostics.

## Validation And Materialization

`POST /validate` checks every completed domain bundle. `POST /materialize` validates first, then converts accepted
bundles into Gym rows consumed by `synthetic_tool_use_agent` and `synthetic_tool_use_simulation`.

See the parent [README](../README.md) for server startup and invocation.
