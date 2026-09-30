# SOL evaluator integration

This resource server connects Gym to an explicitly configured SOL evaluator.
It supports the `simple_agent` verification interface and stateless
`gym eval reverify`. The evaluator owns solution extraction, native GPU
execution, workload coverage checks, candidate/error attribution, and the SOL
score formula. Gym transports responses and reports complete-coverage metrics.
No dataset, subset selection, prompts, baseline table, or evaluation results
are bundled here. Keep those files and all runtime artifacts **outside this
repository**, including in local development.

This is an unbaselined integration (`verified: false`), not a new benchmark
release. The CPU tests exercise transport, retry recovery, and aggregation with
synthetic evaluators; they do not establish GPU correctness or performance.

## Configure a trusted evaluator

For new generation, use `configs/sol_execbench.yaml` with an external YAML
override for its required fields and a configured model server. For replay,
use the resources-only configuration below; no model or agent process is needed.
Paths must be absolute. A private manifest looks like:

```json
{
  "schema_version": 1,
  "protocol_sha256": "<SHA256 of the evaluator protocol>",
  "samples_per_task": 8,
  "tasks": [{"task_id": "example-task"}]
}
```

`manifest_sha256` pins the exact bytes of that file. `evaluator_command` is a
trusted argv list, for example an isolated runtime launcher followed by its
Python interpreter and adapter script. The server appends
`--request /absolute/request.json --result /absolute/result.json`. It never
passes candidate text to a shell. Pin the adapter, native evaluator, images,
inputs, workloads, and scoring dependencies in the external protocol; the
adapter must validate those bindings before admission.

The server is **not a candidate-code sandbox** and does not install a compiler,
CUDA runtime, or the SOL evaluator. Run the configured adapter inside the
appropriate isolated, pinned execution environment. That launcher or runtime
must also own cleanup of descendants that create separate process sessions.
Runtime preparation must not silently upgrade the evaluator or alter its
measurement settings.

Use one resource-server process for each exclusively assigned physical GPU.
`gpu_uuid` is recorded and supplied as `CUDA_VISIBLE_DEVICES`; requests within
that process execute serially. Separate processes must not share that GPU.
`runner_timeout_s` is the adapter watchdog, distinct from the native evaluator's
measurement budget. Configure it to leave time for the native evaluator's own
timeouts and cleanup. Wrapper failures remain infrastructure failures.

## Request and result contract

Gym inputs carry the exact original `responses_create_params`, `task_id`, and
`verifier_metadata`. Saved rollout files carry `response`, `_ng_task_index`, and
`_ng_rollout_index`. Put source hashes and replay provenance in input
`verifier_metadata`: Gym replay forwards the saved response, not arbitrary
extra fields from the old rollout. Never import old rewards or timing results
as fresh verification. Preserve response text bytes and reported usage; do not
fabricate generation-time token IDs by re-tokenizing saved text.

The evaluator request file contains `schema_version`, `request_id`, `task_id`,
`protocol_sha256`, `gpu_uuid`, the original `responses_create_params`, the full
Gym `response`, and `verifier_metadata`.
It must write one JSON object at the requested result path and exit zero:

```json
{
  "request_id": "<echo the request ID>",
  "task_id": "example-task",
  "protocol_sha256": "<echo the pinned protocol hash>",
  "outcome": "PASSED",
  "infrastructure_error": false,
  "solved": true,
  "sol_score": 0.25,
  "native_result": {"detail": "Native traces and process logs retained by the adapter"}
}
```

Only the evaluator may establish a candidate failure. A passing result requires
its exact full workload coverage, correctness, and valid timing checks. An
unknown infrastructure failure must be unsolved with `sol_score: null`.
`EVALUATION_TIMEOUT` is always an infrastructure outcome. Retain native stdout,
stderr, timeout diagnostics, workload UUIDs, and trace files in the attempt
directory. A native timeout may allow the next candidate; any other
infrastructure error stops this worker until operator reconciliation.

The resource server validates result identity and invariants. Gym requires a
numeric `reward`, so unresolved records use a **masked** transport value of zero
while retaining `sol_score: null`. This zero is not a measured candidate score.

## Replay and recovery

Create an external resources-only YAML file, replacing the placeholder values
with your pinned manifest, evaluator command, and allocated physical GPU UUID:

```yaml
# /private/sol/runtime.yaml
sol_execbench:
  resources_servers:
    sol_execbench:
      entrypoint: app.py
      num_workers: 1
      manifest_path: /private/sol/manifest.json
      manifest_sha256: "<SHA256 of the exact manifest bytes>"
      evaluator_command:
        - /private/sol/pinned-runtime-launcher
        - /private/sol/evaluator-adapter.py
      artifact_root: /private/sol/new-evaluation-artifacts
      gpu_uuid: "<allocated GPU UUID>"
      runner_timeout_s: 1800
      timeout_zero_sensitivity: false
```

Do not include the bundled agent/model config in this replay file. Run Gym's
existing command; it starts the configured resources server itself:

```bash
gym eval reverify --config /private/sol/runtime.yaml \
  --inputs /private/sol/inputs.jsonl \
  --rollouts /private/sol/saved-responses.jsonl \
  --output /private/sol/fresh-results.jsonl --concurrency 1
```

For paired comparisons, the external orchestrator must preserve model order
and assignment to the same GPU; this server's semaphore alone does not do so.
Each `(task_index, rollout_index)` must be globally unique within a replay file,
including when mixing model arms. Aggregate each model separately. For sharded
runs, use `--disable-aggregation` and aggregate only after assembling all shards.

Transport retries with the same immutable request coalesce and reuse a validated
completed attempt. This only deduplicates retries within the new evaluation;
the adapter must never read old experiment results as new measurements. Each
attempt retains its request, process record, stdout, stderr, evaluator result,
and accepted result. An interrupted attempt without a valid accepted result is
unresolved and is **not automatically re-executed**. Preserve it and reconcile
the scheduler/process state before explicitly creating a replacement run.

## Metrics

This server overrides the entire `/aggregate_metrics` endpoint because Gym's
default aggregator removes masked samples before custom metric hooks. Its
external manifest fixes the task set and repeat count, including missing slots.

- `official/sol_bestK`: `max(0, best passing SOL score)` of K candidates per task,
  averaged equally across all expected tasks; unsolved tasks contribute zero.
  This headline zero floor does not alter raw scores or the observed per-task
  best passing score, and scores above one are not clipped.
- `official/correctness_at_1`: passing candidates divided by all expected slots.
- `official/pass_at_K`: tasks with at least one passing candidate divided by all
  expected tasks. These rates are fractions, not percentages.
- Every official metric is null with any missing or unresolved evaluation.
- Optional `timeout_zero/*` values are explicitly labeled sensitivity estimates.
  They require every record and permit only native `EVALUATION_TIMEOUT` holes.
  Raw outcomes are unchanged; other errors or missing records suppress them.

Per-task counts and observed best passing scores are retained for diagnosis.
Use `gym eval aggregate` and this server's official keys for SOL reporting;
generic `gym eval profile` measured-subset means are not SOL completion metrics.
Keep hardware-specific scoring anchors distinct from freshly measured reference
latencies, and retain the evaluator's original score definition.
