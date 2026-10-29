# Real-world Agentic IF verifier

See the [benchmark runbook](../../benchmarks/agentic_if_realworld/README.md) for the
measurement contract, private registry setup and execution instructions.

This server owns frozen criteria and strict-v2.2 scoring. The policy receives the
captured input only. Pass/fail rewards are 1/0; technical and unresolved outcomes
are masked with a failure kind/reason. Judge receipts record model, reasoning,
response identity and provider token usage when supplied.

All runtime helpers live under `benchmark/`; no separate benchmark checkout or
external component root is required. Original source hashes remain distinct from
public packaging hashes. The original rubric and judge prompt are downloaded
unchanged from an approved authenticated private registry for real runs. Public
CI uses a separately identified synthetic prompt. Real tasks, per-item criteria,
original policy documents and private receipts are never examples in Git.
Only independently constructed synthetic fixtures are committed.

`REVERIFY_MODE=STATELESS` supports saved native Gym responses. The adaptive
profile additionally verifies immutable upstream delivery receipts. A shared
receipt filesystem must support POSIX advisory locks. Duplicate live callers
share one saved grade; an abandoned in-flight dispatch requires reconciliation.

## Public CI examples

`data/example.jsonl`, `data/example_metrics.json` and `data/example_rollouts.jsonl`
contain five independent synthetic literal tasks and collected results. The local
policy deliberately passes three and fails two; all five answers are complete.
The fixtures contain no real benchmark rows or criteria and are never part of the
969-task benchmark. CI validates both their contents and Gym's five-example gate.

To regenerate, use the synthetic environment exports and local mock server in the
benchmark runbook with a fresh private directory at `/private/agentic-if/ci`.
Set `AGENTIC_IF_DATA_DIR`, `AGENTIC_IF_PREPARED_PATH` and
`AGENTIC_IF_RECEIPT_DIR` to its `data`, `prepared.jsonl` and `receipts` paths.
Inspect and unlink an existing ignored benchmark `data/` directory symlink that
points elsewhere. Preserve older file-only layouts as described in the benchmark
runbook before removing their empty directory.
Then run from the Gym root:

```bash
python -m resources_servers.agentic_if_realworld.benchmark.synthetic "$AGENTIC_IF_DATA_DIR" --examples
gym eval prepare --benchmark agentic_if_realworld
gym eval run --benchmark agentic_if_realworld --split benchmark --num-repeats 1 \
  --concurrency 1 --output /private/agentic-if/ci/rollouts.jsonl \
  ++observability_enabled=true ++model_call_capture_dir=/private/agentic-if/ci/captures \
  ++overwrite_metrics_conflicts=true
cp /private/agentic-if/ci/prepared.jsonl resources_servers/agentic_if_realworld/data/example.jsonl
gym dataset collate '+config_paths=[benchmarks/agentic_if_realworld/config.yaml]' \
  +mode=example_validation +output_dirpath=/private/agentic-if/ci/collated \
  '++agentic_if_realworld_resources.resources_servers.agentic_if_realworld.datasets=[{name:example,type:example,jsonl_fpath:resources_servers/agentic_if_realworld/data/example.jsonl,num_repeats:1}]'
cp /private/agentic-if/ci/collated/example_metrics.json resources_servers/agentic_if_realworld/data/example_metrics.json
cp /private/agentic-if/ci/rollouts.jsonl resources_servers/agentic_if_realworld/data/example_rollouts.jsonl
gym env test +entrypoint=resources_servers/agentic_if_realworld +should_validate_data=true
```

Only copy these independently generated synthetic artifacts into Git. Captures,
judge receipts and real-data runs always stay outside the checkout.

The public component CI checks need neither the private registry nor inference
credentials. From an installed Gym checkout, run:

```bash
gym env test +entrypoint=resources_servers/agentic_if_realworld +should_validate_data=true
gym env test +entrypoint=responses_api_models/agentic_if_text +should_validate_data=true
```
