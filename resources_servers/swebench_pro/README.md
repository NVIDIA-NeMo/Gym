# SWE-bench Pro

This resources server evaluates patches for the public
[ScaleAI SWE-bench Pro](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro) benchmark with NeMo Gym
`AsyncSandbox` providers.

The integration follows the evaluator from
[`scaleapi/SWE-bench_Pro-os`](https://github.com/scaleapi/SWE-bench_Pro-os/tree/ca10a60a5fcae51e6948ffe1485d4153d421e6c5):
it starts the task image identified by `dockerhub_tag`, resets `/app` to the base commit, applies the candidate
patch, runs the task-specific script, parses its JSON output, and requires every fail-to-pass and pass-to-pass
test to pass.

## Prepare data

For a Hermes rollout using this resources server, see the
[sandboxed Hermes launch instructions](../../responses_api_agents/hermes_sandboxed_agent/README.md).
The agent uses the same task preparation and verifier as OpenCode. Command-based
agents can request `create_pty=false` at `/seed_session`; the default still creates
a terminal. Connectable providers also return `sandbox_descriptor`, and agents
can release an abandoned task through the cookie-scoped `/close_session` endpoint.

Optional local SIFs must have provenance manifests produced by
`python -m resources_servers.swebench_pro.image_cache`. Set `image_template` to
`/cache/sifs/{image_digest_hex}.sif`; the helper names files by the pinned digest.
Before starting a container, the server checks the original registry URI and
the SIF checksum against the manifest. Seed and verify responses include
`image_provenance`; the registry path remains unchanged when no template is set.

```bash
uv run python benchmarks/swebench/pro/prepare.py
```

Preparation pins and embeds the task-specific run scripts, parsers, and Dockerfile metadata from the upstream
evaluator commit. The resources server therefore does not access GitHub while serving verification requests.

## Golden-patch smoke test

Start the resources server with an OpenSandbox provider:

```bash
gym env start \
  --config resources_servers/swebench_pro/configs/swebench_pro.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  +swebench_pro_resources_server.resources_servers.swebench_pro.is_verifying_golden_patch=true
```

In another terminal, verify one row:

```bash
python resources_servers/swebench_pro/client.py \
  +benchmark_jsonl=benchmarks/swebench/data/swebench_pro_benchmark.jsonl
```

Run a bounded batch by setting both the row limit and sandbox concurrency:

```bash
python resources_servers/swebench_pro/apply_golden_patch.py \
  +benchmark_jsonl=benchmarks/swebench/data/swebench_pro_benchmark.jsonl \
  +limit=10 \
  +concurrency=2
```

The upstream evaluator and bundled task scripts are MIT licensed. NeMo Gym's adapter code is Apache-2.0.
