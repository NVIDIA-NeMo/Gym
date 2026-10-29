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

## SWE-bench Pro V2

V2 is the validated public split from upstream tag
[`v2.0.0`](https://github.com/scaleapi/SWE-bench_Pro-os/tree/66f92766bba642462d4bbe5479e83f91f9211862/v2):
642 tasks (`benchmarks/swebench/pro_v2`) and the HARD-51 subset (`benchmarks/swebench/pro_v2_hard`), from HF
revision `2d52cb3df914a3fcf80c7f66738b3a88ae37fc50`. V1 (`benchmarks/swebench/pro`) is unchanged.

- Preparation checks upstream's `v2/SHA256SUMS` and embeds each Harbor task's `instruction.md` as the prompt, its
  `tests/` verifier and reference patch, and the GHCR image digest. A row with `verifier_files` is graded by V2.
- Locked protocol: `agent_network_lock` gives the agent sandbox an OpenSandbox `network_policy` that allows only
  the host of `sandbox_model_server`; other providers fail closed, and so does a head server on that host (the V2
  configs set `absolute_ip_server_types: [responses_api_models]`, and the lock refuses a head server off loopback).
  The agent's diff is applied to a fresh sandbox
  from the pristine image (as `v2/tooling/patch_replay.py` does) and the task's `tests/test.sh` decides the reward.
- OpenCode installs offline from the OpenSandbox S3 cache, and prepare stages a pinned static `rg` for upload.
- Budgets are upstream's: 50 minutes for the agent (the patch made so far is still graded), 3000 s for `test.sh`.
- Verdicts are upstream's: `reward.txt` is final, a `test.sh` past its budget and a patch that cannot be extracted
  score 0. Only environment faults (no reward, a rejected test patch, a failed exec) are retried on a new sandbox
  and then returned with `mask_sample=true` and `failure_kind: verifier_error`; test output, which the patched code
  writes, never overrides a reward.

The upstream evaluator and bundled task scripts are MIT licensed. NeMo Gym's adapter code is Apache-2.0.
