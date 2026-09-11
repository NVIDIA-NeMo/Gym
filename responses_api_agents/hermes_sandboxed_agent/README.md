# Sandboxed Hermes with SWE-bench Pro

Runs **NousResearch/hermes-agent@v2026.8.31** inside the task container prepared by
Gym's existing SWE-bench Pro resources server. Hermes uses its terminal and file
tools in `/app`; Pro extracts and grades the patch through its existing verifier.
The model server runs separately. See [ASSESSMENT.md](ASSESSMENT.md) for a short
walkthrough and code review order, and [VALIDATION.md](VALIDATION.md) for results.

## Prepare the Hermes runtime

On Linux x86_64, build the runtime once:

```bash
DEPS_DIR=/absolute/path/hermes-runtime \
  bash responses_api_agents/hermes_sandboxed_agent/prepare_runtime.sh
```

Mount that directory read-only at `/opt/hermes` in task containers. Gym and
Hermes have conflicting OpenAI SDK pins, so keep their Python installations
separate. The runtime contains the exact source checkout, its Python interpreter,
resolved dependency versions and a commit manifest. Task startup verifies the
runtime and does not download or install Hermes.

## Launch with Pro

Compose the same four components used by the OpenCode sandboxed agent: model,
provider, agent and benchmark. This example uses Apptainer with local task images.
Prepare the selected SIF from its pinned registry digest; the helper records a
checksum manifest beside it and validates both files on cache reuse:

```bash
python -m resources_servers.swebench_pro.image_cache \
  --dataset resources_servers/swebench_pro/data/example.jsonl \
  --image-dir /cache/sifs --instance-id INSTANCE_ID_FROM_DATASET
```

SIF filenames use the digest's hexadecimal part. The server rejects missing
manifests, mismatched registry digests and changed SIF checksums. Old caches
without manifests must be prepared again. The runtime bind must be accessible
from the node running Gym. Create an overlay directory on disk before launching:

```bash
gym env start \
  --config responses_api_models/openai_model/configs/openai_model.yaml \
  --config nemo_gym/sandbox/providers/apptainer/configs/apptainer.yaml \
  --config responses_api_agents/hermes_sandboxed_agent/configs/hermes_sandboxed_agent.yaml \
  --config resources_servers/swebench_pro/configs/swebench_pro.yaml \
  +policy_base_url=http://MODEL_HOST:MODEL_PORT/v1 \
  +policy_api_key=gym \
  +policy_model_name=REAL_MODEL_NAME \
  +hermes_sandboxed_agent.responses_api_agents.hermes_sandboxed_agent.model=REAL_MODEL_NAME \
  +hermes_sandboxed_agent.responses_api_agents.hermes_sandboxed_agent.resources_server.name=swebench_pro_resources_server \
  +sandbox.apptainer.create.overlay_root=/absolute/disk/path/overlays \
  '+sandbox.apptainer.exec.default_binds=[/absolute/path/hermes-runtime:/opt/hermes:ro]' \
  '+swebench_pro_resources_server.resources_servers.swebench_pro.image_template=/cache/sifs/{image_digest_hex}.sif'
```

The resources-server reference must be supplied explicitly on this Gym revision.
The model proxy holds upstream credentials; Hermes receives only its proxy URL
and a dummy key. If using registry images instead of local SIFs, omit
`image_template` to use Pro's existing repository/digest selection.

Prepare the complete public dataset with the existing Pro preparation script,
then select a small subset for initial validation:

```bash
python benchmarks/swebench/pro/prepare.py
gym eval run --no-serve --agent hermes_sandboxed_agent \
  --input /path/to/pro-subset.jsonl \
  --output responses_api_agents/hermes_sandboxed_agent/results/pro.jsonl \
  --num-repeats 1 --concurrency 1
```

Pro also ships five prepared example rows at
`resources_servers/swebench_pro/data/example.jsonl`; these include pinned task
scripts and image digests and can be used for the first smoke run.

## Small HTTP smoke run

`smoke.py` starts actual Gym model, agent and Pro HTTP servers in three processes.
It avoids Ray for a small single-node run and uses the real Hermes runner and Pro
verifier. It does not allocate a model server.

```bash
python -m responses_api_agents.hermes_sandboxed_agent.smoke \
  --dataset resources_servers/swebench_pro/data/example.jsonl \
  --provider-config /path/to/provider.yaml \
  --sif-dir /cache/sifs --output /fresh/path/pro-results \
  --model-url http://MODEL_HOST:MODEL_PORT/v1 --model REAL_MODEL_NAME \
  --instance-id INSTANCE_ID_FROM_DATASET \
  --max-turns 30 --max-tokens 8192 --sandbox-timeout 1800
```

For a CPU startup check, use `--model-url http://127.0.0.1:9/v1`,
`--max-turns 1` and `--api-timeout 2`. An expected connection error demonstrates
startup and failure handling only; the smoke command exits 1 and records an
excluded attempt. A live endpoint consumes model inference
capacity even when the sandbox job itself uses no GPUs.

Cluster deployment is maintained in the Slurm evaluations repository. See its
[evaluation configuration](https://gitlab-master.nvidia.com/interactive-agents/slurm-evaluations/-/blob/jnolan/hermes-sandboxed-pro/evaluations/swebench-pro-hermes.yaml)
on the matching `jnolan/hermes-sandboxed-pro` branch.

To check the reference patch with Gym's verifier directly, use:

```bash
python -m responses_api_agents.hermes_sandboxed_agent.check_verifier \
  --smoke-results /path/to/completed/pro-smoke-results \
  --provider-config /path/to/provider.yaml \
  --output /path/to/reference-control.jsonl
```

`--provider-config` optionally overrides the recorded sandbox settings for the
current run. This control starts fresh verification containers without Hermes
or model inference.

Results include inputs, redacted config, server logs, model-call captures,
Hermes conversation and execution details, extracted patches and Pro test output.
Runtime diagnostics record the source commit, Python import path and working
directories. Image provenance records the selected image path, original registry
digest and local SIF checksum for both the agent and verifier containers.

`rollouts.jsonl` contains scored attempts; `failures.jsonl` contains excluded
attempts. `summary.json` lists all attempts, and `metrics.json` reports attempted,
scored, excluded, coverage and accuracy. Accuracy is over scored attempts only;
it is `null` when none were scored. Coverage must accompany any reported accuracy.

## Interface and checks

The agent runs through `/run`, which prepares a benchmark session. Text input and
terminal/file tools are supported. Unsupported multimodal or tool-history input
fails explicitly. Failed or unfinished runs retain `verifier_reward`, omit
`reward` and `response` from their HTTP result, and set Gym's existing
`_ng_failure_class=agent_run_error` marker. Gym's collector puts them in its
`*_failures.jsonl` sidecar and excludes them from scores. Incomplete verification
is excluded too. Completed, conclusive wrong answers still score zero. The
agent's `/aggregate_metrics` also filters incomplete rows for direct callers.

This agent requires a resources server that accepts `create_pty=false` and
returns a full `sandbox_descriptor`; it uses `cleanup_url_path` when provided.
Only the Pro integration has been exercised. The current Verified server returns
a bare handle, so changing the benchmark configuration alone will not make
Verified work on Apptainer. Its interface migration and a second-benchmark run
remain shelved with Verified; cross-benchmark swappability is unproven.

Rich Hermes invocation/compaction observation bundles are not yet
implemented; no training token IDs are fabricated.

```bash
pytest responses_api_agents/hermes_sandboxed_agent/tests \
  resources_servers/swebench_pro/tests tests/unit_tests/test_apptainer_provider.py
ruff check responses_api_agents/hermes_sandboxed_agent \
  resources_servers/swebench_pro nemo_gym/sandbox/providers/apptainer/provider.py
```
