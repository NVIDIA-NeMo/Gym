# Sandboxed Hermes with SWE-bench Pro

Runs **NousResearch/hermes-agent@v2026.9.7** inside the task container prepared by
Gym's existing SWE-bench Pro resources server. Hermes uses its terminal and file
tools in `/app`; Pro extracts and grades the patch through its existing verifier.
The model server runs separately. See [ASSESSMENT.md](ASSESSMENT.md) for the
component map and code review order.

## Prepare the Hermes runtime

On Linux x86_64, build both runtimes once. Pro includes GNU and Alpine/musl images:

```bash
DEPS_DIR=/absolute/path/hermes-runtime \
  bash responses_api_agents/hermes_sandboxed_agent/prepare_runtime.sh
DEPS_DIR=/absolute/path/hermes-runtime/musl ARCH=x86_64-unknown-linux-musl \
  bash responses_api_agents/hermes_sandboxed_agent/prepare_runtime.sh
```

Mount that directory read-only at `/opt/hermes` in task containers. Gym and
Hermes have conflicting OpenAI SDK pins, so keep their Python installations
separate. The runtime contains the exact source checkout, its Python interpreter,
resolved dependency versions and a commit manifest. Task startup verifies the
runtime and does not download or install Hermes. The default `hermes-python`
launcher selects the interpreter for each task image.

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
from the node running Gym:

```bash
gym env start \
  --config responses_api_models/openai_model/configs/openai_model.yaml \
  --config responses_api_agents/hermes_sandboxed_agent/configs/apptainer.yaml \
  --config responses_api_agents/hermes_sandboxed_agent/configs/hermes_sandboxed_agent.yaml \
  --config resources_servers/swebench_pro/configs/swebench_pro.yaml \
  +swebench_pro_example_agent=null \
  +swebench_pro_example_resources_server=null \
  +policy_base_url=http://MODEL_HOST:MODEL_PORT/v1 \
  +policy_api_key=gym \
  +policy_model_name=REAL_MODEL_NAME \
  +hermes_sandboxed_agent.responses_api_agents.hermes_sandboxed_agent.model=REAL_MODEL_NAME \
  +hermes_sandboxed_agent.responses_api_agents.hermes_sandboxed_agent.resources_server.name=swebench_pro_resources_server \
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

Cluster deployment uses the Slurm evaluations repository's existing pipeline;
select its [evaluation configuration](https://gitlab-master.nvidia.com/interactive-agents/slurm-evaluations/-/blob/jnolan/hermes-sandboxed-pro/evaluations/swebench-pro-hermes.yaml).
For a reference-patch control, use Pro's existing
[golden-patch commands](../../resources_servers/swebench_pro/README.md#golden-patch-smoke-test).

## Interface and checks

The agent runs through `/run`, which prepares a benchmark session. Text input and
terminal/file tools are supported. Unsupported multimodal or tool-history input
fails explicitly. Harness failures retain `verifier_reward`, omit
`reward` and `response` from their HTTP result, and set Gym's existing
`_ng_failure_class=agent_run_error` marker. Gym's collector puts them in its
`*_failures.jsonl` sidecar and excludes them from scores. Incomplete verification
is excluded too. Reaching a turn, output-token or wall-time budget after model
output retains the patch's score and records
`response.metadata.budget_exhausted=true`. Missing or failing graded tests score zero.
The agent inherits Gym's standard aggregation; Slurm reporting includes coverage.

```bash
pytest responses_api_agents/hermes_sandboxed_agent/tests \
  resources_servers/swebench_pro/tests tests/unit_tests/test_apptainer_provider.py
ruff check responses_api_agents/hermes_sandboxed_agent \
  resources_servers/swebench_pro nemo_gym/sandbox/providers/apptainer/provider.py
```
