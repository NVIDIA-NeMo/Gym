# Mini-SWE-Agent 2 Sandbox Agent

A NeMo Gym Responses API agent that integrates
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) v2 for evaluating
language models on SWE-bench style software engineering tasks through the public
`nemo_gym.sandbox` API.

This agent intentionally keeps only the sandbox-backed path. It does not carry
over the older Docker/Singularity mini-SWE integration.

## Contents

- [Mini-SWE-Agent 2 Sandbox Agent](#mini-swe-agent-2-sandbox-agent)
  - [Contents](#contents)
  - [Overview](#overview)
  - [Dataset Information](#dataset-information)
  - [Configuration](#configuration)
    - [Agent Configuration](#agent-configuration)
    - [Model Parameters](#model-parameters)
  - [Usage](#usage)
    - [Server](#server)
    - [Collect Rollouts](#collect-rollouts)
  - [Sandbox Environment Adapter](#sandbox-environment-adapter)
    - [Environment Lifecycle](#environment-lifecycle)
  - [Contributing](#contributing)
  - [Licensing Information](#licensing-information)
    - [Dependencies](#dependencies)

## Overview

`mini_swe_agent_2` runs mini-swe-agent's synchronous SWE-bench harness while
creating and executing each task environment through Gym's provider-neutral
sandbox facade. The validated path in this directory is:

- mini-swe-agent `2.1.0`
- SWE-bench task rows, including SWE-bench Verified
- `env: sandbox`
- `responses_api_agents.mini_swe_agent_2.sandbox_environment.MiniSWESandboxEnvironment`
- OpenSandbox through `nemo_gym.sandbox.providers.opensandbox`

For each `/run` request, `MiniSWEAgent.run()` loads mini-swe-agent's built-in
`swebench.yaml`, injects sandbox settings, runs mini-swe-agent in a Ray remote
task, evaluates the generated patch with the SWE-bench harness, and returns a
Gym verify response with reward `1.0` only when the instance is resolved and the
evaluation report includes test status.

`MiniSWEAgent.setup_webserver()` also registers `/v1/responses`, but
`MiniSWEAgent.responses()` is intentionally not implemented in this agent. The
supported eval path is `/run`, typically via `gym eval run --no-serve`.

## Dataset Information

- Eval data - [princeton-nlp/SWE-bench_Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)
  is the primary validation target. It contains 500 human-validated SWE-bench
  test instances.
- The rollout input JSONL should preserve the SWE-bench instance fields needed
  by `swebench`, such as `instance_id`, `repo`, `base_commit`,
  `problem_statement`, `patch`, `test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`,
  and related version fields.
- Each row must also include `responses_create_params`. Extra top-level
  SWE-bench fields are accepted by the agent request model and passed into
  mini-swe-agent as the instance dictionary.

Example row shape:

```json
{
  "instance_id": "django__django-13410",
  "repo": "django/django",
  "base_commit": "...",
  "problem_statement": "...",
  "patch": "...",
  "test_patch": "...",
  "FAIL_TO_PASS": ["..."],
  "PASS_TO_PASS": ["..."],
  "responses_create_params": {
    "input": [],
    "temperature": 0.6,
    "top_p": 1.0,
    "max_output_tokens": 16384
  }
}
```

When `image_name` is present on a row, the agent uses it directly. Otherwise it
derives the SWE-bench image from `instance_id` and `subset`:

- `subset: verified` uses `docker.io/swebench/sweb.eval.x86_64.<id>:latest`
  with `__` replaced by `_1776_`.
- Other subsets use `docker.io/xingyaoww/sweb.eval.x86_64.<id>:latest` with
  `__` replaced by `_s_`.

The default OpenSandbox config uses explicit Docker Hub image refs so cluster
mirroring can happen in the container runtime instead of Gym-side image
rewrites.

## Configuration

### How sandboxes are configured

There is one concept to learn: **a sandbox is a named block**, and an agent points
at it by name.

```yaml
# A named sandbox: <name> maps to <provider> maps to that provider's config.
sandbox:                 # instance name (the handle the agent references)
  opensandbox:           # provider registry key -> provider class
    connection: { ... }  # provider-specific config
```

```yaml
# An agent selects a sandbox by name:
sandbox_provider: sandbox
```

The framework only ever resolves *a name -> one provider config*. Everything else
falls out of how you name and reference blocks:

- **Single sandbox (default).** Ship a `sandbox` block; the agent defaults to
  `sandbox_provider: sandbox`. Done.
- **Swap providers (no agent edit).** Every shipped provider config binds the same
  name `sandbox`, so swapping providers is just swapping one config path in
  `+config_paths`.
- **Multiple / mixed / same-type sandboxes.** Give blocks **distinct instance
  names** (e.g. `opensandbox_foo`, `opensandbox_baz`) and reference each by name.
  See [Advanced: multiple sandboxes](#advanced-multiple-sandboxes).

> Names are arbitrary instance names, not provider types. Two config files that
> bind the **same** name merge last-wins (that is the swap mechanism); to run
> several at once, use distinct names.

### Agent Configuration

The agent config is **provider-neutral**: it selects a sandbox by name via
`sandbox_provider`, and the named block lives in a separate provider config file.
This decouples the agent from any specific sandbox provider so you can swap
providers by swapping a single config path in `+config_paths` — no edits to the
agent config.

Path - `responses_api_agents/mini_swe_agent_2/configs/mini_swe_agent_2.yaml`

```yaml
mini_swe_agent_2:
  responses_api_agents:
    mini_swe_agent_2:
      entrypoint: app.py
      domain: coding
      description: Software engineering tasks driven by mini-swe-agent harness on a Gym sandbox.
      value: Improve agentic software engineering capabilities.
      model_server:
        type: responses_api_models
        name: policy_model
      concurrency: 64
      env: sandbox
      # Name of the sandbox to use; defined in a separate provider config (see
      # "Sandbox Provider Configuration" below).
      sandbox_provider: sandbox
      sandbox_spec:
        ttl_s: 18000
        ready_timeout_s: 1200
        resources:
          cpu: 2
          memory_mib: 8192
          disk_gib: 20
        provider_options:
          platform:
            os: linux
            arch: amd64
        metadata:
          benchmark: swebench-verified
          harness: mini-swe-agent
      sandbox_environment_kwargs:
        cwd: /testbed
        conda_env: testbed
        activate_conda: true
        user: root
      run_golden: false
      step_timeout: 600
      eval_timeout: 1800
      skip_if_exists: false
      step_limit: 250
```

`sandbox_provider` accepts either a name reference (resolved from a top-level
sandbox block in the merged config, the recommended decoupled form) or an inline
single-key provider mapping (`{provider_name: {...}}`) when you prefer to keep
everything in one file.

### Sandbox Provider Configuration

Each provider ships its own config file that defines a named sandbox block. The
default OpenSandbox config is:

Path - `nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml`

```yaml
sandbox:                      # name referenced by the agent's sandbox_provider
  opensandbox:                # provider registry key -> provider class
    connection:
      domain: ${oc.env:OPENSANDBOX_DOMAIN,opensandbox-server.opensandbox-system.svc.cluster.local}
      api_key: ${oc.env:OPENSANDBOX_API_KEY}
      protocol: http
      request_timeout_s: 300
      use_server_proxy: true
    create:
      request_timeout_s: 1200
      timeout_s: 1200
      skip_health_check: true
      retries: 10
      retry_delay_s: 5.0
      retry_max_delay_s: 90.0
    probe:
      timeout_s: 60
      deadline_s: 180
      stable_count: 2
      stable_delay_s: 1.0
    operations:
      retries: 5
      retry_delay_s: 1.0
      retry_max_delay_s: 45.0
      command_retries: 0
      close_timeout_s: 30
```

To use a different provider, add a config file under
`nemo_gym/sandbox/providers/<provider>/configs/<provider>.yaml` that defines a
`sandbox` block (the name the agent references) with that provider's registry key,
then point `+config_paths` at it instead — no agent edit required.

Optional `sandbox_resource_profiles` can be configured as a list of resource
maps. When present, the agent hashes `instance_id` and deterministically merges
one profile into `sandbox_spec.resources`. This is useful for spreading
SWE-bench tasks across a small set of resource sizes without changing the input
data.

### Advanced: multiple sandboxes

The default convention (every provider file binds the name `sandbox`) is optimized
for the single-sandbox case and path-only swapping. To run more than one sandbox
in the same `ng_run`, give each block a **distinct instance name** and reference it
explicitly. Because names are arbitrary, this covers every multi-sandbox case
without any framework change:

- **Different providers at once** (e.g. one agent on OpenSandbox, a grader on
  another provider):

  ```yaml
  sandbox_rollout:
    opensandbox: { ... }
  sandbox_grading:
    docker: { ... }
  ```

- **Two configs of the same provider type** (e.g. two OpenSandbox endpoints — note
  the same inner `opensandbox` key, distinct outer instance names):

  ```yaml
  opensandbox_foo:
    opensandbox: { connection: { domain: foo... } }
  opensandbox_baz:
    opensandbox: { connection: { domain: baz... } }
  ```

Each agent then references the instance it needs (`sandbox_provider:
opensandbox_foo`). Whether a single agent consumes one or several sandboxes is
part of that agent's config contract; `mini_swe_agent_2` uses exactly one sandbox
per task.

> Reminder: do not give two included config files the same instance name unless you
> intend swap-by-replace — same name merges last-wins.

### Model Parameters

`MiniSWEAgent.run()` maps supported Responses API fields into mini-swe-agent
chat-completions kwargs:

- `temperature`, `top_p`, `top_logprobs`, and `parallel_tool_calls` pass through.
- `max_output_tokens` becomes `max_tokens`.
- `responses_create_params.metadata.extra_body` must be a JSON object and is
  passed as `extra_body`.
- `responses_create_params.metadata.chat_template_kwargs` must be a JSON object
  and is nested under `extra_body.chat_template_kwargs`.
- `tool_choice` comes from the agent config when set, otherwise from the request.
  The special value `bash` expands to the OpenAI function choice for the `bash`
  tool.

Keep the requested generation budget compatible with the live vLLM deployment.
For example, a deployment served with `--max-model-len 32768` will reject
`max_output_tokens=49152`. In earlier smoke testing, that upstream vLLM rejection
surfaced in mini-swe-agent as repeated:

```text
No tool calls found in the response. Every response MUST include at least one tool call.
```

That symptom was not a sandbox failure and was not a reason to force the `bash`
tool. The successful smoke kept `tool_choice=auto` and lowered
`max_output_tokens` to `16384`.

## Usage

### Server

Set the policy model endpoint in `env.yaml` or with equivalent Hydra overrides:

```yaml
policy_base_url: http://<vllm-service>.<namespace>.svc.cluster.local:8000/v1
policy_api_key: dummy-key
policy_model_name: <served-model-name>
```

Start the mini-swe-agent 2 server by composing three config paths: the
provider-neutral agent config, a sandbox provider config, and a policy model
server config. To swap providers, change only the sandbox provider path. The
values below show a representative SWE-bench eval setup with OpenSandbox:

```bash
gym env start \
    --config responses_api_agents/mini_swe_agent_2/configs/mini_swe_agent_2.yaml \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    --model-type vllm_model \
    +mini_swe_agent_2.responses_api_agents.mini_swe_agent_2.concurrency=64 \
    +mini_swe_agent_2.responses_api_agents.mini_swe_agent_2.step_timeout=600 \
    +mini_swe_agent_2.responses_api_agents.mini_swe_agent_2.eval_timeout=1800 \
    +mini_swe_agent_2.responses_api_agents.mini_swe_agent_2.step_limit=50 \
    +mini_swe_agent_2.responses_api_agents.mini_swe_agent_2.run_golden=false \
    '+mini_swe_agent_2.responses_api_agents.mini_swe_agent_2.sandbox_spec.resources={cpu: 0.5, memory_mib: 4096, disk_gib: 8}' \
    '+mini_swe_agent_2.responses_api_agents.mini_swe_agent_2.sandbox_spec.metadata={benchmark: swebench-verified, harness: mini_swe_agent_2, endpoint_label: hosted-vllm, run_family: mini-swe-agent-2-pass8}'
```

Use a model server config that matches the policy endpoint you are serving. The
example above uses `vllm_model`, which is the common path for hosted vLLM
`/v1/chat/completions` endpoints.

### Collect Rollouts

Collect eval rollouts from a SWE-bench-style JSONL file:

```bash
gym eval run --no-serve \
    --agent mini_swe_agent_2 \
    --input data/mini_swe_verified_smoke8.jsonl \
    --output results/mini_swe_agent_2_pass8.jsonl \
    --limit 8 \
    --num-repeats 8 \
    --concurrency 64 \
    --max-output-tokens 32768 \
    --temperature 0.6 \
    --top-p 0.95 \
    "++responses_create_params.metadata.chat_template_kwargs='{\"enable_thinking\": true}'"
```

`gym eval run --no-serve` also writes
`results/mini_swe_agent_2_pass8_aggregate_metrics.json`
with per-task eval status, pass@k, resolved task counts, and eval error rates.
After collecting repeated rollouts, run `gym eval profile` on the collected
output when you want the standalone profiler JSONL as well:

```bash
gym eval profile \
    --input data/mini_swe_verified_smoke8.jsonl \
    +materialized_inputs_jsonl_fpath=results/mini_swe_agent_2_pass8_materialized_inputs.jsonl \
    +rollouts_jsonl_fpath=results/mini_swe_agent_2_pass8.jsonl \
    +pass_threshold=1.0
```

The profiler writes `*_reward_profiling.jsonl` and `*_agent_metrics.json`
next to the rollouts file.

The agent writes per-instance mini-swe-agent configs and result artifacts under
`results/<subset>/<policy_model_name>/`.

Use the agent's `step_timeout` and `eval_timeout` overrides above to bound tool
and verifier execution. If you launch from a custom Kubernetes wrapper, add any
outer per-sample guard there.

## Sandbox Environment Adapter

`MiniSWESandboxEnvironment` adapts mini-swe-agent's synchronous environment
contract to `nemo_gym.sandbox.Sandbox`.

When `env` is `sandbox`, the agent resolves `sandbox_provider` (name reference or
inline mapping) to a single-key provider config and Gym injects this environment
config before calling mini-swe-agent:

```yaml
environment:
  environment_class: responses_api_agents.mini_swe_agent_2.sandbox_environment.MiniSWESandboxEnvironment
  image: <swebench task image>
  provider:
    opensandbox:
      connection: ...
  spec:
    resources: ...
    provider_options:
      platform: ...
    metadata: ...
```

### Environment Lifecycle

`MiniSWESandboxEnvironment.__init__()`:

- Validates that a sandbox provider was configured.
- Builds a `SandboxSpec` from the task image, environment variables, metadata,
  resources, and provider-specific options.
- Adds standard metadata such as `nemo_gym_agent=mini_swe_agent_2` and
  `instance_id`.
- Creates a `Sandbox` facade and calls `Sandbox.start(...)`.

`execute()`:

- Receives mini-swe-agent's command action.
- Applies the configured working directory and timeout.
- Optionally wraps the command in `conda activate <env>` for SWE-bench images
  that expect a prebuilt conda environment.
- Calls `Sandbox.exec(...)` as the configured user, root by default.
- Returns mini-swe-agent's expected sync response shape:

```python
{
    "output": "...",
    "returncode": 0,
    "exception_info": "",
}
```

`_check_finished()` preserves mini-swe-agent's submit sentinel behavior. If the
command output begins with `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` and the
command succeeded, it raises `minisweagent.exceptions.Submitted` with the final
submission payload.

`cleanup()` calls `Sandbox.stop(...)` to release provider-owned resources and
stop the sync facade's private loop.

## Contributing

Please refer to the main NeMo Gym documentation for contributing guidelines.

## Licensing Information

- **Code**: Apache 2.0
- **SWE-bench Verified**: MIT

### Dependencies

- **nemo_gym**: Apache 2.0
- **mini-swe-agent**: MIT
- **SWE-bench / swebench**: MIT
