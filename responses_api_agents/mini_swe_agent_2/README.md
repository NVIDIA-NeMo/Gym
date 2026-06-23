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
supported eval path is `/run`, typically via `ng_collect_rollouts`.

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

### Agent Configuration

Path - `responses_api_agents/mini_swe_agent_2/configs/mini_swe_agent_opensandbox.yaml`

```yaml
mini_swe_agent_2:
  responses_api_agents:
    mini_swe_agent_2:
      entrypoint: app.py
      domain: coding
      description: Software engineering tasks driven by mini-swe-agent harness on OpenSandbox.
      value: Improve agentic software engineering capabilities.
      model_server:
        type: responses_api_models
        name: policy_model
      concurrency: 64
      env: sandbox
      sandbox_provider:
        opensandbox:
          connection:
            domain: opensandbox-server.opensandbox-system.svc.cluster.local
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
          sandbox-api: opensandbox-sdk
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

Optional `sandbox_resource_profiles` can be configured as a list of resource
maps. When present, the agent hashes `instance_id` and deterministically merges
one profile into `sandbox_spec.resources`. This is useful for spreading
SWE-bench tasks across a small set of resource sizes without changing the input
data.

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

Start the mini-swe-agent 2 server with the OpenSandbox provider and a policy
model server. The values below show a representative SWE-bench eval setup:

```bash
CONFIG_PATHS="responses_api_agents/mini_swe_agent_2/configs/mini_swe_agent_opensandbox.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml"

ng_run "+config_paths=[$CONFIG_PATHS]" \
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
ng_collect_rollouts \
    +agent_name=mini_swe_agent_2 \
    +input_jsonl_fpath=data/mini_swe_verified_smoke8.jsonl \
    +output_jsonl_fpath=results/mini_swe_agent_2_pass8.jsonl \
    +limit=8 \
    +num_repeats=8 \
    +num_samples_in_parallel=64 \
    '+responses_create_params={max_output_tokens: 32768, temperature: 0.6, top_p: 0.95, metadata: {chat_template_kwargs: "{\"enable_thinking\": true}"}}'
```

`ng_collect_rollouts` also writes
`results/mini_swe_agent_2_pass8_aggregate_metrics.json`
with per-task eval status, pass@k, resolved task counts, and eval error rates.
After collecting repeated rollouts, run `ng_reward_profile` on the collected
output when you want the standalone profiler JSONL as well:

```bash
ng_reward_profile \
    +input_jsonl_fpath=data/mini_swe_verified_smoke8.jsonl \
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

When `env` is `sandbox`, Gym injects this environment config before calling
mini-swe-agent:

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
