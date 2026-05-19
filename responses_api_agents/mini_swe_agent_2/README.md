# Mini-SWE-Agent 2 Sandbox Agent

`mini_swe_agent_2` is the Gym integration for mini-swe-agent v2. It runs
mini-swe-agent's synchronous SWE-bench harness while creating and executing the
task environment through the public `nemo_gym.sandbox` API.

This package intentionally keeps only the sandbox path. It does not carry over
the older Docker/Singularity mini-SWE integration.

## Current Path

The code in this directory is wired for:

- mini-swe-agent `2.1.0`
- SWE-bench task rows, including SWE-bench Verified
- `env: sandbox`
- `responses_api_agents.mini_swe_agent_2.sandbox_environment.MiniSWESandboxEnvironment`
- OpenSandbox through `nemo_gym.sandbox.providers.opensandbox`
- OpenTelemetry sandbox observability through `nemo_gym.sandbox.observability`

Use `configs/mini_swe_agent_opensandbox.yaml` as the Gym server config.

## Code Map

- `app.py` defines the Gym `MiniSWEAgent` FastAPI server and `/run` endpoint.
- `sandbox_environment.py` adapts mini-swe-agent's sync environment interface to
  `nemo_gym.sandbox.Sandbox`.
- `configs/mini_swe_agent_opensandbox.yaml` is the OpenSandbox-backed server
  config.
- `tests/test_app.py` covers request conversion, config generation, Ray runner
  wiring, response shaping, reward extraction, and observability spans.
- `tests/test_sandbox_environment.py` covers mini-swe-agent submit sentinel
  handling.

`MiniSWEAgent.setup_webserver()` also registers `/v1/responses`, but
`MiniSWEAgent.responses()` is intentionally not implemented in this agent. The
supported execution path is `/run`.

## Run Flow

For each `/run` request, `MiniSWEAgent.run()`:

1. Reads the policy model server from Gym global config.
2. Loads mini-swe-agent's built-in `swebench.yaml` config.
3. Converts relevant Responses API rollout parameters into mini-swe-agent
   `model.model_kwargs`.
4. Injects the sandbox provider, sandbox spec, and sandbox environment kwargs.
5. Writes a per-instance mini-swe-agent config to
   `results/<subset>/<policy_model_name>/_configs/<instance_id>.sandbox.yaml`.
6. Launches `run_swegym_with_optional_sandbox()` in a Ray remote task.
7. Converts the saved mini-swe-agent trajectory back into Gym's Responses API
   shape.
8. Runs SWE-bench grading and returns reward `1.0` only when the report says the
   instance resolved and includes test status.

Inside the Ray task, `_run_swegym_v2()` calls mini-swe-agent v2 roughly as:

```python
env = get_environment(environment_config)
model = get_model(config=model_config)
agent = DefaultAgent(model, env, **agent_config)
info = agent.run(instance["problem_statement"])
eval_report = _run_eval_v2(...)
env.cleanup()
```

`run_golden: true` skips model rollout, applies the task's gold patch, and then
runs evaluation.

## Configuration

The server config must set `env: sandbox` and provide `sandbox_provider`.
`sandbox_spec` and `sandbox_environment_kwargs` are optional but normally needed
for SWE-bench images.

Example shape:

```yaml
mini_swe_agent_2:
  responses_api_agents:
    mini_swe_agent_2:
      entrypoint: app.py
      model_server:
        type: responses_api_models
        name: policy_model
      concurrency: 64
      env: sandbox
      sandbox_provider:
        name: opensandbox
        kwargs:
          domain: opensandbox-server.opensandbox-system.svc.cluster.local
          api_key: ${oc.env:OPENSANDBOX_API_KEY}
          protocol: http
          use_server_proxy: true
      sandbox_spec:
        timeout_s: 18000
        ready_timeout_s: 1200
        resources:
          cpu: "1"
          memory: 8Gi
          ephemeral-storage: 20Gi
        platform:
          os: linux
          arch: amd64
        image_rewrites:
        - from: swebench/
          to: mirror.gcr.io/swebench/
        metadata:
          benchmark: swebench-verified
          harness: mini-swe-agent
      sandbox_environment_kwargs:
        cwd: /testbed
        conda_env: testbed
        activate_conda: true
        user: root
        delete: true
      step_timeout: 600
      eval_timeout: 1800
      step_limit: 250
```

`sandbox_resource_profiles` can be configured as a list of resource maps. When
present, the agent hashes `instance_id` and deterministically merges one profile
into `sandbox_spec.resources`. This is useful for spreading SWE-bench tasks
across a small set of resource sizes without changing the input data.

## Model Parameters

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

## Sandbox Environment

`MiniSWESandboxEnvironment` is the adapter that lets mini-swe-agent's
synchronous environment contract use Gym's sandbox facade.

When `env` is `sandbox`, Gym injects this environment config before calling
mini-swe-agent:

```yaml
environment:
  environment_class: responses_api_agents.mini_swe_agent_2.sandbox_environment.MiniSWESandboxEnvironment
  image: <swebench task image>
  provider:
    name: opensandbox
    kwargs: ...
  spec:
    resources: ...
    platform: ...
    metadata: ...
```

`MiniSWESandboxEnvironment.__init__()`:

- Validates that a sandbox provider was configured.
- Builds a `SandboxSpec` from the task image, environment variables, metadata,
  resources, platform, volumes, provider-specific extensions, and health-check
  settings.
- Applies Gym image rewrites before creating the sandbox.
- Adds standard metadata such as `nemo_gym_agent=mini_swe_agent_2` and
  `instance_id`.
- Creates a `Sandbox` facade and calls `Sandbox.create(...)`.

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

`cleanup()` calls `Sandbox.close(..., delete=config.delete)` and then
`Sandbox.shutdown()` to release provider-owned async resources and stop the sync
facade's private loop.

## Why The Sandbox Facade Has A Loop Runner

Gym exposes two public sandbox classes:

- `AsyncSandbox` is the async-native API for Gym servers and high-concurrency
  rollout code.
- `Sandbox` is the sync facade for synchronous integrations such as
  mini-swe-agent v2.

The provider layer remains async by design. Provider calls such as `create`,
`exec`, `read_file`, `write_file`, and `close` may perform network I/O and
should not block a shared event loop.

mini-swe-agent v2's environment API is synchronous today. It constructs the
environment synchronously and calls `env.execute(...)` as a normal blocking
method from `DefaultAgent.run(...)`. If `execute()` returned a coroutine,
mini-swe-agent would not await it, and the agent would break.

`Sandbox` owns the sync-to-async bridge:

- mini-swe-agent sees a normal synchronous environment.
- All Gym sandbox provider calls run on one dedicated asyncio loop.
- The same loop is used for create, exec, and cleanup, which matters because
  SDK clients and handles can be event-loop-affine.
- The facade avoids calling `asyncio.run()` for every command, which would
  create and destroy event loops repeatedly and can fail if a loop is already
  running in the current thread.

## Image Selection

If the input instance has `image_name`, the agent uses it directly. Otherwise it
derives the SWE-bench image from `instance_id` and `subset`:

- `subset: verified` uses `swebench/sweb.eval.x86_64.<id>:latest` with `__`
  replaced by `_1776_`.
- Other subsets use `xingyaoww/sweb.eval.x86_64.<id>:latest` with `__` replaced
  by `_s_`.

Configured `sandbox_spec.image_rewrites` then apply inside
`MiniSWESandboxEnvironment`, for example rewriting `swebench/` to
`mirror.gcr.io/swebench/`.

## Smoke Validation

The sandbox environment path was smoke-tested on Kubernetes with mini-swe-agent
v2, OpenSandbox SDK mode, `tool_choice=auto`, and one Qwen3.5 27B vLLM replica.

Run:

```text
job: hemild-mini-swe2-sandbox-16k-r64xf
run_dir: /mnt/rl-workspace/hemild/gym_eval/refactor/runs/mini_swe_sandbox_environment_smoke/20260518-033858-mini-swe2-sandbox-smoke-direct
```

Result:

```text
rows: 4
reward_sum: 4.0
pass@1: 100.0%
wall_time_s: 343
```

Resolved instances:

- `pytest-dev__pytest-6202`
- `sympy__sympy-15809`
- `django__django-13410`
- `django__django-16429`

The pod completed without restarts, and the logs did not show
`SandboxApiException`, `TimeoutError`, image pull failures, or OpenSandbox
create/exec failures.

## When To Revisit

Revisit this adapter if mini-swe-agent v2 gains native async environment
support. At that point this environment can switch from `Sandbox` to
`AsyncSandbox`, make creation explicit through an async factory, and expose
async `execute` and `cleanup` methods directly.
