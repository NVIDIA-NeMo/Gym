# Mini-SWE-Agent v2 Sandbox Environment

This note explains how `sandbox_environment.py` is used when the Gym
`mini_swe_agent_2` runs mini-swe-agent v2 with `env: sandbox`.

## Where It Fits

The Gym agent entrypoint is `responses_api_agents/mini_swe_agent_2/app.py`.
For each `/run` request, `MiniSWEAgent.run()` builds the mini-swe-agent
configuration and launches `run_swegym_with_optional_sandbox()` in a Ray task.

When `env` is `sandbox`, Gym injects the sandbox provider and sandbox spec into
the per-instance mini-swe-agent config:

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

mini-swe-agent v2 then calls:

1. `get_environment(environment_config)`
2. `DefaultAgent(model, env, **agent_config)`
3. `agent.run(problem_statement)`
4. `env.execute(...)` once per tool command
5. Gym calls `env.cleanup()` in a `finally` block

`MiniSWESandboxEnvironment` is the adapter that lets that synchronous
mini-swe-agent environment contract use Gym's sync sandbox facade.

## Environment Lifecycle

`MiniSWESandboxEnvironment.__init__()`:

- Validates that a sandbox provider was configured.
- Builds a `SandboxSpec` from the task image, environment variables, metadata,
  resources, platform, volumes, and provider-specific extensions.
- Applies Gym image rewrites before creating the sandbox.
- Adds standard metadata such as `nemo_gym_agent=mini_swe_agent_2` and
  `instance_id`.
- Creates a `Sandbox` facade and calls `Sandbox.create(...)`.

`execute()`:

- Receives mini-swe-agent's command action.
- Applies the configured working directory and timeout.
- Optionally wraps the command in `conda activate <env>` for SWE-bench images
  that expect a prebuilt conda environment.
- Calls `Sandbox.exec(...)`.
- Returns mini-swe-agent's expected sync response shape:

```python
{
    "output": "...",
    "returncode": 0,
    "exception_info": "",
}
```

`_check_finished()`:

- Preserves mini-swe-agent's submit sentinel behavior.
- If the command output begins with `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` and
  the command succeeded, it raises `minisweagent.exceptions.Submitted` with the
  final submission payload.

`cleanup()`:

- Calls `Sandbox.close(..., delete=config.delete)`.
- Calls `Sandbox.shutdown()` to release provider-owned async resources and stop
  the sync facade's private loop.

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

## Can This Environment Be Natively Async?

Not without changing the mini-swe-agent integration boundary.

A natively async environment would be cleaner from Gym's point of view, but the
current mini-swe-agent v2 contract is sync. To make `MiniSWESandboxEnvironment`
natively async, one of these would need to happen:

- mini-swe-agent upstream adds an async environment protocol and awaits
  `execute`, `cleanup`, and possibly environment construction.
- Gym forks or wraps the mini-swe-agent v2 agent loop with an async-aware runner.
- Gym moves sandbox orchestration outside mini-swe-agent's environment object
  and exposes only sync command execution back to mini-swe-agent.

Until then, the sync `Sandbox` facade is the smallest compatibility layer. It
keeps the official mini-swe-agent v2 agent loop untouched while still letting
Gym use the async sandbox provider API everywhere under the hood.

## Smoke Validation

The refactored `MiniSWESandboxEnvironment` path was smoke-tested on Kubernetes
with mini-swe-agent v2, OpenSandbox SDK mode, `tool_choice=auto`, and one
Qwen3.5 27B vLLM replica.

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
create/exec failures. This validates that the mini-swe-agent v2 harness can use
the Gym sandbox API sync facade end to end for SWE-bench rollouts.

## Model Generation Budget Gotcha

Keep the requested generation budget compatible with the live vLLM deployment.
During smoke testing, an earlier run used `max_output_tokens=49152` against a
single-replica Qwen3.5 deployment started with `--max-model-len 32768`. vLLM
rejected the request because the requested output token budget exceeded the
served model length. The Gym vLLM proxy converted that upstream failure into an
empty chat completion, and mini-swe-agent v2 surfaced it as repeated:

```text
No tool calls found in the response. Every response MUST include at least one tool call.
```

That symptom was not a sandbox failure and not a reason to force the `bash`
tool. The successful smoke kept `tool_choice=auto` and lowered
`max_output_tokens` to `16384`.

## When To Revisit

Revisit this adapter if mini-swe-agent v2 gains native async environment
support. At that point this environment can switch from `Sandbox` to
`AsyncSandbox`, make creation explicit through an async factory, and expose
async `execute` and `cleanup` methods directly.
