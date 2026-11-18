# Prime Agent

Runs [Prime Agent](https://github.com/PrimeIntellect-ai/prime-agent) in headless JSON mode.

On the host path, Prime Agent must be on `PATH`. The adapter installs `prime_agent_version` with
the official installer on the first host request when the default command is missing; agent
sessions install Prime Agent inside the task sandbox instead (see below).

Set `policy_base_url`, `policy_api_key`, and `policy_model_name` in `env.yaml`.

```bash
gym env start --environment prime_agent_math --model-type openai_model

gym eval run --no-serve --agent prime_agent_math_agent \
  --input environments/prime_agent_math/data/example.jsonl \
  --output environments/prime_agent_math/data/example_rollouts.jsonl \
  --limit 5
```

Math and reasoning gym example environments are in `environments/prime_agent_math` and
`environments/prime_agent_reasoning_gym`.

Each request uses an isolated `HOME` and one-shot Prime Agent worker. `kernel_venv` is shared so
IPython dependencies are reused.

`model` uses `<provider>/<model-id>`. Define the provider in `models_config` or set `model_server`
to use a Gym model server. See `configs/prime_agent.yaml` for all options.

## Agent sessions in task sandboxes

Capability level: Evaluation. A real-model smoke on SWE-bench Pro scored 3 of 3 tasks and resolved 2.

[configs/prime_agent_sandboxed.yaml](configs/prime_agent_sandboxed.yaml) runs Prime Agent **0.7.0**
inside the task sandbox of a `single_agent_turn` Environment Server, for benchmarks whose Resources
Server supplies `SandboxAccess`, such as SWE-bench Pro. Training token-ID capture is off by default. With
`model_server`, model calls go to the Gym Model Server under the rollout's prefix, so capture needs only run-level
`token_id_capture.enabled: true` and either the agent's `token_id_capture: true` or run-level
`token_id_capture.all_agents: true`.

```yaml
config_paths:
  - resources_servers/swebench_pro/configs/swebench_pro.yaml
  - responses_api_agents/prime_agent/configs/prime_agent_sandboxed.yaml
  - environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml

single_agent_turn_legacy:
  environment_servers:
    single_agent_turn_legacy:
      resources_server:
        name: swebench_pro_resources_server
      agent_server:
        name: prime_agent
      resources_tool_transports: []
```

- **Runtime.** Session setup installs a private Node.js 22.23.2 and the `prime-agent-0.7.0.tgz`
  release package with npm (`--ignore-scripts`) under `/tmp/nemo-gym-prime-runtime-0.7.0-node-22.23.2`,
  checks the version it reports, and reuses a ready runtime. The agent server downloads the official
  Node.js build and the release package (GitHub release) once into its asset cache, checks the SHA-256 values pinned in
  `sandbox.py`, and uploads them; the sandbox checks them again. The kernel's uv archive has no pinned checksum. Prime Agent runs as that Node with the
  package's CLI, so the task's Node and `PATH` are untouched. The sandbox needs a glibc image, bash,
  `sha256sum`, `tar`, `gzip`, `flock`, `mktemp` and network access to the npm registry and the release host (for the package's
  own dependencies).
- **Kernel.** Prime Agent's only built-in tool is an IPython kernel. Setup also installs a private
  Python (the mini-SWE bootstrap build, uploaded with its uv) under the runtime's `kernel` directory,
  with the packages Prime Agent's own kernel bootstrap installs, pinned in `install_prime_runtime.sh`,
  and the package's `prime-agent-runtime`, then points `PRIME_AGENT_KERNEL_PYTHON` at it. uv installs
  them from the package index, so the sandbox also needs network access to PyPI. The kernel takes about
  440 MB. IPython and Jupyter may write their state under the task's `HOME`.
- **Network.** A sandbox the agent creates keeps the provider's egress: `network_access` accepts only
  `inherit`, since setup needs the npm registry, the release host and PyPI. A borrowed sandbox keeps
  its lender's network.
- **Isolation.** Models, tools and state live under `PRIME_AGENT_CODING_AGENT_DIR` in a private
  session directory, so tool shells keep the task's `HOME`. `SandboxAccess.env` is forwarded except
  `PRIME_AGENT_*`, `PI_*` and `OPENAI_*`. Close removes only the session directory.
- **Requests.** The user message is the task and the system or developer message, `instructions`
  and `system_prompt` are appended to the system prompt; both are argv values under 128 KiB. Other
  request fields are refused with HTTP 422. `command`, `extra_args` and `env` overrides, required
  tool grants and a `SandboxAccess` user override are refused at seed.
- **Results.** Output and usage come from Prime Agent's `message_end` events. A model error that
  Prime Agent retries is skipped, and a run whose final message is an error stop fails with the model request's status when Prime Agent reports one (for example
  `model_request_error` 503), a `length` stop is `incomplete`, a supervisor timeout with output is
  `incomplete`, as is a rollout stop at any point. A supervisor timeout before any model reply, a
  nonzero exit or a run without an answer fails. A failed stdout read raises, so the run is retried. Model-call ownership, tool
  timing and child invocations are reported as observation gaps.
