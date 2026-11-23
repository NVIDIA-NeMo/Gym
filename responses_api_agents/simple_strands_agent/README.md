# Simple Strands Agent

The pinned Simple Strands Agent dependency is installed with the agent server. The native system
prompt is retained and additional system prompts are appended.

## Agent sessions with tools in the task sandbox

[configs/simple_strands_agent_sandboxed.yaml](configs/simple_strands_agent_sandboxed.yaml) runs the agent
under a `single_agent_turn_legacy` Environment Server, for benchmarks whose Resources Server supplies
`SandboxAccess`, such as SWE-bench Pro. Capability level: **Evaluation**.

```yaml
config_paths:
  - resources_servers/swebench_pro/configs/swebench_pro.yaml
  - responses_api_agents/simple_strands_agent/configs/simple_strands_agent_sandboxed.yaml
  - environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml

single_agent_turn_legacy:
  environment_servers:
    single_agent_turn_legacy:
      resources_server:
        name: swebench_pro_resources_server
      agent_server:
        name: simple_strands_agent
      resources_tool_transports: []
```

- **Execution.** The agent loop and its model calls stay on the agent server, in the same runner
  process as the host path. The runner's `SandboxEnvironment` sends every `bash` and
  `str_replace_editor` operation to the task sandbox through the sandbox provider, in
  `SandboxAccess.workdir` with `SandboxAccess.env`, so nothing is installed in the sandbox. It needs
  only bash, and coreutils `timeout` to bound each command: without it, a command that outlives its
  deadline is reported as exit 124 but may keep running, since a provider deadline such as Docker's
  stops only the local client. The provider configuration reaches the runner on stdin, never on disk.
- **Network.** An agent-owned sandbox keeps the provider's egress: there is no `network_access`
  setting yet. Model calls stay on the agent server, so the sandbox itself needs no network.
- **Lifecycle.** Seed connects to the borrowed sandbox (or starts an agent-owned one with
  `sandbox_provider`) and checks the workdir. One activation runs the agent once, and identical retries
  join it. Close stops the runner process group if it is still running, then disconnects from a
  borrowed sandbox or stops an owned one. A tool command running when the runner stops ends at its
  `shell_timeout` where coreutils `timeout` exists. An agent-owned sandbox needs a connectable provider
  (docker, apptainer, opensandbox or e2b), since the runner reconnects to it from its own process.
- **Refusals.** Required tool grants, a `SandboxAccess` user override, a relative workdir and an
  agent-owned sandbox on a provider that cannot reconnect are refused at seed with HTTP 422. `/run` needs `resources_server` (the host path).
- **Results.** Output and usage are converted as on the host path. When SSA's iteration limit (`max_turns`)
  or the `timeout` deadline ends the run, the transcript so far is returned as an `incomplete` response,
  which is still verified. At the deadline that is the transcript SSA last saved, every four messages.
  A model request's HTTP error fails as `model_request_error` with its status, and any other runner
  failure, including a sandbox outage, fails as `unknown`, which is retried. Model-call ownership and tool
  timing are reported as observation gaps.
