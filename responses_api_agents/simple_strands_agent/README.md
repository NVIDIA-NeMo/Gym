# Simple Strands Agent

The pinned Simple Strands Agent dependency is installed with the agent server. The native system
prompt is retained and additional system prompts are appended.

## Agent sessions with tools in the task sandbox

[configs/simple_strands_agent_sandboxed.yaml](configs/simple_strands_agent_sandboxed.yaml) runs the agent
under a `single_agent_turn` Environment Server, for benchmarks whose Resources Server supplies
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
  `SandboxAccess.workdir` with `SandboxAccess.env`, so nothing is installed in the sandbox; it needs
  only bash (and coreutils `timeout` for per-command SIGTERM; the provider deadline bounds commands
  without it). The provider configuration reaches the runner on stdin, never on disk.
- **Lifecycle.** Seed connects to the borrowed sandbox (or starts an agent-owned one with
  `sandbox_provider`) and checks the workdir. One activation runs the agent once; identical retries
  join it. Close stops the runner process group if it is still running, then disconnects from a
  borrowed sandbox or stops an owned one. A tool command running when the runner stops ends at its
  `shell_timeout`.
- **Refusals.** Required tool grants, a `SandboxAccess` user override and a relative workdir are
  refused at seed with HTTP 422. `/run` needs `resources_server` (the host path).
- **Results.** Output and usage are converted as on the host path. When SSA's iteration limit (`max_turns`)
  ends the run, the transcript so far is returned as an `incomplete` response, which is still verified. Model-call ownership and tool
  timing are reported as observation gaps.
