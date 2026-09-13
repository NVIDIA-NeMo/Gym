# Harbor Terminus 2 Agent

This agent runs Harbor's `Terminus2` control loop in the task sandbox supplied by
the NeMo Gym resources server. It adapts the small Harbor environment interface
that Terminus uses (`exec` and `is_dir`) to `AsyncSandbox`; task state therefore
remains owned by the resources server.

Before launching, complete [OpenSandbox access and setup](https://docs.nvidia.com/nemo/gym/main/infrastructure/sandbox/opensandbox#setup)
for sandbox credentials, endpoint configuration, and resource limits.

```bash
gym env start \
    --config responses_api_models/vllm_model/configs/vllm_model.yaml \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    --config responses_api_agents/terminus_2_sandboxed_agent/configs/terminus_2_sandboxed_agent.yaml \
    --config resources_servers/terminal_bench_2_1/configs/terminal_bench_2_1.yaml \
    ++terminus_2_sandboxed_agent.responses_api_agents.terminus_2_sandboxed_agent.resources_server.name=terminal_bench_2_1_resources_server
```

To run one row from a benchmark JSONL after starting the servers:

```bash
gym eval prepare --config benchmarks/terminal_bench_2_1/terminus_2.yaml

python responses_api_agents/terminus_2_sandboxed_agent/client.py \
    +benchmark_jsonl=benchmarks/terminal_bench_2_1/data/benchmark.jsonl
```

The agent calls the configured model server exclusively through the Responses
API. Its returned response contains every model request and response from the
Terminus trajectory. Set `dump_trajectory: true` to also have Harbor write its
per-turn JSON trajectory files; it is `false` by default.

## Tmux binary: online or pre-staged

With `remote_tmux_binary_path: null`, [Harbor's setup](https://github.com/laude-institute/harbor/blob/v0.22.0/src/harbor/agents/terminus_2/tmux_session.py)
uses tmux on `PATH` or attempts installation. Installation needs permissions,
dependencies and network access to package repositories or source downloads.

To preinstall tmux, expose a compatible binary through a mount, task image, or
custom resources-server upload before the agent runs. For S3-hosted files, arrange
a mount or transfer into each task sandbox. For OpenSandbox, configure
[volume options](https://docs.nvidia.com/nemo/gym/main/infrastructure/sandbox/opensandbox#sandboxspec-provider-options)
under `terminal_bench_2_1_resources_server.resources_servers.terminal_bench_2_1.sandbox_config.provider_options`;
the resources server creates the sandbox and the agent reconnects to it.

Save this as `offline-assets.yaml` and add `--config offline-assets.yaml` to server
startup. Use the binary's path inside the task sandbox:

```yaml
terminus_2_sandboxed_agent:
  responses_api_agents:
    terminus_2_sandboxed_agent:
      remote_tmux_binary_path: /opt/gym-assets/tmux/3.7c/tmux-3.7c-linux-x86_64
```

The agent copies the binary to `/usr/local/bin/tmux`, so its sandbox user needs write
access there. An existing tmux earlier on `PATH` can take precedence.
