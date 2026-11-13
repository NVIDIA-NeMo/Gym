# AgentVista evaluation agent

This adapter evaluates the public, pinned [AgentVista source](https://github.com/hkust-nlp/AgentVista/tree/a3385d51d0c3dd0c8501350a08d52065e3e94ab4)
through Gym's rollout-correlated model endpoint. Choose `harness: default` for the
original greedy loop or `harness: opencode` for the pinned OpenCode CLI and MCP
tools. Both use the original answer judge. This component is evaluation-only.

This is a trusted-host external-agent-loop integration. Its local Python tool
is not a sandbox: model-written code can read the worker's filesystem and process
environment. An environment-variable allowlist is not a security boundary.
Use dedicated disposable infrastructure and trusted credentials/data; do not use
this adapter for adversarial or reward-optimization training. A sandbox-backed
resources server and stock `opencode_agent` pairing is a separate migration.

## Configuration

| Key | Meaning |
|---|---|
| `source_root` | Verified upstream checkout; all 26 pinned files are checked per worker run |
| `dataset`, `image_folder`, `task_selector` | Original task records, images and selected stable IDs |
| `model_server`, `model_name` | Gym model-server reference and policy model name |
| `output_root`, `worker_python` | Isolated episode storage and separate worker interpreter |
| `harness` | `default` or `opencode` |
| `default_max_turns`, `default_max_images` | Original published run-command limits: 30 turns, 100 images |
| `command`, `opencode_version` | OpenCode executable and required version 1.17.11 |
| `concurrency`, `max_steps` | Episode slots and OpenCode step limit (defaults 1 and 150) |
| `context_tokens`, `output_tokens` | Context/output budgets (defaults 65536/8192) |
| `temperature`, `top_p` | Fallback sampling (0/1); row values override these defaults |
| `mcp_timeout_ms`, `agent_timeout`, `episode_timeout` | Tool, actor and active-slot worker deadlines |

The benchmark configs are `agentvista_default` and `agentvista_opencode`.
The legacy relay has no queue-wide deadline; the worker deadline begins after
slot admission. Responses carry source/task bindings, native trajectories,
execution-health metadata, extracted predictions and original judge diagnostics.
Empty model answers and actor deadlines with retained output are judged.
Infrastructure failures, an unavailable tool server, and judge outages stay
unscored. Scores from older harness versions must not be combined with version 3.

Install server dependencies with:

```bash
uv sync --locked --project responses_api_agents/agentvista_agent
gym env test +entrypoint=responses_api_agents/agentvista_agent
```

The server's managed lock does not install the optional image/code worker.
Build that worker from the compiled `requirements-worker.txt` lock, or use the
shared runtime-image build recipe. The `.in` file holds upgrade constraints;
both image builders install the same CPU-only, hashed dependency set.

Integration tests using only local Gym fixtures run without upstream assets.
To additionally exercise the original tools and real CLI, provision the pinned
upstream checkout and worker dependencies, then run:

```bash
export AGENTVISTA_ROOT="$WORKSPACE/agentvista-source"
export AGENTVISTA_TEST_OPENCODE="$WORKSPACE/bin/opencode"
pytest responses_api_agents/agentvista_agent/tests
```

These fixtures use local fake policy/judge endpoints and need no provider keys
or GPUs. For provider configuration, data preparation and evaluation commands,
see [the benchmark README](../../benchmarks/agentvista/README.md).
