# AgentVista evaluation agent

This adapter evaluates the public, pinned [AgentVista source](https://github.com/hkust-nlp/AgentVista/tree/a3385d51d0c3dd0c8501350a08d52065e3e94ab4)
through Gym's rollout-correlated model endpoint. Choose `harness: default` for the
original greedy loop or `harness: opencode` for the pinned OpenCode CLI and MCP
tools. Both use the original answer judge. This component is evaluation-only.

The Python code tool in **both harnesses** runs in a mandatory Linux namespace
sandbox. NeMo Gym's `AsyncSandbox` supervises a fixed Bubblewrap launcher; its
`local` provider alone provides no isolation. Bubblewrap creates the isolated
filesystem, process and network namespaces and clears the environment. There is
no host execution fallback. The adapter remains evaluation-only. Session lifecycle and verification run through
Gym’s Environment and Resources Servers.

The sandbox contains read-only Python runtime/libraries and three hash-verified
upstream tool modules, a temporary writable workspace, and one socket channel.
Task images cross that channel as bytes. Dataset files, episode configuration,
judge/tool credentials, host homes and the Docker socket are never mounted or
forwarded. Original stateful Jupyter execution and image variables are retained. Kernel
startup uses Jupyter's lifecycle API and a 60-second readiness handshake rather
than the original ten-second connection-file polling limit on shared storage.
Image results return as bytes; model-supplied paths never reach the host image
loader. Network access from model-written code is disabled; the original web
search/image search/visit tools remain trusted host-side services.

Bubblewrap (`bwrap`) and permitted unprivileged user namespaces are required on
the worker host/container. Both image builders install Bubblewrap. On systems
whose container policy disables nested namespaces, enable the approved namespace
policy or use a compatible worker; an unavailable sandbox makes the episode
unscorable rather than running Python on the host. Runtime prefixes must contain
only executable/library files, not task data or secrets. Namespace isolation
shares the host kernel and is not a VM or a claim of protection against kernel
vulnerabilities. Resource admission remains the deployment's responsibility.

## Configuration

| Key | Meaning |
|---|---|
| `source_root` | Verified upstream checkout; all 26 pinned files are checked per worker run |
| `dataset`, `image_folder`, `task_selector` | Original task records, images and selected stable IDs |
| `model_server`, `model_name` | Gym model-server reference and policy model name |
| `output_root`, `worker_python` | Isolated episode storage and separate worker interpreter |
| `code_sandbox_python` | Gym-installed interpreter supervising the sandbox; defaults to the agent-server interpreter |
| `harness` | `default` or `opencode` |
| `default_max_turns`, `default_max_images` | Original published run-command limits: 30 turns, 100 images |
| `command`, `opencode_version` | OpenCode executable and required version 1.17.11 |
| `concurrency`, `max_steps` | Episode slots and OpenCode step limit (defaults 1 and 150) |
| `context_tokens`, `output_tokens` | Context/output budgets (defaults 65536/8192) |
| `temperature`, `top_p` | Fallback sampling (0/1); row values override these defaults |
| `mcp_timeout_ms`, `agent_timeout`, `episode_timeout` | Tool, actor and active-slot worker deadlines |

The benchmark configs are `agentvista_default` and `agentvista_opencode`.
Gym’s `single_agent_turn_legacy` Environment Server adapts flat evaluation rows,
seeds an agent session, activates it once, closes it, and calls the AgentVista
Resources Server to judge the retained native predictions. The worker deadline
begins after slot admission. Responses carry source/task bindings, native trajectories,
execution-health metadata, extracted predictions and original judge diagnostics.
Empty model answers and actor deadlines with retained output are judged.
Infrastructure failures, an unavailable tool server, and judge outages stay
unscored. Scores from older harness versions must not be combined with version 5 (agent sessions and separate verification).

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

## Sandbox checks

`tests/test_code_sandbox.py` exercises real namespaces: answer-file and symlink
reads, host `/proc` access, credential environment exclusion, network separation,
and cancellation of detached descendants. Transport/image tests reject arbitrary
host image references and preserve sandbox failures as unscorable. The original
default-loop and pinned OpenCode CLI fixtures additionally check persistent
variables, image crops and image feedback. Run these with a complete pinned worker
Python environment and `AGENTVISTA_SANDBOX_PYTHON` pointing to the Gym-installed
interpreter when invoking workers directly (the Agent Server sets this in worker
configuration automatically). No benchmark prompts, task answers or judge rules
are changed.

## Session contract (version 5)

The agent implements seed, activation and close; it does not own `/run` or score
aggregation. Activate `/v1/responses` only with a seeded session. Prepared rows
carry hash-bound task identity in `agent_context.agent.values.verifier_metadata`.
Re-prepare older rows; version 4 rows are refused. Both benchmark configs now use
`single_agent_turn_legacy` for the flat-row API and `resources_servers/agentvista`
for verification and aggregation. Eval Factory's `--agent agentvista_agent`
selection and both harness names remain the same.

The native worker returns terminal judge inputs without calling the judge. Gym
removes `VERIFIER_*` credentials from its environment; the original loop's
terminal scoring callback records the exact assistant response list instead.
The Resources Server runs the unchanged judge separately with the original
question/reference and checks the task/image binding again. Standalone workers
retain their direct judging mode for compatibility; Gym always enables deferred
verification. Provider errors remain masked, and a measured wrong answer remains
an unmasked zero. The Resources Server never accepts a worker-assigned reward.

The adapter refuses borrowed task sandboxes and required external tool grants
with a terminal 422: its native tools and mandatory code namespace sandbox are
benchmark-specific. It supports one server worker because admission and retry
bindings live in that process. Close cancels an active worker before acknowledging
cleanup and returns an explicit observation gap: native transcripts are preserved,
but typed tool/model joins and complete usage are unavailable. Training,
checkpoint continuation, and arbitrary benchmark pairings are unsupported.
