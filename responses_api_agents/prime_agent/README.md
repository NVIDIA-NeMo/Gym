# Prime Agent

Runs [Prime Agent](https://github.com/PrimeIntellect-ai/prime-agent) in headless JSON mode.

Prime Agent must be on `PATH`. The adapter installs `prime_agent_version` with the official
installer when the default command is missing.

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

## SDK child sessions

`child_app.py` is an evaluation-only entrypoint of this adapter for Prime Agent **0.7.0**
(public revision `be9e2fa0714e7cd1c6bd9bdb1b554d2cc6550387`). It creates a depth-one
child with Prime's own IPython tool and agent-message skill. The parent is a recording
controller; it makes no model calls. This mode requires a borrowed task sandbox and
uses the agent-session API through `single_agent_turn` or `single_agent_turn_legacy`.
The Resources Server prepares the repository and destroys the sandbox after verification.

The implementation calls the public SDK; it does not vendor Prime's source. See
[the pinned SDK](https://github.com/PrimeIntellect-ai/prime-agent/blob/be9e2fa0714e7cd1c6bd9bdb1b554d2cc6550387/packages/coding-agent/src/core/sdk.ts).
Release archives are checked against SHA-256 on the Agent Server before upload.
Installation requires Node 22, npm, Python 3 with venv support, bash, and network
access to npm/PyPI. The paired task image is `node:22-bookworm`; its resource setup
requires root for system dependencies. Python and npm transitive dependencies are
not locked yet; this entrypoint remains experimental pending reproducible runtime
packaging and the remaining harness merge checks.

Child configuration uses `model_server`, `model`, `prime_agent_version`,
`context_window`, `max_output_tokens` (per model call), `max_turns` (complete turns,
including the last turn's tools), `temperature`, `timeout`, `kernel_packages`,
`sandbox_install_timeout_seconds`, and `session_close_timeout_seconds`.
`capture_directory`, when set, exports controller receipts into the Resources Server's
sandbox for verification; `artifact_directory` retains raw SDK captures on the Agent Server.
Runtime files live outside the task repository and are removed at close.
The child inherits the task's HOME and does not rewrite repository files for setup.

Only one user text prompt, optionally preceded by a system prompt, is supported.
Request-level sampling, tools, supplied multi-turn history, sandbox user/environment
overrides, continuation, and required external tool grants are rejected. Optional
external grants are unused. Host CLI options (`command`, `extra_args`, `env`,
`workspace_root`, `kernel_venv`, `models_config`, and `thinking`) belong to `app.py`;
the child always uses its pinned SDK, isolated runtime, and thinking off.

All model requests use Gym's rollout-prefixed model URL. Until the parent receives
a report, the runtime forces the registered IPython tool; after delivery it allows
an automatic final response. Compaction, auto-refinement, and retries are disabled.
The 65,536-token output reservation shrinks to fit remaining context with an
8,192-token margin. Do not remove or override that per-call cap in the Model Server.

Capability level: **Evaluation**. Close returns explicit observation gaps; this mode
claims neither training nor full trajectory conformance. Raw SDK events, controller
receipts, effective prompts/tools, model requests, and supervisor cleanup receipts
are saved. Function-calling shell probes do not apply: the only exposed tool is
IPython. Terminal-Bench support is not claimed. The first pairing is Partner IF's
repository/report task. A second pairing and runtime reproducibility checks remain
required before this experimental entrypoint is ready to merge.
