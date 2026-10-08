# OpenClaw Agent

Runs OpenClaw CLI (`openclaw agent --local --json`).
OpenClaw runs its own tools internally.
Resources server is used for verifier.

Minimal, meant to be extended, and currently eval-only. 

## Quick start

OpenClaw must be installed (or it is auto-installed on first start). 
Make sure `env.yaml` is also set.

```bash
gym env start \
  --config environments/openclaw_math/config.yaml \
  --model-type openai_model

gym eval run --no-serve --agent openclaw_math_agent \
  --input environments/openclaw_math/data/example.jsonl \
  --output openclaw_rollout.jsonl --limit 3
```

## Model id

OpenClaw drops the leading `<provider>/` to form the upstream id,
so we include an extra prefix, such as for `nvidia/...` ids:

```yaml
model: nvinf/nvidia/meta/llama-3.3-70b-instruct
openclaw_config:
  models:
    providers:
      nvinf:
        api: openai-completions
        baseUrl: ${policy_base_url}
        apiKey: ${policy_api_key}
        models:
        - {id: nvidia/meta/llama-3.3-70b-instruct, name: nvidia/meta/llama-3.3-70b-instruct, api: openai-completions}
```

Alternatively, set `model_server` to a Gym model server and set `model` to its served model id. The
agent creates the OpenClaw provider entry automatically. Without `model_server`, the existing
provider configuration is unchanged.

## Config fields

- `concurrency`: max simultaneous `run()` calls
- `command`: the OpenClaw command, split on spaces so a multi-word launcher works (e.g. `npx openclaw`)
- `model`: `<provider>/<model-name>` (see Model id)
- `model_server`: optional Gym model server used to generate the provider entry
- `context_window`: context limit for a generated model entry
- `max_output_tokens`: output limit for a generated model entry
- `workspace_root`: where per-request workspaces are created and deleted
- `openclaw_agent_id`: passed to `--agent`
- `thinking`: passed to `--thinking` (off, low, medium, high, ...)
- `system_prompt`: prepended to the user message
- `setup_timeout`: seconds for `openclaw setup`
- `timeout`: seconds for the `openclaw agent` run
- `extra_args`: extra flags appended to `openclaw agent`
- `env`: extra env vars for the subprocess (e.g. provider API keys)
- `openclaw_config`: deep-merged into the generated `openclaw.json`
- `openclaw_version`: exact version to pin on install (npm ranges such as `^2026.9.0`
  are rejected); overridden by the `OPENCLAW_VERSION` env var, and falls back to
  `setup_openclaw.DEFAULT_OPENCLAW_VERSION`. An already-installed `openclaw` is only
  reused when `openclaw --version` reports exactly the resolved version; anything else
  is reinstalled, so changing the override or the config pin takes effect on the next
  startup. After an install, the launcher selected on `PATH` must report the requested
  version, or startup fails.
  Note: releases ≥ 2026.9.0 store session history in SQLite instead of JSONL, which
  the agent's transcript reader does not support yet — tool results and interrupted
  sessions are not captured, so stick to 2026.6.11 until that lands.
- `node_bin_dir`: directory put before `PATH` when running `openclaw`. Setup uses the
  same order for its version probes and for `npm`, so a bundled runtime is checked
  as the rollout will use it.
- `OPENCLAW_NODE_VERSION` (env only): Node.js version fetched when `npm` is absent
  (default `24.21.0`, the newest release of the Node 24 LTS line OpenClaw supports).
  The build is picked for the host platform — Linux, macOS and Windows on x64 and
  arm64. Other platforms raise, since nodejs.org publishes no build for them;
  install Node.js yourself and put `npm` on `PATH`.

Runtime compatibility is validated before reuse: a system `npm` is only used when
its `node` satisfies the `engines.node` range of the requested OpenClaw release
(`setup_openclaw.OPENCLAW_ENGINES_NODE` for pinned releases, `npm view` for
others), and a cached local toolchain is only
reused when it reports exactly the requested Node version. Incompatible runtimes
are replaced, not bypassed silently.

See `configs/openclaw_agent.yaml`.
