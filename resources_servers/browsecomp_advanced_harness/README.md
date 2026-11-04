# BrowseComp Advanced Harness

RL environment for the [BrowseComp](https://openai.com/index/browsecomp/) benchmark — a challenging web-browsing research task
requiring multi-hop investigation through search and page-level browsing.

## Overview

This resources server provides browsing tools backed by [Tavily](https://tavily.com/) or Exa:

- **`search`** — parallel web search returning top results with content snippets
- **`browse`** — full-page content extraction from given URLs
- **`bash_command`** — the existing read-only pocketshell interpreter over retrieved pages, executed inside a
  resources-owned sandbox when `workspace: per_session` is enabled

Reward is computed via an LLM judge that compares the model's final answer against the ground truth.

## Key Config Parameters

Add to your `env.yaml` (or pass via CLI):

```yaml
tavily_api_key: <YOUR_TAVILY_KEY>   # or a list of keys for rotation
exclude_domains_file_path: /path/to/excluded_domains.json

judge_model_base_url: <YOUR_JUDGE_MODEL_URL>
judge_model_api_key: ""
judge_model_name: Qwen/Qwen3-235B-A22B-Instruct-2507
```

## Workspace sandbox

Workspace tools require a configured sandbox provider. Add these fields under the resources server's
`resources_servers` block and compose the corresponding provider configuration:

```yaml
workspace: per_session
sandbox_provider: sandbox
sandbox_config:
  image: python:3.13-slim
  workdir: /workspace
  ttl_s: 7200
sandbox_install_timeout_seconds: 900
sandbox_close_timeout_seconds: 60
```

For example, composing `nemo_gym/sandbox/providers/docker/configs/docker.yaml` defines the named `sandbox` provider.
The shared sandbox API also supports other providers; Enroot needs an image available on its cluster. Provider
`default_metadata` is merged with `sandbox_config.metadata`, then the resource server and session labels.

Each resources session owns one tool sandbox. An agent that runs its loop in a sandbox owns a separate sandbox;
neither component borrows the other's container. This works with providers such as Enroot that do not support
serializing and reconnecting to a sandbox. Search and browse retain their existing responses and write immutable
retrieved pages. Only those pages and their manifest are copied into the tool workspace. Search/judge credentials
and verifier inputs are not copied into the sandbox.

The shell uses the checked-out pocketshell implementation and `SandboxPythonRuntime` with the installed `regex`
version pinned into its isolated Python runtime. The first seed builds that runtime if needed; subsequent sessions
reuse its host-side artifact cache. Setup failures surface to the caller and retain ownership when cleanup fails,
so a later close can retry. Commands are supervised inside the sandbox; their guard, supported commands, output
truncation, and exit-code envelope are unchanged. There is no fallback to host execution after a sandbox failure.

`allow_host_execution: true` explicitly enables the former host interpreter for development when no provider is
configured, or permits the non-isolating `local` provider. It defaults to `false`; production and benchmark configs
must provide an isolating provider. Configurations with `workspace: none` do not need a tool sandbox.

## Sessions

Each rollout has its own session: the provider-call metrics that `/verify` reports and, with `workspace: per_session`, the disk workspace that `search` and `browse` write pages to and `bash_command` reads.

- **Environment Server.** `/seed_session` takes a `ResourcesSeedSessionRequest` and keys the session by its `resources_session_id`. The seed is refused with 422 when `task_data` lacks the `question` and `ground_truth` that `/verify` needs, or, with a workspace, when the ID is not a plain directory name. `/close_session` removes the workspace and the metrics, and is idempotent. A close that cannot remove the workspace fails and keeps the session, so the caller can retry it. After a close, a seed for the same ID is refused with 409, and a tool call for it gets 404 (`session_lost`) instead of recreating the workspace. `/verify` of a closed session returns a masked sample (`reward: 0`, `mask_sample: true`, `failure_kind: session_lost`). The reward depends only on the response and the task, so the mask is kept on purpose: a closed session reads the same on every server under the resources session contract. `single_agent_turn` verifies before it closes, so only an out-of-order verify gets the mask.
- **An agent's `/run`.** `/seed_session` takes the flat row and keys the session by the cookie. `/verify` releases the sandbox and removes the workspace, since this path sends no close.

`/verify` and `/close_session` stop the owned sandbox before deleting its workspace and metrics. A failed stop leaves
the session available for a close retry. Shutdown also stops any remaining tool sandboxes. Tool calls, verification,
and close are serialized per session, so cleanup cannot delete files while a tool is running. `/verify` drops the
session's metrics after reporting them, unless `dump_session_id_to_metrics_on_exit` keeps them for the dump at shutdown.

## Running

```bash
# If you want to run with browsecomp benchmark instead of the example samples, need to change the datasets part to the one like `benchmarks/browsecomp/config.yaml` in `resources_servers/browsecomp_advanced_harness/configs/browsecomp_advanced_harness.yaml`
config_paths="resources_servers/browsecomp_advanced_harness/configs/browsecomp_advanced_harness.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml"

ng_run "+config_paths=[${config_paths}]"
```


# Licensing information
Code: ?
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
