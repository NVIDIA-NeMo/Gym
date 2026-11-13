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

Enroot must provide a PID namespace separate from the resources-server host, with `/proc` mounted for that same
namespace. Setup checks the running sandbox's namespace IDs before permitting model commands; a version string
or `ENROOT_UNSHARE_PID` setting alone is insufficient because older Enroot builds may ignore that setting. If the
check fails, upgrade or configure Enroot with working PID isolation, or select another isolating provider. The
failed session follows the usual sandbox cleanup and close-retry path. A probe that shows the host's namespace or a
mismatched `/proc` mount refuses the seed with 422 once the sandbox is stopped, since no retry can pass it; a probe
that does not finish fails the seed with a retryable error. Only Enroot is probed. Choose any other
provider for sandboxes that cannot see the host filesystem: Apptainer, for example, bind-mounts the host's home
directory and `/tmp` unless configured otherwise.

Each resources session owns one tool sandbox. An agent that runs its loop in a sandbox owns a separate sandbox;
neither component borrows the other's container. This works with providers such as Enroot that do not support
serializing and reconnecting to a sandbox. Search and browse retain their existing responses and write immutable
retrieved pages. Only those pages and their manifest are copied into the tool workspace. Search/judge credentials
and verifier inputs are not copied into the sandbox.

The shell uses the checked-out pocketshell implementation and `SandboxPythonRuntime` with the installed `regex`
version pinned into its isolated Python runtime. The first seed builds that runtime if needed; subsequent sessions
reuse its host-side artifact cache. Setup failures surface to the caller and retain ownership when cleanup fails,
so a later close can retry. Commands are supervised inside the sandbox; their guard, supported commands, output
truncation, and exit-code envelope are unchanged. A command that outlives the supervisor deadline answers
`[exit_code=-1]` and a sandbox failure answers `[exit_code=-2]`, as on the host. Only the supervisor's deadline
reads as the command's timeout; a provider timeout (an upload, the exec itself, or reading the output) is a sandbox
failure. There is no fallback to host execution after a sandbox failure. A sandbox that cannot start or cannot run a
command marks its session, and `/verify` keeps the judged reward but masks the sample (`mask_sample: true`,
`failure_kind: provider_unavailable`), since the model lost its tool to the infrastructure. A masked sample is
returned even when the judge cannot be reached. After a failed start, later commands answer `[exit_code=-2]` without
starting another sandbox. A blocked command answers without starting a sandbox. A missing provider block, an
unknown provider, or an unknown `sandbox_config` key refuses the seed with 422.

`allow_host_execution: true` explicitly enables the former host interpreter for development when no provider is
configured, or permits the non-isolating `local` provider. It defaults to `false`; production and benchmark configs
must provide an isolating provider. Configurations with `workspace: none` do not need a tool sandbox.

## Search cache

An optional flat-file cache stores raw search and browse responses so that repeated queries, for example the
repeats of one task in a training run, skip the provider call. It is off unless `search_cache_dir` is set. Add these
fields under the resources server's `resources_servers` block:

```yaml
search_cache_dir: /shared/browsecomp-cache   # unset or empty disables the cache; relative paths resolve
                                             # against this server's directory
search_cache_fuzzy_threshold: 0.9            # optional, (0, 1]: serve the closest cached search on an exact miss
search_cache_read_only: false                # serve from a root you cannot write, without writing to it
verbose_phase_log: false                     # log every cache get and put, not only the per-rollout totals
```

- **It changes what the agent retrieves.** Entries are shared across runs and across providers: an Exa entry
  answers a Tavily run and is formatted as Exa output. Keep the cache off for baselines and score comparisons.
- **Fuzzy matching serves another query's results** when the two score above the threshold, for example a query
  about 2018 answered with the cached results for 2019. Search only; browse always matches the URL exactly.
- **Each hit is checked against the request.** The current run's excluded domains and `max_results` apply to every
  hit. An Exa search entry answers only the same `exa_search_type`, and only a request that writes pages when it
  holds page text. A browse page that Exa cut at a smaller character budget is a miss.
- **Writes are best-effort.** A failed write is logged as `[cache][put_error]` and counted in the `[cache]`
  summary at `/verify`; the tool call still returns the provider's result. Startup refuses a writable cache on a root
  this process cannot write, and a read-only cache on a root that does not exist.
- **The cache grows without bound.** Nothing evicts entries; prune the directory between experiments as needed.
- **Fuzzy index.** With fuzzy matching on, each process appends the queries it caches to its own shard under
  `<root>/index/`. Merge the shards into the read-only SQLite FTS5 index offline, while no run writes to the root:

  ```bash
  python resources_servers/browsecomp_advanced_harness/build_cache_index.py /shared/browsecomp-cache
  ```

## Sessions

Each rollout has its own session: the provider-call metrics that `/verify` reports and, with `workspace: per_session`, the disk workspace that `search` and `browse` write pages to and `bash_command` reads.

- **Environment Server.** `/seed_session` takes a `ResourcesSeedSessionRequest` and keys the session by its `resources_session_id`. The seed is refused with 422 when `task_data` lacks the `question` and `ground_truth` that `/verify` needs, or, with a workspace, when the ID is not a plain directory name. `/close_session` removes the workspace and the metrics, and is idempotent. A close that cannot remove the workspace fails and keeps the session, so the caller can retry it. After a close, a seed for the same ID is refused with 409, and a tool call for it gets 404 (`session_lost`) instead of recreating the workspace. `/verify` of a closed session returns a masked sample (`reward: 0`, `mask_sample: true`, `failure_kind: session_lost`). The reward depends only on the response and the task, so the mask is kept on purpose: a closed session reads the same on every server under the resources session contract. `single_agent_turn` verifies before it closes, so only an out-of-order verify gets the mask.
- **An agent's `/run`.** `/seed_session` takes the flat row and keys the session by the cookie. `/verify` releases the sandbox and removes the workspace, since this path sends no close. The sandbox starts with the first `bash_command`, not at the seed. A run whose agent never sends `/verify`, as `browsecomp_agent` does after a timeout or an aborted run, keeps its sandbox until the provider's `ttl_s` reclaims it or the server shuts down; Enroot ignores `ttl_s`. Sandboxed runs should use the Environment Server path, which always closes.

`/close_session`, and `/verify` on the `/run` path, stop the owned sandbox before deleting its workspace and metrics.
A failed stop leaves the session available for a retry. `/verify` of an episode session releases nothing,
so a re-sent verify still finds the session's metrics and its sandbox failure; its close releases them. Tool calls,
verification, and close are serialized per session, so cleanup cannot delete files while a tool is running. Releasing
a session drops its metrics, unless `dump_session_id_to_metrics_on_exit` keeps them for the dump at shutdown.

Shutdown tries to stop any remaining tool sandboxes, but only on a best-effort basis. `gym env` kills the server one
second after its interrupt unless `server_shutdown_timeout_seconds` is raised to cover the closes, and Enroot ignores
`ttl_s`, so an Enroot sandbox left behind stays until `enroot remove` deletes it.

## Running

The example pairing uses `single_agent_turn_legacy`: existing flat dataset rows still work, while the environment
server owns resources/agent session setup, verification, and close. The agent loop runs in its own sandbox and
receives the resources session through the `direct_http` transport; its `resources_server` config is `null`.
The config includes the Docker provider and `use_absolute_ip: true` so the sandbox can reach the Gym servers;
`absolute_ip_server_types` limits that to the model, resources and agent servers, so the head server, whose config
endpoint returns the merged config, stays on loopback.
Replace the included Docker provider config to run on a different backend, and set both sandbox image fields for that provider. Do not merge two provider blocks under `sandbox`.

The example dataset exposes only `search` and `browse`, so the example config keeps `workspace: none` and inline
page content. To enable the resources-owned tool sandbox, use `workspace: per_session` with dataset rows exposing
`bash_command`, as the BrowseComp benchmark does. This preserves the example prompts and tool schemas.

```bash
config_paths="resources_servers/browsecomp_advanced_harness/configs/browsecomp_advanced_harness.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml"

ng_run "+config_paths=[${config_paths}]"
```

To run the benchmark instead, use `benchmarks/browsecomp/config.yaml` as the first config path. Its dataset,
workspace, and tool schemas are configured together.


# Licensing information
Code: ?
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
