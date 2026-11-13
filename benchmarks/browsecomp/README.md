# Run instructions
1. Update your env.yaml file with the necessary keys and configs. Set `exclude_domains_file_path` to a JSON file containing `{"notices": []}` when no additional domain exclusions are needed; an empty list or `{}` does not match the parser schema.
```yaml
browsecomp_benchmark_resources_server:
  resources_servers:
    browsecomp_advanced_harness:
      tavily_api_key: ???
      exclude_domains_file_path: ???
Qwen3-235B-A22B-Instruct-2507-FP8:
  responses_api_models:
    vllm_model:
      base_url: ???
      api_key: ???
```

2. Prepare the benchmark dataset
```bash
config_paths="benchmarks/browsecomp/config.yaml"
ng_prepare_benchmark "+config_paths=[$config_paths]"
```

3. Run the benchmark against a VLLMModel
```bash
WANDB_PROJECT=
EXPERIMENT_NAME=
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/browsecomp/config.yaml"
ng_e2e_collect_rollouts \
    "+config_paths=[${config_paths}]" \
    +wandb_project=$WANDB_PROJECT \
    +wandb_name=$EXPERIMENT_NAME \
    ++output_jsonl_fpath=results/$EXPERIMENT_NAME.jsonl \
    ++split=benchmark \
    ++policy_base_url=??? \
    ++policy_api_key=??? \
    ++policy_model_name=???
```

## Run through the Environment Server

`config.yaml` now runs the benchmark through `single_agent_turn_legacy` with sandboxed agent and resources tool execution. It requires a Docker daemon by default. Replace the included Docker provider config and both sandbox images when selecting another provider; do not merge two provider blocks under `sandbox`. The sandboxes reach the model, resources and agent servers at the host's address (`use_absolute_ip: true`); `absolute_ip_server_types` keeps the head server, whose config endpoint returns the merged config, on loopback. BrowseComp sessions are not exported to checkpoints, so after a crash an episode starts over from its input.

Tasks, prompts and correct-answer grading are unchanged. The sandbox path adds a 3600-second activation cap by default (`sandbox_runner_timeout_seconds`); when `rollout_timeout_s` is unset, that cap also bounds the loop. Account for this limit when comparing with previous step-only runs. Compared with the previous host `/run` recipe, failure handling changes the scored denominator:

| Case | Previous host `/run` recipe | Current sandbox session recipe |
| --- | --- | --- |
| Rollout reaches `rollout_timeout_s` | Reward 0 without a judge call, `timed_out: 1` | The same |
| Model call returns 400, 413 or 422 | Reward 0, with `agent_error` | Verified with an empty answer, which scores 0 |
| Model call returns 5xx after the client's retries, or the sandbox cannot connect to the model or tools, at the activation's start or for about 90 seconds later | Set aside unscored (`kill_shaped`) for resume | A non-terminal failure row (`_ng_failure_*`), unscored, which `gym eval run --resume` retries |
| Model call returns 408, 425 or 429 after the client's retries | Reward 0, with `agent_error` | A non-terminal failure row, which a resume retries |
| Model call returns any other 4xx | Reward 0, with `agent_error` | A terminal failure row, not scored and not retried |
| Any other error in the agent turn | Reward 0, with `agent_error` | A non-terminal failure row, which a resume retries |

The sandbox checks that it can connect to the model, tool and token-count endpoints before the loop starts. A connection lost later is resent for about 90 seconds; if the endpoint does not come back, the activation fails with a non-terminal failure row instead of scoring as a timeout.

So an error that the previous recipe scored 0 now leaves the scored set instead of counting as a wrong answer. Check the failure rows and the coverage file alongside the accuracy.

The [BrowseComp adapter README](../../responses_api_agents/browsecomp_agent/README.md) documents Evaluation capability, explicit evidence gaps, sandbox ownership and unsupported controls. For a hosted model without `/tokenize`, set `save_model_call_using_vllm_tokenize_endpoint: false` explicitly. The research agent and read-only shell each run in an independently owned sandbox; the Resources Server keeps search credentials and verification on the host.

## OpenCode

Two presets run BrowseComp with OpenCode on the `opencode_search` Resources Server, which gives each question a sandbox
and grades the final answer with an LLM judge:

| Preset | Agent | How an episode runs |
| --- | --- | --- |
| `configs/browsecomp_opencode.yaml` | `opencode_sandboxed_search_agent`, the deprecated `opencode_sandboxed_agent` | A `legacy_agent` relay calls the agent's own `/run`, which seeds the Resources Server, runs OpenCode in the sandbox it gets back, and calls `/verify` |
| `configs/browsecomp_opencode_sessions.yaml` | `browsecomp_opencode_agent`, an `opencode_agent` | `single_agent_turn_legacy` seeds the Resources Server, which starts the sandbox, stages the OpenCode plugins in it and lends it to the agent's session. After the session closes, `/verify` reads how the run ended from the plugins' log and stops the sandbox before grading; `/close_session` then fences the session, and releases the sandbox if verify did not run |

The session preset repeats the relay preset's settings where `opencode_agent` has them: OpenCode 1.17.11 with its
stock system prompt and thinking on; the same questions and user turn (`prepare_opencode.py` writes them without the
system prompt and tool list of `prepare.py`'s rows, which `opencode_agent` refuses or adds to its instructions);
permission rules that resolve as the relay's; the same four OpenCode plugins with the same settings (the web-search
guard and its domain exclusion list, a 400-call turn cap, no subagents, and the opt-in arms through the same
environment variables); compaction at 192,000 tokens of a 262,144-token window, with compaction summaries left out of
the graded answer; no output cap; a 21,600-second run budget; and the same sandbox and grading. A run that reaches the
turn cap and then answers, or that the context window ends, keeps its judged reward and is masked on both; a run that
the turn-budget plugin aborts is graded and masked only on the relay (see [Failure accounting](#failure-accounting)).

The two presets' scores are separate measurements until a paired comparison of them, on the same questions with the
same model and judge in the same time window, shows they agree. The session preset is for evaluation only, as
`opencode_agent`'s session path is; training keeps the relay preset. What still differs:

- **Working directory.** The `Working directory` line of OpenCode's system prompt reads `/workspace` in the session
  preset, where the relay's runs use `/`, which `opencode_agent` refuses; the `Workspace root folder` line below it is
  the project root OpenCode finds from that directory. Relative paths, the default root of OpenCode's glob and grep
  tools, and the default directory of bash commands (when bash is on) move with it.
- **Truncation-file path.** When a tool result is too long, OpenCode saves it to a file and names that file in its
  notice to the model. The file lies in `opencode_agent`'s private OpenCode directories, so the path reads differently;
  the rest of the notice is the same.
- **Web-search key.** One key for every row, `EXA_API_KEY` in the lent sandbox's environment; the Resources Server
  refuses an `EXA_API_KEYS` pool there at startup. The relay agent picks a key for each row from a pool
  (`EXA_API_KEYS`), which is the same while the pool holds one key.
- **Process context the model does not see.** `opencode_agent` runs OpenCode with private HOME and XDG directories,
  loads its own task-environment plugin before the staged ones, and starts OpenCode with `--format json` and its own
  session title.
- **Token usage.** Both presets count input tokens as uncached input plus cache reads and writes, and output tokens as
  output plus reasoning, over every assistant message, compaction summaries included. The relay takes each message's
  total from the total OpenCode stored; the session preset adds input and output. The session preset reports cached
  and reasoning token details as null where OpenCode stored 0 or a message has no counts; the relay reports OpenCode's
  numbers.

### Failure accounting

A session-preset run that fails follows the Environment Server's outcome policy (`agent_outcome_policy`: the defaults
in `nemo_gym/agents/agent_outcome.py`, plus `model_context_limit: verify` from the preset). The relay agent grades
whatever export the run left, unless its `/run` fails:

| Case | Relay preset | Session preset |
| --- | --- | --- |
| The last model call fails with 400, 413 or 422 | Graded | Graded |
| The last model call fails with 408, 425, 429 or 5xx, or with a connection error OpenCode marks retryable | Graded | A non-terminal failure row, which `gym eval run --resume` retries |
| The last model call fails with any other 4xx | Graded | A terminal failure row, not scored and not retried |
| The run reaches the turn cap and then answers | Graded and masked (`max_iteration`) | The same |
| The context window ends the run | Graded and masked (`context_window`) | The same |
| The turn-budget plugin aborts the run after its grace calls | Graded and masked (`max_iteration`) | The agent answers 502: a non-terminal failure row |
| OpenCode exits non-zero but leaves an intact export | Graded | The agent answers 502: a non-terminal failure row |
| OpenCode leaves no usable export, for example after a crash | The agent's `/run` answers 500: a failure row | The agent answers 5xx: a non-terminal failure row |
| The export holds an assistant part type the relay agent does not convert (any but text, reasoning, tool, step-start, step-finish and patch), such as `retry` | The agent's `/run` answers 500: a failure row | Graded without that part |
| The seed cannot create the working directory in the sandbox | Not applicable: the agent's sandbox keeps the image's working directory | The Resources Server's seed answers 422 when `mkdir` exits non-zero, which every episode would repeat: a terminal failure row. A `mkdir` that does not complete answers 500: a non-terminal failure row |
| The plugins did not load as configured | The agent's `/run` answers 500 when the sentinel's line is missing: a failure row | `/verify` answers 422 when the plugin log, even a missing or empty one, lacks the sentinel's line or a required line: a terminal failure row |
| `/verify` cannot read the plugin log | Not applicable: the agent reads its run output | `/verify` answers 500 when the read fails, times out or comes back cut short: a non-terminal failure row |
| The run budget runs out | Graded, not masked | The same |
| The episode outlives `default_episode_timeout_seconds` | No episode deadline | A non-terminal failure row ("Episode timed out") |
| The judge gives no verdict in any attempt | Masked, `judge_unparseable` (a run that ended on its budget keeps that kind) | The same |
| The judge never answers | A `judge_failed` failures-sidecar row | The same |
| `judge_max_consecutive_failures` rollouts in a row get no verdict | `/verify` answers 500 for that rollout and each later one without a verdict, until a verdict resets the count, so the agent's `/run` answers 500: a failure row | The same `/verify` 500: a non-terminal failure row |

A relay cell that reads "Graded" assumes an export the relay agent can convert. A relay `/run` that answers 500 becomes
a failures-sidecar row (`agent_run_error`, which a resume retries) only with `++route_failures_to_sidecar=true`;
without it, the first one ends the run. A session-preset failure is a failure row either way. Run both presets with
`++route_failures_to_sidecar=true` and compare the failure rows by `failure_kind`. When the Resources Server names a
kind for a `/verify` that failed, such as `opencode_search:plugins_not_loaded`, the session row carries it in its
`_ng_failure_events`, as the cause of the failed episode.

### Time limits

The session preset's limits, from admission to cleanup:

| Limit | Setting | Enforced by | On expiry |
| --- | --- | --- | --- |
| Admission | `queue_timeout_seconds: 86400` for an episode waiting for one of `max_concurrent_episodes: 64` slots | `single_agent_turn_legacy` | A non-terminal failure row ("Episode admission timed out") |
| Sandbox start | `sandbox_config.ready_timeout_s: 1200` | The sandbox provider, at `opencode_search`'s seed | The seed answers 500: a non-terminal failure row |
| OpenCode setup | `sandbox_install_timeout_seconds: 600` for each setup command | `opencode_agent`, at its seed | The agent's seed answers 500: a non-terminal failure row |
| Run | `timeout: 21600`, from when the session's one activation holds a concurrency slot | `opencode_agent` | OpenCode is stopped; the `incomplete` response is graded on what the run produced, not masked |
| Agent close | `session_close_timeout_seconds: 300` for each phase of the close, and `cleanup_timeout_seconds` for the whole close | `opencode_agent`; `single_agent_turn_legacy` for the whole close | The close fails, and the episode with it, before `/verify`: a non-terminal failure row |
| Episode | `default_episode_timeout_seconds: 25200`, from the seed through `/verify` | `single_agent_turn_legacy` | A non-terminal failure row ("Episode timed out"); the cleanup then closes both sessions |
| Judge | `judge_timeout_seconds: 300` for each request of the first attempt and `judge_retry_timeout_seconds: 120` for each request of a later one, up to `judge_max_attempts: 4`, `judge_retry_delay_seconds: 30` apart. Each attempt is one `call_judge`, which tries up to 3 times after a timeout, a lost connection, 408, 429 or 5xx | `opencode_search`, in `/verify` | No verdict in any attempt: masked, `judge_unparseable`. No reply at all: a `judge_failed` failures-sidecar row |
| Cleanup | `cleanup_timeout_seconds: 600` for each step after the episode: the agent's close if it is still open, then the resources server's | `single_agent_turn_legacy` | Logged ("Episode cleanup timed out") and recorded in the row's failure report as `session_release_failed`; the row keeps its result, and a sandbox still running is stopped when the server shuts down or reclaimed after `ttl_s` |
| Sandbox lifetime | `sandbox_config.ttl_s: 27000` | The sandbox provider | The provider reclaims the sandbox |

The episode deadline covers the seed, the run, the agent's close, and `/verify`'s plugin-log read (60 s at most) and
sandbox stop (120 s at most). After a run that used its whole budget it leaves the judge at most about 20 minutes: a
judge that needs longer, which its attempts allow, times the episode out. With the export copy on
(`episode_export_copy_dir`, see [Compare the two presets](#compare-the-two-presets)), `/verify` also copies the
exports before it stops the sandbox, for up to 300 s, which leaves the judge at most about 16 minutes. A job time limit
shorter than the run budget ends such runs first. `ttl_s` stays above the deadline plus two cleanup steps, since
`/verify` stops the sandbox before it calls the judge and, after a deadline, the cleanup stops it. The relay preset has
no episode deadline: its agent's `sandbox_timeout: 21600` bounds the run, and the judge's limits are the same.

### Row fields

The relay agent adds its run's fields to each row after `/verify`; a session row carries the agent's trajectory and
observations and the plugin log instead:

| Relay row | Session row |
| --- | --- |
| `opencode_failed`, `opencode_exit_code`, `opencode_error_type` | No such fields: `response.status` says how the run ended, and a run the outcome policy does not grade is a failure row. The agent's invocation and sandbox observation in `ng_agent_observations` give the run's status, error type and exit code |
| `opencode_export_found`, `opencode_finished` | None: a run that leaves no assistant result fails before `/verify` |
| `opencode_results_fpath` (OpenCode's export, saved on the host), `opencode_run_stdout`, `opencode_run_stderr` | `opencode_plugin_log`, the plugins' lines. OpenCode's exports reach the host only with the Resources Server's `episode_export_copy_dir` set: `opencode_export_copy` then names the host directory that holds them (see [Compare the two presets](#compare-the-two-presets)) |
| `opencode_plugins_loaded`, `opencode_terminal_reason` | The same fields, read from the plugin log |
| `opencode_attachment_count` | None: the rows hold no images |
| `ng_trajectory` and `ng_agent_observations`, only when the agent correlates rollouts, for example with `++observability_enabled=true` | Both, from the agent's close |
| `response.output`: the assistant items, and the user messages OpenCode adds during the run, such as the one that continues it after a compaction | The assistant items only |
| An errored tool call's `function_call_output`: empty | The tool's error text, and both items' status is `incomplete` |
| A last reply cut at the output limit: `status: incomplete` with `incomplete_details.reason: max_output_tokens` | `status: incomplete` with `incomplete_details: null` |
| `response.metadata`: none | `harness_execution`, `opencode_version`, `harness_hostname`, `harness_pid` and `agent_timeout_sec`, and `ng_agent_outcome` on a failed response |

Grading reads only the last assistant text, so none of these changes a reward. Two relay agent settings have no session
counterpart. `execution_failure_reward_zero`, which gives a run with `opencode_failed` set a reward of 0, is off in the
relay preset (its default); a failed session run follows `agent_outcome_policy`. The relay sets OpenCode's
`chunkTimeout` to its 21,600,000 ms run budget, so it never ends a model call before the run ends, and the session
preset sends `timeout: false` and no `chunkTimeout`: neither preset puts a deadline on a single model call. A later
OpenCode release defaults `chunkTimeout` to five minutes, so check this before changing `opencode_version`.

### Run the session preset

The preset needs these values, none of which has a default, so every command that loads it, `gym eval prepare`
included, fails without them:

```bash
# Files already inside the sandbox, for example on a mounted cache, with their SHA-256 digests: the OpenCode 1.17.11
# binary, and ripgrep 15.1.0, the build OpenCode 1.17.11 downloads. opencode_agent checks both before each session.
export BROWSECOMP_OPENCODE_EXECUTABLE_PATH=...
export BROWSECOMP_OPENCODE_EXECUTABLE_SHA256=...
export BROWSECOMP_OPENCODE_RIPGREP_PATH=...
export BROWSECOMP_OPENCODE_RIPGREP_SHA256=...
```

```yaml
# env.yaml: the judge, with its key as for the relay preset.
search_judge_base_url: ...
search_judge_model_name: ...
inference_api_key: ...
```

`python -m benchmarks.browsecomp.prepare_opencode`, run from the repository root, writes the rows without them.

The web-search key reaches the sandbox in its environment. Supply it from a file that is never committed, such as
env.yaml:

```yaml
search_browsecomp_resources_server:
  resources_servers:
    opencode_search:
      episode_sandbox_env:
        EXA_API_KEY: ...
```

Then start the servers and collect rollouts. Compose a sandbox provider config, as for the relay preset:

```bash
gym env start \
    --config benchmarks/browsecomp/configs/browsecomp_opencode_sessions.yaml \
    --config responses_api_models/vllm_model/configs/vllm_model.yaml \
    --config <sandbox provider config> \
    ++policy_base_url=... ++policy_api_key=... ++policy_model_name=...

gym eval run --no-serve <the same --config and ++ arguments> \
    --agent browsecomp_opencode_agent \
    --input benchmarks/browsecomp/data/browsecomp_opencode_benchmark.jsonl \
    --output results/browsecomp_opencode_sessions/rollouts.jsonl \
    ++route_failures_to_sidecar=true
```

- OpenCode calls the policy model from the sandbox, so the preset sets `use_absolute_ip: true` and gives this host's
  address to the policy model only. The Resources Server, the agent, the Environment Server and the judge stay on
  loopback.
- With the search cache on (`NEMO_GYM_EXA_CACHE_ROOT`), the sandbox searches through the Resources Server's `/mcp`
  proxy: add `++absolute_ip_server_types=[responses_api_models,resources_servers]`. On loopback the server refuses to
  start.
- The preset requires only plugin lines that every arm prints. `NEMO_GYM_DISABLE_BASH=1` removes bash in both presets;
  set `NEMO_GYM_FIX_GREP_DESCRIPTION=1` with it, as the relay preset advises, so that grep's description no longer
  sends the model to bash and the task tool. For that arm, also require the lines its plugins print, in an overlay that
  restates the whole list (the first and third lines extend the preset's two):

  ```yaml
  search_browsecomp_resources_server:
    resources_servers:
      opencode_search:
        required_plugin_markers:
        - "[nemo-gym-plugin] tool_surface task=off bash=off"
        - "[nemo-gym-plugin] tool_surface grep_description=patched removed=2/2"
        - "[nemo-gym-plugin] web_guard version=2 contamination=on exclude_domains=47 search_transport=mcp"
        - "[nemo-gym-plugin] bash_guard=on"
  ```

### Compare the two presets

- Run both from the same Gym commit and in the same time window, since web search and a hosted judge change over
  time. Use the same questions (the same `BROWSECOMP_RUN_FULL` for both prepare scripts), policy model, judge (the
  relay preset's, named in `search_judge_base_url` and `search_judge_model_name`), sandbox provider, web-search key
  and opt-in arms, and `++route_failures_to_sidecar=true` for both.
- Pin the session preset to the OpenCode binary the relay agent installs, so both run one build.
- Compare question by question: each row's `reward`, `mask_sample` and `failure_kind`, and the failure rows.
- Compare token counts by `response.usage.input_tokens` and `output_tokens`, which both presets count the same way;
  for a total, add the two on both, since the relay's `total_tokens` is OpenCode's stored total. Leave out the cached
  and reasoning details (see Token usage above).
- Each session row's `opencode_plugin_log` shows the tool surface the run had, the search guard's settings and how
  the run ended. To keep each session's OpenCode export as well, for example to check where compaction started, set
  `BROWSECOMP_OPENCODE_EXPORT_DIR` to a fresh sandbox directory outside `/workspace` and `/tmp/nemo-gym-opencode*`,
  such as `/tmp/nemo-gym-exports`, not `/tmp` itself, which also holds the staged plugins and the plugin log. The agent
  writes the exports there and the Resources Server reads them from there. Then give the run a host directory of its
  own in an overlay:

  ```yaml
  search_browsecomp_resources_server:
    resources_servers:
      opencode_search:
        episode_export_copy_dir: /absolute/host/directory/for/this/run
  ```

  `/verify` copies the files before it stops the sandbox, to
  `<episode_export_copy_dir>/<capture key>/<task id>/attempt-<attempt>/`, and records the outcome in the row's
  `opencode_export_copy`. A copy that fails, runs past 300 seconds or exceeds `episode_export_max_bytes` (256 MiB) is
  recorded there and never changes the grade.
- `++observability_enabled=true` with a `++model_call_capture_dir` records each model call in the rollouts, as the
  harness conformance checker needs. It also adds a plugin to OpenCode, so use it in a run of its own.

### Scripted smoke

To check the wiring without a model, point the policy model and the judge at stub servers on this host that answer
chat completions (`/v1/chat/completions` and `/v1/models`) with scripted replies. A judge stub that always gives the
same verdict makes the rewards of the two presets comparable row by row. The overrides below name the judge
for either preset, and `++use_absolute_ip=true`, which the session preset already sets, lets the relay's sandbox reach
the model server too:

```bash
SMOKE="++policy_base_url=http://127.0.0.1:8900/v1 ++policy_api_key=dummy ++policy_model_name=scripted \
    ++search_judge_model.responses_api_models.vllm_model.base_url=http://127.0.0.1:8901/v1 \
    ++search_judge_model.responses_api_models.vllm_model.model=scripted-judge ++inference_api_key=dummy \
    ++use_absolute_ip=true ++uvicorn_logging_show_200_ok=true"

gym env start --config benchmarks/browsecomp/configs/browsecomp_opencode_sessions.yaml \
    --config responses_api_models/vllm_model/configs/vllm_model.yaml --config <sandbox provider config> $SMOKE

gym eval run --no-serve --config benchmarks/browsecomp/configs/browsecomp_opencode_sessions.yaml \
    --config responses_api_models/vllm_model/configs/vllm_model.yaml --config <sandbox provider config> $SMOKE \
    --agent browsecomp_opencode_agent --input smoke.jsonl --output results/smoke-sessions/rollouts.jsonl \
    --limit 5 --num-repeats 2 --concurrency 2 ++route_failures_to_sidecar=true
```

With `++uvicorn_logging_show_200_ok=true` the logs show each episode's calls in order: `/seed_session`, the agent
session, the model calls, the agent's close, `/verify` and `/close_session`. Run the relay preset the same way, with
`--agent opencode_sandboxed_search_agent` and `prepare.py`'s rows; it has no `/close_session`.

- Nothing checks the staged plugins' bytes per episode, so check them once here: in one lent sandbox, the SHA-256
  digests of the `/tmp/nemo-gym-plugin-*.js` files equal those of the repository files `episode_sandbox_files` names.
- To check a row of the failure table above, script the stubs to end a run that way, for example by answering a model
  call with an error status, and compare each preset's row with the table.
