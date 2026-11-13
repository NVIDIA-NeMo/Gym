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
system prompt and tool list of `prepare.py`'s rows, which `opencode_agent` refuses or adds to its instructions); the same
tools map and permission rules; the same four OpenCode plugins with the same settings (the web-search guard and its
domain exclusion list, a 400-call turn cap, no subagents, and the opt-in arms through the same environment variables);
compaction at 192,000 tokens of a 262,144-token window, with compaction summaries left out of the graded answer; no
output cap; a 21,600-second run budget; and the same sandbox and grading. A run cut off by the turn cap or the context
window keeps its judged reward and is masked on both.

The two presets' scores are separate measurements until a paired comparison of them, on the same questions with the
same model and judge in the same time window, shows they agree. What still differs:

- **Working directory.** OpenCode's system prompt names its working directory: `/workspace` in the session preset,
  where the relay's runs use `/`, which `opencode_agent` refuses. Relative paths and the default root of OpenCode's glob
  and grep tools move with it.
- **Truncation-file path.** When a tool result is too long, OpenCode saves it to a file and names that file in its
  notice to the model. The file lies in `opencode_agent`'s private OpenCode directories, so the path reads differently;
  the rest of the notice is the same.
- **Failure accounting.** A run that fails, for example when OpenCode exits with an error, follows the Environment
  Server's outcome policy: it is retried or reported in the failures sidecar, where the relay graded what the run
  produced. A run that failed at the context window is graded on both. A `/verify` that cannot read the plugin log
  answers 500, so the episode fails and a resumed run retries it; a log that lacks a required line is refused with 422
  and not graded. Run both presets with `++route_failures_to_sidecar=true` and compare the failure rows by
  `failure_kind`.
- **Web-search key.** One key for every row, from the lent sandbox's environment. The relay agent picks a key for each
  row from a pool (`EXA_API_KEYS`), which is the same while the pool holds one key.
- **Process context the model does not see.** `opencode_agent` runs OpenCode with private HOME and XDG directories,
  loads its own task-environment plugin before the staged ones, and starts OpenCode with `--format json` and its own
  session title.
- **Artifacts.** The relay agent saves OpenCode's export on the host and puts OpenCode's output in each row
  (`opencode_results_fpath`, `opencode_run_stdout`, `opencode_run_stderr`). The session preset's rows carry the agent's
  trajectory and observations and the plugin log (`opencode_plugin_log`) instead.

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
- `NEMO_GYM_DISABLE_BASH=1` removes bash in both presets. With it, also require the plugins' report of the tool
  surface, in an overlay that restates the whole list:

  ```yaml
  search_browsecomp_resources_server:
    resources_servers:
      opencode_search:
        required_plugin_markers:
        - "[nemo-gym-plugin] web_guard version=2 contamination=on"
        - "[nemo-gym-plugin] tool_surface task=off bash=off"
  ```

### Compare the two presets

- Run both from the same Gym commit and in the same time window, since web search and a hosted judge change over
  time. Use the same questions (the same `BROWSECOMP_RUN_FULL` for both prepare scripts), policy model, judge (the
  relay preset's, named in `search_judge_base_url` and `search_judge_model_name`), sandbox provider, web-search key
  and opt-in arms, and `++route_failures_to_sidecar=true` for both.
- Pin the session preset to the OpenCode binary the relay agent installs, so both run one build.
- Compare question by question: each row's `reward`, `mask_sample` and `failure_kind`, and the failure rows.
- Each session row's `opencode_plugin_log` shows the tool surface the run had, the search guard's settings and how
  the run ended. `BROWSECOMP_OPENCODE_EXPORT_DIR` also keeps each session's OpenCode export, for example to check where
  compaction started; the Resources Server stops the sandbox at `/verify`, so point it at storage that outlives the
  sandbox, such as a provider volume, outside `/workspace`.
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
