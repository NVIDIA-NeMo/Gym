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

## Run another harness

`opencode.yaml` runs [OpenCode](../../responses_api_agents/opencode_agent/README.md) (`opencode_agent`, OpenCode
1.17.11) in place of `browsecomp_agent`, with the same Resources Server, judge, tasks and prompts as `config.yaml`.
This is the session-based pairing. The older `configs/browsecomp_opencode.yaml` preset uses the deprecated
`opencode_sandboxed_agent`, `opencode_search`, OpenCode's Exa web tools and a different judge; its scores measure a
different setup.

The preset assumes a served context length of at least 262,144 tokens. For another deployment, set
`context_window` to the served context length minus `max_output_tokens` before running:

```bash
gym eval prepare --benchmark browsecomp/opencode
gym eval run \
    --model-type vllm_model \
    --benchmark browsecomp/opencode \
    --split benchmark \
    --concurrency 32 \
    --output results/browsecomp_opencode.jsonl \
    --model-url <vllm_base_url> \
    --model-api-key <vllm_api_key> \
    --model <served_model_name>
```

- The rows come from `data/browsecomp_benchmark_harness_neutral.jsonl`, written by `prepare_harness_neutral.py`: the
  rows `prepare.py` writes, without the tool list. OpenCode and the other session harnesses reject a tool list in the
  request.
- OpenCode reaches `search`, `browse` and `bash_command` over MCP: the Resources Server sets
  `expose_tools_over_mcp: true` and the Environment Server sets `resources_tool_transports: [mcp]`. The Resources
  Server advertises the descriptions and parameters the rows give `browsecomp_agent`
  (`resources_servers/browsecomp_advanced_harness/tool_definitions.py`). The model sees them as
  `browsecomp_benchmark_resources_server_<tool>`. The search keys stay on the Resources Server, and `bash_command`
  still runs in the Resources Server's read-only shell sandbox, where the saved pages are.
- OpenCode runs in a sandbox it creates from `python:3.13-slim` and installs its pinned runtime there at the start of
  each episode, from GitHub. Its permission rules deny every tool but the BrowseComp ones: its own shell and file tools
  would run in its sandbox, which holds none of the saved pages, and its web tools would skip the domain exclusions.
  With Docker the sandbox keeps the default network; with a provider that has network policies, such as OpenSandbox,
  set `network_access: model_and_tools` on the agent only with an image containing the pinned OpenCode runtime,
  plugin dependencies and `rg`. Configure `sandbox_executable_path` and `sandbox_executable_sha256` as described in
  [Offline sandbox runtimes](../../responses_api_agents/opencode_agent/README.md#offline-sandbox-runtimes-and-native-model-identity).
  The shipped `python:3.13-slim` image needs download access and cannot install the runtime under that policy.
- The Environment Server admits at most 32 episodes, matching the agent's `concurrency`, with a queue timeout of
  86,400 seconds. Queuing happens before sandbox setup and the 6,000-second episode deadline. Change
  `max_concurrent_episodes` together with the agent's `concurrency`.
- OpenCode's own system prompt comes first, and the row's system prompt is added to it as instructions.
  `browsecomp_agent`'s context resets, progress board and step nudges do not apply: OpenCode compacts its context
  20,000 tokens below `context_window`. The run has a 3,600-second `timeout` and no step cap. A timed-out OpenCode
  run is judged on its last assistant text, which can be an intermediate message or compaction summary, and the
  BrowseComp `timed_out` metric remains 0. `browsecomp_agent` instead sets that metric and scores a timeout as 0
  without calling the judge; use the OpenCode response's incomplete/timeout signal to identify its timeouts.
- Every call requests `max_output_tokens` (65,536 in this preset). A call that reaches it ends the whole OpenCode
  run with `incomplete_reason=max_output_tokens`, and the judge receives the truncated output.
- `mcp_tool_timeout_seconds: 900` sets OpenCode's MCP connection and SDK request timeout, but Bun's `fetch` still
  caps a call at about 300 seconds. A search or browse waiting out provider rate limits can therefore fail sooner
  than it would under `browsecomp_agent`.
- MCP tool results reach OpenCode as JSON text such as `{"results_string":"line 1\nline 2"}`. `browsecomp_agent`
  unwraps this envelope and passes raw multiline text to the model.
- The harness and its system prompt differ from `browsecomp_agent`, so the two scores are not the same measurement.
- Change a harness setting with `++browsecomp_opencode_agent.responses_api_agents.opencode_agent.<key>=<value>`.
