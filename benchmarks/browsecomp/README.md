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
