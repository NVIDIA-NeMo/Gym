# WebArena response agent

`webarena_agent` runs the screenshot-based, structured-tool WebArena policy loop. It
communicates only with NeMo Gym model and resources servers; browser, Playwright,
Xvfb, pyautogui, task evaluation, and tool execution remain in
`resources_servers/webarena`.

The agent:

- injects the Nemotron WebArena system prompt;
- requires one Responses API function call per environment step;
- preserves `seed_obs` and the complete flat multimodal trajectory;
- removes only old browser screenshots from future model requests while
  retaining all text, tool history, and task input images;
- aggregates model usage, uses bounded model retries and phase deadlines, and
  records typed termination metadata; and
- requests live-browser verification and then closes each environment
  best-effort in a `finally` path.

Configuration requires `resources_server` and `model_server`. The default
rollout policy uses 50 environment steps, three recent browser screenshots,
three model calls per step, and `return_transitions: false`.

## Model server requirements

The agent consumes native Responses API `function_call` output items. Text that
merely resembles XML or JSON tool markup is not executable. A vLLM endpoint must
therefore use automatic tool choice and the parser required by its checkpoint,
for example:

```bash
vllm serve /path/to/model \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

Choose tool and reasoning parsers to match the model/chat template. The Gym
`vllm_model` configuration's `uses_reasoning_parser` value must agree with the
endpoint. WebAgent always sends `parallel_tool_calls: false` and requires exactly
one structured call per environment step.

The model's multimodal limit and context length must cover retained browser
screenshots, task input images, and tool history. Lower
`max_recent_screenshots` when testing a model with a small image or context
budget.

Use `responses_api_models/vllm_model/configs/vllm_model.yaml` for evaluation.
Training uses `vllm_model_for_training.yaml` so each generated turn carries
`prompt_token_ids`, `generation_token_ids`, and `generation_log_probs`.

## Saved rollout shape

One JSONL row is one complete rollout, not one turn. The initial screenshot is
stored under `response.seed_obs`; the flat multi-turn sequence is stored under
`response.output` as reasoning/messages, function calls, function outputs, and
subsequent browser observations. Live-browser verification happens inside
`/run` before `/close`, so saved rows cannot be reverified later with
`ng_reward_profile`.

## Licensing information

Code: Apache 2.0

Data: N/A
