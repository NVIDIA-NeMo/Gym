# AgentDojo

[AgentDojo](https://github.com/ethz-spylab/agentdojo) ([arXiv:2406.13352](https://arxiv.org/abs/2406.13352), "AgentDojo:
A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents", NeurIPS 2024 Datasets and
Benchmarks) runs tool-using agents over four suites -- `banking`, `slack`, `travel` and `workspace` -- clean and under
indirect prompt injection. Each selector runs one user task, either clean or with one injection task delivered by the
`important_instructions` attack, through the pinned upstream harness (`ethz-spylab/agentdojo@v0.1.35`, commit
`a75aba7631d3ca5fb7ab938965c97ead2f9ff84b`, benchmark `v1.2.2`), which scores utility and attack success against the
suite's own environment state.

| Suite | User tasks | Injection tasks | Clean rows | Attacked rows |
| --- | ---: | ---: | ---: | ---: |
| banking | 16 | 9 | 16 | 144 |
| slack | 21 | 5 (numbered 1–5) | 21 | 105 |
| travel | 20 | 7 | 20 | 140 |
| workspace | 40 | 14 | 40 | 560 |
| **total** | **97** | **35** | **97** | **949** |

The counts are `get_suites("v1.2.2")` at the pinned commit; the agent tests check the task ids against it. Rows carry
task ids only, and upstream loads each prompt and environment. The task matrix, metrics, defenses, and the ways this
adapter differs from upstream's harness are documented in
[`responses_api_agents/agentdojo_agent/README.md`](../../responses_api_agents/agentdojo_agent/README.md).

## Usage

```bash
# Write data/agentdojo_benchmark.jsonl (1,046 selectors). Expected sha256:
# e59fd9b9894bfd6eed7a706d165721778dd4820c4af2ca68635f4430726f27f8
gym eval prepare --benchmark agentdojo

# Start servers, then collect the undefended arm at temperature 0.
gym env start --benchmark agentdojo --model-type vllm_model
gym eval run --no-serve --benchmark agentdojo --model-type vllm_model \
    --agent agentdojo_benchmark --input benchmarks/agentdojo/data/agentdojo_benchmark.jsonl \
    --temperature 0.0 --output results/agentdojo-undefended.jsonl
```

Pass `--temperature 0.0`. The adapter forwards only the sampling settings the run sets, so without it the endpoint's
own default applies and the arms are sampled rather than greedy, which widens run-to-run variance. Upstream's
`OpenAILLM` nominally samples at 0 but sends `temperature or NOT_GIVEN`, so its literal request omits a 0.0; leaving
the flag off reproduces that request, not a deterministic run.

A defense is selected on the agent server with `default_defense`, not by rewriting rows, so every arm scores the same
1,046-selector file. The pinned upstream registers four (`agentdojo.agent_pipeline.agent_pipeline.DEFENSES`):
`tool_filter`, `transformers_pi_detector`, `spotlighting_with_delimiting` and `repeat_user_prompt`. Pass one when
starting the servers:

```bash
gym env start --benchmark agentdojo --model-type vllm_model \
    ++agentdojo_benchmark.responses_api_agents.agentdojo_agent.default_defense=tool_filter
```

Read `agentdojo/benign_utility`, `agentdojo/utility_under_attack` and `agentdojo/attack_success_rate` from the
aggregate metrics, plus `agentdojo/tool_filter_empty_selection_rate` under `tool_filter`. They are computed after masked
rollouts leave the denominator; Gym's `coverage/masked_rollouts` says how many did.

Two operational notes:

- **The agent's `concurrency` is fixed at 1**, and the config refuses anything higher (see the agent README). Run arms
  or models as separate processes for throughput.
- **`tool_filter` needs tools in the prompt under `tool_choice="none"`.** Check the endpoint with the two-request probe
  in the agent README before reading a `tool_filter` score; an empty-selection rate near 1 means the serving stack
  dropped the tool list, not that the defense failed.

## Comparing with the AgentDojo paper

Upstream's results table (`docs/results-table.html` at the pinned tag) reports undefended `important_instructions`
runs on benchmark v1.1.x: the same 97 user tasks, but 629 attacked pairs, because `workspace` had 6 injection tasks
there and has 14 in v1.2.2. For example, `gpt-4o-2024-05-13` scores 69.07% benign utility, 50.08% utility under attack
and 47.69% targeted attack success; `claude-3-5-sonnet-20241022` scores 79.38%, 72.50% and 1.11%. Measured here on
v1.2.2, undefended, at temperature 0:

| Model | Benign utility | Utility under attack | Attack success |
|---|---:|---:|---:|
| `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` | 86.60% | 88.09% | 0.95% |
| `moonshotai/Kimi-K3` | 91.75% | 91.57% | 0.32% |
| `Qwen/Qwen3.5-122B-A10B-FP8` | 89.69% | 78.93% | 28.35% |
| `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` | 87.63% | 86.30% | 4.00% |

Two differences in how the harnesses run. Upstream's `important_instructions` interpolates the victim model's name
into the attack ("to you, GPT-4"); this adapter's default `attack_model_alias: local` interpolates "Local model", as
upstream does for local models. And upstream's `benchmark.py` sets `utility = False, security = True` when a rollout
raises a provider `ServerError` or internal-server `ApiError`, or hits `context_length_exceeded`, where its
`security` means the injection *succeeded*: an infrastructure failure is scored as a successful attack. This adapter
masks those rollouts instead, so they leave the denominator rather than inflate the attack success rate.

## License

AgentDojo code and task data: MIT.
