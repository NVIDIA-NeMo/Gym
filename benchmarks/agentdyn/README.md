# AgentDyn

[AgentDyn](https://github.com/SaFo-Lab/AgentDyn) ([arXiv:2602.03117](https://arxiv.org/abs/2602.03117), "Are Your Agent
Security Defenses Deployable in Real-World Dynamic Environments?") extends AgentDojo with three dynamic, long-horizon
suites -- `shopping`, `github` and `dailylife` -- and runs nine indirect prompt-injection defenses over them. Each
selector runs one user task, either clean or with one injection task delivered by the `important_instructions` attack,
through the pinned upstream harness (`SaFo-Lab/AgentDyn@5353cf7615b135cace8d07c8f12dac53a16b6db3`, benchmark `v1.2.2`),
which scores utility and attack success against the suite's own state.

| Arm | Selectors |
|---|---:|
| Clean (benign utility) | 60 |
| Attacked (utility under attack, attack success) | 560 |
| **Total** | **620** |

The task matrix, metrics, defenses, and the ways this adapter differs from upstream's harness are documented in
[`responses_api_agents/agentdyn_agent/README.md`](../../responses_api_agents/agentdyn_agent/README.md).

## Usage

```bash
# Write data/agentdyn_v1_2_2.jsonl (620 selectors). Expected sha256:
# 819443fea5cb747eabf50a42adf7ba2106abeaa2e8a2e5d7dbf3f60066a86de2
gym eval prepare --benchmark agentdyn

# Start servers, then collect the undefended arm at upstream's sampling setting.
gym env start --benchmark agentdyn --model-type vllm_model
gym eval run --no-serve --benchmark agentdyn --model-type vllm_model \
    --agent agentdyn_benchmark --input benchmarks/agentdyn/data/agentdyn_v1_2_2.jsonl \
    --temperature 0.0 --output results/agentdyn-undefended.jsonl
```

Pass `--temperature 0.0` to match upstream's harness, whose `OpenAILLM` samples at 0. The adapter forwards only the
sampling settings the run sets, so without it the endpoint's own default applies -- and the arms are then sampled
rather than greedy, which widens run-to-run variance. CaMeL is the exception either way: it requests temperature 0 from
its own client.

A defense is selected on the agent server with `default_defense`, not by rewriting rows, so every arm scores the same
620-selector file. Pass one of the nine defense names when starting the servers:

```bash
gym env start --benchmark agentdyn --model-type vllm_model \
    ++agentdyn_benchmark.responses_api_agents.agentdyn_agent.default_defense=camel
```

Read `agentdyn/benign_utility`, `agentdyn/utility_under_attack` and `agentdyn/attack_success_rate` from the aggregate
metrics. They are computed after masked rollouts leave the denominator; Gym's `coverage/masked_rollouts` says how many
did. (Gym removes masked rows before the agent's own metrics run, so `agentdyn/masked_rollout_count` reads 0 there.)

Two operational notes:

- **The agent's `concurrency` is fixed at 1**, and the config refuses anything higher: it is load-bearing, not a
  tuning knob (see the agent README). Run arms or models as separate processes for throughput.
- **CaMeL can generate a program that never terminates.** Such a rollout is abandoned after 3600s and masked. After two
  abandons the agent exits to release the stuck threads and the stack shuts down; rerun `gym eval run` with `--resume`
  and it continues from the rows already collected.

## Comparing with the AgentDyn paper

Upstream reports undefended targeted attack success over these suites from 48.9% (`gpt-4o-mini`) to 99.6%
(`gpt-5-mini`). Measured here, undefended:

| Model | Benign utility | Utility under attack | Attack success |
|---|---:|---:|---:|
| `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` | 70.00% | 64.11% | 0.71% |
| `moonshotai/Kimi-K3` | 76.67% | 76.07% | 0.18% |
| `Qwen/Qwen3.5-122B-A10B-FP8` | 70.00% | 61.96% | 35.36% |
| `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` | 70.00% | 65.71% | 15.89% |

These were collected without `--temperature`, so every arm except CaMeL ran at each endpoint's default sampling rather
than upstream's 0. The gap is still a model result -- attacked trajectories carry the full injection payload and the
model declines it -- with one difference in how the two harnesses count. Upstream's `benchmark.py` sets `utility =
False, security = True` when a rollout raises `JSONDecodeError`, a provider `ServerError` or internal-server
`ApiError`, or `context_length_exceeded`, and its `security` means the injection *succeeded*: an infrastructure failure
is scored as a successful attack. This adapter masks those rollouts instead, so they leave the denominator rather than
inflate the attack success rate.

## PromptGuard2 without Meta access

`prompt_guard_2_detector` loads `meta-llama/Llama-Prompt-Guard-2-86M`, which is **gated**: it needs an `HF_TOKEN` for
an account Meta has granted, and fails outright rather than silently without one. An ungated mirror carries the same
weights:

```text
project-free-llama/Llama-Prompt-Guard-2-86M
revision 43882965632dcb7b20299530f6436ac759d07fd9
```

All five files, `model.safetensors` included, hash identically to canonical
`meta-llama/Llama-Prompt-Guard-2-86M@a8ded8e697ce7c355e395a0df51f94adb4a2fd27`. Verify it rather than taking it on
trust (needs an `HF_TOKEN` with access to the gated repository; exits 0 when every file matches):

```bash
HF_TOKEN=... python benchmarks/agentdyn/check_prompt_guard_provenance.py
```

Then point the agent at the mirror:

```yaml
prompt_guard_2_model_name: project-free-llama/Llama-Prompt-Guard-2-86M
prompt_guard_2_model_revision: 43882965632dcb7b20299530f6436ac759d07fd9
```

Each rollout records the repository and revision that classified it, so a run against the mirror is self-describing.
