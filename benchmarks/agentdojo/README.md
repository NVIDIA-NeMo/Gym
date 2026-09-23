# AgentDojo

Official [AgentDojo](https://github.com/ethz-spylab/agentdojo) (Debenedetti et al., NeurIPS 2024 Datasets and
Benchmarks) at tag `v0.1.35`, commit `a75aba7631d3ca5fb7ab938965c97ead2f9ff84b`, benchmark version `v1.2.2`, run by
`responses_api_agents/agentdojo_agent` (see its README for the adapter contract, result fields and defenses).

```bash
gym eval prepare --benchmark agentdojo
```

writes `data/agentdojo_benchmark.jsonl`: every user task once clean, and every (user task, injection task) pair
under the `important_instructions` attack.

| Suite | User tasks | Injection tasks | Clean rows | Attacked rows |
| --- | ---: | ---: | ---: | ---: |
| banking | 16 | 9 | 16 | 144 |
| slack | 21 | 5 (numbered 1–5) | 21 | 105 |
| travel | 20 | 7 | 20 | 140 |
| workspace | 40 | 14 | 40 | 560 |
| **total** | **97** | **35** | **97** | **949** |

1,046 rows; sha256 `e59fd9b9894bfd6eed7a706d165721778dd4820c4af2ca68635f4430726f27f8`. The counts match
`get_suites("v1.2.2")` at the pinned commit, and the agent tests check the task ids against it. Rows carry task ids only; upstream loads each prompt and environment.
`prepare.py` needs no network and no AgentDojo install.

## Arms

A defense is selected per run with the agent's `default_defense`, so every arm scores the same file:

| Arm | `default_defense` |
| --- | --- |
| undefended | `null` |
| tool filter | `tool_filter` |
| prompt-injection detector | `transformers_pi_detector` |
| spotlighting | `spotlighting_with_delimiting` |
| repeated user prompt | `repeat_user_prompt` |

These are all the defenses the pinned upstream registers. CaMeL, Progent, DRIFT, PIGuard and PromptGuard 2 are not
part of official AgentDojo.

```yaml
# e.g. env.yaml or a second --config
agentdojo_benchmark:
  responses_api_agents:
    agentdojo_agent:
      default_defense: tool_filter
```

## Metrics

`agentdojo/benign_utility` over clean rows; `agentdojo/utility_under_attack` and `agentdojo/attack_success_rate` over
attacked rows. These are upstream's utility, utility-under-attack and targeted attack success rate. Masked rows
(adapter or provider failures) are counted separately and excluded.

## Throughput

The agent runs one rollout at a time per process (`concurrency: 1`, fixed; see the agent README). `gym eval run
--concurrency` above 1 only queues requests at that agent. Run arms or models in parallel as separate
`gym env start` stacks with disjoint ports.

## Smoke

`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`, undefended, 5 clean rows (the agent's `data/example.jsonl`) and
4 attacked rows (`banking/user_task_0 × injection_task_0`, `slack/user_task_0 × injection_task_1`,
`travel/user_task_0 × injection_task_0`, `workspace/user_task_0 × injection_task_0`): benign utility 4/5, utility
under attack 3/4, attack success 0/4, 0 masked. The injected text was present in a tool result of all four attacked
rollouts. Each of the four defenses also ran on the banking and travel attacked rows with 0 masked rows. This is an
integration check, not a baseline.

## License

AgentDojo code and task data: MIT.
