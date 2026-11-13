# Agentic-VBench

Evaluate video-editing agents on the 100 pinned [Agentic-VBench](https://github.com/PhiloLabs/agentic-vbench)
tasks: Repair (18), Assembly (18), Sequencing (28) and Repurpose (36). The tasks are Harbor task folders and run
through Gym's generic [`harbor` environment server](../../environment_servers/harbor/README.md) with any sandboxed
session harness; the committed entry is [`environments/harbor/agentic_vbench/`](../../environments/harbor/agentic_vbench/README.md).
This folder holds the code that prepares it.

| File | Role |
| --- | --- |
| `prepare_harbor_env.py` | Writes the 100 pinned tasks into the entry (`python -m benchmarks.agentic_vbench.prepare_harbor_env`) |
| `checkout.py` | Fetches the pinned upstream revision into the Gym cache root and lists the four families' tasks |
| `staging.py` | The per-task adjustments: Hub judge routing, assembly judge guard, golden-asset retries and inconclusive verdicts |
| `hub_judge.py` | Drop-in for the repurpose tasks' `judge.py`: the unchanged rubric judge with both providers served by the NVIDIA Inference Hub |

## Measurement contract

| | |
|---|---|
| Source | `PhiloLabs/agentic-vbench` @ `410a75fe3dae37de4344fc9e5317da089505330a` (Apache-2.0); the four leaderboard families, the "understanding" family is excluded as upstream does |
| Task contract | The task's own `instruction.md` is the prompt; the model works in the task's own image with the harness's tools and must inspect extracted frames itself (no media is injected into the prompt). Reference solutions stay under `solution/`, which the agent's sandbox does not see |
| Harness | Any session harness the `harbor` server can run in a sandbox; `opencode_agent` with `image_input` is the one under validation (the published numbers come from OpenCode 1.14.39 inside the task container) |
| Budget | Each task declares 1800 s for the agent and 1800 s for the verifier; `[gym.defaults] timeout_multiplier` scales both |
| Reward | The task's `reward.json`, 0..1, higher is better; the leaderboard number is the equal-family mean (`reward x 25 / family size` averaged over the four families), computed from the rows by task family |
| Judge | Repurpose tasks carry an LLM rubric judged over the NVIDIA Inference Hub on `NVINFERENCE_API_KEY` (or `INFERENCE_API_KEY`). `AGENTIC_VBENCH_JUDGE_PROTOCOL=nvinference-native` (default) uses the upstream pair, Claude Opus 4.7 and Gemini 3.1 Pro; `nvinference-hybrid` uses Claude Opus 4.8 and Gemini 3.5 Flash with bounded thinking. Hub-judged scores are protocol-equivalent, not byte-identical, to the native SDK judges; the protocol is recorded per task in the entry's `.records/` |
| Infrastructure failures | A judge-service outage, a missing judge key or a failed golden-asset download writes `inconclusive.json`; the server masks the rollout (`harbor:verifier_inconclusive`) instead of scoring 0 |
| Determinism | Stochastic (sampling, container timing, judge) |

## Reference results

| Model | Avg | Repurpose | Sequencing | Repair | Assembly |
|---|---:|---:|---:|---:|---:|
| Super 3.5 GA (vLLM 0.26, TP4, temperature 0.6, top-p 0.95) | 12.2 ± 0.4 | 16.5 | 5.2 | 15.7 | 11.3 |
| Kimi K3 (NVIDIA Inference Hub) | 36.3 ± 1.8 | 39.2 | 40.8 | 25.6 | 39.5 |
| Qwen3.8-Flash-Next (vLLM, temperature 1.0, top-p 0.95) | 37.5 ± 2.5 | 46.4 | 30.0 | 17.9 | 55.8 |

Avg is the unweighted mean of the four family means in percent over three complete rollouts (request seeds 201,
202 and 203); ± is the population standard deviation across the seeds. These numbers were measured in 2026-10 with
the benchmark's own runtime (Harbor 0.6.6 with OpenCode 1.14.39 in the task container) driven from NVIDIA Eval
Factory, with the `nvinference-hybrid` judge pair and failed rollouts scored 0. They are the reference the entry is
compared against, not results of the entry: the entry has not been baselined with `gym eval run` yet (no oracle
validate output and no full-set run are recorded), so `experimental` stays true. Do not report a subset or a
single-seed result as the benchmark score.
