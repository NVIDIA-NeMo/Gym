# Agentic-VBench

Evaluate video-editing agents on the 100 pinned [Agentic-VBench](https://github.com/PhiloLabs/agentic-vbench)
tasks: Repair (18), Assembly (18), Sequencing (28) and Repurpose (36). The tasks are Harbor task folders and run
through Gym's generic [`harbor` environment server](../../environment_servers/harbor/README.md) with any sandboxed
session harness; the committed entry is [`environments/harbor/agentic_vbench/`](../../environments/harbor/agentic_vbench/README.md).
This folder holds the code that prepares it.

| File | Role |
| --- | --- |
| `prepare_harbor_env.py` | Writes the 100 pinned tasks into the entry (`python -m benchmarks.agentic_vbench.prepare_harbor_env`), or a root of generated training tasks in the same layout into the training entry [`environments/harbor/agentic_vbench_train/`](../../environments/harbor/agentic_vbench_train/README.md) (`--tasks-root <root>`) |
| `checkout.py` | Fetches the pinned upstream revision into the Gym cache root and lists the four families' tasks |
| `staging.py` | The per-task adjustments: Hub judge routing, assembly judge guard, bounded golden-asset retries, exit 75 for verifier-side outages, `/workspace/output` as the judged and repair steps' artifact; and the staging of the OpenCode compaction plugins into every step's workdir |
| `hub_judge.py` | Drop-in for the repurpose tasks' `judge.py`: the unchanged rubric judge with both providers served by the NVIDIA Inference Hub |
| `family_metrics.py` | Per-family means and the equal-family mean of a rollouts file (`python -m benchmarks.agentic_vbench.family_metrics <rollouts.jsonl>`), from the family each task's `task_data.json` names |
| `opencode_plugins/` | OpenCode plugins for sessions that compact (summary ordering, task-prompt anchor, the agent header the model server's summary recovery reads); loaded by the training entry's `compaction_plugins.yaml` |

## Measurement contract

| | |
|---|---|
| Source | `PhiloLabs/agentic-vbench` @ `410a75fe3dae37de4344fc9e5317da089505330a` (Apache-2.0); the four leaderboard families, the "understanding" family is excluded as upstream does |
| Task contract | The task's own `instruction.md` is the prompt; the model works in the task's own image with the harness's tools and must inspect extracted frames itself (no media is injected into the prompt). Reference solutions stay under `solution/`, which the agent's sandbox does not see |
| Harness | Any session harness the `harbor` server can run in a sandbox; `opencode_agent` with `image_input` (set by the entry's overlay) is the one under validation (the published numbers come from OpenCode 1.14.39 inside the task container) |
| Budget | Each task declares 1800 s for the agent and 1800 s for the verifier; the entry's `dataset.toml` gives the verifiers the published 3x allowance (`[gym.defaults] verifier_timeout_multiplier = 3.0`) and leaves the agent budgets as declared |
| Reward | The task's `reward.json`, 0..1, higher is better; the leaderboard number is `(mean_repair + mean_assembly + mean_sequencing + mean_repurpose) / 4` with the unscaled family means, equivalently the mean of `reward_i x 25 / size_of_family_i` over the complete 100 rows; a subset is not the benchmark score, and a masked row is not a 0 |
| Judge | Repurpose tasks carry an LLM rubric judged over the NVIDIA Inference Hub on `NVINFERENCE_API_KEY` (or `INFERENCE_API_KEY`); without Hub access the 36 Repurpose tasks cannot be scored; the other 64 need no credentials (prepare them with `AGENTIC_VBENCH_HARBOR_FAMILIES=repair,assembly,sequencing` and run with the entry's `harbor_sandbox_no_judge.yaml`). `AGENTIC_VBENCH_JUDGE_PROTOCOL=nvinference-upstream` (default) uses the upstream pair, Claude Opus 4.7 and Gemini 3.1 Pro; `nvinference-hybrid` uses Claude Opus 4.8 and Gemini 3.5 Flash with bounded thinking. Hub-judged scores are protocol-equivalent, not byte-identical, to the upstream SDK judges; the protocol is recorded per task in the entry's `.records/`. The Gemini judge inlines the output clip as base64 where the SDK would upload it; a clip over the 14 MiB inline budget goes as an ffmpeg proxy (at most 720p, audio kept, recorded in the result's provenance), and a request the route still refuses for its size or length is that rubric item's error, as upstream |
| Infrastructure failures | A judge route the key cannot use, a Hub outage, a missing judge key, the judge's allowance of time waiting on the Hub (1200 s) running out, or a failed golden-asset download make the verifier exit 75 without a reward; the server masks the rollout (`harbor:verifier_unavailable`) instead of scoring 0. The judged and repair tasks verify in a separate sandbox, the only path on which the server honors the code, so the agent's sandbox cannot produce it; the verifier derives it from the Hub's answers and the download's outcome, never from the agent's output, whose size or length only fails its own items |
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
Factory. Two conventions differ from this entry's defaults: the reference runs judged with the `nvinference-hybrid`
pair, where the entry defaults to `nvinference-upstream` (select hybrid with `AGENTIC_VBENCH_JUDGE_PROTOCOL` to
compare), and they scored failed rollouts 0, where the entry masks a verifier that could not grade. They are the
reference the entry is compared against, not results of the entry: the entry has not been baselined with
`gym eval run` yet (no oracle validate output and no full-set run are recorded), so `verified` stays false in
`dataset.toml`. Do not report a subset or a single-seed result as the benchmark score.
