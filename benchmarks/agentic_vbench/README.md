# Agentic-VBench

Evaluate video-editing agents on the 100 pinned [Agentic-VBench](https://github.com/PhiloLabs/agentic-vbench)
tasks: Repair (18), Assembly (18), Sequencing (28) and Repurpose (36). The `agentic_vbench_agent`
adapter runs each task verbatim through the benchmark's own harness, upstream Harbor 0.6.6 with
OpenCode 1.14.39, in the task's own container, and reads the reward the task's own verifier wrote.
It does not re-implement the benchmark as a question-answering task and never re-scores a trajectory.

## Measurement contract

| | |
|---|---|
| Source | `PhiloLabs/agentic-vbench` @ `410a75fe3dae37de4344fc9e5317da089505330a` (Apache-2.0); the four leaderboard families, the "understanding" family is excluded as upstream does |
| Task contract | The task prompt is handed to OpenCode unchanged; the model works in the task container with OpenCode's tools and must inspect extracted frames itself (no media is injected into the prompt) |
| Harness | Harbor 0.6.6, OpenCode 1.14.39, OpenCode's default agent configuration (`opencode_subagents: default`); the policy is reached through Gym's policy model server, so served model name, seed and sampling overrides apply |
| Model budget | 262,144 context tokens and 100,000 output tokens advertised to OpenCode; each task's own agent and verifier time limits apply (verifiers get `verifier_timeout_multiplier` x their limit) |
| Reward | The task's `reward.json`, 0..1, higher is better; `equal_family_reward` = reward x 25 / family size, whose task-weighted mean is the leaderboard number |
| Aggregate | `mean/equal_family_reward` (equal mean of the four family means), reported as the average of three complete rollouts with request seeds 201, 202 and 203; Gym's task-weighted `mean/reward` is kept alongside |
| Judge | Repurpose tasks carry an LLM rubric. `judge_protocol: nvinference-hybrid` (default) scores it through the NVIDIA Inference Hub (Claude Opus 4.7 + Gemini 3.5 Flash) on `INFERENCE_API_KEY`/`NVINFERENCE_API_KEY`; `null` keeps the task's native Anthropic/Gemini judges, which then need `ANTHROPIC_API_KEY`/`GEMINI_API_KEY` in `credentials_file`. Hub-judged scores are protocol-equivalent, not byte-identical, to the native judges; the protocol used is recorded per trial in `judge_protocol.json` |
| Determinism | Stochastic (sampling, container timing, judge) |

## Runtime

Task containers need a Docker API. With Docker available locally the stock Harbor environment is
used (`backend: podman` for rootless Podman). On clusters where the Gym job itself cannot run
containers, `backend: remote` (the default) talks over mTLS to a rootless Podman backend on
another node, found through `backend_dir/client.env` (`DOCKER_HOST`, `DOCKER_TLS_VERIFY`,
`DOCKER_CERT_PATH`); the Eval Factory configuration for this benchmark ships the backend job
(`scripts/agentic_vbench/` there). About 12 concurrent trials saturate one 64-core backend
node, and the first trial of each task builds its image there (task Dockerfiles download their
media from Hugging Face), so prebuild the images or expect long first trials.

Inside a Gym job everything else is materialized once under the cache root
(`$AGENTIC_VBENCH_CACHE_ROOT`, else `$HF_HOME/../agentic_vbench`): the pinned benchmark checkout
and a static Docker CLI. The agent's own venv hosts `harbor==0.6.6`. Explicit paths
(`AGENTIC_VBENCH_ROOT`, `AGENTIC_VBENCH_HARBOR_PYTHON`, `AGENTIC_VBENCH_DOCKER_CLI_DIR`) serve
launchers that stage these themselves; `AGENTIC_VBENCH_MODEL_BASE_URL`/`AGENTIC_VBENCH_MODEL_ID`
point OpenCode at an endpoint the launcher serves instead of Gym's policy server.

Run-level knobs (`agentic_vbench_*` in `config.yaml`): `concurrency` (12), `judge_protocol`,
`verifier_timeout_multiplier` (3), `context_tokens`, `output_tokens`, `max_turns` (unlimited),
`opencode_subagents` (`default`/`off`/`on`), `keep_task_images` (false; the benchmark deletes
task images after each trial) and `episode_timeout_seconds` (12 h). `AGENTIC_VBENCH_TASKS`
selects `all` (default) or a space-separated list of families and exact task ids for
preparation; preparation validates the whole pinned inventory even for a subset.

```bash
export AGENTIC_VBENCH_ROOT=/path/to/pinned/agentic-vbench      # optional: reuse an existing checkout
export AGENTIC_VBENCH_BACKEND_DIR=/path/with/client.env          # remote backend
export INFERENCE_API_KEY=...                                     # Hub judge (Repurpose)
gym eval prepare --benchmark agentic_vbench
gym eval run --benchmark agentic_vbench --split benchmark --num-repeats 1 \
    --output results/agentic_vbench_seed201.jsonl \
    ++policy_model.responses_api_models.vllm_model.sampling_overrides.seed=201
```

Set `++observability_enabled=true ++model_call_capture_dir=<absolute dir>` for a local run;
`gym eval submit` switches capture on itself. OpenCode's calls go through the
`/ng-rollout/<id>` prefix of the policy server, so they are captured per rollout.

## Results and failure handling

Every scored row keeps reward, family, task id, verifier status, the original Harbor trajectory
and the episode artifact path (`<rollouts>/agentic_vbench_episodes/<task>-<key>/`). A completed
zero-reward trajectory is a result and is never rerun. An infrastructure failure (environment
start timeout, unreachable backend, Harbor runner crash) raises instead of inventing a zero:
with `route_failures_to_sidecar` the run continues and the rollout lands in the failures
sidecar, and a resumed job (`resume_from_cache`) redispatches it; the episode a killed job left
behind is archived as `*.incomplete-<timestamp>` and run again, which is an infrastructure
retry, not a second trajectory. `count_failure_classes_as_zero: [agent_run_error]` scores what
is still unscored as 0, the leaderboard's convention for failed rollouts.

The adapter produces no token ids or logprobs; it is for evaluation, not RL training. Do not
report a subset or a single-seed result as the benchmark score.

## Reference results (100 tasks x 3 seeds, Hub judge, OpenCode 1.14.39)

| Model | Avg | Repurpose | Sequencing | Repair | Assembly |
|---|---:|---:|---:|---:|---:|
| Super 3.5 GA (vLLM 0.26, TP4, temperature 0.6, top-p 0.95) | 12.2 ± 0.4 | 16.5 | 5.2 | 15.7 | 11.3 |
| Kimi K3 (NVIDIA Inference Hub) | 36.3 ± 1.8 | 39.2 | 40.8 | 25.6 | 39.5 |
| Qwen3.8-Flash-Next (vLLM, temperature 1.0, top-p 0.95) | 37.5 ± 2.5 | 46.4 | 30.0 | 17.9 | 55.8 |

Avg is the unweighted mean of the four family means in percent; ± is the population standard
deviation across the three seeds; failed rollouts score 0. Measured through NVIDIA Eval Factory
on nsc-svg with this agent (2026-10).
