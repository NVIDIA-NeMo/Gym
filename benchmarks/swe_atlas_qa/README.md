# SWE Atlas Codebase Q&A

The 124 Codebase Q&A tasks from [SWE Atlas](https://github.com/scaleapi/SWE-Atlas), run through `harness_agent`
and graded by the unchanged upstream rubric evaluator. The tasks, answers, rubrics, and trajectories are
evaluation-only. Do not train on them.

## Run

```bash
gym eval run --config benchmarks/swe_atlas_qa/mini_swe.yaml --split benchmark \
  --output runs/swe-atlas/mini_swe.jsonl ++route_failures_to_sidecar=true
```

The run prepares data first: `prepare.py` downloads upstream revision `c6673fe2`, rejects modified task assets, and
writes the gitignored `resources_servers/swe_atlas_qa/data/benchmark.jsonl`.

| Setting | Value |
|---|---|
| `policy_base_url`, `policy_model_name`, `policy_api_key` | OpenAI-compatible policy endpoint |
| `SWE_ATLAS_JUDGE_BASE_URL`, `SWE_ATLAS_JUDGE_MODEL`, `SWE_ATLAS_JUDGE_API_KEY` | Rubric judge, reachable from the verifier sandbox. Claude Opus 4.5 matches the leaderboard. |
| `SWE_ATLAS_GYM_MODEL_HOST` | Address of this host that task sandboxes can reach |
| `SWE_ATLAS_HARNESS_BUNDLE` | Harness bundle path or URL |
| `SWE_ATLAS_HARNESS_PYTHON` | Python command inside the sandbox (default `python3`) |
| `OPENSANDBOX_DOMAIN`, `OPENSANDBOX_API_KEY` | OpenSandbox connection |

- Task sandboxes can reach only the model host. A workstation behind a VPN usually is not reachable, so run Gym on a
  reachable host or forward a reachable port to it and set `++policy_model.responses_api_models.vllm_model.host=127.0.0.1`.
- Raise `swe_atlas_qa_agent.responses_api_agents.harness_agent.concurrency` (default 1) for a full run.
  `route_failures_to_sidecar` keeps one failed rollout from ending the run. Count rows in `rollouts_failures.jsonl` as zero.

## Harness bundle

The bundle is extracted to `/work/gym_mount` in each task sandbox. It holds `nemo_gym/`, the harness's
`responses_api_agents/<agent>/` package, and the harness runtime. The task images are x86_64, and three of them
(k6, MinIO, and SFTPGo) are Alpine/musl, so ship a runtime with its own glibc loader. Serve a URL bundle from the
model host. Keep `SWE_ATLAS_HARNESS_PYTHON` to `PATH` changes, since other variables leak into the agent's commands.

## Harnesses

`benchmarks/swe_atlas_qa/<harness>.yaml` holds a tested config for `claude_code`, `cline`, `codex`, `deepagents`,
`hermes`, `kilocode`, `mini_swe`, `openclaw`, `opencode`, `pi`, `prime`, `simple_strands`, and `terminus_2`. Version
pins and `/work/gym_mount` paths must match your bundle. `deepagents` needs #5051. `--benchmark swe_atlas_qa` alone
runs OpenCode.

## Results

| Harness | Policy | Score, 124 tasks x 3 | Leaderboard |
|---|---|---|---|
| `mini_swe` | GLM-5.2 | 50.8% (189/372, failures as 0) | [48.12 +/- 5.07](https://labs.scale.com/leaderboard/sweatlas-qna) |
