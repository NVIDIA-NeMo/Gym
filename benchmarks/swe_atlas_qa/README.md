# SWE Atlas Codebase Q&A

The 124 Codebase Q&A tasks from [SWE Atlas](https://github.com/scaleapi/SWE-Atlas), run through `harness_agent`
and graded by the unchanged upstream rubric evaluator. The tasks, answers, rubrics, and trajectories are
evaluation-only. Do not train on them.

## Run

```bash
gym eval run --config benchmarks/swe_atlas_qa/mini_swe.yaml --split benchmark \
  --output runs/swe-atlas/mini_swe.jsonl +use_absolute_ip=true ++route_failures_to_sidecar=true
```

The run prepares data first: `prepare.py` downloads upstream revision `c6673fe2`, rejects modified task assets, and
writes the gitignored `resources_servers/swe_atlas_qa/data/benchmark.jsonl`.

Set these in `env.yaml` or on the command line:

| Key | Value |
|---|---|
| `policy_base_url`, `policy_model_name`, `policy_api_key` | OpenAI-compatible policy endpoint |
| `judge_base_url`, `judge_model_name`, `judge_api_key` | Rubric judge, reachable from the verifier sandbox. Claude Opus 4.5 matches the leaderboard. |
| `harness_bundle` | Harness bundle path or URL, see below |
| `harness_python` | Python command inside the sandbox (default `python3`) |

Also configure the OpenSandbox provider's connection settings.

- Task sandboxes call the Gym model server and can reach only its host. `+use_absolute_ip=true` makes it listen on
  this machine's address. When sandboxes cannot reach this machine (for example behind a VPN), relay a reachable
  port to it and set `++swe_atlas_qa_resources_server.resources_servers.swe_atlas_qa.sandbox_model_base_url=<relay URL>`.
- Raise `swe_atlas_qa_agent.responses_api_agents.harness_agent.concurrency` (default 1) for a full run.
  `route_failures_to_sidecar` keeps one failed rollout from ending the run. Count rows in `rollouts_failures.jsonl` as zero.

## Harness bundle

`harness_bundle` is a tarball with `nemo_gym/`, the harness's `responses_api_agents/<agent>/` package, and a Linux
x86_64 runtime for the harness. Each task sandbox unpacks it to `/work/gym_mount` and starts it with `harness_python`.
The runtime must also run on the three Alpine images (k6, MinIO, and SFTPGo). A URL must be on the model host, the only
host task sandboxes can reach.

## Harnesses

`benchmarks/swe_atlas_qa/<harness>.yaml` holds a tested config for `claude_code`, `cline`, `codex`, `deepagents`,
`hermes`, `kilocode`, `mini_swe`, `openclaw`, `opencode`, `pi`, `prime`, `simple_strands`, and `terminus_2`. Version
pins and `/work/gym_mount` paths must match your bundle. `deepagents` needs #5051. `--benchmark swe_atlas_qa` alone
runs OpenCode.

## Results

| Harness | Policy | Score, 124 tasks x 3 | Leaderboard |
|---|---|---|---|
| `mini_swe` | GLM-5.2 | 50.8% (189/372, failures as 0) | [48.12 +/- 5.07](https://labs.scale.com/leaderboard/sweatlas-qna) |
