# Legal Agent Bench

This resource server runs the public
[Legal Agent Benchmark (LAB)](https://github.com/harveyai/harvey-labs/tree/f46ef86e4788545622db25dcffa3aebb7a139929)
through NeMo Gym and Harbor. The integration is pinned to upstream commit
`f46ef86e4788545622db25dcffa3aebb7a139929`: 1,749 tasks and the public
`docx`, `pptx`, and `xlsx` skills.

NeMo Gym schedules rollouts, the custom Harbor agent works with each task's
documents inside Docker, and the task-local verifier scores every rubric
criterion with an OpenAI-compatible judge model.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Docker with a running daemon (the only supported container backend)
- An OpenAI-compatible policy endpoint and judge endpoint
- At least 10 GB of free working space for preparation and the first Docker build

The pinned source download is about 579 MiB; allow a few GiB of free working
space during preparation. The first task also builds a
document-tooling Docker image and can take several minutes. Later tasks reuse
Docker layers.

From a fresh clone, create the repository environment:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync --extra dev
docker info >/dev/null
```

Add endpoint settings to the gitignored root `env.yaml`:

```yaml
policy_base_url: https://your-policy-endpoint.example/v1
policy_api_key: your-policy-key
policy_model_name: your-policy-model

judge_base_url: https://your-judge-endpoint.example/v1
judge_api_key: your-judge-key
judge_model_name: your-judge-model
```

The judge credentials are injected only into the regenerated, gitignored
runtime task tree. They are never written into the pristine cache.

## Prepare explicitly (recommended)

From the repository root:

```bash
python resources_servers/legal_agent_bench/prepare.py
```

The command downloads the pinned LAB source archive from GitHub with retries and visible
progress, verifies SHA-256
`e45cbdf3236b22866e034bcc62fb23bf00ef2f2e49db7a0cd8a4b07dbae9212c`,
rejects unsafe archive entries, generates deterministic Harbor tasks, and
builds each cache in staging before replacing the previous valid cache. A
handled preparation failure leaves the previous cache in place.

Useful options:

```bash
python resources_servers/legal_agent_bench/prepare.py --asset tasks
python resources_servers/legal_agent_bench/prepare.py --asset skills
python resources_servers/legal_agent_bench/prepare.py --force
python resources_servers/legal_agent_bench/prepare.py \
  --tasks-dir /custom/task-cache \
  --skills-dir /custom/skills-cache
```

## Prepare through the normal Gym workflow

The committed `data/all.jsonl` is a lightweight 1,749-row task index, so
`ng_prepare_data` works without downloading binary assets. If you skip
`prepare.py`, `ng_run` prepares missing task and skill caches during startup.

Set the two configs used by the benchmark:

```bash
CONFIG_PATHS=resources_servers/legal_agent_bench/configs/legal_agent_bench.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml
```

Validate the five examples:

```bash
ng_prepare_data "+config_paths=[${CONFIG_PATHS}]" \
  +output_dirpath=results/legal_agent_bench_prepare \
  +mode=example_validation
```

To collate the full validation index instead:

```bash
ng_prepare_data "+config_paths=[${CONFIG_PATHS}]" \
  +output_dirpath=results/legal_agent_bench_prepare \
  +mode=train_preparation
```

## Run and smoke test

Start the servers:

```bash
ng_run "+config_paths=[${CONFIG_PATHS}]"
```

On a clean cache, startup visibly downloads and prepares the pinned assets.
Every startup regenerates the runtime task tree so old judge credentials cannot
be reused.

In a second activated terminal, collect one rollout:

```bash
CONFIG_PATHS=resources_servers/legal_agent_bench/configs/legal_agent_bench.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml

ng_collect_rollouts "+config_paths=[${CONFIG_PATHS}]" \
  +agent_name=legal_agent_bench_harbor_agent \
  +input_jsonl_fpath=resources_servers/legal_agent_bench/data/example.jsonl \
  +output_jsonl_fpath=results/legal_agent_bench_smoke_rollout.jsonl \
  +num_samples_in_parallel=1 \
  +limit=1
```

The default `full_task` reward is LAB's official all-criteria score: a task
earns `1.0` only when every criterion passes. For diagnostic partial credit,
start with:

```bash
ng_run "+config_paths=[${CONFIG_PATHS}]" \
  +legal_agent_bench.resources_servers.legal_agent_bench.reward_mode=criteria_pass_rate
```

The verifier evaluates up to six criteria concurrently, matching the upstream LAB
default. Adjust the resource setting when the judge endpoint has a lower
concurrency limit:

```bash
ng_run "+config_paths=[${CONFIG_PATHS}]" \
  +legal_agent_bench.resources_servers.legal_agent_bench.judge_parallelism=2
```

This setting is forwarded to each task verifier as
`LAB_JUDGE_PARALLELISM`.

The task container requires network access because its verifier calls the
configured judge endpoint. The agent and verifier share that container, so the
agent also has network access during a rollout. This differs from the upstream LAB
closed-network reference sandbox and should be recorded when comparing runs.

## Caches and outputs

The default paths are:

- Pristine generated tasks: `data/cache/harbor_tasks/legal_agent_bench`
- Public skills: `data/cache/harness/skills`
- Credential-bearing runtime tasks: `data/runtime/harbor_tasks/legal_agent_bench`
- Harbor jobs: `results/legal_agent_bench/harbor_jobs`
- Rollout output: the path passed to `ng_collect_rollouts`

The runtime tree hardlinks immutable documents from the pristine cache when the
filesystem permits it, avoiding a second copy of the large document corpus.
Set `auto_prepare_assets: false` to require a prepopulated valid cache and avoid
network access at startup.

Each successful Harbor trial contains `result.json`, `verifier/reward.json`,
`verifier/scores.json`, `verifier/transcript.jsonl`, `agent/trajectory.json`,
and `agent/artifacts/lab-run/transcript.jsonl`. The agent config artifact
should list exactly `docx`, `pptx`, and `xlsx`.

## Troubleshooting

- A checksum, corrupt archive, unsafe path, wrong task count, or missing skill
  fails before replacing an existing valid cache.
- A missing judge setting produces a verifier error in the trial artifacts.
  Confirm the endpoint permits the exact `judge_model_name`.
- Treat a nonzero `judge_error_count` or `verifier_error` as a judge or
  infrastructure failure, not an ordinary model failure, even though Harbor
  receives a numeric zero reward so it can preserve a complete trial result.
- If Docker appears idle on the first rollout, inspect `docker ps` and the
  `ng_run` terminal; Harbor is normally building the task image.
- Do not copy or publish `data/runtime/`: it can contain local judge credentials.
- Results are revision-specific and should not be compared directly with runs
  that use a different task snapshot or skill set.

## Licensing and source modifications

Harvey LAB is MIT-licensed. Its license and the details of the minimal modified
runtime are in `vendor/harvey_labs/`. Task documents and public skills are
downloaded from the pinned public repository and are never tracked here.
