# JobBench

[JobBench](https://job-bench.github.io/) evaluates agents on the multi-source data preprocessing that professionals
most want offloaded: reconciling contradictory records, cross-referencing spreadsheets, tracing citations. Tasks come
from Workbank, a worker-desire survey spanning 35 white-collar occupations, so the benchmark asks what workers
actually want automated rather than what happens to be automatable.

This resources server wraps the upstream [job-bench-eval](https://github.com/Job-Bench/job-bench-eval) harness.
An OpenCode agent works in a sandbox seeded with one task's `task_folder/` and writes deliverables to an output
directory; the server then pulls that directory back, renders every file as text, and grades it rubric-by-rubric with
an LLM judge.

| | |
|---|---|
| Splits | `main` (65 tasks, the leaderboard split), `easy` (63 simplified tasks) |
| Dataset | [`JobBench/job-bench`](https://huggingface.co/datasets/JobBench/job-bench) |
| Agent | `opencode_sandboxed_agent` |
| Reward | Weighted normalized score — `total_score / max_score`, in `[0, 1]` |
| Domain | `agent` |

## Reward

Each task ships a `RUBRICS.json` of weighted rubrics, and each rubric holds several criteria. A rubric scores its full
weight only when the judge passes **every** criterion; otherwise it scores zero. The reward is the weighted sum over
the task's maximum weight, matching upstream's `normalized_score`. `pass_rate` (the fraction of rubrics fully passed)
is also reported but is not the reward.

Because it is a weighted rubric score, the reward is continuous rather than binary — a partially correct deliverable
lands between 0 and 1.

## Prepare

`prepare.py` pulls the dataset from Hugging Face into a gitignored control-plane cache and writes the model-visible
JSONL. Rubrics, task cards and `files_required_to_search/` stay in the cache and never reach the agent or the JSONL.

```bash
python -m resources_servers.job_bench.prepare --split main
```

The cache lands at `resources_servers/job_bench/data/cache/tasks/<split>/<profession>/<taskN>/`, and the benchmark
JSONL at `benchmarks/job_bench/data/job_bench_main_benchmark.jsonl`. Re-running is a no-op; pass `--force` to
re-download. `--no-download` prepares from an existing cache.

## Build the sandbox image

JobBench tasks manipulate xlsx, docx, pdf and sqlite files and produce plots, so the sandbox needs a data stack. There
is no per-task image upstream, so this server ships one:

```bash
docker build -t job-bench-sandbox:latest resources_servers/job_bench
docker push <your-registry>/job-bench-sandbox:latest
export JOB_BENCH_SANDBOX_IMAGE=<your-registry>/job-bench-sandbox:latest
```

`sandbox_image` defaults to `job-bench-sandbox:latest` and reads `JOB_BENCH_SANDBOX_IMAGE` when set.

## Run OpenCode rollouts

Launch with a model config and a sandbox-provider config. Add `+use_absolute_ip=true` when the sandbox needs a
host-routable model-server address.

```bash
gym env start \
    --config benchmarks/job_bench/opencode.yaml \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    --config responses_api_models/<model>/configs/<model>.yaml \
    +job_bench_opencode_resources_server.resources_servers.job_bench.judge_model_server.name=<judge_model>
```

```bash
gym eval run --no-serve \
  --agent job_bench_opencode_sandboxed_agent \
  --input benchmarks/job_bench/data/job_bench_main_benchmark.jsonl \
  --output results/job_bench_rollouts.jsonl
```

## Judge

Grading is one chat completion per rubric against `judge_model_server`, not an agentic judge loop — that is what keeps
a full pass affordable. Deliverables are rendered to text up front (`max_chars_per_file`, default 200k, guards against
a model dumping a multi-MB input into its output). Rubrics whose text mentions a plot, figure, histogram, Q-Q or
similar additionally get up to `max_vision_images` images attached, pulled from standalone image files, docx-embedded
media and notebook outputs; those rubrics need a judge that accepts multimodal input.

Upstream validates rubric behavior against `grok-4.3` only. Other judges are exploratory — treat cross-judge
comparisons with care.

A judge call that fails at the transport level raises `JudgeError`, and Gym routes the row to the failures sidecar
rather than scoring it zero. A reply that arrives but cannot be parsed is a genuine failed grade and scores zero.

## Network posture and `files_required_to_search/`

Upstream's runner copies only `task_folder/` into the agent's workspace and imposes no network restriction; several
tasks expect the agent to rediscover reference material online, and the upstream README notes that agents without
browsing score lower. This server matches that by default:

- `enforce_agent_no_network: false` — the sandbox reaches the internet, and `benchmarks/job_bench/opencode.yaml`
  re-enables OpenCode's `webfetch`/`websearch` tools that the SWE presets disable.
- `include_search_files: false` — `files_required_to_search/` is withheld, as upstream withholds it.

Two knobs change this, both of which move results off the leaderboard's footing:

- `enforce_agent_no_network: true` gives a hermetic run (deny-all egress, plus the model server when
  `sandbox_model_server` is set). Search-dependent tasks will score near zero.
- `include_search_files: true` mounts the search corpus at `/workspace/files_required_to_search`. Combined with the
  above it makes runs fully deterministic, but hands the agent material it was meant to find, inflating scores.

Changing `include_search_files` also changes the agent prompt, so re-run `prepare.py` with
`--include-search-files` to keep the JSONL consistent with the server.

## Sandbox layout

```
/workspace/task_folder/     # task materials, including TASK_INSTRUCTIONS.txt
/workspace/output/          # the agent writes its final deliverables here; this is what gets graded
```

Only `/workspace/output` is collected. Anything the agent leaves in `task_folder/` or elsewhere is ignored, exactly as
upstream ignores it.

## Tests

```bash
gym env test --resources-server job_bench
```
