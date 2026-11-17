---
name: nemo-gym-running-evals
description: >
  Run a model or agent evaluation with NeMo Gym and get a result you can trust: a first local run
  against an API or vLLM endpoint, a named benchmark (IFBench, GPQA, SWE-bench, Terminal-Bench and
  others), or a long run on a Slurm cluster with a self-hosted vLLM model. Covers installing Gym,
  writing the model config (vLLM service, reasoning on or off, parsers), smoking a small run first,
  submitting, resuming and recovering interrupted runs, and checking health before you report a
  score. Use when asked to "run an eval", "evaluate model Y on benchmark Z", "try this model on a
  benchmark", "benchmark my endpoint", "serve this checkpoint for evaluation", "run it on the
  cluster", "the job timed out", "resume the run", "is the run done", "can I trust this score" or
  "compare BF16 and the quantized model fairly". Not for building a benchmark or harness (use
  add-benchmark or add-harness), reading and comparing finished results (use
  nemo-gym-analyzing-results) or debugging a broken rollout (use nemo-gym-debugging).
---

# Run a Gym evaluation to a result you can trust

Take an evaluation from a request ("run model Y on benchmark Z") to a score with its health checked. Read only the reference the path needs; each reference is complete on its own.

## 1. Pick the path

| You have | Path | Read |
| --- | --- | --- |
| An endpoint (hosted API, or a vLLM server already running) and want a first run or a smoke | Local run | [references/local-and-api-run.md](references/local-and-api-run.md) |
| A checkpoint to serve, a cluster with Slurm, and a long or large run | Slurm run: compose the config, submit, recover | [references/model-config.md](references/model-config.md), then [references/slurm-run-and-recover.md](references/slurm-run-and-recover.md) |
| A task folder (Harbor format) | Local run with a sandbox | local-and-api-run.md, section 4 |
| A benchmark you must add, or a harness | Not this skill | `add-benchmark`, `add-harness` |

If the request does not say, ask for what decides it: where the model is served (endpoint URL, or checkpoint plus cluster, login host, account, partition, images and output directory), which benchmark, and whether this is a smoke or a result to report. Never invent a cluster path, account or key.

## 2. Rules that hold on every path

1. **Smoke first.** Run 5 tasks with 1 repeat before anything large (`--limit 5 --num-repeats 1`). A smoke tells you the wiring works; its score is never the benchmark's score.
2. **Pin what you run.** Install Gym at a release (`nemo-gym==<version>`) or a full commit hash, never a moving branch, and write the version next to the score. A Slurm submit config pins `driver.gym_install.ref` to a full 40-character commit.
3. **Do not change the evaluation.** The benchmark's data, prompts, grading, judge, token budgets and timeouts are what the score measures. Only `num_repeats` may change, and only upward; client concurrency may change. The model section is yours.
4. **Report what the score covers.** A score is the benchmark's only when it comes from every task at no fewer than the benchmark's `num_repeats` (its `manifest.yaml`). A `limit` or a subset is a smoke or a named subset score, labelled as one.
5. **Health before score.** A run can be healthy and score low: that is the model or its deployment, so fix the deployment, not the evaluation. A run whose evaluation-side faults (judge, verifier, sandbox, harness, transport, model-server 5xx) reach 2% of the expected trials fails health, and a run whose grading was not recorded reads `UNVERIFIED`: report both with their counts, do not present them as verified.
6. **Never touch a run's files.** Do not edit, delete or regenerate saved answers or journals. Recover through `gym eval resubmit ... --recover`, never plain `gym eval resubmit`.

## 3. Local run, in short

```bash
uv venv --python 3.13.14 && source .venv/bin/activate && uv pip install nemo-gym
gym list benchmarks                                  # choose a name
gym eval prepare --benchmark <name>                  # when the benchmark has a prepare step
gym eval run --benchmark <name> --model-type openai_model \
    --model <model> --model-url <url> --model-api-key "$KEY" \
    --split benchmark --output results/<name>/rollouts.jsonl --limit 5 --num-repeats 1
gym eval profile ...                                 # pass@1, pass@k, per-task variance
```

Details, the env.yaml form, task folders and the failure table: [references/local-and-api-run.md](references/local-and-api-run.md).

## 4. Slurm run, in short

1. **Compose the config** when there is none (follow [references/model-config.md](references/model-config.md), sections 1 to 6): read the model card, write the `vllm` service, set the reasoning toggle and parsers, point each benchmark at its own config under `benchmarks/<name>/` instead of copying it, and validate without GPUs.
2. **Submit pinned and resumable**, a smoke first:

   ```bash
   gym eval submit --config submit.yaml --resolve-only    # the resolved config, validated
   gym eval submit --config submit.yaml --dry-run         # the job scripts it would queue
   gym eval submit --config submit.yaml                   # submit; note the run's GYM_JOB_ID
   ```

   Every long benchmark needs `resumable` set under `driver.benchmarks.<name>`; `max_walltime` is the whole chain's budget, not one job's.
3. **Inspect, never guess.** The read-only inspection is your record and your next action:

   ```bash
   gym eval resubmit "$GYM_JOB_ID" <benchmark> --recover --inspect \
     --score-agent <agent name> --score-key '["agent_metrics", "mean/reward"]'
   ```

   Read `decision.action`, `gate_task.status` and `operator_verdict`; the exit code is 0 only on `SUCCESS`.
4. **Do the one action the decision table names**: run the same command without `--inspect`, with the option the row names and its reason. A decision with no row, or a stop you cannot explain: stop and report it. The table is in [references/slurm-run-and-recover.md](references/slurm-run-and-recover.md), section 4.
5. **Done** only when the inspection says `SUCCESS`. A finished job or a full score is not acceptance.

## 5. Troubleshooting

| Symptom | First thing to check |
| --- | --- |
| Endpoint unreachable, `500` on `/run`, or `model endpoint(s) never answered` | The URL and key in `env.yaml`; that the server is up. |
| Answers end mid-thought, or are all empty | The output budget and `--max-model-len`; the reasoning parser and `uses_reasoning_parser`. |
| Tool-using benchmark scores about zero | The tool-call parser and the template's `tools` branch (model-config.md, sections 1 and 6). |
| Job ends `TIMEOUT` or the chain stopped | Inspect, then the matching row of the decision table (slurm-run-and-recover.md, sections 2 and 4). |
| `server_spinup_timeout` | `--recover --server-spinup-timeout-seconds N --spinup-timeout-reason "<why>"`. |
| `UNVERIFIED` | The grading record: `trial_counts.unobserved_reason` names the server. Report it; no flag changes it. |
| Slurm can't start the judge on CPU (`QOSMinGRES`) | Pass `--judge-partition` and `--judge-qos` (slurm-run-and-recover.md, section 4). |
| A sandbox died during the task | Read the benchmark's notes before counting the score: [references/benchmark-notes.md](references/benchmark-notes.md), and section 7 of slurm-run-and-recover.md. |

## 6. Comparing models fairly

Keep the benchmark config, sampling the benchmark sets, `num_repeats`, judge, token budgets, timeouts and Gym version identical; parallelism, instances, `--max-num-seqs` and concurrency may differ. A quantized export of a model you also run in BF16 uses the BF16 config except for the fields quantization requires: [references/model-config.md](references/model-config.md), sections 8 and 9. To read and compare the finished runs, use `nemo-gym-analyzing-results`.

## Exit checks

- [ ] A smoke ran first and its health and answers looked right.
- [ ] The Gym version or commit is pinned and recorded with the score.
- [ ] The inspection (Slurm) shows `SUCCESS`, or the remaining stop is explained; a local run's aggregate metrics and health were read.
- [ ] Every operator action (walltime, retries, repin, unsafe resume) has its reason in the report.
- [ ] A reported score names its coverage (all tasks and repeats, or a smoke, or a named subset) and any `UNVERIFIED` or unhealthy count.

## References

- [references/local-and-api-run.md](references/local-and-api-run.md): install, endpoint, smoke, named benchmark, failures.
- [references/model-config.md](references/model-config.md): model card to `vllm` service, reasoning toggle, benchmark config, validation, fair comparison, quantized variants.
- [references/slurm-run-and-recover.md](references/slurm-run-and-recover.md): submit, chains, inspection, the decision table, health and acceptance, fresh runs.
- [references/benchmark-notes.md](references/benchmark-notes.md): facts about single benchmarks that change how a result reads.
