# Nemotron 3.5 Super reward profiling

Reward-profiles the RL training blend by running the policy over every training dataset and
summarizing per-task reward with `gym eval profile`.

`manifest.yaml` covers the **26 no-judge, no-sandbox environments** (623,006 rows). The 10
environments that need an LLM judge or a sandbox are intentionally excluded; they are tracked
separately because they need extra infrastructure.

The sweep machinery itself is generic and lives in `nemo_gym/sweep/`. Only this manifest is
Nemotron-specific.

## Why one job instead of one job per environment

Rollout collection dispatches each row to the agent named in its own `agent_ref`
(`nemo_gym/rollout_collection.py`). Training rows already carry that field, so datasets for
different environments can be concatenated into a single input file and served by one Gym
deployment composed from the union of their configs.

## Validate

Checks that every config exists, declares the agent its entry names, and that the data's
`agent_ref` matches. Reads only the head of each data file, so it takes seconds.

```bash
python -m nemo_gym.sweep validate \
    benchmarks/nemotron_3.5_super/reward_profiling/manifest.yaml
```

A mismatch is an error, not a warning: pairing a dataset with the wrong config would score
rollouts with the wrong verifier and still look successful.

## Build

Concatenates every entry's rows into one input JSONL and writes the composed `config_paths`.

```bash
python -m nemo_gym.sweep build \
    benchmarks/nemotron_3.5_super/reward_profiling/manifest.yaml \
    --out-dir <work-dir> \
    --policy-base-url http://<router-ip>:8000/v1 \
    --policy-model-name <checkpoint-path> \
    --output-jsonl <work-dir>/rollouts.jsonl
```

Outputs `input.jsonl`, `sweep_config.yaml`, `build_report.json`, and prints the `gym eval run`
command. Use `--limit-per-entry N` for a smoke run.

## Run

Against an existing vLLM endpoint (see `benchmarks/nemotron_3.5_super/sbatch_external_vllm.sh`),
using the command `build` printed. Then:

```bash
gym eval profile \
    --inputs <work-dir>/rollouts_materialized_inputs.jsonl \
    --rollouts <work-dir>/rollouts.jsonl
```

## Notes

- `--no-serve` is required. Without it `--input` is silently replaced by the collated split.
- `num_repeats` resolves per agent, so entries sharing an agent share a repeat count. `validate`
  reports which entries those are. Set a per-entry `num_repeats` to override the global default.
- Set `num_samples_in_parallel` explicitly; unset means unbounded concurrency.
- Pass `--resume` and a stable output path. Rollout collection clears the output file otherwise,
  and a sweep of this size will not finish inside one `batch` allocation.
- `agent_ref_override` rewrites `agent_ref` while concatenating. Use it only to deliberately run a
  dataset through a different agent; the override is recorded in `build_report.json`.
