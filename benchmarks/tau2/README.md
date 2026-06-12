# Tau2 / Tau3 Banking Benchmarks

This benchmark directory uses one public prepare entrypoint:

```bash
python benchmarks/tau2/prepare.py
```

With no arguments it prepares the original Tau2 benchmark domains: `airline`,
`retail`, and `telecom`.

Tau3 `banking_knowledge` is opt-in through prepare arguments:

```bash
python benchmarks/tau2/prepare.py banking_knowledge --retrieval-config terminal_use
python benchmarks/tau2/prepare.py banking_knowledge --retrieval-config alltools
python benchmarks/tau2/prepare.py banking_knowledge --all
```

Gym config files live under `benchmarks/tau2/configs/`:

- `tau2.yaml`: original Tau2 `airline`, `retail`, and `telecom`.
- `banking_terminal_use.yaml`: Tau3 `banking_knowledge` with `terminal_use`.
- `banking_alltools.yaml`: Tau3 `banking_knowledge` with `alltools`.

The banking configs use the same `prepare.py` as Tau2 and opt into Tau3 with
dataset-level `prepare_args`:

```yaml
prepare_script: benchmarks/tau2/prepare.py
prepare_args:
  dataset: banking_knowledge
  retrieval_config: terminal_use
```

Prepare implementation details are under `benchmarks/tau2/prepare_utils/`.
The generated JSONL files are written to `benchmarks/tau2/data/` and are
ignored by git.

## Pinned Tau Data Source

Benchmark preparation clones the pinned Tau data-generation branch:

```text
https://github.com/bxyu-nvidia/tau2-bench@jk/bxyu-nemo-gym-data-upstream-main-tau3
```

That branch owns `dump_nemo_gym_data.sh`; Gym runs it and then reads the
generated `nemo_gym_data` JSON files. Override the source for local testing
with:

```bash
NEMO_GYM_TAU2_BENCH_DATA_REPO_URL=/path/to/tau2-bench \
NEMO_GYM_TAU2_BENCH_DATA_REF=jk/bxyu-nemo-gym-data-upstream-main-tau3 \
python benchmarks/tau2/prepare.py banking_knowledge --retrieval-config terminal_use
```

The new data branch keeps the original `bxyu/nemo_gym_data` interface: a
`dump_nemo_gym_data.sh` script emits `nemo_gym_data/<dataset>/*.json` rows with
Tau config, task, system prompt, and Responses-style tool schemas. Internally it
uses Tau's build layer directly instead of patching `run_single_task`, which
keeps the branch smaller while preserving the generated row contract.

## Runtime Checks

`terminal_use` and `alltools` need sandbox tooling at rollout time. `alltools`
also needs `OPENAI_API_KEY` for dense retrieval. Check a retrieval config with:

```bash
python -m benchmarks.tau2.prepare_utils.runtime terminal_use
python -m benchmarks.tau2.prepare_utils.runtime alltools
```
