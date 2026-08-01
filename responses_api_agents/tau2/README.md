# Description
```bash
gym env start \
    --config benchmarks/tau2/configs/tau2.yaml \
    --model-type openai_model \
    ++nemo_gym_log_dir=results/tau2 \
    '++gpt-5_2-2025-12-11.responses_api_models.openai_model.openai_api_key=${openai_api_key}' \
    '++gpt-5_2-2025-12-11.responses_api_models.openai_model.extra_body._delete_key=max_output_tokens'
```

## Tau3 banking_knowledge

The banking_knowledge configs use the Tau2 bridge with Tau3 data and retrieval
configuration in the prepared rows:

```bash
gym env start \
    --config benchmarks/tau2/configs/banking_bm25_grep_artificial_analysis.yaml \
    --model-type openai_model \
    ++nemo_gym_log_dir=results/tau2_banking_bm25_grep_artificial_analysis \
    '++gpt-5_4-mini-2026-03-17.responses_api_models.openai_model.openai_api_key=${openai_api_key}'
```

```bash
gym env start \
    --config benchmarks/tau2/configs/banking_terminal_use.yaml \
    --model-type openai_model \
    ++nemo_gym_log_dir=results/tau2_banking_terminal_use \
    '++gpt-5_2-2025-12-11.responses_api_models.openai_model.openai_api_key=${openai_api_key}'
```

```bash
gym env start \
    --config benchmarks/tau2/configs/banking_alltools.yaml \
    --model-type openai_model \
    ++nemo_gym_log_dir=results/tau2_banking_alltools \
    '++gpt-5_2-2025-12-11.responses_api_models.openai_model.openai_api_key=${openai_api_key}'
```

`banking_bm25_grep_artificial_analysis` uses
`gpt-5.4-mini-2026-03-17` with medium reasoning as the user simulator and runs
five repeats over all 97 tasks. BM25+grep retrieval itself is offline and needs
no sandbox tooling or retrieval API key. `terminal_use` requires local sandbox
tooling: `srt`, `rg`, `bwrap`, and `socat`. `alltools` uses the same sandbox
tooling and also requires `OPENAI_API_KEY` for dense retrieval at rollout time.
You can check a mode with:

```bash
python -m benchmarks.tau2.prepare_utils.runtime bm25_grep
python -m benchmarks.tau2.prepare_utils.runtime terminal_use
python -m benchmarks.tau2.prepare_utils.runtime alltools
```

By default the bridge installs runtime Tau from
`https://github.com/bxyu-nvidia/tau2-bench@bxyu/nemo_gym_stable`
and fetches the Tau `data/` tree from the same ref. Override with
`NEMO_GYM_TAU2_BENCH_REPO_URL` and `NEMO_GYM_TAU2_BENCH_REF` when testing a
different PR branch or commit.

Benchmark prepare uses a separate pinned data-generation branch,
`https://github.com/bxyu-nvidia/tau2-bench@bxyu/nemo_gym_data`.
That branch owns `dump_nemo_gym_data.sh`; Gym clones it, runs the dump script,
and reads the resulting `nemo_gym_data` JSON files. Override that source with
`NEMO_GYM_TAU2_BENCH_DATA_REPO_URL` and `NEMO_GYM_TAU2_BENCH_DATA_REF`.

The generic banking prepare script can also materialize every pinned
banking_knowledge retrieval config:

```bash
python benchmarks/tau2/prepare.py banking_knowledge --all
```

See `benchmarks/tau2/README.md` for the benchmark config and prepare layout.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
- tau2: MIT
