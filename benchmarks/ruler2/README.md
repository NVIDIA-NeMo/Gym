# RULER v2 (`ruler2`) benchmark

Long-context benchmark suite from
[NVIDIA-NeMo/Skills/dataset/ruler2](https://github.com/NVIDIA-NeMo/Skills/tree/main/nemo_skills/dataset/ruler2).

Twelve sub-tasks across three families:

| Family | Sub-tasks | Source data |
| --- | --- | --- |
| `mk_niah_*` (multi-key needle-in-haystack) | basic, easy, medium, hard | synthetic + `cais/mmlu` |
| `mv_niah_*` (multi-value NIAH) | basic, easy, medium, hard | synthetic + `cais/mmlu` |
| `qa_*` (long-doc QA) | basic, easy, medium, hard | `hotpotqa/hotpot_qa` |

Each sub-task is generated synthetically with tokenizer-aware length
control: the generator binary-searches the haystack/document count so
total tokens (using the model's tokenizer) stay under a configurable
`max_seq_length`.

Per-sub-task verifier routing — wired into the JSONL by `prepare.py` so
the [`ruler2` resources server](../../resources_servers/ruler2/README.md)
verifies each row correctly:

| Sub-task | `eval_type` | `match_type` |
| --- | --- | --- |
| `mk_niah_basic`, `mk_niah_easy` | `ruler2` | `all` |
| `mk_niah_medium`, `mk_niah_hard` | `multichoice` | — |
| `mv_niah_basic`, `mv_niah_easy`, `mv_niah_hard` | `ruler2` | `all` |
| `mv_niah_medium` | `ruler2` | `2steps` |
| `qa_basic`, `qa_easy`, `qa_medium`, `qa_hard` | `ruler2` | `part` |

## Prepare benchmark data

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/ruler2/config.yaml]"
```

Knobs (set via env or pass to `prepare.py` directly):

| Variable | Default | Meaning |
| --- | --- | --- |
| `RULER2_MODEL` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | Tokenizer for length control |
| `RULER2_MAX_SEQ_LENGTH` | `16384` | Total token budget per sample |
| `RULER2_DATASET_SIZE` | `100` | Per-sub-task sample count |
| `RULER2_TASKS` | (all 12) | Comma-separated subset of tasks |

The first run installs `wonderwords`, `nltk`, `inflect`, `transformers`,
`tiktoken`, `tenacity`, `datasets`, `tqdm`, and `editdistance` into a
private `.venv/` at `benchmarks/ruler2/`. Subsequent runs reuse it.

## Start environment

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/ruler2/config.yaml"
ng_run "+config_paths=[$config_paths]"
```

## Collect rollouts

```bash
ng_collect_rollouts \
    +agent_name=ruler2_benchmark_simple_agent \
    +input_jsonl_fpath=benchmarks/ruler2/data/ruler2_benchmark.jsonl \
    +output_jsonl_fpath=results/ruler2_rollouts.jsonl \
    +num_repeats=4
```

## Metrics

`compute_metrics()` emits:

- Overall `pass@k/accuracy` and `pass@1[avg-of-k]/accuracy` over the
  whole 12-task suite.
- Per-sub-task breakdown like `mk_niah_basic/pass@1[avg-of-4]/accuracy`,
  `qa_hard/pass@4/accuracy`, etc. via `compute_subset_metrics(subset_key="task")`.
- A Tier-3 composite suite mean: arithmetic mean of the per-sub-task
  accuracy across all 12 sub-tasks, emitted as
  `ruler2_suite_avg/pass@1[avg-of-k]/accuracy` (only when all 12
  sub-tasks are present in the run). Mirrors the
  [`ruler2_score.compute_score`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/ruler2/ruler2_score.py)
  aggregation in NeMo Skills.
