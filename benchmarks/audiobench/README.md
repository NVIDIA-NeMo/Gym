# AudioBench

Port of the [AudioBench](https://github.com/AudioLLMs/AudioBench) benchmark
suite from [NeMo Skills](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/dataset/audiobench)
into NeMo Gym. **63 sub-datasets** across four scoring buckets:

| Bucket | Count | Verifier | Resource server | Agent |
|---|---:|---|---|---|
| Judge (open-ended LLM-graded) | 32 | 0–5 LLM rating prompt, byte-equivalent with Skills' `judge/audiobench.yaml` | [`audiobench_judge`](../../resources_servers/audiobench_judge/) | `audiobench_judge_simple_agent` |
| ASR | 21 | Whisper + audiobench-style WER (digits→words) | [`asr_with_pc`](../../resources_servers/asr_with_pc/) (`task_type=ASR`, `normalization_mode=audiobench`) | `audiobench_asr_simple_agent` |
| BLEU translation | 6 | sacrebleu sentence BLEU | `asr_with_pc` (`task_type=BLEU`) | `audiobench_bleu_simple_agent` |
| Exact-match (spoken-mqa) | 4 | lowercase + punct-strip + whitespace-collapse | `asr_with_pc` (`task_type=EXACT_MATCH`) | `audiobench_exact_match_simple_agent` |

Source-of-truth registry (HF repo + split + bucket + license + instruction
override per sub-dataset): [`DATASETS.py`](DATASETS.py).

## How the bucketing works

Each sub-dataset belongs to exactly one bucket. The bucket determines:
* which `agent_ref` is stamped on every row in `prepare_<bucket>.py`,
* which prepared JSONL the row lands in (`data/audiobench_<bucket>.jsonl`),
* which agent's resource server runs `verify()` on the rollout.

`ng_collect_rollouts` reads the agent name from each row's `agent_ref` and
dispatches accordingly, so a single rollout collection can mix sub-datasets
from one bucket; the per-row `dataset_name` field is then used by the
resource server's `compute_metrics()` (via `compute_subset_metrics`) to
break results down per sub-dataset inside each agent's metric block.

## Preparation

Download all 63 sub-datasets at once via the standard pipeline (one job
per bucket):

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/audiobench/config.yaml]"
```

Or per-bucket directly:

```bash
python benchmarks/audiobench/prepare_judge.py            # all 32 judge datasets
python benchmarks/audiobench/prepare_asr.py              # all 21 ASR datasets
python benchmarks/audiobench/prepare_bleu.py             # all 6 covost2
python benchmarks/audiobench/prepare_exact_match.py      # all 4 spoken-mqa
```

Each per-bucket script accepts `--datasets foo,bar,baz` and
`--max-samples-per-dataset N` for selective preparation:

```bash
# only prepare a smoke set of the judge bucket
python benchmarks/audiobench/prepare_judge.py \
    --datasets alpaca_audio_test,audiocaps_test,mmau_mini --max-samples-per-dataset 5
```

License-gated sub-datasets (the 18 IMDA splits hosted on
[`MERaLiON/Multitask-National-Speech-Corpus-v1`](https://huggingface.co/datasets/MERaLiON/Multitask-National-Speech-Corpus-v1))
will fail to download until the user accepts the license on the HF page
and authenticates with `huggingface-cli login`. The prepare script
**skips** any sub-dataset that raises `GatedRepoError` (or any other HF
loading error) and prints a per-skipped warning at the end — the rest of
the bucket still lands.

## Running

Start the four servers (audiobench_judge + three asr_with_pc variants +
the model under test):

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/audiobench/config.yaml"
ng_run "+config_paths=[$config_paths]"
```

Collect rollouts for one bucket at a time (each writes its own metric
block in `rollouts_aggregate_metrics.json`):

```bash
ng_collect_rollouts \
    +agent_name=audiobench_judge_simple_agent \
    +input_jsonl_fpath=benchmarks/audiobench/data/audiobench_judge.jsonl \
    +output_jsonl_fpath=results/audiobench_judge_rollouts.jsonl \
    +num_repeats=4
```

Repeat with `audiobench_asr_simple_agent`, `audiobench_bleu_simple_agent`,
`audiobench_exact_match_simple_agent`. Or concatenate the four bucket
JSONLs into a single mixed input — `ng_collect_rollouts` reads each row's
`agent_ref` and dispatches to the right server, then groups results by
agent at aggregation time.

## Verification

| Metric | Bucket | Source |
|---|---|---|
| `judge_score` (mean rating × 20, 0–100) and `pass@k` (rating ≥ 3) | judge | LLM judge call, prompt byte-equivalent with Skills `judge/audiobench.yaml` |
| corpus `wer` + per-sub-dataset breakdowns | asr | `jiwer` over Whisper-normalized + digits→words text |
| sentence `bleu` (0..1) and per-sub-dataset breakdowns | bleu | `sacrebleu` |
| `pass@k/accuracy` and per-sub-dataset breakdowns | exact_match | string equality after lowercase + punctuation strip |

Per-sub-dataset breakdowns are emitted by each resource server's
`compute_metrics()` via `compute_subset_metrics(subset_key="dataset_name")`,
so e.g. the judge agent's metric block contains both the cross-dataset
headline and `alpaca_audio_test/pass@1[avg-of-1]/accuracy`,
`audiocaps_test/pass@1[avg-of-1]/accuracy`, … keys.

## Skills↔Gym structural parity

| Aspect | Skills | This port |
|---|---|---|
| Source files for the entire AudioBench port | 5 (dispatcher prepare.py + 2 template `__init__.py` + parent `__init__.py` + judge YAML) | ~10 (one benchmark dir with 4 thin prepare scripts + shared lib + DATASETS registry) |
| Sub-dataset selection at prepare time | `ns prepare-data audiobench --datasets X,Y,Z` | `python prepare_<bucket>.py --datasets X,Y,Z` |
| Per-sub-dataset directory creation at runtime | yes (template `__init__.py` copied per sub-dataset) | no — sub-dataset is a row-level `dataset_name` field |
| Mixed-sub-dataset rollout collection | one `ns eval` per benchmark group | one `ng_collect_rollouts` per bucket; mixed agents supported via per-row `agent_ref` |
| Per-sub-dataset metric breakdown | computed inside each `*Metrics.evaluate` | `compute_subset_metrics(subset_key="dataset_name")` inside each server's `compute_metrics()` |

The Gym port is a strict superset of Skills' coverage: same 63 sub-datasets,
same scoring logic (judge prompt + threshold + score formula byte-equivalent;
ASR / BLEU / exact-match implementations mirror Skills' `audio_metrics.py`),
plus per-sub-dataset breakdowns inside each agent block instead of one big
aggregate.
