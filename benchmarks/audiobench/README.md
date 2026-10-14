# AudioBench

Port of the [AudioBench](https://github.com/AudioLLMs/AudioBench) suite from
[NeMo Skills](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/dataset/audiobench)
into NeMo Gym: **63 sub-datasets** in four benchmarks, one per scoring method.

| Benchmark | Sub-datasets | Verifier | Resources server |
|---|---:|---|---|
| `audiobench/judge` | 32 | LLM judge rates the answer 0–5 against the reference (prompt byte-equivalent with Skills' `judge/audiobench.yaml`) | [`audiobench_judge`](../../resources_servers/audiobench_judge/) |
| `audiobench/asr` | 21 | Whisper normalization + digits→words, corpus WER | [`asr_with_pc`](../../resources_servers/asr_with_pc/) (`task_type: ASR`, `normalization_mode: audiobench`) |
| `audiobench/bleu` | 6 | sacreBLEU (CoVoST 2 speech translation) | `asr_with_pc` (`task_type: BLEU`) |
| `audiobench/exact_match` | 4 | Lowercase, punctuation-stripped exact match (Spoken-MQA) | `asr_with_pc` (`task_type: EXACT_MATCH`) |

Each benchmark lives in its own directory (`judge/`, `asr/`, `bleu/`,
`exact_match/`) with a `config.yaml`, `manifest.yaml`, and `prepare.py`.
They share the sub-dataset registry [`DATASETS.py`](DATASETS.py), the
download and row mapping in [`_prepare_lib.py`](_prepare_lib.py), the prompt,
and the `data/` directory.

## Data provenance

[`DATASETS.py`](DATASETS.py) lists every sub-dataset's Hugging Face repo,
split, and data directory (the same mapping as Skills' `audiobench/prepare.py`),
and `HF_REVISIONS` pins each repo to a fixed commit. Most sources are
`AudioLLMs/*` test sets; the IMDA sub-datasets come from
[`MERaLiON/Multitask-National-Speech-Corpus-v1`](https://huggingface.co/datasets/MERaLiON/Multitask-National-Speech-Corpus-v1)
and Spoken-MQA from [`amao0o0/spoken-mqa`](https://huggingface.co/datasets/amao0o0/spoken-mqa).
Licenses vary per sub-dataset (CC-BY-4.0, CC-BY-NC-4.0, Apache-2.0, and the
IMDA National Speech Corpus license); check each source before use.

Every prepared row carries `subset_for_metrics` (the sub-dataset slug), so the
resources server reports per-sub-dataset metrics next to the benchmark-level
ones.

## Audio handling

`prepare.py` writes one WAV per row to `benchmarks/audiobench/data/audio/`
and references it from `responses_create_params.metadata.audio_path`. The
`vllm_model` adapter reads that file, base64-encodes it, and splices an
`audio_url` content block into the user message.

By default `audio_path` is the absolute path of the local `data/audio/`
directory. When the model server runs on another host or in a container, pass
`audio_prefix` with the directory where `data/audio/` is mounted there.

These benchmarks require an audio-capable vLLM endpoint. The resources
servers only accept vLLM-based model adapters (`--model-type vllm_model`);
set `--model-url` to the vLLM server.

## Prompt

[`prompts/default.yaml`](prompts/default.yaml) is a fixed system prompt plus
`user: "{instruction}"`. `instruction` is set per row: the sub-dataset's own
instruction from the source row, or a per-sub-dataset override in
`DATASETS.py` (ASR language cues, translation direction, Spoken-MQA answer
format).

## Prepare benchmark data

```bash
gym eval prepare --benchmark audiobench/judge
gym eval prepare --benchmark audiobench/asr
gym eval prepare --benchmark audiobench/bleu
gym eval prepare --benchmark audiobench/exact_match
```

`prepare()` accepts `datasets` (sub-dataset slugs from that benchmark),
`max_samples_per_dataset`, and `audio_prefix`:

```bash
gym eval prepare --benchmark audiobench/judge \
    '+prepare_script_args={datasets: [alpaca_audio_test, mmau_mini], max_samples_per_dataset: 5}'
```

The same options are available from the command line, e.g.
`python benchmarks/audiobench/judge/prepare.py --datasets alpaca_audio_test --max-samples-per-dataset 5`.

If a source repo is license-gated on Hugging Face, accept its license and run
`hf auth login`. A `license_gated` sub-dataset that is still inaccessible is
skipped with a warning; any other download failure stops the prepare.

## Judge model

`audiobench/judge` sends each answer to a separate judge model, by default
`openai/gpt-oss-20b` on `https://integrate.api.nvidia.com/v1` through
`/v1/chat/completions` (temperature 0, 512 output tokens). Set
`judge_api_key` in `env.yaml`, and `judge_base_url` / `judge_model_name` to use
a different judge. A failed judge call masks the row instead of scoring it as
wrong.

## Running servers

```bash
gym env start \
    --model-type vllm_model \
    --model-url http://<vllm-host>:<port>/v1 \
    --model <audio-capable-model> \
    --benchmark audiobench/judge
```

`--benchmark` is repeatable, so one server set can host all four benchmarks.

## Collecting rollouts

```bash
gym eval run --no-serve \
    --agent audiobench_judge_audiobench_judge_simple_agent \
    --output results/audiobench_judge_rollouts.jsonl
```

Agents for the other benchmarks are `audiobench_asr_asr_with_pc_simple_agent`,
`audiobench_bleu_asr_with_pc_simple_agent`, and
`audiobench_exact_match_asr_with_pc_simple_agent`. Each benchmark runs one
repeat per task (`num_repeats: 1`); pass `--num-repeats N` for pass@k.

## Verification

| Benchmark | Headline metrics | Per-sub-dataset |
|---|---|---|
| `audiobench/judge` | `judge_score` (mean rating × 20, 0–100); `accuracy` (rating ≥ 3) | `<slug>/pass@1[avg-of-k]/judge_score`, `.../accuracy` |
| `audiobench/asr` | corpus `wer` | `<slug>/pass@1[avg-of-k]/accuracy` (per-row WER < 0.5) |
| `audiobench/bleu` | corpus `bleu` (0–100) | `<slug>/pass@1[avg-of-k]/bleu` (mean sentence BLEU) |
| `audiobench/exact_match` | `accuracy` | `<slug>/pass@1[avg-of-k]/accuracy` |

The judge prompt, rating parser, and score formula match Skills'
`AudioMetrics`; ASR, BLEU, and exact-match scoring mirror Skills'
`audio_metrics.py`.
