# BunsenBench Chemistry MCQ Benchmark

BunsenBench Chemistry MCQ is the public Gym version of the BunsenBench chemistry MCQ benchmark.
The source manifest is hosted in the upstream Hugging Face dataset
[`nvidia/bunsen-bench`](https://huggingface.co/datasets/nvidia/bunsen-bench).
The `chemistry_mcq` config stores source locators, pinned Hugging Face revisions,
hashes, filter flags, and BCT labels without redistributing source question
text, choices, or answers.

## Prepare

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/bunsen_chem/config.yaml]"
```

Preparation loads the `chemistry_mcq` config from a pinned
`nvidia/bunsen-bench` commit, downloads that dataset's `tools/reconstitute.py`
helper, and uses it to fetch the pinned upstream sources and validate
source/canonical problem hashes from the manifest. Gym then writes the generated runnable JSONL to
`benchmarks/bunsen_chem/data/bunsenbench_chemistry_mcq_benchmark.jsonl`.
Generated JSONL under `data/` is gitignored; rerun preparation to recreate it.

Access to `nvidia/bunsen-bench` and GPQA-Diamond can require Hugging Face account
permissions. Set `HF_TOKEN` for an account with access to the upstream manifest
dataset and pinned source datasets.

## Sources

The v0.1.0 source mix is:

- MMLU-Redux high school chemistry
- MMLU-Redux college chemistry
- MMLU-Pro chemistry
- GPQA-Diamond chemistry
- SuperGPQA chemistry
- ChemBench MCQ rows from all nine public configs

The Gym implementation uses the upstream helper and the manifest's pinned
Hugging Face revisions. It fails loudly if a source row is missing, changed,
ambiguous, or inaccessible.

ChemBench rows with zero or multiple positive `target_scores` are excluded for
v0.1.0 because the public verifier uses a single expected answer letter.

## Licensing

Code: Apache 2.0

The benchmark mixes upstream datasets with different licenses. Consult each
source dataset for redistribution and evaluation terms:

| Source | Typical license |
|--------|-----------------|
| MMLU-Redux (chemistry) | MIT |
| MMLU-Pro (chemistry) | MIT |
| GPQA-Diamond (chemistry) | MIT |
| SuperGPQA (chemistry) | Apache 2.0 |
| ChemBench MCQ configs | MIT |

The benchmark config uses `license: TBD` until composite licensing is finalized.
