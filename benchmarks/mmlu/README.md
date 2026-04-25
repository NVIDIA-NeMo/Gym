# MMLU

[Massive Multitask Language Understanding (MMLU)](https://arxiv.org/abs/2009.03300) is a 57-subject multiple-choice benchmark (four options A–D).

## Data preparation

This matches **NeMo-Skills** ``nemo_skills/dataset/mmlu/prepare.py``:

- Download **Hendrycks** ``data.tar`` from ``https://people.eecs.berkeley.edu/~hendrycks/data.tar``
- Read per-subject CSVs under ``data/{split}/`` with columns ``question``, ``A``, ``B``, ``C``, ``D``, ``expected_answer``
- Use the same **subcategories** map for ``subset_for_metrics``

The Gym benchmark JSONL adds ``options_text`` / ``options`` / ``uuid`` for ``mcqa`` + ``benchmarks/mmlu/prompts/default.yaml``. Default split for ``ng_prepare_benchmark`` is **test** (``benchmarks/mmlu/data/mmlu_benchmark.jsonl``).

## Usage

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/mmlu/config.yaml]"

# Same flags as NeMo-Skills (writes ``mmlu_dev.jsonl`` / ``mmlu_val.jsonl`` for non-test splits).
python benchmarks/mmlu/prepare.py --split test

ng_run "+config_paths=[benchmarks/mmlu/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"

ng_collect_rollouts \
  "+config_paths=[benchmarks/mmlu/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
  +output_jsonl_fpath=results/mmlu.jsonl
```
