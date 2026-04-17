# Omniscience Benchmark

Factual knowledge QA benchmark from [AA-Omniscience-Public](https://huggingface.co/datasets/ArtificialAnalysis/AA-Omniscience-Public).

## Overview

Tests factual recall across 6 domains: Humanities, Health, Software Engineering, STEM, Law, and Finance. Uses an LLM judge for verification (4-tier: CORRECT, INCORRECT, PARTIAL_ANSWER, NOT_ATTEMPTED).

## Usage

```bash
# Prepare data
uv run ng_prepare_data +benchmark=omniscience

# Run benchmark
uv run ng_collect_rollouts +benchmark=omniscience \
    "+config_paths=[responses_api_models/vllm_model/configs/vllm_model.yaml]" \
    +output_jsonl_fpath=results/omniscience_rollouts.jsonl
```

## Key Metrics

- **reward (pass@k)**: Fraction of CORRECT verdicts
- **omniscience_index**: `(correct - incorrect) / total` — penalizes hallucination
- **is_hallucination**: Rate of INCORRECT (confident wrong) answers

## Requires

A judge model server (`genrm_model`) must be configured. The judge grades model responses against gold answers using the AA-Omniscience grading rubric.
