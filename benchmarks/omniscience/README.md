# Omniscience Benchmark

Factual knowledge QA benchmark from the [AA-Omniscience v1.1 public set](https://huggingface.co/datasets/ArtificialAnalysis/AA-Omniscience-Public-v1.1).

## Overview

Tests factual recall across 6 domains: Humanities and Social Sciences, Health, Software Engineering, Science Engineering and Mathematics, Law, and Finance. Uses an LLM judge for verification (4-tier: CORRECT, INCORRECT, PARTIAL_ANSWER, NOT_ATTEMPTED).

## Dataset

AA-Omniscience v1.1 is the version Artificial Analysis reports. Its full set has 3,000 questions; the public set used here is 300 of them (60 each in Science Engineering and Mathematics and in Software Engineering, 45 in each of the other four domains). `prepare.py` pins revision `1f9cbcec` of the public set and writes the dataset version and revision into each row.

v1.1 removed saturated questions and all 600 questions of the v1 public set, added new questions, and fixed ambiguous ones. Its `question_id` values 1-300 name different questions than the same ids in v1. Scores are not comparable with v1, which earlier versions of this benchmark used (`ArtificialAnalysis/AA-Omniscience-Public`, 600 questions).

The answer prompt names the question's domain and subtopic, as in Artificial Analysis's answer prompt.

## Usage

```bash
# Prepare data
gym dataset collate +benchmark=omniscience

# Run benchmark
gym eval run --no-serve \
    --model-type vllm_model \
    --output results/omniscience_rollouts.jsonl \
    +benchmark=omniscience
```

## Key Metrics

- **reward (pass@k)**: Fraction of CORRECT verdicts
- **omniscience_index**: `(correct - incorrect) / total` — penalizes hallucination
- **is_hallucination**: Rate of INCORRECT (confident wrong) answers

## Requires

A judge model server (`genrm_model`) must be configured. The judge grades model responses against gold answers using the AA-Omniscience grading rubric. Artificial Analysis grades v1.1 with GPT-5.6 Luna (medium reasoning effort); scores are comparable with theirs only under the same grader.

The default judge is GPT-5.6 Luna at medium reasoning effort on `https://inference-api.nvidia.com/v1` (`benchmarks/omniscience/judge_luna.yaml`), with a 32,768-token judge output cap. Set `JUDGE_API_KEY`, or set `omniscience_judge_base_url`, `omniscience_judge_api_key` and `omniscience_judge_model_name` at the top level of your config to use another endpoint, key or model.
