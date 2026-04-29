# SimpleQA Benchmark

[SimpleQA-Verified](https://huggingface.co/datasets/codelion/SimpleQA-Verified)
short-form factual QA. The dataset is the verified subset of OpenAI's
[SimpleQA](https://openai.com/index/introducing-simpleqa/), a 4,326-question
benchmark designed to measure factual recall and abstention calibration.

## Overview

Tests the model's ability to:

- **recall** rare/short factual answers (people, dates, numeric identifiers,
  obscure technical references), and
- **abstain** rather than guess when it doesn't know — the headline metric
  rewards CORRECT and penalizes INCORRECT but treats NOT_ATTEMPTED as a
  neutral outcome.

Verification uses an LLM judge that grades on a 3-tier scale (CORRECT,
INCORRECT, NOT_ATTEMPTED). See
`resources_servers/simpleqa/README.md` for the verification details.

## Headline metric

F1 = 2·P·R / (P + R), where

- P = correct / (correct + incorrect) (= `accuracy_given_attempted`)
- R = correct / total

This is the SimpleQA-Verified scoring convention from
https://www.kaggle.com/code/nanliao7/simpleqa-verified-benchmark-starter-code.
The server emits per-tier `pass@k` (`correct`, `incorrect`, `not_attempted`)
and derives `f1` + `accuracy_given_attempted` at every aggregation level.

## Prepare benchmark data

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/simpleqa/config.yaml]"
```

## Running servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/simpleqa/config.yaml"
ng_run "+config_paths=[$config_paths]"
```

## Collecting rollouts

```bash
ng_collect_rollouts \
    +agent_name=simpleqa_simpleqa_simple_agent \
    +input_jsonl_fpath=benchmarks/simpleqa/data/simpleqa_benchmark.jsonl \
    +output_jsonl_fpath=results/simpleqa_rollouts.jsonl \
    +prompt_config=benchmarks/simpleqa/prompts/default.yaml \
    +num_repeats=4
```

For reasoning models, start the policy vLLM server with
`--reasoning-parser <name>` (e.g. `deepseek_r1` for Nemotron-3 Nano) so
`<think>...</think>` is stripped before grading.
