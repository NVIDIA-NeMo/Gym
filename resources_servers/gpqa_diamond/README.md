# GPQA-Diamond Resource Server

## Overview

This resource server evaluates GPQA-Diamond multiple-choice responses using the
same verifier logic as `resources_servers/mcqa`.

- Task type: single-turn MCQ
- Domain: `knowledge`
- Answer format: final answer letter in `\boxed{X}`
- Grading mode: `strict_single_letter_boxed`

## Server Composition

Use GPQA-Diamond with:

- `responses_api_agents/simple_agent`
- `responses_api_models/*` (typically `policy_model`)
- `resources_servers/gpqa_diamond`

The server verifies the model response and returns reward `1.0` for exact
letter match against `expected_answer`, else `0.0`.

## Dataset Format

Each JSONL row follows the MCQA request schema:

- `responses_create_params.input[0].content`: user prompt containing question + options
- `options`: list of letter-to-text maps, e.g. `[{"A": "..."}, {"B": "..."}]`
- `expected_answer`: one of `A/B/C/D`
- `grading_mode`: `strict_single_letter_boxed`
- `metadata`: passthrough metadata (`explanation`, `subset_for_metrics`, `difficulty`)
- `uuid`: unique row id

See `data/example.jsonl` for concrete examples.

## Preprocessing Raw GPQA-Diamond

Full train/validation data is not stored in this repo, following the same
pattern as `resources_servers/mcqa`. Dataset locations are declared in
`configs/gpqa_diamond.yaml` via `gitlab_identifier` and
`huggingface_identifier`.

`dataset_preprocess.py` always downloads GPQA-Diamond raw data from HuggingFace,
stores the raw file, then writes the Gym-formatted train file into `data/`.

From the repository root:

```bash
python3 resources_servers/gpqa_diamond/dataset_preprocess.py
```

This generates:

- `resources_servers/gpqa_diamond/data/diamond_raw.jsonl`
- `resources_servers/gpqa_diamond/data/train.jsonl`

`data/example.jsonl` is a curated repo artifact and is not modified by the
preprocess script.

## Example Usage

```bash
config_paths="responses_api_agents/simple_agent/configs/simple_agent.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/gpqa_diamond/configs/gpqa_diamond.yaml"

ng_run "+config_paths=[$config_paths]" \
    +simple_agent.responses_api_agents.simple_agent.resources_server.name=gpqa_diamond

ng_collect_rollouts \
    +agent_name=simple_agent \
    +input_jsonl_fpath=resources_servers/gpqa_diamond/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/gpqa_diamond/data/example_rollouts.jsonl \
    +limit=3
```

`ng_collect_rollouts` also writes sidecar files next to `output_jsonl_fpath`, matching
the same pattern as `test_rollouts*`:

- `*_materialized_inputs.jsonl`
- `*_reward_profiling.jsonl`
- `*_agent_metrics.json`

## Licensing

Code: Apache 2.0
