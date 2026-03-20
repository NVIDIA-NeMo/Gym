# GPQA-Diamond Resources Server

## Overview

This resources server evaluates GPQA-Diamond multiple-choice responses with a
GPQA-specific verifier built on top of `resources_servers/mcqa`.

- Task type: single-turn MCQ
- Domain: `knowledge`
- Dataset prompt format: final line `Answer: LETTER`
- Also accepted by the verifier: `\boxed{X}` and custom regex extraction via
  `template_metadata.output_regex`
- Grading mode: `strict_single_letter_boxed`

## Server Composition

Use GPQA-Diamond with:

- `responses_api_agents/simple_agent`
- `responses_api_models/*` (typically `policy_model`)
- `resources_servers/gpqa_diamond`

The server verifies the model response and returns reward `1.0` for exact
letter match against `expected_answer`, else `0.0`.

Answer extraction priority is:

1. `template_metadata.output_regex` if present
2. GPQA fallback parsing of the final answer from `\boxed{...}` or `Answer: X`

## Dataset Format

Each JSONL row follows the MCQA request schema:

- `responses_create_params.input[0].content`: user prompt containing question + options
- `options`: list of letter-to-text maps, e.g. `[{"A": "..."}, {"B": "..."}]`
- `expected_answer`: one of `A/B/C/D`
- `grading_mode`: `strict_single_letter_boxed`
- `template_metadata`: optional per-row metadata, including optional `output_regex`
- `metadata`: passthrough metadata (`explanation`, `subset_for_metrics`, `difficulty`)
- `uuid`: unique row id

See `data/example.jsonl` for concrete examples.

Notes:

- The generated GPQA prompt asks the model to finish with `Answer: LETTER`.
- Although dataset rows keep `grading_mode: strict_single_letter_boxed` for
  compatibility with the shared MCQA schema, the GPQA server's custom fallback
  parser accepts both `Answer: X` and boxed final answers.

## Preprocessing Raw GPQA-Diamond

Full train data is not stored in this repo.

`dataset_preprocess.py` downloads the GPQA Diamond split from
`Idavidrein/gpqa` on HuggingFace, writes a normalized raw dump, and then writes
the Gym-formatted training set into `data/`.

From the repository root:

```bash
python3 resources_servers/gpqa_diamond/dataset_preprocess.py
```

This generates:

- `resources_servers/gpqa_diamond/data/diamond_raw.jsonl`
- `resources_servers/gpqa_diamond/data/train.jsonl`

`data/example.jsonl` is a curated repo artifact and is not modified by the
preprocess script. There is currently no `validation.jsonl` for this resources
server.

## Example Usage

Using a local Nemotron 3 model with `local_vllm_model`:

```bash
config_paths="responses_api_agents/simple_agent/configs/simple_agent.yaml,responses_api_models/local_vllm_model/configs/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml,resources_servers/gpqa_diamond/configs/gpqa_diamond.yaml"
ng_run "+config_paths=[${config_paths}]" \
    '++policy_model=${inherit_from:NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}' \
    +simple_agent.responses_api_agents.simple_agent.resources_server.name=gpqa_diamond \
    "++NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.responses_api_models.local_vllm_model.vllm_serve_kwargs.mamba_ssm_cache_dtype=float32" \
    "++NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.responses_api_models.local_vllm_model.vllm_serve_kwargs.enable_prefix_caching=False"
```

Generic example with `openai_model`:

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
Configured train dataset license metadata: MIT

