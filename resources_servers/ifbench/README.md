# IFBench Resources Server

## Overview

This resources server evaluates instruction following using AllenAI's [IFBench](https://github.com/allenai/IFBench) library. It checks whether a model response satisfies the constraints embedded in each prompt, covering 57 distinct instruction types.

- Task type: single-turn open-ended instruction following
- Domain: `instruction_following`
- Grading mode: `fraction` — reward is the fraction of instructions followed per response

## Server Composition

Use IFBench with:

- `responses_api_agents/simple_agent`
- `responses_api_models/*` (typically `policy_model`)
- `resources_servers/ifbench`

The server evaluates each instruction in `instruction_id_list` against the model response and returns a reward between 0.0 and 1.0.

## Dataset Format

Each JSONL row:

- `id`: row index
- `instruction_id_list`: list of IFBench instruction IDs (e.g. `["count:numbers", "words:start_verb"]`)
- `prompt`: the full user-facing prompt with constraints embedded in plain English
- `kwargs`: list of parameter dicts, one per instruction (unused keys are `null`)
- `grading_mode`: `"fraction"` (must be present in every row)

See `data/example.jsonl` for concrete examples.

## IFBench Library Setup

The server clones the AllenAI IFBench repo at a pinned commit on first startup and adds it to `sys.path`. Python dependencies (spaCy, nltk, syllapy, etc.) are installed via `requirements.txt`.

If the server runs inside a container that restricts outbound network access, the clone will fail. In that case, run `setup_ifbench.py` outside the container first:

```bash
python resources_servers/ifbench/setup_ifbench.py
```

This clones the repo, applies patches, and writes the `.installed` marker. On subsequent startups the server skips the clone and uses the pre-cloned copy.

## Example Usage

```bash
config_paths="benchmarks/ifbench/config.yaml,responses_api_models/openai_model/configs/openai_model.yaml"

ng_run "+config_paths=[$config_paths]"

ng_collect_rollouts \
    +agent_name=ifbench_benchmark_simple_agent \
    +input_jsonl_fpath=resources_servers/ifbench/data/example.jsonl \
    +prompt_config=benchmarks/ifbench/prompts/default.yaml \
    +output_jsonl_fpath=/tmp/ifbench_rollouts.jsonl \
    +num_repeats=1 \
    "+responses_create_params={max_output_tokens: 4096, temperature: 0.0}"
```

## Licensing

Code: Apache 2.0
IFBench library: Apache 2.0 (AllenAI)
