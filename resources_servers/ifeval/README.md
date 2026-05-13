# IFEval Resources Server

Port of NeMo-RLVR's `nemo_rl/environments/ifeval_environment.py` to NeMo-Gym.

### Overview

Verifies model responses against IFEval-style instruction constraints using the
25-function taxonomy from `if_functions.py` (`IF_FUNCTIONS_MAP`). Each constraint
is a dict with a `func_name` and the kwargs that function accepts; reward is 1.0
iff *all* constraints are satisfied (binary grading), 0.0 otherwise.

This is a separate environment from `instruction_following`, which uses the
`verifiable-instructions` registry with a different schema (`instruction_id_list`
+ `kwargs`). Use `ifeval` when porting datasets that already encode constraints
in the RLVR `func_name` format; use `instruction_following` for the IFEval/IFBench
dataset format.

### Input schema

`verifier_metadata`:
- `ground_truth` (required): constraint spec. May be:
  - a JSON-encoded string,
  - a single dict `{"func_name": ..., ...}`,
  - or a list of such dicts.

A `<think>...</think>` prefix in the model's output is stripped before
verification. An unclosed `<think>` (no `</think>`) yields reward 0.0 without
flagging a verifier failure.

### Example dataset row

`ground_truth` lives at the **top level** of each JSONL row (not nested under
`verifier_metadata`). The agent forwards top-level row fields into the verify
request body alongside `responses_create_params`/`response`.

```json
{
  "ground_truth": [{"func_name": "validate_no_commas"}],
  "responses_create_params": {
    "input": [{"role": "user", "content": "Write a poem about autumn. Your response should not contain any commas."}],
    "tools": [],
    "parallel_tool_calls": false
  }
}
```

### Available constraint functions

See `if_functions.py`. Highlights:

| `func_name` | kwargs |
|---|---|
| `verify_keywords` | `keyword_list: list[str]` |
| `verify_keyword_frequency` | `word: str, N: int, quantifier: str?` |
| `validate_forbidden_words` | `forbidden_words: list[str]` |
| `verify_letter_frequency` | `letter: str, N: int, quantifier: str?` |
| `validate_response_language` | `language: str` |
| `verify_paragraph_count` | `N: int` |
| `validate_word_constraint` | `N: int, quantifier: str` |
| `verify_sentence_constraint` | `N: int, quantifier: str` |
| `validate_paragraphs` | `N: int, first_word: str, i: int` |
| `verify_postscript` | `postscript_marker: str` |
| `validate_placeholders` | `N: int` |
| `verify_bullet_points` | `N: int` |
| `validate_title` | — |
| `validate_choice` | `options: list` |
| `validate_highlighted_sections` | `N: int` |
| `validate_sections` | `N: int, section_splitter: str` |
| `validate_json_format` | — |
| `validate_repeat_prompt` | `original_prompt: str` |
| `validate_two_responses` | — |
| `validate_uppercase` / `validate_lowercase` | — |
| `validate_frequency_capital_words` | `N: int, quantifier: str` |
| `validate_end` | `end_phrase: str` |
| `validate_quotation` | — |
| `validate_no_commas` | — |

### Usage

```bash
config_paths="responses_api_agents/simple_agent/configs/simple_agent.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/ifeval/configs/ifeval.yaml"
ng_run "+config_paths=[$config_paths]" \
  "+simple_agent.responses_api_agents.simple_agent.resources_server.name=ifeval"

ng_collect_rollouts \
    +agent_name=simple_agent \
    +input_jsonl_fpath=resources_servers/ifeval/data/example.jsonl \
    +output_jsonl_fpath=results/ifeval_rollouts.jsonl \
    +num_repeats=5
```

### Testing

```bash
ng_test +entrypoint=resources_servers/ifeval/
```

### Licensing

Code: Apache 2.0. `if_functions.py` is reproduced from NeMo-RLVR (Apache 2.0).
