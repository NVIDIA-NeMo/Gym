# BunsenBench Chemistry MCQ Resources Server

## Overview

BunsenBench Chemistry MCQ verifies chemistry multiple-choice outputs for the public BunsenBench Gym benchmark. It stores materialized answer letters internally, accepts exact choice-text answers, and passes source/taxonomy metadata through to aggregate metrics.

- Task type: single-turn MCQ
- Domain: `knowledge`
- Prompt format: XML `<choice>` exact-match (see [benchmarks/bunsenbench_chemistry_mcq/prompts/default.yaml](../../benchmarks/bunsenbench_chemistry_mcq/prompts/default.yaml))

The examples in `data/example.jsonl` are synthetic. They are not redistributed benchmark-source questions.

## Server Composition

Use BunsenBench Chemistry MCQ with:

- `responses_api_agents/simple_agent` via `bunsenbench_chemistry_mcq_simple_agent`
- `responses_api_models/*` (typically `policy_model` or `openai_model`)
- `resources_servers/bunsenbench_chemistry_mcq` (config key: `bunsenbench_chemistry_mcq`)

The server verifies the model response and returns reward `1.0` when the extracted answer letter matches `expected_answer`, else `0.0`.

## Dataset Format

### Committed smoke test (`data/example.jsonl`)

Each row includes:

- `responses_create_params.input[0].content`: user prompt with question, XML choices, and answer instructions
- `options`: list of letter-to-text maps, e.g. `[{"A": "H2O"}, {"B": "CO2"}]`
- `expected_answer`: gold letter after deterministic option shuffle
- `uuid`: stable row id (e.g. `bunsen:example:1`)
- `metadata`: upstream version tag, source, BCT labels, and upstream locator fields
- `agent_ref`: `{"type": "responses_api_agents", "name": "bunsenbench_chemistry_mcq_simple_agent"}`

Regenerate with:

```bash
python resources_servers/bunsenbench_chemistry_mcq/create_examples.py
```

### Full benchmark eval set

The full public benchmark is prepared under [benchmarks/bunsenbench_chemistry_mcq/](../../benchmarks/bunsenbench_chemistry_mcq/). See [benchmarks/bunsenbench_chemistry_mcq/README.md](../../benchmarks/bunsenbench_chemistry_mcq/README.md) for `ng_prepare_benchmark` and upstream access requirements.

Benchmark rows are materialized without `responses_create_params.input`; the prompt template is applied at rollout time from `question` and `options_text`.

## Input Schema

Required fields:

- `responses_create_params`: OpenAI Responses create params (required for committed examples; optional pre-prompt for benchmark rows).
- `expected_answer`: gold answer letter, such as `A` or `D`.

Supported choice fields:

- `options`: MCQA-style list of single-key dicts, such as `[{"A": "H2O"}, {"B": "CO2"}]`.
- `choices`: list of choice texts. Letters are assigned by position (`A`, `B`, `C`, ...).
- `choices`: list of dicts with `letter`/`label` and `text`/`content` keys.

Source/taxonomy fields may be top-level or nested under `metadata`:

- `source`
- `bct_field`
- `bct_subfield`

### Grading modes and MCQA fields

BunsenBench Chemistry MCQ uses **custom deterministic extraction** in `BunsenChemResourcesServer.verify()`. It does **not** use shared MCQA `grading_mode` or `template_metadata.output_regex` behavior from `resources_servers/mcqa`.

## Answer Extraction

Extraction is deterministic and supports:

- `Answer: A`
- `The answer is A`
- `\boxed{A}` and `\boxed{\text{A}}`
- `<answer>A</answer>`, `<choice>CO2</choice>`, or `<response>A</response>`
- Exact choice text on the final answer line

Exact choice-text matching normalizes common chemistry Unicode variants: subscripts, superscripts, plus/minus variants, multiplication signs, middle dots, micro signs, and compatibility characters.

## Aggregate Metrics

The standard `/aggregate_metrics` framework hook calls `BunsenChemResourcesServer.compute_metrics()`. This server emits overall pass/majority/no-answer metrics plus grouped metrics:

- `by_source/<source>/...`
- `by_bct_field/<field>/...`
- `by_bct_subfield/<field>/<subfield>/...`

Metric group segments are slugged for stable keys.

### Error Mode Breakdown

Beyond accuracy, each rollout is classified into exactly one mutually exclusive error mode (stored on the verify response as `error_mode` and rolled up under `error_modes/<mode>` as a percentage of all rollouts, with `error_modes/<mode>/count` raw counts):

- `correct` — the extracted choice matched the gold answer.
- `wrong_answer` — a valid choice was identified but it was incorrect (a plain wrong answer).
- `refusal` — the model declined to answer (refusal content type or phrases like "I'm sorry, I can't…").
- `early_termination` — generation was cut off before a choice was produced (truncated/`incomplete` response, `max_output_tokens`, or an unclosed `<think>` block).
- `malformed_choice` — a well-formed `<choice>…</choice>` tag was emitted but its content matched no option.
- `format_violation` — no answer could be recovered and the required `<choice>` format was not followed.

## Example Usage

```bash
config_paths="responses_api_agents/simple_agent/configs/simple_agent.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/bunsenbench_chemistry_mcq/configs/bunsenbench_chemistry_mcq.yaml"

ng_run "+config_paths=[$config_paths]"

ng_collect_rollouts \
    +agent_name=bunsenbench_chemistry_mcq_simple_agent \
    +input_jsonl_fpath=resources_servers/bunsenbench_chemistry_mcq/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/bunsenbench_chemistry_mcq/data/example_rollouts.jsonl \
    +limit=5
```

`ng_collect_rollouts` may write sidecar files next to `output_jsonl_fpath` (materialized inputs, reward profiling, agent metrics). Only `example_rollouts.jsonl` is committed for smoke testing; regenerate offline artifacts with `create_examples.py` or a live rollout run as needed.

## Tests

```bash
ng_test +entrypoint=resources_servers/bunsenbench_chemistry_mcq
```

## Licensing

Code: Apache 2.0

Benchmark upstream sources use mixed licenses (MMLU-Redux, MMLU-Pro, GPQA-Diamond, SuperGPQA, ChemBench). See [benchmarks/bunsenbench_chemistry_mcq/README.md](../../benchmarks/bunsenbench_chemistry_mcq/README.md) for the source list and license notes.
