# BunsenChem Resources Server

BunsenChem verifies chemistry multiple-choice outputs for the public BunsenBench Gym benchmark. It stores materialized answer letters internally, accepts exact choice-text answers, and passes source/taxonomy metadata through to aggregate metrics.

The examples in `data/example.jsonl` are synthetic. They are not redistributed benchmark-source questions.

### Input schema

Required fields:
- `responses_create_params`: OpenAI Responses create params.
- `expected_answer`: gold answer letter, such as `A` or `D`.

Supported choice fields:
- `options`: MCQA-style list of single-key dicts, such as `[{"A": "H2O"}, {"B": "CO2"}]`.
- `choices`: list of choice texts. Letters are assigned by position (`A`, `B`, `C`, ...).
- `choices`: list of dicts with `letter`/`label` and `text`/`content` keys.

Source/taxonomy fields may be top-level or nested under `metadata`:
- `source`
- `bct_field`
- `bct_subfield`

### Answer extraction

Extraction is deterministic and supports:
- `Answer: A`
- `The answer is A`
- `\boxed{A}` and `\boxed{\text{A}}`
- `<answer>A</answer>`, `<choice>CO2</choice>`, or `<response>A</response>`
- Exact choice text on the final answer line

Exact choice-text matching normalizes common chemistry Unicode variants: subscripts, superscripts, plus/minus variants, multiplication signs, middle dots, micro signs, and compatibility characters.

### Aggregate metrics

The standard `/aggregate_metrics` framework hook calls `BunsenChemResourcesServer.compute_metrics()`. This server emits overall pass/majority/no-answer metrics plus grouped metrics:
- `by_source/<source>/...`
- `by_bct_field/<field>/...`
- `by_bct_subfield/<field>/<subfield>/...`

Metric group segments are slugged for stable keys.

### Running

```bash
config_paths="responses_api_agents/simple_agent/configs/simple_agent.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/bunsen_chem/configs/bunsen_chem.yaml"

ng_run "+config_paths=[$config_paths]"

ng_collect_rollouts \
    +agent_name=bunsen_chem_simple_agent \
    +input_jsonl_fpath=resources_servers/bunsen_chem/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/bunsen_chem/data/example_rollouts.jsonl \
    +limit=5
```

### Tests

```bash
ng_test +entrypoint=resources_servers/bunsen_chem
```
