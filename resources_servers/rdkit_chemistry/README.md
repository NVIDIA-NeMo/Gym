# rdkit_chemistry Resources Server

## Overview

This resources server verifies chemistry question answering over RDKit-computable
molecular properties drawn from the ChEMBL database.

- Task type: single-turn numeric prediction
- Domain: `knowledge`
- Methods: `direct` (parametric knowledge only) and `mcp-python` (model may call a
  Python tool with RDKit available to compute the answer)
- Dataset prompt format: user message containing a natural-language question, a
  SMILES string, and a format instruction; the model must respond with a single
  integer or binary `0`/`1` flag

Questions cover four property types:

| Property type | Examples | Expected response |
|---|---|---|
| `count` | HeavyAtomCount, NumValenceElectrons | Single integer |
| `bool` | PassesRo5, PassesVeber | `0` or `1` |
| `presence` | HasAmide | `0` or `1` |
| `fragment` | fr_Al_COO, fr_Al_OH | Single integer |

## Reward Signal

All property types use exact match: 1.0 if `round(predicted) == round(actual)`, else 0.0.
When no parseable number can be extracted from the response, `reward = 0.0`.

## Server Composition

Use `rdkit_chemistry` with:

- `responses_api_agents/simple_agent`
- `responses_api_models/*` (typically `policy_model`)
- `resources_servers/rdkit_chemistry`

For `mcp-python` rows the agent must have access to `ns_tools` for Python code
execution; use `rdkit_chemistry.yaml` which includes the `ns_tools` and agent definitions.

## Dataset Format

Each JSONL row:

- `responses_create_params.input[0].content`: user prompt (question + SMILES + format instruction)
- `responses_create_params.tools`: `[]` for `direct`, `[stateful_python_code_exec]` for `mcp-python`
- `expected_answer`: ground-truth numeric value (string or int)
- `property_type`: one of `count`, `bool`, `presence`, `fragment`
- `property`: RDKit property name, e.g. `NumValenceElectrons`
- `chembl_id`: ChEMBL molecule identifier
- `smiles`: canonical SMILES string
- `method`: `direct` or `mcp-python`

See `data/example.jsonl` for concrete examples.

## Example Usage

```bash
config_paths="resources_servers/rdkit_chemistry/configs/rdkit_chemistry.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"

ng_run "+config_paths=[${config_paths}]"

ng_collect_rollouts \
    +agent_name=rdkit_chemistry_agent \
    +input_jsonl_fpath=resources_servers/rdkit_chemistry/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/rdkit_chemistry/data/example_rollouts.jsonl
```

## Licensing

Code: Apache 2.0
Dataset derived from ChEMBL (CC-BY-SA 3.0)
