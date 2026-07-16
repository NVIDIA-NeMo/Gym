# rdkit_chemistry Resources Server

> **⚠️ Deprecated.** The scoring path in this server never depended on chemistry;
> it has been generalized into the domain-agnostic
> [`litmus_agent`](../litmus_agent) resources server (answer extraction + a
> swappable reward-rule taxonomy, plus an optional sandbox-backed Python
> code-execution tool). New work should target `litmus_agent`; export legacy
> chemistry rows there via its `property_type` → `answer_type` back-compat
> mapping. `rdkit_chemistry` is retained only for existing runs and will not
> receive new features.

## Overview

This resources server verifies chemistry question answering over RDKit-computable
molecular properties drawn from the ChEMBL database.

- Task type: single-turn numeric prediction
- Domain: `knowledge`
- Methods: `direct` (parametric knowledge only) and `mcp-python` (model may call a
  Python tool with RDKit available to compute the answer)
- Dataset prompt format: user message containing a natural-language question, a
  SMILES string, and a format instruction; the model must respond with a numeric
  value in the requested answer format

Questions cover five property types:

| Property type | Examples | Expected response |
|---|---|---|
| `count` | HeavyAtomCount, NumValenceElectrons | Single integer |
| `bool` | PassesRo5, PassesVeber | `0` or `1` |
| `presence` | HasAmide | `0` or `1` |
| `fragment` | fr_Al_COO, fr_Al_OH | Single integer |
| `float` | MolWt, TPSA, QED | Floating point number |

## Reward Signal

Discrete property types use exact match: 1.0 if `round(predicted) == round(actual)`, else 0.0.
Float properties use tight numeric equality.
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
- `answer_format`: optional string key `fmt_00` through `fmt_30`, selecting the
  regex used to extract the final answer

See `data/example.jsonl` for concrete examples.

Legacy rows without `answer_format` are still accepted. For those rows,
`use_box_format: true` maps to boxed extraction and `use_box_format: false`
maps to double-parentheses extraction. When `answer_format` is present, it
takes precedence over `use_box_format`.

## Example Usage

```bash
gym env start \
    --resources-server rdkit_chemistry \
    --model-type openai_model

gym eval run --no-serve \
    --agent rdkit_chemistry_agent \
    --input resources_servers/rdkit_chemistry/data/example.jsonl \
    --output resources_servers/rdkit_chemistry/data/example_rollouts.jsonl
```

## Licensing

Code: Apache 2.0
Dataset derived from ChEMBL (CC-BY-SA 3.0)
