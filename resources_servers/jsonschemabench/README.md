# JSONSchemaBench

> Keywords: Evaluation, Structured Outputs, JSON Schema

This environment evaluates JSON text against the original JSONSchemaBench schema with
`jsonschema.Draft202012Validator` and format checking. It deliberately does not make optional
properties required or add `additionalProperties: false`; those transformations change the
benchmark contract and belong to strict structured-output training environments instead.

The input row must provide `responses_create_params`, `schema_str`, and `schema_type: json`.
Optional `problem_type` and source-identity fields are copied into the rollout for subset metrics
and provenance.

Run the example with:

```bash
gym eval run \
  --resources-server jsonschemabench/jsonschemabench \
  --agent jsonschemabench_simple_agent \
  --input resources_servers/jsonschemabench/data/example.jsonl \
  --output results/jsonschemabench_example.jsonl
```
