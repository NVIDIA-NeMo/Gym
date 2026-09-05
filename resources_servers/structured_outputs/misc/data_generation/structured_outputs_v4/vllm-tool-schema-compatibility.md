# vLLM Tool Schema Compatibility

## Decision

Structured Outputs v4 keeps the shared Nemo Gym Responses-to-Chat conversion
semantics-preserving. The converter should not silently relax tool schemas. If
vanilla vLLM rejects a JSON Schema construct, the v4 data generator should make
that compatibility choice explicitly in the generated data.

## Why This Is A Data Decision

The v4 rows are intended to test tool-call behavior against the serving stack
we actually run. If the data contains tool schemas that vanilla vLLM/Outlines
cannot compile, rollout collection fails before the model can generate. Hiding
that in `responses_api_models/vllm_model/app.py` would make the shared wrapper
behave differently from vanilla vLLM and would affect unrelated environments.

Some schemas can be valid JSON Schema, and even reasonable OpenAPI-style
schemas, while still being incompatible with the vLLM tool grammar path. That
is a serving compatibility problem, not a reason to change the shared Gym
conversion layer.

## Two Schema Surfaces

Each generated row now has two related but intentionally different schema
surfaces:

```text
schema_str
  strict verifier schema
  used only by resources_servers/structured_outputs/app.py

responses_create_params.tools[].parameters
  vLLM-compatible tool schema
  sent to the model endpoint after Responses-to-Chat conversion
```

`schema_str` remains the strict verifier surface. It makes all declared fields
required, normalizes enum/nullable/ref quirks, and closes objects with
`additionalProperties: false`.

The tool `parameters` schema starts from the same strictified schema, then
applies a vLLM compatibility transform.

This separation only works because the verifier owns the reward. The
model-facing tool schema is a generation constraint, not the source of truth.
If the model emits extra properties, those properties should still fail against
the strict `schema_str`. In this task, a new property that was not in the
schema is a hallucinated field, not an acceptable extension.

## Observed vLLM Failures

The most important failure was boolean schema values in tool parameters:

```text
Unsupported JSON Schema structure false
```

The common source is object closure:

```json
{
  "type": "object",
  "properties": {
    "title": {
      "type": "string"
    }
  },
  "required": ["title"],
  "additionalProperties": false
}
```

In JSON Schema, `additionalProperties: false` is valid. It means that keys
outside `properties` are not allowed. In the vLLM/Outlines tool grammar path we
tested, that boolean schema value could be rejected while compiling tool
parameters.

The failures often surfaced on `$defs` plus `oneOf` / `anyOf` distractor rows,
but composition was not the root problem by itself. Those rows carried many
strict object branches through nested tool schema positions, so there were many
more chances to hit a boolean closure keyword.

Other serving-compatibility failures were separate normalization issues:

- `Grammar error: Unknown format: uri`
- `Grammar error: schema must be an object or boolean`

For v4 generation, source schemas are normalized before both verifier and tool
schema construction: unsupported `format` annotations are removed, invalid
scalar entries in `properties` maps are dropped, and bool-like strings such as
`additionalProperties: "false"` are converted to real booleans.

## Compatibility Transform

The generator removes boolean object/list closure keywords from tool
parameters:

```json
{
  "type": "object",
  "properties": {
    "summary": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "title": {
          "type": "string"
        }
      },
      "required": ["title"]
    }
  },
  "required": ["summary"],
  "additionalProperties": false
}
```

becomes this for `responses_create_params.tools[].parameters`:

```json
{
  "type": "object",
  "properties": {
    "summary": {
      "type": "object",
      "properties": {
        "title": {
          "type": "string"
        }
      },
      "required": ["title"]
    }
  },
  "required": ["summary"]
}
```

The strict `schema_str` still contains `additionalProperties: false`, so the
verifier can still reject extra keys.

The generator also replaces remaining boolean schema nodes in tool parameters
with `{}`. This avoids vanilla vLLM/Outlines errors such as:

```text
Unsupported JSON Schema structure false
```

Values under non-schema keywords such as `enum`, `const`, `default`, and
`examples` are preserved.

## Tradeoff

The tool grammar is less strict than the verifier. This means vLLM can generate
some extra keys that the tool grammar would otherwise block. That is acceptable
for this dataset because the reward verifier still checks the stricter
`schema_str`, so extra keys become failed examples rather than false positives.

The important constraint is that this relaxation is explicit in the generated
v4 data. It is not hidden in the shared model wrapper.

Do not apply this transform blindly to other environments. If another serving
stack supports full boolean schemas in tool parameters, it may be better to send
the strict tool schema directly. The v4 generator is making a target-serving
choice for the vLLM path used in these rollouts.

## Repro Check

After regenerating data, scan the tool schemas for boolean schema nodes:

```bash
cd /lustre/fsw/portfolios/llmservice/users/jkyi/current/nemo/Gym-github

python resources_servers/structured_outputs/misc/check_tool_call_jsonl.py \
    -i resources_servers/structured_outputs/data/structured_outputs_v4_tool_call.jsonl
```

Expected result:

```text
Errors: 0
```
