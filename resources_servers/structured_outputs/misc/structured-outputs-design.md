# Structured Outputs Data and Verifier Design

This note captures reusable design guidance from the structured outputs
resource server and data-generation paths. It is meant to inform future
structured-output datasets and verifiers. It is not a rollout operations guide.

For implementation details, see:

- `misc/data_generation/structured_outputs_v3/README.md`
- `misc/data_generation/structured_outputs_v4/README.md`
- `misc/data_generation/structured_outputs_v4/vllm-tool-schema-compatibility.md`

## Contents

- [Start With The Response Surface](#start-with-the-response-surface)
- [Task And Instruction Variation](#task-and-instruction-variation)
- [Verifier Scope](#verifier-scope)
- [Row And Verifier Contract](#row-and-verifier-contract)
- [Schema Normalization](#schema-normalization)
- [Tool-Call Specific Design](#tool-call-specific-design)
- [Serving Compatibility And Debuggability](#serving-compatibility-and-debuggability)
- [Coverage Dimensions](#coverage-dimensions)
- [Validation Feedback Loop](#validation-feedback-loop)

## Start With The Response Surface

The first design decision is the response surface:

- **Text-format structured outputs**: the model emits JSON, YAML, XML, TOML, or
  CSV text.
- **Tool-call structured outputs**: the model emits a function call whose
  arguments carry the structured output.

This choice should drive the whole row contract. It affects prompt content,
`responses_create_params`, verifier parsing, metadata, and the failure modes
worth testing.

For text-format tasks, showing the schema in the prompt makes sense because the
model needs to serialize text in the requested format. For tool-call tasks, the
schema should usually not be repeated in the prompt because the schema is
already in `responses_create_params.tools`. The natural prompt is just the
document plus a short task instruction.

## Task And Instruction Variation

Task variation is a core data-design lever for generalization. If every row is
"document + schema + output this format", the model can overfit to a narrow
interaction pattern instead of learning schema adherence as a portable behavior.

The v3 text-output generator uses several task families for this reason:

- **Direct extraction**: map a document into the requested structured format.
- **Translation**: convert a completed structured output from one format to
  another while preserving the schema.
- **Related multistep**: answer once, then reformat the previous answer in a
  follow-up turn.
- **Unrelated multistep**: ignore prior unrelated history and follow the latest
  schema/output instruction.
- **Schema-only generation**: produce plausible data from a schema without a
  document.
- **Error correction**: repair broken structured output so it validates against
  the schema.

These are not just volume multipliers. They test different invariances:
format conversion, latest-instruction following, context distraction, schema
interpretation without document grounding, and validation-oriented repair.

Instruction placement matters for the same reason. The v3 templates vary
whether schema and task instructions appear in system vs user messages, before
or after the document, split across messages, or packed into one message. This
prevents the environment from measuring only one prompt layout. Record this as
metadata when possible, because placement can become a real reward slice.

Do not copy every task family into every response surface. Tool-call structured
outputs usually make the most sense as direct document-to-tool-call generation,
because translation, error correction, and schema-only generation can become
unnatural when the schema is hidden in a tool definition. In that case,
generalization should come from short-instruction templates, system/user
placement, document placement, tool names, wrapper shapes, and distractors
rather than forcing text-output task categories into a tool-call setting.

## Verifier Scope

Structured outputs can test schema adherence, semantic correctness, or both.
These are different objectives.

The current structured outputs verifier is primarily a schema-adherence
verifier. It parses the model response, validates it against the schema, and
does not prove that every extracted value is factually complete or correct.
In observed structured-output training runs, this did not show degradation in
other measured capabilities. One plausible hypothesis is that the semantic
extraction task was simple enough that the reward mostly pressured structural
compliance. Treat that as a hypothesis, not proof.

Semantic verification is worth adding when content correctness is part of the
reward target. It can catch outputs that are structurally valid but
semantically wrong. The cost is additional ambiguity, implementation effort,
model or judge dependence, and new failure modes. If semantic verification is
omitted, document that the reward measures schema validity, not factual
completeness.

When using schema-only verification, avoid making the document-understanding
task so hard that reward becomes ambiguous. If the extraction task is hard, low
reward may no longer mean "bad schema following"; it may mean the model failed
the extraction task.

## Row And Verifier Contract

Design the row contract before scaling generation. The row should contain the
minimum fields needed for the agent to call the model and for the verifier to
parse the response deterministically.

Common text-output fields include:

- `responses_create_params.input`
- `schema_str`
- `schema_type`
- `problem_type`
- `schema_repr`
- `source_format`
- `num_turns`
- `instruction_layout` when instruction placement is varied
- `source_record_id`

Tool-call rows need additional fields that tell the verifier what to inspect:

- `responses_create_params.tools`
- `responses_create_params.tool_choice`
- `responses_create_params.parallel_tool_calls`
- `response_mode`
- `tool_choice`
- `parallel_tool_calls`
- `tool_name`
- `tool_payload_key`
- `tool_schema_mode`
- `num_tools`
- `num_distractors`
- `distractor_style`
- `tool_union_mode`

These fields are not just bookkeeping. They make reward breakdowns explainable.
If a model fails mostly on a specific task family, schema representation,
instruction layout, wrapper mode, distractor style, or response mode, that
should be visible in aggregate metrics.

## Schema Normalization

Source schemas are not clean just because they were previously verified.
Different source formats produce different quirks, and downstream serving
stacks may support only a subset of JSON Schema.

Normalize schemas before generation where possible:

- convert source formats into a JSON Schema dict
- normalize non-array enum containers into enum arrays
- convert `nullable` into an explicit `null` type union
- normalize local refs and definition locations
- remove annotation-only keywords that serving grammars reject, such as some
  `format` values
- drop invalid scalar entries in `properties` maps
- normalize bool-like strings on schema-valued keywords
- handle CSV and XML schema representations explicitly because they are easy to
  make lossy

Keep the verifier schema strict. The verifier should be the reward source of
truth and should reject missing fields, wrong types, and unexpected fields when
that is the intended task.

Model-facing schemas and verifier schemas may need different compatibility
surfaces. Make that difference explicit in data generation, and keep the
verifier as the authoritative reward surface. A relaxed model-facing schema is
not permission for the model to invent new properties; unexpected fields should
still fail against the strict verifier schema.

For the v4 vLLM tool-call case, see
`data_generation/structured_outputs_v4/vllm-tool-schema-compatibility.md`.

## Tool-Call Specific Design

For tool-call structured outputs, the function call is the answer artifact.
Tool execution is not semantically necessary unless the environment is testing
real tool-use behavior. The verifier can inspect the emitted function call
directly.

In Gym, that means the agent path should disable tool execution or use an
equivalent no-op behavior for these rows. Otherwise the framework may try to
route the answer-shaped function call as an actual tool action, which is a
different environment.

The minimum tool-call row contract should make this explicit:

- `responses_create_params.tools` contains the model-facing function schema
- `responses_create_params.tool_choice` sets the endpoint tool-selection policy
- `responses_create_params.parallel_tool_calls` records whether the endpoint is
  allowed to emit parallel tool calls
- `response_mode` is `tool_call`
- the environment uses an agent that does not execute tool calls
- verifier metadata records the expected `tool_name`, optional
  `tool_payload_key`, `tool_choice`, and `parallel_tool_calls`

Tool-call structured-output rows should reward exactly one emitted function
call. This remains true even when `parallel_tool_calls: true` is present in the
request for coverage. In that case, parallel-call permission is an input
condition, not a change in the answer contract. Missing tool calls and multiple
tool calls should both receive zero reward.

For v4-style tool-call rows, prefer `tool_choice: auto` over
`tool_choice: required`. In vLLM, `required` forces tool calls through an
internal structured-output constrained-decoding path. That can be useful for
some serving modes, but it is not the intended task surface here. The dataset
should expose tools and let the model decide whether to call one; the verifier
then assigns zero reward for no call, multiple calls, the wrong call, or
malformed arguments. This also handles documents where there may be little or
nothing useful to extract.

The tool schema should be the model-facing schema surface. The prompt should
describe the document task, not restate the full schema. This prevents the task
from collapsing back into text-prompt schema following.

Malformed generated tool-call arguments should be treated as bad model outputs
when the endpoint has already accepted the request and begun generation. This
is analogous to malformed JSON text in a text-output task. Schema/compiler
errors are different: they indicate the model-facing schema or serving stack is
incompatible and should fail before rollout collection.

## Serving Compatibility And Debuggability

Serving compatibility is part of structured-output data design when the schema
is sent to the model endpoint. A valid verifier schema can still be rejected by
the serving stack before generation, especially when a grammar compiler handles
tool schemas.

Keep these failure classes separate:

- **request/schema compile failure**: the endpoint rejects
  `responses_create_params.tools` before generation. Fix the generated
  model-facing schema or choose a different serving stack.
- **model output parse failure**: the endpoint accepted the request, generated,
  and then could not parse the completion into the expected response surface.
  Treat this as a model sample failure only if the collection path can record
  it as a failed rollout rather than crashing the whole job.
- **verifier failure**: Gym parsed the response, but the payload did not
  validate against `schema_str`.

Do not hide serving incompatibility inside a shared Responses-to-Chat
converter. If a dataset targets vanilla vLLM/Outlines, generate an explicit
vLLM-compatible tool schema surface and keep the verifier schema strict.

When Gym only reports nested 500s, add temporary boundary logging before
changing data semantics. The useful boundaries are:

- rollout collection `/run` response: row id, agent, status, and nested body
- agent `run`: row summary, model-call failure body, verifier failure
  body
- model adapter `/v1/responses`: request shape, provider error body, and
  whether failure happened during conversion, provider call, or postprocess

These logs should be temporary or explicitly gated. They are a debugging tool
for classifying the layer, not a data or reward change.

## Coverage Dimensions

Variation is useful only when it tests a real behavior.

Useful coverage dimensions include:

- task family: direct, translation, multistep, schema-only, error correction
- schema source type: JSON, YAML, XML, TOML, CSV
- schema representation in the prompt for text-output tasks
- instruction placement and message layout
- output response mode: text vs tool call
- tool argument shape: direct object vs wrapper key
- tool name style: semantic vs numbered
- distractor count and distractor rendering

Composition keywords such as `oneOf`, `anyOf`, and `$defs` are useful when
they represent real competing branches, such as target plus distractor schemas.
They are not useful as decoration. Avoid adding `oneOf`, `anyOf`, or `allOf`
to no-distractor rows unless the data actually tests a meaningful schema choice.
Also validate whether the target model can use the composition shape at all.
For tool-call response surfaces, schema-composition shapes should be probed
against the target model and serving stack before inclusion. In one probe,
inline `oneOf`, inline `anyOf`, `$defs` + `oneOf`, `$defs` + `anyOf`, and
`$defs`-only multi-branch schemas produced invalid typed tool arguments. The
dominant failure was a JSON-stringified object inside the selected payload key.
A plain single-tool multi-key object, separate tools, and numbered tools passed
the same probe, so those shapes are safer defaults.

Distractors should also be balanced deliberately. If one distractor style
dominates by accident, aggregate results will mostly measure that style rather
than the intended coverage mix.

## Validation Feedback Loop

Some validation and debugging belongs in the design loop because it tells you
whether the dataset contract is valid.

Useful checks before large-scale use:

- generate example rows and inspect the actual prompts/tools the model sees
- visualize the distribution of design dimensions
- statically scan generated schemas for constructs the target serving stack
  cannot compile
- run direct endpoint probes when a model-serving grammar may reject the schema
- run a small Gym smoke to validate row shape, agent behavior, and verifier
  parsing
- use full rollouts to inspect distribution-level reward slices and unexpected
  verifier errors

These checks are evidence about the data and verifier design. Broader
infrastructure and run-management issues belong in debugging docs, not in the
dataset design itself.

When failures occur, classify the failing layer before changing the data:

- parser failure: response cannot be parsed into the expected response surface
- model-output parse failure: endpoint accepted the request and began
  generation, but the generated response cannot be parsed into the expected
  response surface. For tool-call tasks, malformed function-call arguments are
  a bad model sample, not invalid data.
- verifier failure: parsed object does not validate against `schema_str`
- row-contract failure: metadata does not tell the verifier what to inspect
- serving-schema failure: tool schema cannot be compiled by the endpoint
- prompt-design failure: model is asked for the wrong response surface

Schema normalization should be driven by these concrete failures, not by
blindly adding compatibility transforms.
