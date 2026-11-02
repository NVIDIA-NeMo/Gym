---
name: nemo-gym-pivot-datasets
description: >-
  Use when creating, validating, or documenting Nemo Gym pivot datasets from rollout,
  trajectory, chat-completion, Responses API, or tool-call artifacts. Covers Gym
  Responses-style row conversion, reconstructing model calls from flattened rollout output,
  parallel tool-call (function_call_batch) labels, reasoning placement and label leakage,
  pivot selection, single-step tool-use configs, agent_ref alignment, verifier knobs,
  expected-action row contracts, and train/eval usage.
---

# Nemo Gym Pivot Datasets

## Paper Reference

This skill operationalizes [PivotRL](https://arxiv.org/html/2603.21383v1): create local
single-step pivot datasets from successful trajectories, prefer informative mixed-reward states,
and train with verifier-based local rewards rather than exact trajectory imitation.

## Invocation Check

Use this skill when the task is to turn existing agent trajectories or rollout artifacts into a
Nemo Gym pivot dataset, or to validate whether a pivot JSONL/config pair can be used for
single-step local RL or evaluation.

Before writing a converter, inspect representative source rows and the target resource server.
Do not assume the source field names are the contract. Convert by reconstructing the semantic
pieces needed by Gym's Responses-style row format.

## Core Workflow

1. Inspect the source data shape and count the candidate assistant decision points.
2. Reconstruct model calls from the source's flattened output list, and cross-check the count
  against whatever per-trajectory model-call count the source records. A decision point is one whole
  model call, so this step decides what a row even is.
3. Identify the semantic fields needed for each pivot:
  - model-call input context before the pivot action
  - available tools at that decision point
  - expected assistant action
  - reward/verifier target if it is separate from the demonstrated action
  - optional provenance such as task id, source trajectory id, rollout id, uuid, depth, and original metadata
4. Convert each accepted decision point into one pivot JSONL row.
5. Generate or update the matching Gym config so the pivot-format JSONL can be used directly.
6. Validate with the bundled validator and, when available, the target Gym resource-server models.
7. Write metrics that make skipped rows, action types, tool names, depth, and provenance coverage easy
  to inspect, including the parallel / single / chat split and a batch-size histogram.

## Row Shape

Read [references/row-contract.md](references/row-contract.md) when implementing or reviewing a
converter. For `single_step_tool_use_with_argument_comparison`, the essential row fields are:

- `responses_create_params`: Responses API-style input and tool specs for the model call.
- `expected_action`: one `function_call`, one `function_call_batch` (parallel calls emitted in a
  single response), or one `message`.
- `agent_ref`: row-level agent routing that matches the generated config. Optional when the config
  routes the dataset instead.

Do not copy optional null fields into `responses_create_params`; omit them unless the target
contract explicitly wants them.

## One Model Call, One Row

A pivot covers one whole model call. No calls is a `message`, exactly one call is a `function_call`,
and two or more calls in the same response are one `function_call_batch` — never split across rows,
never dropped, and never a one-element batch. Tool calls take precedence over assistant text, so a
turn that narrates *and* calls is labelled with its calls.

Reconstructing "one model call" from a flattened output list is the hard part, and the same pass
decides which reasoning belongs to the prefix and which would leak the label. Read
[references/rollout-artifact-pitfalls.md](references/rollout-artifact-pitfalls.md) before writing a
converter against raw rollout artifacts.

## Conversion Patterns

Read [references/conversion-patterns.md](references/conversion-patterns.md) when the source data
is not already in pivot shape. The rule is to normalize by meaning, not by source container.

Useful reference scripts live under `scripts/reference/`. They are copied from real conversions and
may contain dataset-specific paths, assumptions, or older branch behavior, so treat them as examples
to borrow from rather than canonical commands to run unchanged:

- `generic_pivot_dataset_reference.py`: generic source rows to pivot rows.
- `chat_messages_to_pivot_dataset_reference.py`: chat-completion messages to pivot rows.
- `conversational_messages_to_pivot_dataset_reference.py`: conversational message trajectories to pivot rows with reasoning/provenance handling.
- `tool_messages_to_pivot_dataset_reference.py`: message/tool-use style rows to pivot rows.
- `responses_output_to_pivot_dataset_reference.py`: Gym rollout artifacts whose trajectory is a flat
  list of Responses API output items. Start here — it is self-contained and runnable, and it is the
  only one that has to reconstruct model calls, so it demonstrates segmentation, parallel batch
  labels, reasoning placement and `reference_output` together.

The four message-list converters were copied from real pipelines; two of them import modules that do
not exist in this repository and cannot be run as written.

## Pivot Selection

Use clean, positive source trajectories for the demonstrated pivots. When multiple source
trajectories exist for a task, prefer tasks whose source trajectory group has mixed rewards
instead of all success or all failure; this avoids spending data on tasks that were trivial or
impossible for the source model. Treat that source-task filter as preferred, not mandatory, because
the source model and downstream policy may have different capabilities.

When possible, profile candidate pivots with local on-policy rollouts from the downstream or
initial policy. Use at least 8 sampled local rollouts per candidate as the default. Keep candidates
with mixed local rewards, discard all-1 and all-0 reward groups, and if data is abundant, drop the
easiest/high-pass-rate pivots first so training concentrates on hard but learnable states.

When every source trajectory carries the same reward, the source-task filter has nothing to act on.
Report that in the metrics rather than skipping it silently, keep the full set, and rely on local
on-policy profiling for selection.

## Config And Training

Read [references/config-training-and-agent-ref.md](references/config-training-and-agent-ref.md)
when creating the Gym YAML or explaining how to train/evaluate from the dataset.

Key points:

- The pivot JSONL is the training/eval dataset; point the config's train dataset entry directly at it.
- `agent_ref.name` in each row must match the agent block used by the config unless the launcher overrides routing intentionally.
- `word_count_similarity_threshold` is the main string-argument matching knob for the single-step tool-use verifier. Its maximum useful value is 0.5, not 1.0: identical strings score exactly 0.5, so anything higher rejects every multi-word string argument.
- `parallel_tool_call_rewarding` must be on for `function_call_batch` datasets, or the number of calls a response makes is not part of the verdict.
- Use `tool_choice: "auto"` for these rows; `tool_choice: "required"` can route some inference engines into structured decoding paths.
- Validate configs and datasets together; a valid JSONL file can still be unusable if the agent/resource-server names do not line up.

## Validation

Run the bundled validator before calling a pivot dataset done:

```bash
python scripts/validate_pivot_dataset.py --path /path/to/pivot.jsonl --agent-ref expected_agent_name
```

When the Gym repo is available, also validate against the resource-server Pydantic models:

```bash
python scripts/validate_pivot_dataset.py \
  --path /path/to/pivot.jsonl \
  --agent-ref expected_agent_name \
  --gym-repo /path/to/Gym-github
```

Use `--require-field` and `--require-any-field` only when a dataset-specific workflow needs extra
provenance checks. Provenance is useful for debugging and filtering, but it is not required by the
resource-server request model.

The validator accepts all three expected-action types (`function_call`, `function_call_batch` and
`message`) and prints an end summary split between parallel, single and chat pivots with a
batch-size histogram. Pass `--no-require-agent-ref` for datasets routed by their config rather than
by the row.

Two further scripts, for the cases the base validator cannot cover:

```bash
# Once a single validation pass is measured in tens of minutes, shard it by byte range.
python scripts/validate_pivot_dataset_sharded.py --path /path/to/pivot.jsonl \
    --workers 32 --gym-repo /path/to/Gym --config /path/to/agent_config.yaml

# Whenever the converter had to reconstruct model calls from a flattened output list.
# With --source-path alone this reports source anomalies instead of auditing a pivot file.
python scripts/audit_pivot_turn_boundaries.py --source-path /path/to/rollouts.jsonl \
    --pivot-path /path/to/pivot.jsonl
```

## Reference Loading

Load a reference only when the task needs that detail:

- [references/row-contract.md](references/row-contract.md) — writing or reviewing row fields,
  `expected_action` types, provenance.
- [references/conversion-patterns.md](references/conversion-patterns.md) — the source is not already
  in pivot shape and needs mapping by meaning.
- [references/rollout-artifact-pitfalls.md](references/rollout-artifact-pitfalls.md) — converting
  from raw rollout artifacts, reasoning placement, validating or running very large pivot files.
- [references/config-training-and-agent-ref.md](references/config-training-and-agent-ref.md) —
  writing the Gym YAML, verifier knobs, threshold tuning, pivot selection for training.
