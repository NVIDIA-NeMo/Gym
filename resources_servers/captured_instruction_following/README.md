# Captured Instruction Following

Generates one new model response from a captured evaluation input. It supports two explicit protocols:

| Protocol | Model input | Verification |
| --- | --- | --- |
| `native_next_turn` | Original Responses API messages and declared function schemas | Compare the first generated function call with a supplied native reference. Tools are not executed. |
| `single_turn_text_replay` | One text prompt containing a frozen conversation and its tool descriptions/results | Generation only. Until criteria are reviewed, return `reward: 0`, `mask_sample: true`, `scoring_status: not_scored`, and `scoring_reason: criteria_require_review`. |

An unscored placeholder zero is not a model failure. This component never calls an LLM judge. It does not reconstruct a live browser, filesystem, API, or coding environment, and it does not claim equivalence to an original benchmark's end-to-end reward.

## Import captured traces

This follows the captured-prefix approach in [Gym PR #5013](https://github.com/NVIDIA-NeMo/Gym/pull/5013), with a separate cohort and explicit native-call scoring. It leaves the existing frozen Agentic IF Realworld release unchanged.

Create a private JSONL inventory with `benchmark`, `orig_trajectory_id`, `task_index` (for repeat grouping), `source_record_path`, and the original file's `source_record_sha256`. Relative paths resolve against the inventory directory. Each source record contains its matching `benchmark` and `trajectory_id`, plus `source_trajectory.responses_create_params` and `source_trajectory.response.output`. Historical reward, verifier metadata, and analysis can remain in the source record; the importer retains them in the private baseline rather than treating them as new scores.

For existing native references, supply a private JSONL reference map with matching `benchmark` and `orig_trajectory_id`, `comparison_profile`, and `expected_call: {name, arguments}`. The reference must match the source verifier's `expected_function` and `expected_params`. A missing or inconsistent native profile is an error. Text-only cases remain unreviewed; a failure label or an old zero reward cannot establish reviewed criteria.

```bash
python -m resources_servers.captured_instruction_following.import_traces \
  --manifest /private/captured-inputs/inventory.jsonl \
  --reference-map /private/captured-inputs/reference-map.jsonl \
  --output-dir /private/captured-inputs/gym-dataset
```

The output contains `prepared.jsonl`, `baseline_outputs.jsonl`, `provenance.jsonl`, and a hash-bound `release_manifest.json`. The importer refuses an existing destination and refuses to write private exports inside a Git checkout. Use `--expected-count` to validate a known cohort size. Sampling defaults are explicit evaluation settings; all captured request fields remain available in the private baseline and transformation receipts.

Native rows preserve the original request's semantic input and tool definitions. Text rows embed the complete applicable tool registry and captured history in one user message, with actual tools disabled. If the recorded episode has tool results, the prefix includes the exact earlier events through the last result and holds out the complete subsequent assistant turn, including its reasoning. These earlier actions are supplied history, not actions retested by the new checkpoint. No original held-out answer or grading metadata is supplied as model input.

## Row contract

Each row contains `responses_create_params`, top-level `benchmark`, `orig_trajectory_id`, and `task_group_id`, plus `verifier_metadata`:

```json
{
  "schema_version": "captured-instruction-following/v1",
  "evaluation_mode": "native_next_turn",
  "source_identity": {
    "benchmark": "fictional_workflow",
    "orig_trajectory_id": "synthetic-native-001",
    "task_group_id": "synthetic-native-001-group"
  },
  "source_request_sha256": "<canonical original-request digest>",
  "evaluation_request_sha256": "<canonical evaluated-request digest>",
  "private_baseline_ref": {
    "path": "<private baseline artifact>",
    "row_id": "<source row>",
    "sha256": "<canonical source-record digest>"
  },
  "criteria_status": "native_reference_available",
  "comparison_profile": "case_insensitive_scalars",
  "expected_function": "get_open_invoice",
  "expected_params": {"customer_id": "SYN-CUST-1"},
  "cohort_size": 5
}
```

For text replay, use `criteria_status: pending_review` and omit all three native scoring fields. `task_data.py` is the dependency-light validation schema; Gym flattens `verifier_metadata` for task-data validation.

The importer validates source-record and original-request hashes offline. The resources server checks the top-level identity against `source_identity` and binds the evaluated request to `evaluation_request_sha256`; a mismatch is masked. It never opens the baseline artifact or uses a historical reward to score a new response. Digests use sorted-key, compact, UTF-8 JSON with `ensure_ascii=false` and finite numbers. Hash the request after validating it through the target Gym request model with `model_dump(mode="json", exclude_unset=True)`; Python integer and float serialization can differ. Sampling or request overrides require a freshly prepared binding.

## Native comparison

The function name is case sensitive. Only the first generated `function_call` is compared, and expected parameters form a subset of its arguments. Extra arguments and subsequent calls are ignored by this protocol and explicitly reported as `extra_argument_keys` and `ignored_call_count`.

The reviewed profile selects scalar comparison:

- `case_insensitive_scalars`: ordinary value equality, or equal lowercased string representations. Surrounding whitespace remains significant.
- `normalized_scalars`: preserve null; otherwise lowercase and trim the string representation, recognize `true/1/yes` and `false/0/no`, then attempt integer or decimal conversion. These are scalar profiles, not recursive structured-object matching.

A missing expected argument is compared as null. Thus an expected null can match a missing key; `missing_expected_keys` reports the omission. Empty expected parameters require only a matching function with valid JSON-object arguments. Malformed JSON, duplicate argument keys, nonfinite numbers, and non-object arguments fail. Text that describes a call is not accepted as an actual function call. A completed response with no function call is an unmasked failure; an incomplete response or provider error is masked.

## Coverage and metrics

The metrics endpoint reports `passed`, `failed`, `scored`, and `not_scored`, plus item coverage. Masked generations do not enter failure counts or accuracy. `scored_subset_accuracy` describes measured samples only. `complete_accuracy` stays null if any observed sample is unscored or any expected item is missing.

Set `verifier_metadata.cohort_size` on every row or configure `expected_items` on the resource instance to measure coverage against the complete cohort. Conflicting sizes are reported; the server does not invent a denominator. Repeated attempts increase trial counts, while coverage counts distinct `(benchmark, orig_trajectory_id)` identities.

## Run and test

The config uses `simple_agent` with `execute_tools: false` and `max_steps: 1`, routed through `single_agent_turn_legacy` with no tool transports. This uses an agent session; the direct Simple Agent `/run` route does not support disabled tool execution. There is no episode deadline in this config. Per-request token limits and model-server timeouts apply.

```bash
gym env start --resources-server captured_instruction_following --model-type openai_model
gym eval run --no-serve \
  --agent captured_instruction_following_simple_agent \
  --input resources_servers/captured_instruction_following/data/example.jsonl \
  --output results/captured_if_synthetic.jsonl
gym env test --resources-server captured_instruction_following
```

`data/example.jsonl` contains five entirely fictional smoke tasks. `example_rollouts.jsonl`, when present, contains explicitly scripted policy fixtures, not live checkpoint measurements. Their flags prohibit treating them as benchmark results. Publish only fictional examples and generic code; keep captured source rows, historical answers, private baselines, and annotations in their authorized data store.

The verifier records its calculation with `observe_programmatic_grading` (the resources server owns grading). It is stateless and can reverify saved responses. Unit tests exercise actual Gym agent sessions with a scripted model transport, native comparison failures, masked text, request/identity mismatches, and coverage accounting. A real model smoke is separate rollout evidence and must be recorded before claiming the integration has been baselined.
