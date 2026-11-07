# GYM-FM-02 grading criteria

These four requirement IDs are the complete pass criteria. Do not add requirements.
Harbor reward 1 requires all four requirements to be supported.

## Grading criteria

### 1. `carlos_task`

Pass when: Gym evaluated the exact Carlos request in task_request.

- A real model response is recorded. The outer agent answering directly does not count.

### 2. `gym_evaluation`

Pass when: Gym actually ran the evaluation through its CLI or direct Python API.

- Shell scripts, Python subprocess wrappers, command variables and readiness waits are acceptable.
- Execution records must establish that Gym performed the evaluation. Configuration, a script or a saved result alone is insufficient.
- Reliable successful completion of this evaluation is required.

### 3. `gym_agent`

Pass when: This evaluation used the Gym agent selected by `GYM_AGENT`,
provided to the judge as `gym_agent.harness`.

- Actual execution must establish that the selected agent ran.
  The outer coding agent, component names or configuration alone
  do not establish it.

### 4. `saved_scored_result`

Pass when: At least one inspectable, newly generated, genuinely scored Carlos rollout is saved under /workspace and linked to that same completed evaluation.

- Copied samples, fabricated output, skipped verification, collection failures and empty placeholders do not count.
- Other files, failed retries, duplicate copies and tied timestamps do not invalidate a separate valid result.

## Saved-result evidence

These eight facts explain evidence for `carlos_task` and `saved_scored_result`.
They are not additional agent instructions.

### `one_scored_carlos_rollout_saved`

The selected saved result contains exactly one Carlos row with the exact task_request.

### `gym_reward_is_valid`

The Gym reward is a finite number between zero and one, not a boolean. Gym scores email quality, so a valid zero is acceptable for the workflow.

### `gym_verification_not_skipped`

Gym verification was not skipped.

### `rollout_is_not_collection_failure`

The selected row has no collection failure and is not masked out of scoring.

### `model_interaction_saved`

The response has an ID, a model label and model output: a tool call, nonblank assistant text, refusal or reasoning summary. Tool results alone and blank placeholders are insufficient.

### `rollout_not_copied_from_bundled_samples`

Its output does not match the supplied known bundled-output fingerprints.

### `response_belongs_to_attempt`

The response's created_at timestamp belongs to this attempt and is consistent with the chosen evaluation. When container times are available, allow one second before container start through 120 seconds after verification begins. Times are evidence, not proof of execution attribution.

### `recorded_tool_calls_have_results`

Each recorded model tool call has a matching result. Tool errors are allowed, and tool calls are not mandatory.
