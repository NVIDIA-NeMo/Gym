# Rollout Health RFC Ambiguities

The rollout-health RFC is the product specification for this branch. This document records only decisions that the RFC leaves open; it is not a second specification. Each numbered section states the gap, the behavior implemented in this branch, and the alternatives considered.

## Code map

- [`BaseVerifyResponse`](nemo_gym/base_resources_server.py) defines the common fields returned by a Gym verifier. The agent-specific `response` inside it can have different shapes.
- [`TrajectoryRecord`, `TrajectoryTurn`, `AgentInvocation`, and `ModelCallRef`](nemo_gym/rollout_observability.py) define Gym's standard trajectory and correlation data.
- [`CheckSpec`, `Finding`, and `RolloutDigest`](nemo_gym/health/types.py) define the health-check data contracts and per-rollout result.
- [The check registry and check functions](nemo_gym/health/checks.py) normalize persisted evidence and emit findings for one rollout.
- [`run_health_checks`](nemo_gym/rollout_health.py) coordinates byte-offset indexing, worker processes, reduction, report writing, and the CLI-facing workflow.
- [`RolloutCollectionHelper`](nemo_gym/rollout_collection.py) collects and aggregates rollouts. It also attaches `ng_trajectory` to new rollout records.
- [`gym eval health-check`](nemo_gym/cli/main.py) is the standalone command. The same runner executes automatically after `gym eval run` and `gym eval aggregate`.

## Glossary

- **Task:** One benchmark problem. A task may be attempted more than once.
- **Rollout or repeat:** One agent attempt at a task. `_ng_task_index` identifies the task and `_ng_rollout_index` identifies the repeat.
- **Artifact:** A file saved by an evaluation run, such as rollout records, aggregate metrics, or health reports.
- **JSONL:** A text file containing one complete JSON value per line. Gym stores one rollout record per non-empty line.
- **Rollout record:** One JSON object in the rollout JSONL file. It contains the task and repeat identity, the agent result, verifier output, and any observability data attached during collection.
- **Observability data:** Evidence recorded so a completed run can be inspected, such as conversations, model calls, token counts, status codes, and timing. Gym writes the final standard form under the rollout record's `ng_trajectory` key. It does not change the rollout or its score.
- **Canonical trajectory:** The one `TrajectoryRecord` stored under `ng_trajectory`. This is the only observability representation read by health checks.
- **Observation gap:** An `ObservationGap` in `ng_trajectory.gaps`. The observability code writes a gap when it could not collect, normalize, or join some evidence exactly. A gap is a fact about evidence quality, not a health verdict by itself.
- **Transcript:** Saved conversation evidence for the user, agent, and tools. Health checks read its agent turns only from `ng_trajectory.turns`.
- **Agent turn:** One unit of agent activity. It may include reasoning, an assistant message, and one or more function calls. A user message or function-call output starts the next interaction.
- **Agent invocation:** One execution of a root agent or subagent. An invocation can contain several conversation items, turns, tool calls, and model calls. The code type is `AgentInvocation`.
- **Model call:** One request sent to a language model and its response.
- **Model-call evidence:** The request, response, token, status, and timing data stored in `ng_trajectory.model_calls`.
- **Model-call reference:** An identifier that connects one canonical trajectory turn to one canonical model call. `ModelCallRef` accepts either `model_call_id` or the pair `model_ref` and `response_id`.
- **Binding:** A successful match between a reference in `TrajectoryTurn.model_calls` and exactly one entry in `TrajectoryRecord.model_calls`.
- **Correlation:** The information needed to bind trajectory turns to canonical model calls.
- **Policy model:** The model being evaluated.
- **Auxiliary model:** A model used to support the evaluation, such as a user simulator or judge. Its calls may appear in the same canonical model-call list as policy-model calls.
- **Check:** One registered health rule. `CheckSpec` gives it a stable ID, an evaluation scope, a finding subject, and required inputs. Evaluation scope says how much data the rule reduces: one rollout, one task across repeats, or the run as a whole. Finding subject says what the evidence identifies, such as an agent turn or model call.
- **Finding:** Evidence that a check failed for a rollout or task. A check emits `Finding`; it never emits a verdict.
- **Healthy:** Every enabled rollout check had enough input to run, and none produced a finding.
- **Unhealthy:** At least one enabled rollout check produced a finding.
- **Unobserved:** No enabled rollout check produced a finding, but at least one could not run because required evidence was missing.
- **Shard:** One part of a larger rollout file. Separate workers may write shards that are later aggregated.

## Transcript and check behavior

### 1. Missing canonical agent turns

**RFC gap.** Current Gym stores agent turns in `TrajectoryRecord.turns`. The RFC does not say how a check should behave when the canonical trajectory or its turns are unavailable.

**Chosen behavior.** Health checks read agent turns only from `ng_trajectory.turns`. They do not reconstruct turns from `ng_trajectory.invocations`, `response.output`, or any agent-specific record shape.

- No valid `ng_trajectory`, a `trajectory_projection_failed` gap, or a `turns_unavailable` gap makes `rollout_missing_agent_turns` and `agent_turn_hollow` unobserved.
- A valid canonical trajectory whose turns are available but contain no agent activity produces `rollout_missing_agent_turns`.
- Each canonical turn is reduced to whether it contains non-empty answer or reasoning content, a tool call, or an explicit model-call reference.

**Alternatives considered.** Reconstruct historical turns from invocation conversations or `response.output`, or reject the whole rollout. Reconstruction makes the result depend on guessed agent-specific formats; rejection would prevent the other checks from running.

### 2. Whether reasoning counts as message content

**RFC gap.** `agent_turn_hollow` means “no message and no tool calls,” but the RFC does not say whether a reasoning-only turn has a message.

**Chosen behavior.** Any non-empty answer, content, text, output text, reasoning content, reasoning summary, or encrypted reasoning content means the turn is not hollow. A tool call also means the turn is not hollow.

**Alternative considered.** Count only user-visible answer text. That would mark reasoning-only model activity as hollow.

### 3. Deterministic dispatch and zero-token calls

**RFC gap.** The RFC exempts deterministic-dispatch steps from `model_call_zero_completion_tokens` without explaining the term.

**Chosen behavior.** A deterministic dispatch is agent logic that produces a step without calling the model. Because `model_call_zero_completion_tokens` examines explicitly bound policy-model calls, such a step has no bound call and is automatically outside this check. Every bound call reporting zero completion tokens produces a finding. Unreferenced canonical model-call entries are not evaluated because they may belong to a user simulator or judge. The runner does not infer an exemption from request parameters or metadata.

**Alternative considered.** Add or infer a special marker on model calls. No such marker exists in the specified artifacts, so doing this would invent a new contract.

### 4. Binding trajectory turns to model calls

**RFC gap.** The model-call checks need to know which calls belong to the policy trajectory, but the RFC does not define how to make that connection or distinguish absent evidence from contradictory evidence.

**Chosen behavior.** Both sides of the match come from the same canonical `TrajectoryRecord`: references come from `TrajectoryTurn.model_calls`, and call evidence comes from `TrajectoryRecord.model_calls`. The runner accepts only an explicit `ModelCallRef`: either `model_call_id`, or both `model_ref` and `response_id`. It never matches by position, payload similarity, or model name. All policy-model-call checks consume the same binding result.

- A turn without a reference does not produce a finding. It may be deterministic agent logic rather than a model call.
- If the canonical trajectory lacks usable references or complete model-call evidence, binding-dependent checks are unobserved.
- An explicit reference that resolves to no model call or more than one model call produces a `trajectory_capture_mismatch` finding only when the canonical trajectory does not say that model-call evidence is incomplete.
- The observability gap codes `model_call_reference_unmatched`, `model_call_reference_ambiguous`, and `model_call_reference_conflict` are also explicit contradictions and produce `trajectory_capture_mismatch` findings.
- Only calls that resolve exactly once become bound policy-model calls.

**Alternatives considered.** Read raw capture files again, match by list position, treat every model-call entry as a policy-model call, or fail every turn without a reference. These choices duplicate observability work, can assign auxiliary calls to policy turns, or turn missing evidence into false rollout failures.

### 5. Separating correspondence, token, and model-call failures

**RFC gap.** The RFC's correspondence family combines cross-artifact mismatches, missing token fields, token-total mismatches, failed calls, and retries under one check. These findings have different subjects and different missing-input behavior.

**Chosen behavior.** The implementation separates them into checks whose IDs begin with their finding subject:

- `trajectory_capture_mismatch` reports explicit reference contradictions recorded in or derived from the canonical trajectory. Extra unreferenced model-call entries do not fail because they may be auxiliary calls.
- `model_call_missing_token_counts` reports a bound call missing prompt-token or completion-token counts. With no bound calls, it is unobserved. Zero is a present count and belongs to `model_call_zero_completion_tokens`.
- `rollout_token_count_mismatch` compares complete top-level rollout totals with the sum of a complete set of bound calls. Missing totals, missing per-call counts, or incomplete bindings make it unobserved.
- `model_call_failed` reports one finding per bound call with HTTP status 400 through 599, a recorded error category, or response status `failed`, `error`, or `cancelled`. Its detail records whether that call was the final bound call. Unreferenced failed model-call entries are not attributed to the policy model.
- `model_call_zero_completion_tokens` and `model_call_runaway_generation` also evaluate only bound policy-model calls.

Run statistics count all `TrajectoryRecord.model_calls` because they describe the canonical artifact rather than assigning policy ownership. The report field names containing `capture` are retained because the RFC fixes those schema names; health does not read capture sidecars.

**Alternatives considered.** Keep one overloaded check, sum all capture calls, or treat every unreferenced failure as a dropped policy retry. Separate checks make each ID independently understandable and ignorable, while bound-call evaluation avoids attributing judge or user-simulator failures to the policy.

### 6. Deciding whether a length-limited response is empty

**RFC gap.** `model_call_runaway_generation` applies when a call ends because of the length limit and has empty content, but model providers serialize response content differently.

**Chosen behavior.** The runner looks for non-empty text, content, output text, answer, encrypted content, reasoning, or reasoning summary in OpenAI Responses output, Chat Completions choices, and Messages-style content. Any of those fields means the response is not empty.

**Alternative considered.** Implement separate content rules for each model provider's response format. That would make results depend on provider-specific code paths rather than one rule over saved artifacts.

## Verdicts and task aggregation

### 7. Combining check results into one rollout verdict

**RFC gap.** The RFC explains how individual checks are evaluated but does not define how their results become the rollout's single verdict.

**Chosen behavior.** Verdict priority is:

1. `unhealthy` when any enabled check produced a finding.
2. Otherwise `unobserved` when any enabled check lacked required input.
3. Otherwise `healthy`.

This makes `healthy` a strong claim that every enabled rollout check ran and passed.

**Alternatives considered.** Call a rollout healthy when every computable check passes, even if other checks lack evidence, or omit the overall verdict whenever coverage is partial.

### 8. Inputs to `task_consistently_unhealthy`

**RFC gap.** The RFC says this task-level check uses “computable repeat verdicts” but does not define that phrase.

**Chosen behavior.** A computable repeat is a rollout whose verdict is `healthy` or `unhealthy`, not `unobserved`. The task check runs when at least two repeats are computable and produces a finding when all computable repeats are unhealthy. Additional unobserved repeats do not prevent the finding; this follows R4.

**Alternative considered.** Require every repeat for the task to be observed before evaluating the task check.

### 9. Inputs to `task_no_successful_model_calls`

**RFC gap.** The RFC does not say whether this task-level check may draw a conclusion from only the repeats that have model-call evidence.

**Chosen behavior.** Every repeat must have a complete set of explicit policy-call bindings in its canonical trajectory. A list of model calls without turn references is insufficient because it may contain auxiliary calls. If any repeat lacks complete bindings, this task check is unobserved. When every repeat is observed, the check produces a finding only if none contains a successful bound call.

**Alternative considered.** Evaluate only the observed repeats. That could report failure even though an unobserved repeat contained a successful call.

### 10. Whether tasks receive their own verdict

**RFC gap.** The report schema gives each task repeat counts and task-level flags but defines no task verdict field.

**Chosen behavior.** Task entries contain only `repeats`, `healthy`, `unhealthy`, `unobserved`, and `flags`, exactly as specified.

**Alternative considered.** Add a task verdict. That would extend the report schema without a rule for deriving it.

### 11. Meaning of the `issues` counts

**RFC gap.** The RFC calls `run.issues` a histogram—a mapping from check IDs to counts—but does not say whether those numbers count findings or affected rollouts and tasks.

**Chosen behavior.** Each number counts individual `Finding` objects. The histogram contains every registered check ID, including IDs with a zero count.

**Alternatives considered.** Count affected rollouts and tasks once per check, or omit check IDs whose count is zero.

## Artifact discovery and failure handling

### 12. Finding the canonical observability input

**RFC gap.** The RFC describes rollout records and separate capture artifacts, but current Gym already combines its observability evidence into a final `TrajectoryRecord` before persisting each rollout.

**Chosen behavior.** Standalone, post-run, and post-aggregate health checks all read exactly one observability source: `ng_trajectory` inside each rollout record. The CLI has no capture-directory option, and the health runner never discovers or opens model-call sidecars. The collection observability code may still use sidecars internally before it creates the final trajectory; that is outside the health workflow.

**Alternatives considered.** Let health rediscover sidecars, read `ng_model_call_capture`, or choose among several representations. Multiple sources require precedence rules, can disagree after files move, and duplicate normalization and correlation already owned by the observability stack.

### 13. Translating observation gaps into health behavior

**RFC gap.** `TrajectoryRecord.gaps` records evidence that observability could not collect or join exactly, but the RFC does not say which gaps mean missing input and which describe an affirmative contradiction.

**Chosen behavior.** The observability stack records facts; the health runner alone derives findings and verdicts from them.

- `trajectory_projection_failed` makes every trajectory-dependent check unobserved.
- `turns_unavailable` makes turn-dependent checks unobserved.
- `model_calls_unavailable`, `model_call_capture_incomplete`, `model_call_capture_records_unreadable`, and `model_call_capture_unreadable` make binding-dependent checks unobserved.
- `model_call_reference_unmatched`, `model_call_reference_ambiguous`, and `model_call_reference_conflict` are explicit contradictions and produce `trajectory_capture_mismatch` findings.
- `model_call_ownership_unavailable` does not fail a check by itself. An unreferenced call may belong to a judge, user simulator, or other auxiliary model.
- Other gap codes are not silently turned into rollout-health failures. A check uses them only when it has an explicit rule.

This preserves the difference between “the evidence is absent” and “the available evidence disagrees.”

**Alternatives considered.** Treat every gap as unhealthy, ignore all gaps, or add a health check for every observability code. Those choices respectively create false failures, hide missing-input states, or make the health registry an accidental mirror of observability internals.

### 14. Recording why a check is unobserved

**RFC gap.** The canonical trajectory can explain why evidence is missing through `ObservationGap`, while `rollout_verdicts.jsonl` allows only a list of unobserved check IDs, not a reason for each ID.

**Chosen behavior.** The runner reads the relevant gap codes while deciding whether each check has input, then writes only the check IDs allowed by the RFC schema. The original reasons remain inspectable in the rollout's `ng_trajectory.gaps`.

**Alternative considered.** Copy gap reasons into every verdict row. That would duplicate evidence and change the specified report schema.

### 15. Custom collection drivers

**RFC gap.** A custom collection driver may or may not participate in Gym observability, and the RFC does not define a separate persisted driver-health contract.

**Chosen behavior.** Health has no driver-specific branch. If a driver produces a valid final `ng_trajectory`, its available checks run. If it does not, the dependent checks are unobserved. Files adjacent to the rollout are never used to infer observability.

**Alternative considered.** Mark every custom driver unobserved by configuration alone. That would ignore valid canonical trajectories produced by conforming drivers.

### 16. Finding the rollout JSONL from a run directory

**RFC gap.** The standalone command receives a directory, but the RFC does not require a rollout filename.

**Chosen behavior.** The command reads `<run-dir>/rollouts.jsonl` by default and performs no filename discovery. A nonstandard path must be supplied explicitly with `--rollouts-file PATH`; relative paths resolve under the run directory and absolute paths are used as written. If the selected file does not exist, the command fails even when another JSONL is present.

**Alternative considered.** Infer a rollout file when exactly one plausible JSONL is present. Even unique-looking files can be prepared inputs or unrelated outputs, so inference could silently check the wrong artifact.

### 17. Health checking explicitly selected rollout shards

**RFC gap.** `gym eval aggregate` can run with `merge_shards=false`. In that mode, aggregation does not create the merged rollout file named by `output_jsonl_fpath`, but the RFC still requires an automatic health check.

**Chosen behavior.** The aggregate command passes the rollout shard paths explicitly selected by `input_glob` to the health runner. The runner checks those files as one logical run and writes one combined health report in the directory containing `output_jsonl_fpath`. It does not search for additional JSONL files and does not create a merged rollout file.

This does not affect the standalone command: `gym eval health-check` still requires `<run-dir>/rollouts.jsonl` or an explicit `--rollouts-file` argument.

**Alternatives considered.** Create a merged rollout file despite `merge_shards=false`, or omit the automatic health check.

### 18. Reporting an unreadable rollout record

**RFC gap.** The RFC says parsing must be tolerant and an unparseable record is a finding. It defines neither an issue ID for that finding nor a fallback identity for the report row.

**Chosen behavior.** The implementation adds the separate `record_unreadable` check ID. Because an unreadable entry cannot supply its real rollout identity, the report uses a clearly synthetic `_ng_task_index` containing the input-file position and physical line number, such as `__unreadable_record__:input-0:line-42`; `_ng_rollout_index` is `0`. The finding locator records the selected source-file path and line number. Checks that need the parsed record are marked unobserved, preventing meaningless follow-on findings. The synthetic namespace prevents an unreadable entry from being grouped with an ordinary numeric task index.

**Alternatives considered.** Report the parse failure under an unrelated semantic check, or omit the line from reports. The first corrupts another check's issue count; the second violates the requirement to classify every rollout record.

### 19. Running where process pools are unavailable

**RFC gap.** The execution model requires multiple worker processes and also says verification must always complete. Some operating systems cannot start Python's `ProcessPoolExecutor` because required process-synchronization support is unavailable.

**Chosen behavior.** The runner normally uses `ProcessPoolExecutor`. If pool creation or execution fails for a platform reason, it emits a warning and processes the same inputs serially.

**Alternative considered.** Fail the health check and therefore fail post-run aggregation after aggregate metrics were already produced.

### 20. Handling a check execution error

**RFC gap.** The RFC requires tolerant parsing but does not define how to report an unexpected exception raised while evaluating one check.

**Chosen behavior.** Expected missing or unsupported input makes the affected check unobserved. If a check nevertheless raises an unexpected exception, the runner emits a separate `check_execution_error` finding that identifies the failed check and exception type, marks that check unobserved, and continues evaluating the remaining checks. `record_unreadable` remains reserved for records or canonical trajectories that cannot be parsed.

**Alternatives considered.** Report the exception as `record_unreadable`, which incorrectly blames a successfully parsed record, or abort verification before the other checks can run.

### 21. Duplicate task and rollout identities

**RFC gap.** Reports are keyed by `_ng_task_index` and `_ng_rollout_index`, but the RFC does not say what to do when the input JSONL contains the same pair more than once.

**Chosen behavior.** Every non-empty input line still receives a rollout verdict and contributes to run-level counts. Before task-level checks and repeat counts, records with the same identity are collapsed into one repeat:

- Copies with the same verdict produce that verdict once.
- Copies with conflicting verdicts make that repeat unobserved for task-level evaluation.

**Alternatives considered.** Reject the run, keep only the first or last copy everywhere, or classify identity collisions as `record_unreadable`. Validation found that counting duplicate lines as separate repeats could create a false `task_consistently_unhealthy` finding.

### 22. Running only a selected subset of checks

**RFC gap.** The RFC defines a fixed check registry and a whole-run `--no-health-check` option, but not a way to exclude individual checks.

**Chosen behavior.** All checks remain registered and enabled by default. Operators may explicitly ignore known check IDs for one run. An ignored check does not execute and does not affect findings, unobserved states, rollout verdicts, or task flags. `quality_summary.json` records sorted IDs in `run.ignored_checks` and records an `ignored` count in each check's coverage. `rollout_verdicts.jsonl` keeps the RFC schema, so readers must use the summary to know that verdicts came from a reduced check set.

**Alternatives considered.** Remove checks from the registry, treat ignored checks as unobserved, or silently change the default check set. This extension exists for explicit validation of historical corpora and does not weaken default production behavior.
