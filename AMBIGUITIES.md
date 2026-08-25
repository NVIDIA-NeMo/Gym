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
- **Artifact:** A file saved by an evaluation run, such as rollout records, aggregate metrics, model-call captures, or health reports.
- **JSONL:** A text file containing one complete JSON value per line. Gym stores one rollout record per non-empty line.
- **Rollout record:** One JSON object in the rollout JSONL file. It contains the task and repeat identity, the agent result, verifier output, and any observability data attached during collection.
- **Observability data:** Evidence recorded so a completed run can be inspected, such as conversations, model calls, token counts, status codes, and timing. It does not change the rollout or its score.
- **Transcript:** Saved conversation evidence for the user, agent, and tools. New Gym records standardize this evidence in `ng_trajectory`.
- **Agent turn:** One unit of agent activity. It may include reasoning, an assistant message, and one or more function calls. A user message or function-call output starts the next interaction.
- **Agent invocation:** One execution of a root agent or subagent. An invocation can contain several conversation items, turns, tool calls, and model calls. The code type is `AgentInvocation`.
- **Model call:** One request sent to a language model and its response.
- **Model-call capture:** Saved request, response, token, status, and timing evidence for model calls.
- **Sidecar:** A separate file associated with a rollout. Raw model-call capture usually lives in a `*.capture.jsonl` sidecar.
- **Embedded projection:** A normalized copy of capture data stored inside the rollout record, in `ng_trajectory.model_calls` or `ng_model_call_capture`.
- **Model-call reference:** An identifier that connects transcript evidence to a captured model call. `ModelCallRef` accepts either `model_call_id` or the pair `model_ref` and `response_id`.
- **Binding:** A successful match between a transcript model-call reference and a captured model call.
- **Correlation:** The information needed to make bindings between transcript activity and captured model calls.
- **Policy model:** The model being evaluated.
- **Auxiliary model:** A model used to support the evaluation, such as a user simulator or judge. Its calls may appear in the same capture as policy-model calls.
- **Check:** One registered health rule. `CheckSpec` gives it a stable ID, an evaluation scope, a finding subject, and required inputs. Evaluation scope says how much data the rule reduces: one rollout, one task across repeats, or the run as a whole. Finding subject says what the evidence identifies, such as an agent turn or model call.
- **Finding:** Evidence that a check failed for a rollout or task. A check emits `Finding`; it never emits a verdict.
- **Healthy:** Every enabled rollout check had enough input to run, and none produced a finding.
- **Unhealthy:** At least one enabled rollout check produced a finding.
- **Unobserved:** No enabled rollout check produced a finding, but at least one could not run because required evidence was missing.
- **Driver bypass:** A custom collection path that does not use Gym's normal model-server capture and correlation path.
- **Shard:** One part of a larger rollout file. Separate workers may write shards that are later aggregated.

## Transcript and check behavior

### 1. Missing canonical agent turns

**RFC gap.** Current Gym stores agent turns in `TrajectoryRecord.turns`. Some records do not contain those turns, and the RFC does not say whether turn-based checks should reject them, become unobserved, or use weaker evidence.

**Chosen behavior.** `ng_trajectory.turns` is the canonical input. The runner retains two compatibility fallbacks, but labels their turn-based results as best-effort with one aggregated warning per run. The warning reports how many records used each fallback. Sources are tried in this order and never combined:

1. `ng_trajectory.turns`: each `TrajectoryTurn` is one agent turn.
2. `ng_trajectory.invocations`: this is current Gym data, but it is coarser than turns. Each `AgentInvocation` with conversation evidence is treated as one unit. An invocation may contain several messages and model calls. Its `model_calls` list is stored on the `AgentInvocation`, not on individual conversation items, so the runner cannot safely determine which call produced each message.
3. `response.output`: this legacy fallback reconstructs turns by grouping adjacent reasoning, assistant-message, and function-call items. A user message or function-call output ends the group.

For example, this `response.output` sequence contains two agent turns:

```text
reasoning
assistant message
function call
function-call output  # ends the first agent turn
assistant message     # second agent turn
```

For structural health checks, each selected turn or invocation is reduced to three facts: whether it contains non-empty text or reasoning, whether it contains a tool call, and which model-call references it contains. The record has agent activity when at least one selected unit contains any of those facts. Model-call checks do not use these compatibility fallbacks: they require explicit references from canonical `TrajectoryTurn.model_calls` and otherwise become unobserved.

**Alternatives considered.** Mark turn-based checks unobserved whenever `TrajectoryRecord.turns` is missing, reject such records, or treat every `response.output` item as a separate turn. Treating each item separately produced false `agent_turn_hollow` findings when an empty assistant message and its real function call were sibling items in the same model response.

### 2. Whether reasoning counts as message content

**RFC gap.** `agent_turn_hollow` means “no message and no tool calls,” but the RFC does not say whether a reasoning-only turn has a message.

**Chosen behavior.** Any non-empty answer, content, text, output text, reasoning content, reasoning summary, or encrypted reasoning content means the turn is not hollow. A tool call also means the turn is not hollow.

**Alternative considered.** Count only user-visible answer text. That would mark reasoning-only model activity as hollow.

### 3. Deterministic dispatch and zero-token calls

**RFC gap.** The RFC exempts deterministic-dispatch steps from `model_call_zero_completion_tokens` without explaining the term.

**Chosen behavior.** A deterministic dispatch is agent logic that produces a step without calling the model. Because `model_call_zero_completion_tokens` examines explicitly bound policy-model calls, such a step has no bound call and is automatically outside this check. Every bound call reporting zero completion tokens produces a finding. Unowned capture entries are not evaluated because they may belong to a user simulator or judge. The runner does not infer an exemption from request parameters or metadata.

**Alternative considered.** Add or infer a special marker on captured calls. No such marker exists in the specified artifacts, so doing this would invent a new contract.

### 4. Binding trajectory turns to captured model calls

**RFC gap.** The model-call checks need to know which captured calls belong to the policy trajectory, but the RFC does not define how to make that connection or distinguish absent evidence from contradictory evidence.

**Chosen behavior.** The runner creates one shared binding result from canonical `TrajectoryTurn.model_calls`. It accepts only an explicit `ModelCallRef`: either `model_call_id`, or both `model_ref` and `response_id`. It never matches by position, payload similarity, or model name. All policy-model-call checks consume the same bound calls.

- A turn without a reference does not produce a finding. It may be deterministic agent logic rather than a model call.
- If the trajectory contains no usable references, binding-dependent checks are unobserved.
- An explicit reference that resolves to no captured call or more than one captured call produces `trajectory_capture_mismatch` findings because the two artifacts contradict each other.
- Only calls that resolve exactly once become bound policy-model calls.

**Alternatives considered.** Match by list position, treat every capture entry as a policy-model call, or fail every turn without a reference. These choices can connect auxiliary calls to policy turns or turn missing observability into false rollout failures.

### 5. Separating correspondence, token, and model-call failures

**RFC gap.** The RFC's correspondence family combines cross-artifact mismatches, missing token fields, token-total mismatches, failed calls, and retries under one check. These findings have different subjects and different missing-input behavior.

**Chosen behavior.** The implementation separates them into checks whose IDs begin with their finding subject:

- `trajectory_capture_mismatch` reports explicit references that resolve to zero or multiple captured calls. An unreadable capture line is also an affirmative artifact defect under this check. Extra unreferenced capture entries do not fail because they may be auxiliary calls.
- `model_call_missing_token_counts` reports a bound call missing prompt-token or completion-token counts. With no bound calls, it is unobserved. Zero is a present count and belongs to `model_call_zero_completion_tokens`.
- `rollout_token_count_mismatch` compares complete top-level rollout totals with the sum of a complete set of bound calls. Missing totals, missing per-call counts, or incomplete bindings make it unobserved.
- `model_call_failed` reports one finding per bound call with HTTP status 400 through 599, a recorded error category, or response status `failed`, `error`, or `cancelled`. Its detail records whether that call was the final bound call. Unreferenced failed capture entries are not attributed to the policy model.
- `model_call_zero_completion_tokens` and `model_call_runaway_generation` also evaluate only bound policy-model calls.

Raw run statistics remain unchanged: they count all stored capture calls because they describe the artifact rather than assigning policy ownership.

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

### 9. Inputs to `task_no_healthy_model_calls`

**RFC gap.** The RFC does not say whether this task-level check may draw a conclusion from only the repeats that have capture evidence.

**Chosen behavior.** Every repeat must have a complete set of explicit policy-call bindings. Raw capture alone is insufficient because it may contain auxiliary calls. If any repeat lacks complete bindings, this task check is unobserved. When every repeat is observed, the check produces a finding only if none contains a successful bound call.

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

### 12. Finding capture files from the standalone command

**RFC gap.** `gym eval health-check <run-dir>` accepts only the run directory. Capture output can be configured at an absolute path, and the RFC defines no manifest that records that path.

**Chosen behavior.** The standalone command recursively searches the supplied run directory for `*.capture.jsonl`. Automatic execution after `run` or `aggregate` receives the configured capture directory directly and does not need discovery.

**Alternatives considered.** Add another standalone CLI argument not defined by the RFC, or make every capture-dependent check unobserved in standalone mode.

### 13. Using model-call evidence embedded in a rollout

**RFC gap.** A rollout may retain model-call evidence inside `ng_trajectory` or `ng_model_call_capture` after raw sidecars have moved, or when sidecar filenames do not match the rollout identity. The RFC does not state which source wins.

**Chosen behavior.** A sidecar whose filename matches the rollout is preferred. If none matches, a non-empty embedded model-call projection counts as observed capture. An embedded but empty call list does not prove that the rollout used Gym's capture path, so capture-dependent checks remain unobserved.

**Alternatives considered.** Ignore embedded evidence, or treat an empty embedded list as proof of a successfully captured rollout with zero calls. The latter produced false `task_no_healthy_model_calls` findings for collection drivers that bypass Gym's model server.

### 14. Recording why a check is unobserved

**RFC gap.** The RFC distinguishes capture disabled, an agent that cannot correlate calls, and driver bypass. Its `rollout_verdicts.jsonl` schema allows only a list of unobserved check IDs, not a reason for each ID.

**Chosen behavior.** The runner distinguishes these causes while deciding whether input exists, then writes only the check IDs allowed by the schema.

**Alternative considered.** Add an unobserved-reasons object to each verdict row. That would change the specified report schema.

### 15. Treating custom collection drivers as capture bypass

**RFC gap.** None. The RFC's state table says a custom collection driver bypasses the standard correlation path. The remaining choice is whether to trust capture-looking files found beside such a run.

**Chosen behavior.** Automatic execution marks all capture-dependent checks unobserved when a custom collection driver is configured, even if files named like captures are present.

**Alternative considered.** Evaluate those files opportunistically. Their presence does not prove they follow Gym's standard rollout-correlation contract.

### 16. Finding the rollout JSONL from a run directory

**RFC gap.** The standalone command receives a directory, but the RFC does not require a rollout filename.

**Chosen behavior.** The command first looks for `rollouts.jsonl`. If absent, it accepts exactly one top-level JSONL after excluding prepared input files whose names contain `materialized`, failure files, capture files, and `rollout_verdicts.jsonl`. More than one candidate is an error rather than an arbitrary choice.

**Alternative considered.** Select the first candidate by filename order. That could silently check the wrong artifact.

### 17. Running after aggregation without shard merging

**RFC gap.** `gym eval aggregate` can run with `merge_shards=false`, while the RFC requires automatic health checks after aggregation but does not say whether health checking requires a merged rollout file.

**Chosen behavior.** The runner reads each selected shard directly and writes one combined health report beside the requested aggregate output. It does not create a merged rollout artifact.

**Alternatives considered.** Create an unrequested merged JSONL, or skip the required health check when shard merging is disabled.

### 18. Reporting an unreadable rollout record

**RFC gap.** The RFC says parsing must be tolerant and an unparseable record is a finding. It defines neither an issue ID for that finding nor a fallback identity for the report row.

**Chosen behavior.** The implementation adds the separate `record_unreadable` check ID. For an unreadable JSONL entry, its zero-based position among all non-empty input lines becomes `_ng_task_index` and `_ng_rollout_index` becomes `0`. Checks that need the parsed record are marked unobserved, preventing meaningless follow-on findings.

**Alternatives considered.** Report the parse failure under an unrelated semantic check, or omit the line from reports. The first corrupts another check's issue count; the second violates the requirement to classify every rollout record.

### 19. Running where process pools are unavailable

**RFC gap.** The execution model requires multiple worker processes and also says verification must always complete. Some operating systems cannot start Python's `ProcessPoolExecutor` because required process-synchronization support is unavailable.

**Chosen behavior.** The runner normally uses `ProcessPoolExecutor`. If pool creation or execution fails for a platform reason, it emits a warning and processes the same inputs serially.

**Alternative considered.** Fail the health check and therefore fail post-run aggregation after aggregate metrics were already produced.

### 20. Handling a check that cannot parse one of its fields

**RFC gap.** The RFC requires tolerant parsing but does not say how to distinguish a health finding from a check implementation failing on malformed input.

**Chosen behavior.** The runner catches the check failure, emits a `record_unreadable` finding whose detail names the affected check, and marks that check unobserved. Other checks continue.

**Alternatives considered.** File the exception under the affected semantic check, which would pollute that check's issue histogram, or abort all verification.

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
