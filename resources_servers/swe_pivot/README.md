# SWE Pivot Verifier

Verifier for PivotRL training on coding agent trajectories. Scores rollout actions at pivot turns — critical decision points where the model's choice determines fix success or failure.

## Background

PivotRL (arXiv:2603.21383) trains on intermediate turns from SFT trajectories where sampled actions exhibit high reward variance. Instead of full end-to-end rollouts, it does local single-turn rollouts at these pivots and uses GRPO to upweight correct actions.

This verifier provides the reward signal for those rollouts. It checks whether the model's rollout action is functionally equivalent to the demonstrated (correct) action from the SFT trajectory.

## End-to-End Pipeline

```
 SFT Trajectories (74,925)
         |
         | filter_difficult_patches.py
         | (>= 2 files, > 25 changed lines)
         v
 Difficult Patches (10,096)
         |
         | convert_to_minimax_format.py
         | (merge reasoning_content, list content, add tool name)
         v
 Formatted Data (10,096 all / 4,395 OpenHands)
         |
         | judge_pivots.py
         | (LLM-as-judge, N runs majority vote, 6 failure patterns)
         v
 Pivot Samples (~11K est, ~6% of turns)
         |
         | split into train.jsonl / validation.jsonl
         v
 +-----------------------+        +------------------+
 |  NeMo Gym Training    |        |  SWE Pivot       |
 |  (GRPO on pivots)     | -----> |  Verifier        |
 |                       |  POST  |  (this server)   |
 |  For each pivot:      | /verify|                  |
 |  - Sample G rollouts  | -----> |  Returns reward   |
 |  - Score with verifier|        |  0.0 - 1.0       |
 |  - Compute advantage  |        |                  |
 |  - Update policy      |        |                  |
 +-----------------------+        +------------------+
```

## Verification Flow

```
           Rollout action (model output)
           Expected action (SFT demonstration)
                        |
                        v
              +-------------------+
              | Level 0: Tool     |     "Is it using the right kind of tool?"
              | Name Match        |     (edit vs bash vs search vs read)
              +-------------------+
                   |         |
                 PASS       FAIL --> reward = 0.0 (TOOL_NAME_MISMATCH)
                   |
                   v
              +-------------------+
              | Level 1: Target   |     "Is it operating on the correct file?"
              | Match             |     (basename match for edits, verb match for bash) # regex based file paths
              +-------------------+
                   |         |
                 PASS       FAIL --> reward = 0.0 (TARGET_MISMATCH)
                   |
                   v
              +-------------------+
              | Level 2: Argument |     "Is the edit content similar?"
              | Similarity        |     (old_str and new_str checked independently,
              |                   |      both must pass threshold, capped at 2K chars)
              +-------------------+
                   |         |
                 PASS       FAIL --> reward = 0.0 (SIMILARITY_BELOW_THRESHOLD)
                   |
                   v
              +-------------------+
              | Level 4: Diff-    |     "Is the edit minimal?"
              | Size Shaping      |     (small penalty for bloated edits)
              +-------------------+     (continuous mode only)
                   |
                   v
              reward = 0.0 - 1.0
```

## Reward Modes

Configurable via `reward_mode` in the server config.

### Binary Mode (`reward_mode: "binary"`)

```
                  Level 0       Level 1         Level 2
 Rollout ------> tool name ---> target file ---> similarity >= 0.8?
                  match?         match?              |        |
                  |    |         |    |             YES       NO
                  NO   YES      NO   YES            |        |
                  |              |                   v        v
                  v              v               reward=1  reward=0
              reward=0       reward=0
```

Clean 0/1 signal. No partial credit, no diff-size shaping. Best for standard GRPO where binary rewards produce the clearest advantage signal. The PivotRL paper uses binary rewards for SWE-Bench.

### Continuous Mode (`reward_mode: "continuous"`, default)

```
 After passing Level 0 + Level 1:

 similarity score
     |
     |  0.0          0.4           0.8           1.0
     |   |____________|_____________|_____________|
     |   |  reward=0  |  0.5 - 1.0  |  1.0       |
     |   |            |  (linear)    |  (full)    |
     |   |            |              |            |
     |   +-- FAIL     +-- partial    +-- PASS     |
     |                    credit                  |
     v
 Then apply diff-size shaping:
     reward = reward * (1.0 - alpha * min(edit_size / 500, 1.0))
```

Richer signal with partial credit for similar-but-not-identical actions and a small penalty for bloated edits. Best when you want the model to learn fine-grained distinctions between good and great actions.

## Verification Levels (Detail)

### Level 0: Tool Name Match (always on)

Checks whether the rollout uses the same **tool category** as the expected action. Handles diversified tool names across three agent frameworks:


| Category   | Tool Name Variants (63 total names covered)                                                                                                                                                                                                                |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `edit`     | `str_replace_editor`, `code_editor`, `text_editor`, `file_editor`, `edit_file`, `modify_file`, `update_file`, `file_edit`, `edit`, `modify`, `apply_patch`, `patch_file`, `apply_diff`, `apply_changes`, `write`, `write_file`, `save_file`, `create_file` |
| `bash`     | `execute_bash`, `run_bash`, `bash_exec`, `shell_run`, `shell`, `bash`, `terminal`, `shell_command`, `shell_cmd`, `run_shell_cmd`, `exec_command`, `exec_cmd`                                                                                               |
| `search`   | `grep`, `search_text`, `find_pattern`, `text_search`, `glob`, `find_files`, `file_glob`, `match_files`, `grep_files`, `search_files`, `find_in_files`, `code_search`, `list_dir`, `ls`, `show_dir`, `ls_dir`, `list_directory`                             |
| `read`     | `read`, `read_file`, `view_file`, `cat_file`, `read_file_content`                                                                                                                                                                                          |
| `planning` | `task_tracker`, `todo_tracker`, `plan_tracker`, `todo_read`, `todo_write`, `update_plan`, `list_tasks`, `update_tasks`, `read_todos`, `write_todos`, `edit_plan`, `modify_plan`                                                                            |
| `finish`   | `finish`, `submit`                                                                                                                                                                                                                                         |
| `think`    | `think`                                                                                                                                                                                                                                                    |


All names also match their CamelCase variants (e.g. `StrReplaceEditor`, `ExecuteBash`).

### Level 1: Target Match (`enable_target_match`)


| Tool Category   | What's Compared                    | Example                                                                                 |
| --------------- | ---------------------------------- | --------------------------------------------------------------------------------------- |
| `edit`          | File basename + editor sub-command | `/workspace/src/models.py:str_replace` vs `/workspace/src/views.py:str_replace` -> FAIL |
| `edit` (view)   | File basename + view_range overlap | `view foo.py [1,50]` vs `view foo.py [200,250]` -> FAIL (no overlap)                   |
| `bash`          | Command verb + path basenames      | `grep -r 'foo' src/models/` vs `grep -r 'foo' src/views/` -> FAIL (different path)     |
| `search`/`read` | Target path basename               | `grep` on `src/` vs `grep` on `src/` -> PASS                                            |

Note on `view_range`: 70% of view commands include a `[start_line, end_line]` range. If both rollout and expected have a range, they must overlap — viewing a completely different section of the file means the model sees different code, leading to different hypotheses. If either side has no range (viewing the whole file), the check passes.

Note on bash paths: File/directory paths are extracted from shell commands via regex and compared by basename. `grep -r 'foo' src/models/` vs `grep -r 'foo' src/views/` fails because `models` != `views`.

Note on `security_risk`: This OpenHands-specific parameter is present in many tool arguments but is **not compared** at any level. Only `command`, `path`, `view_range`, `old_str`, and `new_str` are checked.


This catches the **consumer-vs-producer** failure pattern (P1) — editing the wrong file gets reward 0.

### Level 2: Argument Similarity (`enable_argument_similarity`)

Uses `difflib.SequenceMatcher` on the edit/command content, capped at 2,000 characters to avoid O(n*m) blowup on large edits (affects only ~9% of samples).

For `str_replace` edits, `old_str` and `new_str` are compared **independently**:

```
old_str similarity = SequenceMatcher(rollout.old_str, expected.old_str)  --> "right location?"
new_str similarity = SequenceMatcher(rollout.new_str, expected.new_str)  --> "right fix?"

similarity = min(old_str_sim, new_str_sim)   --> both must pass
```

Both must independently meet the threshold — getting the location right but writing the wrong fix (or vice versa) results in low similarity.

| Tool Category | What's Compared |
| ------------- | --------------- |
| `edit` (str_replace) | `old_str` and `new_str` separately, take min |
| `edit` (apply_patch) | Full patch/diff content |
| `edit` (write/create) | File content |
| `bash` | Full shell command string |
| Other | Returns 1.0 (not applicable) |

This catches **over-engineering** (P4) and **execution errors** (P6).

### Level 4: Diff-Size Shaping (`enable_diff_size_shaping`)

Applies a small penalty for large edits, rewarding minimality:

```
penalty = alpha * min(edit_content_length / 500, 1.0)
reward  = reward * (1.0 - penalty)
```

Default `alpha = 0.1`, so the max penalty is 10%. Only applies in continuous mode and only to edit tool calls. This is a tiebreaker among correct actions — directly addresses the over-engineering pattern.

## Pivot Tags and Failure Patterns

Each pivot sample carries tags indicating which failure pattern it targets. The verifier doesn't branch on tags — all levels apply uniformly. But the tags explain WHY each level matters:

```
 P1 Consumer-vs-Producer (39%)  -----> Level 1 catches this
     Agent edits symptom site            (wrong file = reward 0)
     instead of root cause

 P2 Backs Off Correct Fix (14%) -----> Level 2 catches this
     Agent weakens fix after             (changed content = low similarity)
     test failure

 P3 Incomplete Fix (12%)        -----> Level 0/1 catches this
     Agent stops before checking         (search vs finish = wrong tool)
     sibling code paths

 P4 Over-Engineering (21%)      -----> Level 2 + Level 4 catches this
     Agent builds elaborate fix          (low similarity + diff-size penalty)
     when 1-3 lines suffice

 P5 Hypothesis Roulette (68%)   -----> Level 1 or 2 catches this
     Agent picks wrong fix               (depends on whether hypothesis
     strategy post-localization           manifests as file or content choice)

 P6 Execution Errors (18%)      -----> Level 2 catches this
     Correct file, botched               (similar but not identical content)
     implementation
```

## Configuration

### Server Config

```yaml
swe_pivot:
  resources_servers:
    swe_pivot:
      entrypoint: app.py
      domain: agent
```

### Config Options (`SwePivotResourcesServerConfig`)


| Option                       | Default        | Description                                             |
| ---------------------------- | -------------- | ------------------------------------------------------- |
| `reward_mode`                | `"continuous"` | `"binary"` (0 or 1) or `"continuous"` (0.0 to 1.0)      |
| `enable_target_match`        | `true`         | Level 1: check target file/command                      |
| `enable_argument_similarity` | `true`         | Level 2: SequenceMatcher on args                        |
| `similarity_threshold`       | `0.4`          | Minimum similarity for partial credit (continuous only) |
| `similarity_full_credit`     | `0.8`          | Similarity threshold for full credit                    |
| `enable_diff_size_shaping`   | `true`         | Level 4: penalize large edits (continuous only)         |
| `diff_size_alpha`            | `0.1`          | Max diff-size penalty                                   |


### Recommended Configs

**Conservative (PivotRL paper style)**:

```python
reward_mode = "binary"
enable_target_match = False           # Only tool name match, like the paper
enable_argument_similarity = False
enable_diff_size_shaping = False
```

**Balanced (recommended)**:

```python
reward_mode = "binary"
enable_target_match = True
enable_argument_similarity = True
similarity_full_credit = 0.8
enable_diff_size_shaping = False      # Binary mode ignores this anyway
```

**Fine-grained**:

```python
reward_mode = "continuous"
enable_target_match = True
enable_argument_similarity = True
similarity_threshold = 0.4
similarity_full_credit = 0.8
enable_diff_size_shaping = True
diff_size_alpha = 0.1
```

## Data Format

Input data is the output of `judge_pivots.py`. Each sample:

```json
{
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      {"type": "reasoning", "summary": [...]},
      {"role": "assistant", "content": [...], "type": "message"},
      {"type": "function_call", "name": "execute_bash", "arguments": "..."},
      {"type": "function_call_output", "output": "...", "call_id": "..."}
    ],
    "tools": [...],
    "parallel_tool_calls": true
  },
  "expected_action": {
    "type": "function_call",
    "name": "str_replace_editor",
    "arguments": "{\"command\": \"str_replace\", \"path\": \"/workspace/foo.py\", ...}"
  },
  "ref_patch": "diff --git a/foo.py b/foo.py\n...",
  "pivot_tags": ["P1", "P5"],
  "agent_ref": {
    "type": "responses_api_agents",
    "name": "single_step_tool_use_with_argument_comparison_agent"
  }
}
```

The verifier receives `expected_action` via `expected_answer` or `metadata.expected_action` and compares it against the model's rollout `response`.

## Verify Endpoint

**Request**: `POST /verify`

```json
{
  "responses_create_params": {"input": [...], "tools": [...]},
  "response": {"output": [{"type": "function_call", "name": "str_replace_editor", "arguments": "..."}]},
  "expected_answer": "{\"type\": \"function_call\", \"name\": \"str_replace_editor\", \"arguments\": \"...\"}"
}
```

**Response**:

```json
{
  "reward": 0.85,
  "tool_name_match": true,
  "target_match": true,
  "similarity_score": 0.92,
  "diff_size_penalty": 0.03,
  "failure_reason": "none"
}
```

### Failure Reasons


| Code                         | Meaning                                          |
| ---------------------------- | ------------------------------------------------ |
| `none`                       | All checks passed                                |
| `expected_action_invalid`    | Could not parse expected action from request     |
| `model_output_invalid`       | No function_call found in model response         |
| `tool_name_mismatch`         | Rollout used a different tool category           |
| `target_mismatch`            | Rollout targeted a different file/command        |
| `similarity_below_threshold` | Edit/command content too different from expected |
| `unknown_error`              | Unexpected exception during verification         |


## Relationship to Terminal Pivot

This server follows the same pattern as `resources_servers/terminal_pivot/`:


|                     | Terminal Pivot                               | SWE Pivot                                                           |
| ------------------- | -------------------------------------------- | ------------------------------------------------------------------- |
| Domain              | Terminal commands (keystrokes)               | Coding agent tool calls                                             |
| Action format       | JSON with `commands[].keystrokes`            | Responses API `function_call`                                       |
| Schema validation   | OpenAPI schema for command batches           | Tool name category matching                                         |
| Similarity          | `SequenceMatcher` on concatenated keystrokes | `SequenceMatcher` on tool arguments (edit content or shell command) |
| Task complete check | `task_complete` / `is_task_complete` flag    | Not applicable (single-turn pivots)                                 |
| Reward modes        | Continuous (similarity score)                | Binary or continuous (configurable)                                 |
| Extra shaping       | None                                         | Diff-size penalty for edit minimality                               |
| Tool diversity      | Fixed schema per harness                     | 63 tool name variants across 7 categories                           |


## Testing

```bash
cd resources_servers/swe_pivot
python -m pytest tests/test_verifier.py -v
```

Tests cover:

- All 63 tool name variants mapped to correct categories
- CamelCase variant coverage
- Argument parsing (JSON string, dict, invalid)
- Each verification level in isolation
- Full flow integration (perfect match, wrong tool, wrong file, over-engineering, execution error)
- Edge cases (empty args, missing fields, cross-category)
- Binary vs continuous reward mode behavior

