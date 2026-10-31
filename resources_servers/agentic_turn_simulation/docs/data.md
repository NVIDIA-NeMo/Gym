# Data

The row schema, every `verifier_metadata` key the environment reads, and how to build rows from your own agent
sessions, recorded or synthetic. For the summary, see the [README](../README.md#data).

## Row schema

One JSON object per line. This row (trimmed from `data/example.jsonl`, with a `reference_turn` added) is a Claude Code
conversation; the policy plays the turn that answers the last user message.

```json
{
  "responses_create_params": {
    "input": [
      {"type": "message", "role": "system", "content": "You are a coding assistant.\n\n<env>\nWorking directory: /home/user/demo\nPlatform: linux\n</env>"},
      {"type": "message", "role": "user", "content": "Make a file called hello.txt that says hi."},
      {"type": "function_call", "call_id": "t1", "name": "Write", "arguments": "{\"file_path\": \"/home/user/demo/hello.txt\", \"content\": \"hi\\n\"}"},
      {"type": "function_call_output", "call_id": "t1", "output": "File created successfully at: /home/user/demo/hello.txt"},
      {"type": "message", "role": "assistant", "content": "Done."},
      {"type": "message", "role": "user", "content": "What does hello.txt say?"}
    ],
    "tools": [
      {"type": "function", "name": "Read", "description": "Read a file.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}, "strict": false},
      {"type": "function", "name": "AskUserQuestion", "description": "Ask the user a question.", "parameters": {"type": "object", "properties": {"questions": {"type": "array", "items": {"type": "object"}}}, "required": ["questions"]}, "strict": false}
    ]
  },
  "verifier_metadata": {
    "harness": "claude_code",
    "harness_version": "2.1.90",
    "record_kind": "interactive",
    "session_id": "dummy-claude_code-1",
    "reference_turn": [
      {"type": "function_call", "call_id": "r1", "name": "Read", "arguments": "{\"file_path\": \"/home/user/demo/hello.txt\"}"},
      {"type": "function_call_output", "call_id": "r1", "output": "1\thi\n2\t"},
      {"type": "message", "role": "assistant", "content": "It says hi."}
    ]
  },
  "agent_ref": {"type": "responses_api_agents", "name": "agentic_turn_simulation_agent"}
}
```

- `input` is the history exactly as recorded, ending with the user message the turn answers. The system prompt is the
  harness's own. Earlier tool calls carry their real results.
- `tools` is the exact list the harness offered (trimmed here), with each tool's `parameters` schema as the harness
  sent it: the argument gates check the policy's calls against it. A call to a tool not in it is code-gated as
  `nonexistent_tool`. Native rows must list their tools: a row without `tools` code-gates every call, except that
  `sandbox` rows default to the 11 sandbox tools and Cline rows to Cline's tool names. Cline rows carry no function
  tools (`"tools": []` or no `tools` key).
- `reasoning`, `instructions`, `tool_choice` and `parallel_tool_calls` are optional and passed to the policy as given.
  `instructions`, when present, counts as part of the system prompt, which the judge and the simulator see only when
  the harness shows it (see `show_agent_system_prompt`).
- `agent_ref` points rollouts at the paired agent.

## `verifier_metadata` keys

| Key | Type | Read by | Meaning |
|---|---|---|---|
| `harness` | string | harness selection, metrics | The harness the row was recorded in (`sandbox`, `claude_code`, `hermes`, `openclaw`, `opencode`, `oh_my_opencode`, `kilo_code`, ...). Missing or unknown: `generic`. `cline` alone selects `generic`; Cline rows also need `protocol: cline_xml`. |
| `harness_version` | string | result formats | The harness release, for result formats that changed between versions (`"2.1.90"`, `"0.17.0"`). Missing: formats the session's earlier results do not settle go to the LLM. |
| `protocol` | `native` or `cline_xml` | harness selection | How the agent calls tools. `cline_xml` selects the Cline harness. Default `native`. |
| `record_kind` | string | judge framing, metrics | `interactive` (a person wrote the request) or `subagent` (a parent agent delegated it; the judge reads the requester as that agent). |
| `categories` | list of strings | metrics | Task categories; `turn_score_by_category` (rows without any count as `uncategorized`). Echoed in the verify response. |
| `corrected_turn_item_range` | `{"steerability": [lo, hi]}` | judge | Half-open index range into `input` of the earlier agent turn that this user message corrects. The judge sees it wrapped in `<rejected_turn>`. Native-protocol rows only. |
| `reference_turn` | list of Responses items | `score_source: reference_turn` | The recorded reply to this message, with its real tool results. |
| `obstacle` | `false` or `{"kind": ...}` | obstacle injection | Replaces the row's draw when `obstacle_injection` is on: `false` never arms the row, `{"kind": "transient" \| "blocked_route" \| "empty_search"}` arms that kind ([design](design.md#recoverable-obstacles)). Any other value is an error. |

Any other key (an id of your own, a split) is carried through to the rollout, and read only by the metrics when
`metric_breakdown_keys` lists it. The data-build scripts read the row id from `row_id` and the session from
`session_id`.

## Building rows from your own traces

### One row per user request

Convert each session into a Responses-format trajectory: the system prompt first, then user and assistant messages,
reasoning items before the output they belong to, and `function_call` / `function_call_output` pairs. Every user
request written by a person (or by your task generator, for synthetic rows) that has a reply becomes one row. For a
session a parent agent started (`record_kind: subagent`), the delegation message is the row.

Harness-injected user-role text is never a row: system reminders, "continue" nudges, environment-details-only
messages, tool results that a text protocol sends as user messages (Cline), compaction summaries, heartbeats and
scripted messages. A message that mixes a person's request with injected blocks counts as a request.

### The input is the untouched prefix

The input is every item up to and including that message, exactly as recorded: no truncation, no rewriting, and the
harness-injected text inside user messages kept, because that is what the model saw. Rows too long for a policy are
masked as `context_overflow` (with `propagate_context_overflow_errors: true` on a `vllm_model` policy; without it they
end as `max_output_tokens` and score 0), not cut. Use the harness's real system
prompt and the exact tool list that harness version sent. When a recording lacks the tool list, reconstruct it from the
harness release (for example by running it offline against a local request recorder and capturing its `tools` array)
rather than guessing: a call to a missing tool is code-gated, and a missing system prompt makes the simulator and judge
invent the user's machine.

### Record kinds

Mark each session `interactive` (a person typed at least one message) or `subagent` (a parent agent wrote the prompt).
Skip sessions with no request written by a person: scheduled jobs, scripted workers, and the harness's side requests
(memory updates, compaction, title generation).

### Privacy and placeholders

If your traces were redacted, review privacy and licensing before use; the rules below handle common placeholder
damage (`restore_sandbox_home` and `last_user_placeholder`). A trace with no visible redaction can still contain
personal data or credentials.

### Data rules

Fix a row only where the true value is known exactly; otherwise drop it, and never guess. Every rule has a name and
one of three levels:

- **normalise**: an exact edit that restores what the trace really held. The row stays.
- **validity drop**: the row would teach the wrong thing or cannot be scored fairly. It leaves every file built from
  the source. Validation rows are flagged instead of dropped, so scores can be reported with and without them.
- **run selection**: a choice for one training run. It applies to that run's rows, never to a source file.

| Rule | Level | What it does | Why | Implemented by |
|---|---|---|---|---|
| `intact_tool_history` | validity drop | Drops a session whose tool history is broken: a call without a call id, a name or arguments; an output that answers no earlier call; a call without a result outside the final step; a call to an unlisted tool or with undeclared arguments. | The env replays the history as the agent's own, and the policy learns from calls that never worked. | the row builder |
| `complete_reference_turn` | validity drop | Drops a row whose recorded reply is cut off or interrupted: it does not end with assistant text or a call to the harness's stop tool (it ends right after a tool result, after reasoning only, empty, or with a placeholder such as `[interrupted]`). | The reference turn shows the judge a finished reply; a cut-off one is masked. | the row builder |
| `trivial_turn` | validity drop | Drops turns that ask for nothing, such as polls. | Any reply passes, so the row carries no signal. | the row builder |
| `restore_sandbox_home` | normalise | In `sandbox` rows, replaces a redacted home directory such as `/home/[NAME]` with the sandbox's home `/home/user` in every string of the row: input, tools, instructions and `reference_turn`. | `/home/user` is the sandbox's own account directory, not personal data. The policy, the simulator and the judge must see the path the sandbox really has. | `clean_rows.restore_sandbox_home` |
| `flattened_tool_history` | validity drop | Drops a row in which an assistant message is a text rendering of a tool step (flattened tool history) (`[Tool result: ...]` as the message, or a `[Called tools: ...]` footer). Call syntax a model really wrote into its text stays. | The model never saw its history in that form. The policy would learn to narrate tool calls and imitate fake inline results instead of calling tools. | `clean_rows.flattened_tool_history` |
| `absent_input` | validity drop | Drops a row whose last user message refers to an input the row does not carry: an inline media placeholder whose media the trace left out (`[Image #1]`, `[Pasted text #2 ...]`), an id from a review list (`--reviewed`), or the same typed text (30 characters or more) as a reviewed row under the same system prompt. A message in which the harness says it could not read the input stays unless the review list names it. | The policy can only ask for the input or make it up. | `clean_rows.absent_input` |
| `incomplete_tool_list` | validity drop | Drops a row whose harness offered tools that its tool list lacks: a Hermes subagent row, an OpenClaw row whose `## Tooling` section names missing tools, a Claude Code row whose system prompt has `# MCP Server Instructions`. | The env would code-gate a call the real agent could make, and the missing schemas cannot be guessed safely. | `clean_rows.incomplete_tool_list` |
| `last_user_placeholder` | validity drop | Drops a row whose last user message holds a redaction placeholder from a vocabulary file (`--placeholders`). The whole message counts, including blocks a harness injected into it. Placeholders earlier in the history stay. Runs after `restore_sandbox_home`, so a restored path is not a placeholder. | The request the policy must answer is unanswerable or ambiguous. The policy can only guess or ask what the placeholder stands for, a judge may pay that ask, and RL can amplify the habit. The row is neither a clean act target nor a clean ask target. | `clean_rows.last_user_placeholder` |
| `duplicate_same_state` | validity drop | Of rows with the same or nearly the same typed last user message (harness blocks removed; 5-gram Jaccard >= 0.9) in the same conversation state (same system prompt, a matching reply before it, or both first turns), keeps one. | A copy upweights one state. | `clean_rows.duplicates` |
| `duplicate_repeat` | validity drop | Drops a substantive request (30 characters or more) that was re-sent from another effective session. A later copy in the kept row's own effective session is a new state: one row of that state stays. | The same request from elsewhere is the same target twice. | `clean_rows.duplicates` |
| `session_cap` | run selection | Keeps at most `--cap` rows per effective session, spread over the session's turns. An effective session merges sessions that hold a same-state copy of each other's rows (one conversation stored under several ids). Effective sessions are computed over every input row, including rows an earlier rule drops, so re-cleaning a cleaned file can drop a few more repeats. | One long session, or one conversation stored several times, must not dominate a training step. | `clean_rows.session_cap` |
| mixed selection | run selection | Keeps rows the starting policy sometimes passes and sometimes fails, under the run's judge. | A group whose rollouts all get the same reward carries no gradient. | [Picking RL rows](#picking-rl-rows) |
| prompt length | run selection | Drops rows whose prompt leaves too little room in the training context, and rows that overflowed in profiling. | Such rows end masked or cut off. | [Picking RL rows](#picking-rl-rows) |
| source exclusion | run selection | Leaves out whole sources that a run may not use. | A run uses only the sources it is allowed to. | the mix builder |

`scripts/clean_rows.py` runs its rules in the table's order on one source file at a time: rows are compared only
with rows of the same file. Normalisations run first, so every drop sees the normalised row. A normalisation edits the
raw line and is checked against the same edit applied to the decoded row; a drop never edits a line. `--levels` picks
the levels to run, and `--plan` writes the ids each rule changed or dropped. Inputs that belong to a dataset stay with
the data, not in this repository: review-list ids (`--reviewed`), the redaction step's placeholder vocabulary
(`--placeholders`), and harness text specific to a dataset that message comparisons should ignore (`--patterns`).

### Arguments that are not JSON objects

Recorded arguments that do not parse as a JSON object break some chat templates. Carry them as
`{"__unparseable_arguments__": "<the raw text>"}`. In the history this is just data; in a policy's own call the key
marks malformed arguments for the code gate (every harness but `sandbox`, whose only marker is
`__malformedToolArguments`).

### Reasoning items

Keep the recorded reasoning and let the policy's chat template decide what it sees. Gym's schema requires a
reasoning item's `id` and `summary`, and `vllm_model` passes the `summary` text to the template. Put recorded
reasoning in `summary`, never in `content`: an item with `content` or `encrypted_content` cannot be converted, and the
rollout fails.

```json
{"type": "reasoning", "id": "rs_3", "summary": [{"type": "summary_text", "text": "The config lives in src/kiln/config.py ..."}]}
```

The simulator and the judge never see reasoning.

### The reference turn

`reference_turn` holds the items after the row's message up to where the agent's reply ends, with their real tool
results. A reasonable cut for native protocols is the first assistant message with text that a user or system message
follows, leaving out turns the harness started on its own after it (heartbeats, task notifications); for Cline, the
last assistant message before the next row's message. A reference turn that ends right after a tool result is masked.

### Splits

Split train and validation by session, not by row: rows of one session share most of their history.

## Picking RL rows

Profile the starting policy before training, with the exact simulator, judge, caps and sampling settings you will
train with:

```bash
gym eval run --no-serve \
  --agent agentic_turn_simulation_agent \
  --input data/candidates.jsonl \
  --output results/profile.jsonl \
  --num-repeats 4 --temperature 1.0 --max-output-tokens 65536

gym eval profile \
  --inputs results/profile_materialized_inputs.jsonl \
  --rollouts results/profile.jsonl
```

Then keep rows where the policy's unmasked rollouts are **mixed**: at least one passes, at least one fails, and at
least two were measured. A group where every rollout gets the same reward carries no gradient, and a row that never
passes in a few tries is weak evidence either way. This writes the mixed rows of one run to `data/rl_rows.jsonl`:

```bash
python - results/profile_materialized_inputs.jsonl results/profile.jsonl data/rl_rows.jsonl <<'EOF'
import json, sys
from collections import defaultdict

inputs, rollouts, out = sys.argv[1:]
rewards = defaultdict(list)
for line in open(rollouts):
    r = json.loads(line)
    if not r.get("mask_sample"):
        rewards[r["_ng_task_index"]].append(r["reward"])
mixed = {t for t, rs in rewards.items() if len(rs) >= 2 and 0 < sum(rs) < len(rs)}
seen = set()
with open(out, "w") as f:
    for line in open(inputs):
        row = json.loads(line)
        if row["_ng_task_index"] in mixed and row["_ng_task_index"] not in seen:
            seen.add(row["_ng_task_index"])
            f.write(json.dumps({k: v for k, v in row.items() if not k.startswith("_ng_")}) + "\n")
print(f"mixed rows: {len(mixed)} of {len(rewards)} with a measured rollout")
EOF
```

The rows come from the materialized inputs, so their `responses_create_params` carry the profiling run's
`--temperature` and `--max-output-tokens`; drop those keys if the trainer should set them.

`_ng_task_index` numbers the rows of one run's input; for runs split into shards, group by `verifier_metadata.row_id`
instead.

Also drop rows whose prompt leaves too little room in the training context for the turn, and rows that overflowed in
profiling. Hold out whole sessions, and keep the held-out sessions identical across RL and SFT experiments so they stay
comparable. Expect selection to filter out a large share of candidate rows.

## Rebuilding from new raw data

Each step runs per source, since rows are compared only within one file. `S` is
`resources_servers/agentic_turn_simulation/scripts`, and `PY` is a Python with this server's requirements, run with
the Gym root on `PYTHONPATH`.

1. Build rows from the traces as described above, with `row_id`, `session_id`, `harness` and `record_kind`, and apply
   the row-builder rules (`intact_tool_history`, `complete_reference_turn`, `trivial_turn`).
2. Split by session into `train.jsonl` and `val.jsonl`.
3. Normalise the train rows, drop the invalid ones, and check the split for leaks:

   ```bash
   $PY $S/clean_rows.py train.jsonl train.clean.jsonl --levels normalise,validity \
       --placeholders tags.json --plan train.plan.json --holdout val.jsonl
   ```

4. Flag validation rows instead of dropping them, with no output file:
   `$PY $S/clean_rows.py val.jsonl --levels normalise,validity --placeholders tags.json --plan val.plan.json`. The
   plan lists the rows to report with and without. If a judge test set needs normalised rows, `--levels normalise`
   with an output file writes them, every row kept.
5. For a run, cap the rows per session in one call with every level, for each source the run may use (leave out the
   sources it may not). The cap then still sees the copies that link a conversation's clones; a later `--levels run`
   call on cleaned rows would cap each clone separately:

   ```bash
   $PY $S/clean_rows.py train.jsonl run.capped.jsonl --placeholders tags.json --cap 5 --plan run.plan.json
   ```

6. Pick the run's rows from `run.capped.jsonl`: drop long prompts and keep mixed rows
   ([Picking RL rows](#picking-rl-rows)), writing `run.picked.jsonl`.
7. Mix the sources: `python3 $S/mix_rows.py --seed 0 train_mix.jsonl a.picked.jsonl b.picked.jsonl`.
8. Check that every row loads in the env: `$PY $S/validate_rows.py train_mix.jsonl`.

Keep each plan with the data: it lists, per rule, the ids the rule changed or dropped, so a rebuild can be compared
with it.
