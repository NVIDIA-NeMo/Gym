# Agentic Turn Simulation

Train and evaluate agents on **whole turns of agent-harness conversations (recorded or synthetic)**. A *harness* is
the application around the model: its system prompt, its tools and how tool calls travel (Claude Code, Hermes,
OpenClaw, opencode, Kilo Code, Cline, ...). Each row is a conversation up to a user message. The policy plays the
agent's entire next turn, as many model and tool calls as it needs, against **simulated tools**; an LLM judge and a few
code checks decide whether the turn was right.

Whole turns matter because an agent's costliest mistakes (claiming a success its tool results contradict, handing
over partial work as complete) show up at the end of a turn, not in its first step. No dataset ships:
`data/example.jsonl` holds five hand-written dummy rows for a smoke test (one per harness, trivial requests), and
[docs/data.md](docs/data.md) shows how to build rows from your own traces.

The environment pairs with the agent
[`agentic_turn_simulation_agent`](../../responses_api_agents/agentic_turn_simulation_agent/README.md).

- [docs/design.md](docs/design.md): why whole-turn simulation, and how the simulator, the judge and masking work.
- [docs/harnesses.md](docs/harnesses.md): each harness's formats, Cline's XML protocol, adding a harness.
- [docs/data.md](docs/data.md): the row schema, building rows from your own traces, picking RL rows.
- [docs/training.md](docs/training.md): RL considerations and what to monitor during training.

## How it works

```
row: history up to a user message + the harness's system prompt and tools
  |
/seed_session ....... store the history, pick the row's harness, replay recorded results into a model of its files
  |
agent loop .......... policy step -> tool calls -> /simulate_tool_call -> answered in code if exact, else by the LLM
  |                   ends on: a final message (or an empty reply), a stop tool, the step (model call) or time cap,
  |                   a context overflow, a cut-off reply with a call or nothing visible (Cline: the mistake limit)
  |
/verify ............. masked? -> code gate -> LLM judge -> reward 1.0 if the turn is correct, else 0.0
```

- A **stop tool** is the harness's "ask the user" or "finish" tool. A call to one never runs: the reply's other calls
  run (on Cline rows, only those before it), then the turn ends; the user's next message is never simulated.
- The **simulator** answers in code what follows from what the conversation shows, mostly reads, writes and edits of
  files whose content is known and shell commands that only print such a file (`cat`, `head`, `sed -n`, ...), in the
  harness's own result format. Everything else (other shell commands, web tools, unseen files) goes to a simulator LLM
  told to be faithful rather than helpful.
- A **code gate** fails the turn without a judge call when it calls a tool the row does not offer, passes
  arguments its tool's schema rejects, or writes a tool call into its message ([Scoring](#scoring) lists them).
  An opt-in gate also fails a reply garbled with letters of a script the conversation never shows.
- **Obstacles** (optional, off by default): in a share of rows, one early web call fails the way the harness's
  recordings show (a transient error, a blocked URL, an empty search), always recoverably, so that giving up costs
  reward ([design](docs/design.md#recoverable-obstacles)).
- A **masked** rollout (`mask_sample: true`) is one whose outcome says nothing about the policy: the wall-clock cap,
  a row too long for the policy's context (when the policy server reports it), a simulator or judge failure. Gym
  leaves it out of the score instead of counting a 0. A turn the policy left unfinished (step cap, runaway output,
  a later context overflow, an empty final reply) scores 0.

## Quick start

You need two OpenAI-compatible model servers (vLLM; see Gym's
[vLLM guide](../../fern/versions/latest/pages/model-server/vllm.mdx)):

- **Policy**: tool calling on (`--enable-auto-tool-choice --tool-call-parser <parser>`), a reasoning parser for a
  reasoning model, and as long a context as you can give it: histories are never truncated. A row that does not fit
  comes back as an empty cut-off reply and scores 0; it is masked as `context_overflow` only when the policy server
  reports the overflow (`propagate_context_overflow_errors: true` on a `vllm_model` policy; Gym's default is false).
  Filter such rows out either way ([data](docs/data.md#picking-rl-rows)).
- **Simulator and judge**: a fixed model for every policy you compare, never the policy itself, at reasoning effort
  `high` with its reasoning parser and full context window. Use a strong reasoning model, and check its verdicts on a
  few hand-checked turns before trusting scores ([design](docs/design.md#the-judge)). A judge request holds up to 200k
  characters of history; one that does not fit masks the rollout as `judge_failed`.

```bash
# Servers (terminal 1): the simulator and the judge share one model server
gym env start \
  --resources-server agentic_turn_simulation \
  --model-type vllm_model \
  --model-url http://<policy-host>:8000/v1 --model <policy-model-name> --model-api-key EMPTY \
  +judge_base_url=http://<judge-host>:8000/v1 +judge_api_key=EMPTY +judge_model_name=<judge-model-name>

# Rollouts on the example rows (terminal 2)
gym eval run --no-serve \
  --agent agentic_turn_simulation_agent \
  --input resources_servers/agentic_turn_simulation/data/example.jsonl \
  --output results/agentic_turn_simulation_example.jsonl \
  --num-repeats 1 --temperature 1.0 --max-output-tokens 65536
```

- Outputs land next to `--output`: one rollout per line (`reward`, the verdicts, `stop_reason`, `mask_sample`,
  `failure_kind`), the metrics in `..._aggregate_metrics.json` (see [Scoring](#scoring)), and
  `..._materialized_inputs.jsonl`, which `gym eval reverify` and `gym eval profile` read. A rollout whose `/run`
  fails ends the run; add `++route_failures_to_sidecar=true` to record it in `..._failures.jsonl` and keep going.
- Run one resources-server worker (`num_workers` unset or 1, the default): sessions live in its memory, and a call
  that reaches another worker finds no session.
- The simulator and judge set throughput. For more than a few hundred rollouts, or to serve them from separate models,
  see [docs/training.md](docs/training.md#serving-the-simulator-and-the-judge).

### Reference evaluation settings

Reference evaluations use these settings:

- 4 repeats per row (`--num-repeats 4`) and a policy reply budget of 65,536 tokens (`--max-output-tokens 65536`).
- Agent caps `max_time_seconds` 1,200 (set with
  `++agentic_turn_simulation_agent.responses_api_agents.agentic_turn_simulation_agent.max_time_seconds=1200` on
  `gym env start`) and `max_steps` 50 (the default).
- A GLM 5.3-class reasoning model as simulator and judge, at reasoning effort `high` and temperature 1 (the shipped
  defaults).

The turn's time cap counts wall time, the waits for simulated tool calls included, and the simulator and judge share
one model server by default. Serve enough simulator and judge capacity, about 20 rollouts in flight per replica:
otherwise more turns hit the cap and are masked, and the score leaves out the slowest turns.

## Scoring

`/verify` runs these steps in order:

| Step | When | Result |
|---|---|---|
| Masked | The agent stopped on `time_cap`, the context overflowed before its first step (`context_overflow` at `steps` 0: the row does not fit; only when the policy server reports the overflow), or a tool result in the turn is a simulator failure. | `mask_sample: true`, no judge call |
| Code gate | A call in the turn fails a code gate (below), or the turn is unfinished: it ended on `max_steps`, `max_output_tokens`, a later `context_overflow`, or `empty_output` from the policy (`unfinished_turn`; a failed request's empty reply is judged, see [design](docs/design.md#masking)). | `reward: 0.0`, `code_gate` set, no judge call |
| Judged | Otherwise: each judge in `judges` returns its verdicts. | `reward: 1.0` if the turn is correct, else `0.0` |
| Masked | No judge returned a usable verdict: one with every field the verdict schema (`prompts/evaluation_schema.json`) requires, each a string and one of the values the schema lists. Unusable ones are retried (`judge_max_attempts`). | `mask_sample: true`, `failure_kind: judge_failed` |

The code gates check each call (and policy message) of the turn in order; the first failure sets `code_gate`, even
when the policy repaired the call later in the turn:

- `nonexistent_tool`: a tool the row does not offer.
- `malformed_arguments`: arguments that are not a JSON object (or carry a recording's malformed-arguments marker).
- `unknown_argument`: a key the tool's schema forbids (`additionalProperties: false`).
- `wrong_argument_type`: a value of the wrong JSON type. Values are not coerced: a string holding JSON is a string.
- `missing_argument`: a required argument left out, unless the schema gives it a `default` (the harness fills it in).
  Only for harnesses verified to reject such calls: `sandbox`, `claude_code`, `openclaw`
  ([why](docs/design.md#code-gates)).
- `call_in_text`: an assistant message that holds tool-call markup outside code: Qwen3-Coder-style XML
  (`<parameter=path>`, `</function></tool_call>`) or a `<tool_call>` block holding a Hermes JSON or GLM call. The
  harness shows such a call to the user instead of running it. Markup in code as Markdown reads it (a closed fenced
  block, an indented block, an inline code span) does not count, and neither does a lone `<tool_call>` token; a
  fence left open hides nothing. Not for Cline rows, whose calls are text.
- `foreign_script` (opt-in, checked after the others and `unfinished_turn`): with `foreign_script_gate_min_letters`
  set, a final reply that holds at least that many letters of scripts the turn's input never shows
  (`foreign_script_letters`), such as Chinese characters in a Persian sentence. Code, URLs, markup and quoted text do
  not count, and neither do Latin, lone Greek letters, a script written in lines of its own (a translation) or kanji
  after kana ([design](docs/design.md#code-gates)).

The argument gates check nested objects and arrays too, but not a failed `anyOf`/`oneOf` alternative. The schema is
the row tool's `parameters` (JSON Schema 2020-12 unless it names its `$schema`). A tool without an object schema, or
with one `jsonschema` cannot use, is not checked (so neither is a `sandbox` row without `tools`), and other
violations (enums, ranges, patterns) are left to the judge. In `reference_turn` mode only `nonexistent_tool` applies.

The judge answers four questions in one call:

- **Steerability** (`addresses_concern`): did the agent understand the request and engage with it? A clean refusal
  or one focused question counts.
- **Correctness** (`correct`; `blocked_by` names the cause of a failure): did it do the right thing and stop in the
  right place? Incorrect includes: complying with a request it should have refused, misreading the ask, only planning
  when it could act, repeating an approach that already failed, giving up on a recoverable error, reporting a blocker
  while an untried way forward remains, asking again after the go-ahead, asking about a choice left to it, claiming a
  success the tool results contradict or never checked, presenting partial work as complete, stating invented
  specifics as sourced, and delivering something visibly broken. An honest shortfall can be correct.
- `request_questionable` (is the request itself questionable) and `discard` (can the turn be judged fairly at
  all): reported, never part of the reward.

**Reward** is 1.0 when the turn is both correct and steerable (`correct_ladder: yes`), else 0.0. With several judges,
each judge's `correct_ladder` is computed and the majority wins; ties pay nothing.

The verify response adds these fields to the request's:

| Field | Meaning |
|---|---|
| `score_source` | The configured `score_source`. |
| `stop_reason`, `steps` | What ended the agent's turn and how many model steps it kept, from the agent's response metadata ([agent README](../../responses_api_agents/agentic_turn_simulation_agent/README.md)). |
| `ended_on_stop_tool` | The stop tool whose call ended the turn (`stop_reason` when it names one of the row's stop tools), else `null`. |
| `executed_calls` | The turn's calls that ran (got a result), stop tools excluded. With `ended_on_stop_tool` set, `0` means the turn stopped (asked, or handed back) without acting. |
| `foreign_script_letters` | Letters in the turn's final reply (its last messages and stop-tool question) of scripts the turn's input never shows: garbled text, `0` when none ([design](docs/design.md#code-gates)). Reported with the `foreign_script` gate on or off; `null` when the recorded turn is scored. |
| `code_gate` | The code gate that failed the turn (including `unfinished_turn`), else `null`. |
| `addresses_concern`, `correct`, `correct_ladder`, `blocked_by`, `request_questionable`, `discard`, `discard_reason` | The combined verdicts. |
| `judge_verdicts` | Each judge's parsed verdict, `null` for a judge with none. |
| `shell_failures` | Failed shell calls, `null` where not measured. |
| `simulated_calls` | Calls answered `exact`, by the `llm`, `failed`, or with an `injected` obstacle; `{}` when `/verify` finds no session, as under `gym eval reverify`. |
| `categories`, `harness` | The row's `categories` and the harness it ran under. |
| `obstacle` | With `obstacle_injection` on: whether the row was armed, whether the turn met the failure, whether the policy replied after it and what it did next ([design](docs/design.md#recoverable-obstacles)). |
| `failure_kind`, `failure_reason` | Masked rollouts only: `agentic_turn_simulation:<stop reason>` for `time_cap`, a too-long row's `context_overflow` and any other reason in `masked_stop_reasons`, `agentic_turn_simulation:simulator_failed`, `agentic_turn_simulation:recording_cut_off` in `reference_turn` mode, or `judge_failed`; and why. |

`stop_reason`, `steps`, `ended_on_stop_tool` and `executed_calls` describe the policy's turn, also in `reference_turn`
mode. A trainer can read them, and the agent's per-call token usage (`response.metadata.model_calls`, see the
[agent README](../../responses_api_agents/agentic_turn_simulation_agent/README.md#token-usage)), without knowing the
row's harness.

**Metrics** (over measured rollouts; Gym counts masked ones in `coverage/masked_rollouts`):

| Metric | Meaning |
|---|---|
| `turn_score` | Mean reward (key metric) |
| `steerable_rate`, `code_gate_rate` | Share of measured turns the judge found engaged with the request (code-gated count as not) / that were code-gated (key metrics) |
| `questionable_request_rate` | Share of requests the judge found questionable |
| `stop_reasons` | Count per agent stop reason (measured rollouts only, so masked reasons never appear) |
| `failure_attribution` | Share of failed turns by cause, keyed by the `code_gate` value, else the `blocked_by` value, else the literal `steer=no` (also incorrect turns no judge attributed) |
| `shell_failures_per_turn` | Mean failed shell calls, where measured |
| `obstacle_armed_rate`, `obstacle_fired_rate` | With `obstacle_injection` on: share of turns armed / of armed turns that met the failure |
| `obstacle_turn_score`, `obstacle_retried_rate`, `obstacle_quit_rate` | Of the turns that met the failure and replied after it: mean reward / share that called the same URL or query again / share that got no other source (no web call after it, none beside it in its reply that got a result) and made at most one call of any kind after it |
| `turn_score_by_category` | Per `categories` value (`uncategorized` for rows without any) |
| `turn_score_by_<key>` | Per value of each `metric_breakdown_keys` key (by default `turn_score_by_harness` and `turn_score_by_record_kind`), when some row carries it (`none` for rows without it) |

### Reporting an eval

Quote the score with masked rollouts counted as failures next to `turn_score` (masked excluded), and the masked
count by `failure_kind`. Gym reports neither the masked-as-failures score nor a count per `failure_kind`, so compute
them from the rollouts file:

```bash
python - results/agentic_turn_simulation_example.jsonl <<'EOF'
import json, sys
from collections import Counter
rows = [json.loads(line) for line in open(sys.argv[1])]
measured = [r for r in rows if not r.get("mask_sample")]
passed = sum(r.get("reward") == 1.0 for r in measured)
print(f"masked as 0: {passed / len(rows):.3f}  turn_score (masked excluded): {passed / max(len(measured), 1):.3f}")
print("masked by failure_kind:", dict(Counter(r.get("failure_kind") for r in rows if r.get("mask_sample"))))
EOF
```

With `obstacle_injection` on, take the `obstacle_*` rates over the same rows, masked ones included
(`obstacle_metrics(rows)`, from `resources_servers.agentic_turn_simulation.obstacles`): turns that persist past a
failure hit `time_cap` more often than turns that quit.

Time-cap masks rise when the simulator and judge are overloaded, so compare runs served at similar load. To
re-judge saved rollouts after changing the judge (`/verify` recomputes the reward from the request alone), use
`gym eval reverify`. It calls only `/verify`, so the policy need not be running, but it first waits for every
configured model URL to answer, the agent's `policy_model` included: point the `+policy_*` values at the judge
server, which it checks anyway.

```bash
gym eval reverify --resources-server agentic_turn_simulation --model-type vllm_model \
  +policy_base_url=http://<judge-host>:8000/v1 +policy_api_key=EMPTY +policy_model_name=unused \
  +judge_base_url=http://<judge-host>:8000/v1 +judge_api_key=EMPTY +judge_model_name=<judge-model-name> \
  --inputs results/agentic_turn_simulation_example_materialized_inputs.jsonl \
  --rollouts results/agentic_turn_simulation_example.jsonl \
  --output results/agentic_turn_simulation_example_reverified.jsonl
```

## Data

Each row is a Responses-format task:

- `responses_create_params.input`: the conversation up to and including the user message the turn answers, exactly as
  recorded (system prompt, earlier turns, real tool calls and results).
- `responses_create_params.tools`: the tools the harness offered, with their real `parameters` schemas. A call to
  any other tool, or with arguments its schema rejects, is code-gated.
- `verifier_metadata.harness`: the harness the row was recorded in ([Harnesses](#harnesses)); rows without it run
  under `generic`, and Cline rows also need `protocol: cline_xml`. The other keys
  ([docs/data.md](docs/data.md#verifier_metadata-keys)) add metric breakdowns, version-specific formats and a
  reference turn.

## Harnesses

| `harness` | Stop tools | Answered in code | Notes |
|---|---|---|---|
| `generic` (no `harness`, or an unknown one) | `ask_user`, `AskUserQuestion`, `clarify`, `question`, `ask_followup_question`, `attempt_completion` | nothing | the LLM answers everything (an unknown name is logged once) |
| `sandbox` | `ask_user` | `read_file`/`edit_file` of known files, `bash` read-backs of known files, `write_file`, `present_file`, image and speech generation, malformed or unknown calls | A Linux sandbox tool set: a fixed set of 11 tools with JSON results |
| `claude_code` | `AskUserQuestion`, `ExitPlanMode` | `Read`/`Edit` of known files; `Write` over a known file; `Bash` read-backs of known files | |
| `hermes` | `clarify` | `read_file` of known files; `write_file`/`patch` for versions before 0.12; `terminal` read-backs of known files | JSON results |
| `openclaw` | `ask_user` | `read`/`edit` of known files; `write`; `exec` read-backs of known files | |
| `opencode`, `oh_my_opencode` | `question`, `plan_exit` | `read` of known files; `write`/`edit` of files no language server checks; `bash` read-backs of known files | |
| `kilo_code` | `question`, `plan_exit`, `ask_followup_question`, `attempt_completion` | as `opencode`, for its CLI tools | |
| any, with `protocol: cline_xml` | `ask_followup_question`, `attempt_completion`, `plan_mode_respond` | nothing | Cline: XML tool calls in the reply text |

Each harness has its own simulator prompt. All but `generic` and `sandbox`, which have their own judge frames, add
a short description of the agent to a shared judge frame; the rubrics are the same for all. See
[docs/harnesses.md](docs/harnesses.md) for the formats and for adding a harness.

## Configuration

| Field | Default | Meaning |
|---|---|---|
| `tool_simulator_model_server`, `judges[].model_server` | required | Model servers (chat completions) for the simulator and the judges. Several judges vote. |
| `tool_simulator_chat_completions_create_params`, `judge_chat_completions_create_params` | required | Request parameters (`messages: []`; the server fills them). Shipped: `max_tokens: 16384` / `20000`, `temperature: 1.0`, `reasoning_effort: high`. |
| `judges[].overrides` | `{}` | Per-judge parameter overrides (`null` removes a parameter). |
| `tool_simulator_max_attempts`, `judge_max_attempts` | `4`, `6` | Attempts per call. A simulator call that fails them all masks the rollout; a judge that fails them all leaves the vote (masked only when no judge is usable). |
| `tool_simulator_max_tokens_cap`, `judge_max_tokens_cap` | `49152`, `65536` | Ceiling when a cut-off reply doubles `max_tokens`. |
| `tool_simulator_max_concurrency`, `judge_max_concurrency` | `64`, `32` | Concurrent requests (`null` or `0`: unlimited). |
| `retry_backoff_max_seconds` | `20.0` | Cap of the exponential backoff between attempts. |
| `score_source` | `policy` | `reference_turn` judges the row's recorded turn instead (a baseline). The policy still runs, so use a small one and one repeat. |
| `masked_stop_reasons` | `[time_cap]` | Agent stop reasons that mask a rollout; list more (e.g. `max_steps`) to mask those too. A `context_overflow` before the first step is masked regardless. |
| `unfinished_stop_reasons` | `[max_steps, max_output_tokens, context_overflow, empty_output]` | Agent stop reasons scored 0 as `unfinished_turn` without a judge (a masked reason is masked first; `empty_output` only when the empty reply is the policy's, not a failed request's: see [design](docs/design.md#masking)). |
| `foreign_script_gate_min_letters` | `null` | Score 0 as `foreign_script`, without a judge, a policy turn whose `foreign_script_letters` reaches this many (`null`: off; the count is reported either way). |
| `session_ttl_seconds` | `21600` | Sessions that never reach `/verify` are dropped after this long. |
| `show_agent_system_prompt` | `null` | Show the judge and simulator the row's system prompt. `null`: every harness but `sandbox`. |
| `obstacle_injection` | `{enabled: false, row_rate: 0.3, kinds: {transient: 0.25, blocked_route: 0.5, empty_search: 0.25}, call_kinds: [fetch, search], harnesses: null, max_call_index: 2, seed: 0}` | Recoverable failures injected into the policy's web calls ([design](docs/design.md#recoverable-obstacles)). When enabled, a hash of `seed` and the row's input arms `row_rate` of the rows (`verifier_metadata.obstacle` overrides it), each with one kind drawn by `kinds`; the failure hits the first fitting web call of an allowed `call_kinds` among the turn's web calls 0..`max_call_index` that its kind can fail (fetches for `blocked_route`, searches for `empty_search`), never first in a reply that calls a stop tool. `harnesses: null` is every harness with recorded texts (`sandbox`, `claude_code`, `openclaw`, `opencode`); a row without `harness` runs as `generic` and is never armed. |
| `metric_breakdown_keys` | `[harness, record_kind]` | `verifier_metadata` keys that `turn_score` is broken down by (dotted for a nested key, such as `difficulty.level`, which becomes `turn_score_by_difficulty_level`). |
| `judge_system_prompt_note_file` | `judge-system-prompt-note.md` | How the judge should read that prompt: facts in it are grounded and its instructions are ones the agent had to follow. |

The agent's settings (`max_steps`, `max_time_seconds`, output-budget limits) are in its
[README](../../responses_api_agents/agentic_turn_simulation_agent/README.md).

## Design and training

[docs/design.md](docs/design.md) explains the choices: whole turns against a simulator rather than single steps,
reference actions or live sandboxes; a simulator that is exact where it can be and faithful elsewhere; correctness
gated by steerability as the reward; and masking what the environment, not the policy, decided.

[docs/training.md](docs/training.md) covers data selection, serving the simulator and judge, caps and masking, and
what to monitor during RL (stop-tool share, reasoning length, ask-the-user rate).

## Limitations

- **Simulated tools.** Compare policies with each other, not with the recorded turn, which saw real results. Web
  content is unknowable, so simulated searches tend to find what the agent looks for, and over a turn the LLM
  simulator leans toward letting the agent's decisive action succeed.
- **Read-backs through the shell** are answered in code only in their plain forms (`cat`, `head`, `tail`, `sed -n`
  and, in the sandbox, `wc` of one known file; outside the sandbox, by absolute path). Others (`grep`, a pipe, several
  files) are answered by the LLM and can disagree with the file; agents that verify heavily can lose time chasing the
  mismatch and hit the time cap.
- **Files a shell command changes without naming them** (`python fix_cfg.py && make build`) keep their old content
  in the workspace model, so a later read of one is answered in code with stale content.
- **One turn.** The user's next message is not simulated; the judge decides whether stopping to ask was warranted.
  The rubric fails avoidable questions, but judging stops remains its least certain call.
- **Rows too long for the policy** score 0 by default, which penalizes a policy with a small context window on the
  longest rows; masking them (`propagate_context_overflow_errors: true`) flatters it instead. Filter them out. A
  context overflow later in the turn scores 0 either way.
- **Argument gates trust the row's schemas.** A harness that coerces a value or ignores an extra key runs calls the
  gate fails. This is rare, but Hermes runs some wrong-type calls, and calls to stop tools (such as the sandbox's
  `ask_user`) have no recorded result to check ([design](docs/design.md#code-gates)).
- **`generic` answers nothing in code**, and how closely the simulator matches each harness's real results is not
  measured.
- **`foreign_script_letters` misses some garbling**: letters of a script the turn's input shows (a garbled file the
  policy wrote and read back unescaped included), fewer than three letters of a script, Latin words in a non-Latin
  reply, and a reply written wholly in the wrong language.
- **`discard` is reported, not masked**: turns the judge cannot judge fairly are still scored.
- **No image inputs**: rows whose history holds images are not supported.
- **A model judging itself** may prefer its own turns; add a second judge to `judges` for close comparisons.

## Code map

- `app.py`: config, payload models, endpoints, the verify flow and metrics.
- `harnesses/`: one module per harness (`opencode.py` holds opencode and Kilo Code; `base.py` has the `Harness` class
  and the generic harness; `sandbox.py` has its own `SandboxWorkspace`). `__init__.py` has the registry: `HARNESSES`,
  `ALIASES` and `select_harness`.
- `workspace.py`: the model of the agent's files, the file operations answered in code, and the shell read-backs
  (`ReadBack`) answered from it.
- `foreign_script.py`: the count of foreign-script letters in the final reply (`foreign_script_letters`), with its
  rule in the module docstring.
- `obstacles.py`: the optional recoverable obstacles: arming a row, which calls fail, what the simulator sees, the
  verify report.
- `simulator.py`, `judge.py`, `render.py`: the simulator's view of a call and its reply parsing; the judge's prompt,
  verdict parsing, voting and code gates; how conversations are rendered for both.
- `cline_xml.py`: Cline's XML tool-call parser and messages (shared with the agent). `items.py`: helpers over
  Responses items.
- `prompts/`: simulator and judge prompts, rubrics and the verdict schema.

## Tests

```bash
gym env test --resources-server agentic_turn_simulation
gym env test +entrypoint=responses_api_agents/agentic_turn_simulation_agent
```

## Licensing information

Code: Apache 2.0

Data: the example rows are synthetic, Apache 2.0.

Dependencies

- nemo_gym: Apache 2.0
