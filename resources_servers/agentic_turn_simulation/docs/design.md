# Design

Why the environment works the way it does. For usage, see the [README](../README.md).

## The problem

Agent harnesses (coding CLIs, IDE agents, personal assistants) produce long multi-step sessions with rich system
prompts and tool lists; scoring a turn needs the results of the agent's own calls. The question is: given the
conversation so far, did the agent handle this request well? A request is usually answered with a multi-step turn
(several model calls, each with tool calls, until the agent hands control back).

Vocabulary used throughout: a **step** is one model call with its text and tool calls; a **turn** is every step
between one user message and the agent handing control back.

## Alternatives considered

**Feedback labels as the reward.** User ratings, where available, record reactions, not correctness, and are not used
as reward: users approve what looks finished and cannot see an invented fact or an
unchecked claim, a turn's label often depends on the user's next message rather than on the turn, and labels are sparse
and lopsided.

**Matching a reference action.** Comparing the policy's call with a demonstrated correct call needs a gold action. A
recording holds what the original agent did, including its failures, and a request usually has many acceptable
answers. A rubric replaces the gold action.

**Scoring a single step.** Rebuilding the history, asking for one step and judging it needs no environment, but it
over-rates turns that start well and fail later: the plan looks right, and the failure (a test that never ran, a file
that came out in the wrong format) appears only after the tool results come back. Whether a turn was right is a
question about the whole turn.

**Live sandbox replay.** Re-running the turn in a real sandbox needs the exact pre-turn state: every file the earlier
turns wrote, whatever earlier commands installed, the web as it was. A trace has none of it, and for most harnesses
the environment is the user's own machine, which cannot be rebuilt at all. Recorded traces may also be incomplete.

## Why whole-turn simulation

The policy plays the whole turn, and a simulator stands in for the tools. This keeps what makes a turn hard (the agent
must act, read its own results and report honestly) without needing the original environment. The judge then scores
the turn the way a careful human would: with every tool result in front of it, checking the agent's final claims
against what its tools returned, without the single-step blind spot.

## The simulator

### Exact where knowable, an LLM otherwise

A call is answered in code when its result follows from what the conversation shows: a read or edit of a file whose
whole content is known, a shell command that only prints such a file, a write whose confirmation depends on nothing
unknown, and (for the sandbox harness) its error for a malformed or unknown call and its per-turn generation limits.
Code answers have no simulation error and cost nothing, so every call that can be answered exactly is. Everything else
goes to the LLM: other shell commands, web search and fetches, and files nobody has shown. Each code answer is written
in the harness's own result format, because the policy has seen that format in the history and a different one would
be a tell.

### Faithful, not helpful

The simulator prompt tells the LLM to run the call exactly as written against the machine as the conversation shows
it: a bug, a wrong path or a failing test fails with the real error. It also says not to invent failures: most calls a
competent agent makes succeed. Both directions matter, because an optimistic simulator pays for "claim a fix and
re-run", and a pessimistic one punishes correct work.

What the LLM sees depends on the harness ([harnesses.md](harnesses.md)). At most it gets the harness's system prompt
(first and last 30k characters), the tool definitions, the head and tail of the conversation, the pending call, and
the known contents of up to three files the call names. `sandbox` gets no tool definitions and, by default, no system
prompt; `generic` gets the system prompt but no tool definitions and no workspace. The LLM never sees the current
step's other calls (only the results of the ones already answered), since the real tool runs without knowing them.

The simulator runs at high reasoning effort. A lower-effort simulator tends to be optimistic: calls that should fail
come back as success, often as "the fix worked". Prompt rules written against specific errors tend to fit those calls
more than fresh ones; facts taken from the conversation's own tool outputs generalize better than more rules.

A reply that cannot be used is retried: an empty or cut-off reply with double the output budget (up to
`tool_simulator_max_tokens_cap`), a reply in the wrong format at the same budget. After the last attempt, the tool
result is a marked failure (a JSON error object whose message starts `[tool simulator failed:`), never an invented
result, and the rollout is masked: the failure is the environment's, not the policy's. Only that whole object
masks; the phrase alone, which a file or a command's output may contain, does not.

### The workspace model

The workspace maps each path to its content, or to "exists, content unknown". At seed time it replays the recorded
history with its real results; during the turn it learns from every answer, so a file written in step 2 reads back
exactly in step 5, through the file tool or through the shell. The harness workspace (Claude Code, Hermes, OpenClaw,
opencode, Kilo Code, Cline) never guesses a home or working directory; the sandbox workspace knows its machine and
resolves paths against `/home/user`. The rules are in [harnesses.md](harnesses.md#the-harness-workspace).

**Shell read-backs.** Agents often check their own writes through the shell (`cat`, `head`, `sed -n`) rather than a
read tool, and an LLM answer to such a read-back can disagree with the file it sees in its context: an agent that
verifies carefully then chases a mismatch the environment made, and can run into the step or time cap. So a shell
command that only prints a known file is answered in code, in the shell tool's own result format: `cat P`,
`head -n N P`, `tail -n N P` (also `-N`, `+N`), `wc -l P`, `wc -c P` and `sed -n 'A,Bp' P`, with one plain path and
no pipe, glob, redirection, variable or second command. The output is what GNU coreutils prints. Anything else, such
as `grep`, a pipe or a second file, still goes to the LLM. The sandbox answers all of these, with paths under the
call's `cwd` (by default the home directory) and a stable `duration_ms` in the range recorded read-backs took, the one
part of the result that cannot be known. The harness shells answer only by absolute path, since the shell's working
directory is not known; they leave `wc` to the LLM, because GNU and BSD `wc` pad their output differently and the
user's operating system is not known either, and likewise `head` of 0 lines (BSD `head` rejects the count) and a `sed`
range that ends on a last line without a line break.
Each harness frames the output as its recordings show (trimmed, or with its own note for no output) and declines what
its recordings leave open, such as how far it trims surrounding whitespace.

**Forgetting.** Both workspaces forget conservatively, by the same rules: every known content after a command that can
change files it does not name (a formatter or fixer such as `prettier --write`, `black`, `--fix`, `go fmt` or
`npm run format`, a package manager that rewrites the manifest such as `npm install x`, `yarn add x` or
`npm version patch`, or a version-control operation such as `git checkout .` or `git stash`), and otherwise the files a
mutating command names, by path or by basename however short (`sed -i` in any spelling, `perl -pi`, redirects, `mv`,
`rm`, `curl -o`, `wget`, `sort -o`, `dos2unix`, PowerShell's `Set-Content`, ...). The harness workspace also forgets
everything after a subagent or another opaque tool. A file changed by a command that names neither it nor a known
pattern (`python fix.py`) keeps its old content, so a later read-back of it is answered with stale content.

Being conservative costs exact answers: a forgotten file goes to the LLM, which still sees the rest of the
conversation but can answer a read-back inconsistently.

### What is deliberately not simulated

- **Harness noise at random.** Recordings can include the harness's own infrastructure failures (generic errors,
  backend timeouts). The simulator does not reproduce them at random: a failure the policy cannot get past would
  penalize it for someone else's infrastructure. The exceptions are a harness's own answer to a malformed call, which
  the policy caused, and the opt-in [recoverable obstacles](#recoverable-obstacles), which the policy can always get
  past.
- **The user's next message.** A stop tool (the harness's "ask the user" or "finish" tool) ends the turn unexecuted.
  Simulating the user would add a second simulator to trust and to exploit, and the judge can already decide whether
  stopping to ask was right.

## The judge

**One call, analysis first.** A single judge call returns every verdict. Its JSON schema puts a short `analysis`
first, so the judge reasons about what was asked and what the agent actually did before it commits. One call also keeps
the verdicts consistent with one reading of the turn.

**Four verdicts.**

| Verdict | Question |
|---|---|
| Steerability (`addresses_concern`) | Did the agent understand the request and engage with it? A clean refusal or one focused question counts. |
| Correctness (`correct`, `blocked_by`) | Did it take the right action and end the turn in the right place? `blocked_by` attributes a failure: `model`, `infra`, `possibly_interrupted`, `refusal`, `impossible`, `other`. |
| Request (`request_questionable`) | Is the request itself questionable (credentials, defeating an access control, a jailbreak)? Judged on the request only. |
| Discard (`discard`, `discard_reason`) | Can the turn be judged fairly at all (it refers to something outside the trace, the complaint is about the harness UI, it was cut off before the action)? |

**Correctness is the reward; steerability gates it.** Almost every turn engages with its request, so steerability alone
carries little signal; correctness splits turns into more balanced classes and is the question that matters. A turn
that misread or ignored the request cannot have done the right thing, so `correct_ladder` is `no` whenever steerability
is `no`. The correctness rubric checks claims against evidence: success that the turn's own tool results contradict (or
that was never checked), partial work presented as complete, invented specifics presented as sourced, and a visibly
defective deliverable presented as final are all incorrect. An honest shortfall ("the build could not run here; these
are untested") can still be correct, but only after the reasonable routes were tried. Stopping early is incorrect too:
a blocker report while the conversation still offers an untried way forward, asking again after the user's go-ahead,
and a question about something the agent could have picked and stated (a style, a format, a choice the user left to it)
when the work could have started.

**What the judge sees.** The history (capped at 200k characters) and the whole turn, with every tool argument and
result in both cut to 2,000 characters (head and tail kept), but never the policy's reasoning: the reward is for what
the agent did and said. When `corrected_turn_item_range` marks the earlier turn the user is correcting, it is wrapped
in `<rejected_turn>` and marked as the turn the user is correcting.

**Voting and ties.** With several judges, only usable verdicts count: a usable verdict has every field the verdict
schema requires, each a string and one of the values the schema lists, so a vote with a missing or null steerability
can never pay. A judge reply that is not usable (a cut-off one included, unless every field was already written) is
retried up to `judge_max_attempts`. Each judge's `correct_ladder` is `no` if its steerability is `no`, else its
`correct`, and the reward follows the majority of those ladders; the other axes are voted separately. Voting the ladder rather than the axes keeps the reward consistent with each judge's own reading:
two judges that each fail the turn for a different reason still fail it. Ties go to `no` for steerability,
correctness and the ladder, to `ok` for `request_questionable`, and to `no` for `discard`, because a reward should not
be paid on a coin flip. `blocked_by` comes from the judges that marked the turn incorrect, so a dissenting "none"
cannot erase an attribution.

**High reasoning effort.** A lower-effort judge tends to be lenient rather than noisier: it accepts invented figures,
unverified claims and "impossible" without the agent trying the tools, which hides part of the gap between models and
gives RL a target.

**The agent's system prompt.** Harness system prompts carry the working directory, OS, date, model identity, project
rules and memory. Without them, the judge can call answers taken from the system prompt "fabricated" and the simulator
can invent a different machine for each policy. The prompt is shown in an `<agent_system_prompt>` block before the
conversation, with a note telling the judge how to use it: a fact taken from it is grounded, an instruction in it is
one the agent had to follow. The block also lets the judge check system-prompt adherence. The `sandbox` harness hides
the prompt by default because its judge frame already describes the sandbox; `show_agent_system_prompt: true` shows it
everywhere.

**Checking the judge.** Before trusting scores from a judge model, compare its verdicts with your own reading of a
few hand-checked whole turns, and re-run it on the same turns to see how stable its verdicts are. Fail verdicts that
vary between re-runs average out over many rollouts.

**Why the rubric fails avoidable stops.** A rubric that accepts almost any focused question and credits honest blocker
reports is open to a cheap strategy: end a clear task in a question before any tool call, re-ask after "continue" or
"do it", or file an "I can't access X" report with the named file never opened, each paid about as well as finished
work. The clauses above close that route. Judging stops stays the soft spot; track the share of turns ending in a stop
tool during training.

## Code gates

Some defects code sees exactly, where a judge would be slower and noisier. A policy's turn scores 0 without a judge
call when a call names a tool the row does not offer (`nonexistent_tool`), passes arguments that are not a JSON object
(`malformed_arguments`), or passes arguments its tool's schema rejects: a key the schema forbids (`unknown_argument`),
a value of the wrong JSON type (`wrong_argument_type`), or, for some harnesses, a missing required argument
(`missing_argument`). The first failing call of the turn names the gate. A call the policy repairs later in the turn
still fails it, as a call to a tool that does not exist does: the gate is about calling tools as their schemas say, and
forgiving a repair would teach the policy to lean on the harness's error messages instead.

**Only what the real harness rejects.** A gate that fails calls the real harness would have run teaches a rule the
deployment does not have, so what each harness does with such calls sets what is gated: a call's result in a trace
shows whether the harness rejected its arguments.

- Unknown keys and wrong types are gated for every harness with tool schemas, with values taken as sent (no coercion).
  Claude Code, OpenClaw and opencode answer wrong-type calls with a validation error, but Hermes runs some of them, so
  on Hermes rows this gate can fail a call the harness would run.
- Missing required arguments split by harness. The sandbox, Claude Code and OpenClaw answer such calls with a
  validation error, so they are gated there (`Harness.gates_missing_arguments`), except for a required key whose schema
  declares a `default`: Claude Code's schemas list some such keys as required (`AskUserQuestion`'s `multiSelect`,
  `TaskOutput`'s `timeout`), and calls without them run normally. Hermes and opencode run calls that leave out an
  argument their schemas mark required (Hermes's `patch` mode, opencode's `bash` description) normally; gating them
  would fail correct turns. Kilo Code shares opencode's tools but whether it rejects such calls is not verified,
  `generic` rows come from unknown harnesses, and Cline's tools are documented in prose rather than schemas, so none of
  these gate it.
- Calls to stop tools (the sandbox's `ask_user`, Claude Code's `AskUserQuestion`) are gated by the same rules, but
  a stop call ends the turn, so no trace holds a result showing whether the harness rejects its arguments. That gate
  rests on the schema alone.
- Other violations (enum values, ranges, patterns) are left to the judge: harnesses both accept and reject them. So is
  a schema that `jsonschema` cannot use.

**Calls written as text.** On a native harness a call travels as a function call. One the policy writes into its
message instead, in its chat template's markup (`<parameter=path>`, `</function></tool_call>`, a `<tool_call>` block
holding a call), never runs: the harness shows the markup to the user. A policy message holding such markup fails the
turn (`call_in_text`). Typical cases are a file written out in the message followed by its `write_file` or
`present_file` call, or a question closed by `</function></tool_call>`; a judge can read such text as delivered and
pass the turn. Markup inside code as Markdown reads it (a closed fenced block, a longer fence quoting a shorter one, an
indented block, an inline span of any number of backticks) does not count, so a turn that explains a template is still
judged, and neither does a lone `<tool_call>` token in degenerate text. A fence left open does not hide what follows: a
leaked file can open one in its content and never close it. Cline rows are exempt: their calls are text by design.

**Garbled scripts (opt-in).** A policy that writes a non-Latin language sometimes splices a few characters of another
script into its reply, such as Chinese characters or Cyrillic letters inside a Persian or Arabic sentence. The judge
seldom fails such a turn: its analysis can name the garbling and still call it cosmetic, so a judge-only reward puts
little pressure against it. Code sees it without a language model.
`foreign_script_letters` counts the letters in the turn's final reply of scripts that the turn's input (the history and
the turn's tool results) never shows, after removing code, URLs, markup and quoted text, where any script is
legitimate. Four exclusions remove the false positives of a looser rule. Latin never counts: product names,
identifiers and glosses are Latin everywhere. Greek counts only as words, since lone Greek letters are math. A script
written mostly in lines of its own is a deliberate passage, such as a translation or a whole reply in that language.
Kanji count as Japanese when the input has kana. `foreign_script.py` states the rule. What it can still flag is a
reply written wholly in a language the user did not use and a name written in its own script.

The count is reported for every policy turn, and the gate (`foreign_script_gate_min_letters`, checked after the other
gates and `unfinished_turn`) is off by default, for three reasons. First, how many turns it fails depends on the
policy's languages and the languages of the data. Second, a flagged turn may hold only a few foreign letters, so
where to put the threshold is a judgment call. Third, its precision depends on the data, so check a sample of flagged
turns on yours. Turn it on for training when the policy garbles, since the judge alone does not penalize it.

A "the shell never recovered" gate was rejected: such turns were mostly the environment's fault and honestly reported,
so shell failures are a metric (`shell_failures`), not a gate.

## Masking

**What is masked.** A rollout is returned with `mask_sample: true` and a `failure_kind`, so that scores leave it out
rather than count it as a 0, only when its outcome says nothing about the policy: the turn hit the wall-clock cap
(`time_cap`), the row does not fit the policy's context (the context overflowed before the first step), the
simulator could not answer, or no judge returned a usable verdict.

**Why time caps and too-long rows.** A time cap measures serving speed more than behavior: a turn it cuts off is
usually slow per step rather than long, and the same model hits it more often on an overloaded simulator and judge than
on a healthy one. A row that does not fit fails before the policy acts, and on practically every rollout of that row: a
property of the row and the policy's context window, not of behavior.

**Why step caps and runaway output score 0.** The step cap (`max_steps`) and an output cap that cut a call or left
nothing visible (`max_output_tokens`) follow from what the policy did: it kept calling tools without converging, or
generated without end. Other rollouts of the same row can finish within the caps. Masking these turns would hide
them from the reward and pay nothing for avoiding them. Judging them does not work either: a judge can pass a cut-off
turn that ended on a tool call without ever reporting back, crediting the progress made before the cut (or a doom loop
whose individual actions look right). So they score 0 without a judge (`code_gate: unfinished_turn`).

**An empty final reply scores 0 too.** A reply with neither a message nor a call (`empty_output`) leaves the turn
unfinished just as a cap does: the user gets nothing after the turn's tool calls. Such a reply carries reasoning,
often a tool call drafted there and never emitted, so it is the policy's doing, and a judge can still pass it. A failed
request's reply is different, so it is judged as before: one with no items at all, not
even reasoning, which is what a model server passes on for a failed request, and one the model server marked
unfinished (an `error`, or a `failed`, `cancelled`, `queued` or `in_progress` status), since a Responses-native server
passes on an aborted request's partial output, reasoning and all. `completed`, `incomplete` (cut off by the output
budget) and no status at all are the policy's. vLLM's chat placeholders never get here: they carry an empty message
and end the turn as `max_output_tokens`.

**Later context overflows are a trade-off.** An overflow after the first step scores 0 too. The policy filled its
context with its own calls and their output, but usually on rows with long histories, where other rollouts often
overflow as well, so it also measures the row's length against the policy's context window. Judging it penalizes a
small-context policy on long rows; masking it would hide a policy that floods its context. It scores 0 so that
keeping tool output in bounds pays; list `context_overflow` in `masked_stop_reasons` to mask it instead.

**Reporting.** Neither score is enough alone: counting masked rollouts as failures mixes serving load into the
score, and excluding them flatters a policy whose hardest turns run long. The README's
[Reporting an eval](../README.md#reporting-an-eval) prints both. When time-cap masks differ between runs, also compare
the runs with their time-capped rollouts left out of both scores.

`masked_stop_reasons` changes what is masked, and a trainer can mask or rescore by `stop_reason` itself; see
[training.md](training.md#wall-clock-caps-and-masking).

## Recoverable obstacles

**Why.** The simulator is told not to invent failures, so web calls in training almost always succeed and a policy
never practices getting past one. Real web tools fail often (403s, timeouts, empty searches), and a policy that never
meets a failure may give up at the first one: a blocker report or a question to the user where a retry or another
source would have worked. `obstacle_injection` (off by default) puts such failures into the
environment, always in a form the policy can get past, so that quitting costs reward and recovering pays.

**How.**

- A share of rows (`row_rate`) is armed, by a hash of `seed` and the row's input: every rollout of a row meets the same
  obstacle, so the advantage compares behaviors rather than luck. `verifier_metadata.obstacle` fixes it per row, for
  eval sets. `score_source: reference_turn` never arms.
- An armed row draws one kind (`kinds`, by default 0.25 / 0.5 / 0.25), each recoverable by construction:
  - `transient`: the call fails once (an HTTP 500, a timeout); the same call made again is answered as usual. An
    identical call in the same reply met the same outage and fails too.
  - `blocked_route`: one URL fails (HTTP 403) for the rest of the turn; other sources work. Never a URL the user gave:
    when the user named the page, handing the problem back can be right.
  - `empty_search`: one query finds nothing for the rest of the turn; a reworded query is answered as usual.
- It fails the first web call (a page fetch or a search, `call_kinds`) that its kind applies to, if that call is among
  the turn's first `max_call_index + 1` web calls its kind can fail: page fetches for a block, searches for an empty
  search, either for a transient error. Only those count, so calls made first (file reads, shell commands, searches
  before the first fetch) do not push a web call out of the window. A web call in a reply that also calls a stop tool
  is never the first to fail: the agent runs it, but the turn ends with that reply, so the policy could not react,
  and the turn would count as a quit. URLs and queries are compared normalized (host case, `www.`, trailing slash,
  fragment; query case and spacing).
- The failure is the harness's own recorded result for that tool (`Harness.obstacle_texts`), so it looks like the
  failures already in the policy's history, in the format of the tool's results in the turn. Where the simulator's
  format varies, the text follows the tool's latest result the conversation shows: sandbox search results carry a
  `status` when recorded, but the simulator writes them without one (its prompt's format) unless the conversation shows
  recorded ones, so an injected search result has a `status` only when the latest search result shown does. Harnesses
  without recorded texts are never armed: `generic` and Cline (formats unknown), Hermes (its web results changed shape
  across versions, and `web_extract` takes several URLs at once) and Kilo Code (its web failure texts are not
  verified). Rows must name their harness to be armed: a sandbox row without `harness: sandbox` runs as `generic` and
  never is.
- The failed call is answered in code: no simulator call, and the workspace learns nothing from it. The simulator sees
  a note in its place ("a transient outage, healthy again"; "this URL is unreachable from any tool, curl included";
  "this query finds nothing"), since its "stay consistent" and "pages fail sometimes" rules would otherwise fail the
  retry too.
- Calls are told apart by their position in the turn (some parsers reuse call ids across steps), and everything
  follows from the row, the config and the turn, so `/verify` recomputes it without the session. It reports
  `obstacle`: `armed`, `fired`, `kind`, `tool`, `call_index` (among those web calls), `retried` (later calls to the
  same URL or query), `other_route` (later web calls elsewhere), `seen` (the policy replied after the failure) and
  `ended_within_1` (at most one call ran after the failure). These count only calls that ran, a stop tool's sibling
  calls included: a retry made beside a question to the user counts. Calls made beside the failed one in its reply ran
  before the policy saw the failure, so they are no retry, but a web call elsewhere among them that got a result (by
  the tool's `failed` check) is a source the turn had and counts in `other_route`: a turn that fetched another page in
  parallel and answered from it did not quit. A turn cut off before the policy's next reply (`time_cap`, `max_steps`,
  `context_overflow`, a reply cut off or failed) did nothing about the failure: `seen` is false, `ended_within_1` null,
  and the retry and quit rates and `obstacle_turn_score` leave it out. A reply holding only reasoning (`empty_output`)
  is the policy's: it saw the failure. `shell_failures` never counts the injected failure.
- The judge is not told: it sees the failure as the policy did. Recovering is judged like any other turn, and quitting
  falls under the correctness rubric's "gave up on a recoverable error". Masking and the code gates are unchanged.

**Risks.**

- **Blind retries.** A policy that always retries passes transients and fails blocks; blocks are half the mix so that
  "retry once, then switch" is what pays. A retry loop ends at the step cap and scores 0 (`unfinished_turn`).
- **Detection.** The texts are the harnesses' own and the kind and position vary, so an injected failure looks like a
  real one. The policy may also learn to retry real 404s, which costs little.
- **Avoiding the web.** A policy can dodge obstacles by not calling web tools early: watch `obstacle_fired_rate`
  (armed turns that met the obstacle) next to `turn_score`.
- **A lenient judge.** An honest quit after one failure can still pass. The correctness rubric asks for a retry after
  a transient error before a failure is reported, and attributes to `infra` only failures that persist; still compare
  the pass rates of quits and recoveries on a fixed set before training on it.
- **Time caps.** Persisting takes time, so turns that recover hit `time_cap` (masked) more often than turns that quit.
  Compute quit rates over all rollouts, masked ones included.
- **A note out of view.** The simulator sees the head and tail of a long conversation; in a turn long enough to push
  the note out, a later `curl` of a blocked URL can succeed.

## Harnesses

Harnesses differ in their tools, result formats, stop tools and wire protocol, but the session flow, the agent loop,
masking, the rubrics, voting and metrics are the same for all of them. One `Harness` object per harness isolates the
differences; a fork per harness would duplicate the shared parts and let them drift apart.

| Varies per harness | Shared |
|---|---|
| Tools answered in code, their argument names and result formats | Seed, simulate and verify flow |
| Simulator prompt and reply format (JSON, text, or JSON-or-text) | Agent loop, stop reasons and masking |
| Stop tools and tool protocol (native or Cline's XML) | The four rubrics, the verdict schema and voting |
| A short description of the agent for the judge | Code gates (per-harness malformed-argument markers and missing-argument gating) |
| How the shell reports a failure (`shell_failures`) and frames a read-back's output | Metrics and breakdowns |
| Its web tools' recorded failures (`obstacle_texts`) | Obstacle arming, injection and reporting |

**Generic fallback.** A row without a known harness runs under `generic`: every call goes to the LLM, which infers
the machine from the history and copies each tool's format from its earlier results. An honest LLM answer is better
than an exact answer in the wrong format. The stop tools are the known harnesses' ask-the-user tools plus
`attempt_completion`; plan-approval tools (`ExitPlanMode`, `plan_exit`, `plan_mode_respond`) are not among them.

## Reference-turn mode

`score_source: reference_turn` judges the row's recorded turn (`verifier_metadata.reference_turn`, with its real tool
results) instead of the policy's. Argument checks are off, since recorded traces may be incomplete, and
a recording that ends right after a tool result is masked. The policy still runs once per row; use a small model.

It measures how the original agent did with real tools, so it is a way to check the judge against your own reading of
the same turns. Compare policies with each other rather than with the reference: the reference saw real results and the
policies see simulated ones, and replaying the original's own calls on simulated results already scores lower than its
real turn.
