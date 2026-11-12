# Harnesses

How each harness answers tool calls, ends turns and frames the judge. For the overview table, see the
[README](../README.md#harnesses); for why each harness is its own object, see [design.md](design.md#harnesses).

## Selection

The harness is chosen from the row's `verifier_metadata`, at `/seed_session` and again at `/verify`:

1. `protocol: cline_xml` selects `cline_xml`, whatever `harness` says.
2. Otherwise `harness` selects the harness of that name (`oh_my_opencode` is an alias of `opencode`).
3. A row without `harness` runs under `generic`. An unknown harness name also runs under `generic`, and the server
   logs the name once.

Cline rows must carry `protocol: cline_xml`; `harness: cline` alone runs under `generic`. Every verify response names
the harness in `harness`.

All harnesses but `sandbox` show the row's system prompt to the judge and the simulator by default, read
`record_kind: subagent` as a request from a parent agent, and treat both `__malformedToolArguments` and
`__unparseable_arguments__` in a call's arguments as malformed.

The argument gates (unknown keys and wrong types against the row's tool schemas) apply to every harness whose rows
carry tool schemas; a `sandbox` row without `tools` gets none. Only `sandbox`, `claude_code` and `openclaw` also gate
a call that leaves out a required argument without a `default` (`gates_missing_arguments`), because their recorded
calls show the harness rejecting one. `hermes` and `opencode` ran recorded calls without some arguments their schemas
mark required, and it is not verified for `kilo_code` ([design](design.md#code-gates)).

## The harness workspace

Every harness but `generic` and `sandbox` shares one workspace model, fed by each harness's own file tools:

- **Paths are taken as given.** Linux and macOS paths are normalized as they are; Windows paths (drive letter or UNC)
  are compared case-insensitively with either slash. Nothing is resolved against a home or working directory the trace
  does not state, so a file named two ways is not matched and the LLM answers.
- **What it learns.** Writes and edits that succeeded, reads that returned the whole file, and the whole file when a
  write or edit result shows it. It learns from recorded results at seed time and from every answer in the turn.
- **What it forgets.** The files a mutating shell command names, by path or by basename however short (`sed -i` or
  `--in-place` in any option spelling, `perl -pi`, `awk -i inplace`, redirects, `tee`, `mv`, `rm`, `cp`, `patch`,
  `tar`, `curl -o`, `wget`, `sort -o`, `dos2unix`, `python -c`, PowerShell's `Set-Content`, `Remove-Item`, ...);
  everything after a formatter or fixer (`prettier`, `black`, `--fix`, `--write`, `go fmt`, `cargo fix`,
  `npm run format`, ...), a package manager that rewrites the manifest (`npm install x`, `yarn add x`, `npm pkg set`,
  `npm version`, ...), a version-control operation (`git checkout`, `reset`, `pull`, ...) or an opaque tool
  (subagents, code execution, patch tools); and the files named in the arguments of any other tool not known to be
  read-only.
- **When it answers.** Only when the result follows from the known content and the arguments. A result that depends
  on something it cannot know (whether a file existed, a linter's verdict, wording that changed between harness
  versions) goes to the LLM. Version-dependent wording follows what earlier results in the session show, else
  `harness_version`.
- **Shell read-backs.** `cat`, `head`, `tail` and `sed -n 'A,Bp'` of a known file by its absolute path, with no other
  shell syntax ([design](design.md#the-workspace-model)), get the shell tool's result as the harness formats it (per
  harness below). A shell call that sets anything but its command, description, timeout or working directory (a
  background run, a PTY) goes to the LLM, and so do `wc`, `head` of 0 lines (BSD `head` rejects the count), relative
  paths, output with a carriage return, and a `sed` range that ends on a last line without a line break.
- **Repeated reads.** Harnesses that answer a repeated read of an unchanged range with an "already read" stub (Claude
  Code, Hermes) leave that read to the LLM.
- **What the LLM sees.** The known contents of up to three files the call names, 30k characters each.

## Per harness

### `generic`

The fallback for rows without a harness, or with an unknown one. Every call goes to the simulator LLM,
which infers the OS, shell and paths from the system prompt and the history and copies each tool's output format from
its earlier results. The reply is the raw tool output, verbatim; an empty or cut-off reply is retried. There is no
workspace model and the simulator does not see the tool definitions.

- Stop tools: `ask_user`, `AskUserQuestion`, `clarify`, `question`, `ask_followup_question`, `attempt_completion`.
- `shell_failures`: not measured (`null`).

### `sandbox`

An agent in a Linux sandbox with 11 tools (`bash`, `read_file`, `write_file`, `edit_file`, `present_file`,
`web_search`, `fetch_page`, `image_search`, `generate_image`, `generate_speech`, `ask_user`) and a workspace at
`/home/user`. Results are JSON objects.

- **In code:** `write_file` (`{"hash", "status": "success"}`); `read_file` of a known file (`content`, `kind`,
  `lines`, `size`, `status`, and `truncated` past the read cap); `bash` read-backs of a known file (`cat`, `head`,
  `tail`, `wc -l`/`-c`, `sed -n`; paths under the call's `cwd`, by default the home directory) as a finished command,
  unless the output passes the 20,000-character stream cap; `edit_file` on a known file, matching the first
  occurrence while tolerating whitespace differences, or its "Context not found" error; `present_file`;
  `generate_image` and `generate_speech` up to the harness's per-turn limit, then its limit error; `read_file` of an
  image the history mentions (a note that the image was returned as visible content); and malformed arguments, unknown
  tools and schema violations, which get the harness's generic error (`Something went wrong. Please try again.`).
- **By the LLM:** other `bash` commands, `web_search`, `fetch_page`, `image_search`, and reads and edits of unknown
  files. The reply must hold one JSON object in the tool's format.
- **Paths:** the policy, the simulator and the judge see the same paths; nothing is rewritten. Rows must carry the
  sandbox home `/home/user`, including in `verifier_metadata.reference_turn` (the data rule `restore_sandbox_home` in
  [data.md](data.md#data-rules) restores it).
- **Workspace:** unlike the harness workspace above, it resolves relative and `~` paths against `/home/user`. It
  forgets by the same rules (everything after a formatter, a manifest-rewriting package manager or a version-control
  operation, else the files a mutating `bash` command names), matching a command's files by plain substring of the
  path or its basename. A `wc` read-back under `~` prints the expanded path, as bash does.
- **Judge:** a frame describing the sandbox; the row's system prompt is hidden unless `show_agent_system_prompt: true`.
- Stop tools: `ask_user`. `shell_failures`: `bash` results with a non-zero `exit_code` or `status: "timeout"`.

### `claude_code`

- **`Read`:** `<n>\t<line>` per line, from 1 or from `offset`, honoring `limit`; an empty file gets its empty-file
  reminder. Memory files (`CLAUDE.md`, `AGENTS.md`, `.claude/`), repeated reads and very long reads go to the LLM.
- **`Edit`:** exact-match replacement with `replace_all`, and its `<tool_use_error>` messages (no change, string not
  found, several matches). Misses involving non-ASCII text or typographic quotes go to the LLM.
- **`Write`:** an overwrite of a file whose whole content the workspace knows gets the update confirmation. A new,
  unread, partly read or since-changed file goes to the LLM (it is either created or refused as unread).
- **Confirmation suffix:** the `Write` and `Edit` confirmations end with a suffix that changed between versions. It
  follows what earlier results in the session show, else `harness_version`; when neither settles it, the LLM answers.
- **`Bash` read-backs:** the output without its trailing line breaks, or `(Bash completed with no output)`. Output
  whose first line is blank or that ends in other whitespace goes to the LLM (Claude Code trims more than line
  breaks), and so does `PowerShell`.
- Shell tools: `Bash`, `PowerShell`. Opaque: `Agent`, `Task`, `Workflow`, `SendMessage`, worktree tools.
- Stop tools: `AskUserQuestion`, `ExitPlanMode`. `shell_failures`: an `Exit code N` first line or a timeout on `Bash` /
  `PowerShell`.

### `hermes`

- **`read_file`:** its JSON result with line numbers, the default page size of the row's version, and its totals and
  hints. Line numbers follow an earlier read in the session; otherwise `harness_version` (padded before 0.17,
  unpadded from 0.17); with neither, the LLM answers.
- **`write_file`, `patch`:** answered only for rows that state a version below 0.12, since later versions added result
  keys the workspace cannot know; rows without `harness_version` go to the LLM. `patch` is answered only for files its
  linters do not check, with a unique exact match (or several non-overlapping ones with `replace_all`) that ends more
  than three lines before the end of the file.
- **Replies** are JSON-or-text: a reply that opens a JSON object must hold a complete one; anything else is the output
  as text.
- **`terminal` read-backs:** `{"output", "exit_code": 0, "error": null}` with the output stripped. Once a session's
  results show its command screening (a flagged command gets an `approval` note), read-backs go to the LLM.
- Shell tool: `terminal`. Opaque: `delegate_task`, `execute_code`, `process`.
- Stop tools: `clarify`. `shell_failures`: a non-zero `exit_code` in a `terminal` result.

### `openclaw`

- **`read`:** the file's text as it is, for whole-file reads only (no `limit`; `offset` absent or 1), up to 30k
  characters. Empty files go to the LLM.
- **`write`, `edit`:** one-line confirmations; `edit` takes `edits: [{oldText, newText}]`, each matched in the original
  text and required to be unique, with errors in its JSON envelope. Misses (in versions that add closest-match hints,
  or rows without `harness_version`), edits involving typographic characters or trailing spaces, and writes of unknown
  files once the session shows sandbox refusals go to the LLM.
- **Flat `edit` form:** from 2026.5.28 (OpenClaw's own `edit`), a top-level `oldText`/`newText` pair becomes the last
  `edits` entry, as OpenClaw prepares the call, for the argument gates, the workspace and the simulator; the judge sees
  the call as made. Earlier versions (pi-coding-agent's `edit`, unverified) and rows without `harness_version` are
  gated on the schema as it is.
- **`exec` read-backs:** the output without its trailing line breaks, or `(no output)`; output that ends in other
  whitespace goes to the LLM.
- Shell tool: `exec`. Opaque: `apply_patch`, `process`, session and subagent tools.
- Stop tools: `ask_user`. `shell_failures`: `(Command exited with code N)`, command not found, killed by a signal, or
  timed out on `exec`.

### `opencode` (alias `oh_my_opencode`)

- **`read`:** `<path>`, `<type>file</type>` and `<content>` with `N: line` lines, then its end-of-file or page notice,
  for POSIX absolute paths. Windows paths go to the LLM.
- **`write`, `edit`:** `Wrote file successfully.` and `Edit applied successfully.`, only for files no language server
  checks (a language server may append diagnostics); `edit` only with a unique exact match or `replaceAll`. Sessions
  that show permission rules go to the LLM.
- **`bash` read-backs:** the output as printed. No output (its note differs between versions) and long output go to
  the LLM. A `read` result does not show whether the file ends with a line break, so a read-back that reaches the end
  of a file known only from a read goes to the LLM too.
- Shell tools: `bash`, `execute_command`, `interactive_terminal`. Opaque: `task`, patch tools, `new_task`.
- Stop tools: `question`, `plan_exit`. `shell_failures`: not measured (its `bash` results show no exit code).

### `kilo_code`

Its CLI generation shares opencode's tools and formats and is answered the same way. The legacy extension's tools
(`read_file`, `apply_diff`, `execute_command`, ...) all go to the LLM.

- Stop tools: `question`, `plan_exit`, `ask_followup_question`, `attempt_completion`.
- `shell_failures`: `Exit code: N` on legacy `execute_command` results; rows without that tool are not measured.

### `cline_xml`

Cline rows (`protocol: cline_xml`) use Cline's XML tool protocol, described below. Their `tools` list is empty: the
tool definitions shown to the simulator and the judge are rebuilt from the tool documentation in the row's system
prompt. Every call goes to the LLM, because each Cline result ends with editor state (open tabs, the time, context
usage) that the workspace cannot know. The workspace still follows files (`read_file`'s `N | line` labels, and the
whole file Cline shows after a write or edit), so the simulator sees their current contents.

- Stop tools: `ask_followup_question`, `attempt_completion`, `plan_mode_respond`.
- `shell_failures`: `execute_command` results that report a failed exit code or a timeout.
- No argument gates: Cline rows carry no tool schemas.
- A recorded turn that ends with `ask_followup_question` is judged without the user's answer.

## Cline's XML protocol

Cline gives the model no function tools. The model writes calls in its reply text:

```xml
<read_file>
<path>src/config.ts</path>
</read_file>
```

Cline runs a reply's calls in order and answers all of them in the next user message: each result starts with a
header such as `[read_file for 'src/config.ts'] Result:`, and the message ends with `<environment_details>`.

**Parsing** (the agent and the server use the same rules): only Cline's known tool names are calls; parameters are the
call's top-level `<param>value</param>` children; tool tags inside a call's body belong to that call; an unclosed tag
runs to the end of the text.

**The agent** sends a reply's calls to `/simulate_tool_call` in order and returns the results as one user message;
see the [agent README](../../../responses_api_agents/agentic_turn_simulation_agent/README.md#clines-xml-protocol).

**The server** adapts the transcript for the simulator, the judge and the code gates: XML calls become function calls,
an answering user message is split at the result headers into one output per call (text before the first header, such
as the user's feedback on an approval, goes with the first), and `attempt_completion` / `plan_mode_respond` become the
agent's message. It frames each simulated result with Cline's header and keeps the environment details only on the
reply's last result.

**Policies trained on native tool calls** may not follow the protocol. A model can wrap the XML in its own tool-call
markup (`<tool_call>read_file>...</read_file>`), which Cline cannot parse and which the serving stack's tool-call
parser may strip from the reply. The turn then gets Cline's "no tools used" errors and ends on `mistake_limit`, as it
would in Cline. Check a few Cline rollouts before comparing policies on Cline rows.

## Adding a harness

Start with `generic` and add only what the harness needs.

1. **Run it under `generic`.** Set `harness: <name>` on the rows and run a few rollouts. Check that the turn ends where
   the harness would end it: if its ask-the-user or finish tool is not among `generic`'s stop tools, the agent will
   execute it instead of stopping.
2. **Register it with its stop tools**: the harness's ask-the-user, plan-approval and finish tools. A call to one
   never runs and ends the turn once the reply's other calls have run. A new module in `harnesses/` declares it, and
   `harnesses/__init__.py` lists it in `HARNESSES`:

   ```python
   GOOSE = Harness(
       name="goose",
       simulator_prompt="tool-simulator-goose.md",
       judge_prompt="judge-system-harness.md",
       judge_environment="judge-environment-goose.md",
       stop_tools=("ask_user",),
       simulator_sees_tools=True,
   )
   ```

   The server loads every registered harness's simulator prompt at startup and its judge environment line at
   `/verify`, so write `prompts/tool-simulator-<name>.md` and `prompts/judge-environment-<name>.md` (steps 3-4) before
   starting it, or begin with `simulator_prompt="tool-simulator-generic.md"` and no `judge_environment`.
3. **Write its simulator prompt**, `prompts/tool-simulator-<name>.md`. Start from a similar harness's prompt: describe
   the environment (the user's own machine, inferred from the history), each tool's output format and errors in your
   own words from recorded results, and keep the "How to simulate" rules as they are. Pick the reply format: raw text,
   or JSON-or-text for harnesses whose tools answer in JSON. With `simulator_sees_tools=True` the simulator also sees
   the row's tool definitions.
4. **Add the judge environment line**, `prompts/judge-environment-<name>.md`: one sentence on what the agent is and
   where it runs. It fills the shared harness judge frame; the rubrics stay the same.
5. **Optional: a shell-failure count** read from the harness's shell result format, reported as `shell_failures`.
6. **Optional: `gates_missing_arguments=True`** when the harness's tools reject a call that leaves out a required
   argument. Check its recorded calls first: if calls missing a required argument got normal results, leave it off.
7. **Optional: file tools and their format.** Map the harness's read, write and edit tools to the canonical operations
   by their argument names; write the format that renders and reads back their results; and list its shell tools,
   opaque tools and read-only tools so the workspace knows what to forget. Decline (leave to the LLM) every result the
   workspace cannot determine exactly; a wrong exact answer is worse than an LLM answer.
8. **Optional: `obstacle_texts`**, so that [recoverable obstacles](design.md#recoverable-obstacles) can be injected into
   its rows: for each web tool, the argument holding the URL or query, its results for a failed call as the
   recordings show them (a transient error, a blocked page, a search without results), and `failed`, which tells a
   failed result of the tool from one that brought something. Compare the texts with the simulator's results for that
   tool, not only the recordings': a field the simulator leaves out gives an injected failure away (a text can follow
   the tool's latest result shown, as the sandbox's searches do). A kind without a recorded text is never injected; a
   harness without any is never armed.
9. **Test with synthetic rows.** Exact answers must match the harness's recorded results byte for byte. Then replay a
   sample of recorded calls through the simulator and compare its results with the real ones before trusting scores.

A harness with its own wire protocol (like Cline's) also needs agent support for parsing calls and returning results.
