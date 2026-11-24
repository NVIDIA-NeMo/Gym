# Harbor conversion checks

Checks and probes for converting a native Gym environment (JSONL rows, a Resources Server, an agent and YAML) to
Harbor task folders. They cover the parts of a conversion that `gym dataset validate` and `python -m
scripts.harbor_parity` do not: whether the instruction is the native prompt, whether hidden data reaches the agent,
negative controls and reward-forgery probes, a scripted model for native oracle runs, the first model request of the
two paths, and a reproducible task subset.

Run them from the repository root. They use the standard library only and stream JSONL files line by line.

```bash
python -m scripts.harbor_conversion_checks <command> --help
```

Exit codes: 0 pass or done, 1 a check failed, 2 a usage or input error (printed to stderr).

Gate order: G1a determinism (two conversions into two parent folders with the same file names, `diff -r` empty) -> G2
oracle (`gym dataset validate`) -> G1b content check (`check`; for a parameterized template on
`results/harbor/<dataset folder name>/rendered` after G2, for one folder per task on the second conversion, which no
validate touches) -> G3 negative controls (`controls`, from a never-validated conversion, under their own dataset
folder name, into a folder that must not exist) -> G4 native smoke -> G5 Harbor smoke -> G6 comparison
(`first-request`, `python -m scripts.harbor_parity`). The `convert-env-to-harbor` skill's `references/gates.md` has
each command.

## `subset`: pick the task ids (step 3)

```bash
python -m scripts.harbor_conversion_checks subset data/train.jsonl --id-field instance_id --count 50 --seed 20261122 \
    --include data/supported_instance_ids.txt --exclude unawardable_ids.txt --out parity_ids.txt
```

`--id-field` is a dotted field of the row, or `_index` for the 0-based line. Ids are ordered by
`sha256(f"{seed}:{id}")`, kept only if listed in `--include`, dropped if listed in `--exclude` (for example rows whose
reference answer the native grader does not award), and the first `--count` are written after a header naming the
rows file's sha256 and line count. The same command with `--count 5` writes the first 5, the smoke subset: convert it
as its own dataset rather than selecting it with `--limit`, which picks tasks in a different order on each path.
Fewer eligible ids than `--count` exits 1 and writes nothing; a duplicate id exits 2. It prints how many `--include`
ids are rows (an include list is usually the server's list for the whole dataset) and lists `--exclude` ids that are
not rows (a mistyped id shows up there).
Pass the file to the adapter's `--ids-file`.

## `check`: content checks of the converted folders (G1b)

```bash
python -m scripts.harbor_conversion_checks check results/harbor/<taskset>/rendered \
    --native-rows native_subset.jsonl --native-id _index --id-map id_map.json --prefix-file system_prompt.txt \
    --hidden-from resources_servers/<server>/task_data.py --template <template> --template-hidden expected_answer \
    --json check.json
```

DATASET is a task folder or a folder of them, found as Gym finds them (task folders at any depth up to 6 levels,
hidden folders skipped, symbolic links not followed, two task folders of one name refused). A parameterized template
is refused with exit 2; pass `results/harbor/<dataset folder name>/rendered`, which only `gym dataset validate` or
`gym eval run` writes, under the directory `gym` ran in. There is no render-only command, so for a template, `check`
runs after G2. The checks:

- `instruction` (needs `--native-rows`, `--native-id`): fails when `instruction.md`, minus leading canary lines,
  differs from `<prefix file><system messages joined by a blank line><blank line><user message>` of the task's native
  row, or the row has not exactly one user message. Gym reads `instruction.md` in text mode, so a native prompt with
  CRLF (GitHub issue bodies) reaches the model with LF; such a task fails with "differs only in line endings" until
  you pass `--normalize-newlines`, which counts those tasks in the detail line. Record it as a difference.
- `hidden-values` (needs `--hidden FIELD` or `--hidden-from task_data.py`): fails when a hidden native value is in
  `instruction.md`, a `steps/<name>/instruction.md`, the effective `[environment].env` variables (`task.toml`'s with
  the dataset's `dataset.toml` `[gym.tasks."<task>".environment.env]` merged over them, as Gym's loader merges them),
  the effective `mcp_servers`, `skills_dir` and `healthcheck` (which the harbor Environment Server hands the harness
  or writes into the agent's sandbox), or a file under `environment/` or a `steps/<name>/workdir/` (uploaded into the
  agent's working directory before its turn). The `tests/` and `solution/` folders, the task's and each step's, stay
  with the verifier and the oracle and are not scanned. A `dataset.toml` that Gym would not load, or an override of a
  shape this does not model, fails as not scanned. Every file, the name and value of every effective `env` variable,
  and every string of those other settings goes through one traversal: each base64 run of 1024+ characters (one
  line, or wrapped as `base64` wraps) is decoded, a gzip is gunzipped (every member of the stream) and a tar is
  untarred, and each of those layers is traversed the same way, so a base64 bundle and the same file on disk are
  scanned alike. A zip, xz or bz2 file, base64 payload or tar member is reported as not scanned. JSON-escaped copies
  of a value are found, but case-changed copies and other encodings are not scanned and must be read by hand
  (`first-request` is the stronger evidence).
- `hidden-template` (needs `--template`, `--template-hidden NAME`): fails when a template file Gym renders
  (`<stem>.template<suffix>`) outside `solution/` and `tests/` has the placeholder `{{ NAME }}` or
  `{{ task_data.NAME }}`. These are the only placeholders Gym renders; any other `{{ ... }}` is refused at render.
- `secrets`: fails when any file has an API key or private key pattern: `sk-`, `nvapi-`, `hf_`, `ghp_` (and the
  other `gh?_` GitHub tokens), `github_pat_`, `glpat-`, Slack `xox?-`, `AKIA` or a `-----BEGIN ... PRIVATE KEY-----`
  block.
- `local-paths`: fails when any file names your home directory, the current directory, `/lustre/`, `/raid/` or
  `/scratch/`. Your home and the current directory count only as whole paths (`/home/al` is not found in
  `/home/alice`), and a home of `/` or `/root` is not scanned. Another user's home directory is not flagged, only
  yours: read the prompt and the shipped files for such paths by eye.

A check whose inputs are absent prints `SKIP`. What a check cannot scan in full fails it, naming the file: a zip, xz
or bz2 file, base64 payload or tar member, a gzip that does not gunzip, ends early or has other bytes after it, a run
laid out as base64 that does not decode, a layer nested more than 8 deep (each base64 run, gunzip and tar member is
one layer), or anything past 256 MiB of decoded bytes per file or `env` name or value (all layers together).

`secrets` and `local-paths` scan every file under DATASET, so on a folder that was already validated they also scan
the records validation wrote there (`compose-images.json`, `compose-images.json.lock`, `manifest.toml`).

The task id is the folder name. `--id-map` maps a native id to a task id when the adapter renames tasks (the same
file `harbor_parity --id-map` takes). The prefix file is used byte for byte. For a pairing that joins `system_prompt`
and the user message with two newlines, write `system_prompt + "\n\n"` with Python, for example
`python -c "import yaml; c = yaml.safe_load(open(CONFIG)); open(OUT, 'w').write(<path to system_prompt> + '\n\n')"`.
A YAML `|` block already ends in one newline, which leaves two blank lines before the question; shell `$(...)` strips
that newline, and the `instruction` check then fails on every task.

`--hidden-from` reads the Resources Server's own `task_data.py` and hides every `TaskData` field whose
`json_schema_extra["consumed_by"]` lists neither `"prompt"` nor `"provenance"`: what only the verifier reads. Ids and
repository names (`provenance`) are meant to be visible. Inherited fields count: a base class is followed through
`from MODULE import NAME` to that module's file in the repository, and a base it cannot follow (or a `TaskData` that is
not a class) exits 2; name the fields with `--hidden` instead.

Hidden values are the string and number leaves of each hidden field: the whole value if it has 8 or more characters,
each line of 40 or more characters of a multi-line value (shorter lines are boilerplate such as
`if __name__ == "__main__":`), and the added lines of 20 or more characters of a patch. A string holding a JSON array
(`FAIL_TO_PASS`) also counts each item; a string holding a JSON object (tool-call `arguments`) counts only whole,
since its fields are values the agent looks up in the environment's data. The detail line counts each distinct
hidden string of a task once and sums over the tasks, so each test name in a list counts once and a large count (1089
for 5 SWE rows) is expected. Short values (under 8 characters) are covered only by `--template-hidden` on a
parameterized template; on a dataset of one folder per task they are not scanned. The keys of a hidden mapping (a
status map keyed by test id) are not scanned, only its values: where the native row also holds those keys as a field
of its own (a list of test ids), name that field with `--hidden`. An explicit `--hidden` field that none of the tasks'
native rows has exits 2 (a typo, or a nested field named without its parent); a `--hidden-from` field missing from
every row is noted as `absent from every row`. A value is dropped, and not counted, when the agent sees it by design,
so the count can be lower than the number of hidden leaves:

- it occurs in the row's prompt or in the `tools` the row offers (tool names and descriptions), so pass rows that
  still carry their `tools` for a tool environment;
- it equals the task's effective `[environment]` `docker_image` or `workdir` (`task.toml`'s, with a `dataset.toml`
  override applied) or an image a `FROM` line of `environment/Dockerfile` names, with or without its `@sha256:`
  digest. No other `[environment]` setting is exempt: `env`, `mcp_servers`, `skills_dir` and `healthcheck` reach the
  agent and are scanned.

A short generic value can also occur by coincidence in the code the adapter ships (a test named `continue` inside a
vendored module). Read each match, and list the ones that are not row data in a file passed as `--allow-values` (one
value per line, `#` comments): their matches under `environment/` or a step's `workdir/` are printed as `allowed:` lines and counted, not
failed. A match can sit in a base64 bundle inside `environment/Dockerfile`, reported as
`<task>/environment/Dockerfile[base64 at N]:<member>`, so a common test name such as `continue` matches the vendored
seed code. A match in an `instruction.md` or in `[environment].env` always fails. Keep the file with the evidence.

This check restates the adapter's rule; comparing the first model request of both paths (`first-request`) is the
stronger evidence. Loading, unknown `task.toml` keys and skipped tasks are `gym dataset validate`'s job.

## `controls`: negative controls and reward-forgery probes (G3)

```bash
python -m scripts.harbor_conversion_checks controls <dataset> <dataset>_empty --empty
python -m scripts.harbor_conversion_checks controls <dataset> <dataset>_wrong --solution-dir wrong_solution/
python -m scripts.harbor_conversion_checks controls <dataset> <dataset>_plant --plant-reward
python -m scripts.harbor_conversion_checks controls <dataset> <dataset>_conftest --plant-conftest
python -m scripts.harbor_conversion_checks controls <dataset> <dataset>_hooks --plant-hooks
gym dataset validate <dataset>_empty --sandbox <provider> --config harbor_arm.yaml
```

DATASET is a parameterized template or a folder of task folders at any depth (found as `check` finds them): use a
conversion no validate has touched. OUT must not exist, and its name is the control's dataset folder name, which keys
its results, so give each control its own (`<dataset>_wrong`, not the oracle's). The copy keeps everything (symlinks
as links) except each `solution/`, which is replaced:

- `--empty`: a `solve.sh` that does nothing.
- `--solution-dir DIR`: DIR's files, a wrong answer (for a text answer, `answer.template.md` holding
  `\boxed{-987654321}`; for a tool environment, a solution that makes one wrong call).
- `--plant-reward`: writes `{"reward": 1.0}` to `/logs/verifier/reward.json` and `1` to `reward.txt`, puts `python3`
  and `python` stubs that write them again into `/usr/local/bin`, and keeps rewriting both for `PROBE_SECONDS`
  (default 120) seconds. In a shared sandbox it tests whether the agent can forge the reward; under separate
  verification it shows that separate verification is on.
- `--plant-conftest`: leaves a `conftest.py` in the working directory, the repository a patch hook captures. When the
  verifier's pytest imports it, it writes the same reward files, starts a daemon rewriting them for `PROBE_SECONDS`
  seconds and, unless `PROBE_KILL=0`, kills every process with an argument starting with `PROBE_KILL_PREFIX` (default
  `/tests/`). Use it when the verifier runs agent code.
- `--plant-hooks`: the same payload, started from what the verifier's install or test commands run in a repository of
  another language: npm lifecycle scripts (`preinstall`, `prepare`, `pretest`, `prejest`, `posttest`; written only
  when the image has `node`), an `init()` in a Go test file next to the first `*_test.go`, and a root `conftest.py`.
  Rust `build.rs` and Maven plugins are not covered: write your own `--solution-dir` probe for those.

A multi-step task's `steps/<name>/solution/` folders are dropped (the server would place them in preference), so the
one replaced top-level `solution/` applies to every step.

`PROBE_ROOT` (default empty) prefixes every path the probes write, for trying them outside a sandbox.
`PROBE_KILL_PREFIX` (default `/tests/`, also when empty) is the argument prefix of the processes the kill step kills:
outside a sandbox, set it to a folder of your own (or set `PROBE_KILL=0`), since the default kills every process of
your user with an argument under `/tests/`. Every control must score 0 and must not be masked: read the rewards from
`results/harbor/validation/<dataset folder name>/rollouts_run_summary.json` (`gym dataset validate` exits 1 on a
control by design and labels its rows `oracle solved` even at reward 0). A probe may score 0 with `failure_kind`
`harbor:missing_reward`.

### A partial-credit probe (a parity check, not a control)

When the native grader gives partial credit, check that both arms give the same reward between 0 and 1 on every
task, not only at the endpoints. This copy is never a 0 control: do not count it as one.

1. For each task, derive one partly wrong answer from the native row (for example the reference with one item
   changed), and check with the native grader that it scores strictly between 0 and 1. In a knights-and-knaves
   puzzle, the reference with the first person's role flipped scores 0.65 for two people.
2. `python -m scripts.harbor_conversion_checks controls <template> <template>_partial --solution-dir partial/`, where
   `partial/answer.template.md` writes the answer in the task's format around `{{ task_data.partial_answer }}`.
3. Add `"partial_answer": "..."` to the `task_data` of every row of `<template>_partial/tasks.jsonl`, and declare
   `partial_answer: str` in `<template>_partial/task_data.py`. `gym dataset validate` refuses an undeclared key with
   exit 2.
4. Native arm: run `fake-model` with a replies file of `{"match": <question>, "text": <the same answer in the native
   format>}`.
5. Compare with `python -m scripts.harbor_parity ... --strict` without `--expect`. It passes when every task's reward
   is equal on both arms.

Only `solution/` renders the field. Keep it out of agent-visible files and confirm with
`check --template <template>_partial --template-hidden partial_answer`.

## `fake-model`: a scripted model for native oracle and control runs

```bash
python -m scripts.harbor_conversion_checks fake-model --port 18731 --replies replies.jsonl \
    --reply "I could not solve this."
# in another shell: the native arm with the scripted model
gym eval run ... --model-type openai_model --model <model name> --model-url http://127.0.0.1:18731/v1 \
    ++observability_enabled=true ++model_call_capture_dir=calls/
```

An OpenAI chat-completions endpoint on 127.0.0.1 (`--port 0` picks a free port and prints it). It serves chat
completions only (OpenCode); an agent that calls the Responses API (`/v1/responses`, such as simple_agent) needs its
own fake upstream. `POST /v1/chat/completions` answers with JSON, or server-sent events when the request has
`"stream": true` (with `usage` on the last chunk when `stream_options.include_usage` is set). `GET /v1/models` lists
the last model requested. Each reply is the `text` of the first `replies.jsonl` row `{"match": ..., "text": ...}`
whose `match` occurs in the last user message, as sent or with OpenCode's argv quoting undone; else `--reply`; else
`no-match`. The file is re-read on every request, so one server can serve an oracle run and then a wrong-answer run.
For a text-answer environment, build the file from the native rows, for example
`{"match": <question>, "text": "Answer: \\boxed{<expected>}"}`. It keeps no log: capture the requests with
`++observability_enabled=true ++model_call_capture_dir=DIR`. Environments whose grade depends on tool calls use their
own native control instead (a golden patch, or a direct `/verify`).

## `first-request`: compare what the model saw on both paths (G6)

```bash
python -m scripts.harbor_conversion_checks first-request native_calls/ harbor_calls/ --out first_request/ \
    --json first.json
```

Each folder holds the `*.capture.jsonl` files a run writes with `++observability_enabled=true
++model_call_capture_dir=DIR`, one per rollout. A file's first agent request is its earliest request that offers
tools, which skips OpenCode's title call. Tasks are paired by the first user message: exactly, or with OpenCode's argv
quoting undone (noted as `argv_prompt_quoting`) and CRLF or CR read as LF (noted as `newlines`: Gym reads
`instruction.md` in text mode); notes are not failures. For each pair it compares the model, the tools (their
names, then each tool's whole definition, such as its description, parameter schema and `strict`, in any order of
tools and of JSON keys; a changed definition reports `tools differs: the definition of NAME`), every other top-level
request key except `messages`, `tools`, `stream` and `stream_options` (the sampling settings: `temperature`,
`max_tokens`, `tool_choice`, `top_k`, `seed`, `chat_template_kwargs` and so on), the message roles, and every
message but the system and developer ones, whole and in order: content text and content parts (in order), tool calls,
tool results and any other field (a change reports `messages differ: non-system message N (role)`). Two things are
tolerated: the first user message's text, which the pairing already matched (exactly, or with argv quoting undone and
newlines read as LF; its other content parts are still compared), and the system prompt, which is diffed on its own
and whose dynamic lines (provider prefix, working directory, date) are reported, not failed. It prints a unified diff
of the system prompts, and counts `AGENTS.md`, `CLAUDE.md`, `Instructions from` and
`<available_skills>` in every message but the assistant's. `--out` writes each pair's system prompt and user message
to `DIR/<sha256 of the user message, 12 hex>/{native,harbor}.{system,user}.txt`. A capture file with no request that
offers tools (a rollout that died before its first agent call) is listed as `no agent request`, not counted as
unpaired. Every capture is compared: the repeats of a prompt (and distinct tasks that share one) pair in file-name
order, a capture left over on one side is unpaired, and `--out` writes the second and later pairs of a prompt to
`<key>.2`, `<key>.3` and so on.

It exits 1 when there is no pair at all (0 pairs fails), on an unpaired task, a model, tool, sampling, role
sequence or non-system message difference, or a flag whose count differs between the sides or that sits on a
system-prompt line that differs (a host's `Instructions from: .../AGENTS.md` on one side). OpenCode's
built-in prompt names `AGENTS.md` and lists its built-in skills in `<available_skills>` identically on both paths: such
flags are noted, not failed. Other system-prompt lines that differ, such as the working directory or the date, are
reported only.
