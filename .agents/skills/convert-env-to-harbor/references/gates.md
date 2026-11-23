# Probes, gates and evidence

Commands run from the repository root unless noted. `<model flags>` is `--model-type <type> --model <model>
--model-url <url>`; `<provider>` the sandbox provider; `harbor_arm.yaml`, `harbor_arm_harness.yaml` and
`native_arm.yaml` the run configs of the [adapter contract](families.md#adapter-contract). `harbor_arm.yaml` goes on
every Harbor-arm `gym dataset validate` and `gym eval run` (for a Compose family it carries the deployment settings);
`harbor_arm_harness.yaml` on `gym eval run` only. `<map.json>` is the adapter's `--id-map-out`. Use absolute paths
for datasets, configs and capture directories. Results land under `results/` in the directory you run from, keyed by
the dataset folder's name: `gym dataset validate` in `results/harbor/validation/<dataset folder name>/`, `gym eval
run` in `results/harbor/<dataset folder name>/`. Both render a parameterized template into
`results/harbor/<dataset folder name>/rendered/` and leave the template unchanged; on a folder of task folders they
write `manifest.toml` (unless git tracks the folder) and, when a task has a Dockerfile or a Compose file, the image
records (`compose-images.json` and its `.lock`) into the dataset folder itself.

Reading results: `gym dataset validate` exits 1 whenever a task's oracle does not score 1, which is what a negative
control should do, and it labels such a row "oracle solved" with reward 0. Read
`results/harbor/validation/<dataset folder name>/rollouts_run_summary.json`: `counts`, and per task `tasks[].reward`,
`tasks[].mask_sample`, `tasks[].failure_kind` and `tasks[].verifier.mode`. `failure_kind` null means a measured 0,
`harbor:missing_reward` means no reward file. The line `Get command status failed: Sandbox ... not found` after a
Compose group stops is noise.

Gate order: G1a determinism (two conversions into two parent folders with the same file names, `diff -r` empty) -> G2
oracle -> G1b content check (for a parameterized template on `results/harbor/<dataset folder name>/rendered` after G2;
for one folder per task on the second G1a conversion, which no validate touches) -> G3 negative controls (from a
never-validated conversion, under their own dataset folder name, into a folder that must not exist) -> G4 -> G5 -> G6.

## Infrastructure probes P1-P6

Write the probes into a scratch folder outside the repository. Each needs `instruction.md`; the scaffold has one.

| Probe | Command | Pass |
| --- | --- | --- |
| P1 sandbox | `gym dataset validate environments/hello_world --sandbox <provider>` | `oracle solved 1.0` |
| P2 image pull | `gym dataset init probe --root <dir> --image <image>` then `gym dataset validate <dir>/probe --sandbox <provider>` | `oracle solved 1.0`; one per image family or language |
| P3 runtime | In P2's `<dir>/probe/hello/solution/solve.sh`, before its last line: `set -e`, `python3 -c 'import sys; assert sys.version_info >= (3, 10)'`, `command -v timeout bash git` (plus what grader and hooks need); validate again | `oracle solved 1.0`; `oracle failed` means something is missing |
| P4 egress | Two probe datasets from P2's scaffold, each fetching every host the install, grader or tools use (PyPI, GitHub, npm, crates, the Go proxy, the Maven mirror in `responses_api_agents/swe_agents/maven_mirror/`) with a line such as `python3 -c 'import urllib.request; urllib.request.urlopen("https://pypi.org/simple/", timeout=20)' \|\| exit 1`. Agent sandbox: put the lines in `solution/solve.sh` before `echo hello`. Verifier sandbox: in a second copy, set `environment_mode = "separate"` under `[verifier]` in `task.toml` and replace `tests/test.sh` with `#!/bin/bash`, `mkdir -p /logs/verifier`, the fetch lines, `echo 1 > /logs/verifier/reward.txt` (a fresh verifier has no `hello.txt`, so the scaffold's check would score 0; and the scaffold's `test.sh` has no `set -e`, so without `\|\| exit 1` a failed fetch still writes 1) | 1.0 on both; a failed fetch shows as `oracle failed` (agent) or 0 with `harbor:missing_reward` (verifier) |
| P5 Compose | A two-service task as in [Multi-Container Tasks](../../../../fern/versions/latest/pages/data/task-folders/compose-tasks.mdx): `main` reads a file a sidecar binds and a value the sidecar installed; `gym dataset validate <probe> --sandbox opensandbox --config <deployment.yaml>` with `sandbox.opensandbox.networking.enabled: true` | 1.0 (workplace_assistant: 90 s, no shared storage needed) |
| P6 judge (unpiloted) | `test.sh` calls the judge with a key from `[verifier].env` `${NAME}`, filled by the harbor server's `secrets`; nothing in the agent's sandbox | 1.0 only when the judge answered |

An image without Python cannot run the `python3` fetch line: test a host with curl instead, and count any HTTP answer
as reachable, since a bare registry root answers 404 (files.pythonhosted.org, a Maven mirror) or 429 (Maven Central)
and `curl -f` would fail on it:

```bash
reach() {
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$1")
  [ "$code" != 000 ] || { echo "unreachable $1"; return 1; }
}
reach https://repo.maven.apache.org/maven2/ || exit 1
```

**Per-row images (SWE):** one probe per question and image costs 3 image pulls per language. Build one dataset with
two tasks per image instead, one shared and one with `environment_mode = "separate"`, whose `test.sh` prints the
diagnostics (Pythons and versions, `timeout bash git curl tar sha256sum`, `grep CapBnd /proc/self/status`, one
`reach` line per host) and writes 1 only when P3 holds; read the answers from each task's `test-stdout.txt`. The
swe_rebench dry run probed 4 images this way in one 173 s run.

## G1a determinism

```bash
for d in a b; do
  mkdir -p <data>/$d   # some adapters (scale_swe) exit 2 when the side outputs' folder does not exist
  python resources_servers/<server>/harbor/adapter.py --rows <rows.jsonl> --output-dir <data>/$d/<dataset> \
      --ids-file <ids.txt> --native-rows-out <data>/$d/native.jsonl --id-map-out <data>/$d/id_map.json
done
diff -r <data>/a <data>/b
ruff check <adapter files> && ruff format --check <adapter files>
```

Pass: `diff -r` silent. Convert into two parent folders with the same file names: the dataset README names the
inputs and side outputs by basename, so differently named side files differ there. Run it before any validate (see
the results paragraph above). Also run the adapter once from a copy outside any git checkout.

## G1b content check

```bash
python -m scripts.harbor_conversion_checks check <checked folder> \
    --native-rows <data>/a/native.jsonl --native-id <field|_index> [--id-map <data>/a/id_map.json] \
    [--prefix-file <file>] --hidden <field> --hidden-from resources_servers/<server>/task_data.py \
    [--template <data>/a/<dataset> --template-hidden <name>] --json g1_check.json
```

`<checked folder>`: `check` refuses a parameterized template with exit 2 and has no render step; only a validate or
eval run writes `results/harbor/<dataset folder name>/rendered`, under the directory `gym` ran in, so for a template
pass that folder after G2. For one folder per task, pass `<data>/b/<dataset>`, which no validate touches.
Pass: `check: <p> passed, 0 failed`.
Two options record what you reviewed instead of failing on it: `--normalize-newlines` for native prompts with CRLF (a
difference to record, Step 5), and `--allow-values <file>` for short hidden values that match shipped code by
coincidence (swe_rebench: test names such as `continue` inside vendored modules; it covers files under
`environment/` only: a match in `instruction.md` or `[environment].env` always fails); keep both in the evidence.
Loading and unknown `task.toml` keys are checked by `gym dataset validate` (G2): an overlay task loads only after
prepare has recorded its base image.

## G2 oracle

```bash
gym dataset validate <data>/a/<dataset> --sandbox <provider> --config harbor_arm.yaml [--concurrency 4]
```

Pass: every task `oracle solved 1.0`, 0 masked; with separate verification every row shows `verifier_mode: separate`.
Native oracle on the smoke dataset's `--native-rows-out`: the fake-model oracle or the server's golden mode
([native-baseline.md](native-baseline.md#model-free-native-runs)). Then:

```bash
python -m scripts.harbor_parity --native <native oracle rollouts.jsonl> --native-key <field|_ng_task_index> \
    [--id-map <map.json>] --harbor results/harbor/validation/<dataset>/ --strict --expect 1
```

`--native-key _ng_task_index` with the id map when the task ids are not a field of the rows; see the
[adapter contract](families.md#adapter-contract).

Before the full parity run, validate every task of the parity dataset.

## G3 negative controls

```bash
B=<data>/b/<dataset>   # the second G1a conversion, which no validate has touched
python -m scripts.harbor_conversion_checks controls $B <data>/<dataset>_empty --empty
python -m scripts.harbor_conversion_checks controls $B <data>/<dataset>_wrong --solution-dir <wrong solution dir>
python -m scripts.harbor_conversion_checks controls $B <data>/<dataset>_plant --plant-reward
# only when the verifier runs agent code (pytest; --plant-hooks for npm, Go and pytest repositories):
python -m scripts.harbor_conversion_checks controls $B <data>/<dataset>_conftest --plant-conftest
python -m scripts.harbor_conversion_checks controls $B <data>/<dataset>_hooks --plant-hooks
gym dataset validate <data>/<dataset>_wrong --sandbox <provider> --config harbor_arm.yaml   # one run per control
```

The output folder must not exist, and its name is the control's dataset folder name: results are keyed by it, so a
control named like the oracle dataset would share the oracle's results folder.

`<wrong solution dir>` holds the files of a wrong answer (math: `answer.md` with `\boxed{-987654321}`;
workplace_assistant: one state-changing call no row asks for). Choose "wrong", not "empty", where an empty answer
can be right (empty ground truth), and one the scorer gives no partial credit (reasoning_gym: a number, no
names; to compare partial credit itself, run the
[partial-credit probe](../../../../scripts/harbor_conversion_checks/README.md#a-partial-credit-probe-a-parity-check-not-a-control),
which is not a control). Pass: every control scores 0 unmasked, `failure_kind` null for `--empty` and
`--solution-dir` (a plant probe may show `harbor:missing_reward`). Native side: the same wrong and empty answers
through the fake model or the native control of the family, then `harbor_parity --strict --expect 0` on each pair.

## G4 native smoke and G5 Harbor smoke

The smoke dataset (SKILL.md Step 3) whole on both arms, no `--limit`: the native arm takes rows in file order and the
Harbor arm sorted task ids, so a limit can run different tasks. Same harness, model flags and concurrency, model-call
capture on.

```bash
# G4: see native-baseline.md for the command that fits the native config
# G5:
gym eval run <smoke dataset> --agent opencode_agent --sandbox <provider> --config harbor_arm.yaml \
    --config harbor_arm_harness.yaml <model flags> --concurrency 4 \
    ++observability_enabled=true ++model_call_capture_dir=<abs harbor capture dir>
```

Pass: every task scored on both arms, 0 failed, masked explained, and both capture directories hold one
`*.capture.jsonl` per rollout. Read the rollouts, not the exit code. A rollout or server log that says
`model_calls_unavailable` means the two `++` capture flags were missing on that arm (workplace_assistant never got
captures this way; scale_swe did once they were passed): rerun with them, G6 needs both.

## G6 comparison

```bash
python -m scripts.harbor_conversion_checks first-request <native capture dir> <harbor capture dir> \
    --out first_request/ --json first_request.json
python -m scripts.harbor_parity --native <G4 rollouts.jsonl> --native-key <field|_ng_task_index> \
    [--id-map <map.json>] --harbor results/harbor/<taskset>/ --min-tasks 3 --json g6.json --markdown g6.md
gym dev compare <G4 rollouts.jsonl> <G5 rollouts.jsonl> --key <field>   # only when both arms carry the field
```

`gym dev compare` needs one id field in both files; SWE native rows carry `instance_id` and Harbor rows
`_ng_task_id.task_id`, so read flips from `harbor_parity`'s per-task table there.

`first-request` passes when at least one pair exists, every task is paired, and no pair reports `model differs`,
`tools differs` (the tool names, then each complete tool definition, in any order of tools and of object keys),
`sampling differs` (every top-level request key but `messages`, `tools`, `stream` and
`stream_options`, so `tool_choice` counts) or `roles differ`, and no contamination flag (`AGENTS.md`, `CLAUDE.md`,
`Instructions from`, `<available_skills>`, counted in every message but the assistant's) has a count that differs
between the sides or sits on a system-prompt line that differs. A capture file without an agent request is listed as
`no agent request`. A flag on the native side only is host contamination (fix the isolation in
[native-baseline.md](native-baseline.md#isolation)); a flag on the Harbor side only is something in the task or
image the native arm lacks. OpenCode's own system prompt lists its built-in skill as `<available_skills>` even with
an empty HOME (math 5/5), and its coding prompt names `AGENTS.md` ("suggest writing it to AGENTS.md"; scale_swe 3/3,
not math): a flag on both sides on identical system-prompt lines is noted, not failed.
Confirm it in the pair's `native.system.txt` and `harbor.system.txt` under `--out`. Differences in the system
prompt's working-directory, date or provider-prefix lines are reported only; write them down. A pair matched only
after undoing argv quoting means the Harbor arm needs `argv_prompt_quoting`; one matched only with CRLF read as LF
(`newlines`) is the Step 5 newline difference. The smoke table says nothing about the
mean; it shows both paths run the same thing.
`gym eval compare` pools repeats of one benchmark; `harbor_parity` pairs two task-id spaces and adds strict mode and
rate gaps, which parity needs.

## Full parity

50 tasks x 2 repeats per arm (SKILL.md Step 10), after an oracle `gym dataset validate` of every Harbor task scored
1.0. Then:

```bash
python -m scripts.harbor_parity --native <native rollouts.jsonl> --native-key <field|_ng_task_index> \
    [--id-map <map.json>] --harbor results/harbor/<taskset>/ --json parity.json --markdown parity.md
```

Pass: G2 and G3 strict PASS on both arms, and statistical PASS: every task scored on both arms, the paired 95%
bootstrap interval of Harbor minus native holds 0, and the masked and failed rate gaps are at most 0.05. Fewer paired
tasks than `--min-tasks` (default 10) is INCONCLUSIVE. Exit 0 PASS, 1 FAIL, 2 input error (an unmatched task, a bad id
map or file), 3 INCONCLUSIVE. A FAIL from a rate gap can be infrastructure: rerun before you call it a conversion
defect. If the means differ, diff the first requests again before anything else. A FAIL can rest on a few flips: on
about 50 tasks x 2 repeats, 3 tasks with one repeat flipped in one direction PASS and 4 FAIL, although the sign test
gives p = 0.125. Then diff a differing task's first requests, and add repeats or tasks and rerun, or use a
`--tolerance` stated before the run; never change it after seeing the result (`scripts/harbor_parity/README.md`,
"Choosing `--tolerance`"). A native mean at or near 0 or 1 limits what a PASS shows (SKILL.md Step 3).

## evidence.json template

```json
{
  "env": "<environment and native pairing>",
  "gym_commit": "<sha>",
  "rows": {"file": "<basename>", "sha256": "<hex>", "lines": 0, "source_revision": "<hf revision or url@sha>"},
  "ids": {"smoke": "<ids file>", "parity": "<ids file>"},
  "configs": ["harbor_arm.yaml", "harbor_arm_harness.yaml", "native_arm.yaml"],
  "gates": [
    {"gate": "P1", "command": "<exact command>", "run_dir": "<results dir>", "status": "pass|fail",
     "result": "<rewards, masked, failed, time>", "files": ["<small copies: run summary, a verifier log>"]}
  ]
}
```

One entry per probe and gate, including failed attempts and why they failed.

## Conversion log template

```markdown
# <environment> -> Harbor: conversion log
## 1. What the native environment does   (seed, tools, state, verify, prompt, harness, image; read in code)
## 2. Decisions                          (numbered: what and why)
## 3. Differences from native            (numbered, each with its expected effect on reward)
## 4. Pitfalls hit                       (what happened, how you got past it)
## 5. Requests outside the adapter       (Gym changes the native arm needed)
## 6. Run record                         (label or run dir, what, result)
```

## Differences checklist

Go through every line; each that applies becomes a numbered difference with its expected effect on reward.

- **Where the harness runs**: host CLI vs sandbox, tools on the host vs in a sandbox, extra packages in the image.
- **Model traffic**: the native local CLI makes an extra title-generation call per episode that a Harbor session does
  not (reasoning_gym: 99 of 100 native rollouts). A known difference with no reward effect; list it for cost.
- **Prompt**: folded system prompt or system role, CLI quoting, trailing newlines, dropped `tools` /
  `parallel_tool_calls`, harness system prompt lines (working directory, date and timezone, provider prefix).
- **Sampling**: temperature or top_p the rows carried natively; output-token cap per call (`max_output_tokens`).
- **Tools and permissions**: tool list on the wire, permission profile, skills offered, and each tool's complete
  definition: a description can name a host path (OpenCode's `bash` names `<TMPDIR>/opencode`, the host's
  temporary folder on a native local CLI; reasoning_gym; the remedy is a neutral `TMPDIR`,
  [native-baseline.md](native-baseline.md#isolation)).
- **Transcript scope**: native may grade subagent sessions or skip whitespace-only parts; the Harbor session export
  may not (math: a `task`-tool subagent's answer is graded natively, never on Harbor).
- **Grading input**: the response vs a journal; call order (harness-reported vs arrival order) under parallel calls.
- **Grader**: vendored functions, restated glue, CPU and timeout of the sandbox it now runs in, a grader crash
  (native 500 = failed row vs Harbor missing reward = 0).
- **Retries**: native verification retry on no verdict (`inconclusive_verification_retries`), seed retries, retryable
  vs terminal seed failures.
- **Timeouts and masking**: a native timeout scored 0 vs masked on Harbor; exit 75 cases; budgets
  (`[agent].timeout_sec`, episode timeout, verifier timeout).
- **Resources and scheduling**: `resource_requests` (a harbor `sandbox_provider_options` setting, not a task field),
  sandbox ready timeout (harbor 900 s vs the server's own), sandbox metadata labels, `sandbox_config.env`, the CPU
  hints `cpu_cap_env` adds.
- **Metrics**: per-subset metrics (`subset_for_metrics` exists natively only), pass@k or extra verify keys.
- **Egress and cost**: per-seed installs, image pulls, wall time per episode.
