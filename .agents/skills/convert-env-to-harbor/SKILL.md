---
name: convert-env-to-harbor
description: >
  Guide for converting a native NeMo Gym environment (JSONL rows, a Resources Server with verify(), tools and
  per-session state, an agent, YAML) into Harbor task folders run by the `harbor` Environment Server, and for proving
  parity: the same harness, model and tasks score the same on the native path and the converted path. Use when the
  user asks to convert, port or migrate an environment or benchmark to Harbor or to task folders, write a per-server
  Harbor adapter, turn rows into task folders, or compare native and Harbor rewards. Covers the go/no-go decision,
  infrastructure probes, the native baseline, images, instruction parity, hidden data, reward safety, the adapter,
  gates G1-G6 and the full parity run. Requires an existing native Gym environment; not for importing an upstream
  benchmark as new task folders (use add-benchmark). Triggered by: "convert to Harbor", "task folders from rows",
  "Harbor parity", "Harbor adapter for <server>", "move <environment> off the Resources Server".
---

# Convert a NeMo Gym Environment to Harbor Task Folders

The Gym page [Convert an Environment to Task Folders](../../../fern/versions/latest/pages/data/task-folders/convert-an-environment.mdx)
is the contract. When it and this skill disagree, the page wins. This skill adds the order of work, the decisions, the
gates with their commands, and the traps three pilot conversions hit: `math_with_judge` (text answer), `scale_swe`
(SWE, per-row image) and `workplace_assistant` (tools with state), plus two converted from this skill alone as dry
runs: `reasoning_gym` (text answer, partial credit) and `swe_rebench` (SWE, 7 languages). Their adapters live in
`resources_servers/<server>/harbor/`; read the one of your family before you write code.

Words: the *native arm* runs the rows through the Resources Server; the *Harbor arm* runs the converted task folders
through the `harbor` Environment Server. *Parity* = same harness, same model flags, same tasks on both arms, mean
reward within run-to-run noise. Task-folder terms follow
[Task Folder Format](../../../fern/versions/latest/pages/data/task-folders/task-folder-format.mdx) and
[Task Folder Verifiers](../../../fern/versions/latest/pages/data/task-folders/task-folder-verifiers.mdx). Read
those two pages before writing a task folder; do not rely on memory of upstream Harbor.

Throughout, `<model flags>` means `--model-type <type> --model <model> --model-url <url>` with credentials in
`env.yaml`, `<provider>` is the sandbox provider, and `python -m scripts.<tool>` runs from the repository root.
Keep a `CONVERSION_LOG.md` and an `evidence.json` outside the repository from the first step ([gates.md](references/gates.md)
has both templates); their content goes into the PR body.

## Scope

Proven end to end (G1-G6 and reward-safety probes, OpenCode on both arms):

| Family | Pilot | What it proved |
| --- | --- | --- |
| Text answer graded from the response, judge off | `math_with_judge` via `environments/opencode_math` | Python verifier reads `response`; grader copied by `ast`; overlay install |
| SWE: per-row image, patch, hidden tests | `scale_swe`; `swe_rebench` (dry run) | Overlay seed, collect hook, separate verifier, vendored grader; oracle and controls on Python, JS, TS, Java, Go images, oracle only on Rust and Dart |
| Resources Server tools with per-session state | `workplace_assistant` | Compose sidecar MCP server with a call journal, separate verifier; G6 without a first-request comparison (no captures were taken) |

Allowed only with the extra gate named: judge on (probe P6); a delegated or per-row verifier (Step 1, resolved
config); a tool replaced by the harness shell (Step 1 says what proves it); task images without
Python 3.10+ (P3 decides the verifier design); a tool that calls an external stateless service such as a search API
(P4; record a live-service difference); no same-harness native pairing on main (Step 0).

Stop and stay on the Resources Server path for: simulated users, cohort or group rewards, a judge the verifier cannot
reach, single-turn environments no harness can run as an agent episode (DAPO math ran with `simple_agent` before
its OpenCode pairing), and tool state held in an external system or too large to rebuild per
episode (a generic code executor whose only state is its own interpreter session is not external state, even when
that session lives in a separate sandbox service: Step 1 replaces it).

## Step 0 Decide

1. **Pick the path.** Harbor path: the whole benchmark fits in task folders whose verifier writes the reward; Gym
   runs them with no Resources Server and no benchmark code of its own. Resources Server path: grading needs Python
   state on the host, tools Gym serves, a judge model, or the server's own sandbox policy. The rest of this skill
   assumes the Harbor path; see [Task folders or a Resources Server](../../../fern/versions/latest/pages/agent-interactions/harbor.mdx#task-folders-or-a-resources-server).
2. **Go/no-go on the native arm.** Parity needs a native pairing with the same harness (the pilots used
   `opencode_agent`; math's pairing is the `legacy_agent` relay to a local CLI, which counts). When the server ships
   several configs, take the one whose agent type is the one you run on Harbor (`opencode_agent`, not
   `opencode_sandboxed_agent` or `simple_agent`; swe_rebench ships all three). Go when such a pairing exists on main
   and scores a known row: the reference answer, or the fake-model oracle
   ([native-baseline.md](references/native-baseline.md)). If none exists and the server is stateless or already
   implements the session hooks (`has_session_state`, `initialize_session_state`, `release_session_state`), add only
   the pairing config, in its own PR (`environments/opencode_math/config.yaml`, a `legacy_agent` relay to
   `opencode_agent`'s `legacy_app.py`, is the template for a text answer). If it keeps per-session state without
   them, either move the Resources Server to the Environment Server session contract first, in its own PR (it can
   take more than two days; follow
   [Resources Server APIs](../../../fern/versions/latest/pages/environment-tutorials/resources-server-apis.mdx#state-management)
   and the state hooks in [add-benchmark Step 3](../add-benchmark/SKILL.md#step-3-implement-verify)), or run the
   native arm on a demonstrator of the missing fix and label every number with it (`workplace_assistant` did:
   "native + typed-seed + name-normalization demonstrator"), or stop.
3. **Ask the owner for what you cannot invent:** the model endpoint and its credentials, the sandbox provider (a
   Compose sidecar needs OpenSandbox with networking on main), whether an image registry the provider can pull from
   exists (none: overlay Dockerfiles and Compose sidecars only), and approval for downloads (datasets, images).

Before moving on:
- [ ] The path, the native pairing and its go/no-go result are in the log.
- [ ] Endpoint, provider, registry answer and download approval are recorded.

## Step 0b Infrastructure probes

Run these before any adapter code; each is one `evidence.json` entry. Commands and pass criteria are in
[gates.md](references/gates.md#infrastructure-probes-p1-p6).

| Probe | What it proves | Needed when |
| --- | --- | --- |
| P1 | `gym dataset validate environments/hello_world --sandbox <provider>` scores 1.0 | always |
| P2 | each image family (or language) pulls: `gym dataset init probe --root <dir> --image <image>`, then validate | per-row or custom images |
| P3 | the image has what grader and hooks need (`python3` >= 3.10, `timeout`, `bash`, `git`), checked in the probe's `solve.sh` | a verifier or hook runs in the task image |
| P4 | egress from the agent sandbox, and from the verifier sandbox under separate verification, to what installs, graders or tools fetch | any `RUN` install, sidecar install or online tool |
| P5 | a two-service Compose task with `networking.enabled` scores 1.0 | tools move into a sidecar |
| P6 | the judge is reachable from the verifier sandbox, with credentials delivered to it only | judge on |

A probe that fails changes the design (Step 4), not the probe. P3 failing means "no Python in the image": install a
sha256-pinned standalone Python at seed and in `test.sh` (swe_rebench's `python.sh`; needs P4 egress to GitHub
releases), or use a verifier image (needs a registry). With one image per row, probe every image family in one
combined dataset ([gates.md](references/gates.md#infrastructure-probes-p1-p6)). When run capacity is scarce, P1 may
cite a passing run on the same provider, and for a family whose only egress is a package-index install
in an overlay, G2 with separate verification is P4's evidence (it installs in both sandboxes); a failure then costs an
adapter redesign instead of a probe.

Before moving on:
- [ ] Every probe your family needs has a passing `evidence.json` entry, or the design change it forced is logged.

## Step 1 Classify and read the effective config

1. Resolve the exact native run config and read values there, never from the shipped YAML alone:
   ```bash
   OPENSANDBOX_API_KEY=unused gym env resolve --config <native run config> [--config <its overlays>] \
       [--config responses_api_models/openai_model/configs/openai_model.yaml] \
       [--config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml] \
       ++policy_base_url=http://unused ++policy_api_key=unused ++policy_model_name=<parity model> \
       > native_resolved.yaml
   ```
   The placeholders let it resolve without an `env.yaml` (else: "`policy_base_url` ... not set") and keep
   credentials out of the dump. Add the model config when the agent names `policy_model` but the config defines no
   model server ("references responses_api_models/'policy_model', which is not defined"), and the provider config
   when the server starts sandboxes (its key comes from the environment variable).

   Then list every field of the server's config class (in its `app.py`) and of the agent's config, with its resolved
   value, and decide for each whether the task can reproduce it (the adapter refuses the rest, Step 8). Look first for
   the delegated verifier (the server whose `verify()` really runs), defaults that apply when a key is absent (math's
   `should_use_judge` defaults to true), the agent `entrypoint`, `system_prompt` and `system_prompt_mode`, `timeout`,
   `thinking`, `token_id_capture`. An empty config class (reasoning_gym) is an answer too.
2. Answer each question from the code, and write the answer in the log:
   - **Where does the harness run?** An agent entrypoint like `legacy_app.py` with no agent session runs a local CLI
     on the agent-server host: no sandbox and no image (math). Do not assume "the native image".
   - **Seed:** image, seed commands, their time limits and tolerance, retries.
   - **Tools and session state:** which tools, which state, whether it is deterministic (no wall clock, no
     randomness, no network).
   - **What `verify()` reads:** the final message, all assistant messages joined (math), the tool calls
     (workplace_assistant), a captured patch plus hidden tests (scale_swe).
   - **Reward range:** binary or partial credit (reasoning_gym's knights_knaves scores 0.3 + 0.7 x the matching
     fraction). With partial credit the oracle must still score exactly 1, and a wrong-answer control must be one the
     scorer gives 0 (read the scorer: a plausible wrong answer scores above 0). To show both arms give the same
     partial reward per task, run the [partial-credit probe](../../../scripts/harbor_conversion_checks/README.md#a-partial-credit-probe-a-parity-check-not-a-control).
   - **Judge, per-row verifier, metrics** beyond `reward`.
3. Choose the family in [families.md](references/families.md). For tools: re-host them in a Compose sidecar when
   state matters or grading reads the calls (in-repo Python tools); replace them with harness built-ins only for a
   generic executor such as code execution, and put the packages the rows import into the image. What proves such
   a conversion: both arms run without the tool, the same gates and the same parity rule apply between those two
   arms, and the result is reported as a new baseline, because neither arm is comparable with the environment run
   with its tool. No pilot exercised this path.

Stop and ask the owner when verify() depends on something outside the episode (another rollout, a live service whose
answers change, a judge you cannot reach from a sandbox).

Before moving on:
- [ ] `native_resolved.yaml` is saved, and the log answers every question above from code.

## Step 2 Native baseline

Follow [native-baseline.md](references/native-baseline.md). Run 3-5 rows on the native arm with the parity harness
before writing the adapter, and diff the two arms' `rollouts_resolved_config.yaml` once both exist. Most native
surprises in the pilots were here: a local CLI reading the host's `AGENTS.md` and home directory into the system
prompt (math), a policy model on `127.0.0.1` that a remote sandbox cannot reach (scale_swe hung silently until its
episode timeout), and a native pairing that failed at seed or scored 0 on tool-name spelling (workplace_assistant).

Before moving on:
- [ ] The native smoke ran with the same harness and model flags you will use on the Harbor arm.
- [ ] The native first model request was captured and contains no host instructions or personal skills.

## Step 3 Task ids

- Give every task a stable id from a field of the row. When rows have no id (math), use the 0-based line index, and
  record the rows file's sha256 and line count in the dataset README: another source revision gives the same id to
  another problem.
- Grade every reference answer with the server's grader on the host before choosing ids (math: 17,398 rows in 5.5
  min): a loop that passes each row's reference, in the answer format the native prompt asks for, to the grader.
  Run it in the server's venv, which `gym env test --resources-server <server>` builds (in
  `resources_servers/<server>/.venv`, or under `uv_venv_dir` when set). Once the adapter exists you can instead
  convert all rows and call the task's `tests/verifier.py` on each `tasks.jsonl` row (reasoning_gym). For SWE, run
  the server's golden-patch mode on the candidate rows instead
  ([families.md](references/families.md#swe-per-row-image-patch-capture-hidden-tests-scale_swe-swe_rebench)): it found
  a swe_rebench row whose tests download data the sandbox cannot reach. Exclude references the grader cannot award,
  and rows with no reference. For SWE, also run the candidates with an empty patch (`controls --empty` on the
  converted candidates, then `gym dataset validate`) and exclude a row the empty patch resolves: in scale_swe one
  parity row's FAIL_TO_PASS test passes on the base commit. Re-run a reference that fails once on a flaky test
  before you exclude its row.
- Pick the parity subset reproducibly; the smoke subset is its first 5 ids:
  ```bash
  python -m scripts.harbor_conversion_checks subset <rows.jsonl> --id-field <field|_index> --count 50 --seed <S> \
      [--include <supported_ids.txt>] [--exclude <unawardable_ids.txt>] --out parity_ids.txt
  grep -v '^#' parity_ids.txt | head -5 > smoke_ids.txt
  ```
  For a file `subset` wrote, the same command with `--count 5` gives the same 5 ids; for an ids file drawn another
  way (the math, scale_swe and workplace_assistant pilots) it does not.
  Filters that need the network (image exists) stay in your own script. When the rows are public, ship the parity
  ids as `resources_servers/<server>/harbor/configs/parity_ids.txt`, as the pilot adapters do.
- Avoid the ceiling. A subset whose native mean is at or near 0 or 1 can only show a difference in one direction:
  with the native arm at 1, a PASS cannot show a Harbor advantage, and it still passes with a few tasks lost on
  Harbor. Pick tasks, or a task mix, the model solves only some of the time: check the G4 smoke mean, or run one
  native repeat on the candidate ids. reasoning_gym's 2-person puzzles scored 100/100 natively and 98/100 on Harbor:
  PASS, but it shows only that Harbor loses no more than the interval's lower bound (-0.05). In math, 47 of the 50
  tasks scored 1 on every rollout of both arms.
- Convert the smoke and parity subsets as separate datasets (the Step 8 adapter, once per ids file), each with its
  own native rows and id map, and run every gate on a whole dataset. Never select tasks with `--limit`: the native arm
  takes rows in file order and the Harbor arm task ids in sorted order, so `--limit 3` ran different tasks on the two
  arms (reasoning_gym) and G6 could not pair them.
- Keep native rows as raw lines: re-serialising JSON changes bytes (`ensure_ascii`).

Before moving on:
- [ ] The parity and smoke ids files, their seed and the rows file's sha256 are in the log.

## Step 4 Image

Prefer, in order: a digest-pinned `[environment].docker_image`; an overlay `environment/Dockerfile` (`FROM` +
`RUN`, run at every seed, nothing built or pushed); a build-mode Dockerfile (needs a registry the provider pulls
from); a Compose sidecar for tools. See [Dockerfile modes](../../../fern/versions/latest/pages/data/task-folders/task-folder-format.mdx#dockerfile-modes).

- Pin `FROM` by digest, read from the registry before any run ([families.md](references/families.md#pinning)), or
  from the `compose-images.json` a validate records (in the dataset folder, or in `rendered/` for a template).
- Pin installed packages to the Resources Server's `uv.lock`, transitive ones included, after checking the lock for
  packages with no wheel; the commands are in [families.md](references/families.md#pinning). A package the server
  venv carries indirectly can change behaviour (pyarrow changed pandas string storage in workplace_assistant).
- Each overlay `RUN` line runs as one `sh -c` argument: keep it under 128 KiB. Overlay mode has no `COPY`: ship files
  (code, a hashed `requirements.txt`) as deterministic base64 of a gzip or tar.gz (mtime 0) in one `RUN` line and use
  them in a second.
- An overlay `RUN` install needs egress at every seed (P4). Measure its cost with a copy of the dataset whose
  Dockerfile has no `RUN`: math went from about 32 s to 37-41 s per oracle episode, and the copy scoring
  `harbor:missing_reward` proved the install was needed.

Before moving on:
- [ ] Every image is pinned by digest and every installed package by version and hash.

## Step 5 Instruction parity

A Harbor task gives the agent one user message, `instruction.md`. Compare it with what the model receives natively,
not with what Gym code builds.

- **Fold** what the native path puts in the prompt: a system prompt the local CLI prepends (math, read from the
  config: a YAML `|` block keeps its trailing newline), or a `system` role message (workplace_assistant: system text,
  blank line, user text). Do not move it to the harness's `system_prompt`: in sandbox sessions that becomes a
  system-side instructions file, which is not what the local path does.
- **Drop** and record: row `tools` and `parallel_tool_calls` (OpenCode's sandbox session refuses them with a 422),
  and sampling parameters. Harbor rows cannot carry sampling; set the same default on the model server with
  `++policy_model.responses_api_models.openai_model.extra_body.temperature=<value>` on both arms.
- **Rewrite** only through harness settings: OpenCode's `run` quotes a prompt passed as one argv word, so the Harbor
  arm sets `argv_prompt_quoting: true` while `instruction.md` stays unquoted.
- **Newlines:** Gym reads `instruction.md` in text mode, so a native prompt with CRLF (GitHub issue bodies in SWE
  rows) reaches the Harbor model with LF. It cannot be reproduced: record it, and pass `--normalize-newlines` to
  `check`.
- Check the rule the adapter applies on every task (the id map comes from the adapter, Step 8). The prefix file is
  the folded text byte for byte; write it with Python, since shell `$(...)` drops the YAML block's trailing newline:
  ```bash
  python -c 'import sys, functools, yaml; v = functools.reduce(lambda d, k: d[k], sys.argv[2].split("."),
      yaml.safe_load(open(sys.argv[1]))); sys.stdout.write(v + "\n\n")' native_resolved.yaml \
      <agent instance>.responses_api_agents.<agent type>.system_prompt > prefix.txt
  python -m scripts.harbor_conversion_checks check <rendered dataset> --native-rows <native rows> \
      --native-id <field|_index> [--id-map <map.json>] [--prefix-file prefix.txt]
  ```
  `check instruction` only restates the adapter's rule. The stronger evidence is `first-request` in G6. A worked
  example is G1b under [Gates in the math adapter README](../../../resources_servers/math_with_judge/harbor/README.md#gates).

Before moving on:
- [ ] Every folded, dropped or rewritten part of the prompt is a numbered difference.

## Step 6 Hidden data

Reference answers, hidden tests and ground truth live only in `task_data` (`tasks.jsonl` or `task_data.json`),
`tests/` and `solution/` (oracle only). No verifier code runs on the host: at verification, after the agent's turn,
the server uploads the task's fields, the agent's response and `tests/` into the verifier's sandbox
([Task Folder Verifiers](../../../fern/versions/latest/pages/data/task-folders/task-folder-verifiers.mdx)). In shared
mode that sandbox is the agent's own, which is one reason separate verification is the default (Step 7). Never put
hidden data in `instruction.md`, `environment/`, the image, skills or the agent's working directory.

```bash
python -m scripts.harbor_conversion_checks check <rendered dataset> --native-rows <native rows> \
    --native-id <field|_index> [--id-map <map.json>] --hidden <field> \
    --hidden-from resources_servers/<server>/task_data.py --template <template dir> --template-hidden <name> \
    --json check.json
```

Short hidden values (under 8 characters, such as a numeric answer) are counted, not scanned: name them with
`--template-hidden` so the check fails on any template file outside `solution/` and `tests/` that references them.
A longer generic value can match shipped code by coincidence (a test named `continue` in a vendored module): read
each match, and list the ones that are not row data in a file for `--allow-values` (it covers files under
`environment/` only; a match in `instruction.md` or the agent-visible `[environment]` settings always fails). A
`--hidden` field that no native row has exits 2 (a typo or a wrong dotted path).

Before moving on:
- [ ] `check` reports 0 failed on the folder G1b names (Step 9).

## Step 7 Verifier and reward safety

Read [reward-safety.md](references/reward-safety.md) before writing the verifier. The rules:

- Run the environment's own grading code, vendored unchanged; never rewrite it. A grader that is a server method
  reading Responses objects by attribute is vendored as a stand-in class
  ([families.md](references/families.md#vendoring-a-grader)).
- Verify in a **separate** sandbox by default. In shared mode an agent with root replaced `python3` and scored 1.0
  without answering (math, 5/5). An overlay task gets separate verification from the server option
  `separate_verifier_for_artifacts: true` plus an `artifacts` entry in `task.toml`.
- Exit 75 without a reward (a mask) only where the native server masks, and only on facts the agent cannot write.
  Anything a collect hook writes in the agent's sandbox is agent-controlled: map it to a completed 0, never a mask.
- When the verifier executes agent-written code (tests, a `conftest.py`, and the install step before them: `npm ci`,
  `pip install -e .`, `mvn compile`), a separate sandbox is not a trust boundary: protect the reward path and every
  file the grader has mapped by ownership, drop capabilities around the whole eval script, and fail closed on any
  import or file open after the tests start, as `scale_swe` does (reward-safety.md rule 2). Some routes stay open
  until the harbor server reads the reward outside the verifier sandbox; list them in the adapter README.
- Prove it with controls that must score 0 unmasked ([gates.md](references/gates.md#g3-negative-controls)):
  `controls --plant-reward`, plus `--plant-conftest` (pytest) or `--plant-hooks` (npm, Go, pytest) when the verifier
  runs agent output; for other languages write your own `--solution-dir` probe.

Before moving on:
- [ ] Every exit 75 in `test.sh` maps to a case where the native server masks, on a fact the agent cannot write.
- [ ] The run config that makes verification separate is named in the adapter README.

## Step 8 Adapter

Write `resources_servers/<server>/harbor/`, where `<server>` is the server the native pairing names, even when its
`verify()` delegates. Vendor the grader that really runs there, unchanged (a delegated server's too), and record what
the delegating server adds (passed-through `mask_sample` or `failure_kind`, extra metrics) as numbered differences;
never extend another pairing's adapter. The full contract (files, CLI flags and exit codes,
the id map, deterministic bytes, the dataset README, vendoring headers, run configs, tests) is in
[families.md](references/families.md#adapter-contract). In short:

- `adapter.py` streams `--rows`, writes `--output-dir`, selects `--ids-file`, and writes `--native-rows-out` and
  `--id-map-out`; it refuses with exit 2 every native setting the task cannot reproduce.
- Two conversions into two parent folders with the same file names must `diff -r` empty (G1a), so the dataset
  README names inputs by basename and the output as `<dir>`; the adapter runs outside a git checkout.
- Run configs: `harbor_arm.yaml` (server and deployment settings, on every Harbor validate and run),
  `harbor_arm_harness.yaml` (harness settings, `gym eval run` only) and `native_arm.yaml`.

Before moving on:
- [ ] Each adapter test fails when you revert the behaviour it covers.
- [ ] `ruff check` and `ruff format --check` pass on the adapter files only.

## Step 9 Gates G1-G6

Run them in this order; [gates.md](references/gates.md) has each command, pass criterion and evidence entry. Never
claim a gate you did not run: paste the command, the run folder and the result.

| Gate | Pass when |
| --- | --- |
| G1a determinism | two conversions into two parent folders with the same file names: `diff -r` empty, before any validate |
| G2 oracle | Harbor `gym dataset validate` with `harbor_arm.yaml` scores 1.0 on the smoke dataset, and the native oracle scores 1.0 on its native rows; `harbor_parity --strict --expect 1` PASS |
| G1b content check | `check` passes: for a parameterized template on `results/harbor/<dataset folder name>/rendered` after G2 (it exits 2 on the template); for one folder per task on the second G1a conversion, which no validate touches |
| G3 negative | controls from a never-validated conversion, under their own dataset folder name, into a folder that must not exist: empty or wrong solution and the plant probes score 0 unmasked on both arms; `--strict --expect 0` PASS |
| G4 native smoke | the smoke dataset's native rows, parity harness and model flags, model-call capture on |
| G5 Harbor smoke | the smoke dataset, same harness and flags, `harbor_arm.yaml` and `harbor_arm_harness.yaml`, capture on |
| G6 comparison | `first-request`: at least one pair, every task paired, no model, tool (names and complete definitions), sampling, role or non-system message difference, no contamination flag whose count or system-prompt line differs between the sides; `harbor_parity --min-tasks 3` per-task table; masked and failed counts |

`gym dataset validate` exits 1 on a negative control by design, and labels a 0 "oracle solved": read the rewards and
`failure_kind` from `rollouts_run_summary.json`. A smoke where every task scores 1 on both arms says nothing about
the mean. A rollout that logs `model_calls_unavailable` ran without
`++observability_enabled=true ++model_call_capture_dir=<abs dir>` on the command that started the servers (for a held
native stack, `gym env start`, not only the client): G6 is not done until both arms have captures.

Before moving on:
- [ ] Every gate has an `evidence.json` entry with its command, run folder and result, failed attempts included.

## Step 10 Full parity and handoff

1. **Estimate first.** Median episode time in the full runs, Harbor vs native: math 93 s vs 13 s, workplace_assistant
   141 s vs 57 s, reasoning_gym 99 s vs 10.6 s (about 9x: its overlay builds three packages from source in both
   sandboxes at every seed). SWE episodes were not slower: scale_swe 231 s vs 217 s; swe_rebench's 3-task smoke run
   took 406 s on Harbor, 431 s native. Sandboxes per Harbor episode = agent + separate verifier + sidecars. Total time
   is about tasks x repeats x 2 arms x episode time / concurrency. Tell the owner the estimate.
2. **Run** the parity dataset (50 tasks, whole) x 2 repeats per arm, same harness, model flags and concurrency,
   model-call capture on. Before the model run, validate the whole Harbor dataset with the oracle: every task must
   score 1.0. A local CLI native arm needs a fresh empty HOME for every run
   ([native-baseline.md](references/native-baseline.md#isolation)).
3. **Judge** with `python -m scripts.harbor_parity` ([gates.md](references/gates.md#full-parity)): exit 0 PASS,
   1 FAIL, 2 input error (an unmatched task, a bad id map), 3 INCONCLUSIVE. If the means differ, diff the first
   requests again before anything else. If the native mean is at or near 0 or 1, say in the report what the PASS
   cannot show (Step 3).
4. **Hand off:** the dataset README's regeneration command, the run configs, the numbered differences list and the
   parity report. Post-training must be able to regenerate and run the dataset from the adapter README alone. A report
   prints its input paths as passed: for a report you paste into a pull request, run `harbor_parity` from the run
   folder with relative paths.

Before moving on:
- [ ] `harbor_parity` printed PASS on the full run, or the log says why not and what was rerun.

## Constraints

- Never change the native environment's data, prompts or verifier to make the arms agree. Changes outside the
  adapter (Gym bugs found on the native arm) go in their own PR or issue; label a native arm that needs them.
- Vendor graders unchanged; restate only dispatch glue, and record it as a difference.
- Every difference from native (grading, prompt, tools, sampling, timeouts, masking, retries, resources) is a
  numbered entry with its expected effect on reward ([gates.md](references/gates.md#differences-checklist)).
- No secrets, internal hostnames or local paths in task folders or adapter output (`check` scans for them).
- New source files carry the NVIDIA SPDX header; line length 119; code outside `nemo_gym/` imports public names.
- Pilot logs (`CONVERSION_LOG.md`) stay out of the repository; their content goes into the PR body.

## References

- [families.md](references/families.md): the adapter contract, the three recipes, vendoring, pinning, known
  deviations of the pilot adapters.
- [native-baseline.md](references/native-baseline.md): running the native arm cleanly; known gaps on main.
- [reward-safety.md](references/reward-safety.md): how the reward can be forged or masked, and the fixes that held.
- [gates.md](references/gates.md): P1-P6, G1-G6 and full-parity commands, evidence and log templates, the
  differences checklist.
- Scripts: `scripts/harbor_conversion_checks/README.md` (`check`, `controls`, `fake-model`, `first-request`,
  `subset`) and `scripts/harbor_parity/README.md`.
- Docs: [Author Task Folders](../../../fern/versions/latest/pages/data/task-folders/harbor-tasks.mdx),
  [Task Folder Patterns](../../../fern/versions/latest/pages/data/task-folders/task-folder-patterns.mdx),
  [Multi-Container Tasks](../../../fern/versions/latest/pages/data/task-folders/compose-tasks.mdx),
  [Harbor Environment Server](../../../fern/versions/latest/pages/agent-interactions/harbor.mdx).
- Related skills: [add-harness](../add-harness/SKILL.md) (a harness for the pairing), [add-benchmark](../add-benchmark/SKILL.md),
  [nemo-gym-score-triage](../nemo-gym-score-triage/SKILL.md) (when the arms disagree).
