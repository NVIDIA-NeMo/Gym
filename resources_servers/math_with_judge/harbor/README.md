# math_with_judge as Harbor task folders (the opencode_math pairing)

`adapter.py` converts one native pairing, `environments/opencode_math` (DAPO-Math-17k rows graded by
`math_with_judge` with the LLM judge off, answered by OpenCode running as a local CLI), into a parameterized Harbor
dataset: `task-template/` plus one `tasks.jsonl` row per selected native row. The `harbor` Environment Server runs
it with no Resources Server. Other uses of `math_with_judge` (judge on, other prompts or agents) need their own
adapter.

Run every command below from the repository root, in the Python environment Gym is installed in (the adapter needs
PyYAML and `nemo_gym`, to check each row against the template's `TaskData`; that environment has both). `<data>` is
a folder outside the repository. The tools are
described in [`harbor_conversion_checks`](../../../scripts/harbor_conversion_checks/README.md) and
[`harbor_parity`](../../../scripts/harbor_parity/README.md); the task-folder format is in
[the task-folder docs](../../../fern/versions/latest/pages/data/task-folders/).

## What the conversion does

| Native | Harbor task |
| --- | --- |
| Row: one user message, `question`, `expected_answer` | `tasks.jsonl` row `{"task_id": "dapo17k-<n>", "task_data": {"question", "expected_answer"}}` |
| The local CLI receives `f"{system_prompt}\n\n{user message}"` as one argv word | `instruction.md` = the agent config's `system_prompt` (its YAML block keeps a trailing newline), two newlines, the question. The CLI's argv quoting is reproduced by the harness setting `argv_prompt_quoting`, not in the file |
| No image: OpenCode and its bash tool run on the agent-server host | Overlay `environment/Dockerfile`: a public uv/Python 3.13 image pinned by digest, plus one `RUN` that installs math-verify and its dependencies at the versions `../uv.lock` pins |
| `verify()` joins the `output_text` of every assistant message, then a last-`\boxed{}` gate and math-verify in a forked child with a timeout | `tests/verifier.py` repeats only that glue; `tests/math_with_judge_grader.py` is generated: the grading functions copied out of `../app.py` unchanged, with the native config's `library_verifier_timeout_seconds` |
| Reference answer in the row | `solution/answer.md` = `\boxed{<expected_answer>}` (oracle only) |
| Grading on the Resources Server host | Grading in a fresh sandbox of the task image (`artifacts` in `task.toml` + `configs/harbor_arm.yaml`) |

Hidden data: the expected answer is only in `tasks.jsonl` and the `solution/` template. The instruction, the image
and the agent's working directory never hold it. The verifier reads it from `task_data` after the agent's turn.

The adapter reads `environments/opencode_math/config.yaml` (or `--native-config`, a `gym env resolve` dump of the
exact native run or the pairing YAML) and refuses, with exit 2, what a task cannot reproduce: the LLM judge
(`should_use_judge`, which defaults to true), `format_tolerant_answer_extraction`, an agent entrypoint other than
`legacy_app.py`, `system_prompt_mode` other than `append`, a `library_verifier_timeout_seconds` that is not a
positive number, and an agent `timeout`, `thinking` or `opencode_version` that differs from the Harbor arm's
(`[agent].timeout_sec` of `task-template/task.toml`, 900; `thinking: false` and `opencode_version: 1.17.11` in
`configs/harbor_arm_harness.yaml`) or is not set. To run another value, change it on both sides first. It also
exits 2 for malformed YAML, a non-empty `--output-dir`, an ids file with no ids, a duplicate or unknown id, and a
row that is not a JSON object holding one plain-text user message and a string `expected_answer`, or whose
`responses_create_params` holds anything besides `input` (a per-row `metadata.agent_timeout_sec`, sampling). A row
whose task fields the template's `TaskData` (`task-template/task_data.py`) refuses also exits 2, checked the way Gym
checks it at prepare.

The adapter does not check the other harness settings of `configs/harbor_arm_harness.yaml`: `sandbox_model_id`,
`argv_prompt_quoting`, `max_output_tokens` and `opencode_config.permission`, nor the session defaults the file
leaves alone (`extra_args`, `context_window`). Each must give the Harbor session what the native agent's run
uses. To check, diff the `rollouts_resolved_config.yaml` that `gym eval run` writes beside each arm's rollouts, then
run `python -m scripts.harbor_conversion_checks first-request <native capture dir> <harbor capture dir>` (G6): the
local CLI does not apply `max_output_tokens` (its resolved config shows the unused default, 131072), so only the
captured requests show the cap it sends. The pilot used `thinking: false`, `opencode_version: 1.17.11`,
`sandbox_model_id: ${policy_model_name}`, `argv_prompt_quoting: true`, `max_output_tokens: 32000` (OpenCode's
default cap, which the local CLI keeps), `permission` with `bash`, `edit`, `webfetch` and `skill` all `allow`, and
the session defaults `extra_args: []` and `context_window: 262144`, which the native config does not set either.

## Regenerate

```bash
python environments/opencode_math/prepare.py --output <data>/dapo17k_train.jsonl
python -m scripts.harbor_conversion_checks subset <data>/dapo17k_train.jsonl --id-field _index \
  --count 50 --seed <seed> --out <data>/parity_ids.txt   # only for a new subset; see the parity ids below
python resources_servers/math_with_judge/harbor/adapter.py --rows <data>/dapo17k_train.jsonl \
  --output-dir <data>/dapo17k_harbor --ids-file <data>/parity_ids.txt \
  --native-rows-out <data>/dapo17k_native.jsonl --id-map-out <data>/dapo17k_id_map.json
```

- Task id `dapo17k-<n>` is row `n` (0-based) of the rows file, which has no id field. `--ids-file` takes those
  indexes or the task ids, one per line, `#` comments allowed, order kept.
- `--native-rows-out` writes the same rows, in the same order, each line unchanged apart from its line ending
  (written as LF), for the native arm. `--id-map-out` maps each of its lines (what native rollouts carry as
  `_ng_task_index`) to the task id.
- The dataset's root `README.md` (render skips it) records the provenance: the rows file's name, sha256 and line
  count, the selected count, the ids file's sha256, the native config's sha256, the Gym commit (`unknown` outside
  a git checkout; `-dirty` appended when `app.py` or `harbor/` has uncommitted changes), the sha256 of the
  generated grader and of the `app.py` it came from, and the regeneration command with the basenames of the input
  and side-output files. `prepare.py` does not pin the Hugging Face revision, so check the rows sha256: another
  revision can give the same id to another problem. The parity rows are `prepare.py`'s output from
  `YouJiacheng/DAPO-Math-17k-dedup` at revision `6e26a33abdabd3e6aaa1d742326b790758f7dbc5`, split `train`: 17398
  lines, sha256 `e78d1d5422c80bb7305ed1e780579702830233496a1c5885499eda6e16c22a5f`.
- The 50 parity ids (`configs/parity_ids.txt`) are `random.Random(20261122).sample(sorted row indexes, 50)` in
  draw order, taken from the rows whose reference answer the grader scores 1 (all 17398 qualify); the first 5 are
  the smoke subset (command under Gates). `subset` orders ids differently, so it does not reproduce this list; use
  it only for new subsets.
- A dataset regenerated from this revision differs from the one the parity run used only in comments (the
  `environment/Dockerfile` comment, the grader's vendoring header) and the dataset README, but `environment/` is
  hashed into the environment id, so it needs one `gym dataset validate` before a static load (see Known limits).

## Run both arms

The dataset is tamper-safe only when run with `configs/harbor_arm.yaml`: it makes the harbor server verify every
task in a fresh sandbox. `configs/harbor_arm_harness.yaml` makes the sandbox session send what the native local
CLI sends; it goes on `gym eval run` only.

```bash
H=resources_servers/math_with_judge/harbor/configs
gym eval run <data>/dapo17k_harbor --agent opencode_agent --sandbox opensandbox \
  --config $H/harbor_arm.yaml --config $H/harbor_arm_harness.yaml \
  --model-type openai_model --model <model> --model-url <url> --num-repeats 2 --concurrency 4
```

Native arm on the same rows, with `configs/native_arm.yaml`. It routes the CLI through `policy_model` (the same
model flags pick the model on both arms) and isolates it from the machine. Supply its two values: a workspace
outside any git repository and a fresh empty HOME for every run (create both first). Put the OpenCode release the
sandbox installs (1.17.11) first on `PATH`; otherwise the local path installs its own copy into the repository.

```bash
N="--config environments/opencode_math/config.yaml --config $H/native_arm.yaml"
ISO="++opencode_math_agent.responses_api_agents.opencode_agent.workspace_root=<workspace dir>"
ISO="$ISO ++opencode_math_agent.responses_api_agents.opencode_agent.env.HOME=<fresh empty dir>"
gym env start $N $ISO --model-type openai_model --model <model> --model-url <url>
# in a second shell, once the stack is ready; stop the stack afterwards
gym eval run --no-serve $N $ISO --agent opencode_math_agent --input <data>/dapo17k_native.jsonl \
  --output <run dir>/rollouts.jsonl --model-type openai_model --model <model> --model-url <url> \
  --num-repeats 2 --concurrency 4
```

Compare with `python -m scripts.harbor_parity --native <run dir>/rollouts.jsonl --native-key _ng_task_index
--id-map <data>/dapo17k_id_map.json --harbor results/harbor/dapo17k_harbor/`.

## Gates

Run them in this order. Each gives its command and what a pass looks like.

**Smoke subset.** Convert the first 5 parity ids as a dataset of their own:

```bash
H=resources_servers/math_with_judge/harbor/configs
grep -v '^#' $H/parity_ids.txt | head -5 > <data>/smoke_ids.txt
```

Select tasks with `--ids-file` only; the adapter has no `--limit`. Do not pick a subset with `gym`'s `--limit`
either: it takes native rows in file order and task ids in sorted order, and the ids are in draw order, so the two
arms would run different tasks.

**G1a, determinism (first).** Convert twice, into two parent folders with the same file names:

```bash
for d in a b; do
  python resources_servers/math_with_judge/harbor/adapter.py --rows <data>/dapo17k_train.jsonl \
    --output-dir <data>/$d/dapo17k_harbor --ids-file <data>/smoke_ids.txt \
    --native-rows-out <data>/$d/dapo17k_native.jsonl --id-map-out <data>/$d/dapo17k_id_map.json
done
diff -r <data>/a <data>/b
```

Expected: each conversion prints `Wrote 5 tasks to ...` and `diff -r` prints nothing. The dataset `README.md` records
the basenames of the input and side-output files, so differently named side files make `diff -r` differ there.

**G2, oracle.**

```bash
gym dataset validate <data>/a/dapo17k_harbor --sandbox opensandbox --config $H/harbor_arm.yaml
```

Expected: exit 0. Read `results/harbor/validation/dapo17k_harbor/rollouts_run_summary.json` under the directory
`gym` ran in: `counts.masked` 0, `counts.failed` 0, and every entry of `tasks` has `reward` 1.0 and
`verifier.mode` `separate`. 5 tasks took about 90 s (each verification about 40 s). The native oracle (a fake model
that answers `\boxed{<expected_answer>}`, `scripts.harbor_conversion_checks fake-model`) scores 1.0.

**G1b, static check (after G2).** The dataset is a parameterized template: `check` refuses it with exit 2 and needs
the rendered task folders. Only a validate or eval run writes them, at `results/harbor/<dataset folder
name>/rendered` under the directory `gym` ran in. The prefix file is the agent's `system_prompt` followed by two
newlines, which the adapter puts before every question. Write it with Python: shell `$(...)` drops the YAML block's
trailing newline, and `instruction` then fails on every task.

```bash
python -c 'import sys, yaml
agent = yaml.safe_load(open(sys.argv[1]))["opencode_math_agent"]["responses_api_agents"]["opencode_agent"]
open(sys.argv[2], "w").write(agent["system_prompt"] + "\n\n")' environments/opencode_math/config.yaml <data>/prefix.txt
python -m scripts.harbor_conversion_checks check results/harbor/dapo17k_harbor/rendered \
  --native-rows <data>/a/dapo17k_native.jsonl --native-id _index --id-map <data>/a/dapo17k_id_map.json \
  --prefix-file <data>/prefix.txt --hidden expected_answer \
  --hidden-from resources_servers/math_with_judge/task_data.py \
  --template <data>/a/dapo17k_harbor --template-hidden expected_answer
```

Expected, exit 0:

```text
PASS instruction: 5 task(s) equal their native prompt
PASS hidden-values: none of 0 hidden value(s) of 5 row(s) in instruction.md or environment/; 5 short value(s) [...]
PASS hidden-template: no agent-visible template file renders expected_answer
PASS secrets: no secret pattern in 40 file(s)
PASS local-paths: no local path in 40 file(s)
check: 5 passed, 0 failed, 0 skipped
```

`hidden-values` scans 0 values because these answers are shorter than the 8 characters it scans (most DAPO answers
are). `hidden-template`, which shows that no agent-visible template file names `expected_answer`, covers them.

**G3, negative control.** Build it from the second conversion, which no validate has touched, into a folder that
must not exist, then validate the copy:

```bash
mkdir -p <data>/wrong_solution
printf '%s\n' '\boxed{-987654321}' > <data>/wrong_solution/answer.template.md
python -m scripts.harbor_conversion_checks controls <data>/b/dapo17k_harbor <data>/dapo17k_harbor_wrong \
  --solution-dir <data>/wrong_solution
gym dataset validate <data>/dapo17k_harbor_wrong --sandbox opensandbox --config $H/harbor_arm.yaml
```

Expected: `controls` prints `wrote --solution-dir ... for 5 task(s)`. Validate exits 1 on a control by design and
labels the rows `oracle solved` even at reward 0, so read
`results/harbor/validation/dapo17k_harbor_wrong/rollouts_run_summary.json`: `counts.masked` 0, `counts.failed` 0,
every `reward` 0.0. The wrong-answer, empty and `--plant-reward` controls score 0 on both arms.

**G4-G6.** Native and Harbor smoke runs on the same 5 tasks (the commands under "Run both arms" on
`<data>/a/dapo17k_harbor` and `<data>/a/dapo17k_native.jsonl`), then `first-request` on captured requests and
`harbor_parity`.

Tests: `gym env test --resources-server math_with_judge` builds the server's venv and runs
`tests/test_harbor_adapter.py` with the server's own tests. With that venv already built:
`cd resources_servers/math_with_judge && .venv/bin/python -m pytest tests/test_harbor_adapter.py`.

## Parity result

50 DAPO-Math-17k tasks (`configs/parity_ids.txt`, drawn as "Regenerate" describes), 2 repeats per
arm, OpenCode 1.17.11 on both arms (`opencode_math_agent` as a local CLI natively, `opencode_agent` sandbox
sessions on Harbor), the same model behind an OpenAI-compatible model endpoint, concurrency 4, on OpenSandbox.

| Arm | Rollouts | Masked | Failed | Mean reward | 95% bootstrap interval |
| --- | --- | --- | --- | --- | --- |
| native | 100 | 0 | 0 | 0.940 | [0.860, 1.000] |
| Harbor | 100 | 0 | 0 | 0.940 | [0.860, 1.000] |

- Difference (Harbor - native) over 50 paired tasks: +0.000, paired bootstrap 95% interval [+0.000, +0.000].
  No task's mean differs.
- Reference solutions: 50/50 = 1.0 on both arms (Harbor oracle with every row verified in a separate sandbox,
  native oracle with a fake model).
- The copied grader gave the same verdict as the native `verify()` on all 200 real responses, each graded against
  its expected answer and a wrong one, plus 7 synthetic cases: 407 cases, 0 disagreements.
- The user message reached the model byte-identical on both arms (checked on all 100 Harbor rows, and on the wire
  for 5 paired tasks), with the same tools and the same 32000-token output cap. The Harbor half of that 5-task wire
  comparison was captured during the conversion's smoke runs, before the parity run; the Harbor config and dataset
  did not change in between.

Caveats:

- The parity is near the ceiling. 47 tasks scored 1.0 on all four rollouts and 3 scored 0 on all four, so the
  interval is [0, 0] because no task differs, not because the noise is that small. The task-level standard error
  is 0.034 per arm. The run can exclude shifts of about ±0.07, not small ones on hard problems. Before training or
  evaluating on harder problems, rerun on tasks the model solves 20-80% of the time with more repeats.
- The wrong-answer, empty and tamper controls were run on 5 tasks (all 0 on both arms), not on 50.
- A Harbor episode takes about 7 times longer: median 93 s vs 13 s, because each one starts two sandboxes and installs
  math-verify twice and OpenCode once.

## Differences from the native path

1. **Where OpenCode runs.** Natively on the agent-server host (bash, Python and network of that host); on Harbor in
   a 2-CPU, 4 GB sandbox with Python 3.13 and SymPy. Small for math: most rollouts use no tool.
2. **Model routing.** `native_arm.yaml` replaces the pairing's own provider with Gym's `policy_model`, so both arms
   use the same model server. No effect on parity.
3. **System prompt.** Equal except the provider prefix in the model-id sentence, the working directory, the
   built-in skill's path and the date line. Negligible.
4. **User message.** Byte-identical, CLI quoting included, through `argv_prompt_quoting: true`. Without it every
   task differs (outer quotes, escaped `"`).
5. **Tools and permissions.** The same tools reach the model, by name. One definition differs: the description of
   OpenCode's `bash` tool names its temporary folder, `<TMPDIR>/opencode`, which is the host's temporary folder
   natively and `/tmp/opencode` on Harbor (read from the request logs of the scripted-model runs). No reward effect
   expected; start the native CLI with `TMPDIR=/tmp` to make the two equal. The session applies its default
   permission profile plus the native allows and `skill: allow`, so rules for tools the model does not see differ.
6. **Output cap.** 32000 tokens per call on both (`max_output_tokens: 32000`; the session default is 131072).
7. **Timeout.** 900 s on both. Natively a timeout gives empty output and scores 0; on Harbor a timeout or an
   infrastructure failure may be masked. Matters only for runs that reach 900 s (none did).
8. **Transcript scope.** Native grading reads every OpenCode session, `task` subagents included; Harbor grades the
   root session only. A `\boxed{}` written only by a subagent can score natively, never on Harbor. Rare.
9. **Grading location.** The same functions, join, gate and timeout, but in a separate 2-CPU sandbox without the
   server's concurrency limiter; a slow SymPy case could time out more often.
10. **Metrics.** Harbor writes `reward` and `boxed_answer`; native returns `library_reward`, `extracted_answer` and
    pass@k metrics. No effect on the mean reward.
11. **Egress.** Every Harbor seed (agent and verifier sandbox) installs math-verify from PyPI, so both sandboxes
    need PyPI access. A prebuilt image with math-verify installed removes that once a registry the sandbox
    provider pulls from exists.
12. **Sampling.** Neither arm sets temperature or top_p; a Harbor row cannot carry them. Set any sampling default on
    the model server for both arms.

## Reward safety

With the default shared verification the grader runs in the agent's sandbox, where the agent is root: a control
that replaced `python3` there wrote its own reward without answering. With `separate_verifier_for_artifacts: true`
(`configs/harbor_arm.yaml`) the verifier runs in a fresh sandbox of the task image and the same control scores 0.
Every `task.toml` declares `artifacts = ["/logs/artifacts"]` for this; nothing there is graded. Use the config for
every run, training included.

The native arm has its own exposure: the local CLI runs as your user and can read any file, the native rows (which
hold the answers) included. Keep the rows outside the workspace and HOME.

## Known limits

- Only the `opencode_math` pairing with the judge off and boxed-answer extraction is reproduced; the adapter refuses
  the rest.
- The verifier reads the agent's final Responses API `response`, which only Gym's `harbor` server passes, so
  `gym dataset publish --harbor` refuses these tasks: they run under Gym only.
- A static load of the dataset (`discover_tasks`) needs the base image's record that a `gym dataset validate` run
  writes: run G2 before G1b's `check`.
- Every seed needs PyPI egress (difference 11); under a deny-by-default network policy every episode fails at seed.
- The generated grader's header names the Gym commit, so regenerating at another commit changes that one comment
  line even when `app.py` is unchanged; its sha256 in the header and the dataset README identifies the code.
