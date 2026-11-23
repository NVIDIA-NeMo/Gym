# reasoning_gym as Harbor task folders (the opencode_reasoning_gym pairing)

`adapter.py` converts one native pairing, `environments/opencode_reasoning_gym` (Reasoning Gym rows graded by the
`reasoning_gym` Resources Server, answered by the local OpenCode CLI), into a parameterized Harbor dataset:
`task-template/` plus one `tasks.jsonl` row per selected native row, run by the `harbor` Environment Server with no
Resources Server. It reads `environments/opencode_reasoning_gym/config.yaml` (or `--native-config`) and refuses,
with exit 2, what a task cannot reproduce: an agent entrypoint other than `legacy_app.py`, `system_prompt_mode`
other than `append`, an agent `timeout`, `thinking` or `opencode_version` other than the Harbor arm's (the
template's `task.toml`, `configs/harbor_arm_harness.yaml`, else the sandbox session's default version), a row with
more than one user message or with sampling or tools in `responses_create_params`, a row without
`metadata.source_dataset` or a `question` string, a row without a reference answer (it has no oracle), a row whose
task fields `task-template/task_data.py` refuses (the check Gym runs when it prepares the dataset), a line of
`--rows` that is not JSON, and an `--ids-file` entry that is not a 0-based row index.

This conversion was produced as a dry run of the `convert-env-to-harbor` skill: it was written from the skill alone,
then taken through the skill's gates and a full parity run (below).

Run every command below from the repository root, in the Python environment Gym is installed in. `adapter.py` needs
PyYAML and `nemo_gym` (to check each row against the template's `TaskData`), which that environment has; `prepare.py`
also needs reasoning-gym (see "Regenerate"). The tools are described in
[harbor_conversion_checks](../../../scripts/harbor_conversion_checks/README.md) and
[harbor_parity](../../../scripts/harbor_parity/README.md), the task-folder format in
[the task-folder docs](../../../fern/versions/latest/pages/data/task-folders/).

## Regenerate

```bash
python environments/opencode_reasoning_gym/prepare.py --task knights_knaves --size 200 --seed 42 \
  --output <data>/train_knights_knaves.jsonl
python -m scripts.harbor_conversion_checks subset <data>/train_knights_knaves.jsonl --id-field _index \
  --count 50 --seed 20261122 --out <data>/parity_ids.txt   # byte-identical to configs/parity_ids.txt
python resources_servers/reasoning_gym/harbor/adapter.py --rows <data>/train_knights_knaves.jsonl \
  --output-dir <data>/rg_harbor --ids-file resources_servers/reasoning_gym/harbor/configs/parity_ids.txt \
  --native-rows-out <data>/rg_native.jsonl --id-map-out <data>/rg_id_map.json
```

- Run `prepare.py` in an interpreter with reasoning-gym 0.1.25 installed (the version `../uv.lock` pins), such as
  the server's venv: its output depends on the seed, the size and the reasoning-gym version. The parity rows have
  200 lines and sha256 `5eee91d1e16752263f4c6113796eba1ed46b9bf6a23686ba3053a7c17b579831`; check yours before
  using `configs/parity_ids.txt`, because another rows file gives the same id to another problem.
- `--ids-file` takes 0-based row indexes of `--rows` (as `subset` writes them), not task ids, one per line, `#`
  comments allowed, order kept.
- `configs/parity_ids.txt` holds the 50 parity ids in draw order. The parity run's own ids file listed the same ids
  in the same order and differed only in its header (it named an empty exclusion file), so a dataset converted
  with this file differs from the parity run's in the ids-file sha256 line of the dataset README, not in any task.

## What the dataset holds

- Task id `rg-<n:05d>` (row 98 is `rg-00098`): row `n` (0-based) of the `--rows` file, which has no id field.
  `--id-map-out` maps each line of `--native-rows-out` (native `_ng_task_index`) to its task id. The dataset's root
  `README.md` records the rows file's sha256.
- `instruction.md`: `opencode_reasoning_gym_agent.system_prompt` (its YAML block keeps a trailing newline), two
  newlines, then the row's user message: what the native agent hands the OpenCode CLI. The CLI's argv quoting is
  reproduced by the harness setting `argv_prompt_quoting: true` (`configs/harbor_arm_harness.yaml`), not in the file.
- `tests/verifier.py` grades the agent's final Responses API `response` (only Gym's `harbor` server passes it):
  `tests/reasoning_gym_grader.py` is generated, the answer extraction copied out of `../app.py` unchanged
  (`<answer>` tags, else the last `\boxed{}`, else the last non-empty assistant message); the verifier restates
  only `verify`'s scoring glue (`reasoning_gym.get_score_answer_fn(metadata["source_dataset"])`, any scorer error
  scores 0) and gives the dict an attribute view for the copied method. Partial credit (knights_knaves gives
  `0.3 + 0.7 * matching / n`) passes through unchanged.
- `environment/Dockerfile` is an overlay generated from `requirements.txt`: `FROM` a uv image with gcc, pinned by
  digest, then reasoning-gym and every dependency at the version and hash `../uv.lock` pins, installed from PyPI at
  every seed. `cellpylib`, `pycosat` (C) and `zss` ship no wheel and are built there. Every seed needs PyPI egress.
- `solution/answer.md` is `<answer><reference answer></answer>`. The answer and the scorer's `metadata` (which
  holds the solution) are in `tasks.jsonl` only, never in the instruction or the image (but see "Reward safety").
- `task.toml` declares `artifacts = ["/logs/artifacts"]`: with `separate_verifier_for_artifacts: true` the verifier
  runs in a fresh sandbox of the task image, out of the agent's reach.

## Gates on a 5-task smoke subset

The adapter writes a parameterized template, so the static check needs the folder a validate renders: run the gates
in this order. The expected results are those of an acceptance run of these steps.

```bash
H=resources_servers/reasoning_gym/harbor/configs; R=<data>/train_knights_knaves.jsonl
grep -v '^#' $H/parity_ids.txt | head -5 > <data>/smoke_ids.txt   # 98 138 58 29 100
```

Convert the smoke subset as its own dataset. `adapter.py` has no `--limit`; `--ids-file` writes the task folders and
the native rows from the same ids. Do not select tasks with gym's `--limit` either: Harbor runs tasks in task-id
order and the native arm in rows-file order, so the two paths would run different tasks.

1. **Determinism.** Convert twice, into two parent folders with the same file names, and compare the parents (so the
   native rows and id maps are compared too):

   ```bash
   for x in a b; do
     python resources_servers/reasoning_gym/harbor/adapter.py --rows $R --output-dir <data>/$x/rg_smoke \
       --ids-file <data>/smoke_ids.txt --native-rows-out <data>/$x/rg_smoke_native.jsonl \
       --id-map-out <data>/$x/rg_smoke_id_map.json
   done
   diff -r <data>/a <data>/b
   ```

   Expected: `Wrote 5 tasks to ...` twice and no `diff` output. Differently named side files make `diff -r` of the
   parents differ. The dataset's `README.md` records the file names (not the folders) of `--rows` and `--ids-file`
   and the Gym commit, so other input names or another commit change that file only.
2. **Oracle (G2).**

   ```bash
   gym dataset validate <data>/a/rg_smoke --sandbox opensandbox --config $H/harbor_arm.yaml --concurrency 4
   ```

   Expected: exit 0. Read `results/harbor/validation/rg_smoke/rollouts_run_summary.json` under the directory gym
   ran in: `counts.masked` 0, `counts.failed` 0, and for each of the 5 tasks `reward` 1.0 and `verifier.mode`
   `separate`. It took 181 s; each verification takes 45 to 60 s, mostly the source builds.
3. **Static check (G1).** `check` refuses the template (exit 2). It reads the folder the oracle rendered,
   `results/harbor/rg_smoke/rendered` under the directory gym ran in. The prefix is the pairing's `system_prompt`,
   which already ends in one newline, plus two newlines. Write it with Python (shell `$(...)` drops the newline):

   ```bash
   python -c "import yaml; c = yaml.safe_load(open('environments/opencode_reasoning_gym/config.yaml')); \
   a = c['opencode_reasoning_gym_agent']['responses_api_agents']['opencode_agent']; \
   open('<data>/prefix.txt', 'w').write(a['system_prompt'] + '\n\n')"
   python -m scripts.harbor_conversion_checks check results/harbor/rg_smoke/rendered \
     --native-rows <data>/a/rg_smoke_native.jsonl --native-id _index --id-map <data>/a/rg_smoke_id_map.json \
     --prefix-file <data>/prefix.txt --hidden answer --hidden metadata \
     --template <data>/a/rg_smoke --template-hidden answer --template-hidden metadata
   ```

   Expected, exit 0:

   ```text
   PASS instruction: 5 task(s) equal their native prompt
   PASS hidden-values: none of 15 hidden value(s) of 5 row(s) in instruction.md or environment/; 84 short [...]
   PASS hidden-template: no agent-visible template file renders answer, metadata
   PASS secrets: no secret pattern in 40 file(s)
   PASS local-paths: no local path in 40 file(s)
   check: 5 passed, 0 failed, 0 skipped
   ```

   `answer` and `metadata` are the fields `../task_data.py` gives to `verify` only. Only values of 8 or more
   characters are scanned (the answer sentence, `knights_knaves`, ...); the 84 short ones are names, roles,
   booleans and numbers. `hidden-template` covers them: no agent-visible file renders `answer` or `metadata`.
4. **Negative control (G3).** Build it from a conversion that was never validated, into a folder that must not exist:

   ```bash
   mkdir <data>/wrong_solution
   printf '%s\n' '\boxed{-987654321}' > <data>/wrong_solution/answer.template.md
   python -m scripts.harbor_conversion_checks controls <data>/b/rg_smoke <data>/rg_smoke_wrong \
     --solution-dir <data>/wrong_solution
   gym dataset validate <data>/rg_smoke_wrong --sandbox opensandbox --config $H/harbor_arm.yaml --concurrency 4
   ```

   Expected: validate exits 1 on a control by design, and its table labels every row `oracle solved` even at reward
   0. Read `results/harbor/validation/rg_smoke_wrong/rollouts_run_summary.json`: every `reward` 0.0 with
   `answer_extracted` 1.0 (the answer was read and graded wrong), `counts.masked` 0, `counts.failed` 0.
5. **Native smoke, Harbor smoke, comparison (G4 to G6) and full parity:** "Run both arms" below.

## Run both arms

The dataset is tamper-safe only when run with `configs/harbor_arm.yaml` (separate verification).

```bash
H=resources_servers/reasoning_gym/harbor/configs
gym dataset validate <data>/rg_harbor --sandbox opensandbox --config $H/harbor_arm.yaml   # oracle scores 1
gym eval run <data>/rg_harbor --agent opencode_agent --sandbox opensandbox \
  --config $H/harbor_arm.yaml --config $H/harbor_arm_harness.yaml \
  --model-type openai_model --model <model> --model-url <url> --num-repeats 2 --concurrency 4
```

Native arm on the same rows (`--native-rows-out`), with `configs/native_arm.yaml`. It routes the model through
`policy_model` and isolates the local CLI from the machine. Supply its two values on both commands: a workspace
outside any git checkout and a fresh empty HOME per run (create both first; OpenCode writes state into HOME, so do
not reuse one). Put an OpenCode 1.17.11 binary first on `PATH`; otherwise the local path installs one into the
repository. To capture model requests for the first-request comparison, put the capture flags
(`++observability_enabled=true ++model_call_capture_dir=<dir>`) on `gym env start`, not on the client.

```bash
N="--config environments/opencode_reasoning_gym/config.yaml --config $H/native_arm.yaml"
ISO="++opencode_reasoning_gym_agent.responses_api_agents.opencode_agent.workspace_root=<workspace dir>"
ISO="$ISO ++opencode_reasoning_gym_agent.responses_api_agents.opencode_agent.env.HOME=<fresh empty dir>"
gym env start $N $ISO --model-type openai_model --model <model> --model-url <url>
# in a second shell, once the stack is ready; stop the stack afterwards
gym eval run --no-serve $N $ISO --agent opencode_reasoning_gym_agent --input <data>/rg_native.jsonl \
  --output <run dir>/rollouts.jsonl --model-type openai_model --model <model> --model-url <url> \
  --num-repeats 2 --concurrency 4
```

Compare with `python -m scripts.harbor_parity --native <run dir>/rollouts.jsonl --native-key _ng_task_index
--id-map <data>/rg_id_map.json --harbor results/harbor/rg_harbor/`.

Tests: `gym env test --resources-server reasoning_gym` builds the server's venv and runs
`tests/test_harbor_adapter.py` with the server's own tests. With that venv already built:
`cd resources_servers/reasoning_gym && .venv/bin/python -m pytest tests/test_harbor_adapter.py`.

## Parity result

50 knights_knaves tasks (`configs/parity_ids.txt`, 2 people each), 2 repeats per arm, OpenCode 1.17.11 on both arms
(`opencode_reasoning_gym_agent` as a local CLI natively, `opencode_agent` sandbox sessions on Harbor), the same
model, concurrency 4, on OpenSandbox.

| Arm | Rollouts | Masked | Failed | Mean reward | 95% bootstrap interval |
| --- | --- | --- | --- | --- | --- |
| native | 100 | 0 | 0 | 1.000 | [1.000, 1.000] |
| Harbor | 100 | 0 | 0 | 0.980 | [0.950, 1.000] |

- Difference (Harbor - native): -0.020, paired bootstrap 95% interval [-0.050, +0.000]: PASS. 2 of 50 tasks
  differ, both lower on Harbor, where the model hedged between two answers; the native grader gives that response
  the same 0.
- Model-free checks on all 50 tasks matched exactly on both arms: reference answer 1.0, empty and wrong answers
  0.0, and a partial answer (the first person's role flipped) 0.65.

Caveat: the native arm scored at the ceiling (100 of 100). The run shows that grading and the prompt match and rules
out a Harbor deficit larger than about 5 points. It cannot show a Harbor advantage, or parity on tasks the model
finds hard, because there was no room above native and almost no errors to compare. Before relying on it, rerun
parity on a harder or mixed set (a higher `n_people`, or several reasoning-gym tasks) where the native mean is well
away from 0 and 1.

## Differences from the native path

1. **Where OpenCode runs.** Natively on the agent-server host (its bash, Python and files, the rows file with the
   answers included); on Harbor in a sandbox with Python 3.13 and reasoning-gym installed. Small for these tasks.
2. **reasoning-gym in the agent's sandbox.** The overlay image is the task image of both the agent and the verifier
   sandbox, so the agent can import reasoning-gym (see "Reward safety"). Knights_knaves is solvable by brute force
   anyway. Small.
3. **Prompt.** The system prompt is folded into `instruction.md` as the CLI joins it; argv quoting is reproduced by
   `argv_prompt_quoting: true`. The user message reached the model identical on both arms.
4. **System prompt lines.** The provider prefix in the model-id sentence, the working directory, and the date line
   (host local time natively, UTC on Harbor) differ. None expected.
5. **Verifier errors.** A crash in the verifier outside the scorer gives Harbor `missing_reward` (0); natively the
   same crash is a server 500 and a failed row. A scorer error scores 0 on both. A row without
   `metadata.source_dataset` (a native 500) is refused at conversion.
6. **Timeouts.** The agent has 900 s on both. The Harbor verifier has 120 s; native verify has none.
7. **Model traffic.** The native local CLI makes an extra OpenCode title-generation request before the agent's call;
   the Harbor sandbox session makes none. No effect on reward, but a difference in calls and cost.
8. **Egress.** Every Harbor seed (agent and verifier sandbox) installs from PyPI.

## Reward safety

With `configs/harbor_arm.yaml` the verifier runs in a fresh sandbox of the task image, out of the agent's reach.
Use it for every run, training included.

The overlay installs reasoning-gym, with its dataset generators and scorers, in the agent's sandbox too. An agent
can therefore regenerate a dataset made with a known seed (`prepare.py` defaults to 42, and the parity rows use it)
and look up the reference answer, or call the task's own solver. For training, generate rows with an unpublished
seed. The proper fix is a separate verifier image without reasoning-gym in the agent's image, which needs a registry
the sandbox provider can pull from.

The native arm has its own exposure: the local CLI runs as your user and can read any file, the native rows (which
hold the answers) included. Keep the rows outside the workspace and HOME.

## Known limits

- Per-seed install cost: every seed builds `cellpylib`, `pycosat` and `zss` from source, in both the agent and the
  verifier sandbox. In the parity run the median Harbor episode took 99.2 s against 10.6 s natively (about 9x), and
  100 rollouts took 2560 s against 296 s. A model-free `gym dataset validate` of 50 tasks takes about 1000 s. A
  prebuilt image removes this once a registry exists.
- Every seed needs PyPI egress; under a deny-by-default network policy every episode fails at seed.
- Because the verifier reads `response`, `gym dataset publish --harbor` refuses these tasks: they run under Gym only.

## Regenerate requirements.txt

`environment/Dockerfile` embeds the bytes of `requirements.txt`, header included, so regenerate it exactly like
this, or every generated task changes. In a scratch folder:

```bash
cp <repo>/resources_servers/reasoning_gym/harbor/requirements.in .
uv export --frozen --no-hashes --no-emit-project --no-emit-package nemo-gym \
  --project <repo>/resources_servers/reasoning_gym > constraints.txt
# overrides.txt and excludes.txt: one requirement per line, from [tool.uv] override-dependencies and
# exclude-dependencies of resources_servers/reasoning_gym/pyproject.toml
uv pip compile requirements.in -c constraints.txt --overrides overrides.txt --excludes excludes.txt \
  --only-binary :all: --no-binary cellpylib --no-binary pycosat --no-binary zss --generate-hashes \
  --python-version 3.13 --python-platform x86_64-manylinux_2_28 -o requirements.txt
cp requirements.txt <repo>/resources_servers/reasoning_gym/harbor/requirements.txt
```

The file's second line must then read exactly `#    uv pip compile requirements.in -c constraints.txt ...
-o requirements.txt`, as committed.
