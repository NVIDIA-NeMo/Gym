# workplace_assistant as Harbor tasks

`adapter.py` turns workplace_assistant rows (the Hugging Face `nvidia/Nemotron-RL-agent-workplace_assistant` splits,
or `../data/example.jsonl`) into a parameterized Harbor dataset: `task-template/` plus one `tasks.jsonl` row per
native row. `gym eval run` and `gym dataset validate` render one task folder per row. This environment is the
example for Resources-Server tools with per-session state: the tools and their tables move into a per-episode sidecar.

Run every command from the repository root, in the Python environment Gym is installed in. The adapter imports
`nemo_gym` to check each row against the template's `TaskData`, and nothing else beyond the standard library. Tools
used below:
[harbor_conversion_checks](../../../scripts/harbor_conversion_checks/README.md) and
[harbor_parity](../../../scripts/harbor_parity/README.md). Task-folder docs:
[data/task-folders](../../../fern/versions/latest/pages/data/task-folders/).

## What the conversion does

| Piece | Native | Harbor task |
| --- | --- | --- |
| Agent image | (none with `simple_agent`; OpenCode pairing: a public image) | `main`: the same public image, pinned by digest (`environment/Dockerfile`); it holds nothing of the task |
| Tools and tables | The Resources Server's 27 tools on per-session tables | The Compose service `tools` (`environment/docker-compose.yaml`): `tool_server.py` serves the server's own `utils.py`, `workplace_assistant_tools/` and `csv_data/` over MCP at `http://tools:8000/mcp`, one fresh set of tables per episode |
| Instruction | System message + user message | `instruction.md`: the system message, a blank line, the user message, as one user message |
| Grading | `verify()` replays the response's `function_call` items with `is_correct` | `tools` journals every call (name, JSON arguments) to `/var/lib/wa-journal/calls.jsonl` before running it; a separate verifier sandbox receives the journal as an artifact and scores it with the server's own `is_correct` (`tests/grade.py`) |
| Hidden data | `ground_truth` in the row | `tests/ground_truth.json` (verifier only) and `solution/calls.json` (oracle only) |

The adapter copies the tool and grading code unchanged into `environment/wa/` and `tests/wa/`, under its package
path so its imports work as they are. Each Python copy starts with one line
`# Vendored unchanged from resources_servers/workplace_assistant/<file> at <Gym commit>; source sha256 <hex>; license
Apache-2.0`; the source's bytes follow it. Both sandboxes install `environment/requirements.txt` and
`tests/requirements.txt`, which pin every package (pyarrow included, so pandas stores strings as the server's does)
to the server's `uv.lock` with hashes. Rewrite them with `pin_requirements.py` after the lock changes.

## Generate

```bash
python resources_servers/workplace_assistant/harbor/adapter.py --rows <rows.jsonl> --output-dir <dataset dir> \
    --ids-file <ids.txt> --native-rows-out <native rows.jsonl> \
    --opencode-rows-out resources_servers/workplace_assistant/data/validation_opencode.jsonl --id-map-out <id map.json>
```

- Only `--rows` and `--output-dir` are required. `--native-rows-out` and `--id-map-out` are for the static check
  and `harbor_parity`; `--opencode-rows-out` is for the native arm only (it writes into the checkout).
- `--ids-file`: row `id`s, one per line, `#` comments allowed, order kept. Task `wa-validation-0115` is row `id` 115.
  `--limit N` keeps the first N selected rows, for the tasks and every side file alike.
- `--native-rows-out`: the selected raw lines, unchanged and in the same order (the input of the static check).
- `--opencode-rows-out`: the same rows as the native OpenCode config runs them (no `tools`, `parallel_tool_calls` or
  `agent_ref`, which OpenCode's sandbox session refuses), at the path `configs/native_pairing.yaml` reads, which
  `../data/.gitignore` ignores.
- `--id-map-out`: `{row id: task id}` for `python -m scripts.harbor_parity --native-key id --id-map`.
- `--solution none|wrong` writes negative controls (no call; one email no row asks for). Prefer `wrong`: rows with
  an empty ground truth (11 of the 50 parity rows) score 1 for no call.
- Exit 2, with no dataset written, for a non-empty output folder, an output folder or file that cannot be written,
  a `--limit` below 1, an id in the ids file that is missing or repeated, no row selected, an unreadable rows or ids
  file, an ids file that is not UTF-8, a rows line that is not JSON or not UTF-8, two selected rows with the same
  `id`, a row `id` that is not an integer, a row missing `id`, `category`, `environment_name` or
  `responses_create_params`, or a row the tasks do not reproduce: roles other than one system and one user
  message, message content that is not plain text, a request field other than `input`, `tools`,
  `parallel_tool_calls` and `temperature`, a temperature other than 1.0, or a ground truth that is not a list of
  `{name, arguments}`. It also exits 2 for a row whose task fields the template's `TaskData`
  (`task-template/task_data.py`) refuses, checked the way Gym checks it at prepare. A write that fails partway
  through the dataset also exits 2, and can leave a partial output folder.
- The rows' tools list is not read: every task serves the 27 tools of get_tools(TOOLKITS), as the native OpenCode arm
  does through mcp_tools(); a row offering a different set would be converted without notice.
- `<dataset dir>/README.md` records the rows file's name, sha256 and line count, the task count, the ids file's name
  and sha256, the Gym commit, each vendored file's sha256 and the command that regenerates the dataset. That command
  names the rows and ids files by basename only (paths are relative to where you ran the adapter) and lists which
  side-output flags you passed, without their paths.

## Run both arms

`configs/` holds the settings the parity run used:

- `harbor_arm.yaml` on every Harbor `gym dataset validate` and `gym eval run`: OpenSandbox networking (the sandboxes
  of a Compose group reach each other by service name) and `task_mcp_grants: required` (OpenCode does not start a
  task's MCP servers; the Environment Server grants the task's server to it).
- `harbor_arm_harness.yaml` on the Harbor `gym eval run` only: temperature 1.0 through `openai_model`'s `extra_body`,
  because a Harbor task cannot carry the rows' `temperature`.
- `native_arm.yaml` with `native_pairing.yaml` for the native arm: the same temperature. `native_pairing.yaml` is the
  configuration the parity run's native arm ran with, on a patched copy of the server; it is not a supported pairing
  and does not run on main (below).

```bash
C=resources_servers/workplace_assistant/harbor/configs
MODEL="--model-type openai_model --model <model> --model-url <OpenAI-compatible endpoint>"

# Harbor arm.
gym eval run <dataset dir> --agent opencode_agent --sandbox opensandbox \
    --config $C/harbor_arm.yaml --config $C/harbor_arm_harness.yaml $MODEL --num-repeats 2

# Native arm, from the directory the adapter ran in: the config reads the rows at the path --opencode-rows-out wrote
# (resources_servers/workplace_assistant/data/validation_opencode.jsonl); it refuses --input.
gym eval run --config <repo>/$C/native_pairing.yaml --config <repo>/$C/native_arm.yaml \
    --agent workplace_assistant_opencode_agent --split validation $MODEL --num-repeats 2
```

Use absolute paths for datasets and configs. The model endpoint's API key and the OpenSandbox connection settings
come from your `env.yaml`, as for any Gym run.

**The native arm does not run on main yet.** Two fixes to the Resources Server, each its own follow-up change, must
land first:

1. **Episode seed.** `app.py`'s `seed_session` override answers only the agent-run seed, so every episode seed from
   the Environment Server is refused ("validation errors for ResourcesSeedSessionResponse"). Fix: drop the override
   and implement `has_session_state`, `initialize_session_state` and `release_session_state` over the existing
   per-session tables.
2. **Tool names.** OpenCode names the calls `workplace_assistant_<tool>`, and the MCP exposure's
   `normalize_tool_name` strips only Claude Code's `mcp__<server>__` prefix, so `verify()` scores 0 on every row
   whose ground truth is not empty. Fix: also map `<server name>_<tool>` to `<tool>` when `<tool>` is one of the
   server's exposed tools.

The Harbor tasks need neither: they grade the journal, which holds bare tool names.

## Gates

Run them in this order on a 5-task smoke subset, then on the 50 parity tasks. `<data>` is any folder outside the
checkout (absolute path). The smoke subset is the first 5 parity ids, converted as its own dataset:

```bash
C=$PWD/resources_servers/workplace_assistant/harbor/configs
A=resources_servers/workplace_assistant/harbor/adapter.py
grep -v '^#' $C/parity_ids.txt | head -5 > <data>/smoke_ids.txt
```

**1. Determinism.** Convert twice into two parent folders with the same file names. `diff -r` prints nothing and
exits 0. The dataset README names the rows and ids files and the side-output flags, so both runs need the same ones.

```bash
for x in a b; do python $A --rows <rows.jsonl> --output-dir <data>/$x/smoke --ids-file <data>/smoke_ids.txt \
    --native-rows-out <data>/$x/native_rows.jsonl --id-map-out <data>/$x/id_map.json; done
diff -r <data>/a <data>/b
```

**2. Oracle (G2).** The reference solution must score 1.0 on every task.

```bash
gym dataset validate <data>/a/smoke --sandbox opensandbox --config $C/harbor_arm.yaml
```

It exits 0. Read `results/harbor/validation/smoke/rollouts_run_summary.json` under the directory gym ran in (`smoke`
is the dataset folder's name). Pass: `counts.masked` 0, `counts.failed` 0, and every entry of `tasks` has `reward`
1.0 and `verifier.mode` `separate`. 5 tasks took about 130 s in a test run.

**3. Static check.** This adapter writes a parameterized template, so `check` refuses the dataset folder (exit 2, "is
a parameterized template"). It runs after the oracle, on the task folders `gym dataset validate` rendered into
`results/harbor/smoke/rendered` under the directory gym ran in. No `--prefix-file`: `instruction.md` is the system
text, a blank line and the user text, which is the native prompt `check` builds.

```bash
python -m scripts.harbor_conversion_checks check results/harbor/smoke/rendered \
    --native-rows <data>/a/native_rows.jsonl --native-id id --id-map <data>/a/id_map.json \
    --hidden-from resources_servers/workplace_assistant/task_data.py \
    --template <data>/a/smoke --template-hidden ground_truth
```

Expected on the smoke subset (exit 0):

```text
PASS instruction: 5 task(s) equal their native prompt
PASS hidden-values: none of 6 hidden value(s) of 5 row(s) in instruction.md or environment/
PASS hidden-template: no agent-visible template file renders ground_truth
PASS secrets: no secret pattern in 215 file(s)
PASS local-paths: no local path in 215 file(s)
check: 5 passed, 0 failed, 0 skipped
```

The 6 hidden values are the `arguments` strings of the 6 ground-truth calls (row 115 has two). Tool names are not
scanned: the rows' tool list already shows them.

**4. Negative control (G3).** A wrong call must score 0. Build a fresh conversion with its own folder name, into a
folder that does not exist yet (the adapter refuses a non-empty one), and validate it:

```bash
python $A --rows <rows.jsonl> --output-dir <data>/smoke_wrong --ids-file <data>/smoke_ids.txt --solution wrong
gym dataset validate <data>/smoke_wrong --sandbox opensandbox --config $C/harbor_arm.yaml
```

It exits 1 by design, and its log labels every row "oracle solved" even at reward 0. Read
`results/harbor/validation/smoke_wrong/rollouts_run_summary.json`: every `reward` 0.0, `counts.masked` 0,
`counts.failed` 0, and `verifier.rewards.journal_calls` 1 (the one wrong call reached the tools).

**5. Smoke runs, comparison and full parity (G4 to G6).** The two `gym eval run` commands of
[Run both arms](#run-both-arms) on the same tasks; the native arm needs the rows `--opencode-rows-out` wrote for
them. Compare the runs with `python -m scripts.harbor_parity --native-key id --id-map <id map.json>`.

The adapter test checks determinism, every refusal, the dataset README and that each vendored file is the server's
file byte for byte below its header, and grades fixed cases with the vendored `tests/grade.py` and the server's
`is_correct`. Run it with the server's dependencies: `gym env test --resources-server workplace_assistant`, or
`python -m pytest resources_servers/workplace_assistant/tests/test_harbor_adapter.py` from the repository root in
the server's venv.

## Parity result

The 50 rows of `validation.jsonl` listed in `configs/parity_ids.txt` × 2 repeats per arm, `opencode_agent` (OpenCode
1.17.11) on both arms, the same model behind an OpenAI-compatible model endpoint, temperature 1.0 on both,
OpenSandbox, the same agent image and a 900 s agent timeout. The rows are the Hugging Face dataset
`nvidia/Nemotron-RL-agent-workplace_assistant` at revision `f674d9bba17820d90f14eefe9156fdc018a0db60` (545 lines,
sha256 `6cce92f9cd8807983c4aa8f1006d5fab5e8c4f16ae28b90222739d98905df01b`). To rebuild the parity tasks:

```bash
REVISION=f674d9bba17820d90f14eefe9156fdc018a0db60  # pragma: allowlist secret
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(
    'nvidia/Nemotron-RL-agent-workplace_assistant', 'validation.jsonl', repo_type='dataset',
    revision='$REVISION', local_dir='<rows dir>'))"
python resources_servers/workplace_assistant/harbor/adapter.py --rows <rows dir>/validation.jsonl \
    --output-dir <dataset dir> --ids-file resources_servers/workplace_assistant/harbor/configs/parity_ids.txt \
    --native-rows-out <native rows.jsonl> --id-map-out <id map.json> \
    --opencode-rows-out resources_servers/workplace_assistant/data/validation_opencode.jsonl
```

| Arm | Rollouts scored | Masked | Failed | Mean reward |
| --- | --- | --- | --- | --- |
| Native, with the two fixes above applied in a patched copy of the server | 100 / 100 | 0 | 0 | 0.930 |
| Harbor (this conversion) | 100 / 100 | 0 | 0 | 0.960 |

Harbor − native: **+0.030**, 95% paired bootstrap interval over tasks **[−0.010, +0.090]** (SE 0.026); means differ
on 4 of 50 tasks, Harbor higher on 3. `python -m scripts.harbor_parity` statistical mode: PASS. The reference
solution scored 1.0 on all 50 tasks on both paths (`harbor_parity` strict mode PASS): on Harbor in a
`gym dataset validate` run, natively in a model-free probe of the patched server (`/seed_session`, then `/verify` with
the ground-truth calls under OpenCode's tool names), not a `gym` run. The same probe scored a wrong call 0 on all 50.

Caveats:

- The native arm is not main: it ran a patched server with the two fixes above (tools, tables and `verify()`
  unchanged).
- The native oracle and wrong-call results come from the probe above, not from rollouts. On Harbor, the wrong-call
  control ran on 5 tasks only (all 0), not on the 50.
- 12 of the 50 rows score 1 for doing nothing on both paths: 11 with an empty ground truth and one (row 474) whose
  ground truth sets a value the record already has. Without the 11 empty rows the difference is +0.013.
- The one systematic difference is prompt placement (difference 1 below). Row 90 scored 0/2 natively and 2/2 on
  Harbor. In a follow-up of rows 90 and 531 with 6 repeats per arm, the model treated not-yet-due tasks as overdue
  and changed them in 6 of 12 native rollouts and 0 of 12 Harbor rollouts.
- Model requests were captured on neither arm, so the first request was not compared byte for byte, and Harbor's
  temperature is confirmed in the resolved config, not on the wire.
- Episodes took a median of 57 s natively and 141 s on Harbor (a tools sidecar, per-episode installs and a separate
  verifier sandbox).

## Differences from the native path

1. **Prompt placement.** Natively, OpenCode appends the row's system text (which carries the task's date, "Today's
   date is ...") to its own system prompt. Here it is the top of the user message. Same words, different role. It
   changed behaviour on date-sensitive rows in the parity run (above): natively the model sometimes reasons from
   another date. To make the arms equal, give both the same placement, for example by passing the system text as
   the harness's system prompt on the Harbor arm or by putting it in the user prompt natively.
2. **Sampling.** The rows' `temperature: 1.0` reaches the model through `harbor_arm_harness.yaml`, not the request.
   `parallel_tool_calls: false` is dropped on both arms (OpenCode refuses it).
3. **Grading input.** Natively, the calls the harness reports; here, the calls that reached the tool server. An agent
   that calls the MCP endpoint itself (for example with curl) is graded on those calls here only.
4. **Call order.** Calls are replayed in the order they reached the tool server, natively in the order the harness
   reports them. They differ only for parallel state-changing calls whose order changes the tables.
5. **Masking and crashes.** A verification that cannot run (no journal, a failed install) exits 75 and is masked;
   native has no such path. If `is_correct` itself raises, the native rollout fails, while here it scores 0.
6. **Tool dispatch.** `tool_server.py` restates the server's `route_to_python_function` and Gym's MCP rendering of
   its reply (`app.py` imports `nemo_gym`, so it is not copied). The 27 tool names, descriptions, schemas and replies
   were equal to the native MCP exposure's.
7. **Runtime.** The sidecar and verifier run the image's Python 3.13 with packages pinned to the server's lock; the
   native server runs in its own venv.
8. **Steps.** `simple_agent`'s `max_steps: 6` has no counterpart on either OpenCode arm.

## Reward safety

The dataset is safe to run only with `configs/harbor_arm.yaml` and the task files as generated. Every task declares a
`[verifier.environment]`, so Gym grades it in a fresh sandbox that the agent never touches. The verifier reads only
the journal copied out of `tools`, which `tool_server.py` appends to before running each call; the agent can add to
it only by calling a tool. The ground truth lives in `tests/` and `solution/`, never in `instruction.md` or
`environment/`.

## Known limits

- Every episode installs packages from PyPI in `tools` and in the verifier: both sandboxes need that egress. A
  prebuilt image would remove it but needs a registry the sandbox provider can pull from.
- Tested on OpenSandbox with networking only (the Compose sidecar). Running the tools inside the agent's own image
  instead is not part of this conversion: it needs an image pushed to a registry, and a verifier that cannot trust
  files in the agent's sandbox.
- Uploading `tests/` into a separate verifier, granting a task's MCP server and copying an artifact out of a Compose
  service are Gym behaviour; upstream Harbor would need a verifier image.
