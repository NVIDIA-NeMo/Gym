# Harbor format: task directories (item 6)

Line numbers are at main `eeccee711d`.

- `H` = `environment_servers/harbor/app.py`

Use this file when the PR adds or changes a Harbor task directory, a `dataset.toml`, an `environments/harbor/<entry>/`, the loader under `nemo_gym/data/tasks/harbor/`, the `harbor` Environment Server, or `nemo_gym/verifiers/`.

Harbor and Gym format are equal choices for a new dataset (decided 2026-11-11). `gym env init` prints a preference for Harbor ("use `gym dataset init <name>` for a Harbor-shaped dataset", `nemo_gym/cli/main.py:953-956`); don't turn that into a finding against a Gym-format PR.

## The task directory

- `task.toml`; the directory is not a symlink (`nemo_gym/data/tasks/harbor/task.py:407-412`).
- `instruction.md` for a single-step task, read as text in the locale's default encoding (no explicit UTF-8: `nemo_gym/data/tasks/harbor/task.py:386`) with leading canary lines dropped; a decode error is a task error (`nemo_gym/data/tasks/harbor/task.py:379-395, 482-491`).
- `tests/`, unless the dataset root has a `verifier.py` (`nemo_gym/data/tasks/harbor/task.py:485-487`). `tests/test.sh` takes precedence over `tests/verifier.py` (`nemo_gym/data/tasks/harbor/python_verifier.py:65-90`).
- A multi-step task has `steps/<name>/instruction.md` and a `test.sh` in the step's tests or the task's (`nemo_gym/data/tasks/harbor/task.py:851-918`).
- **An image:** `[environment].docker_image` or `environment/Dockerfile`. The seed refuses a task with neither with 422 (H:1803-1806). A Dockerfile that needs a build has a recorded build (`nemo_gym/data/tasks/harbor/task.py:541-550`).
- Task IDs (directory names) are unique (`nemo_gym/data/tasks/harbor/materialize.py:157-162`). Discovery goes at most 6 levels deep and skips hidden directories (`nemo_gym/data/tasks/harbor/task.py:87-89, 785-815`).
- Linux only; network restrictions only on a separate verifier (`nemo_gym/data/tasks/harbor/task.py:448-477`).
- Unknown `task.toml` keys only log a warning (`nemo_gym/data/tasks/harbor/task.py:415-416`): a typo is silently ignored, so read them.

## `dataset.toml [gym]`

- Every `[gym]` table forbids unknown keys, and so do per-task overrides (`nemo_gym/data/tasks/harbor/dataset_config.py:44, 65, 75, 228-230, 257-259`).
- `train` and `validation` need a license (`nemo_gym/data/tasks/harbor/dataset_config.py:142-146`). `type = "example"` is refused (`nemo_gym/data/tasks/harbor/prepare.py:713-725`).
- `allowed_agents` is non-empty and distinct (`nemo_gym/data/tasks/harbor/dataset_config.py:101-114`), checked at prepare (`nemo_gym/data/tasks/harbor/prepare.py:293-299, 728-739`) and at dispatch, after materialize copies it into the generated dataset entry (`nemo_gym/data/tasks/harbor/materialize.py:259-260`; `nemo_gym/rollouts/rollout_collection.py:5309-5315`). The oracle is always admitted (`nemo_gym/data/tasks/harbor/prepare.py:996-997`).
- Per-task overrides go in `[gym.tasks."<id>"]` (`nemo_gym/data/tasks/harbor/dataset_config.py:56-69, 289-332`).
- Per-harness settings are not data: they live in `AGENT_OVERRIDES` (`nemo_gym/data/tasks/harbor/prepare.py:119-133`) or in the entry's YAML overlays (for example `environments/harbor/swe_interact/swe_interact.yaml`). A per-harness flag in `dataset.toml` is a finding.

## Typed task fields

- Optional `task_data.py` beside `dataset.toml`, and `task_data.json` in each task.
- Strict: unknown keys fail and values are not coerced (`nemo_gym/data/tasks/harbor/task_data.py:4-28`). `_ng_*` keys are reserved (`nemo_gym/data/tasks/harbor/task_data.py:51-57, 213-215`).
- Checked at prepare (`nemo_gym/data/tasks/harbor/prepare.py:308-315, 388`), at seed with 422 (H:705-720), and at collate (`nemo_gym/data/tasks/harbor/task_data.py:276-315`; `nemo_gym/data/dataset_preparation.py:1185-1206`).

## Rows and configs are generated, never hand-written

- The instruction becomes plain-string user content, and `task_data` carries `_ng_digest` (`nemo_gym/data/tasks/harbor/materialize.py:84-88, 113-154`).
- The run config is generated with `environment_routing_mode: taskset` and `environment_server_routes` (`nemo_gym/data/tasks/harbor/materialize.py:265-321`).
- The seed checks the directory digest (409, H:654-656), the row digest (422, H:680-685) and the row's turns (H:1811-1818).
- A hand-edited Harbor JSONL or run config in the diff is a finding: the digest checks refuse it, or worse, an edited prompt runs. The seed never compares the row's `responses_create_params` with `instruction.md` (H:1782-1818).

## Text only: images mask the turn

- ATIF export accepts only text content parts, plus `refusal` in the model's own output (`nemo_gym/trajectory/atif_export.py:1139-1170`; everything else is refused at `:1142-1145, 1155-1162`).
- An `AtifExportError` masks the turn as `harbor:trajectory_export_failed`, reward 0 and `mask_sample: true` (H:230, H:1047-1054).
- So a harness that puts any image in its transcript (in the prompt or in a tool output) masks every turn. This is open issue #5871.
- The kind is namespaced, so recovery never retries it (`serving.md`). A Harbor benchmark with image tools produces only masked rows until #5871 lands.
- No `data/task-folders` or `agent-interactions/harbor.mdx` page says transcripts must be text-only (`known-conflicts.md`).

## Deprecated agents

- `harbor_agent` and `harbor_agent_general` are deprecated (#4826, merged 2026-11-10). Startup warnings: `responses_api_agents/harbor_agent/app.py:236-242`, `responses_api_agents/harbor_agent_general/app.py:283-291`; README banners at lines 3-11.
- The replacement is the `harbor` Environment Server with a session harness, and `harbor_oracle_agent` for validation.
- Eight benchmarks still use the old agents: biomnibench_da, biomysterybench, bixbench_hypothesis, genebench_pro, labbench2_pdf, legal_agent_bench, openthoughts_tblite, skillsbench (`responses_api_agents/harbor_agent/README.md:6-7`; for example `benchmarks/biomnibench_da/config.yaml:7`). Not a precedent.

## `nemo_gym/verifiers/`

- Standard library only, and it imports nothing else from `nemo_gym` (`nemo_gym/verifiers/__init__.py:4-10`). Nothing enforces it except the docstring, a Python 3.8 syntax test and an import-linter exemption (`tests/unit_tests/test_import_linter_coverage.py:32-38`).
- Only its top-level `*.py` files are copied into the sandbox (`nemo_gym/data/tasks/harbor/python_verifier.py:185-192`); a subpackage would not ship.
- Python 3.8 is the floor (`nemo_gym/data/tasks/harbor/python_verifier.py:55-58`).
- Grading code for one benchmark belongs in its `tests/verifier.py` or the dataset's `verifier.py`, not here.

## What the PR ships

From `fern/versions/latest/pages/contribute/environments/adding-a-benchmark.mdx:140-151`:

- **Hub dataset:** `environments/harbor/<name>/` with `README.md`, any `*.yaml` overlays, and a `dataset.toml` whose `[gym] source` is pinned. Git keeps only those files there (`.gitignore:258-266`).
- **Authored directories:** committed outside `environments/harbor/`, without the prepare records.
- **`gym dataset validate` passes.** It fails when no selected task has a `solution/`, when an oracle result is missing or failed, when a selected solution goes `unvalidated`, or when a task is skipped (`nemo_gym/data/tasks/harbor/validation_report.py:103-107, 114-138, 157-159`). A task without `solution/` is left out of the run and listed as `unvalidated` without failing it (`validation_report.py:111-112, 155-156`; `nemo_gym/cli/eval.py:1874-1877`), so read that list. Exclusions are listed in the README.
- No root README table row.
- The doc asks to "run one harness on a few tasks" (`adding-a-benchmark.mdx:147`). This guide asks more (item 14, a review-team rule): a real pairing run with each harness in `allowed_agents`, in addition to the oracle.

## Review questions

- Does `gym dataset validate` pass, which tasks did it list as `unvalidated` (no `solution/`), and are exclusions listed in the README?
- Does every harness in `allowed_agents` keep its transcript text-only on these tasks?
- Is anything per-harness written into the data instead of an override?
- Does a change to the loader or the `harbor` server reach other Harbor entries (`ls environments/harbor/`)? Name them.
