# Configs, evidence and tests (items 12-15)

Line numbers are at main `eeccee711d`.

- `S` = `.agents/skills/add-benchmark/SKILL.md`
- `P` = `.agents/skills/add-benchmark/references/patterns.md`
- `E` = `nemo_gym/cli/env.py`
- `NE` = `fern/versions/latest/pages/contribute/environments/new-environment.mdx`

## Where a benchmark lives

| Path | What it holds | Checked by |
|---|---|---|
| `resources_servers/<name>/` | `app.py`, `configs/*.yaml`, `data/example.jsonl`, `tests/`, `README.md` | CI: the server's tests plus the data validator (`.github/workflows/unit-tests.yml:257-281`) |
| `benchmarks/<name>/` | `manifest.yaml`, `config.yaml`, `prepare.py`, `README.md`, data | `gym env validate <name>` |
| `environments/<name>/` | Same layout, `kind: environment` | `gym env validate <name>` |
| Harbor task directories | `gym dataset init` output | `gym dataset validate` (see `harbor-format.md`) |

`gym env init` prints that it is deprecated for new datasets: use `gym dataset init <name>` for a Harbor-shaped dataset. It stays for custom Resources Servers "until tasksets can ship their own tools" (`nemo_gym/cli/main.py:953-956`). That message is not a review rule: the Gym and Harbor formats are equal choices for a new dataset (decided 2026-11-11).

## What CI checks on a changed Resources Server

With `should_validate_data=true`, which CI forces for every changed `resources_servers/*` (`unit-tests.yml:275`), `gym env test` requires (E:258-307):

- `data/example.jsonl` with exactly 5 rows;
- `data/example_metrics.json` from `+mode=example_validation`;
- `data/example_rollouts.jsonl` with 5 rows;
- no `*conflict*` files.

CI checks only that the rollouts file exists. It does not check that the rows came from a real model. Coverage of a server's own tests is not gated: `E:342` runs plain `pytest`; the 95% bar in `pyproject.toml:787` is for the core library (S:189 says otherwise; see `known-conflicts.md`).

## Config-only PRs (item 12)

A PR that adds or changes only YAML (a pairing, a benchmark recipe, an Environment Server config):

- **Check:** `gym env validate --config <file>` composes the config as a run would, applies the server-block checks (`domain` required, dataset `type` and `license`: `nemo_gym/config/config_types.py:478, 595-598, 795`), and validates each `environment_servers` block against its own config class (E:679-719, E:971-1023). With a manifest it also checks the manifest mirrors the config and parses every row as a rollout input (`nemo_gym/environment/validation.py:637-705, 773-848`).
- **It never** starts a server, runs the verifier, calls a model, or reads `verified`.
- **Evidence:** the validate output plus a real pairing run. Do not ask for unit tests that resolve the YAML and assert its bindings; they test no behaviour (team rule, 2026-11-06).
- **New work on the session path.** A new config wires `single_agent_turn` (preferred, with a taskset), `single_agent_turn_legacy` (flat rows) or `harbor`, not `legacy_agent` unless the benchmark needs only one harness (item 1), and names no deprecated agent and no agent without session support (item 1).

## The `verified` flag (item 12)

- **What writes it.** The `add-verified-flag` hook runs on `^resources_servers/.*/configs/.*\.yaml$` only (`.pre-commit-config.yaml:58-64`). For every `resources_servers` block without `verified`, including a partial override block in a pairing config, it adds `verified: False` (`scripts/add_verified_flag.py:26-40`).
- **What it breaks.** When it changes a file it rewrites it with `yaml.dump` (`add_verified_flag.py:42-45`): the SPDX header, every comment and flow-style lists are lost, anchors are expanded, long strings rewrapped. It exits 1, so the first pre-commit run fails.
  - Ask the author to write `verified: false` in every `resources_servers` block, override blocks included, so the hook leaves the file alone.
  - A config in the diff that lost its SPDX header or comments is the hook's output; ask for the original.
  - `benchmarks/` and `environments/` are not covered by the hook; their manifests use `experimental` instead.
- **What it means.** Nothing in code gates on it. It is read into server metadata (`nemo_gym/config/server_metadata.py:55`), and the README "Verified" column is commented out (`scripts/update_env_list.py:359-381`). 4 of 315 configs set it to `true`.
- **Flipping to `true`** (decided 2026-11-11): the author flips it in their own PR once the add-benchmark exit checks pass, and links the run folder and its `quality_summary.json` (S:268). The PR also shows the baselining numbers below. A reviewer checks that evidence; a flip without it is a finding. `CONTRIBUTING.md:356-357` says maintainers flip it; the guide follows S:268 (`known-conflicts.md` 40).
- **`experimental: false`** in a manifest needs approval from the Gym team (`fern/versions/latest/pages/contribute/environments/adding-a-benchmark.mdx:104`).

## The README table

`update-readme-table` (`scripts/update_env_list.py`) owns the root `README.md` rows. It reads datasets only from `responses_api_agents` blocks and skips a config with none (`update_env_list.py:151-182, 296`). A config with its datasets on the Resources Server or Environment Server block, as the scaffold writes it, gets no row. A hand-edited row is still a finding; a missing row may be the hook's limit, not the author's.

## Evidence (item 13)

**Behaviour-changing environment work needs real smoke rollouts with a model,** and someone must inspect the agent's and the verifier's behaviour. Green unit tests are not enough (repo `AGENTS.md`, "Quality bar"; NE:158-174; S:191-209). A docs-only or metadata-only change may say `N/A` with a reason.

**What the PR body lists** (`AGENTS.md`, "Pull Requests"; `.github/PULL_REQUEST_TEMPLATE.md:3-17`):

- the exact validation run;
- rollout evidence, or `N/A` with a reason;
- impact on published benchmark results;
- for add-benchmark exit checks: the run folder, `quality_summary.json`, and any ignored health checks (S:225, S:268).

**Baselining a new benchmark:**

- A profiling run: `gym eval run` then `gym eval profile --inputs <materialized> --rollouts <jsonl>` (`nemo_gym/cli/main.py:1744-1754`). It writes `<rollouts>_reward_profiling.jsonl`, `_agent_metrics.json` and `_repeat_level_metrics.json` (`nemo_gym/metrics/reward_profile.py:539-563`). Masked rows are left out of quality metrics but counted in completion.
- Code sets no threshold. **Reviewers apply the add-benchmark bar** (P:686-706, decided 2026-11-11):
  - Models: the policy of interest, a closed-source baseline, and an open instruct and an open thinking model (P:690-695).
  - Open models score 30% or more, and closed models score at or above open ones. If either fails, the PR shows what the author investigated (P:699-700).
  - Both instruct and thinking models run, and someone read failure cases in the rollout JSONL, not only the aggregate (P:701-702).
  - The highest-scoring open model is repeated, raising `num_repeats` until variance is under 1% (P:706).
- The public docs' lighter bar (NE:176-186: about 500 samples on Qwen3-4B or 30B, 16 responses each) is not enough for a new benchmark.
- **A training environment** (not a benchmark) needs only a real-model smoke with the agent's and verifier's behaviour inspected (decided 2026-11-11; `CONTRIBUTING.md:337-341`, NE:176-186). Don't ask for a reward profile or a NeMo RL training run (P:708-713 suggests one); welcome them if the author adds them. It merges with `verified: false`; flipping that still needs the baselining numbers.
- **An external benchmark** reproduces its published numbers before and after integration (S:30, S:229).
- **A verifier or reward change** shows before/after numbers on the same inputs, or proof the distribution did not move.
- **Health:** `gym eval health-check` (`nemo_gym/cli/main.py:1581`); compare runs with `gym eval compare` (`main.py:2062`).

**Grading evidence (add-benchmark conformance, S:233-258):**

- a scored row carries a grading artifact from `GRADING_ARTIFACTS` (`nemo_gym/orchestration/recovery.py:4419`), bound to its response and reward;
- the benchmark's characteristic failure is typed and routed, and a genuine wrong answer stays an unmasked 0;
- a cut and resume keeps settled rows.

**A scripted-model smoke** (stub Model Server) is good evidence for failure and retry paths. It does not replace one real-model run.

## Tests (item 14)

- **Each claimed behaviour has a test that fails when the fix is reverted.** No pass-through tests (`AGENTS.md`, "Quality bar").
- **A stateful server** calls `check_resources_session_contract` (`nemo_gym/server_runtime/testing/session_conformance.py:26`), with `verify_body` when its score depends on session state.
- **The verifier is tested on the known-good and the known-bad answer,** and on each infrastructure failure path: masked, not 0.
- **`VERIFIER_FIXTURE`** is exported where fixture-based `gym env test NAME` needs it (E:1028-1043).
- **External tools:** skip with `pytest.mark.skipif(shutil.which("tool") is None, ...)`, or auto-install through `ensure_<tool>()` in `model_post_init` and a `pytest_configure` hook (`AGENTS.md`, "External Tool Auto-Install"). A dependency `conftest` installs fails loudly; a warning that silently skips in CI is a finding.
- **Every wait is bounded** (`asyncio.wait_for`).
- **What the PR ran:** `gym env test --resources-server <name>` for each changed server; every test under `tests/` that names a changed config (`grep -rn <name> tests/`). Sandbox and lifecycle tests that skip on macOS need a Linux run.

## Public text (item 15)

- **Guardwords.** A pre-commit hook (`.pre-commit-config.yaml:36-39`) sees staged files only. The CI job (`.github/workflows/guardwords-check.yml:16-59`) runs on the PR mirror with the PR number, and in practice it also fails on the PR description. Keep cluster names, internal hosts, internal paths and ticket ids out of the title, body, commits and code comments. Write "cluster" instead of a cluster name.
- **Title** in Conventional Commit form; commits signed off (`git commit -s`). `-S` signing is optional (`AGENTS.md`, "Quality bar"; S:294 says otherwise).
- **Provenance.** Adapted external code records source URL, commit SHA and license in the PR body and in every file header (`AGENTS.md`, "Code provenance"). New files carry the SPDX header.
