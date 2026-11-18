# Skill opening

When a task calls for a skill, does the model open its instructions? This benchmark scores the instructions returned in Harbor's recorded tool results. It does not measure task completion or the official SkillsBench score.

**Version 0.2.1 is experimental.** It uses Harbor's existing agents, skill loading, timeouts, and trajectories. The bounded validation below reports new observations. The earlier experiment's evidence was deleted without a backup; its results and 2,092-trial scorer comparison cannot validate or reproduce this release.

## Tasks and configurations

The frozen selection contains 110 positive prompts: 35 BioNeMo, 50 selected SkillsBench, and 25 Addy Osmani cases. [selection.json](selection.json) fixes the prompts, upstream revisions, and each task's original skill catalog. Preparation excludes upstream evaluation answers and retains attribution. Each Harbor task supplies its own catalog through `environment.skills_dir`.

| Configuration | Harbor agent | Version | Guidance |
|---|---|---|---|
| `pi` | Pi | 1.3.0 | Native defaults |
| `codex` | Codex | 0.156.1 | Native defaults |
| `codex_guidance` | Codex | 0.156.1 | Skill-usage instructions enabled in a supplied model catalog |
| `hermes` | Hermes | `v0.22.2` | Native defaults |
| `hermes_guidance` | Hermes | `v0.22.2` | Tool-use enforcement enabled |

[profiles.py](profiles.py) supplies the guidance settings and two compatibility fixes. Hermes installs into `/tmp/hermes` from its pinned release commit and uses the supported version probe. Codex selects the main session identified by its CLI output and retains child-session traces; Harbor 0.25.0 can otherwise select a later child log as the main trajectory. Harbor still owns installation, model execution, skill discovery, timeouts, and conversion of each individual session. There is no custom runner or inference proxy.

Both Codex configurations use a standard custom provider with `supports_websockets=false`. This keeps transport consistent across models and avoids a WebSocket routing failure observed at the NVIDIA endpoint; the same native request succeeded over HTTP. Harbor supplies the endpoint and credentials. This transport setting does not add skill-use instructions.

## Scoring

The host-side [verifier](verifier.py) pairs tool calls with their returned content in Harbor's root trajectory. A read must contain 120 normalized characters from the intended skill body, or the whole body if shorter. Front matter, assistant claims, explicit tool errors, and shared boilerplate without a direct skill read do not count. The [matching rule](scoring.py) preserves the original whitespace and Pi character-counting conventions.

A verified successful read of the intended skill earns 1, even if later work errors or times out. A finished, observable attempt without that read earns 0. Task completion and application of the instructions are not requirements. Missing, malformed, copied, or incomplete trajectory evidence is an error, not a zero. A root-session miss with delegated trajectories or a recorded completed delegation without child logs remains unscored until those traces can be verified; this version does not infer child-agent success. Use the mean reward across valid trials as the skill-opening rate, not pass@k. Report execution errors and failed/unscored counts alongside it, and report the three collections separately.

The task has a 120-second Harbor agent timeout. Agent defaults govern reasoning, retry behavior, tool selection, and output length. Unlike the original experiment, this version does **not** enforce a shared three-request cap or collect independent filesystem-access evidence. It trusts Harbor's recorded tool results. It must have new result tables rather than reuse historical percentages.

## Prepare

Use Linux with Docker, uv, and Python 3.13.14. The configurations and adapters are tested with Harbor 0.25.0. Install Gym's existing Harbor component and pin that runtime:

```sh
uv sync --project responses_api_agents/harbor_agent_general --python 3.13.14 --no-default-groups
uv pip install --python responses_api_agents/harbor_agent_general/.venv/bin/python 'harbor[opensandbox]==0.25.0'
source responses_api_agents/harbor_agent_general/.venv/bin/activate
```

Audit the upstream sources without packaging:

```sh
python -m benchmarks.skill_trigger.prepare --fetch --audit-only \
  --output benchmarks/skill_trigger/data/source-audit
```

**Full packaging is still blocked by unresolved terms for nine cases.** The exact files and reasons are in `selection.json` under `release_blockers`. This refactor does not clear those terms or remove those cases. No upstream skill bodies are committed.

After clearance, prepare a release without `--audit-only`, calculate its manifest SHA-256, and package it:

```sh
python -m benchmarks.skill_trigger.package \
  --release /absolute/path/to/release \
  --manifest-sha256 MANIFEST_SHA256 \
  --output /absolute/path/to/tasks
```

The package contains standard Harbor task directories and `gym.jsonl`. Expected targets and the release manifest remain on the host, outside task images. Preparation refuses existing output directories.

## Run

Supply `OPENAI_API_KEY` in the worker environment. The supplied profiles use an OpenAI-compatible endpoint and Harbor's `openai/MODEL_ID` naming. Pi uses the Responses API. Set paths and the manifest hash explicitly; keep keys out of YAML and command arguments.

Dataset discovery uses literal paths. For custom task locations, save a run overlay
such as `/absolute/path/to/run.yaml` (keep this task path equal to the `tasks`
setting below):

```yaml
config_paths:
- benchmarks/skill_trigger/configs/pi/config.yaml
harbor_agent_general:
  responses_api_agents:
    harbor_agent_general:
      datasets:
      - name: skill_trigger
        type: benchmark
        jsonl_fpath: /absolute/path/to/tasks/gym.jsonl
        prepare_script: benchmarks/skill_trigger/package.py
        num_repeats: 1
```

```sh
gym eval run \
  --config /absolute/path/to/run.yaml \
  --agent harbor_agent_general --model openai/MODEL_ID \
  --split benchmark \
  --output /absolute/path/to/new-run/rollouts.jsonl --num-repeats 3 \
  ++skip_venv_if_present=true \
  ++policy_base_url=https://api.openai.com/v1 \
  ++skill_trigger_settings.release=/absolute/path/to/release \
  ++skill_trigger_settings.manifest_sha256=MANIFEST_SHA256 \
  ++skill_trigger_settings.tasks=/absolute/path/to/tasks \
  ++skill_trigger_settings.jobs=/absolute/path/to/new-run/harbor
```

Change the profile in the overlay’s `config_paths` for another harness. Codex guidance additionally requires `skill_trigger_settings.model_catalog_path` and its SHA-256: use a catalog captured from the same Codex version and baseline model. The extension changes only the selected model's `include_skills_usage_instructions` value. A native model may already enable it; inspect the effective baseline before interpreting this as a treatment contrast.

The `skip_venv_if_present` setting retains the explicitly prepared Harbor 0.25.0 component environment; without it, Gym reinstalls the component lock, which currently selects Harbor 0.24.0 and cannot import these adapters. Verify the component environment version before running.

Gym and Harbor own scheduling, failure collection, and cleanup. This contribution does not change their worker-retry policy. Preserve the resolved configuration, runtime versions, task manifests, trajectories, and failure records. Do not infer complete coverage from process exit alone.

## Development

```sh
python -m pytest --confcutdir=benchmarks/skill_trigger/tests benchmarks/skill_trigger/tests
```

Tests cover source integrity, hidden-label exclusion, packaging, score success/miss/error cases, all five resolved Gym configurations, and the two guidance overrides. Before release, validate representative real-model runs for each profile and inspect their trajectories. Keep incomplete validation as a draft. An experimental release must identify its cleared evaluated subset and inspected model evidence; the nine blocked cases remain unavailable for packaging.


## Revision 0.2.1 and negative controls

Revision 0.2.1 corrects missing-content and signed-exit scoring, task IDs, notices,
and two contradictory positive prompts. Earlier frozen releases and results remain
separate; their observations must not be relabeled as results for revised prompts.
The canonical selection still has 110 positive cases and nine source-blocked cases.

A separate six-case control release tests whether the agent opens any installed
skill for self-contained arithmetic or repetition prompts. It uses two prompts per
collection and the full catalog from that collection's lexicographically first
cleared case. Review these catalogs before execution. These simple controls measure
unnecessary opening on these prompts, not general skill-selection specificity.

Create it from an already cleared release, then package its new manifest using the
same package command above:

```sh
python -m benchmarks.skill_trigger.negative_controls \
  --release /absolute/path/to/cleared-release \
  --manifest-sha256 MANIFEST_SHA256 \
  --output /absolute/path/to/new-negative-release
```

A negative control earns reward 1 for no unnecessary verified skill openings and 0
otherwise. Addy allows its documented session-start meta-skills, `using-agent-skills`
and `context-engineering`, when present in the pinned catalog. The control manifest
records these exemptions. Distinct opened skills are reported as all, session-start,
and unnecessary reads; exemptions never hide reads. Missing evidence or unresolved delegation remains unscored.
Report its **unnecessary-opening rate** (1 minus mean reward) separately from the
positive target-opening rate. Bulk-reading every skill fails the negative controls;
useful multi-skill behavior on positive cases is not penalized.

Use three attempts per control and all five profiles with the same identified model.
Preserve every planned attempt, including failed and pending attempts, in an attempt
ledger matching `report.Attempt`. The summary separates models, runtimes, profiles,
collections, release hashes, and control kinds:

```sh
python -m benchmarks.skill_trigger.report --attempts /absolute/path/to/attempts.jsonl
```

Each summary includes the numerator, scored denominator, unscored count, and pending
count. No scored attempts means an unavailable rate, never zero. This ledger summary
supplements Gym's native `gym eval profile`; it does not infer absent attempts from
successful rollouts. Never pool exploratory probes with benchmark baselines.

## Release 0.2.1 validation

The November 11, 2026 (UTC) pilot used `nvidia/nvidia/nemotron-3.5-super-vl` at `https://inference-api.nvidia.com/v1`, Python 3.13.14, and Harbor 0.25.0 with the five pinned agents above. All five profiles have inspected read=1/no-tools=0 Gym CLI fixtures. Two separately labeled Codex follow-ups closed initial rate-limit failures; those earlier failures remain unscored.

### Frozen positive sample

From the unchanged 110-case selection, exclude catalogs intersecting recorded source blockers: 101 cases remain eligible (35 BioNeMo, 41 SkillsBench, 25 Addy). Before outcomes, rank each collection by ascending `SHA256(seed + NUL + suite + NUL + case_id)` and take four cases. The seed is `gym-pr5031-positive-baseline-v021-20261110`. No outcome-driven replacement occurred. The remaining 89 eligible cases were not evaluated in this pilot, and nine source-blocked cases remain excluded.

Canonical selection SHA-256 at study freeze: `2c78cf0fb5b5a38011f4ab0345288a4084b80df212978b1cf61d18644ab3ee64`. Positive release manifest SHA-256: `24040e3161f2a2b8973a8400bf87ef3c1b93bc3302555632810dd4fbbb66aaed`. The evaluated derivative retains only selected, blocker-free catalogs and preserves source notices. The canonical case membership and its source guard are unchanged. The two corrected prompt cases are outside this 12-case sample; all sampled prompts/catalogs are unchanged. No model results for those rewritten prompts are claimed. The current full selection SHA-256 is `c9b2a15108ebfbbce1d130e5c90bd6e2064e0a78fcf313042c8b44955900f6f0`.

| Collection | Stable task ID | Intended skill |
|---|---|---|
| addy | `case-e77693cc5b01` | `using-agent-skills` |
| addy | `case-cf967863230e` | `planning-and-task-breakdown` |
| addy | `case-9dff18a1894a` | `deprecation-and-migration` |
| addy | `case-8189d9d117c3` | `spec-driven-development` |
| bionemo | `case-23c2cefe07ab` | `complexa-evaluate-pdbs` |
| bionemo | `case-45de3de16826` | `parabricks` |
| bionemo | `case-322349ba33ee` | `codonfm-setup` |
| bionemo | `case-864fc41fbb4a` | `kermt-pretrain-scratch` |
| skillsbench | `case-6a0cf37ce517` | `testing-python` |
| skillsbench | `case-3125f9458362` | `radar-vital-signs` |
| skillsbench | `case-f5b43c7d29b3` | `pymatgen` |
| skillsbench | `case-332af49dfeda` | `light-curve-preprocessing` |

Upstream source revisions are BioNeMo toolkit `5fe8b184cda54c48e73214824df6ee04f040b88d`, SkillsBench `f860ecf02ef53412a030f8661c83d644b739ff01`, and Addy agent-skills `aeac41d4d005999432bb7312f7ff06383a91d59d`, as recorded in the canonical selection. Rollouts used source `a271411d39072d2386ed75fe9b302d4bf61208d8`; subsequent main integration leaves benchmark code unchanged.

### Positive results

All 180 planned attempts finished once: 12 cases × three attempts × five profiles. There are 133 scored observations and 47 unscored outcomes (38 rate limits and nine timeouts without conclusive opening evidence), and zero pending attempts. Each fraction below is the mean binary target-opening reward on scored attempts; errors without conclusive opening evidence are not counted as misses.

| Profile | Collection | Target opened / scored | Unscored | Pending | Planned |
|---|---|---:|---:|---:|---:|
| codex | addy | 0/10 | 2 | 0 | 12 |
| codex | bionemo | 8/11 | 1 | 0 | 12 |
| codex | skillsbench | 1/5 | 7 | 0 | 12 |
| codex_guidance | addy | 0/12 | 0 | 0 | 12 |
| codex_guidance | bionemo | 7/10 | 2 | 0 | 12 |
| codex_guidance | skillsbench | 2/3 | 9 | 0 | 12 |
| hermes | addy | 3/11 | 1 | 0 | 12 |
| hermes | bionemo | 8/10 | 2 | 0 | 12 |
| hermes | skillsbench | 0/5 | 7 | 0 | 12 |
| hermes_guidance | addy | 0/9 | 3 | 0 | 12 |
| hermes_guidance | bionemo | 5/8 | 4 | 0 | 12 |
| hermes_guidance | skillsbench | 2/3 | 9 | 0 | 12 |
| pi | addy | 5/12 | 0 | 0 | 12 |
| pi | bionemo | 12/12 | 0 | 0 | 12 |
| pi | skillsbench | 5/12 | 0 | 0 | 12 |

All 175 available saved trajectories agree under reviewed and revised scoring, including every scored observation. Synthetic regressions demonstrate the corrected missing-content and signed-exit defects; their frequency in the deleted historical experiment is unknown. Six verified main-session openings count despite later errors: one Codex timeout, four Codex-guidance rate errors, and one Hermes-guidance rate error. All 53 execution errors remain recorded. There were ten execution timeouts in total. The prior execution-clean and partial opening-only accounting remain history. The old runner deleted 43 standalone rate-error results; their retained trajectories support these comparisons, and unavailable original files are not reconstructed. The generic Harbor agent now preserves exact failed-result bytes privately before invalidating resume. All frozen input bytes and modes and all 180 effective 120-second agent limits were verified. Agent setup and provider waits can extend total request duration beyond the agent execution window.

These rates are conditional on scorable attempts. Failures and censoring are not assumed random; profiles ran sequentially in fixed case order, so endpoint availability can confound comparisons. The balanced 12-case sample is not population weighted, and its three attempts per task are repeated observations. It establishes neither harness rankings, full-suite performance, task completion, historical reproduction, nor statistical confidence. The intended label is narrow: in three Codex-guidance misses for Addy `using-agent-skills`, the agent successfully read the relevant alternative `debugging-and-error-recovery`. Useful alternative-skill behavior is not task failure.

### Separate negative results

The six controls ran three times per profile: 90 terminal attempts, 82 scored, eight unscored rate-limit failures, and zero pending. All 82 scored attempts recorded zero unnecessary, any, and session-start openings. The eight error trajectories were also inspected; no conclusive unnecessary opening was found, and those attempts remain unscored rather than successful avoidance. Negative release SHA-256: `afc6a528c44da2f56e7a1e81270e5135bdd699b638664a32c8d0f67ea26d8dcc`.

| Profile | Collection | Unnecessary / scored | Any opening / scored | Session-start / scored | Unscored | Pending | Planned |
|---|---|---:|---:|---:|---:|---:|---:|
| codex | addy | 0/6 | 0/6 | 0/6 | 0 | 0 | 6 |
| codex | bionemo | 0/4 | 0/4 | 0/4 | 2 | 0 | 6 |
| codex | skillsbench | 0/6 | 0/6 | 0/6 | 0 | 0 | 6 |
| codex_guidance | addy | 0/4 | 0/4 | 0/4 | 2 | 0 | 6 |
| codex_guidance | bionemo | 0/3 | 0/3 | 0/3 | 3 | 0 | 6 |
| codex_guidance | skillsbench | 0/6 | 0/6 | 0/6 | 0 | 0 | 6 |
| hermes | addy | 0/6 | 0/6 | 0/6 | 0 | 0 | 6 |
| hermes | bionemo | 0/6 | 0/6 | 0/6 | 0 | 0 | 6 |
| hermes | skillsbench | 0/6 | 0/6 | 0/6 | 0 | 0 | 6 |
| hermes_guidance | addy | 0/6 | 0/6 | 0/6 | 0 | 0 | 6 |
| hermes_guidance | bionemo | 0/5 | 0/5 | 0/5 | 1 | 0 | 6 |
| hermes_guidance | skillsbench | 0/6 | 0/6 | 0/6 | 0 | 0 | 6 |
| pi | addy | 0/6 | 0/6 | 0/6 | 0 | 0 | 6 |
| pi | bionemo | 0/6 | 0/6 | 0/6 | 0 | 0 | 6 |
| pi | skillsbench | 0/6 | 0/6 | 0/6 | 0 | 0 | 6 |

These arithmetic/repetition controls do not estimate general skill-selection specificity or positive performance. The earlier 18-attempt Pi/Harbor 0.24 cohort is separate and excluded here. Full catalog-description readback was recorded after the initial Pi run and during Hermes; later profiles followed that review. This timing deviation is disclosed rather than described as a completed pre-execution gate. Addy session-start exemptions remain explicit, and the bulk-read regression still fails indiscriminate reading.
