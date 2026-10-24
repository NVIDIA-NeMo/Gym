# Skill opening across agent harnesses

When a task calls for a skill, does the model open its instructions? This benchmark measures that step in Pi, Codex, and Hermes. Opening a skill does not establish that the model followed it or completed the task.

**Implementation draft.** Full release is blocked by source-license questions. Synthetic Pi execution passes through Harbor and Gym, including a fresh public clone. Scored Codex/Hermes integration and the real-model migration comparison remain incomplete. No new model results are reported.

## Evaluation contract

The frozen selection contains 110 positive prompts: 35 BioNeMo, 50 selected SkillsBench, and 25 Addy Osmani cases. Three repetitions produce 330 trials per model/configuration. These SkillsBench cases test opening skills; they are **not the official SkillsBench task-performance benchmark or leaderboard score**. Negative-trigger accuracy and task completion are outside this release.

| Configuration | Harness | Guidance setting |
|---|---|---|
| `pi` | Pi 1.3.0 | Default harness configuration |
| `codex` | Codex 0.156.1 | Default harness configuration |
| `codex_guidance` | Codex 0.156.1 | Skill-usage instructions from the effective model catalog |
| `hermes` | Hermes 0.22.2 at the pinned source revision | Default harness configuration |
| `hermes_guidance` | Same Hermes revision | Tool-use enforcement |

[selection.json](selection.json) fixes prompts, source revisions, target versions, adaptation provenance, and 43 ordered catalogs. Each task receives its original catalog; a union catalog would change the experiment. All 186 distinct skill directories match the reference inputs. Provenance includes 13 upstream trigger cases, 22 adapted completion cases, 50 authored SkillsBench trigger cases, and 25 Addy ranking-test adaptations.

The budget is three forwarded model requests, including retries and children, 120 seconds from prompt release, and 8,192 output tokens per response. Setup precedes the clock. Record effective reasoning settings, API adapters, and served checkpoint identity where available. Equal requested reasoning settings do not imply equal compute.

## Setup and source audit

On Linux, install Docker, uv, Python 3.13.14, and Node 22 or later. From the checkout root:

```bash
uv sync --project responses_api_agents/skill_trigger --python 3.13.14 --no-default-groups
source responses_api_agents/skill_trigger/.venv/bin/activate
npm ci --ignore-scripts --prefix environments/skill_trigger
```

Keep `uv`, `node`, and `npm` on `PATH`; Gym creates separate server environments. Record resolved dependency versions with each run. Harness versions are pinned, while Gym dependency ranges can resolve newer packages.

Audit the public upstream sources without packaging them:

```bash
python -m benchmarks.skill_trigger.prepare --fetch --audit-only \
    --output benchmarks/skill_trigger/data/source-audit
```

Use a new output directory each time. Audit mode validates all 110 cases but writes metadata only. It does not download scientific software or model weights mentioned by skills.

**Nine cases have unresolved file-level terms:** `rdkit`, `jax-skills`, `xlsx`, `senior-data-engineer`, `pdf`, `pptx`, `file-organizer`, `image-ocr`, and `video-frame-extraction`. Ten bundled document-skill licenses restrict copying and redistribution; JAX declares proprietary terms but omits its license file; a companion PubChem skill declares its license unknown. Exact paths, hashes, and reasons are in `selection.json` under `release_blockers`. The repository's root license does not resolve these conflicts. Removing companion skills would change the benchmark.

Normal preparation and packaging refuse unresolved terms. Only a permitted `prepared-release` artifact can run; an audit manifest cannot. Preparation retains notices and excludes upstream evaluation labels. The release manifest and expected targets remain outside the agent sandbox.

## Synthetic integration check

Build the pinned runtime and run Pi with a local scripted provider:

```bash
docker build -t gym-skill-harnesses:local environments/skill_trigger
LITELLM_LOCAL_MODEL_COST_MAP=True python -m environments.skill_trigger.tests.harbor_smoke \
    --image gym-skill-harnesses:local --harness pi --gateway-host 127.0.0.1 \
    --output /tmp/new-skill-pi-smoke
```

No inference credential is required. This uses synthetic instructions, not a frozen benchmark task. Use a new output directory for each attempt.

| Option | What it exercises |
|---|---|
| Default | Direct Harbor trial, adapter, observation, and verifier |
| `--engine gym-job` | Gym's Harbor Job bridge and response conversion |
| `--engine gym-ray` | Catalog guard and an actual Ray worker |
| `--engine gym-http` | Agent HTTP endpoints and persisted collection |
| `--engine gym-cli` | Ordinary `gym eval run`, head server, isolated environments, Ray, collection, and aggregation |
| `--scenario miss`, `request-limit`, or `provider-error` | Valid miss, shared request limit, or unscored provider failure |
| `--api openai-completions` | Pi's Chat Completions path instead of Responses |

Pi runs its SDK on the host with Docker-backed tools and a network-disabled task container. Codex and Hermes run inside the container and need an explicitly reachable gateway. Use `--harness hermes` or `hermes_guidance` for their developing smoke path; `--gateway-port` allows an operator-approved fixed port. The test does not change firewall rules. Codex guidance requires a pinned effective model catalog and its hash.

Harbor's CPU field accepts whole numbers; Docker Compose enforces the frozen 0.5-CPU/768-MiB limits with `cpu_enforcement_policy: ignore`. The adapter verifies the running container limits. Preserve the reference sidecar's response-header timeout separately from the trial's 120-second limit; a change needs an explicit migration comparison.

## Running the benchmark after release clearance

The [runner](../../environments/skill_trigger/run.py) groups tasks by catalog and delegates execution to Gym's general Harbor bridge. Five [configurations](configs) share [common.yaml](common.yaml). The agent checks each incoming `skills_ref` against the task's frozen catalog before dispatch. Global task indices distinguish skills across groups.

Create `RUNTIME.yaml` with `provider: openai` and `api: openai-responses` for Pi, or the appropriate gateway and pinned model-catalog settings for another profile. Keep `SKILL_MODEL_API_KEY` in the host/worker environment, never in YAML or task files. Then:

```bash
python -m environments.skill_trigger.run plan \
    --release RELEASE --manifest-sha256 SHA256 --output NEW_RUN \
    --image PINNED_IMAGE --harness pi --model MODEL --endpoint BASE_URL \
    --runtime-settings RUNTIME.yaml --pilot
python -m environments.skill_trigger.run execute --output NEW_RUN
```

The plan requires the full frozen release before selecting the nine-case pilot. Omit `--pilot` for all 110 cases with three repetitions. Execute without `env.yaml`, `NEMO_GYM_CONFIG_DICT`, or `NEMO_GYM_EXTRA_ROOTS`, which could override the plan. Explicitly set the image, endpoint/model, release hash, paths, and gateway. Pi also needs its provider/API settings.

Execution runs catalogs sequentially with two concurrent trials per group. Harbor, Ray worker, and collector retries are disabled. Commands, console logs, exits, scored rows, and failures are retained. Resuming only starts groups never attempted; failed or interrupted attempts require inspection. Changed inputs/configs require a new plan. Process exit alone does not prove complete coverage: inspect saved result and failure counts.

## Scoring and evidence

A valid trial earns 1 only when a successful tool result returns the intended skill body within budget, corroborated by independent file access. Matching requires 120 normalized body characters after front matter, or the entire body if shorter. Shared boilerplate alone does not identify a skill. Pi retains its reference UTF-16 character counting. Names, descriptions, failed reads, and access without returned instructions earn no credit.

Same-model delegated reads count only with proven session lineage, returned content, and a shared budget; inherited text does not count. A successful opening before a deadline remains a success if later work times out. Valid misses earn 0. Provider, setup, observation, and incomplete-evidence errors are unscored.

The [shared implementation](../../environments/skill_trigger) keeps observer logs, expected targets, and verifier state outside agent-visible paths. The host verifier requires explicit binary rewards; an agent-written reward file cannot replace evidence. Harbor 0.24.0 converters produce ATIF, with checks for omitted or invented calls/results. Canonical native and child traces remain the scoring source where conversion is lossy.

`skill_opening_rate` averages valid repetitions within each skill, then skills equally; multiply by 100 for a percentage. Report each collection/configuration, `skills_scored`, `trials_scored`, and planned/error counts. It is not pass@3. Keep `count_failure_classes_as_zero=[]` and `count_missing_rollouts_as_zero=false`. A failed bridge response can contain a placeholder zero; the collector's separate scored and failure files are authoritative.

## Validation and remaining work

Run the preparation and component suites:

```bash
python -m pytest --confcutdir=benchmarks/skill_trigger/tests benchmarks/skill_trigger/tests/test_prepare.py
LITELLM_LOCAL_MODEL_COST_MAP=True python -m pytest \
    --confcutdir=environments/skill_trigger/tests environments/skill_trigger/tests
```

The fresh public clone passed source audit, image construction, 185 component tests, and synthetic Pi success through direct Harbor and the ordinary Gym CLI. It used Python 3.13.14, Harbor 0.24.0, and Ray 2.59.0, without source overlays or copied package environments; Docker could reuse build layers. Earlier synthetic checks covered both Pi APIs, misses, limits, provider failures, and persisted error exclusion. The catalog dispatcher also passed two synthetic catalogs with two repetitions each: separate task/catalog identities, scored successes and misses, and no new requests or changed result files on resumption. This does not exercise frozen-release planning or all 43 catalogs.

Codex/Hermes startup and guidance profiles have offline checks. Actual Codex fresh/forked child-session admission passed with scripted replies; this is not full scored integration. An 18-trial archived pilot matched reference read classification across all three harnesses. Older traces lack evidence needed for the new timing/session admission contract; this does not certify complete historical-trial replay or post-hoc child corrections.

Before release:

1. Resolve all source terms and validate every case/configuration without changing the frozen selection.
2. Finish scored Codex/Hermes integration and both guidance profiles; complete historical evidence comparison and catalog-runner validation.
3. Run [pilot.json](pilot.json)'s nine frozen cases through direct Harbor and Gym, across all five configurations and two models, one repetition: 90 trials per path. Check prompt/catalog/settings parity and inspect errors and representative traces. Stochastic outcomes need not match exactly.
4. Demonstrate clean-clone execution with an authorized real model, run required checks, and finish codebase cleanup. Keep the PR draft until these gates pass.

A validated pilot is sufficient for migration; repeating the entire historical model sweep is not required. The external-agent-loop profile uses `resources_server: null`, so fixture-based `gym env test NAME`/`gym env publish NAME` do not apply. Follow the [benchmark contribution guide](../../fern/versions/latest/pages/contribute/environments/adding-a-benchmark.mdx) and [run-level skills conventions](../../fern/versions/latest/pages/agent-server/agent-skills.mdx).

Related work: [Gym skill infrastructure](https://github.com/NVIDIA-NeMo/Gym/issues/1256), [SkillsBench task performance](https://github.com/NVIDIA-NeMo/Gym/pull/1870), and [Harbor Job integration](https://github.com/NVIDIA-NeMo/Gym/pull/4620).
