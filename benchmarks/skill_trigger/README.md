# Skill opening

When a task calls for a skill, does the model open its instructions? This benchmark scores the instructions returned in Harbor's recorded tool results. It does not measure task completion or the official SkillsBench score.

**Version 0.2 is an experimental, new execution profile.** It uses Harbor's existing agents, skill loading, timeouts, and trajectories. The earlier experiment's results and its 2,092-trial scorer comparison do not validate this version. Fresh model runs are required.

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

A completed trial earns 1 for a matching read and 0 for a miss. Missing, malformed, copied, or incomplete trajectory evidence is an error, not a zero. A root-session miss with delegated trajectories remains unscored until those traces can be verified; this version does not infer child-agent success. Use the mean reward across valid trials as the skill-opening rate, not pass@k. Report failed/unscored counts alongside it, and report the three collections separately.

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

```sh
gym eval run \
  --config benchmarks/skill_trigger/configs/pi/config.yaml \
  --agent harbor_agent_general --model openai/MODEL_ID \
  --input /absolute/path/to/tasks/gym.jsonl \
  --output /absolute/path/to/new-run/rollouts.jsonl --num-repeats 3 \
  ++policy_base_url=https://api.openai.com/v1 \
  ++skill_trigger_settings.release=/absolute/path/to/release \
  ++skill_trigger_settings.manifest_sha256=MANIFEST_SHA256 \
  ++skill_trigger_settings.tasks=/absolute/path/to/tasks \
  ++skill_trigger_settings.jobs=/absolute/path/to/new-run/harbor
```

Change the configuration path for another harness. Codex guidance additionally requires `skill_trigger_settings.model_catalog_path` and its SHA-256: use a catalog captured from the same Codex version and baseline model. The extension changes only the selected model's `include_skills_usage_instructions` value. A native model may already enable it; inspect the effective baseline before interpreting this as a treatment contrast.

Gym and Harbor own scheduling, failure collection, and cleanup. This contribution does not change their worker-retry policy. Preserve the resolved configuration, runtime versions, task manifests, trajectories, and failure records. Do not infer complete coverage from process exit alone.

## Development

```sh
python -m pytest --confcutdir=benchmarks/skill_trigger/tests benchmarks/skill_trigger/tests
```

Tests cover source integrity, hidden-label exclusion, packaging, score success/miss/error cases, all five resolved Gym configurations, and the two guidance overrides. Before release, validate representative real-model runs for each profile and inspect their trajectories. Keep the PR draft until source clearance and model validation are complete.
