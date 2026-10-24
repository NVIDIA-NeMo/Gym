# Skill opening across agent harnesses

**Implementation draft: this benchmark is not runnable yet.** Source preparation, task packaging, scoring, trace conversion, shared budgets, inference transport, and the host-side Harbor verifier have component tests. Codex/Hermes launch profiles are ported. The shared adapter has passed synthetic Pi integration through Harbor and Codex live session-attribution checks; complete Codex/Hermes integration and live migration checks remain pending. Five experimental profile declarations are present; their full dataset validation and live migration remain incomplete. No new model results are reported.

The question is whether a model opens the relevant skill instructions when a task calls for them. Task-performance benchmarks measure whether work gets done; this benchmark measures an earlier step. Opening instructions is not evidence that the agent followed them or completed the task.

## Proposed evaluation

Use one shared collection of positive tasks with three primary configurations: Pi, Codex, and Hermes. Preserve each harness's native skill discovery and tool interface. Additional Codex skill-usage guidance and Hermes tool-use guidance are separate configurations, giving five configurations in total.

The target release contains 110 positive prompts drawn from 35 BioNeMo skills, 50 selected SkillsBench skills, and 25 Addy Osmani skills. Three repetitions produce 330 trials per configuration. This is a trigger-test collection, **not the official SkillsBench task-performance benchmark or its leaderboard score**. Negative-trigger accuracy and task completion are outside the initial release.

Each task retains its prompt, source, intended skill, and catalog identity. Full skill directories are made available; expected targets and evaluation labels are hidden. Catalog membership must remain identical across harnesses for a given task. Catalogs may differ between tasks; replacing them with a union catalog changes the experiment.

Source revisions, prompt adaptations, distribution terms, and exact task counts must be reviewed before runnable onboarding. [selection.json](selection.json) pins each prompt, target version, source reference, and ordered catalog; [pilot.json](pilot.json) freezes nine representative cases. Download upstream content reproducibly with notices intact rather than copying an opaque archive. An unavailable or unlicensed source blocks release; it does not justify silently changing the benchmark.

## Source preparation status

All 110 prompt bytes and target versions have been compared to the reference experiment. All 186 distinct skill-directory hashes, across 43 catalogs, match the reference, including companion skills. Provenance is recorded as 13 upstream trigger cases, 22 adapted completion cases, 50 authored SkillsBench trigger cases, and 25 adaptations of Addy's ranking tests. Wording changes are recorded separately from changes to label semantics.

**Full packaging is blocked by file-level license questions affecting nine cases.** Ten bundled document-skill licenses restrict copying and redistribution. One additional JAX skill declares a proprietary license but omits the referenced license file; a companion PubChem skill declares its license unknown. The affected targets are `rdkit`, `jax-skills`, `xlsx`, `senior-data-engineer`, `pdf`, `pptx`, `file-organizer`, `image-ocr`, and `video-frame-extraction`. Several affected files are companion skills. Removing them changes the benchmark and requires an explicit revision. The Apache-2.0 root license of SkillsBench does not resolve these file-level conflicts.

The exact paths, hashes, and reasons are retained in `selection.json` under `release_blockers`. Audit mode validates all inputs and writes metadata without copying skill catalogs:

```bash
python -m benchmarks.skill_trigger.prepare --fetch --audit-only \
    --output benchmarks/skill_trigger/data/source-audit
python -m pytest --confcutdir=benchmarks/skill_trigger/tests benchmarks/skill_trigger/tests/test_prepare.py
```

Use a new output directory for each preparation. Normal preparation refuses unresolved blockers. Only a `prepared-release` artifact may eventually be packaged for execution; `source-audit` is metadata only. The manifest contains labels and original prompt provenance and must remain outside the agent sandbox. Copies of root and nested attribution notices accompany permitted prepared catalogs. No scientific software or model weights referenced by a skill are downloaded or executed during preparation.

Once release terms are resolved, `python -m environments.skill_trigger.package --release RELEASE --manifest-sha256 SHA256 --output NEW_TASKS --image PINNED_IMAGE --harness pi --gateway-host 127.0.0.1` packages every case for Harbor. Select another of the five harness configurations to preserve its network profile. This validates the manifest and skill files, retains each original catalog, and keeps labels outside task images. The same packager is used by the synthetic integration checks; it refuses audit-only artifacts and existing output directories. Packaging also writes `catalogs/<catalog-id>.jsonl` for separate catalog runs.

## Scoring

A valid trial earns 1 when the intended skill body is returned by a successful tool operation within the run budget, corroborated by independent file-access evidence. A skill name, front matter, failed read, or file access without returned instructions is insufficient. Match at least 120 normalized characters of body text, or the entire body when shorter, excluding ambiguous shared boilerplate.

Same-model delegated reads may count when session lineage, returned content, and shared-budget evidence establish the read. Inherited parent output alone does not count. Keep canonical native traces alongside any converted representation.

Valid misses earn 0. Setup, provider, observation, and incomplete-evidence errors are unscored errors, not model failures. A successful opening before a deadline remains a success even if subsequent work times out. Record planned, valid, failed-budget, and infrastructure-error counts alongside scores. Average repetitions within skills, then skills; report each collection and harness separately. The `skill_opening_rate` headline averages valid repetitions within each skill, then skills equally; multiply by 100 for a percentage. `skills_scored` and `trials_scored` accompany it. Also report the collector's planned and error counts, especially with incomplete coverage. This is not pass@3.

The initial comparison uses three forwarded model requests including retries, 120 seconds from prompt release, and 8,192 output tokens per response. Setup precedes the clock. Record reasoning settings and provider adapters; equal requested settings do not imply equal compute. No outcome-dependent retries are permitted.

The shared implementation is under [`environments/skill_trigger`](../../environments/skill_trigger). It extracts successful model-visible tool results, checks session lineage and observation coverage, and preserves the original excerpt-matching rules, including Pi's UTF-16 character counting. Exact-file reads can disambiguate shared text; shell output must identify a skill by its content. Invalid evidence raises an error rather than producing a zero.

Trajectory conversion reuses Harbor 0.24.0's Pi, Codex, and Hermes converters. The wrapper checks call/result IDs and rejects dropped or invented records. Converted trajectories support Gym interoperability; canonical traces remain the scoring source. The shared request allowance counts retries and child requests together, anchors time to prompt release, and cancels streams at the absolute deadline. Component tests do not establish end-to-end correctness; the developing Hermes adapter still needs a successful integrated run.

The trial-scoped gateway uses Gym's shared HTTP client with transport retries and redirects disabled. It records original and forwarded requests, response bytes, and provider status outside the agent-visible logs. Tests use local HTTP endpoints to check stream preservation, errors, deadlines, and blocked requests. The Codex/Hermes launcher waits for host authorization after profile setup; Codex guidance requires effective model metadata, while Hermes guidance changes only `tool_use_enforcement`. The shared adapter connects these components to Pi, Codex, and Hermes. Codex admission verifies canonical session ownership before forwarding, shares limits across children, and embeds their actual trajectories. Complete Codex/Hermes integration remains unverified.

The custom Harbor verifier reads a pinned prepared release and private host evidence outside Harbor's mounted trial logs. It checks the prompt and complete catalog files, rejects audit-only or blocked releases, and returns an explicit binary `reward`. Missing evidence and provider errors raise; an agent-written `reward.json` cannot substitute for the host record. Each attempt is recorded once and retained independently of Harbor trial cleanup. The adapter must still establish that these directories are not mounted into the sandbox.

The Linux Docker component smoke checks per-trial observation, read-only instructions, observer isolation, and termination of agent processes while preserving the container keepalive. It uses synthetic text and no model. Run it explicitly with an existing Python-containing image:

```bash
python -m environments.skill_trigger.tests.docker_smoke \
    --image YOUR_PINNED_PYTHON_IMAGE --output /tmp/skill-container-smoke.json
```

This is not the clean-clone benchmark smoke or proof of end-to-end harness equivalence.

For the offline tests, install Gym's development dependencies and `harbor==0.24.0`, then run from the checkout root:

```bash
LITELLM_LOCAL_MODEL_COST_MAP=True python -m pytest \
    --confcutdir=environments/skill_trigger/tests environments/skill_trigger/tests
```

## Runtime development

Build the pinned Codex/Hermes container from public sources:

```bash
docker build -t gym-skill-harnesses:local environments/skill_trigger
```

The Dockerfile checks the Codex release checksum and retains the exact Hermes Git revision. Codex, Hermes, and Hermes-guidance startup were exercised in this image with networking disabled and stopped before prompt release. This is profile validation, not a scored rollout; Codex-guidance startup still needs its effective model catalog.

The shared adapter uses Harbor's Docker lifecycle, independent file observation, the inference gateway, canonical traces, ATIF conversion, and the host verifier. Pi's SDK runs on the host with Docker-backed tools and no network in the task container. Hermes runs inside the container and needs an explicitly reachable gateway.

For Pi, install Node 22 or later and the pinned SDK dependencies, then run the synthetic smoke:

```bash
npm ci --ignore-scripts --prefix environments/skill_trigger
LITELLM_LOCAL_MODEL_COST_MAP=True python -m environments.skill_trigger.tests.harbor_smoke \
    --image gym-skill-harnesses:local --harness pi --gateway-host 127.0.0.1 \
    --output /tmp/new-skill-pi-smoke
```

The smoke uses a local scripted provider and synthetic instructions, not model inference or a benchmark case. Pi's successful-read and missed-read paths passed with both `openai-responses` and `openai-completions`; three-request stops preserve earlier successful reads, and scripted provider failures remain unscored. Select the API with `--api` and a boundary case with `--scenario miss`, `request-limit`, or `provider-error`. Each attempt needs a new output directory. This is integration evidence, not the live migration comparison or a clean-clone acceptance run.

Add `--engine gym-job` to exercise Gym's actual Harbor Job execution and response conversion. Synthetic Pi success, miss, request-limit, and provider-error cases passed on this path. A provider failure remains marked `harbor_failed` and keeps private evidence after the bridge removes its failed trial result. Gym's collector excludes marked failures by default; keep `count_failure_classes_as_zero=[]` and `count_missing_rollouts_as_zero=false` for this benchmark. The check calls the bridge in-process and does not exercise Ray dispatch, the HTTP server, or collector persistence. It requires Gym's Ray dependency (tested with the lockfile's Ray 2.58.0); no Ray cluster is started.

Codex selects the root from the CLI's `thread.started` event, verifies child ownership against canonical spawn calls/results, and excludes inherited records using the child's explicit history ordinal plus a matching parent prefix. The watcher rejects incomplete or altered files. Live admission checks passed for fresh and forked children using the actual pinned binary, scripted replies, and network-disabled containers. Each admitted three requests across two proven sessions and produced linked ATIF trajectories. These checks use pipes instead of the HTTP gateway and do not score a benchmark trial. To reproduce, run `python -m environments.skill_trigger.tests.codex_live_smoke --image skill-trigger-harnesses --output /tmp/new-codex-check`, adding `--fork` in a separate fresh directory for inherited history.

For Hermes, use `--harness hermes` or `hermes_guidance` and a Docker-reachable host address. Its integrated test remains blocked by the test host's firewall. An authenticated readiness check detects unreachable gateways before prompt release; `--gateway-port` permits an operator-approved fixed port. The test does not alter firewall rules.

Harbor's task CPU field accepts whole numbers. The frozen half-CPU limit is enforced by Docker Compose with Harbor's CPU override disabled (`cpu_enforcement_policy: ignore`); the adapter checks the actual running container still has 0.5 CPU and 768 MiB. The networking path also needs the migration comparison: the prior experiment documented an earlier header timeout in Harbor's sidecar, separate from the 120-second trial limit. Do not silently change that behavior and label the result a historical reproduction.

## Gym integration

Reuse [harbor_agent_general](../../responses_api_agents/harbor_agent_general/README.md) with its native `job` configuration. Gym schedules one task and one agent per input row and owns repetitions/concurrency. Use separate named configurations with shared preparation and scoring. Keep the existing native harnesses rather than replacing them with a common agent loop: harness behavior is part of the subject being measured.

The implemented benchmark should use the `external-agent-loop` integration profile, `prompt_source: agent`, and `experimental: true`. Harbor owns the task execution and verifier. A null resources server means the fixture-based `gym env test NAME` and `gym env publish NAME` commands do not apply; use focused adapter/verifier tests and representative real rollouts as required by the [contribution guide](../../fern/versions/latest/pages/contribute/environments/adding-a-benchmark.mdx).

Follow Gym's [run-level skills conventions](../../fern/versions/latest/pages/agent-server/agent-skills.mdx) for catalog identity and provenance. Execute catalog-homogeneous groups where necessary; verify that the catalog actually staged in the Harbor sandbox matches its recorded identity. Do not assume the bridge stages skills solely because request metadata contains a path.

Three compatibility gates precede live scoring:

- Every adapter must emit schema-valid ATIF from actual events. The bridge requires this even when Harbor can execute an agent without ATIF. Preserve native artifacts and disclose lossy conversion, especially for delegated sessions.
- Budget exhaustion must terminate active work before grading and remain a scored model outcome when evidence is complete. Infrastructure failures remain distinct. The bridge currently removes a failed trial result before retries; preserve attempt evidence and disable automatic retries for initial validation.
- The verifier must supply the selected binary reward explicitly. Missing or malformed reward data must fail validation rather than reaching the bridge's fallback-to-zero behavior.

Use fresh sessions, isolated homes/workspaces, per-trial skill-file copies, and observer startup after catalog staging. Keep credentials, targets, and verifier evidence outside the agent's readable environment. Limit the agent's network to the configured inference route. This proposal is evaluation-only; it does not promise training-compatible token metadata or a recurring hosted evaluation service.

## Gym profiles

The five profiles are [Pi](configs/pi/config.yaml), [Codex](configs/codex/config.yaml), [Codex with guidance](configs/codex_guidance/config.yaml), [Hermes](configs/hermes/config.yaml), and [Hermes with guidance](configs/hermes_guidance/config.yaml). They share [common settings](common.yaml) and the existing Harbor bridge, with a small [catalog check](../../responses_api_agents/skill_trigger/README.md) before dispatch. Configuration and manifest composition checks pass; full `gym env validate` requires the currently blocked prepared release and has not passed.

The declarations default to the BioNeMo catalog as a starting group, **not the complete 110-case benchmark**. Use the [catalog runner](../../environments/skill_trigger/run.py) to prepare one model/profile across all 43 groups (110 cases, three repetitions). `--pilot` selects the frozen nine cases, five catalogs, and one repetition. Each group gets matching input rows and `skills.path`; the catalog guard rejects another group's tasks before inference. Input rows carry global task indices so combining catalog results cannot merge different skills.

After source terms are resolved, create a runtime YAML containing `provider: openai` and `api: openai-responses` for Pi, or the appropriate gateway and pinned model catalog settings for another profile. Then:

```bash
python -m environments.skill_trigger.run plan \
    --release RELEASE --manifest-sha256 SHA256 --output NEW_RUN \
    --image PINNED_IMAGE --harness pi --model MODEL --endpoint BASE_URL \
    --runtime-settings RUNTIME.yaml --pilot
python -m environments.skill_trigger.run execute --output NEW_RUN
```

The plan checks the complete frozen selection and packages all tasks before selecting input rows. No API key belongs in the runtime YAML. Execute from a clean checkout without `env.yaml`, `NEMO_GYM_CONFIG_DICT`, or `NEMO_GYM_EXTRA_ROOTS`, which could override the plan. Execution calls `gym eval run` sequentially per catalog, with two concurrent trials inside a group and a one-attempt limit. It preserves each command, console log, exit code, and Gym's result/failure files. Reinvoking execution only starts never-attempted groups; failures and interrupted attempts remain visible and require inspection. Config or input changes require a new plan. A successful process exit is not proof of complete scored coverage: inspect Gym's result and failure counts before reporting a score. HTTP/collector integration and complete real-model execution remain unverified.

Set the release manifest hash, model name and endpoint, image, task paths, and Docker-reachable gateway explicitly. Pi additionally requires its original `provider` and `api`; Codex guidance requires the pinned effective model catalog and its hash. `SKILL_MODEL_API_KEY` stays in the host/worker environment and is not part of Harbor's task config. Set `NEMO_GYM_MAX_ROLLOUT_ATTEMPTS=1`; Harbor retries are already disabled. The single-repetition defaults are for migration validation, not a replacement for the historical three repetitions.

`--engine gym-ray` on the synthetic smoke exercises the catalog guard and actual Ray worker, using a disposable local runtime that is shut down afterward. The successful Pi check preserved catalog provenance, returned skill text, and reward. It does not exercise the HTTP server or collector persistence and uses no real model.

`--engine gym-http` additionally starts the actual FastAPI agent and legacy relay on loopback, dispatches through Ray, and uses Gym's collector to persist results. Synthetic Pi success, miss, and provider-error checks distinguish scored 1, scored 0, and an unscored failure. Inspect the result and failure files: a failed bridge response carries a placeholder zero, but the collector excludes it from scored rows. This focused check injects the resolved config; it does not test head-server discovery, Gym's process supervisor, aggregation, or the complete catalog-runner CLI. No real model or frozen benchmark task is used.

## Implementation and acceptance checklist

- [ ] Publish reviewed task selection, pinned sources, attribution, adaptation provenance, and deterministic preparation for all 110 cases.
- [ ] Provide Pi, Codex, and Hermes adapters plus the two separately named guidance configurations, preserving native discovery.
- [ ] Implement trusted read evidence, shared request/time limits, schema-valid ATIF, and explicit error/reward handling.
- [ ] Test correct/wrong/failed reads, mentions without reads, shared boilerplate, setup-only accesses, successful reads before timeout, budget exhaustion, missing evidence/rewards, provider failures, and delegated-session attribution.
- [ ] Verify isolation and active-command termination; replay identical evidence through both scoring paths with identical results.
- [ ] Freeze nine representative cases, three per collection, before observing results. Compare direct Harbor and Gym across all five configurations and two models, one repetition: 90 trials per path. Check prompt/catalog/settings parity and inspect traces; stochastic outcomes need not be identical.
- [ ] Prepare and validate all cases/configurations. A small validated migration does not require repeating every historical model result.
- [ ] Demonstrate a documented clean-clone smoke run, run relevant tests and pre-commit checks, and retain sanitized rollout evidence.
- [ ] Review and simplify the final codebase: one preparation path, one scorer, thin adapters, declarative configurations, focused tests, and short documentation. Keep draft status until these gates pass.

Related work: [Gym skill evaluation infrastructure](https://github.com/NVIDIA-NeMo/Gym/issues/1256), [SkillsBench task-performance integration](https://github.com/NVIDIA-NeMo/Gym/pull/1870), and [native Harbor job configuration](https://github.com/NVIDIA-NeMo/Gym/pull/4620). This proposal complements these efforts without changing existing benchmark behavior.


`--engine gym-cli` runs the same synthetic task through `gym eval run`, including the head server, isolated server environments, Ray, collection, and aggregation. A successful Pi read passed this complete path and retained its catalog and nonzero global task index. A separate provider-error check made one request and persisted one unscored failure, with no scored rows. The benchmark disables Ray worker retries as well as Harbor and collector retries. Its reported `skill_opening_rate` averages repetitions within each skill, then averages the skills; it is not pass@k. This single-catalog check does not establish full catalog-runner execution, real-model migration, or clean-clone readiness.
