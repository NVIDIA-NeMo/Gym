# Skill opening across agent harnesses

**Implementation draft: this benchmark is not runnable yet.** Source preparation and its tests are implemented. Harness adapters, scoring, and live migration checks remain pending. This draft does not register a benchmark or report new model results.

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

## Scoring

A valid trial earns 1 when the intended skill body is returned by a successful tool operation within the run budget, corroborated by independent file-access evidence. A skill name, front matter, failed read, or file access without returned instructions is insufficient. Match at least 120 normalized characters of body text, or the entire body when shorter, excluding ambiguous shared boilerplate.

Same-model delegated reads may count when session lineage, returned content, and shared-budget evidence establish the read. Inherited parent output alone does not count. Keep canonical native traces alongside any converted representation.

Valid misses earn 0. Setup, provider, observation, and incomplete-evidence errors are unscored errors, not model failures. A successful opening before a deadline remains a success even if subsequent work times out. Record planned, valid, failed-budget, and infrastructure-error counts alongside scores. Average repetitions within skills, then skills; report each collection and harness separately. The headline is the percentage of valid trials opening the intended skill, with coverage disclosed, not pass@3.

The initial comparison uses three forwarded model requests including retries, 120 seconds from prompt release, and 8,192 output tokens per response. Setup precedes the clock. Record reasoning settings and provider adapters; equal requested settings do not imply equal compute. No outcome-dependent retries are permitted.

## Gym integration

Reuse [harbor_agent_general](../../responses_api_agents/harbor_agent_general/README.md) with its native `job` configuration. Gym schedules one task and one agent per input row and owns repetitions/concurrency. Use separate named configurations with shared preparation and scoring. Keep the existing native harnesses rather than replacing them with a common agent loop: harness behavior is part of the subject being measured.

The implemented benchmark should use the `external-agent-loop` integration profile, `prompt_source: agent`, and `experimental: true`. Harbor owns the task execution and verifier. A null resources server means the fixture-based `gym env test NAME` and `gym env publish NAME` commands do not apply; use focused adapter/verifier tests and representative real rollouts as required by the [contribution guide](../../fern/versions/latest/pages/contribute/environments/adding-a-benchmark.mdx).

Follow Gym's [run-level skills conventions](../../fern/versions/latest/pages/agent-server/agent-skills.mdx) for catalog identity and provenance. Execute catalog-homogeneous groups where necessary; verify that the catalog actually staged in the Harbor sandbox matches its recorded identity. Do not assume the bridge stages skills solely because request metadata contains a path.

Three compatibility gates precede live scoring:

- Every adapter must emit schema-valid ATIF from actual events. The bridge requires this even when Harbor can execute an agent without ATIF. Preserve native artifacts and disclose lossy conversion, especially for delegated sessions.
- Budget exhaustion must terminate active work before grading and remain a scored model outcome when evidence is complete. Infrastructure failures remain distinct. The bridge currently removes a failed trial result before retries; preserve attempt evidence and disable automatic retries for initial validation.
- The verifier must supply the selected binary reward explicitly. Missing or malformed reward data must fail validation rather than reaching the bridge's fallback-to-zero behavior.

Use fresh sessions, isolated homes/workspaces, per-trial skill-file copies, and observer startup after catalog staging. Keep credentials, targets, and verifier evidence outside the agent's readable environment. Limit the agent's network to the configured inference route. This proposal is evaluation-only; it does not promise training-compatible token metadata or a recurring hosted evaluation service.

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
