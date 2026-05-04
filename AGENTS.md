# NeMo Gym Agent Constitution

## North Star

NeMo Gym exists to prove that **the same environment that scores a model can produce its training rollouts**. The library separates an agentic environment into four independently-varied components — dataset, agent harness, model, and reward verifier — and runs them as composable HTTP services that scale to thousands of concurrent rollouts. This is the bridge competing eval-only frameworks don't have. Protect that bridge.

## Non-Negotiables

- Use NeMo Gym's aiohttp client (`nemo_gym.server_utils.request()`), not httpx. httpx/httpcore has O(n²) connection pooling that hangs at 16k+ concurrency. See `fern/versions/latest/pages/reference/design-docs/aiohttp-vs-httpx.mdx`.
- Pass all configuration through Hydra YAML, not environment variables. Sensitive values go in `env.yaml`.
- All commits require DCO sign-off (`-s`) and cryptographic signature (`-S`).
- Tests must pass on Python 3.12+. Async-first; subprocess output decoded with `errors="replace"`.
- Servers must handle errors gracefully and survive 4k–65k concurrent requests without crashing.
- Test coverage ≥ 96% on changed code (gate is `[tool.coverage.report] fail_under = 96.0` in `pyproject.toml`).
- Line length 119; ruff for linting and formatting; double quotes; isort.

## Architecture Boundaries

Three FastAPI server types plus a dataset, all communicating over async HTTP:

- **Dataset** (4th component) — JSONL with one task per line. `responses_create_params.input` is OpenAI message format; `verifier_metadata` is opaque pass-through to the verifier.
- **Agent server** (the agent harness) — orchestrator. Implements `responses()` and `run()`. Lives in `responses_api_agents/`.
- **Model server** — stateless LLM inference. Implements `chat_completions()` and `responses()`. Lives in `responses_api_models/`.
- **Resources server** — verifier (`verify()`), tools, environment state. Lives in `resources_servers/`.

Class hierarchy:
```
BaseServer → SimpleServer → {SimpleResourcesServer, SimpleResponsesAPIModel, SimpleResponsesAPIAgent}
```

Owned by `nemo_gym/`. Public Python API for extension authors. Protocol changes require Steward Notes.

## Stakes

When this repo regresses:

- **Environment authors** lose the contract that makes `verify()` predictable; baselines drift; benchmarks become untrustworthy.
- **Agent harness authors** can't propagate cookies/token-IDs through multi-turn rollouts; RL training breaks silently.
- **Model server authors** ship endpoint-shape mismatches (Responses vs chat completions) that crash mid-rollout at scale.
- **NeMo RL** (downstream trainer) consumes broken rollouts and trains on garbage rewards.
- **Operators** hit cluster-scale failure modes (Ray socket-path overruns, sandbox network leaks, vLLM context-length silent failures) that the docs should have warned about.
- **AI agents reading the docs** can't self-serve when the controlled-vocab frontmatter or `llms.txt` drift from the IA.

## Stop And Ask

Surface human review for any of:

- Public Python API or CLI changes (ng_*, BaseServer subclass signatures).
- New runtime dependencies. Use `nemo-gym[dev]` for dev-only.
- Changes to `verify()` reward semantics, retry behavior, or aggregation rules.
- Hydra config schema changes that could break `+key=value` overrides.
- Changes to inter-server HTTP contract (cookies, session state, JSONL schema).
- Dataset format changes that affect train/validation/example splits.
- Pre-commit hook changes (auto-modifying hooks).
- Release workflow, version-skew, or 0.x → 0.x+1 deprecations.
- Test/code disagreement.
- Unreproduced bugs.

## Anti-Patterns

- Importing httpx, anthropic, or LiteLLM directly. Use `nemo_gym/openai_utils.py` (pinned to `openai<=2.7.2`).
- `os.environ.update(...)` from Python expecting it to propagate into `ng_test`'s isolated venvs. Set env vars in the parent shell.
- `ray.get(...)` inside an async function. Use `await future` (Ray futures are awaitable).
- Mocking the database/server stack in tests. Use real subprocesses + real HTTP.
- Committing `train.jsonl` or `validation.jsonl`. Only `example.jsonl` (5 entries) is git-tracked.
- Skip-ahead Tips ("Already X? Skip to Y") in fast-path docs. Returning users skim.
- Pedagogy openers in tutorials ("Goal:", "In this tutorial you will…").
- Sharing a single `aiohttp.ClientSession` across tests without proper teardown.

## Steward System

Agents read the **root `AGENTS.md`** plus the **closest scoped `AGENTS.md`** for the area they're touching. Root is constitution and routing guide; scoped files are domain stewards.

Scoped stewards own local invariants, refusal patterns, docs, tests, examples, fixtures, and checks. Cross-boundary work needs **Steward Notes** in the PR description listing affected stewards and their accepted findings.

Every steward uses this operating model:

- **Point of View**: who or what the domain represents
- **Protect**: invariants, contracts, quality bars, failure modes
- **Contract Checklist**: concrete surfaces to inspect when this domain changes
- **Advocate**: features and investments the domain should push for
- **Serve Peers**: upstream/downstream domains that need clearer contracts, diagnostics, docs, tests, or ergonomics
- **Do Not**: local anti-patterns
- **Own**: tests, docs, examples, fixtures, maintenance checks

## Stewards

| Domain | Path | Owns |
|---|---|---|
| Core library | `nemo_gym/AGENTS.md` | BaseServer hierarchy, CLI, openai_utils, server_utils, async patterns |
| Resources servers | `resources_servers/AGENTS.md` | verify() contract, environment quality bar, verified flag, baseline variance |
| Agent harnesses | `responses_api_agents/AGENTS.md` | Agent harness protocol, cookies/token-IDs propagation, multi-turn rollouts |
| Model servers | `responses_api_models/AGENTS.md` | Provider compatibility, Responses vs chat completions, retry/concurrency |
| Tests | `tests/AGENTS.md` | Coverage, ng_test venv isolation, async test patterns |
| Docs site | `fern/AGENTS.md` | Fern IA, controlled-vocab frontmatter, voice rubric, llms.txt, redirects |

## Contract Checklist

For cross-surface changes, every accepted finding names required proof and collateral updates, or explicitly says `no collateral: <reason>`:

- CLI / programmatic API
- Inter-server HTTP protocol (request bodies, response shapes, cookies)
- JSONL schema (input + output)
- Hydra config schema
- Tests (unit + functional)
- Docs (reference + concept + how-to + tutorial pages)
- Examples (`data/example.jsonl`, `configs/`)
- Scaffold templates (`ng_init_resources_server`)
- Benchmarks / verified-flag baselines
- llms.txt / changelog / migration notes

Docs and examples move in the same PR as user-facing behavior unless synthesis records why they're unaffected. Contract-affecting PRs include a parity matrix when behavior spans multiple entrypoints.

## Steward Signal Format

Steward findings should be contract-oriented, evidence-backed, and collateral-aware:

```
- Steward:
- Area:
- Severity: P0/P1/P2/P3
- Invariant:
- Evidence:
- User Impact:
- Required Fix:
- Required Proof:
- Collateral:
- Confidence:
```

### Convergence

When two or more stewards independently flag the same finding, that finding is **automatically P0** regardless of either steward's individual severity rating. Cross-steward convergence is a confidence multiplier the synthesis step privileges. Note convergence explicitly in the synthesis ("X stewards independently flagged Y") so future audits can see the pattern.

When the same shape of finding recurs across multiple Content Audit passes — fabricated CLI flags, fabricated config fields, miscounted entities, stale version pins — escalate to a Known Regression Pattern (see § Known Regression Patterns) and treat the pattern as a class to be hunted for proactively, not waited on.

## Steward Swarms

Stewards spawn as independent agents, each reading root + its closest scoped `AGENTS.md`, each advocating only for its domain, each returning findings in the Steward Signal Format above. The implementing agent owns synthesis and final decisions; stewards advise and create useful tension but do not own integrated implementation. Keep PR scope bounded to accepted findings and their proof/collateral; defer unrelated steward suggestions to not-now/follow-up.

Triggers: `ask stewards`, `bugbash`, `review swarm`, `steward synthesis`, `audit docs`, `content audit`, `accuracy pass`.

Stewards run in two modes. Pick based on the trigger and the shape of the change.

### Implementation Review (default)

Triggered by code changes, contract-affecting PRs, or `ask stewards` during a feature/refactor. Stewards defend their invariants against the change. Findings cite the diff, the test, or the missing collateral.

Severity meanings: P0 = breaks an invariant or shipped contract; P1 = degrades it without breaking; P2/P3 = polish, advocacy, follow-up.

### Content Audit

Triggered by doc-shaped PRs (IA refactors, release notes, large content updates, README sweeps), or the explicit phrases `audit docs`, `content audit`, `accuracy pass`. Stewards verify that claims about their domain match the code: command surfaces, output schemas, named entities, version pins, counts, link targets, voice rubric. Findings cite a `<source-file:line>` ↔ `<doc-file:line>` divergence and the corrected text.

Severity meanings shift in this mode:

- **P0** — claim is factually wrong; would break a copy-pasted invocation, name a non-existent entity, or assert a contradicted invariant.
- **P1** — claim is incomplete, stale, or misleading but not actively wrong.
- **P2/P3** — voice rubric, cosmetic, advocacy.

Doc-shaped PRs that don't run a content audit before merge are a known regression source. Make it part of the merge gate, not a follow-up.

#### Doc Autopilot — keeping docs accurate via stewards

The Content Audit mode is the primary mechanism for keeping NeMo Gym docs accurate over time. Stewards are the autopilot — each one already knows what doc pages it owns, what source files are the source of truth, and what shape of claim is stale-by-default. The implementing agent's job is to invoke the swarm at the right cadence and synthesize.

Three triggers shape the cadence:

1. **Merge gate.** Doc-shaped PRs gate on a Content Audit having run. P0/P1 findings actioned or explicitly deferred before merge. Already in the Done Criteria.
2. **Periodic re-audit.** Independent of any PR — spawn a full Content Audit swarm at every release boundary and on a recurring cadence (every 4–6 weeks is a reasonable default). Drift accumulates between PRs even when no doc-shaped PR has landed; periodic re-audit catches it.
3. **Source-triggered targeted re-audit.** When source files change in ways a steward owns, the relevant scoped steward's Content Audit on related doc pages should run. E.g. a change to `nemo_gym/cli.py` triggers a targeted re-audit of `api-reference/cli/*.mdx`; a change to `responses_api_models/*/app.py` triggers `reference/compatibility/providers.mdx` and `reference/servers/model/*.mdx`. Surface this trigger when reviewing a code change that touches public surface contracts.

Each scoped `AGENTS.md` lists the doc pages that steward owns under "Own" — that's the audit surface for that steward in autopilot mode.

### Backlog / roadmap / prioritization

Consult all scoped stewards and produce raw steward signals, confidence, dependencies, risks, convergence, minority reports, ranked backlog, and not-now items.

## Known Regression Patterns

Patterns the steward system has surfaced repeatedly. Stewards in autopilot mode hunt these by default — each pattern names the verification the audit must do, not just the symptom.

- **Fabricated CLI / config fields.** Invented parameter names, output field names, or YAML config fields that don't exist in source. Confirmed cases: `+pass_threshold` and `+output_jsonl_fpath` on `ng_reward_profile` (audit 1); `start_idx` / `end_idx` on `DatasetConfig` (audit 3, two stewards converged independently). **Audit verification:** every CLI flag and every YAML config field documented anywhere must trace to a Pydantic class field or `argparse` declaration. If the audit finds a doc claim that names a flag or field, the next step is `grep` against the relevant source file.
- **Doc-snippet rot.** `ng_*` CLI invocations drift from real signatures (param renames, dropped flags, type changes). Confirmed cases: `ng_collect_rollouts +agent_name` documented as required but is `Optional[str]`; `ng_e2e_collect_rollouts +split` required field omitted from the param table. **Audit verification:** every `bash`-tagged code block in a doc page must round-trip against the actual command's signature.
- **Naming and counting drift.** Any doc claim with a count ("four model servers"), a name (`simple_agent` framed as "single-turn"), or a version pin (`openai<=2.6.1`) is stale-by-default. Confirmed cases: 4 → 6 model servers; openai pin 2.6.1 → 2.7.2; `simple_agent` framed as single-turn when it's a multi-step loop. **Audit verification:** any doc page that names a count, names an entity, or pins a version must be re-verified against current source on every audit pass.
- **Wrong endpoint or contract shapes.** Docs claim an endpoint shape, retry policy, or interaction pattern that doesn't match the implementation. Confirmed cases: `azure_openai_model` documented as `/v1/responses` but it actually calls `chat.completions.create()` and synthesizes Responses-shaped output; "exponential backoff" documented but actual is fixed 0.5s. **Audit verification:** for any claim about *how a server works*, read the `app.py` of that server type and trace one full request path before accepting the claim.
- **Pedagogy creep.** Voice rubric violations (Success Check ceremony, "In this guide you will…", skip-ahead Tips) accumulate when the rubric is enforced by review, not by lint. **Audit verification:** the docs steward greps for the known violation patterns on every audit pass and reports residue.
- **llms.txt drift.** When section structure or page slugs change, `llms.txt` rots silently. **Audit verification:** every URL in `llms.txt` of the form `https://docs.nvidia.com/nemo/gym/latest/<path>` must resolve to an `.mdx` file under `fern/versions/latest/pages/<path>` or `<path>/index.mdx`.
- **Cross-page inconsistency.** Same fact stated differently on different pages — most often in default values, command parameters, or counts. Confirmed case: `+data_source` default documented as `huggingface` on one page and `gitlab` on another (the latter was the regression). **Audit verification:** the cross-steward synthesis step explicitly checks for two pages disagreeing on the same fact.

## Steward Feedback Loop

- **Steward miss** — when a bug escapes an applicable steward, update the checklist, add a regression test, add a docs/snippet check, update the routing rule, or record why the miss should not become policy.
- **Steward overreach** — when a steward repeatedly pulls unrelated work into PRs, narrow the checklist, split the steward, or move the concern to follow-up.
- Repeated high-quality findings should become checklist items.
- Repeated noisy findings should be pruned or clarified.
- Steward guidance evolves from evidence: escaped bugs, late collateral updates, CI/review misses, recurring review comments.

## When To Consult

- Proactively consult stewards for cross-boundary, public-facing, hard-to-reverse, performance-sensitive, concurrency-sensitive, security-sensitive, or contract-affecting work.
- Use the nearest steward for local work.
- Use multiple stewards when ownership lines cross.
- Parallelize steward consultation only when questions are independent.
- Final synthesis and implementation accountability stays with the implementing agent.

## Ask Stewards

Trigger phrases: `ask stewards`, `bugbash`, `review swarm`, `steward synthesis` (Implementation Review); `audit docs`, `content audit`, `accuracy pass` (Content Audit).

For implementation work: consult affected stewards and return synthesis before or during the change. Include accepted/deferred findings, merged duplicates, minority reports, required proof, collateral updates, and not-now items.

For content audit: spawn stewards across all domains the docs touch (default: all six on a site-wide IA refactor); synthesize into a triaged P0/P1/P2 punch list cited at file:line; gate the merge on P0/P1. Always run the swarm in parallel — independent stewards surface convergence (see § Steward Signal Format → Convergence), and the convergence pattern itself is a P0 signal.

For multi-surface work, include a parity matrix:

| Contract | API/CLI | Programmatic | Protocol | Schema/Types | Docs | Examples | Tests |
|---|---|---|---|---|---|---|---|

## Extension Routing

- New environment → `resources_servers/<name>/` + tests + `data/example.jsonl` + config in `configs/`. See `resources_servers/AGENTS.md`.
- New agent harness → `responses_api_agents/<name>/` + tests + config. See `responses_api_agents/AGENTS.md`.
- New model server → `responses_api_models/<name>/` + tests + config. See `responses_api_models/AGENTS.md`.
- New CLI command → `nemo_gym/cli.py` + entry point + tests + docs. See `nemo_gym/AGENTS.md`.
- New docs page → `fern/versions/latest/pages/<section>/` + frontmatter + redirect if moving an existing URL. See `fern/AGENTS.md`.

## Done Criteria

- `pytest` passes; `ruff check --fix . && ruff format .` clean; `pre-commit run --all-files` clean.
- Docs / changelog / migration notes updated where contracts changed.
- Examples / scaffold / templates updated where relevant.
- Performance / concurrency / security notes where relevant.
- Every accepted steward finding has test/docs/example/benchmark proof or an explicit no-impact note.
- For docs: `cd fern && npm run check` clean (60 baseline pre-existing v0.2 false-positives are OK).
- For docs: `python3 scripts/validate_docs_frontmatter.py` clean.
- For doc-shaped PRs (IA refactors, release notes, large content updates, README sweeps): a Content Audit steward swarm has run; P0 and P1 findings have been actioned or explicitly deferred with a recorded reason.

## Review Notes

Flag in PR description:

- Weird tests (skipif, monkeypatching, ad-hoc fixtures)
- Unused public names
- `# noqa` / `# type: ignore` suppressions and the reason
- Dead code
- Benchmark gaps (no reward profile baseline)
- Free-threading assumptions
- Steward disagreement
- Deferred / not-now findings

## Companion Files

- `CLAUDE.md` — Claude Code-specific instructions (commands, common workflows). Stays alongside this file; AGENTS.md is canonical for governance, CLAUDE.md for tactical Claude Code use.
- `CONTRIBUTING.md` — public contribution guide. Cross-references this file for invariants.
- `README.md` — user-facing intro. Don't put governance there.
