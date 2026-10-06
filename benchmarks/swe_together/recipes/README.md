# SWE-Together interactive recipe

Run the canonical 109 tasks through independent Resources, OpenCode, model, and fixed Claude Code judge definitions. The interactive Environment Server owns continuation and cleanup. Resources owns the benchmark prompts, simulated user, task images, and scoring.

The reference profile targets `opencode_gpt` in the upstream [canonical plan](https://github.com/Togetherbench/SWE-Together/blob/891d19eb4b3a64a47c3d49bbd066a311e0133254/canonical_full109.json): OpenCode **1.15.13**, GPT-5.5 through OpenRouter, high reasoning, Gemini 3.1 Pro Preview for simulation and interaction metrics, and Claude Code **2.1.108** with Opus 4.6 for correctness. The pinned native catalog supplies the model limits; OpenCode retains its default 32,000-token per-call output cap and compaction behavior. Task and image pins are included. Keep `verified: false` until reviewed baselines.

High reasoning selects the main agent's variant. Native auxiliary agents retain their upstream defaults: the built-in explore agent inherits the model but omits that variant. An alternate backend must translate explicit reasoning settings while preserving omitted settings; an unconditional model-server override changes these auxiliary calls.

## Prepare and launch

1. Obtain the upstream source at `891d19eb4b3a64a47c3d49bbd066a311e0133254`. Set `SWE_TOGETHER_SOURCE` to the checkout and `SWE_TOGETHER_OUTPUT` to an artifact directory.
2. Prefetch the official Linux amd64 OpenCode 1.15.13 binary and Claude Code 2.1.108 archive. Set `OPENCODE_BINARY` and `CLAUDE_CODE_ARCHIVE`, or configure the checksummed runtime download fields. For restricted egress, prefetch the official ripgrep 15.1.0 Linux x86-64 musl archive and set `OPENCODE_RIPGREP_URL` plus `OPENCODE_RIPGREP_SHA256` (`1c9297be4a084eea7ecaedf93eb03d058d6faae29bbc57ecdaf5063921491599`). The judge supports `CLAUDE_CODE_ARCHIVE_URL` with `CLAUDE_CODE_ARCHIVE_SHA256` to avoid repeated archive uploads. The adapters install and verify their runtimes outside the task repository; task images do not need Node.js for OpenCode. Nine Hyperswitch images also lack Python: configure the separate Python runtime URL and SHA256 variables for Resources, candidate, and judge shown in `reference.yaml` (or equivalent component settings) to a prefetched official Python install-only archive. Use CPython 3.12.8 from Astral build 20250115, Linux x86-64 glibc, SHA-256 `e5435e717c934ed30d4066f64e858497c27f37c1ba547f403b050d9221e50ea4`. Its artifact route must be reachable under each component’s egress policy.
3. Configure four model routes using the variables in `models.yaml`. Each route is independently launchable. Keep real credentials outside candidate sandboxes. Candidate access must expose only its fixed model, with no web-enabled model or non-function tool route.
4. Choose an independent sandbox provider config. Add deployment-specific candidate and judge network policies, including external enforcement evidence. The reference profile fails setup without qualification. Preserve upstream's package allowlist, task-package denials, and direct-IP bypass prevention for a leaderboard comparison.

```bash
python -m benchmarks.swe_together.prepare \
  --source "$SWE_TOGETHER_SOURCE" \
  --images benchmarks/swe_together/recipes/image-pins.json \
  --output benchmarks/swe_together/data/full109.jsonl

gym env start \
  --config benchmarks/swe_together/recipes/reference.yaml \
  --config /path/to/provider-and-network-policy.yaml
```

Once the servers are ready, collect through Gym with the same configuration:

```bash
gym eval run --no-serve \
  --config benchmarks/swe_together/recipes/reference.yaml \
  --config /path/to/provider-and-network-policy.yaml \
  --input benchmarks/swe_together/data/full109.jsonl \
  --output artifacts/swe_together/results.jsonl \
  --num-repeats 2 --concurrency 12
```

For a smoke run, include `--config benchmarks/swe_together/recipes/smoke.yaml` in both commands, before the provider config, and use `--num-repeats 1`. Keep distinct output directories and explicit `_ng_rollout_id` values across separate dispatches so retries retain the original evidence. Set `CANDIDATE_CONCURRENCY` (default 32) at least as high as the combined collector concurrency sent to this Agent Server; waiting for an execution slot consumes the interaction wall budget.

Collection uses the materialized taskset `swe_together:full109`, routed to `interactive_agent` and its `/run` endpoint. Use the same resolved composition for startup and collection. A harness change updates its independent config import, the `swe_together_candidate` reference, and harness settings; benchmark data, simulator, and verifier stay fixed. Other harnesses have not been qualified for continuation.

## Protocol and qualification

The candidate retains one native conversation and private runtime home across activations. Its 4,800-second interaction deadline starts after setup and includes native execution, simulator calls, and waits between activations; each execution is capped at 1,800 seconds. All participating servers must use synchronized clocks for the immutable shared deadline. Resources discards simulator decisions that arrive after that deadline. Resources permits at most 15 resumes and ends after four consecutive no-ops; the first three generate synthetic `continue` inputs. The environment's outer deadline also includes setup, cleanup, grading, and diagnostics. The fixed judge has 50 turns and the upstream task-classified 600/1,200-second deadline.

For health qualification, apply `smoke.yaml` after `reference.yaml`. It uses one repeat, a 600-second candidate budget, and sends empty or tiny submissions to the same agentic judge (`judge_all`), making missing grades visible independently of solve rate. Record any different provider backend, restricted egress, changed budgets, or model transport. A passing smoke establishes lifecycle and grading health; it does not establish leaderboard equivalence.

Preserve source/runtime/image/prompt pins, the secret-free resolved config, every attempt and activation, model-call evidence for each role, simulator inputs/decisions, patches, raw judge verdicts, auxiliary metrics, and cleanup receipts. Report planned, returned, graded, masked, and failed counts separately; retain failed attempts when retrying. A grade of zero is healthy only when it is a valid measurement. Missing judge output is masked, never silently counted as measured zero.

Configuration resolution, lifecycle, native continuation, simulator parity, judge completion, and full baseline are separate qualification gates. No harness-swapping test is claimed by this recipe.

Audit smoke coverage after collection:

```bash
python benchmarks/swe_together/recipes/check_health.py \
  --tasks benchmarks/swe_together/data/full109.jsonl \
  --results artifacts/swe_together/results.jsonl \
  --output artifacts/swe_together/health.json
```

Repeat `--results` to include retained retry files. This checks at least one healthy graded attempt per task; the reference profile’s two-repeat completeness must be audited separately. Confirm provider-side sandbox cleanup against episode/session metadata as well.

Report Gym's generic trajectory health separately. The pinned OpenCode release supplies native turn and session records but omits per-assistant-message identifiers on model requests, leaving exact model-call ownership unavailable. Its generic trajectory status can therefore remain `unobserved` even when the explicit lifecycle, grading, metrics, and cleanup audit passes.

Task-specific upstream agent kwargs travel as the harness-neutral `harbor.agent-kwargs.v1` runtime policy. OpenCode applies the pinned upstream translation. In particular, upstream writes tool denials under `permission.tools`; that nesting does not disable the actual `webfetch`/`websearch` permission in OpenCode 1.15.13. Keep this source behavior visible in provenance and enforce the required network restrictions independently; do not claim these declarations alone enforce web-tool denial.
