---
name: gym-review
description: >
  Review code changes for NeMo Gym anti-patterns and correctness issues. Use when
  reviewing a PR, auditing a benchmark implementation, or checking a resources server,
  agent, or config before merge. Catches: httpx usage (must use aiohttp), ray.get() in
  async context, missing semaphores, non-binary rewards, missing think-block stripping,
  env vars instead of YAML config, test coverage gaps, and cookie propagation issues.
license: Apache-2.0
compatibility: Requires Python 3.10+. Works standalone or inside the NeMo Gym repo.
metadata:
  author: nvidia-nemo-gym
  version: "3.0"
allowed-tools: Bash Read Grep Glob
---

# NeMo Gym Code Review

Review code for anti-patterns that fail in NeMo Gym's async, high-concurrency microservice architecture. **Run the script first. Apply judgment only for things the script can't catch.** Everything else lives in the reference docs.

## Step 1: Run the script

```bash
python scripts/review.py <path>         # scan a directory
python scripts/review.py <path> --json  # machine-readable output
```

The script checks 11 Python rules and 1 YAML rule, prints per-finding `(file:line, rule, description, fix)` entries, and exits 1 if any BLOCK-level findings exist. It's self-contained (stdlib only) and works outside the repo.

Each finding is labeled with a `rule` name — cite the rule name in your review so reviewers can cross-reference. The ruleset and full rationale for each rule is in `references/anti-patterns.md`; production fix templates are in `references/fix-patterns.md`.

## Step 2: Interpret severity

- **BLOCK** — must fix before merge. Production failure modes: connection-pool hangs, blocked event loops, unbounded concurrency, etc.
- **WARN** — should fix. Correctness or completeness issues that don't crash production but do affect training quality or confidence in the benchmark.

If the script is quiet, the code is clean on the patterns the ruleset covers. Say so affirmatively rather than hedging — a clean review report should state "no BLOCK or WARN findings, approval recommended" explicitly.

## Step 3: Judgment beyond the script

Five things the pattern matcher can't verify. Check these manually:

- **Test coverage** — each server should test verify() pass, verify() fail (wrong output), verify() fail (no extraction), verify() fail (timeout or error path), and any partial-reward path. Target ≥95% coverage.
- **`pytest.mark.skipif`** on tests requiring external tools not in stdlib.
- **Unguarded optional fields** — `(body.field or {}).get("key", default)`, not `body.field.get(...)`.
- **YAML instance-name references** — agent `resources_server.name` / `model_server.name` must match top-level instance names in the composed config.
- **Intentional partial rewards** — if `non-binary-reward` fires, confirm the partial credit is documented (e.g. judge fallback, `check_twice_swap`).

For the exact anti-pattern catalog and fix patterns, see the references.

## Step 4: Report

Structure the review around the script's findings and the manual checks above. When the code is clean, state the approval recommendation directly — don't manufacture concerns.
