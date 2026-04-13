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
  version: "2.0"
allowed-tools: Bash Read Grep Glob
---

# NeMo Gym Code Review

Review code for anti-patterns that cause production failures in NeMo Gym's async, high-concurrency microservice architecture (4k-65k concurrent requests).

This skill is **script-first**: run the deterministic checker, then apply judgment for context the script can't catch.

## Step 1: Run the automated checker

Run `scripts/review.py` against the target path. It checks 11 Python rules and 1 YAML rule.

```bash
# Scan a directory (most common — scan the whole server)
python scripts/review.py <path>

# Scan with JSON output (for programmatic use)
python scripts/review.py <path> --json

# Only BLOCK-level findings
python scripts/review.py <path> --severity BLOCK
```

The script exits 1 if any BLOCK-level findings exist, 0 otherwise.

> **Note**: `scripts/review.py` is self-contained — no dependencies beyond the Python standard library. It works outside the NeMo Gym repo.

## Step 2: Interpret the results

The script reports findings at two severity levels:

### BLOCK (must fix before merge)

| Rule | What it catches |
|------|----------------|
| `httpx-usage` | httpx/httpcore imports — O(n^2) connection pooling hangs at 16k+ requests |
| `ray-get-async` | `ray.get()` in async context — blocks the event loop |
| `missing-semaphore` | Subprocess calls without `asyncio.Semaphore` — unbounded at scale |
| `missing-errors-replace` | `.decode()` without `errors="replace"` — crashes on non-UTF8 |
| `env-var-config` | `os.environ`/`os.getenv` for config — must use YAML/Hydra |
| `wrong-client` | litellm/anthropic imports — must use `nemo_gym/openai_utils.py` |
| `missing-cookies` | Agent `server_client.post()` without `cookies=` — breaks stateful sessions |
| `missing-token-ids` | Multi-turn agent without token ID accumulation — breaks RL training |
| `non-binary-reward` | Reward values other than 0.0/1.0 without documentation |

### WARN (should fix)

| Rule | What it catches |
|------|----------------|
| `missing-think-strip` | Parses model output without stripping `<think>` blocks |
| `sync-endpoint` | `def verify`/`def run` instead of `async def` |
| `verified-true` | Config has `verified: true` — confirm baselining was done |
| `missing-gitlab-id` | Train/validation dataset without `gitlab_identifier` |
| `missing-license` | Train/validation dataset without `license` field |

For each finding, the script provides the file, line number, rule name, description, and fix suggestion.

## Step 3: Apply judgment (what the script can't catch)

The script handles pattern matching. These require human/agent judgment:

1. **Test coverage completeness**: Does the server have tests for verify pass, verify fail (wrong output), verify fail (no extraction), verify fail (compilation error if applicable), and verify timeout? Target >= 95% coverage.

2. **`pytest.mark.skipif` for external tools**: Tests requiring tools not in the standard library should use `skipif(shutil.which("tool") is None, ...)`.

3. **Unguarded optional fields**: Access patterns like `body.field.get("key")` should use `(body.field or {}).get("key", default)`.

4. **YAML instance name consistency**: Agent configs reference resources/model servers by name — verify these match actual instance names in the config.

5. **Intentional partial rewards**: If the script flags `non-binary-reward`, check whether the partial credit is documented and intentional (e.g., judge-based servers with `check_twice_swap`).

## Step 4: Report

Structure the review as:

```
## Review: [server/agent name]

### Automated findings
<paste or summarize review.py output>

### Manual checks
- Test coverage: [pass/fail/not applicable]
- Optional field guards: [pass/fail]
- YAML consistency: [pass/fail]

### Summary
X BLOCK, Y WARN — [merge/do not merge]
```

## References

Full context for each anti-pattern and its fix:

- `references/anti-patterns.md` — Why each pattern fails in production, with architecture context
- `references/fix-patterns.md` — Production code patterns: aiohttp adapter, cookie chain, token accumulation, semaphore-subprocess, think-block stripping variants
