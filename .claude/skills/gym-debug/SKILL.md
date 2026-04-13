---
name: gym-debug
description: >
  Diagnose NeMo Gym server failures, rollout errors, and infrastructure issues. Use when
  servers won't start, rollouts fail or hang, rewards are unexpected, or there are
  concurrency/scaling issues. Covers request tracing, log analysis, config validation,
  Ray diagnostics, and common failure modes.
license: Apache-2.0
compatibility: Requires Python 3.12+, access to NeMo Gym server logs.
metadata:
  author: nvidia-nemo-gym
  version: "1.0"
allowed-tools: Bash(python:*) Bash(ng_*) Bash(git:*) Bash(curl:*) Bash(ps:*) Read Grep Glob
---

# NeMo Gym Debugging

## Step 1: Establish the failure mode

Categorize the problem before investigating:

| Symptom | Category | Start at |
|---------|----------|----------|
| Server won't start | Startup | Step 2 |
| Requests hang or timeout | Concurrency | Step 3 |
| Rollouts return all 0.0 rewards | Verification | Step 4 |
| Servers crash under load | Scaling | Step 5 |
| Config errors on launch | Configuration | Step 6 |
| Inconsistent results across runs | Nondeterminism | Step 7 |

## Step 2: Startup failures

1. Check `ng_status` — are all servers reporting healthy?
2. Read server logs for import errors, missing dependencies, or port conflicts
3. If using auto-installed tools, check that `ensure_<tool>()` completed — look for the install directory (e.g. `.lean4/`, `.go/`)
4. Verify `env.yaml` has correct model endpoint config (`policy_base_url`, `policy_api_key`, `policy_model_name`)
5. Check that YAML config composes correctly: `ng_dump_config "+config_paths=[...]"`

## Step 3: Concurrency issues

Symptoms: requests hang, timeouts, server becomes unresponsive at scale.

1. **Missing semaphore**: Check that subprocess calls are bounded by `asyncio.Semaphore`. Unbounded spawning exhausts system resources.
2. **httpx in use**: Any httpx/httpcore usage causes O(n^2) connection pooling hangs at 16k+ requests. Must use aiohttp via `nemo_gym.server_utils.request()`.
3. **`ray.get()` blocking event loop**: Use `await future` for Ray remote tasks.
4. **aiohttp session lifecycle**: The global client is a singleton with connection pooling. Verify it's not being created per-request.
5. **Cookie propagation**: In stateful environments, missing `cookies=request.cookies` on downstream calls causes session loss, leading to repeated initialization or state corruption.

## Step 4: Verification failures

All rollouts returning reward 0.0 when they shouldn't:

1. **Output parsing**: Is the model's response being extracted correctly? Check code extraction regex. Common miss: markdown fences with language tags (` ```python ` vs ` ``` `).
2. **Think blocks**: Thinking models wrap output in `<think>`/`<thinking>` blocks. These must be stripped before parsing.
3. **Test case format**: Does `verifier_metadata` in the JSONL match what `verify()` expects? Field name mismatches are silent failures.
4. **Subprocess execution**: If verify() runs code, check: is the binary installed and on PATH? Is the working directory correct? Is the timeout sufficient?
5. **Manual test**: Call `/verify` directly with a known-good input to isolate whether the issue is in verification or upstream.

```bash
curl -X POST http://localhost:<port>/verify \
  -H "Content-Type: application/json" \
  -d '{"response": {"output_text": "known good answer"}, "verifier_metadata": {...}}'
```

## Step 5: Scaling failures

Servers crash or OOM under high concurrency (4k-65k requests):

1. Check semaphore value — too high exhausts memory, too low bottlenecks throughput
2. Check Ray worker count and memory allocation
3. Look for memory leaks: subprocess output not being released, accumulating results in memory
4. Verify `errors="replace"` on all subprocess decode — non-UTF8 output without this flag can cause exceptions that leak resources

## Step 6: Configuration issues

1. Run `ng_dump_config "+config_paths=[...]"` to see the merged config
2. Check instance name consistency — agent must reference exact names of resources and model servers
3. Verify Hydra override syntax: `+key=value` for new keys, `key=value` for existing
4. Check for YAML indentation issues (especially in dataset sections with nested `gitlab_identifier`)
5. OmegaConf interpolation errors: `${var}` references must resolve

## Step 7: Nondeterminism

Results vary significantly across identical runs:

1. **Temperature**: Ensure `temperature: 1.0` (or your chosen value) is being passed correctly
2. **Random seeds**: If verify() uses randomness (shuffled test cases, random sampling), seed it
3. **Stateful environments**: Check that state is being properly reset between requests — leaked state from one request affects the next
4. **Race conditions**: In multi-turn agents, verify that async operations are properly sequenced

## LLM-as-Judge debugging

Judge-based benchmarks (equivalence_llm_judge, jailbreak_detection) have additional failure modes:

1. **Missing judge model server**: Judge configs require a second model server instance. If the config only defines `policy_model` but the server also needs `judge_model`, all judge calls fail silently or return default rewards.

2. **Judge rate limiting**: `judge_endpoint_max_concurrency` bounds concurrent judge calls. If set too low, rollout collection stalls. If too high, the judge API returns 429s. Check the judge model server logs separately from the resources server logs.

3. **Two-stage reward issues** (jailbreak_detection with `use_combined_reward: true`): The final reward is `safety_reward * quality_reward`. If safety passes (1.0) but quality fails (0.3), the combined reward is 0.3, not 0.0. This is intentional partial credit, not a bug — but can look like inconsistent rewards if you don't know the formula.

4. **Positional bias** (equivalence_llm_judge with `check_twice_swap: true`): The judge runs twice with expected/generated answers swapped. If the two runs disagree, `reward_if_swap_fails` applies (default 0.0). High disagreement rates indicate the judge is sensitive to answer ordering, not answer correctness.

5. **Regex extraction failures**: Judge-based servers often extract answers via regex before judging. Check `question_extract_regex`, `response_extract_regex`, and per-record `template_metadata.output_regex`. When regex fails, the server may fall back to `check_full_generation_on_fail` — giving partial credit (`reward_if_full_generation_succeeds: 0.5`) instead of 0.0.

## Custom VerifyResponse fields

Production servers return more than just `reward`. These extra fields are critical for debugging:

| Server | Extra fields | What they tell you |
|--------|-------------|-------------------|
| code_gen | `extracted_model_code`, `result`, `unit_tests_time_taken`, `reasoning_format_violation_rate` | What code was extracted, what happened when it ran, whether thinking tags were malformed |
| spider2_lite | `extracted_sql`, `execution_match`, `failure_reason` (enum: NO_SQL_EXTRACTED, EXECUTION_ERROR, etc.) | Whether SQL was found, whether it ran, why it failed |
| equivalence_llm_judge | `expected_answer`, `judge_evaluations` [{verdict_label}] | What the judge saw and decided |
| tavily_search | `num_tool_calls`, `metrics` [{function, status, time_taken}] | How many API calls were made and which failed |

When debugging, always read these extra fields from the rollout JSONL — they tell you exactly where in the pipeline things went wrong.

## Ray-specific issues

- **Socket path too long**: On HPC/Lustre with long working directory paths, Ray's AF_UNIX socket exceeds the 107-byte Linux limit. Fix: `export RAY_TMPDIR=/tmp` before running.
- **`ng_test` venv isolation**: `os.environ` changes in Python don't propagate to `ng_test` venvs. Set env vars externally: `RAY_TMPDIR=/tmp ng_test ...`
