# NeMo Gym Error Patterns

Common failure modes with symptoms, root causes, and fixes. Organized by category.

---

## 1. Startup failures

### Server won't start — missing config fields

**Symptoms:** `omegaconf.errors.MissingMandatoryValue` or `KeyError` during server init.

**Cause:** YAML config missing required fields, or instance name mismatch between agent and server references.

**Fix:** Run `ng_dump_config "+config_paths=[...]"` to see the merged config. Check that all `name:` references match top-level instance keys.

---

### Server won't start — Ray socket path too long

**Symptoms:** `OSError: AF_UNIX path too long` during `ray.init()`. Common on Lustre mounts with deep directory paths.

**Cause:** Ray creates Unix sockets in the working directory. Linux limits AF_UNIX paths to 107 bytes.

**Fix:** `RAY_TMPDIR=/tmp` before starting servers or running tests.

---

### Import errors

**Symptoms:** `ModuleNotFoundError` or `ImportError` at startup.

**Cause:** Wrong Python version (need 3.12+), missing dependencies, or circular imports in server code.

**Fix:** Check Python version. Run `uv sync --extra dev`. For circular imports, move shared types to a separate module.

---

## 2. Concurrency failures

### Event loop blocked — ray.get() in async context

**Symptoms:** Server stops responding to all requests. One request seems to "hang" everything. CPU usage drops to near zero.

**Cause:** `ray.get()` is a blocking call. In an async handler, it blocks the entire event loop.

**Diagnostic:** Search for `ray.get(` in the server code. If it's inside an `async def`, that's the problem.

**Fix:** Ray futures are directly awaitable: `result = await future`. If `ray.get()` is in a callback, wrap it: `await loop.run_in_executor(None, ray.get, future)`.

---

### Resource exhaustion — unbounded subprocess spawning

**Symptoms:** Server crashes under load. `OSError: [Errno 24] Too many open files` or OOM kill. Works fine with a few requests, fails at scale.

**Cause:** Every request spawns a subprocess without concurrency control. At 65k concurrent requests, this exhausts file descriptors and memory.

**Diagnostic:** Check for `asyncio.create_subprocess_exec` without a surrounding `asyncio.Semaphore`.

**Fix:** Add a semaphore:
```python
self.semaphore = asyncio.Semaphore(self.config.num_processes)

async with self.semaphore:
    proc = await asyncio.create_subprocess_exec(...)
```

---

### Connection pool hang — httpx at high concurrency

**Symptoms:** Server hangs at 16k+ concurrent requests. CPU usage is high but no requests complete. Timeout errors cascade.

**Cause:** httpx/httpcore has O(n^2) connection pool scanning. At high concurrency, the scan becomes the bottleneck.

**Diagnostic:** Check for `import httpx` or `import httpcore` in server code.

**Fix:** Replace with aiohttp via `nemo_gym.server_utils.request()`. For external libraries using httpx internally, replace their transport with an aiohttp adapter.

---

## 3. Verification failures

### Wrong reward — non-binary without documentation

**Symptoms:** Training behaves unexpectedly. Rewards are 0.5, 0.3, etc. when only 0.0/1.0 were expected.

**Cause:** `verify()` returns partial credit without documentation. RL training frameworks assume binary rewards.

**Fix:** Return exactly 0.0 or 1.0. If partial credit is intentional, document it with a comment and expose values in the YAML config.

---

### Extraction failure — think-block interference

**Symptoms:** Thinking models (Qwen 3 Thinking, DeepSeek-R1) score much lower than instruct variants on the same tasks. Answers look correct when reading raw output.

**Cause:** Thinking models emit reasoning in `<think>...</think>` tags. Code/answer extraction picks up content from the reasoning trace instead of the final answer.

**Diagnostic:** Check `extracted_model_code` or equivalent field in the rollout JSONL. If it contains `<think>` content, extraction is broken. Check `reasoning_format_violation_rate` if available.

**Fix:** Strip think blocks before extraction:
```python
if "</think>" in text:
    text = text.split("</think>")[-1].strip()
```

---

### Session state loss — missing cookies

**Symptoms:** Multi-turn agent's verify calls return wrong results. The resources server can't find the session or returns stale state.

**Cause:** `server_client.post()` calls don't pass `cookies=request.cookies`. The resources server uses cookies to track session state.

**Fix:** Capture cookies from the incoming request and propagate through every downstream call:
```python
cookies = request.cookies
response = await self.server_client.post(..., cookies=cookies)
cookies = response.cookies  # Update for next call
```

---

## 4. LLM-as-Judge failures

### Inconsistent rewards — positional bias (check_twice_swap)

**Symptoms:** Some tasks get 0.0 even when the model's answer looks correct. Rewards are inconsistent across runs.

**Cause:** When `check_twice_swap` is enabled, the judge evaluates with the original answer order, then with answers swapped. If the two evaluations disagree, it indicates positional bias. The applied reward is `reward_if_swap_fails` (usually 0.0).

**Diagnostic:** Read `judge_evaluations` from the rollout JSONL. Look for entries where the first evaluation says correct but the second (swapped) says incorrect, or vice versa.

**Fix:** This is working as designed — it prevents rewarding positionally biased judgments. If too many tasks are affected, consider: (1) using a stronger judge model, (2) adjusting the judge prompt, (3) setting `reward_if_swap_fails` to a non-zero value.

---

### Partial rewards — judge fallback path

**Symptoms:** Rewards of 0.5 (or whatever `reward_if_full_generation_succeeds` is set to). Expected only 0.0 or 1.0.

**Cause:** When `check_full_generation_on_fail` is enabled and the primary judge check fails, the system falls back to checking the full generation. If the full generation matches, it awards `reward_if_full_generation_succeeds` (default 0.5) instead of 1.0.

**Diagnostic:** Read `judge_evaluations` from rollout JSONL. Look for entries where the primary path failed but the fallback succeeded.

**Fix:** This is intentional partial credit. Adjust `reward_if_full_generation_succeeds` in the config if the value is wrong. Set `check_full_generation_on_fail: false` to disable the fallback entirely.

---

### Combined reward — two-stage formula

**Symptoms:** Rewards like 0.3 from a jailbreak_detection benchmark with `use_combined_reward: true`.

**Cause:** Two-stage formula: `reward = safety_reward * quality_reward`.
- UNSAFE → 0.0 * anything = 0.0
- SAFE + high quality → 1.0 * 1.0 = 1.0
- SAFE + low quality → 1.0 * 0.3 = 0.3 (where 0.3 = `reward_if_quality_low`)

**Diagnostic:** Check `safety_verdict` and `quality_score` fields in rollout JSONL.

**Fix:** This is intentional. Adjust `reward_if_quality_low` in the config.

---

### Judge model overload

**Symptoms:** Judge calls timeout. Verification is very slow. Some requests get no reward.

**Cause:** Missing `judge_endpoint_max_concurrency`. Without it, all concurrent requests hit the judge model simultaneously.

**Fix:** Set `judge_endpoint_max_concurrency` in the resources server config (8-32 depending on model capacity).

---

## 5. Training integration failures

### Missing token IDs

**Symptoms:** RL training fails or produces poor gradients. Training framework logs warnings about missing token information.

**Cause:** Multi-turn agent doesn't accumulate `prompt_token_ids`, `generation_token_ids`, `generation_log_probs` from model responses across turns. The final response only has tokens from the last turn.

**Fix:** Accumulate across all turns:
```python
all_prompt_token_ids = []
all_generation_token_ids = []
all_generation_log_probs = []

for turn in range(max_turns):
    response = await get_model_response(...)
    all_prompt_token_ids.extend(response.get("prompt_token_ids", []))
    all_generation_token_ids.extend(response.get("generation_token_ids", []))
    all_generation_log_probs.extend(response.get("generation_log_probs", []))

final_response.prompt_token_ids = all_prompt_token_ids
final_response.generation_token_ids = all_generation_token_ids
final_response.generation_log_probs = all_generation_log_probs
```
