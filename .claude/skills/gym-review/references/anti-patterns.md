# NeMo Gym Anti-Patterns Reference

## Architecture context

NeMo Gym is a microservice architecture with three FastAPI server types (resources, model, agent) communicating over async HTTP. Servers handle 4k-65k concurrent requests. Anti-patterns in this list cause production failures at scale.

---

## BLOCK-level anti-patterns

### 1. httpx-usage

**What**: Any import of `httpx` or `httpcore`.

**Why**: httpx/httpcore has O(n^2) connection pooling. At 16k+ concurrent requests, the connection pool scan becomes the bottleneck and servers hang. This was discovered in production and documented in `docs/infrastructure/engineering-notes/aiohttp-vs-httpx.md`.

**Fix**: All async HTTP must go through `nemo_gym.server_utils.request()`, which uses aiohttp with a singleton connection pool. When wrapping external libraries that use httpx internally, replace their HTTP transport with an aiohttp adapter (see fix-patterns.md § aiohttp-adapter).

---

### 2. ray-get-async

**What**: Calling `ray.get()` in an async function.

**Why**: `ray.get()` is a blocking call. In an async context, it blocks the entire event loop, preventing all other coroutines from running. One blocked `ray.get()` in a verify handler stops the server from processing any other requests.

**Fix**: Ray futures are directly awaitable: `result = await future`. If you must use `ray.get()` (e.g., in a callback), wrap it in `loop.run_in_executor(None, ray.get, future)`.

---

### 3. missing-semaphore

**What**: `asyncio.create_subprocess_exec` or subprocess calls without a bounding `asyncio.Semaphore`.

**Why**: Without concurrency control, every incoming request spawns a subprocess. At 65k concurrent requests, this exhausts file descriptors, memory, and CPU. The server crashes or the OS kills processes.

**Fix**: Initialize a semaphore in `model_post_init()`:
```python
self.semaphore = asyncio.Semaphore(self.config.num_processes)
```
Wrap all subprocess calls:
```python
async with self.semaphore:
    proc = await asyncio.create_subprocess_exec(...)
```

---

### 4. non-binary-reward

**What**: `verify()` returning reward values other than 0.0 or 1.0 without explicit documentation.

**Why**: RL training frameworks assume binary rewards unless configured otherwise. Non-binary rewards silently change training dynamics. Partial credit IS used in some servers (e.g., jailbreak_detection's combined reward, equivalence_llm_judge's fallback), but it must be intentional and documented.

**Fix**: Return exactly 0.0 or 1.0. If partial credit is intentional, add a comment explaining the reward structure and ensure the YAML config exposes the partial reward values (e.g., `reward_if_quality_low: 0.3`).

---

### 5. missing-errors-replace

**What**: `subprocess.stdout.decode()` or `.stderr.decode()` without `errors="replace"`.

**Why**: Model-generated code can produce non-UTF8 output (binary data, corrupted strings). Without `errors="replace"`, the decode raises `UnicodeDecodeError`, which either crashes the request or leaks resources if the exception isn't caught properly.

**Fix**: Always use `.decode(errors="replace")`.

---

### 6. env-var-config

**What**: Using `os.environ` or `os.getenv()` for configuration.

**Why**: NeMo Gym uses Hydra/OmegaConf for all configuration. Environment variables bypass the config system, making deployments non-reproducible and configs non-composable. The ONE exception is `${oc.env:VAR,default}` in YAML for deployment-specific infrastructure values (sandbox hosts, etc.).

**Allowed env vars**: `RAY_TMPDIR`, `PATH`, `LD_LIBRARY_PATH`, `HOME`, `USER`, `TMPDIR`, `CUDA_VISIBLE_DEVICES`.

---

### 7. wrong-client

**What**: Imports of `litellm`, `anthropic`, or OpenAI clients other than NeMo Gym's wrapper.

**Why**: NeMo Gym pins `openai<=2.6.1` for schema compatibility. Other clients have incompatible message formats, don't integrate with the config system, and don't go through the aiohttp transport.

**Fix**: Use `nemo_gym/openai_utils.py` for all LLM calls.

---

### 8. missing-cookies

**What**: Agent server makes `server_client.post()` calls without passing `cookies=request.cookies`.

**Why**: Stateful environments (e.g., multi-turn proof refinement) use cookies to track session state on the resources server. Missing cookies mean the resources server can't associate requests with the correct session, causing state loss or corruption.

**Fix**: Capture cookies from the incoming request and propagate through every downstream call:
```python
cookies = request.cookies
response = await self.server_client.post(..., cookies=cookies)
cookies = response.cookies  # Update for next call
```

---

### 9. missing-token-ids

**What**: Multi-turn agents that don't propagate `prompt_token_ids`, `generation_token_ids`, `generation_log_probs` across turns.

**Why**: RL training requires token-level information to compute policy gradients. If multi-turn agents don't accumulate token IDs from each model call, the training framework can't attribute rewards to specific generation decisions.

**Fix**: Extract from each model response and accumulate:
```python
all_prompt_token_ids.extend(response.get("prompt_token_ids", []))
all_generation_token_ids.extend(response.get("generation_token_ids", []))
all_generation_log_probs.extend(response.get("generation_log_probs", []))
```

---

## WARN-level anti-patterns

### 10. missing-think-strip

**What**: Code that parses model output without stripping `<think>`/`<thinking>` blocks.

**Why**: Thinking models (Qwen 3 Thinking, DeepSeek-R1) emit reasoning in `<think>...</think>` tags. If these aren't stripped, code extraction picks up code from the reasoning trace, answer extraction matches intermediate reasoning, and `reasoning_format_violation_rate` increases.

**Fix**: Strip before parsing:
```python
if "</think>" in text:
    text = text.split("</think>")[-1].strip()
```

---

### 11. sync-endpoint

**What**: `/run` or `/verify` defined as `def` instead of `async def`.

**Why**: Synchronous handlers block the FastAPI event loop. Under concurrent load, this serializes all requests.

---

### 12. test-coverage

**What**: New servers with insufficient test coverage (< 95%).

**Required test cases**: verify pass, verify fail (wrong output), verify fail (no code/answer extracted), verify fail (compilation error if applicable), verify timeout.

---

### 13. missing-skipif

**What**: Tests requiring external tools without `pytest.mark.skipif(shutil.which("tool") is None, ...)`.

**Why**: Tests must pass in CI environments where the tool may not be installed. If the server auto-installs the tool, add a `pytest_configure` hook in `conftest.py` to run the install before test collection — `skipif` evaluates at import time, before fixtures.

---

### 14. unguarded-optional-fields

**What**: Accessing `body.field.get("key")` without guarding against None.

**Fix**: Use `(body.field or {}).get("key", default)`.
