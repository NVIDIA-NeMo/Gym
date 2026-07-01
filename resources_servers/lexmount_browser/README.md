# Lexmount Browser — NeMo-Gym interactive-browser environment

A NeMo-Gym **stateful resources server** that turns a real browser into an RL
environment. Each rollout (`session_id`) owns one isolated live browser context;
the policy drives it with tool calls (navigate / click / type / observe / finish);
`verify()` returns the task reward from the live browser state.

This is the interactive complement to the existing read-only browsing envs
(`google_search`, `browsecomp_advanced_harness`): those search-and-extract text;
this one *operates* pages (stateful, multi-step web agency).

## Pluggable backend

The server depends only on the small `BrowserBackend` contract (`backend.py`):
`open / goto / click / type / observe / current_url / text / close`. Switch backend
with one config line; nothing else in the environment changes.

- `backend: playwright` — open-source reference (headless Chromium). Runnable today,
  used for local dev and CI (no proprietary deps). **Default.**
- `backend: lexmount` — production. Each rollout gets an isolated browser session in
  the Lexmount cloud (browser runs off the training node); we connect over CDP and
  reuse the same page-driving logic.

### Using the Lexmount cloud backend

1. Register at **https://browser.lexmount.com**, create a project, and copy your
   **API key** and **project ID**.
2. Install the SDK and export credentials (never commit them):
   ```bash
   pip install "lexmount>=0.5.13"
   export LEXMOUNT_API_KEY=<your-api-key>
   export LEXMOUNT_PROJECT_ID=<your-project-id>
   export LEXMOUNT_BASE_URL=https://api.lexmount.com   # API base shown in your dashboard
   ```
3. Switch one line in `configs/lexmount_browser.yaml` (nothing else changes —
   tools, observation, and reward are identical to the Playwright backend):
   ```yaml
   resources_servers:
     lexmount_browser:
       backend: lexmount        # was: playwright
   ```

## Tools & observation

Tools: `browser_navigate(url)`, `browser_click(element_id)`,
`browser_type(element_id, text)`, `browser_observe()`, `browser_finish(answer)`.
Observation is a **compact numbered list of interactive elements** (`[id] role: name`)
plus URL/title — deliberately token-cheap (raw HTML/pixels are far too expensive for
small policies and for training context length). `element_id`s come from the most
recent observation.

## Reward

`verify()` scores a per-task success spec in `verifier_metadata`:
`final_url` / `url_contains` / `dom_contains` / `answer_equals`. Sparse 0/1 outcome
reward by default (least reward-hackable); extend `_score()` with new keys as needed.

## Run

### 1. Standalone backend test (no GPU, no Gym serving stack) — proves it works
```bash
uv run --no-project --with playwright python -m playwright install chromium
uv run --no-project --with playwright --with pytest --with pytest-asyncio python -m pytest tests/test_backend.py -q
```
Drives headless Chromium against the bundled offline `site/` (deterministic,
ToS-safe) and asserts navigate/click/type/observe + reward logic end-to-end.

### 2. As a NeMo-Gym environment (rollouts / training)
```bash
gym env start --resources-server lexmount_browser --model-type vllm_model &
gym eval run --no-serve \
  --agent lexmount_browser_simple_agent \
  --input data/example.jsonl \
  --output data/example_rollouts.jsonl --limit 1
```
Then plug into NeMo-RL GRPO via `examples/nemo_gym/run_grpo_nemo_gym.py`.

## Files (Gym `new-environment` spec)
- [x] `app.py` — resources server (seed_session + tools + verify)
- [x] `backend.py` — `BrowserBackend` + `PlaywrightBackend` + `LexmountBackend` (cloud SDK)
- [x] `configs/lexmount_browser.yaml`
- [x] `site/` — bundled offline test site (deterministic tasks/CI)
- [x] `generate_data.py` + `data/example.jsonl` — 5 example tasks (Responses-API inputs)
- [x] `tests/test_backend.py` — standalone e2e backend test
- [x] `requirements.txt`, `README.md`
- [ ] `data/example_rollouts.jsonl` — generate after a model server is wired
- [ ] reward profiling + GRPO training-signal run (Qwen3-4B/30B-A3B) for the PR

## Licensing
- Environment code: Apache 2.0 (matches NeMo-Gym).
- Reference backend: Playwright (Apache 2.0).
- Lexmount cloud SDK: a separate, optional dependency installed by the operator (not bundled); only needed for `backend: lexmount`.
