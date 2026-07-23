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

**Rule reward vs. the validated recipe (read this).** The `verify()` in this PR is
**rule-based** and is the environment default: it checks the final URL / a URL
substring / visible DOM text / the reported answer. It is deterministic, free, and
CI-safe, which is exactly what you want for the bundled offline `site/` tasks. It is
**not** what produced our reference training result. The production-validated recipe
scores whole trajectories with an **LLM judge** (trajectory-level, binary yes/no) —
that is how real WebVoyager tasks (which have no rule-checkable end state) were graded
in the 0721 run cited under [Training](#training-reference-result). Treat the LLM
judge as a planned extension (drop a judge-backed `_score()` branch behind a new
`verifier_metadata` key such as `llm_judge`); do **not** read the rule reward as the
validated reward.

## Run

`example.sh` wraps everything below into one idempotent, fail-fast script with three
stages (`rollout` / `train` / `rollout --backend lexmount`). Read on for what each does.

### 0. Serve a policy model (the #1 reviewer stumbling block)

Stage A collects rollouts, so it needs a model endpoint that satisfies **both**:

1. **Speaks the Responses API** (`POST /v1/responses`). Chat-completions-only
   gateways do NOT work — the agent calls `/v1/responses` on the upstream with no
   chat fallback. Recent vLLM serves `/v1/responses` natively.
2. **Parses tool calls into structured `function_call` items.** If the server
   returns the model's tool-call markup as plain text (e.g. a literal
   `<tool_call>{"name": "browser_observe", ...}</tool_call>` string), the agent
   sees zero tool calls, the browser is never driven, and every rollout
   "succeeds" with **reward 0.0** — silently. For vLLM + Qwen-family models,
   launch with `--enable-auto-tool-choice --tool-call-parser hermes`.

Pick either:

- **A generic OpenAI-compatible endpoint (no local GPU).** Point at any server that
  implements the Responses API and export three vars; `example.sh` / the `openai_model`
  config read them:
  ```bash
  export POLICY_BASE_URL=https://your-endpoint/v1
  export POLICY_API_KEY=sk-...            # use any non-empty token if your gateway ignores it
  export POLICY_MODEL=your-model-name     # e.g. gpt-4.1
  ```
- **A locally served vLLM model (needs a GPU).** vLLM exposes the Responses API; serve
  a small tool-calling model and select the `vllm_model` config:
  ```bash
  export POLICY_KIND=vllm
  export POLICY_MODEL=Qwen/Qwen3-4B       # HF id or local checkpoint path
  ```

> The provided `example_rollouts.jsonl` was collected against a generic
> Responses-API endpoint. No base URL or key is stored in the committed rollouts.

### 1. Standalone backend test (no GPU, no Gym serving stack) — proves it works
```bash
uv run --no-project --with playwright python -m playwright install chromium
uv run --no-project --with playwright --with pytest --with pytest-asyncio python -m pytest tests/test_backend.py -q
```
Drives headless Chromium against the bundled offline `site/` (deterministic,
ToS-safe) and asserts navigate/click/type/observe + reward logic end-to-end.

> **Bare containers/VMs**: Chromium needs system libraries (libnss3, libgbm1,
> ...). If the install prints `Host system is missing dependencies to run
> browsers`, run (root/sudo required — `example.sh` attempts this automatically
> when it can):
> ```bash
> uv run --no-project --with playwright python -m playwright install-deps chromium
> ```
> Network note: setup downloads from pypi.org (deps), astral.sh (uv), and
> cdn.playwright.dev (~190 MB Chromium).

### 2. Stage A — rollouts as a NeMo-Gym environment (no GPU)
```bash
# after exporting POLICY_* from step 0:
bash example.sh rollout
```
which is equivalent to (run from the repo root, current-main CLI):
```bash
gym env start --resources-server lexmount_browser \
  --model-type openai_model --model "$POLICY_MODEL" \
  --model-url "$POLICY_BASE_URL" --model-api-key "$POLICY_API_KEY" &
gym eval run --no-serve --agent lexmount_browser_simple_agent \
  --input resources_servers/lexmount_browser/data/example.jsonl \
  --output resources_servers/lexmount_browser/data/example_rollouts.jsonl --limit 2
```

### 3. Stage B — GRPO training smoke (1 GPU, via NeMo-RL)
```bash
bash example.sh train   # prints the exact NeMo-RL launch; gated behind a GPU
```
Uses `configs/grpo_lexmount_browser_smoke.yaml` (ports SXH's validated 0721
hyperparameters to a 1-GPU smoke; every value is annotated with its provenance) with
NeMo-RL's `examples/nemo_gym/run_grpo_nemo_gym.py`.

### 4. Stage C — same rollout on the Lexmount cloud backend (one flag)

Two Stage-C-specific facts:

- **The SDK must live in the *server* venv** — the resources server runs in its
  own venv at `resources_servers/lexmount_browser/.venv` (created by the first
  `gym env start`, e.g. by running Stage A once). Installing `lexmount` into the
  repo-root venv does nothing for the server process.
- **Stage C rolls out on real-web tasks**, not the bundled offline `site/`: the
  offline tasks are local `file://` URIs, which a cloud browser cannot load.
  `example.sh` uses the 3 bundled WebVoyager sample tasks
  (`data/webvoyager_sample.jsonl`, already in this env's input format) and
  writes rollouts to `data/webvoyager_rollouts.jsonl` (conservative
  `url_contains` reward — see [Data](#data-webvoyager-bridge)).

```bash
bash example.sh rollout            # Stage A once, so the server venv exists
uv pip install --python resources_servers/lexmount_browser/.venv/bin/python "lexmount>=0.5.13"
export LEXMOUNT_API_KEY=... LEXMOUNT_PROJECT_ID=... LEXMOUNT_BASE_URL=...
bash example.sh rollout --backend lexmount
```

## Data (WebVoyager bridge)

`scripts/convert_webvoyager.py` maps WebVoyager-style task JSON
(`{web_name, id, ques, web}`, [MinorJerry/WebVoyager](https://github.com/MinorJerry/WebVoyager),
**MIT license**) into this env's `example.jsonl` format (`initial_url` +
`verifier_metadata`), following the cleaning conventions of the validated 0721
pipeline (row-count / SHA-256 validation, duplicate-id rejection, source-id
preservation, task-agnostic system prompt — no answers or synthetic data injected).
The full 600-task WebVoyager set is **not bundled** (fetch it from upstream); three
sample tasks (Allrecipes--0 / Amazon--0 / GitHub--0) ship **already converted** in
`data/webvoyager_sample.jsonl` (directly usable as rollout input — Stage C uses it);
their raw upstream form is embedded in the converter for `--selftest`.

Because WebVoyager has no rule-checkable ground truth, converted rows carry a
**conservative** `url_contains` spec (agent reached/stayed on the task host) and, with
`--emit-judge-todo`, a `needs_llm_judge: true` marker — a rule pass is necessary but
not sufficient; full success needs the LLM judge (see [Reward](#reward)).
```bash
python scripts/convert_webvoyager.py --selftest                       # no external data
python scripts/convert_webvoyager.py --source WebVoyager_data.jsonl \
    --output data/webvoyager_example.jsonl --limit 3 --emit-judge-todo
```

## Training (reference result)

The production recipe (colleague SXH's 0721 experiment) is a GRPO, full-parameter FSDP
run on **2×8 Ascend 910B** with **Qwen3-8B**: 8 tasks/step × 8 rollouts/task = 64
rollouts/step, 60 steps, 4 epochs, lr 5e-6 constant, context 40960 (4096 prompt /
36864 response), 10 assistant + 10 user turns, trajectory-level **LLM judge** reward.
The cloud (Lexmount) arm's mean reward rose from **≈0.105** (first 10 steps) to
**≈0.289** (last 10 steps). `configs/grpo_lexmount_browser_smoke.yaml` scales this to a
1-GPU smoke and annotates every deviation.

## Files (Gym `new-environment` spec)
- [x] `app.py` — resources server (seed_session + tools + verify)
- [x] `backend.py` — `BrowserBackend` + `PlaywrightBackend` + `LexmountBackend` (cloud SDK)
- [x] `configs/lexmount_browser.yaml`
- [x] `configs/grpo_lexmount_browser_smoke.yaml` — 1-GPU GRPO smoke (NeMo-RL)
- [x] `site/` — bundled offline test site (deterministic tasks/CI)
- [x] `generate_data.py` + `data/example.jsonl` — 5 example tasks (Responses-API inputs)
- [x] `scripts/convert_webvoyager.py` + `data/webvoyager_sample.jsonl` — WebVoyager data bridge
- [x] `tests/test_backend.py` — standalone e2e backend test
- [x] `example.sh` — one-script Stage A/B/C reproduction
- [x] `requirements.txt`, `README.md`
- [x] `data/example_rollouts.jsonl` — 5 rollouts collected against a Responses-API endpoint (reward 1.0 on the offline site)
- [x] reward wiring validated end-to-end (Stage A); GRPO training-signal run documented above (Ascend 910B, Qwen3-8B)

## Licensing
- Environment code: Apache 2.0 (matches NeMo-Gym).
- Reference backend: Playwright (Apache 2.0).
- Example tasks: bundled offline `site/` is original (Apache 2.0). WebVoyager sample
  tasks in `data/webvoyager_sample.jsonl` are from WebVoyager (MIT); the full dataset
  is not redistributed here.
- Lexmount cloud SDK: a separate, optional dependency installed by the operator (not bundled); only needed for `backend: lexmount`.
