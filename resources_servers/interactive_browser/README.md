# Interactive Browser — NeMo-Gym browser RL environment

A NeMo-Gym **stateful resources server** that turns a real browser into an RL
environment. Each rollout (`session_id`) owns one isolated live browser context;
the policy drives it with tool calls (navigate / click / type / observe / finish);
`verify()` returns the task reward from the live browser state.

This is the interactive complement to the existing read-only browsing envs
(`google_search`, `browsecomp_advanced_harness`): those search-and-extract text;
this one *operates* pages (stateful, multi-step web agency).

## Backends

The server depends only on the small `BrowserBackend` contract
(`browser/base.py`): `open / goto / click / type / observe / current_url / text /
close`. Where the browser actually runs is one config block; nothing else in the
environment changes.

| `backend:` | Browser runs | Use it for |
| --- | --- | --- |
| `local_playwright` (**default**) | A Chromium launched on this machine, one per rollout | Local dev, CI, offline `site/` tasks. No account, no quota, no network. |
| `remote_cdp` | Anywhere reachable over CDP, supplied per rollout by a **session provider** | Keeping browsers off the training node: a browser container, another host, or a hosted browser service. |

```yaml
backend:
  local_playwright:
    headless: true
```

```yaml
backend:
  remote_cdp:
    connect_timeout_s: 60
    session_provider:
      static_cdp:
        cdp_url: "http://127.0.0.1:9222"    # e.g. chromium --remote-debugging-port=9222
```

### Session providers

A provider answers one question for `remote_cdp`: *where is this rollout's
browser, and how do I give it back?*

```python
class BrowserSessionProvider(Protocol):
    name: str
    async def acquire(self, spec: BrowserSessionSpec) -> BrowserSessionHandle: ...   # returns a live cdp_url
    async def release(self, handle: BrowserSessionHandle) -> None: ...               # idempotent
```

| Provider | What it does |
| --- | --- |
| `static_cdp` (built in) | Uses an endpoint that already exists (`cdp_url`, or `$BROWSER_CDP_URL`). Release is a no-op. |
| `lexmount` (example) | Creates one isolated cloud browser session per rollout and deletes it afterwards. See [`providers/lexmount/`](providers/lexmount/README.md) and the `interactive_browser/lexmount` config flavor. |

Providers are selected by name (`{provider_name: {kwargs}}`) and imported only
when selected, so an unused provider never pulls in its SDK. Publish one from
your own package with an entry point — no fork required:

```toml
[project.entry-points."nemo_gym.browser_session_providers"]
my_browser_cloud = "my_pkg.provider:MyProvider"
```

`register_session_provider(name, cls)` does the same thing in-process, and
`register_backend(name, builder)` adds a backend that is not CDP-shaped at all
(a vendor SDK with its own DOM API implements `BrowserBackend` directly).

## Tools & observation

Tools: `browser_navigate(url)`, `browser_click(element_id)`,
`browser_type(element_id, text)`, `browser_observe()`, `browser_finish(answer)`.
Observation is a **compact numbered list of interactive elements** (`[id] role: name`)
plus URL/title — deliberately token-cheap (raw HTML/pixels are far too expensive for
small policies and for training context length). `element_id`s come from the most
recent observation.

Element collection stops at `max_elements` (default 50) — probing each interactive
node costs several browser round-trips, so a large page is truncated rather than
fully scanned, and the observation says so.

## Reward

`verify()` scores a per-task success spec in `verifier_metadata`:
`final_url` / `url_contains` / `dom_contains` / `answer_equals`. Sparse 0/1 outcome
reward by default (least reward-hackable); extend `_score()` with new keys as needed.

A spec carrying none of those keys **raises** rather than scoring 0 — a misspelled
key would otherwise give every rollout reward 0, which looks identical to a policy
that never solves the task.

## Session lifetime

A browser is released when the rollout is scored (`verify`) or when the same
`session_id` is re-seeded. There is no independent episode TTL and no
client-side cap on concurrent sessions, so:

- a rollout abandoned without either (trainer crash, client disconnect) leaves
  its browser open until the process exits (local) or the provider reclaims it;
- with a metered provider, size the account quota **above** the rollout
  concurrency, with headroom for sessions still being torn down.

Every acquired provider session is released exactly once — including when the
CDP connect fails after the session was created, and however often `close()` is
called (`tests/test_registry_and_sessions.py`).

## Run

### 1. Backend contract tests (no GPU, no Gym serving stack)
```bash
uv run --no-project --with playwright python -m playwright install chromium
uv run --no-project --with playwright --with pytest --with pytest-asyncio python -m pytest tests -q
```
Runs one contract against **both** shipped backends: `local_playwright`, and
`remote_cdp` against a Chromium the test suite starts with
`--remote-debugging-port` — so the remote path is covered in CI without any
third-party service. Tasks are the bundled offline `site/` (deterministic, ToS-safe).

> On a bare Linux container/VM, Chromium also needs system libraries (libnss3,
> libgbm1, ...): if the install warns `Host system is missing dependencies to
> run browsers`, run (root/sudo required)
> `uv run --no-project --with playwright python -m playwright install-deps chromium`.

### 2. As a NeMo-Gym environment (rollouts / training)

Requirements for the policy endpoint — it must satisfy **both**:

1. **Speak the Responses API** (`POST /v1/responses`). Chat-completions-only
   gateways do not work; recent vLLM serves `/v1/responses` natively.
2. **Parse tool calls into structured `function_call` items.** If the server
   returns tool-call markup as plain text, the agent sees zero tool calls, the
   browser is never driven, and every rollout "succeeds" with **reward 0.0** —
   silently. For vLLM + Qwen-family models, launch with
   `--enable-auto-tool-choice --tool-call-parser hermes`.

```bash
# from the repo root:
gym env start --resources-server interactive_browser \
  --model-type openai_model --model <served-model-name> \
  --model-url https://your-endpoint/v1 --model-api-key <key> &
gym eval run --no-serve --agent interactive_browser_simple_agent \
  --input resources_servers/interactive_browser/data/example.jsonl \
  --output resources_servers/interactive_browser/data/example_rollouts.jsonl --limit 2
```
Then plug into NeMo-RL GRPO via `examples/nemo_gym/run_grpo_nemo_gym.py`.

Remote browsers cannot open this repo's offline `site/` pages (local `file://`
URIs), so remote runs use live-web tasks — `data/example_remote.jsonl`, which the
`interactive_browser/lexmount` flavor points at.

## Files (Gym `new-environment` spec)
- [x] `app.py` — resources server (seed_session + tools + verify)
- [x] `browser/` — `BrowserBackend` contract, `LocalPlaywrightBackend`, `RemoteCDPBackend` + `StaticCDPProvider`, registry
- [x] `providers/lexmount/` — example session provider (hosted browser service, optional SDK)
- [x] `configs/interactive_browser.yaml`, `configs/lexmount.yaml` (remote flavor)
- [x] `site/` — bundled offline test site (deterministic tasks/CI)
- [x] `generate_data.py` + `data/example.jsonl` (offline) + `data/example_remote.jsonl` (live web)
- [x] `tests/` — backend contract (both backends) + selection/session bookkeeping
- [x] `requirements.txt`, `README.md`
- [x] `data/example_rollouts.jsonl` — 5 rollouts collected against a Responses-API endpoint (reward 1.0 on the offline site; reward wiring exercised end-to-end)
- [ ] GRPO training-signal run (train via NeMo-RL's `examples/nemo_gym/run_grpo_nemo_gym.py`)

## Licensing
- Environment code: Apache 2.0 (matches NeMo-Gym).
- Reference backend: Playwright (Apache 2.0).
- Third-party browser-service SDKs (e.g. the `lexmount` provider): separate,
  optional dependencies installed by the operator; not bundled, and not required
  by the default backend.
