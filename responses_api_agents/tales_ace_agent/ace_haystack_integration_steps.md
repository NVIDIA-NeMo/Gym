# ACE + Haystack Memory Agent — NeMo Gym Integration

Integrates `ace_relay_trace.py` (NeMo Relay instrumented ACE agent, `/home/ubuntu/dspy/`) into
NeMo Gym as a first-class agent harness. The agent owns the full DSPy + Haystack structured memory
loop and evolving ACE playbook, calling the TALES resources server via `ServerClient` for
environment interaction.

```
responses_api_agents/tales_ace_agent/    ← this agent
resources_servers/tales/                 ← existing TALES env server (ALFWorld, TextWorld, …)
```

---

## Architecture Overview

```
ng_run / ng_test
│
├── TALESResourcesServer  (resources_servers/tales/app.py)
│     /reset → gym.make("tales/alfworld") + env.reset()
│     /step  → env.step(action) → (obs, reward, done, info)
│     expose_admissible_commands: true  ← REQUIRED for ACE
│
└── TalesAceAgent  (responses_api_agents/tales_ace_agent/app.py)
      /run  → full episode loop
        ├── asyncio.to_thread(_sync_memory_step)   DSPy MemoryUpdate + Haystack embed/recall
        ├── asyncio.to_thread(_reason_traced)      NvidiaChatGenerator + nemo_relay spans
        ├── POST /step  →  ServerClient             env transition
        └── asyncio.to_thread(_sync_reflect)       DSPy Reflect + playbook curate
```

**No model server.** The agent calls NVIDIA NIM directly. `/v1/responses` returns HTTP 501.

**Playbook persistence.** `self._playbook` (a `Playbook` instance) lives on the agent object and
survives across `run()` calls, enabling cross-episode strategy learning.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python ≥ 3.12 | Required by NeMo Gym |
| `uv` | Package manager used by `ng_test` / `ng_run` |
| Java 11+ | Required by some tale-suite environments (e.g. Jericho) |
| `NVIDIA_API_KEY` | NIM API key for all LLM + embedding calls |
| NeMo Relay wheel | `nemo_relay-0.4.0-cp311-abi3-linux_x86_64.whl` |

---

## Environment Setup

### 1. Create the main Gym venv

```bash
cd /home/ubuntu/Gym
uv venv && uv sync --extra dev --group docs
source .venv/bin/activate
```

### 2. Install Java (required for Jericho / some tale-suite envs)

```bash
sudo apt-get install -y default-jdk
```

### 3. Set up NeMo Relay source and wheel

The `nemo_relay` wheel installs only a `.pth` file pointing to `/home/ubuntu/NeMo-Relay/python`.
The actual source must exist at that path.

**3a — Copy source to the path the `.pth` expects:**
```bash
mkdir -p /home/ubuntu/NeMo-Relay/python
cp -r /home/ubuntu/dspy/.venv/lib/python3.12/site-packages/nemo_relay \
      /home/ubuntu/NeMo-Relay/python/
```

**3b — Copy wheel to stable path** (away from volatile `/ephemeral/cache/`):
```bash
cp /ephemeral/cache/uv/sdists-v9/editable/ffca03370d02b0c8/JZHaYRC7MMdzzz7r/nemo_relay-0.4.0-cp311-abi3-linux_x86_64.whl \
   /home/ubuntu/Gym/responses_api_agents/nemo_relay-0.4.0-cp311-abi3-linux_x86_64.whl
```

`requirements.txt` references this stable path:
```
../nemo_relay-0.4.0-cp311-abi3-linux_x86_64.whl
```

### 4. Set the NVIDIA API key

```bash
cp /home/ubuntu/dspy/.env /home/ubuntu/Gym/responses_api_agents/tales_ace_agent/.env
# or:
echo "NVIDIA_API_KEY=nvapi-<your-key>" > responses_api_agents/tales_ace_agent/.env
```

> **Security:** never commit `.env`.

---

## Running the Tests

```bash
cd /home/ubuntu/Gym
source .venv/bin/activate
ng_test "+entrypoint=responses_api_agents/tales_ace_agent"
```

All 34 tests are offline (heavy deps mocked — no API calls, no game engine).

---

## Running a Live Episode

> **Always run from `/home/ubuntu/Gym`** with the main Gym venv.
> Running from inside the agent directory causes port 11000 collision (two `ng_run` processes
> fight over the Ray head server port).

### One-liner: kill stale + start servers + run (recommended)

```bash
cd /home/ubuntu/Gym
bash responses_api_agents/tales_ace_agent/launch.sh
# custom output path:
bash responses_api_agents/tales_ace_agent/launch.sh results/my_run.jsonl
```

`launch.sh` kills any stale `ng_run`/Ray/tales processes, starts both servers in the background
via a single `ng_run` invocation, waits for the head server at port 11000, then fires
`ng_collect_rollouts` against `data/example.jsonl`. Servers shut down automatically on exit or
Ctrl-C.

### Manual: servers in Terminal 1, rollouts in Terminal 2

**Terminal 1 — start both servers:**
```bash
cd /home/ubuntu/Gym
source .venv/bin/activate
ng_run "+config_paths=[responses_api_agents/tales_ace_agent/configs/tales_ace_agent.yaml]" \
       "+servers=[tales_ace_agent]"
```

`ng_run` auto-starts `tales` alongside `tales_ace_agent` because the config declares
`resources_server: {name: tales}`.

**Terminal 2 — collect rollouts:**
```bash
cd /home/ubuntu/Gym
source .venv/bin/activate
ng_collect_rollouts \
    "+input_jsonl_fpath=responses_api_agents/tales_ace_agent/data/example.jsonl" \
    "+output_jsonl_fpath=results/tales_ace_example.jsonl"
```

`ng_collect_rollouts` connects to the head server at `127.0.0.1:11000` and fetches server URLs
from it — no `+config_paths` needed.

### Stopping all servers

```bash
pkill -f "ng_run\|tales_ace_agent\|tales.*app\.py\|uvicorn.*tales\|ray"
sleep 2
pkill -9 -f "ng_run\|tales_ace_agent\|tales.*app\.py\|uvicorn.*tales\|ray" 2>/dev/null
pgrep -a -f "ng_run\|tales_ace\|uvicorn.*tales\|ray" || echo "all clear"
```

---

## Configuration Reference

### `configs/tales_ace_agent.yaml`

```yaml
tales:
  resources_servers:
    tales:
      entrypoint: app.py
      expose_admissible_commands: true   # REQUIRED — ACE reads admissible_commands each step
      framework: alfworld
      task_no: 0
      seed: 0
      split: train
      max_episode_steps: 50

tales_ace_agent:
  responses_api_agents:
    tales_ace_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: tales
      max_steps: 50                      # episode step cap; truncated=True if reached
      trace_dir: traces                  # ATIF + ATOF output directory
      datasets:
      - name: example
        type: example
        jsonl_fpath: responses_api_agents/tales_ace_agent/data/example.jsonl
        num_repeats: 1
        license: Apache 2.0
```

| Field | Default | Effect |
|---|---|---|
| `expose_admissible_commands` | **must be `true`** | TALES server strips admissible_commands when false; agent cannot function |
| `max_steps` | `50` | Episode step cap |
| `trace_dir` | `traces` | Root for trace output; subdirs `atif/` and `atof/` created on startup |

---

## Models Used

All calls go to NVIDIA NIM (`https://integrate.api.nvidia.com/v1`).

| Role | Model | Used for |
|---|---|---|
| Reasoner | `nvidia/nemotron-3-ultra-550b-a55b` | Tool-call action selection per step (Haystack NvidiaChatGenerator) |
| Summarizer / Reflector | `nvidia/llama-3.3-nemotron-super-49b-v1` | DSPy MemoryUpdate + Reflect (DSPy LM) |
| Embedder | `nvidia/nv-embedqa-e5-v5` | Document + query embedding for Haystack InMemoryDocumentStore |

---

## NeMo Relay Tracing

The agent emits structured trace events via `nemo_relay` v0.4.0.

```
traces/
  atif/
    episode-<uuid8>.json     # one structured trajectory per episode (ATIF v1.7)
  atof/
    events.jsonl             # append-mode NDJSON event stream across all runs
```

**Episode scope events**

| Event | Fired | Key fields |
|---|---|---|
| `episode_goal` | start of episode | `goal`, `common_sense_hint`, `framework`, `task_no`, `split` |
| `playbook_injected` | after goal | `bullet_count`, `playbook` |
| `agent_step` | after every `env.step()` | `step`, `observation`, `action_taken`, `progress`, `plan`, `recalled_facts`, `dead_ends`, `result_obs`, `reward`, `is_done` |
| `episode_outcome` | end of episode | `label`, `goal`, `outcome`, `steps`, `total_reward`, `trajectory_tail` |
| `reflection_summary` | after reflector | `goal`, `outcome`, `steps_taken`, `helpful_strategy_ids`, `harmful_strategy_ids`, `next_best_strategies`, `playbook_size_before`, `current_playbook` |
| `playbook_updated` | after curate | `playbook_size_after`, `bullets_delta`, `playbook_snapshot` |

**LLM lifecycle events** (via `nemo_relay.llm.call` / `call_end`)

| Provider | Fired when |
|---|---|
| `memory-updater` | Before/after every DSPy `MemoryUpdate` predict call |
| `reasoner` | Before/after every `NvidiaChatGenerator.run()` attempt |
| `reflector` | Before/after every DSPy `Reflect` predict attempt |

**RPM / failure events**

| Event | Key fields |
|---|---|
| `llm_rate_limit_hit` | `provider`, `model`, `step`, `attempt`, `wait_seconds`, `error` |
| `reasoner_fallback` | `step`, `fallback_action`, `last_error` |

---

## Rate Limiting

| Layer | Mechanism | Where |
|---|---|---|
| **Proactive** | `_throttle()` — sliding 28-RPM window, fires before each NVIDIA API call | `_sync_memory_step`, `_reason_traced`, `_sync_reflect` |
| **Reactive** | RPM retry loop — catches 429 / "rate limit" / "too many requests" / "rpm" | `_reason_traced`, `_sync_reflect` |

```python
_RPM_WAIT_BASE    = 15.0   # seconds (conservative floor)
_RPM_MAX_RETRIES  = 5
_RPM_WAIT_CAP     = 60.0   # doubles each attempt up to this cap
```

On `_reason_traced` exhaustion: heuristic fallback action chosen; `reasoner_fallback` event fired.
On `_sync_reflect` exhaustion: `None` returned; playbook update skipped for that episode.

---

## ACE Playbook — Cross-Episode Learning

The `Playbook` object lives on the agent instance (`self._playbook`) and persists across all
`run()` calls in a session.

**Per-episode cycle:**

1. **Inject** — rendered playbook prepended to reasoner prompt before each step.
2. **Reflect** — after episode ends, `Reflect` DSPy signature extracts `helpful_ids`,
   `harmful_ids`, and up to 3 `new_strategies`.
3. **Curate** (`_curate`) — bumps counters, adds new strategies (deduped by Jaccard > 0.6,
   max 12), prunes items where `harmful - helpful >= 2` and `helpful == 0`.

---

## Memory System (DSPy + Haystack)

Per-episode, per-step cycle:

1. **MemoryUpdate** (DSPy `Predict`) — extracts `new_fact`, `dead_end`, `progress`, `plan`.
2. **Grounding** — `ground_fact()` rejects facts whose named objects don't appear verbatim in the observation.
3. **Embed + store** — grounded facts embedded via `NvidiaDocumentEmbedder` → `InMemoryDocumentStore`.
4. **Recall** — goal + plan + observation embedded; top-10 nearest facts retrieved via `InMemoryEmbeddingRetriever`.
5. **Dead-end tracking** — verified noops added to `mem.dead_ends`; filtered from offered actions before reasoner call.

---

## File Tree

```
responses_api_agents/tales_ace_agent/
├── __init__.py
├── app.py                             # ~460 lines — Gym agent (self-contained port)
├── haystack_memory.py                 # standalone: rung 2 — DSPy + Haystack memory
├── ace_alfworld.py                    # standalone: rung 3 — ACE playbook optimizer
├── ace_relay_trace.py                 # standalone: rung 3 + NeMo Relay tracing
├── ace_haystack_integration_steps.md  # this file
├── launch.sh                          # one-liner: kill + start + run
├── configs/
│   └── tales_ace_agent.yaml           # Hydra config (TALES server + agent)
├── data/
│   └── example.jsonl                  # 3 ALFWorld example tasks
├── requirements.txt                   # dspy, haystack, nvidia-haystack, nemo_relay wheel
├── tests/
│   ├── __init__.py
│   └── test_app.py                    # 34 unit + integration tests (offline, all mocked)
└── .env                               # NVIDIA_API_KEY  ← DO NOT COMMIT

responses_api_agents/
└── nemo_relay-0.4.0-cp311-abi3-linux_x86_64.whl   # stable wheel copy
```

The three standalone scripts (`haystack_memory.py`, `ace_alfworld.py`, `ace_relay_trace.py`) are
the original development rungs ported to this directory. They import each other locally
(`ace_alfworld` → `haystack_memory`; `ace_relay_trace` → both) and share the `.env` file in
this directory. `app.py` does **not** import from them — it is a self-contained Gym integration
that avoids the module-level side effects present in the standalone scripts.

---

## Common Issues

### `ModuleNotFoundError: No module named 'nemo_relay'`

Two distinct causes — check both:

**Cause A — wheel not at stable path:**
```bash
ls /home/ubuntu/Gym/responses_api_agents/nemo_relay-0.4.0-cp311-abi3-linux_x86_64.whl
```
If absent, re-run setup step 3b.

**Cause B — `.pth` source directory missing** (most common after reboot):
```bash
ls /home/ubuntu/NeMo-Relay/python/nemo_relay/__init__.py
```
If absent, re-run setup step 3a.

Quick verify:
```bash
source responses_api_agents/tales_ace_agent/.venv/bin/activate
python -c "import nemo_relay; print('OK')"
```
If still failing, delete the agent venv and let `ng_run` recreate it:
```bash
rm -rf responses_api_agents/tales_ace_agent/.venv
```

### `[Errno 98] address already in use` on port 11000

Stale `ng_run` or Ray process from a previous run. Use `launch.sh` (handles this automatically)
or kill manually:
```bash
pkill -f "ng_run\|tales_ace_agent\|tales.*app\.py\|uvicorn.*tales\|ray"
sleep 2
pkill -9 -f "ng_run\|tales_ace_agent\|tales.*app\.py\|uvicorn.*tales\|ray" 2>/dev/null
```

### `admissible_commands` empty / agent always picks `look`

TALES server running with default `expose_admissible_commands: false`. Must use
`tales_ace_agent.yaml` config which sets it to `true`.

### 429 / rate limit errors

Expected behavior — the reactive retry loop handles them automatically. If they persist,
lower `_RPM_CAP` in `app.py` (line ~68) below 28.

### `load_dotenv` not loading key

`.env` must be in `responses_api_agents/tales_ace_agent/`. The agent loads it relative to
`__file__`, not the working directory.
