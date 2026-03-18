# Browser Gym Integration for NeMo-Gym

## Overview

This integration adds Browser Agent support to NeMo-Gym, enabling LLMs to interact with web applications through visual observations (screenshots) and browser actions (click, type, scroll, etc.). The system captures full trajectories suitable for RL training.

### Architecture

The integration follows NeMo-Gym's three-server architecture:

All three providers follow the same unified flow: **Agent → Adapter → Model Server → External API**. Each provider has a dedicated adapter (in the agent layer) that handles context management and action mapping, and a dedicated model server that acts as a stateless API proxy handling authentication, retries, and transport.

- **OpenAI**: Adapter manages server-side context via `previous_response_id`; `openai_model` server proxies to `api.openai.com`
- **Anthropic**: Adapter manages client-side context (turn-based trimming, tool pair validation, screenshot GC); `anthropic_model` server proxies to Anthropic API
- **Gemini**: Adapter manages client-side context (paired-turn trimming, function pair validation, screenshot GC); `gemini_model` server proxies to Google Gemini API

```
                              EXTERNAL LLM PROVIDERS
                  ┌──────────────────────────────────────────┐
                  │   OpenAI      Anthropic       Google     │
                  │   (CUA         (Sonnet/        (Gemini   │
                  │   Preview)     Opus)           2.5 CU)   │
                  └─────┬──────────┬──────────────────┬──────┘
                        │          │                  │
                        │          │                  │
┌───────────────────────┼──────────┼──────────────────┼───────────────┐
│  NEMO-GYM             │          │                  │               │
│                       │          │                  │               │
│  ┌────────────────────┼──────────┼──────────────────┼───────────┐   │
│  │  ng_collect_rollouts          │                  │           │   │
│  │  (CLI orchestrator)           │                  │           │   │
│  └───────────────┬───────────────┼──────────────────┼───────────┘   │
│                  │ /run          │                  │               │
│                  ▼               │                  │               │
│  ┌───────────────────────────────┼──────────────────┼───────────┐   │
│  │         BROWSER AGENT SERVER  │                  │           │   │
│  │   responses_api_agents/browser_agent             │           │   │
│  │                               │                  │           │   │
│  │  ┌────────────────────────────┴──────────────────┴───────┐   │   │
│  │  │  Adapters (unified flow for all providers)            │   │   │
│  │  │                                                       │   │   │
│  │  │  ┌─────────────┐ ┌──────────────┐ ┌──────────────┐    │   │   │
│  │  │  │ OpenAI      │ │ Anthropic    │ │ Gemini       │    │   │   │
│  │  │  │ • Server-   │ │ • Client-    │ │ • Client-    │    │   │   │
│  │  │  │   side ctx  │ │   side ctx   │ │   side ctx   │    │   │   │
│  │  │  │ • prev_resp │ │ • Turn trim  │ │ • Pair trim  │    │   │   │
│  │  │  │   _id chain │ │ • Tool valid │ │ • Func valid │    │   │   │
│  │  │  └──────┬──────┘ └──────┬───────┘ └──────┬───────┘    │   │   │
│  │  │         └───────────────┼────────────────┘            │   │   │
│  │  │                         │ unified BrowserAction       │   │   │
│  │  └─────────────────────────┼─────────────────────────────┘   │   │
│  │                            │                                 │   │
│  │    ┌───────────────────────┴────────────────────────────┐    │   │
│  │    │  CUA Loop:                                         │    │   │
│  │    │   1. /seed_session → open browser at start_url     │    │   │
│  │    │   2. /step → execute action, return screenshot     │    │   │
│  │    │   3. Get next action (via adapter → model server)  │    │   │
│  │    │   4. Repeat 2-3 until done or max_steps            │    │   │
│  │    │   5. /dump_local_storage → capture LS              │    │   │
│  │    │   6. /close → tear down browser                    │    │   │
│  │    └────────────────────────────────────────────────────┘    │   │
│  └──────────┬────────────────────────────────┬────────────────────┘   │
│             │ /v1/responses                  │ /seed_session          │
│             │                                │ /step, /close          │
│             │                                │ /dump_local_storage    │
│             │                                │ /verify                │
│             ▼                                ▼                        │
│  ┌──────────────────────────────┐  ┌─────────────────────────────┐   │
│  │  MODEL SERVERS               │  │  BROWSER GYM RESOURCE SERVER│   │
│  │                              │  │  resources_servers/browser_ │   │
│  │  ┌────────────────────────┐  │  │  gym                        │   │
│  │  │ openai_model           │  │  │                             │   │
│  │  │ • api.openai.com       │  │  │  ┌───────────────────────┐  │   │
│  │  │ • Token tracking       │  │  │  │   BrowserPool         │  │   │
│  │  │ • Retry/backoff        │  │  │  │   (Playwright         │  │   │
│  │  └────────────────────────┘  │  │  │    Chromium,          │  │   │
│  │                              │  │  │    Semaphore bounded)  │  │   │
│  │  ┌────────────────────────┐  │  │  └───────────────────────┘  │   │
│  │  │ anthropic_model        │  │  │                             │   │
│  │  │ • Stateless API proxy  │  │  │  ┌───────────────────────┐  │   │
│  │  │ • AsyncAnthropic       │  │  │  │    Verification       │  │   │
│  │  └────────────────────────┘  │  │  │  taskId + LS dump     │  │   │
│  │                              │  │  │         │             │  │   │
│  │  ┌────────────────────────┐  │  │  │         ▼             │  │   │
│  │  │ gemini_model           │  │  │  │  POST /api/v1/        │──┼───┼──┐
│  │  │ • Stateless API proxy  │  │  │  │   get_actual_state    │  │   │  │
│  │  │ • genai Client         │  │  │  │         │             │  │   │  │
│  │  └────────────────────────┘  │  │  │         ▼             │  │   │  │
│  │                              │  │  │  assertions → reward  │  │   │  │
│  │                              │  │  │  (0.0 or 1.0)         │  │   │  │
│  └──────────────────────────────┘  │  └───────────────────────┘  │   │  │
│                                    └─────────────────────────────┘   │  │
└──────────────────────────────────────────────────────────────────────┘  │
                                                                         │
                   ┌─────────────────────────────────────┐               │
                   │     GYM ENVIRONMENTS (external)     │◄──────────────┘
                   │                                     │
                   │  SpotHub    DeskZen    PriceLens    │
                   │                                     │
                   │  Each gym exposes:                  │
                   │  - Web app UI (browser target)      │
                   │  - POST /api/v1/get_expected_state  │
                   │    (task discovery for gym URLs)    │
                   │  - POST /api/v1/get_actual_state    │
                   │    (verification via LS assertions) │
                   └─────────────────────────────────────┘
```

**Key points:**
- **All providers** follow the same unified flow: Adapter → Model Server → External API. No direct API calls are made from the agent layer. Each adapter receives an injected `api_caller` that routes requests through the model server.
- **OpenAI** adapter manages server-side context via `previous_response_id` and maps `computer_call` / `computer_call_output` items to `BrowserAction`. The `openai_model` server proxies to `api.openai.com` with token tracking and retry logic.
- **Anthropic** adapter manages full conversation history with turn-based trimming, tool pair validation, and screenshot memory GC. The `anthropic_model` server acts as a stateless API proxy.
- **Gemini** adapter manages full conversation history with paired-turn trimming, function pair validation, and screenshot GC. The `gemini_model` server acts as a stateless API proxy.
- The resource server manages browser lifecycle **and** verification in one process.
- Gym environments are external -- they can be remote (deployed URLs) or local.

### NeMo-Gym Layer Communication

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NeMo-Gym Layers                              │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  CLI / Orchestrator Layer                                     │  │
│  │  ng_collect_rollouts / ng_reward_profile                      │  │
│  │                                                               │  │
│  │  • Reads input JSONL (tasks) or fetches from gym URL          │  │
│  │  • Dispatches /run requests to agent servers                  │  │
│  │  • Writes output JSONL (rollouts with rewards)                │  │
│  └───────────────────────────┬───────────────────────────────────┘  │
│                              │                                      │
│                    POST /run │ (task prompt + verifier_metadata)    │
│                              │ Returns: CUAVerifyResponse           │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Agent Layer  (responses_api_agents/browser_agent)            │  │
│  │                                                               │  │
│  │  Orchestrates the full CUA loop:                              │  │
│  │   • Calls resource server to manage browser                   │  │
│  │   • Uses adapter (all providers) → routes through model srv   │  │
│  │   • Adapter handles context mgmt, action mapping, trimming    │  │
│  │   • Builds trajectory (screenshots, actions, provider data)   │  │
│  │   • Calls resource server to verify at end                    │  │
│  └───────┬────────────────────────────────┬──────────────────────┘  │
│          │                                │                         │
│          │ POST /v1/responses             │ POST /seed_session      │
│          │ (screenshot + context)         │ POST /step              │
│          │ Returns: NeMoGymResponse       │ POST /dump_local_storage│
│          │                                │ POST /close             │
│          ▼                                │ POST /verify            │
│  ┌─────────────────────────────┐          ▼                         │
│  │  Model Layer                │  ┌──────────────────────────────┐  │
│  │                             │  │  Resource Layer              │  │
│  │  ┌────────────────────────┐ │  │  (resources_servers/         │  │
│  │  │  openai_model          │ │  │   browser_gym)               │  │
│  │  │  • api.openai.com      │ │  │                              │  │
│  │  │  • Token tracking      │ │  │  • BrowserPool (Playwright)  │  │
│  │  │  • Retry with backoff  │ │  │  • Action execution          │  │
│  │  │  • Org header          │ │  │  • Screenshot capture        │  │
│  │  └────────────────────────┘ │  │  • localStorage dump         │  │
│  │                             │  │  • Verification via gym API  │  │
│  │  ┌────────────────────────┐ │  └──────────────────┬───────────┘  │
│  │  │  anthropic_model       │ │                     │              │
│  │  │  • Stateless API proxy │ │                     │              │
│  │  │  • AsyncAnthropic      │ │                     │              │
│  │  └────────────────────────┘ │                     │              │
│  │                             │                     │              │
│  │  ┌────────────────────────┐ │                     │              │
│  │  │  gemini_model          │ │                     │              │
│  │  │  • Stateless API proxy │ │                     │              │
│  │  │  • genai.Client        │ │                     │              │
│  │  │  • asyncio.to_thread   │ │                     │              │
│  │  └────────────────────────┘ │                     │              │
│  └─────────┬───────────────────┘                     │              │
│            │                                         │              │
└────────────┼─────────────────────────────────────────┼──────────────┘
             │                                         │
             │ HTTPS                                   │ HTTPS
             ▼                                         ▼
    ┌──────────────────────┐              ┌──────────────────────┐
    │  LLM APIs            │              │  Gym Environments    │
    │                      │              │  (SpotHub, DeskZen,  │
    │  • OpenAI            │              │   PriceLens, etc.)   │
    │    /v1/responses     │              │                      │
    │  • Anthropic         │              │  /api/v1/            │
    │    /v1/messages      │              │   get_actual_state   │
    │  • Google Gemini     │              │                      │
    │    generate_content  │              │                      │
    └──────────────────────┘              └──────────────────────┘
```

**Communication summary:**

| From | To | Endpoint | Payload | Response |
|------|----|----------|---------|----------|
| CLI | Agent | `POST /run` | task prompt + verifier_metadata | CUAVerifyResponse (reward + trajectory) |
| Agent | OpenAI Model | `POST /v1/responses` | screenshot + input items | NeMoGymResponse (computer_call actions) |
| Agent | Anthropic Model | `POST /v1/responses` | pre-built Anthropic API params (messages, tools, betas) | raw Anthropic response dict |
| Agent | Gemini Model | `POST /v1/responses` | serialized contents + config | serialized Gemini response dict |
| Agent | Resource | `POST /seed_session` | start_url, viewport_width, viewport_height | env_id + initial screenshot |
| Agent | Resource | `POST /step` | env_id + BrowserAction | screenshot + current_url |
| Agent | Resource | `POST /dump_local_storage` | env_id | localStorage JSON string |
| Agent | Resource | `POST /close` | env_id | status |
| Agent | Resource | `POST /verify` | response + verifier_metadata + localStorage | reward (0.0 or 1.0) |
| CLI | Gym | `POST /api/v1/get_expected_state` | (none) | verifiers dict (task IDs, prompts, assertions) |
| Resource | Gym | `POST /api/v1/get_actual_state` | taskId + localStorageDump | assertions array |

### Data Flow

1. Agent receives task prompt + `verifier_metadata` (task_id, gym_url, start_url)
2. Agent seeds a browser session via resource server (navigates to start_url)
3. Agent enters CUA loop (same flow for all providers):
   - Adapter receives screenshot → manages context (provider-specific: OpenAI uses server-side `previous_response_id`; Anthropic/Gemini maintain client-side conversation history with trimming, validation, and screenshot GC) → prepares API params → routes through model server (stateless proxy) via injected `api_caller` → parses response → maps provider-specific actions to unified `BrowserAction` → executes via resource server `/step` → repeat
4. On completion (model says "done" or max_steps reached), agent dumps localStorage from the browser
5. Agent closes the browser session
6. Resource server verifies by sending localStorage to the gym's `/api/v1/get_actual_state` endpoint
7. Reward is 1.0 if all assertions pass, 0.0 otherwise

### Provider Support

| Provider | Execution Path | Context Management | Token Tracking |
|----------|---------------|-------------------|----------------|
| **OpenAI** (CUA Preview) | Adapter → model server (`openai_model`) | Server-side via `previous_response_id` | Yes (via `NeMoGymResponseUsage`) |
| **Anthropic** (Sonnet/Opus) | Adapter → model server (stateless proxy) | Client-side in adapter: turn-based trimming, tool pair validation, screenshot GC | Yes (via usage in response) |
| **Gemini** (2.5 Computer Use) | Adapter → model server (stateless proxy) | Client-side in adapter: paired-turn trimming, function pair validation, screenshot GC | Yes (via usage_metadata in response) |

## File Structure

```
resources_servers/browser_gym/
├── app.py                          # Resource server (seed, step, verify, close endpoints)
├── browser_pool.py                 # Playwright browser lifecycle + action execution
├── schemas.py                      # All shared Pydantic schemas
├── setup_playwright.py             # Auto-installs Chromium on first startup
├── configs/browser_gym.yaml        # Server + agent YAML config
├── data/
│   ├── example.jsonl               # Example tasks (committed to git)
│   └── .gitignore
├── tests/
│   ├── conftest.py
│   └── test_app.py
├── requirements.txt
└── README.md                       # This file

responses_api_models/anthropic_model/
├── app.py                          # Anthropic model server (stateless API proxy, /v1/responses)
└── requirements.txt

responses_api_models/gemini_model/
├── app.py                          # Gemini model server (stateless API proxy, /v1/responses)
└── requirements.txt

responses_api_agents/browser_agent/
├── app.py                          # Agent server (responses + run endpoints)
├── trajectory_writer.py            # Debug trajectory output (screenshots + JSON)
├── adapters/
│   ├── __init__.py                 # AdapterFactory registry
│   ├── base.py                     # BaseCUAAdapter abstract class
│   ├── openai_adapter.py           # OpenAI Computer Use Preview (context mgmt, action mapping, API routing)
│   ├── anthropic_adapter.py        # Anthropic Sonnet / Opus (context mgmt, trimming, API routing)
│   └── gemini_adapter.py           # Google Gemini (context mgmt, paired-turn trimming, API routing)
├── tests/
│   ├── __init__.py
│   └── test_app.py
└── requirements.txt
```

## Prerequisites

1. **Install `uv`** (NeMo-Gym uses `uv` to manage virtual environments for each server):
   ```bash
   # macOS
   brew install uv

   # Linux / macOS (alternative)
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Or via pip (any platform)
   pip install uv
   ```

2. **Python 3.12+** and **NeMo-Gym** installed:
   ```bash
   uv venv && uv sync --extra dev
   source .venv/bin/activate
   ```
   After activating the virtual environment, CLI commands like `ng_run`, `ng_collect_rollouts`, etc. become available.
   All BrowserGym-specific dependencies (`playwright`, `anthropic`, `Pillow`, etc.) are declared in
   each server's `requirements.txt` and installed automatically by `ng_run` when it creates per-server
   virtual environments. Chromium is also auto-installed on first server startup via `ensure_playwright()`.

3. **API keys and settings** in `env.yaml` (project root):
   ```yaml
   cua_openai_api_key: "sk-proj-..."
   cua_openai_org: "org-..."
   cua_anthropic_api_key: "sk-ant-..."
   cua_gemini_api_key: "AIza..."

   # Debug trajectories (screenshots + JSON saved to results/cua_debug_trajectories/)
   cua_debug_trajectories: false

   # Optional: override default concurrent rollouts (default: 10)
   num_samples_in_parallel: 10

   # Browser concurrency
   max_concurrent_browsers: 16   # max simultaneous browser sessions (semaphore cap)
   browser_pool_size: 4          # number of separate Chromium processes
   ```

   `max_concurrent_browsers` controls how many browser sessions can exist at once (the semaphore cap). `browser_pool_size` controls how many separate Chromium OS processes share that load — if one crashes, only the sessions on that process are affected.

4. **Gym environments** deployed and accessible (e.g. `https://your-gym-url.com`)

## Running

### Start All Servers

```bash
ng_run "+config_paths=[resources_servers/browser_gym/configs/browser_gym.yaml]"
```

This starts 9 servers:
- 1 resource server (browser management + verification)
- 4 model servers (OpenAI CUA, Anthropic Sonnet, Anthropic Opus, Gemini)
- 4 agent servers (OpenAI, Anthropic Sonnet, Anthropic Opus, Gemini)

First run is slow (~2-3 min) because it creates isolated venvs and downloads Chromium. Subsequent runs are faster with:

```bash
ng_run "+config_paths=[resources_servers/browser_gym/configs/browser_gym.yaml]" +skip_venv_if_present=true
```

### Check Server Health

```bash
ng_status
```

### Collect Rollouts

Tasks can be loaded from either a **local JSONL file** or directly from a **gym URL** (via the gym's `/api/v1/get_expected_state` endpoint). The gym URL approach requires no data preparation -- just point to the gym and go.

Debug trajectories (screenshots, `conversation.json`, `verification.json`) are saved to `results/cua_debug_trajectories/` when `cua_debug_trajectories: true` is set in `env.yaml` (disabled by default). This path is relative to the project root and lives alongside other output artifacts in the gitignored `results/` directory. All agents share this single setting.

#### From a Gym URL (Recommended)

Instead of preparing a JSONL file, you can point `ng_collect_rollouts` directly at a gym URL. It will call the gym's `/api/v1/get_expected_state` endpoint to fetch all available tasks and automatically convert them to the standard input format.

**OpenAI (Computer Use Preview) -- all tasks:**

```bash
ng_collect_rollouts \
  +agent_name=browser_openai_agent \
  +input_gym_url=https://your-gym-url.com \
  +output_jsonl_fpath=results/cua_rollouts_openai.jsonl \
  +num_repeats=5 \
  +num_samples_in_parallel=10 \
  "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}"
```

**Anthropic Claude Sonnet -- all tasks:**

```bash
ng_collect_rollouts \
  +agent_name=browser_anthropic_sonnet_agent \
  +input_gym_url=https://your-gym-url.com \
  +output_jsonl_fpath=results/cua_rollouts_anthropic_sonnet.jsonl \
  +num_repeats=5 \
  +num_samples_in_parallel=10 \
  "+responses_create_params={max_output_tokens: 4096, temperature: 1.0}"
```

**Anthropic Claude Opus -- all tasks:**

```bash
ng_collect_rollouts \
  +agent_name=browser_anthropic_opus_agent \
  +input_gym_url=https://your-gym-url.com \
  +output_jsonl_fpath=results/cua_rollouts_anthropic_opus.jsonl \
  +num_repeats=5 \
  +num_samples_in_parallel=10 \
  "+responses_create_params={max_output_tokens: 4096, temperature: 1.0}"
```

**Gemini 2.5 Computer Use -- all tasks:**

```bash
ng_collect_rollouts \
  +agent_name=browser_gemini_agent \
  +input_gym_url=https://your-gym-url.com \
  +output_jsonl_fpath=results/cua_rollouts_gemini.jsonl \
  +num_repeats=5 \
  +num_samples_in_parallel=10 \
  "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}"
```

**Run specific task(s) by ID (any agent):**

```bash
# Single task:
ng_collect_rollouts \
  +agent_name=browser_openai_agent \
  +input_gym_url=https://your-gym-url.com \
  "+input_gym_task_id=[YOUR-TASK-ID]" \
  +output_jsonl_fpath=results/cua_rollouts_single.jsonl \
  +num_repeats=3 \
  "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}"

# Multiple tasks:
ng_collect_rollouts \
  +agent_name=browser_openai_agent \
  +input_gym_url=https://your-gym-url.com \
  "+input_gym_task_id=[TASK-ID-001,TASK-ID-002,TASK-ID-003]" \
  +output_jsonl_fpath=results/cua_rollouts_subset.jsonl \
  +num_repeats=3 \
  "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}"
```

If any task ID is not found, the command errors with the missing IDs and a list of available ones.

> **Tip:** `num_samples_in_parallel` controls how many rollouts run concurrently (default: 10). Set it to match or stay below the `max_concurrent_browsers` value in the resource server config (default: 16). Override via CLI (`+num_samples_in_parallel=20`) or set a persistent default in `env.yaml`.

**Gym URL parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `input_gym_url` | Yes | Base URL of the gym (e.g. `https://your-gym-url.com`) |
| `input_gym_task_id` | No | Run only specific task ID(s). Use list syntax: `"[ID1,ID2]"`. Errors if any are not found. |

The gym's `/api/v1/get_expected_state` response is expected to return:

```json
{
  "verifiers": {
    "TASK-ID-001": {
      "prompt": "Task description...",
      "assertions": [...]
    }
  }
}
```

Each task's `prompt` (or `task_statement`) becomes the user message, and the task ID key becomes the `verifier_metadata.task_id`. Fields `start_url` and `viewport_size` are used if present in the response, otherwise they default to the gym URL and 1280x720 respectively.

> **Note:** `input_jsonl_fpath` and `input_gym_url` are mutually exclusive -- use one or the other.

#### From a Local JSONL File

**OpenAI (Computer Use Preview):**

```bash
ng_collect_rollouts \
  +agent_name=browser_openai_agent \
  +input_jsonl_fpath=resources_servers/browser_gym/data/example.jsonl \
  +output_jsonl_fpath=results/cua_rollouts_openai.jsonl \
  +num_repeats=1 \
  "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}"
```

**Anthropic Claude Sonnet:**

```bash
ng_collect_rollouts \
  +agent_name=browser_anthropic_sonnet_agent \
  +input_jsonl_fpath=resources_servers/browser_gym/data/example.jsonl \
  +output_jsonl_fpath=results/cua_rollouts_anthropic_sonnet.jsonl \
  +num_repeats=1 \
  "+responses_create_params={max_output_tokens: 4096, temperature: 1.0}"
```

**Anthropic Claude Opus:**

```bash
ng_collect_rollouts \
  +agent_name=browser_anthropic_opus_agent \
  +input_jsonl_fpath=resources_servers/browser_gym/data/example.jsonl \
  +output_jsonl_fpath=results/cua_rollouts_anthropic_opus.jsonl \
  +num_repeats=1 \
  "+responses_create_params={max_output_tokens: 4096, temperature: 1.0}"
```

**Gemini 2.5 Computer Use:**

```bash
ng_collect_rollouts \
  +agent_name=browser_gemini_agent \
  +input_jsonl_fpath=resources_servers/browser_gym/data/example.jsonl \
  +output_jsonl_fpath=results/cua_rollouts_gemini.jsonl \
  +num_repeats=1 \
  "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}"
```

Available agent names: `browser_openai_agent`, `browser_anthropic_sonnet_agent`, `browser_anthropic_opus_agent`, `browser_gemini_agent`

> **Note:** Debug trajectories are disabled by default (`cua_debug_trajectories: false` in `env.yaml`). To enable, set `cua_debug_trajectories: true` in `env.yaml` and restart the servers. This is a server-side config — it takes effect at server startup.

### Reward Profiling

```bash
ng_reward_profile \
  +input_jsonl_fpath=resources_servers/browser_gym/data/example.jsonl \
  +rollouts_jsonl_fpath=results/cua_rollouts.jsonl \
  +output_jsonl_fpath=results/cua_profiled.jsonl \
  +materialized_inputs_jsonl_fpath=results/cua_rollouts_materialized_inputs.jsonl \
  +pass_threshold=1.0
```

### Viewing Results

**Agent-level aggregate (overall pass@1, pass@k):**

```bash
python scripts/print_aggregate_results.py +jsonl_fpath=results/cua_rollouts_reward_profiling.jsonl
```

**Per-task breakdown:**

```bash
python -c "
import json
with open('results/cua_rollouts_reward_profiling.jsonl') as f:
    for line in f:
        data = json.loads(line)
        sample = data.get('sample', {})
        prompt = sample.get('responses_create_params', {}).get('input', [{}])[-1]
        task_text = prompt.get('content', '') if isinstance(prompt, dict) else str(prompt)
        task_id = sample.get('verifier_metadata', {}).get('task_id', 'unknown')
        print(f'Task: {task_id}')
        print(f'  Prompt:    {task_text[:100]}')
        print(f'  pass@1:    {data.get(\"mean/reward\", \"N/A\")}')
        print(f'  pass@k:    {data.get(\"max/reward\", \"N/A\")}')
        print(f'  min:       {data.get(\"min/reward\", \"N/A\")}')
        print(f'  median:    {data.get(\"median/reward\", \"N/A\")}')
        print(f'  std:       {data.get(\"std/reward\", \"N/A\")}')
        print()
"
```

**Agent-level metrics directly:**

```bash
python -c "
import json
with open('results/cua_rollouts_agent_metrics.json') as f:
    for agent in json.loads(f.read()):
        print(f'Agent: {agent[\"agent_ref\"][\"name\"]}')
        print(f'  pass@1 (mean/reward): {agent.get(\"mean/reward\", \"N/A\")}')
        print(f'  pass@k (max/reward):  {agent.get(\"max/reward\", \"N/A\")}')
        print(f'  min/reward:           {agent.get(\"min/reward\", \"N/A\")}')
        print(f'  median/reward:        {agent.get(\"median/reward\", \"N/A\")}')
        print(f'  std/reward:           {agent.get(\"std/reward\", \"N/A\")}')
"
```

**Metrics reference:**

| Metric | Meaning |
|--------|---------|
| `mean/reward` | **pass@1** -- probability of solving the task in a single attempt |
| `max/reward` | **pass@k** -- solved at least once across all repeats |
| `min/reward` | 1.0 only if every single repeat passed |
| `median/reward` | Middle value across repeats |
| `std/reward` | Variance across repeats (lower = more consistent) |

## Debug Trajectory Output

Enable debug mode to save screenshots as PNGs and structured JSON for visual inspection.

### Option 1: Set in `env.yaml` (recommended)

In `env.yaml` (project root), set:

```yaml
cua_debug_trajectories: true
```

Then restart the servers. All agents will save debug trajectories.

### Option 2: Via CLI override on `ng_run`

```bash
ng_run "+config_paths=[resources_servers/browser_gym/configs/browser_gym.yaml]" ++cua_debug_trajectories=true
```

### Output Structure

```
results/cua_debug_trajectories/<env_id>/
├── screenshots/
│   ├── 00_initial.png
│   ├── 01_after.png
│   ├── 02_after.png
│   └── ...
├── conversation.json       # Full agent interaction (actions, URLs, raw provider responses — no base64)
└── verification.json       # Reward, assertions, localStorage dump, verifier_metadata
```

### Extract Debug Output from Existing Rollouts

If you already have rollout output and want to extract screenshots:

```bash
python -c "
import json, base64, os
with open('results/cua_rollouts.jsonl') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        resp = data.get('response', {})
        traj = resp.get('trajectory', {})
        out_dir = f'results/cua_debug_trajectories/rollout_{i+1}/screenshots'
        os.makedirs(out_dir, exist_ok=True)
        for j, step in enumerate(traj.get('steps', []), 1):
            ss = step.get('screenshot_after', '')
            if ss:
                raw = ss.split(',', 1)[-1] if ss.startswith('data:') else ss
                with open(f'{out_dir}/{j:02d}_after.png', 'wb') as img:
                    img.write(base64.b64decode(raw))
        print(f'Extracted rollout {i+1}: {len(traj.get(\"steps\", []))} screenshots')
"
```

## Debugging with VS Code / Cursor

Create `.vscode/launch.json` in the project root with the following content (replace the API key and org in the model server config with your values from `env.yaml`):

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Browser Gym Resource Server",
            "type": "debugpy",
            "request": "launch",
            "program": "app.py",
            "cwd": "${workspaceFolder}/resources_servers/browser_gym",
            "python": "${workspaceFolder}/.venv/bin/python",
            "env": {
                "PYTHONPATH": "${workspaceFolder}",
                "NEMO_GYM_CONFIG_PATH": "browser_gym_resources_server",
                "NEMO_GYM_CONFIG_DICT": "browser_gym_resources_server:\n  resources_servers:\n    browser_gym:\n      entrypoint: app.py\n      domain: agent\n      max_concurrent_browsers: 16\n      default_viewport_width: 1280\n      default_viewport_height: 720\n      host: 127.0.0.1\n      port: 10212\nbrowser_gym_openai_model:\n  responses_api_models:\n    openai_model:\n      entrypoint: app.py\n      openai_base_url: https://api.openai.com/v1\n      openai_api_key: YOUR_OPENAI_API_KEY\n      openai_model: computer-use-preview\n      openai_organization: YOUR_OPENAI_ORG\n      host: 127.0.0.1\n      port: 10213\nbrowser_openai_agent:\n  responses_api_agents:\n    browser_agent:\n      entrypoint: app.py\n      cua_adapter_type: openai\n      cua_model: computer-use-preview\n      max_steps: 250\n      viewport_width: 1280\n      viewport_height: 720\n      cua_debug_trajectories: true\n      cua_debug_output_dir: results/cua_debug_trajectories\n      resources_server:\n        type: resources_servers\n        name: browser_gym_resources_server\n      model_server:\n        type: responses_api_models\n        name: browser_gym_openai_model\n      datasets:\n      - name: example\n        type: example\n        jsonl_fpath: resources_servers/browser_gym/data/example.jsonl\n        num_repeats: 1\n      host: 127.0.0.1\n      port: 13135\nhead_server:\n  host: 127.0.0.1\n  port: 11000\ndry_run: false"
            },
            "justMyCode": false,
            "console": "integratedTerminal"
        },
        {
            "name": "Browser OpenAI Agent",
            "type": "debugpy",
            "request": "launch",
            "program": "app.py",
            "cwd": "${workspaceFolder}/responses_api_agents/browser_agent",
            "python": "${workspaceFolder}/.venv/bin/python",
            "env": {
                "PYTHONPATH": "${workspaceFolder}",
                "NEMO_GYM_CONFIG_PATH": "browser_openai_agent",
                "NEMO_GYM_CONFIG_DICT": "browser_gym_resources_server:\n  resources_servers:\n    browser_gym:\n      entrypoint: app.py\n      domain: agent\n      max_concurrent_browsers: 16\n      default_viewport_width: 1280\n      default_viewport_height: 720\n      host: 127.0.0.1\n      port: 10212\nbrowser_gym_openai_model:\n  responses_api_models:\n    openai_model:\n      entrypoint: app.py\n      openai_base_url: https://api.openai.com/v1\n      openai_api_key: YOUR_OPENAI_API_KEY\n      openai_model: computer-use-preview\n      openai_organization: YOUR_OPENAI_ORG\n      host: 127.0.0.1\n      port: 10213\nbrowser_openai_agent:\n  responses_api_agents:\n    browser_agent:\n      entrypoint: app.py\n      cua_adapter_type: openai\n      cua_model: computer-use-preview\n      max_steps: 250\n      viewport_width: 1280\n      viewport_height: 720\n      cua_debug_trajectories: true\n      cua_debug_output_dir: results/cua_debug_trajectories\n      resources_server:\n        type: resources_servers\n        name: browser_gym_resources_server\n      model_server:\n        type: responses_api_models\n        name: browser_gym_openai_model\n      datasets:\n      - name: example\n        type: example\n        jsonl_fpath: resources_servers/browser_gym/data/example.jsonl\n        num_repeats: 1\n      host: 127.0.0.1\n      port: 13135\nhead_server:\n  host: 127.0.0.1\n  port: 11000\ndry_run: false"
            },
            "justMyCode": false,
            "console": "integratedTerminal"
        }
    ],
    "compounds": [
        {
            "name": "Browser Gym Full Stack (Resource Server + Agent)",
            "configurations": ["Browser Gym Resource Server", "Browser OpenAI Agent"],
            "stopAll": true
        }
    ]
}
```

Then:

1. **Stop `ng_run`** if it's running (ports will conflict)
2. Open **Run and Debug** (Cmd+Shift+D)
3. Select **"Browser Gym Full Stack (Resource Server + Agent)"** and press F5
4. Set breakpoints in:
   - `browser_pool.py` line ~509 (`execute_action`) — see Playwright calls
   - `openai_adapter.py` line ~129 (`_map_openai_action`) — see action parsing
   - `app.py` (agent) line ~327 — see the main CUA loop
5. Trigger a run via curl:

```bash
curl -X POST http://127.0.0.1:13135/run \
  -H "Content-Type: application/json" \
  -d '{
    "responses_create_params": {
      "input": [
        {"role": "user", "content": [{"type": "input_text", "text": "Your task prompt here"}]}
      ]
    },
    "verifier_metadata": {
      "task_id": "YOUR-TASK-ID",
      "gym_url": "https://your-gym-url.com",
      "start_url": "https://your-gym-url.com",
      "viewport": {"width": 1280, "height": 720}
    }
  }'
```

Note: `ng_status` and `ng_collect_rollouts` will not work in debugger mode (no HeadServer). Use curl to trigger runs directly.

## Running Tests

Each server's tests are run individually (NeMo-Gym isolates tests per server via `ng_test`).

### Browser Gym Resource Server

```bash
# Via ng_test (creates isolated venv, installs deps, runs pytest):
ng_test +entrypoint=resources_servers/browser_gym

# Or directly with pytest (faster, uses current venv):
pytest resources_servers/browser_gym/tests/test_app.py -x -v
```

### Browser Agent

```bash
ng_test +entrypoint=responses_api_agents/browser_agent

# Or directly:
pytest responses_api_agents/browser_agent/tests/test_app.py -x -v
```

### OpenAI Model Server

```bash
ng_test +entrypoint=responses_api_models/openai_model

# Or directly:
pytest responses_api_models/openai_model/tests/test_app.py -x -v
```

### Anthropic Model Server

```bash
ng_test +entrypoint=responses_api_models/anthropic_model

# Or directly (requires `anthropic` SDK):
pytest responses_api_models/anthropic_model/tests/test_app.py -x -v
```

### Gemini Model Server

```bash
ng_test +entrypoint=responses_api_models/gemini_model

# Or directly (requires `google-genai` SDK):
pytest responses_api_models/gemini_model/tests/test_app.py -x -v
```

### Running a Subset of Tests

Use `-k` to filter by test class or name:

```bash
# Only Gemini adapter tests
pytest responses_api_agents/browser_agent/tests/test_app.py -x -v -k "Gemini"

# Only denormalization tests
pytest responses_api_agents/browser_agent/tests/test_app.py -x -v -k "Denorm"

# Only key normalization tests
pytest resources_servers/browser_gym/tests/test_app.py -x -v -k "NormalizeKey"
```

### Test Coverage Summary

| Test File | What It Covers |
|---|---|
| `resources_servers/browser_gym/tests/test_app.py` | BrowserPool, BrowserAction schema, key normalization, verify endpoint |
| `responses_api_agents/browser_agent/tests/test_app.py` | Adapter parsing (OpenAI, Gemini), denormalization, URL tracking, adapter factory |
| `responses_api_models/openai_model/tests/test_app.py` | OpenAI model server proxy, model config, responses/chat_completions |
| `responses_api_models/anthropic_model/tests/test_app.py` | Anthropic model server proxy, model fallback, error propagation |
| `responses_api_models/gemini_model/tests/test_app.py` | Gemini model server proxy, content serialization/deserialization, config parsing |

---

## Testing the Verify Endpoint

To test verification in isolation (without running a full CUA task):

```bash
curl -X POST http://127.0.0.1:10212/verify \
  -H "Content-Type: application/json" \
  -d '{
    "responses_create_params": {"input": []},
    "response": {
      "id": "test", "created_at": 0, "model": "test", "object": "response",
      "output": [], "parallel_tool_calls": false, "tool_choice": "auto", "tools": [],
      "env_id": "test",
      "trajectory": {"steps": [], "task_prompt": "", "initial_screenshot": ""},
      "local_storage_dump": "{}"
    },
    "verifier_metadata": {
      "task_id": "YOUR-TASK-ID",
      "gym_url": "https://your-gym-url.com"
    }
  }'
```

## Task Input

Tasks can be provided via a **local JSONL file** (`+input_jsonl_fpath`) or fetched directly from a **gym URL** (`+input_gym_url`). See [Collect Rollouts](#collect-rollouts) for usage examples.

### JSONL Task Format

Each line in the input JSONL:

```json
{
  "responses_create_params": {
    "input": [
      {"role": "user", "content": "Your task description here"}
    ]
  },
  "verifier_metadata": {
    "task_id": "UNIQUE-TASK-ID",
    "gym_url": "https://your-gym-url.com",
    "start_url": "https://your-gym-url.com",
    "viewport": {"width": 1280, "height": 720}
  }
}
```

- `task_id`: Unique identifier sent to the gym's verification API
- `gym_url`: Base URL of the gym (verification calls `{gym_url}/api/v1/get_actual_state`)
- `start_url`: URL the browser navigates to at the start of the task
- `viewport`: Browser viewport dimensions

### Gym URL (No JSONL Required)

When using `+input_gym_url`, tasks are fetched from the gym's `/api/v1/get_expected_state` endpoint and automatically converted to the JSONL format above. Each verifier entry's `prompt` (or `task_statement`) becomes the user message, and the verifier key becomes the `task_id`. Use `"+input_gym_task_id=[ID1,ID2]"` to run specific tasks.

## Rollout Output

Each line in the output JSONL contains:

```json
{
  "reward": 1.0,
  "response": {
    "id": "cua_...",
    "model": "computer-use-preview",
    "output": [{"role": "assistant", "content": [...]}],
    "env_id": "uuid",
    "trajectory": {
      "task_prompt": "...",
      "initial_screenshot": "<base64>",
      "final_message": "...",
      "steps": [
        {
          "action": {"action_type": "click", "coordinate": [476, 279], "button": "left"},
          "screenshot_before": "<base64>",
          "screenshot_after": "<base64>",
          "current_url": "https://...",
          "raw_provider_response": { ... }
        }
      ]
    },
    "local_storage_dump": "{...}"
  },
  "verifier_metadata": { ... }
}
```

The trajectory captures everything needed for RL training: task prompt, every action with before/after screenshots, raw model responses, final reward, and the full localStorage state.

## Verification

Verification uses **localStorage assertions**. After the CUA agent completes a task:

1. The agent dumps the browser's `localStorage` (which the gym app uses to track state)
2. The resource server sends `taskId` + `localStorageDump` to the gym's `/api/v1/get_actual_state` endpoint
3. The gym returns assertions like:
   ```json
   {
     "assertions": [
       {"result": "pass", "title": "Email template was created"},
       {"result": "fail", "title": "Subject line matches expected value", "actual": "...", "expected": "..."}
     ]
   }
   ```
4. Reward is `1.0` if all assertions have `"result": "pass"`, otherwise `0.0`

## Adding a New CUA Provider

All providers follow the same unified flow: **Adapter → Model Server → External API**. No direct API calls from the agent layer.

1. Create an adapter in `responses_api_agents/browser_agent/adapters/your_adapter.py` extending `BaseCUAAdapter`
2. Implement `initialize()`, `step()`, and `reset()` -- handle context management (history, trimming, validation) in the adapter
3. The adapter receives an `api_caller` callable (injected by the agent) that routes API calls through the model server
4. Map provider-specific actions to the unified `BrowserAction` schema
5. Create a stateless model server in `responses_api_models/your_model/app.py` that relays API calls (handles auth, retries, transport)
6. Register the adapter in `responses_api_agents/browser_agent/adapters/__init__.py` (AdapterFactory)
7. Add model server + agent config blocks in `configs/browser_gym.yaml` with `model_server` reference
8. Add `requirements.txt` with the provider's SDK in both server directories

## Adding a New Task

**Option 1: Via gym URL** (recommended for gyms with `/api/v1/get_expected_state`)

Tasks are fetched automatically -- just use `+input_gym_url` when collecting rollouts. No manual JSONL editing needed. To run specific tasks, add `"+input_gym_task_id=[YOUR-TASK-ID]"` (or multiple: `"+input_gym_task_id=[ID1,ID2,ID3]"`).

**Option 2: Via JSONL file**

Append a JSON line to `data/example.jsonl`:

```json
{"responses_create_params": {"input": [{"role": "user", "content": "Your task description"}]}, "verifier_metadata": {"task_id": "YOUR-TASK-ID", "gym_url": "https://your-gym-url.com", "start_url": "https://your-gym-url.com", "viewport": {"width": 1280, "height": 720}}}
```
