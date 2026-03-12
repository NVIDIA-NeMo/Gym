# Browser Gym Integration for NeMo-Gym

## Overview

This integration adds Browser Agent support to NeMo-Gym, enabling LLMs to interact with web applications through visual observations (screenshots) and browser actions (click, type, scroll, etc.). The system captures full trajectories suitable for RL training.

### Architecture

The integration follows NeMo-Gym's three-server architecture:

- **OpenAI**: Uses the `openai_model` server -- token tracking, retry logic via `NeMoGymAsyncOpenAI`
- **Anthropic**: Uses the `anthropic_model` server -- stateful session management, context trimming, action normalization to OpenAI format
- **Gemini**: Uses a direct-API adapter in the agent (Gemini's protocol doesn't fit either schema)

```
                              EXTERNAL LLM PROVIDERS
                  ┌──────────────────────────────────────────┐
                  │   OpenAI      Anthropic       Google     │
                  │   (CUA         (Sonnet/        (Gemini   │
                  │   Preview)     Opus)           2.5 Flash)│
                  └─────┬──────────┬──────────────────┬──────┘
                        │          │                  │
                        │          │                  │ direct API
                        │          │                  │ (adapter)
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
│  │  ┌── Model server path ───┐   │    ┌─────────────┴────────┐  │   │
│  │  │ OpenAI + Anthropic     │   │    │  Gemini adapter      │  │   │
│  │  │ via /v1/responses      │   │    │  (direct HTTP)       │  │   │
│  │  │                        │   │    │                      │  │   │
│  │  │ • computer_call        │   │    │  • Client-side       │  │   │
│  │  │ • computer_call_output │   │    │    context mgmt      │  │   │
│  │  │ • previous_response    │   │    │  • Provider-specific │  │   │
│  │  │   _id chaining         │   │    │    action mapping    │  │   │
│  │  └─────────┬──────────────┘   │    └──────────┬───────────┘  │   │
│  │            │                  │               │              │   │
│  │            │  unified BrowserAction           │              │   │
│  │            └──────────────┬───────────────────┘              │   │
│  │                           │                                  │   │
│  │    ┌──────────────────────┴─────────────────────────────┐    │   │
│  │    │  CUA Loop:                                         │    │   │
│  │    │   1. /seed_session → open browser at start_url     │    │   │
│  │    │   2. /step → execute action, return screenshot     │    │   │
│  │    │   3. Get next action (model server or adapter)     │    │   │
│  │    │   4. Repeat 2-3 until done or max_steps            │    │   │
│  │    │   5. /dump_local_storage → capture LS              │    │   │
│  │    │   6. /close → tear down browser                    │    │   │
│  │    └────────────────────────────────────────────────────┘    │   │
│  └───────────────────────────┬──────────────────────────────────┘   │
│                              │ /verify                              │
│  ┌───────────────────────────┴──────────────────────────────────┐   │
│  │  MODEL SERVERS                                               │   │
│  │                                                              │   │
│  │  ┌────────────────────────┐  ┌────────────────────────────┐  │   │
│  │  │  openai_model          │  │  anthropic_model           │  │   │
│  │  │                        │  │                            │  │   │
│  │  │  • api.openai.com      │  │  • api.anthropic.com       │  │   │
│  │  │  • computer_use_preview│  │  • Stateful sessions       │  │   │
│  │  │  • Org header          │  │  • Context trimming        │  │   │
│  │  │  • Token tracking      │  │  • Action normalization    │  │   │
│  │  │  • Retry with backoff  │  │    (Anthropic → OpenAI)    │  │   │
│  │  └────────────────────────┘  │  • Token tracking          │  │   │
│  │                              │  • Beta flags              │  │   │
│  │                              └────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  BROWSER GYM RESOURCE SERVER                                 │   │
│  │  resources_servers/browser_gym                               │   │
│  │                                                              │   │
│  │  /seed_session  /step  /dump_local_storage  /close  /verify  │   │
│  │                                                              │   │
│  │  ┌──────────────────┐   ┌───────────────────────────────┐    │   │
│  │  │   BrowserPool    │   │        Verification           │    │   │
│  │  │   (Playwright    │   │  taskId + localStorageDump    │    │   │
│  │  │    Chromium,     │   │         │                     │    │   │
│  │  │    Semaphore     │   │         ▼                     │    │   │
│  │  │    bounded)      │   │  POST /api/v1/get_actual_state│────┼───┼──┐
│  │  │                  │   │         │                     │    │   │  │
│  │  │                  │   │         ▼                     │    │   │  │
│  │  │                  │   │  assertions → reward (0/1)    │    │   │  │
│  │  └──────────────────┘   └───────────────────────────────┘    │   │  │
│  └──────────────────────────────────────────────────────────────┘   │  │
└─────────────────────────────────────────────────────────────────────┘  │
                                                                         │
                   ┌─────────────────────────────────────┐               │
                   │     GYM ENVIRONMENTS (external)     │◄──────────────┘
                   │                                     │
                   │  SpotHub    DeskZen    PriceLens    │
                   │                                     │
                   │  Each gym exposes:                  │
                   │  - Web app UI (browser target)      │
                   │  - POST /api/v1/get_actual_state    │
                   │    (verification via LS assertions) │
                   └─────────────────────────────────────┘
```

**Key points:**
- **OpenAI** goes through the `openai_model` server, gaining token usage tracking, centralized retry logic, and full architectural consistency. CUA-specific types (`computer_call`, `computer_call_output`) are supported in `NeMoGymResponseInputItem`.
- **Anthropic** goes through the `anthropic_model` server, which translates between NeMo-Gym's OpenAI-style schema and Anthropic's native protocol. It maintains stateful sessions (conversation history, context trimming) and normalizes Anthropic actions to OpenAI format so the agent uses a single code path.
- **Gemini** still uses a direct-API adapter because its protocol (`FunctionCall`/`FunctionResponse`) doesn't map to either schema.
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
│  │  • Reads input JSONL (tasks)                                  │  │
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
│  │   • Calls model server (or adapter) for next action           │  │
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
│  │  │  • api.anthropic.com   │ │                     │              │
│  │  │  • Stateful sessions   │ │                     │              │
│  │  │  • Context trimming    │ │                     │              │
│  │  │  • Action normalization│ │                     │              │
│  │  │  • Token tracking      │ │                     │              │
│  │  │  • Beta flags          │ │                     │              │
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
    └──────────────────────┘              └──────────────────────┘
```

**Communication summary:**

| From | To | Endpoint | Payload | Response |
|------|----|----------|---------|----------|
| CLI | Agent | `POST /run` | task prompt + verifier_metadata | CUAVerifyResponse (reward + trajectory) |
| Agent | Model | `POST /v1/responses` | screenshot + input items | NeMoGymResponse (computer_call actions) |
| Agent | Resource | `POST /seed_session` | start_url, viewport, task_id | env_id + initial screenshot |
| Agent | Resource | `POST /step` | env_id + BrowserAction | screenshot + current_url |
| Agent | Resource | `POST /dump_local_storage` | env_id | localStorage JSON string |
| Agent | Resource | `POST /close` | env_id | status |
| Agent | Resource | `POST /verify` | response + verifier_metadata + localStorage | reward (0.0 or 1.0) |
| Resource | Gym | `POST /api/v1/get_actual_state` | taskId + localStorageDump | assertions array |

### Data Flow

1. Agent receives task prompt + `verifier_metadata` (task_id, gym_url, start_url)
2. Agent seeds a browser session via resource server (navigates to start_url)
3. Agent enters CUA loop:
   - **Model server path** (OpenAI + Anthropic): sends `NeMoGymResponseCreateParamsNonStreaming` with screenshot to model server `/v1/responses` → receives `NeMoGymResponse` with `computer_call` items → maps to `BrowserAction` → executes via resource server `/step` → sends `computer_call_output` with new screenshot → repeat (context managed via `previous_response_id`)
   - **Adapter path** (Gemini): sends screenshot to adapter → receives `BrowserAction` → executes via resource server `/step` → repeat
4. On completion (model says "done" or max_steps reached), agent dumps localStorage from the browser
5. Agent closes the browser session
6. Resource server verifies by sending localStorage to the gym's `/api/v1/get_actual_state` endpoint
7. Reward is 1.0 if all assertions pass, 0.0 otherwise

### Provider Support

| Provider | Execution Path | Context Management | Token Tracking |
|----------|---------------|-------------------|----------------|
| **OpenAI** (CUA Preview) | Model server (`openai_model`) | Server-side via `previous_response_id` | Yes (via `NeMoGymResponseUsage`) |
| **Anthropic** (Sonnet/Opus) | Model server (`anthropic_model`) | Stateful sessions with turn-based trimming | Yes (via usage in response) |
| **Gemini** (2.5 Flash) | Direct-API adapter | Client-side `_contents` list with paired-turn trimming | No |

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
├── app.py                          # Anthropic CUA model server (stateful, /v1/responses)
└── requirements.txt

responses_api_agents/browser_agent/
├── app.py                          # Agent server (responses + run endpoints)
├── trajectory_writer.py            # Debug trajectory output (screenshots + JSON)
├── adapters/
│   ├── __init__.py                 # AdapterFactory registry
│   ├── base.py                     # BaseCUAAdapter abstract class
│   ├── openai_adapter.py           # OpenAI Computer Use Preview (fallback without model server)
│   ├── anthropic_adapter.py        # Anthropic Sonnet / Opus (fallback without model server)
│   └── gemini_adapter.py           # Google Gemini (primary path -- no model server)
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

3. **API keys** in `env.yaml` (project root):
   ```yaml
   cua_openai_api_key: "sk-proj-..."
   cua_openai_org: "org-..."
   cua_anthropic_api_key: "sk-ant-..."
   cua_gemini_api_key: "AIza..."
   ```

4. **Gym environments** deployed and accessible (e.g. `https://app.spothub.rlgym.turing.com`)

## Running

### Start All Servers

```bash
ng_run "+config_paths=[resources_servers/browser_gym/configs/browser_gym.yaml]"
```

This starts 8 servers:
- 1 resource server (browser management + verification)
- 3 model servers (OpenAI CUA, Anthropic Sonnet, Anthropic Opus)
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

**Gemini 2.5 Flash:**

```bash
ng_collect_rollouts \
  +agent_name=browser_gemini_agent \
  +input_jsonl_fpath=resources_servers/browser_gym/data/example.jsonl \
  +output_jsonl_fpath=results/cua_rollouts_gemini.jsonl \
  +num_repeats=1 \
  "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}"
```

Available agent names: `browser_openai_agent`, `browser_anthropic_sonnet_agent`, `browser_anthropic_opus_agent`, `browser_gemini_agent`

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

### Option 1: Via CLI override (no config change)

```bash
ng_collect_rollouts \
  +agent_name=browser_openai_agent \
  +input_jsonl_fpath=resources_servers/browser_gym/data/example.jsonl \
  +output_jsonl_fpath=results/cua_rollouts.jsonl \
  +num_repeats=1 \
  "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}" \
  ++browser_openai_agent.responses_api_agents.browser_agent.cua_debug_trajectories=true
```

### Option 2: Set in YAML config

In `configs/browser_gym.yaml`, set `cua_debug_trajectories: true` for the desired agent.

### Output Structure

```
/tmp/cua_debug_trajectories/<env_id>/
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
        out_dir = f'/tmp/cua_debug_trajectories/rollout_{i+1}/screenshots'
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

Create `.vscode/launch.json` in the project root with the following content (replace the API key and org with your values from `env.yaml`):

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
            "python": "${workspaceFolder}/env/bin/python",
            "env": {
                "PYTHONPATH": "${workspaceFolder}",
                "NEMO_GYM_CONFIG_PATH": "browser_gym_resources_server",
                "NEMO_GYM_CONFIG_DICT": "browser_gym_resources_server:\n  resources_servers:\n    browser_gym:\n      entrypoint: app.py\n      domain: agent\n      max_concurrent_browsers: 16\n      default_viewport_width: 1280\n      default_viewport_height: 720\n      host: 127.0.0.1\n      port: 10212\nbrowser_openai_agent:\n  responses_api_agents:\n    browser_agent:\n      entrypoint: app.py\n      cua_adapter_type: openai\n      cua_model: computer-use-preview\n      cua_api_key: YOUR_OPENAI_API_KEY\n      cua_org: YOUR_OPENAI_ORG\n      max_steps: 50\n      viewport_width: 1280\n      viewport_height: 720\n      cua_debug_trajectories: true\n      cua_debug_output_dir: /tmp/cua_debug_trajectories\n      resources_server:\n        type: resources_servers\n        name: browser_gym_resources_server\n      datasets:\n      - name: example\n        type: example\n        jsonl_fpath: resources_servers/browser_gym/data/example.jsonl\n        num_repeats: 1\n      host: 127.0.0.1\n      port: 13135\nhead_server:\n  host: 127.0.0.1\n  port: 11000\ndry_run: false"
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
            "python": "${workspaceFolder}/env/bin/python",
            "env": {
                "PYTHONPATH": "${workspaceFolder}",
                "NEMO_GYM_CONFIG_PATH": "browser_openai_agent",
                "NEMO_GYM_CONFIG_DICT": "browser_gym_resources_server:\n  resources_servers:\n    browser_gym:\n      entrypoint: app.py\n      domain: agent\n      max_concurrent_browsers: 16\n      default_viewport_width: 1280\n      default_viewport_height: 720\n      host: 127.0.0.1\n      port: 10212\nbrowser_openai_agent:\n  responses_api_agents:\n    browser_agent:\n      entrypoint: app.py\n      cua_adapter_type: openai\n      cua_model: computer-use-preview\n      cua_api_key: YOUR_OPENAI_API_KEY\n      cua_org: YOUR_OPENAI_ORG\n      max_steps: 50\n      viewport_width: 1280\n      viewport_height: 720\n      cua_debug_trajectories: true\n      cua_debug_output_dir: /tmp/cua_debug_trajectories\n      resources_server:\n        type: resources_servers\n        name: browser_gym_resources_server\n      datasets:\n      - name: example\n        type: example\n        jsonl_fpath: resources_servers/browser_gym/data/example.jsonl\n        num_repeats: 1\n      host: 127.0.0.1\n      port: 13135\nhead_server:\n  host: 127.0.0.1\n  port: 11000\ndry_run: false"
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
3. Select **"CUA Full Stack (Resource Server + Agent)"** and press F5
4. Set breakpoints in:
   - `browser_pool.py` line ~142 (`execute_action`) — see Playwright calls
   - `openai_adapter.py` line ~160 (`_map_openai_action`) — see action parsing
   - `app.py` (agent) line ~192 — see the main CUA loop
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

## JSONL Task Format

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

**Option A: Model server (preferred)** -- full NeMo-Gym architectural consistency:

1. Create `responses_api_models/your_model/app.py` extending `SimpleResponsesAPIModel`
2. Translate provider requests/responses to NeMo-Gym's `NeMoGymResponse` format (with `computer_call` items)
3. Normalize provider actions to OpenAI format so the agent's `_map_openai_action` works
4. Add model server + agent config blocks in `configs/browser_gym.yaml` with `model_server` reference
5. Add `requirements.txt` with the provider's SDK

**Option B: Adapter** -- for providers whose protocol is hard to normalize:

1. Create `responses_api_agents/browser_agent/adapters/your_adapter.py`
2. Extend `BaseCUAAdapter` and implement `initialize()`, `step()`, `reset()`
3. Map provider actions to unified `BrowserAction` schema
4. Register in `adapters/__init__.py`:
   ```python
   _ADAPTER_REGISTRY["your_provider"] = YourCUAAdapter
   ```
5. Add agent config block in `configs/browser_gym.yaml` (no `model_server` reference)

## Adding a New Task

Append a JSON line to `data/example.jsonl`:

```json
{"responses_create_params": {"input": [{"role": "user", "content": "Your task description"}]}, "verifier_metadata": {"task_id": "YOUR-TASK-ID", "gym_url": "https://your-gym-url.com", "start_url": "https://your-gym-url.com", "viewport": {"width": 1280, "height": 720}}}
```
