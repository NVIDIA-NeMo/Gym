# CVDP Agent

Agents for the **CVDP** (Comprehensive Verilog Design Problems) benchmark — hardware
(RTL/SystemVerilog) design and code-comprehension tasks. All flavors here share the same
CVDP resources server, which owns task data, verification, and reward computation. See
`[resources_servers/cvdp/](../../resources_servers/cvdp/)` for benchmark details.

This directory ships **one agent class**, `CVDPAgent` (in `app.py`), with **two flavors**
selected by the `simple_agent` config flag. They differ only in *how the model interacts
with the task*; both read the same task format and grade through the same `/verify`.


| Config flag          | Flavor                   | When to use                                                                                                                                  |
| -------------------- | ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `simple_agent: true`  | Non-agentic, no sandbox  | Model emits the RTL directly (optionally with resources-server tool calls). Fast.                                                            |
| `simple_agent: false` | Agentic, **any harness** | Runs a config-selected coding harness inside a Gym sandbox to edit files and self-test. |


> `CVDPAgent.run()` dispatches on `config.simple_agent`: `_run_simple()` for the non-agentic
> flavor and `_run_agentic()` for the sandboxed harness flavor. Both live in `app.py`, the
> repo-wide convention for an agent's default entrypoint (see every other
> `responses_api_agents/*/`).

---

## 1. `simple_agent: true` — non-agentic, no harness

The simplest flow. There is **no coding harness and no sandbox**. The model itself is asked
to produce the answer (RTL/code), and the resources server grades whatever the model
emitted.

What `_run_simple()` does:

1. `POST /seed_session` to the resources server (per-task state).
2. Drive the model via the agent's own `/v1/responses` loop (`responses()`):
  - call the **model server**,
  - if the model emits `function_call`s, execute them against the **resources server** and
  feed results back,
  - stop when the model produces a final message (or hits `max_steps` / token limit).
3. `POST /verify` with the model's response → reward.
4. Retry the model+verify step up to `llm_parse_retries` times on a parse failure or a model
  error (mirrors CVDP's `LLM_RETRY_COUNT`).

Key point: the model writes the code *in its response*; nothing edits files on disk and no
simulator is run by the agent. This is the flow used by the non-agentic CVDP code-generation
dataset. Config: `[configs/cvdp_agent.yaml](configs/cvdp_agent.yaml)`.

---

## 2. `simple_agent: false` — agentic, any harness

The agentic flavor. The coding harness is **configuration, not code**: you name a Gym agent
in the YAML and it gets booted inside the sandbox to do the work. Proven to grade real RTL
through the unchanged CVDP verifier.

### The core idea

Every Gym coding agent (Claude Code, Hermes, ...) exposes the same method: `responses()`.
`CVDPAgent` never needs to know a harness's CLI — it just boots the named harness
inside the sandbox and calls its `responses()`. Each harness already knows its own CLI.

### How a run flows

```
config names a harness (agent_server_module/class/config_class)
        │
CVDPAgent._run_agentic()
  reads verifier_metadata        ← context_files / target_files / harness_files
  _provision_deps()              → install that harness's software into a deps prefix (once, cached)
  _seed_files()                  → copy safe context files into the workspace (never the hidden harness)
  _build_spec()                  → generate the in-sandbox runner script + workspace files
  upload runtime archive        → provider-neutral harness dependencies
  AsyncSandbox(...).exec(runner) → the runner imports the harness and calls responses()
                                     the harness edits rtl/*.sv with its own tools
  _remote_harvest()              → pull back the produced HDL files
  POST /verify                   ← the unchanged CVDP resources server → reward
```

### What you need to know to use it

- **Pick a harness** by setting three strings in the config:
`agent_server_module`, `agent_server_class`, `agent_config_class`.
- `**agent_kwargs`** are passed straight into that harness's config class, so every key must
be a real field on `<agent_config_class>`. (e.g. for `ClaudeCodeAgentConfig`: `model`,
`anthropic_base_url`, `anthropic_api_key`, `max_turns`, ...). These are **harness-specific**.
- **A deps script must exist** for the harness: `setup_scripts/<key>_deps.sh`, where `<key>`
is derived from the module (`responses_api_agents.hermes_agent.app` → `hermes_agent`). It
builds a self-contained prefix (portable Python + nemo_gym + the harness CLI) that is
packaged once and uploaded into each sandbox.
- The data format **never changes** when you swap harnesses, and grading always goes through
the same CVDP `/verify`.

### Adding a new harness (the whole checklist)

1. Add `setup_scripts/<key>_deps.sh` that installs that harness into `$DEPS_DIR` (start from
  `claude_code_agent_deps.sh` or `hermes_agent_deps.sh`; source `_portable_python.sh` for
   the shared portable-Python + nemo_gym base).
2. Add a config under `configs/` naming the harness and its `agent_kwargs`.

No Python changes — `app.py` and `sandbox_entrypoint.py` stay untouched.

### Sandbox configuration

The agent uses the common sandbox API. `sandbox_provider` accepts an inline config or a
named top-level config. `archive` uploads the runtime without provider-specific mounts.
Use a registry OCI image for Apptainer and OpenSandbox. Local `.sif` images require Apptainer.

### Running

Add the harness settings to your repo-root `env.yaml`:

```yaml
anthropic_model_name: <claude-model>
anthropic_api_key: <your-api-key>
anthropic_base_url: https://api.anthropic.com
cvdp_sim_image: nvidia/cvdp-sim:v1.0.0
```

**Step 1: Start servers**

```bash
gym env start \
    --config responses_api_agents/cvdp_agent/configs/cvdp_agent_generic_claude.yaml \
    --resources-server cvdp \
    --model-type vllm_model
```

**Step 2 — Collect rollouts** (`--agent` is the config's top-level name):

```bash
gym eval run --no-serve \
    --agent cvdp_agent_generic_claude \
    --config responses_api_agents/cvdp_agent/configs/cvdp_agent_generic_claude.yaml \
    --input resources_servers/cvdp/data/gym_agentic_code_generation_no_commercial.jsonl \
    --output results/generic_agent/rollouts.jsonl \
    --num-repeats 5 \
    --concurrency 10 \
    --resources-server cvdp \
    --model-type vllm_model \
    --resume
```

**Step 3 — Generate report:**

```bash
python resources_servers/cvdp/scripts/cvdp_pass_at_k_report.py \
    --rollouts  results/rollouts.jsonl \
    --output    results/report/ \
    --model     cvdp_agent_generic_claude \
    --dataset   <original_cvdp_dataset>.jsonl \
    --k         1
```

To run a different harness, swap `cvdp_agent_generic_claude` for another config (e.g.
`cvdp_agent_generic_hermes`). See `resources_servers/cvdp/README.md` for the full dataset
download/conversion steps and report-output details.

### Supporting files

- `**app.py**` — the `CVDPAgent` class (both flavors) plus the agentic host-side helpers:
`load_runner_source()` (reads `sandbox_entrypoint.py` and drops it into the container),
`harvest()` (collects produced files by glob, skipping files unchanged from what was seeded),
`agent_key()` (maps a module to its deps-script key), and `deps_recipe_key()` (fingerprints the
deps cache).
- `**sandbox_entrypoint.py**` — the guest entrypoint copied verbatim into the sandbox and run as
`python agent_runner.py`. Reads the agent module/class and task inputs from `NV_*` env vars set
by the agent, imports the named harness, calls `responses()`, and writes the trajectory out.
Kept as a plain, lintable module rather than a string template.
- `**setup_scripts/**`
  - `_portable_python.sh` — shared base: downloads a relocatable CPython and `pip install`s
  nemo_gym into `$DEPS_DIR`. (Leading underscore = sourced helper, not run directly.)
  - `claude_code_agent_deps.sh` — adds portable Node + the `@anthropic-ai/claude-code` CLI.
  - `hermes_agent_deps.sh` — adds the `hermes-agent` Python package (version pinned from
  `responses_api_agents/hermes_agent/requirements.txt`).
  - `opencode_agent_deps.sh`: adds portable Node and the pinned `opencode-ai` CLI.
- `**configs/cvdp_agent_generic_claude.yaml**`, `**configs/cvdp_agent_generic_hermes.yaml**`,
`**configs/cvdp_agent_generic_opencode.yaml**`
— example wirings; identical except the `agent_server_*` block, `agent_kwargs`, and deps
script. They show the harness swap is config-only.

### Runtime requirements

- A configured Gym sandbox provider. Apptainer is required only when selected.
- An image visible to that provider. Local `.sif` images are Apptainer-only.
- Network access on the **first** run for the chosen harness (to download portable
Python/Node and the harness CLI). The result is cached via a `.installed` sentinel, so
later runs are offline-fast.
- A reachable model endpoint (configured in the run's `env.yaml`).

---

# Licensing information

Code: Apache 2.0
Data: N/A

Dependencies

- nemo_gym: Apache 2.0
