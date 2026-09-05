# Directory layout and config wiring

## Directory layout

There are 46 agent servers under `responses_api_agents/`. All but one follow this layout:

```
responses_api_agents/<name>_agent/
├── __init__.py
├── app.py              # the server class + `if __name__ == "__main__": X.run_webserver()`
├── configs/
│   └── <name>_agent.yaml
├── tests/
│   └── test_app.py
├── requirements.txt
└── README.md
```

Add files beyond this set only when the shape demands it:

- **A bridge module** when you adapt an external framework's model interface —
  `langchain_deepagents_agent/responses_langchain_bridge.py`. Reviewers ask for this split; keep `app.py`
  to endpoint orchestration and put the model adapter plus conversion functions in the bridge.
- **A concrete-subclass module** when `app.py` holds an abstract base — `reasoning_search_agent.py`,
  `langgraph_agent/reflection_agent.py`. The concrete module becomes the config `entrypoint`.
- **A setup module** when an external tool must be auto-installed —
  `claude_code_agent/setup_claude_code.py`, plus a `scripts/` shell script.
- **An observability module** when parsing a transcript from a separate process —
  `claude_code_agent/observability.py`.

## `requirements.txt`

Always starts with the editable monorepo install:

```
-e nemo-gym[dev] @ ../../
```

Framework pins follow. Any version ceiling needs a comment explaining the conflict and what would
unblock a bump — otherwise it reads as an arbitrary stale pin later. See checklist item 10.

## Config

### The agent's own config

`configs/<name>_agent.yaml` declares the agent with `???` where a composing config must supply a value:

```yaml
langchain_deepagents_agent:
  responses_api_agents:
    langchain_deepagents_agent:
      entrypoint: reasoning_search_agent.py
      resources_server:
        type: resources_servers
        name: ???
      model_server:
        type: responses_api_models
        name: policy_model
      tavily_api_key: ${tavily_api_key}
      max_search_results: 5
```

Structure is: **server-instance name** → `responses_api_agents:` → **class directory name** → fields.
`entrypoint` is `app.py` unless the base class is abstract, in which case it names the concrete module.

Agent-specific config fields (`tavily_api_key`, `max_search_results`, `system_prompt`) go on your own
`XConfig` subclass. Prefer a config field over a module constant — it lets each combo YAML set a
different value without touching code.

### The combined launch config

The runnable config lives with the **resources server**, not the agent, and is named
`<resources_server>_<agent>_model_server.yaml`. It inlines the agent block with `???` resolved, plus any
judge/model servers the benchmark needs:

```
resources_servers/tavily_search/configs/tavily_search_langchain_deepagents_agent_model_server.yaml
```

The `_model_server` suffix is the convention meaning "this config expects `--model-type` to supply the
policy model at launch." That is discoverable only by reading sibling filenames — the CLI flags and
`--help` do not communicate it.

That YAML's top-level key is also the `--agent` name for `gym eval run`:

```bash
gym env start --resources-server tavily_search/tavily_search_langchain_deepagents_agent_model_server \
    --model-type inference_provider/openrouter

gym eval run --no-serve --agent tavily_search_langchain_deepagents_agent_model_server \
    --input resources_servers/tavily_search/data/example.jsonl --output /tmp/out.jsonl
```

Put the exact working commands in a comment at the top of the combined config. Every existing one does,
and it is the fastest way for the next person to run your agent.

### Relative paths

Relative paths in a config resolve against **that server's own process cwd** when Gym spins it up
(e.g. `resources_servers/tavily_search/`), not the repo root and not your shell's cwd. See checklist
item 9.

## Commands

```bash
# run the servers
gym env start --resources-server <resources_server>/<combined_config> --model-type <provider>

# tests for one server (creates an isolated venv on first run — slow)
gym env test --resources-server <name>

# health
gym env status

# dump the merged config
gym env resolve --config ...
```

`gym env test` creates isolated venvs per server, so `os.environ` changes inside Python do not
propagate — set env vars externally (e.g. `RAY_TMPDIR=/tmp gym env test ...`).

## Before opening a PR

Per `CLAUDE.md`:

- Run real rollouts with a real model and inspect agent and verifier behavior. Green unit tests alone
  are explicitly not enough for agent or environment changes.
- `pre-commit run --all-files` (or scope it: `pre-commit run --files responses_api_agents/my_agent/**/*`).
- DCO sign-off: `git commit -s`.
- Coverage must stay >= 96%.
- New source files need the standard NVIDIA SPDX header.
