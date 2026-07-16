# MCP auto-exposure (draft — for design discussion)

**Status: draft / RFC.** Serve a resources server's existing FastAPI tool routes over MCP with
**zero handler changes** and **one opt-in flag**. This consolidates the exploration in PRs #2002 and
#2053 into a single tracked module plus a small framework hook.

## What it does

A resources server sets one class attribute:

```python
class MyResourcesServer(SimpleResourcesServer):
    expose_tools_over_mcp = True     # <- the only addition; no decorators, no handler edits
    ...
```

and its plain `POST /<tool>` routes become advertised + callable over an MCP `/mcp` endpoint.
`run_webserver` calls `maybe_auto_expose(server, app)` automatically after building the app, so the
author writes **no function call**. Handlers keep their `request: Request` param and their
`request.session[SESSION_ID_KEY]` reads exactly as written.

Dispatcher servers (one catch-all route backing many tools whose schemas live in data) additionally
override one method, `mcp_tool_inventory()`, returning `[{name, input_schema, description}]`.

## How a tool call is served

Per route, chosen once at startup by a two-gate detector:

- **Direct dispatch (default).** The frozen handler runs **exactly once**, invoked with a fabricated
  `Request` whose `.session` is materialized directly — no middleware, no routing, no second app
  pass. ~5-7 us/call. Public FastAPI/Starlette/pydantic surface only.
- **Replay fallback.** Where the detector cannot *prove* direct == a real HTTP request (an author's
  custom middleware, or a handler shape direct dispatch doesn't reproduce), the call is re-issued as
  an internal in-process HTTP request through the full app stack (httpx-free). No in-tree server
  needs it today; it guarantees correctness never depends on the fast path.

Both paths verify a signed session token (`X-NeMo-Gym-Session-Token`, same salt/secret derivation as
`base_resources_server.py`'s MCP token), resolve the session, run the handler, and map the result to
MCP. The MCP engine is the official SDK's **public low-level** `Server` — no private-attr access.

## Where the code lives

| File | What |
|---|---|
| `nemo_gym/mcp_auto_exposure.py` | the whole engine in one file: detector, direct dispatcher, replay fallback, route harvest, token mint/verify, `/seed_session` wrap, `/mcp` mount, and `maybe_auto_expose` (the flag gate) |
| `nemo_gym/base_resources_server.py` | +1 line: the `expose_tools_over_mcp` opt-in flag on `SimpleResourcesServer` |
| `nemo_gym/server_utils.py` | +3 lines: `run_webserver` calls `maybe_auto_expose` after the app is built |
| `prototypes/mcp_auto_exposure/` | the live acceptance suite (`run_checks.py`) + demo servers built from **unmodified in-tree** `resources_servers/` |

**Tool files are untouched** — `git diff origin/main -- resources_servers/` is empty.

## Run the checks

```bash
python prototypes/mcp_auto_exposure/run_checks.py    # 37/37 (finance + workplace)
# install fhaviary for +7 aviary checks (44/44)
```

Verifies against pristine `origin/main` handler code: the flag alone exposes the tools; handlers run
over MCP with `request.session` working and sharing per-session state with the HTTP door; typed
schemas harvested; dispatcher via the override; per-session `allowed_tools` filtering; cross-session
isolation; and byte-parity of the HTTP door (only additive deltas: the `mcp` key in `/seed_session`,
the new `/mcp` path). All in-tree tool routes dispatch **direct, zero replay** — verified live for
finance, workplace, aviary, newton_bench, openenv, and ns_tools.

## Notes for the discussion

- **The flag is opt-in (default off)** because auto-exposing *every* route is not always wanted:
  e.g. aviary's `/step`/`/close` are agent/harness plumbing, and `/step` carries `env_id`, so
  exposing it hands a model a parameter that addresses other rollouts' environments. Per-server
  opt-in (plus the toolless-catch-all declaration) is the guard.
- **The detector deserves an audit before defaulting on.** It refuses (falls back to replay) on
  route-level `Depends()`, non-Gym middleware, and unsupported parameter shapes; it dispatches the
  current in-tree servers correctly, but the classifier should get a review pass before broad
  rollout.
- **If adopted, `@gym_tool` (PR #2002) becomes unnecessary for exposure** — this reads the routes an
  author already wrote, so the ~861-line migration across 16 tool files is not needed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
