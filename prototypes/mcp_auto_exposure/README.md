# MCP auto-exposure prototype (draft — for design discussion)

**Status: prototype / RFC. Not for merge as-is.** This is a self-contained proof-of-concept for an
alternative way to serve NeMo Gym resources-server tools over MCP, built for design review against
the [dual-registration PR](https://github.com/NVIDIA-NeMo/Gym/pull/2002). It lives under
`prototypes/` and wires into nothing — no CI, no imports from the rest of the tree.

## The idea

Expose **every existing FastAPI tool route as an MCP tool automatically**, with:

- **R1 — no decorator.** Env authors write plain FastAPI routes exactly as on `main`.
- **R2 — byte-identical handlers.** Handlers keep their `request: Request` parameter and read
  `request.session[SESSION_ID_KEY]`. This module never edits a handler.
- **R3 — every non-basic POST route becomes a tool.** Parameterized catch-all routes (dispatcher
  servers) can't be a single tool, so those servers override one method,
  `mcp_tool_inventory()`, returning `[{name, input_schema, description}]`; those calls dispatch
  through the existing catch-all.
- **R4 — backwards compatible.** The HTTP door is untouched except two additive deltas: the
  `/seed_session` response gains an `mcp` key, and a new `/mcp` endpoint exists.

Contrast with PR #2002 (the `@gym_tool` design): that migrates each tool to a new signature
(`session_id: str` + typed params) and dispatches MCP calls **directly**, at the cost of ~861 lines
changed across 16 server files. This prototype changes **zero** tool files, at the cost of a
**replay** step (below).

## The mechanism: the replay bridge

A frozen `request: Request` handler can only be invoked correctly by the HTTP machinery itself
(the MCP SDK can't even register a `Request`-typed function). So an MCP `tools/call` is served by
**re-issuing it as an internal in-process HTTP request** through the app's own ASGI stack:

```
MCP tools/call
  -> verify the signed session token (X-NeMo-Gym-Session-Token; itsdangerous, same salt/secret
     derivation as base_resources_server.py's MCP token)
  -> mint the SessionMiddleware cookie for that session id (the server owns the secret)
  -> re-issue POST /<tool> internally through the FULL app stack (httpx-free hand-built ASGI call);
     SessionMiddleware populates request.session, FastAPI validates the body, the unmodified
     handler runs
  -> map the HTTP response to an MCP result (2xx JSON -> content + structuredContent;
     otherwise -> isError carrying the HTTP status and body text)
```

MCP-side engine: the official SDK's **public low-level** `mcp.server.lowlevel.Server` +
`StreamableHTTPSessionManager` — no private-attribute access.

## Files

| File | What |
|---|---|
| `autoexpose.py` | the whole engine: route harvest, `/seed_session` wrap, token mint/verify, the two MCP handlers, the ASGI replay bridge, `/mcp` mount |
| `servers.py` | builds test servers from the **unmodified in-tree** `resources_servers/` (finance = typed routes, workplace = 27-tool dispatcher, aviary = plumbing case when `fhaviary` is installed) |
| `run_checks.py` | live acceptance suite — real uvicorn servers driven by the official MCP client |

## Run it

```bash
# from the repo root, with the dev env installed
python prototypes/mcp_auto_exposure/run_checks.py
```

Expected: `37/37 checks passed` (finance + workplace). Install `fhaviary` to add the aviary
plumbing-hazard case (7 more checks). The suite proves, against **pristine `origin/main` handler
code imported directly from the tree**: R1 (zero decorators), R2 (handlers run over MCP with
`request.session` working, sharing per-session state with the HTTP door), R3 (typed schemas
harvested, dispatcher via the override, aviary's `step`/`close` exposed), R4 (tool-route HTTP
responses byte-identical; only the two documented additive deltas), plus per-session
`allowed_tools` filtering and cross-session isolation.

## Known trade-offs (for the discussion, not hidden)

- **Replay is a 2x in-process pass** (middleware + routing + validation run again). Measured
  ~260-290 us/call on a real server — dominated by Gym's own `add_session_id` `BaseHTTPMiddleware`
  (~191 us), which the normal HTTP door also pays. Negligible under any real tool (web search,
  sandbox, model turn), but real for megabyte payloads or extreme call rates.
- **`aviary` shows the exposure hazard**: `/step` and `/close` are agent/harness plumbing, not
  model-facing tools, yet auto-exposure advertises them — and `/step`'s body carries `env_id`, so
  an auto-exposed `step` lets a model address other rollouts' environments. This is why exposure
  should likely be opt-in per server, and why a toolless-catch-all / plumbing-route declaration is
  needed (finance's `mcp_toolless_catchall_paths` is a first cut).
- **Schema harvest reads `route.body_field`**, a semi-internal FastAPI attribute whose spelling has
  moved across versions; the code fails loudly if it can't resolve it (rather than silently
  advertising a wrong schema).
- **Direct dispatch (no replay) is being explored separately** as a possible middle ground —
  synthesizing `gym_tool` registrations from the routes so the reviewed engine serves MCP without
  the replay pass.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
