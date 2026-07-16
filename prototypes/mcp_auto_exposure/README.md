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
(`session_id: str` + typed params) and dispatches MCP calls directly, at the cost of ~861 lines
changed across 16 server files. This prototype changes **zero** tool files and **also dispatches
directly** (below) — so it aims to combine both designs' wins.

## The mechanism: direct dispatch (default) + replay fallback

An MCP `tools/call` must run a frozen `request: Request` handler. Two strategies, chosen per route
by a startup **detector** (`hybrid_dispatch.py`):

**Direct dispatch (the default, no second pass).** The handler is called exactly once, with a
fabricated `Request` whose `session` is served by a public `Request` subclass overriding the
documented `.session` property (FastAPI's own "Custom Request and Route Class" pattern) — no
cookie mint, no middleware pass, no ASGI re-entry. Measured ~5–7 µs/call vs ~270 µs for replay
(~40× cheaper). Uses only public FastAPI/Starlette/pydantic surface.

**Replay fallback (correctness net).** Where the detector cannot *prove* direct == a real HTTP
request — a server with **custom middleware**, or a handler using FastAPI dependency-injection /
streaming / other shapes direct dispatch doesn't reproduce — that route (or the whole server) falls
back to re-issuing the call as an internal in-process HTTP request through the full ASGI stack
(httpx-free). A `git grep` of all in-tree resources servers shows none currently need the fallback,
but it guarantees correctness never depends on the fast path.

Both paths: verify the signed session token (`X-NeMo-Gym-Session-Token`; itsdangerous, same
salt/secret derivation as `base_resources_server.py`'s MCP token) → resolve the session →
run the handler → map the result to MCP (`2xx` JSON → content + structuredContent; otherwise →
`isError` carrying the status and body text). MCP-side engine: the official SDK's **public
low-level** `mcp.server.lowlevel.Server` + `StreamableHTTPSessionManager` — no private-attr access.

In the acceptance run below, all 32 tools (finance's 5 + workplace's 27) dispatch **direct, zero
replay**; adding a custom middleware flips that server to replay automatically.

## Files

| File | What |
|---|---|
| `autoexpose.py` | the engine: route harvest, `/seed_session` wrap, token mint/verify, the two MCP handlers, direct-dispatch + replay-fallback wiring, `/mcp` mount |
| `hybrid_dispatch.py` | the direct dispatcher + the two-gate detector (server-level middleware audit, route-level signature bind) that picks direct vs replay per route |
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

- **Direct dispatch's residue is one non-public spelling.** The read side (`request.session`) is
  public; the fabricated request serves the session via a public `Request` subclass. The detector
  reads `app.user_middleware` (a stable Starlette attribute, not in the docs index) to decide
  direct-safety; the fully-public productization is for `SimpleServer` to wrap the app's public
  `add_middleware` and record what env authors add. No FastAPI DI internals (`solve_dependencies`,
  `body_field`, `dependant`) are touched.
- **Replay fallback keeps its 2x cost** where it fires (custom middleware, unsupported handler
  shapes) — ~260-290 us/call, dominated by Gym's own `add_session_id` `BaseHTTPMiddleware` (~191 us,
  which the HTTP door also pays). No in-tree server needs it today; a front-door split (routing
  `/mcp` before the middleware) would remove the wasted pass-1 session-mint for both paths.
- **`aviary` shows the exposure hazard**: `/step` and `/close` are agent/harness plumbing, not
  model-facing tools, yet auto-exposure advertises them — and `/step`'s body carries `env_id`, so
  an auto-exposed `step` lets a model address other rollouts' environments. This is why exposure
  should likely be opt-in per server, and why a toolless-catch-all / plumbing-route declaration is
  needed (finance's `mcp_toolless_catchall_paths` is a first cut).
- **Detector hardening is required before repo-wide rollout.** The two gates refuse (fall back to
  replay) on route-level `Depends()`, non-Gym middleware, and unsupported parameter shapes; ad-hoc
  probing found further constructions it should also refuse. As-built it dispatches the current
  in-tree servers correctly, but the classifier needs an audit plus a per-route replay default for
  anything it has not proven.
- **If adopted, the `@gym_tool` migration could revert.** Because this engine synthesizes the
  equivalent registration from each existing route, the ~861 lines PR #2002 changed across 16 tool
  files would become unnecessary — one engine, zero tool-file changes, direct dispatch.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
