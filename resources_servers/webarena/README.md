# WebArena

This resources server runs Classic WebArena tasks in persistent Ray browser
actors. Each actor owns one Xvfb display and one Chromium process and leases a
fresh browser context to one episode at a time. `max_browser_workers` is the
global browser capacity of the single resources-server process.

The server supports the Classic `string_match`, `url_match`, and
`program_html` evaluators. Evaluation runs against the live page before the
episode context is returned to the pool.

## System requirements

- Linux with `Xvfb` and `xdpyinfo`
- Playwright Chromium and its native shared libraries
- writable `/tmp` for X11 sockets and Ray
- Access to a WebArena deployment

On Ubuntu:

```bash
apt-get update
apt-get install -y xvfb x11-utils
```

Create the Gym-managed server environment before installing Chromium. Playwright
browser revisions are version-specific, so installing from another Python
environment can download the wrong revision:

```bash
cd /path/to/Gym
source .venv/bin/activate
export RAY_TMPDIR=/tmp

ng_run \
  "+config_paths=[resources_servers/webarena/configs/webarena.yaml]" \
  ++webarena_resources_server.resources_servers.webarena.max_browser_workers=1 \
  ++webarena_resources_server.resources_servers.webarena.initial_browser_workers=1 \
  +dry_run=true

resources_servers/webarena/.venv/bin/python \
  -m playwright install --with-deps chromium
```

`--with-deps` requires root privileges and installs libraries such as
`libnspr4.so`. On a managed cluster, use a container that already includes the
Chromium runtime libraries if package installation is unavailable.

The WebArena requirements intentionally do not list `python-xlib`. `pyautogui`
already installs `python3-xlib` on Linux; installing both distributions is unsafe
because both own `Xlib/X.py`, and parallel uv installation can leave a mixed
package that fails during browser-worker startup.

Configure the deployment before running:

```bash
export WA_SHOPPING=http://shopping.example
export WA_SHOPPING_ADMIN=http://shopping-admin.example
export WA_REDDIT=http://reddit.example
export WA_GITLAB=http://gitlab.example
export WA_WIKIPEDIA=http://wikipedia.example
export WA_MAP=http://map.example
export WA_CLASSIFIEDS=http://classifieds.example
```

Optional LLM evaluator settings are `WEBARENA_JUDGE_MODEL`,
`WEBARENA_JUDGE_BASE_URL`, `WEBARENA_JUDGE_API_KEY`, and
`WEBARENA_JUDGE_TIMEOUT`.

Start this server together with a policy model:

```bash
export RAY_TMPDIR=/tmp
ng_run "+config_paths=[resources_servers/webarena/configs/webarena.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

The agent sends stable inline task payloads to `/seed_session`; the resource
server does not look tasks up by JSONL row index. Mutating calls carry request
IDs and use single-flight deduplication, so transport retries do not repeat
browser actions.

## Dependency troubleshooting

| Error | Cause and fix |
| --- | --- |
| `Xvfb` or `xdpyinfo` not found | Install `xvfb` and `x11-utils` in the host/container. |
| Playwright executable does not exist | Run `playwright install chromium` from `resources_servers/webarena/.venv/bin/python`. |
| Chromium cannot load `libnspr4.so` or another shared library | Run that same environment's `playwright install-deps chromium`, or use `install --with-deps chromium`. |
| `Xlib.X` lacks `FamilyServerInterpreted` | The venv contains overlapping `python-xlib` and `python3-xlib` files. Recreate the server venv after updating requirements; for an old venv, force-reinstall one implementation last. |
| Ray fails with long Unix-socket paths | Set `RAY_TMPDIR=/tmp`. |

The default config prewarms 16 browser actors. For a smoke test, override both
`max_browser_workers` and `initial_browser_workers` to `1`; keeping the initial
worker nonzero validates Xvfb and Chromium during server startup.
