# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""In-pod full-page screenshot pipeline: playwright inside the agent's own sandbox.

Only the EVALUATION path uses this. Training ships the workspace to the grading service and
lets it render; evaluation renders here instead, so that it does not depend on that service
running -- evaluation is usually done in an environment that has no such long-lived service.

``_SHOT_SCRIPT`` is kept **byte for byte**. Every non-obvious line in it encodes a failure
that was actually hit and diagnosed in a pod: proxy auth handed to the wrong playwright
field, a loopback bypass that the dict API silently drops, blank ``file://`` renders of
runtime-rendered pages, reveal-on-scroll content captured at opacity 0, sticky headers
painted at an in-flight scroll offset, parallax backgrounds going black below the fold. Read
the inline comments before changing any of it; none of these failures announce themselves --
they all arrive as a plausible-looking screenshot that scores badly.

The sandbox image must carry node, playwright and chromium. A missing chromium surfaces as
``render_failed``, which the caller masks, rather than as a bad score.
"""

from __future__ import annotations

import base64
import json
import os
import shlex


_SHOT_SCRIPT = r'''
import sys, json, base64, argparse
from playwright.sync_api import sync_playwright
ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--actions-b64", default="")
ap.add_argument("--proxy", default="")  # comma-separated failover chain; empty entry = direct
ap.add_argument("--vw", type=int, default=1440)
ap.add_argument("--vh", type=int, default=900)
ap.add_argument("--serve-root", default="")  # serve this dir over local http and rewrite --url onto it
a = ap.parse_args()
if a.serve_root:
    import threading, functools
    from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
    class _QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args, **kw):  # keep stdout clean for SHOT_B64 parsing
            pass
    _handler = functools.partial(_QuietHandler, directory=a.serve_root)
    _srv = ThreadingHTTPServer(("127.0.0.1", 0), _handler)
    threading.Thread(target=_srv.serve_forever, daemon=True).start()
    _page = a.url.rsplit("/", 1)[-1] if a.url.rsplit("/", 1)[-1].endswith(".html") else "index.html"
    a.url = "http://127.0.0.1:%d/%s" % (_srv.server_address[1], _page)
actions = json.loads(base64.b64decode(a.actions_b64).decode()) if a.actions_b64 else []
BASE_LAUNCH = {"headless": True, "args": ["--no-sandbox", "--disable-dev-shm-usage",
                                          "--proxy-bypass-list=127.0.0.1;localhost"]}

def _proxy_cfg(purl):
    from urllib.parse import urlparse as _up
    _pu = _up(purl)
    cfg = {"server": _pu.scheme + "://" + _pu.hostname + ":" + str(_pu.port)}
    if _pu.username:
        cfg["username"] = _pu.username
        cfg["password"] = _pu.password or ""
    return cfg

proxy_chain = [s.strip() for s in a.proxy.split(",")] if a.proxy else [""]
if a.proxy and "" not in proxy_chain:
    proxy_chain.append("")
FATAL_NET = ("ERR_PROXY_CONNECTION_FAILED", "ERR_TUNNEL_CONNECTION_FAILED",
             "ERR_SOCKS_CONNECTION_FAILED", "ERR_NAME_NOT_RESOLVED",
             "ERR_INTERNET_DISCONNECTED", "ERR_CONNECTION_TIMED_OUT",
             "ERR_CONNECTION_REFUSED", "ERR_CONNECTION_RESET", "ERR_TIMED_OUT",
             "ERR_EMPTY_RESPONSE")
errs = []
fatal = []  # fatal external request failures of the CURRENT load attempt
render_env = {"proxy_failed": False, "recovered": None, "attempts": [], "fatal": []}

def _on_request_failed(req):
    try:
        u = req.url or ""
        if not u.startswith("http") or u.startswith("http://127.0.0.1") or u.startswith("http://localhost"):
            return  # local server / file:// subresources are page defects, not env
        f = req.failure or ""
        if any(k in f for k in FATAL_NET):
            fatal.append(f + " " + u[:120])
    except Exception:
        pass

SETTLE_JS = """async () => {
  try {
    document.documentElement.style.setProperty('scroll-behavior', 'auto', 'important');
    if (document.body) document.body.style.setProperty('scroll-behavior', 'auto', 'important');
  } catch (e) {}
  if (document.fonts && document.fonts.ready) { try { await document.fonts.ready; } catch (e) {} }
  const H = () => Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);
  const step = Math.max(400, Math.floor(window.innerHeight * 0.8));
  for (let y = 0; y <= Math.min(H(), 30000); y += step) {
    window.scrollTo(0, y);
    await new Promise(r => setTimeout(r, 120));
  }
  window.scrollTo(0, H());
  await new Promise(r => setTimeout(r, 400));
  window.scrollTo(0, 0);
  await new Promise(r => setTimeout(r, 300));
}"""
FREEZE_CSS = ("*,*::before,*::after{animation-delay:0s !important;animation-duration:.01s !important;"
              "transition-delay:0s !important;transition-duration:.01s !important;"
              "background-attachment:scroll !important;scroll-behavior:auto !important;}")

with sync_playwright() as p:
    b = None
    pg = None

    def _open(purl):
        global b, pg
        if b is not None:
            try:
                b.close()
            except Exception:
                pass
        kw = dict(BASE_LAUNCH)
        if purl:
            kw["proxy"] = _proxy_cfg(purl)
        b = p.chromium.launch(**kw)
        pg = b.new_context(viewport={"width": a.vw, "height": a.vh}).new_page()
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errs.append("pageerror: " + str(e)))
        pg.on("requestfailed", _on_request_failed)

    def _load():
        del fatal[:]
        try:
            resp = pg.goto(a.url, wait_until="load", timeout=25000)
            if resp is not None and a.url.startswith("http") and resp.status >= 500:
                fatal.append("PAGE_HTTP_%d %s" % (resp.status, a.url[:120]))
        except Exception as e:
            errs.append("goto failed: " + str(e))
        pg.wait_for_timeout(2000)

    for i, purl in enumerate(proxy_chain):
        label = (purl.split("@")[-1] or "direct") if purl else "direct"
        try:
            _open(purl)
            _load()
        except Exception as e:
            errs.append("launch failed via " + label + ": " + str(e))
            del fatal[:]
            fatal.append("LAUNCH_FAILED " + label)
        if fatal:
            render_env["attempts"].append({"via": label, "fatal": fatal[:3]})
            if pg is not None and not fatal[0].startswith("LAUNCH_FAILED"):
                _load()
        if not fatal and pg is not None:
            if render_env["attempts"]:
                render_env["recovered"] = "via " + label
            break
    render_env["proxy_failed"] = bool(fatal) or pg is None
    render_env["fatal"] = fatal[:5]
    if pg is None:  # every route failed to even open a browser: report env, no shot
        sys.stdout.write("CONSOLE_ERRORS:" + json.dumps(errs[:15]) + "\n")
        sys.stdout.write("RENDER_ENV:" + json.dumps(render_env) + "\n")
        sys.exit(0)
    try:
        pg.evaluate(SETTLE_JS)
        pg.add_style_tag(content=FREEZE_CSS)
        pg.wait_for_timeout(300)
    except Exception as e:
        errs.append("settle failed: " + str(e))
    for act in actions:
        t = act.get("type")
        try:
            if t == "click":
                pg.click(act["selector"], timeout=5000)
            elif t == "fill":
                pg.fill(act["selector"], act.get("value", ""), timeout=5000)
            pg.wait_for_timeout(800)
        except Exception as e:
            errs.append("action " + json.dumps(act) + " failed: " + str(e))
    try:
        pg.evaluate("window.scrollTo(0, 0)")
        pg.wait_for_timeout(250)
    except Exception as e:
        errs.append("scroll-top failed: " + str(e))
    png = pg.screenshot(full_page=True, type="jpeg", quality=75)
    b.close()
sys.stdout.write("SHOT_B64:" + base64.b64encode(png).decode() + "\n")
sys.stdout.write("CONSOLE_ERRORS:" + json.dumps(errs[:15]) + "\n")
sys.stdout.write("RENDER_ENV:" + json.dumps(render_env) + "\n")
'''


def _build_shot_cmd(url: str, *, actions_b64: str = "", proxy: str = "") -> str:
    """Pod-side command that runs _SHOT_SCRIPT against `url`. Shared by ScreenshotTool
    (feeds the multimodal grader) and capture_fullpage (lands the JPEG to disk).

    ``WEBDEV_GRADE_HTTP=1`` switches ``file://`` rendering to a loopback http server inside
    the shot script (see ``--serve-root``) so runtime-rendered pages actually execute. **The
    launcher sets this on**, and it must stay on: browsers block module loading, fetch and
    XHR on the file protocol, so a React, Vue or ES-module deliverable renders as an empty
    body no matter how good it is -- measured at 44 of 182 shots blank, essentially all of
    them runtime-rendered. Since React is named in 30% of the training briefs, the old
    default was scoring a large share of the task pool as necessarily broken and teaching the
    policy to avoid the very stack the brief asked for.

    Scores across this boundary are NOT comparable: a React page goes from certainly-0 to
    gradeable. Set it to 0 only to reproduce a pre-flip run.
    """
    script_b64 = base64.b64encode(_SHOT_SCRIPT.encode()).decode()
    serve_root = ""
    if os.environ.get("WEBDEV_GRADE_HTTP") == "1" and url.startswith("file://"):
        serve_root = os.path.dirname(url[len("file://") :]) or "/"
    return (
        f"echo {script_b64} | base64 -d > /tmp/_webdev_shot.py && "
        f"python3 /tmp/_webdev_shot.py --url {shlex.quote(url)} "
        f"--actions-b64 {shlex.quote(actions_b64)} --proxy {shlex.quote(proxy)}"
        + (f" --serve-root {shlex.quote(serve_root)}" if serve_root else "")
    )


def _parse_shot_b64(r: dict) -> tuple[str, str]:
    """Pull (SHOT_B64, CONSOLE_ERRORS) out of an env.execute result; empty b64 on miss."""
    out = r.get("output") or ""
    shot_b64, console_errors = "", "[]"
    for line in out.splitlines():
        if line.startswith("SHOT_B64:"):
            shot_b64 = line[len("SHOT_B64:") :].strip()
        elif line.startswith("CONSOLE_ERRORS:"):
            console_errors = line[len("CONSOLE_ERRORS:") :].strip()
    return shot_b64, console_errors


def _parse_render_env(r: dict) -> dict:
    """Pull the RENDER_ENV json out of an env.execute result. {} when absent
    (old pod script) or unparseable — callers treat that as env-healthy."""
    for line in (r.get("output") or "").splitlines():
        if line.startswith("RENDER_ENV:"):
            try:
                return json.loads(line[len("RENDER_ENV:") :].strip())
            except Exception:
                return {}
    return {}
