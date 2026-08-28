# Native WebVoyager browser resource server

This component owns the public-site browser environment for the maintained
native WebVoyager profile. It exposes Gym's common web session API while
keeping WebVoyager-only behavior out of the WebArena/VisualWebArena runtime:

- headed Chromium with Playwright lifecycle and PyAutoGUI coordinate actions;
- selective or forced US proxy routing, including the DuckDuckGo HTML fallback
  used by rewritten Google Search tasks;
- CAPTCHA detection and CapSolver integration;
- WebVoyager init scripts, public-site navigation retries, and evidence capture.

The shared native driver supplies browser, action, screenshot, and artifact
mechanics. `resources_servers/native_web` supplies the separate local-site
WebArena and VisualWebArena policy. WebVoyager reward is not computed here:
the agent closes this browser after retaining immutable evidence, then sends a
standard `/verify` request to `resources_servers/webvoyager_judge`.

One process supports one live session on one X display. Scale by launching
isolated replicas with distinct `DISPLAY` values. The runtime image must
provide Xvfb, Chromium, `xclip`, and the benchmark fonts.

Runtime credentials are read from the environment:

- `WA_BROWSER_PROXY_SERVER`
- `CAPSOLVER_API_KEY`
- optional `WA_CAPTCHA_PROVIDER` or `WA_CAPTCHA_SOLVER`

Credentials, proxy authentication, CAPTCHA solution tokens, and complete URL
paths are never logged.

The native tool schema caps one scroll action at 50 wheel clicks, and the
shared driver clamps the value again at execution time. This prevents malformed
model output such as `scroll_amount=100000` from blocking a worker.
