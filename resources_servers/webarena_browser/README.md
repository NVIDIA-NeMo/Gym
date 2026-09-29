# WebArena browser resource server

This component applies WebArena-specific site and evaluator policy to Gym's
shared `visual_browser` interaction runtime. It does not introduce a second
browser protocol: WebArena and WebVoyager both use headed Chromium, Playwright
for lifecycle/navigation, PyAutoGUI for visible coordinate input, the common
`computer_use` action envelope, and the same browser-session provider seam.

The resource-specific responsibilities are local-site URL substitution,
benchmark-account login, mutable-site locking, and the pinned WebArena
evaluator. WebVoyager's public-site proxy, CAPTCHA, and Gemini evidence judge
remain isolated in `resources_servers/visual_browser`.

Collision plans capture selected API and live-page state before and after a
rollout so one task cannot silently receive credit for another task's
mutation. The evaluator source is pinned to reference revision
`3b775dc538931ead0cb6b4922349da9c6d493dab`; see
`reference_evaluation/PROVENANCE.md`.

One process supports one active session on one X display. Distributed Gym
workers scale through isolated resource-server processes or containers, not
threads sharing a display. Mutable WebArena deployments additionally require
an external reset/isolation policy between independent benchmark runs.

The runtime requires Xvfb, Chromium, PyAutoGUI, `xclip`, and the Playwright
browser revision matching `playwright==1.55.0`. Configure the self-hosted sites
with `WA_SHOPPING`, `WA_SHOPPING_ADMIN`, `WA_REDDIT`, `WA_GITLAB`,
`WA_WIKIPEDIA`, `WA_MAP`, and `WA_HOMEPAGE`. Model-backed evaluator tasks also
require `WEBARENA_JUDGE_API_KEY`; its endpoint and model are selected through
`WEBARENA_JUDGE_BASE_URL` and `WEBARENA_JUDGE_MODEL`.

Use `configs/webarena_browser.yaml`. Lifecycle logs are emitted under
`nemo_gym.resources_servers.webarena_browser` without credentials or complete
URL paths.
