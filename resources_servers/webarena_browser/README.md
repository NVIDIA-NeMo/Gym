# WebArena browser resource server

This component applies WebArena-specific site and evaluator policy to Gym's
shared `visual_browser` interaction runtime. It does not introduce a second
browser protocol: WebArena and WebVoyager both use headed Chromium, Playwright
for lifecycle/navigation, PyAutoGUI for visible coordinate input, the common
`computer_use` action envelope, and the same browser-session provider seam.

The resource-specific responsibilities are local-site URL substitution,
benchmark-account login, mutable-site locking, and the pinned WebArena-family
evaluators. VisualWebArena additionally resolves benchmark reference images
inside a confined read-only root. WebVoyager's public-site proxy, CAPTCHA, and
Gemini evidence judge remain isolated in `resources_servers/visual_browser`.

Collision plans capture selected API and live-page state before and after a
rollout so one task cannot silently receive credit for another task's
mutation. The evaluator source is pinned to reference revision
`3b775dc538931ead0cb6b4922349da9c6d493dab`; see
`reference_evaluation/PROVENANCE.md`.

For collision-planned VisualWebArena `program_html` targets, semantic freshness
is checked against the complete before and after observations. Retained context
such as negation remains part of each changed claim; an old matching phrase
cannot validate a new negative confirmation. Additional matching copies and
paraphrases can qualify, while merely moving existing content cannot.

The complete final observation must still satisfy every configured predicate.
When the baseline already passes, a new literal `must_include` occurrence can
establish freshness directly; otherwise the semantic check uses the configured
WebArena judge with both observations. This can add one judge request per
changed target record, with both observations included without truncation.
Judge transport or response errors leave verification inconclusive.

The literal shortcut deliberately retains the reference `must_include`
semantics: it counts required words or phrases without interpreting negation.
For `must_include: [saved]`, both `saved` → `saved\nnot saved` and
`saved\nnot` → `saved\nnot\nsaved` add a matching literal and score one.
The `fuzzy_match` path instead sends these complete observations to the
semantic judge. A new paraphrase without a new literal, including
`saved\nnot` → `saved\nnot\npersisted`, also goes to that judge; the retained
negation cannot be dropped to manufacture a positive claim.

One process supports one active session on one X display. Distributed Gym
workers scale through isolated resource-server processes or containers, not
threads sharing a display. Mutable WebArena deployments additionally require
an external reset/isolation policy between independent benchmark runs.

The runtime requires Xvfb, Chromium, PyAutoGUI, `xclip`, and the Playwright
browser revision matching `playwright==1.55.0`. Configure the self-hosted sites
with `WA_SHOPPING`, `WA_SHOPPING_ADMIN`, `WA_REDDIT`, `WA_GITLAB`,
`WA_WIKIPEDIA`, `WA_MAP`, `WA_CLASSIFIEDS`, and `WA_HOMEPAGE`. Model-backed
evaluator tasks also require `WEBARENA_JUDGE_API_KEY`; its endpoint and model
are selected through `WEBARENA_JUDGE_BASE_URL` and `WEBARENA_JUDGE_MODEL`.

Use `configs/webarena_browser.yaml`. Lifecycle logs are emitted under
`nemo_gym.resources_servers.webarena_browser` without credentials or complete
URL paths.
