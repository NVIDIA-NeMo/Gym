# WebArena

This benchmark is a thin Gym adapter over BrowserGym WebArena. The prepare
script converts the official 812 task configs into normalized `web_task` rows;
BrowserGym still owns task setup, authentication, Playwright execution, and the
native WebArena evaluator.

By default the script finds the sibling checkout used during development:

```text
../webarena/config_files/test.raw.json
```

For another layout, set `WEBARENA_SOURCE_CONFIG`. Then run:

```bash
gym eval prepare --benchmark webarena
```

Before rollout, deploy the WebArena websites and set all `WA_*` URLs documented
by `resources_servers/browsergym_web/README.md`. Of the official tasks, 82 use
semantic fuzzy matching and another 36 can invoke the unachievable-answer
judge. Pass `webarena_evaluator_model=<model>` and
`web_evaluator_base_url=<openai-compatible-url>`, and place the credential in
`OPENAI_API_KEY` (or select another name with `web_evaluator_api_key_env`). A
missing model on one of those 118 tasks is a masked configuration failure, not
a score of zero.

The first implementation is single-session because a fresh browser context
does not isolate mutable site state.
