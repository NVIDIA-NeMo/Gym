# Browserbase WebVoyager

This resources server maps the canonical BrowserEnv DOM tool surface into a native NeMo Gym environment.

It uses:
- Browserbase for browser sessions
- Stagehand for DOM-native browser actions
- the standard Gym `simple_agent` loop for multi-step tool calling

The server keeps browser/session handles hidden in server-side session state keyed by Gym's per-episode session cookie.

## Tool Surface

- `navigate(url)`
- `observe(instruction)`
- `act(instruction)`
- `extract(instruction, schema_json)`

All four tools return plain-text outputs so the model sees the same string-level semantics as the canonical BrowserEnv DOM mode.

## Verification

The initial verifier is intentionally lightweight: it grades the rollout by comparing the final assistant answer against `expected_answer` from the dataset row.

This is enough for smoke tests and rollout plumbing. If you need judge-model verification later, extend `verify()` in `app.py`.

## Credentials

Set these environment variables before running the server unless you pass direct config overrides:

- `BROWSERBASE_API_KEY`
- `BROWSERBASE_PROJECT_ID`
- `MODEL_API_KEY`

## License

Example data in `data/example.jsonl` is synthetic and provided under the repository's Apache 2.0 license.
