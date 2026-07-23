# Synthetic Tool-Use Scenario Generation

This Gym resource server owns the third stage of the [synthetic tool-use pipeline](../synthetic_tool_use/README.md).
Its `app.py` exposes `POST /generate`, calls the configured scenario `ModelServerRef`, and writes validated scenario
JSONL shards to the shared run directory.

The package owns:

- `app.py`: Gym server configuration and HTTP route
- `stage.py`: scope scheduling, generation, validation, deduplication, and artifact writes
- `schema.py`: structured scenario models and generated schema
- `assets.py` and `prompts/`: system/user prompt and response-schema assets

Requests may select an inclusive `domain_start` and exclusive `domain_end`. Only domains with completed policy/tool
artifacts are eligible for scenario generation.
