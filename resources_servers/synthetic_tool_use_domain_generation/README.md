# Synthetic Tool-Use Domain Generation

This Gym resource server owns the first stage of the [synthetic tool-use pipeline](../synthetic_tool_use/README.md).
Its `app.py` exposes `POST /generate`, calls the configured domain `ModelServerRef`, and writes accepted candidates to
the shared manifest-backed run directory.

The package owns:

- `app.py`: Gym server configuration and HTTP route
- `stage.py`: generation, parsing, deduplication, and artifact writes
- `rendering.py`: follow-up prompt rendering
- `assets.py` and `prompts/`: prompt loading and prompt assets

The response includes the current generation report. Domain generation always operates on the run as a whole;
downstream servers support source-index partitioning.
