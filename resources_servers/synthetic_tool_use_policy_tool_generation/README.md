# Synthetic Tool-Use Policy And Tool Generation

This Gym resource server owns the second stage of the [synthetic tool-use pipeline](../synthetic_tool_use/README.md).
Its `app.py` exposes `POST /generate` and calls configured policy/tool and judge `ModelServerRef`s.

The package owns:

- `app.py`: Gym server configuration and HTTP route
- `stage.py`: generation, refinement, validation, judging, and artifact writes
- `profiles.py` and `profiles/`: general and proactive behavior profiles
- `rendering.py`: deterministic prompt inputs and reference sampling
- `prompts/` and `references/`: active prompts, archived prompt iterations, and golden references

Requests may select an inclusive `domain_start` and exclusive `domain_end`. Completed `policy.md` and `tools.jsonl`
artifacts are validated before resume skips a domain.
