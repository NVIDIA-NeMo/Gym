# Environment adapter

This internal Resources Server runs trusted environments defined by `environment.yaml`.

It loads and validates tasks, owns a fresh sandbox for each episode, invokes the verifier selected for that task, and cleans up the episode.
It supports singleton tasks, JSONL tasksets with shared instructions and verification, and directory-based tasks with per-task instructions and verifiers.
Managed MCP processes remain to be implemented.

For file-backed tasksets, each JSONL row contains `task_id` and `task_data`.
The environment defines one shared `instruction`, `task_model`, and `verifier`; instruction placeholders use `{{ task_data.field }}`, and an exact `task_data.field` verifier-input value resolves to that row's validated data.
For directory-based tasks, each entry under `tasksets.<name>.tasks` selects its own instruction and verifier, and its mapping key is the default task ID.

Environment authors use the files in their environment directory rather than configuring this server or writing a replacement.
A custom Resources Server is only needed for behavior outside the standard environment lifecycle, such as externally managed durable services, specialized seeding, or authoritative state that the adapter cannot own.
