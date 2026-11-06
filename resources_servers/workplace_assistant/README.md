# Description

1. Environment: This is a tool use - multi step agentic environment that tests the agents ability to execute tasks in a workplace setting. Workplace assistant contains a sandbox environment with five databases, 27 tools, and 690 tasks. These tasks represent common business activities, such as sending emails and scheduling meetings.
2. Domain: Business activities
3. Source of prompts: 
- Full set of prompts (1260): https://huggingface.co/datasets/nvidia/Nemotron-RL-agent-workplace_assistant
4. Example prompt: Reply to carlos's last email about 'Task Update on Develop prototype for report generation' with 'Thanks for the update - I will get back to you tomorrow.'


Commands - 
Spin up server:

```
gym env start \
    --model-type openai_model \
    --resources-server workplace_assistant
```

Collect trajectories:
```
gym eval run --no-serve \
    --agent workplace_assistant_simple_agent \
    --input resources_servers/workplace_assistant/data/train.jsonl \
    --output results/workplace_assistant_trajectory_collection.jsonl \
   --limit 1
```

## Sessions

Each rollout gets its own copy of the five databases, created at `/seed_session`.

The shared `SimpleResourcesServer` lifecycle owns seed identity, locking, verification cleanup and retry records, close, and checkpoint retirement. Workplace supplies its state hooks and a seed adapter that validates typed task data against `TaskData`. Checkpoint export/restore remains specific to the database tables: typed identity advances with the restored attempt through the public restore API, and a typed seed may adopt a restored session that has no identity, such as one from an older flat checkpoint, without resetting its tables.

An Environment Server, such as `single_agent_turn`, seeds with a typed `ResourcesSeedSessionRequest` and closes with `/close_session`. The databases are keyed by the caller's `resources_session_id`; repeating an active seed returns the existing databases. `/verify` releases the databases and active binding, even if scoring raises, before checkpoint ownership ends. A compact retry record lets an HTTP verify retry replay the trajectory without retaining the databases. `/close_session` names the session, so it works without the seed's cookie; it releases the session only for the episode that owns it (409 otherwise), whether the session is active or already verified.

Closed and verified IDs cannot be seeded again for `session_close_retry_window_seconds` (default 300 seconds). The records expire after that fixed window; callers must use unique typed IDs and finish retries within it. A malformed typed seed or invalid workplace `task_data` gets 422.

A tool call after verify, close, or session loss gets 404 with `failure_kind: session_lost`. Verify retries are scored only while the server remembers the completed session and it has not been explicitly closed. Never-seeded, closed, restarted, or expired sessions return a masked sample (`reward: 0`, `mask_sample: true`).

Offline `gym eval reverify` is unsupported (`REVERIFY_MODE = UNSUPPORTED`): saved trajectories do not carry a live seeded session. `++force=true` bypasses the CLI guard but does not restore session provenance, so those requests are masked rather than rescored. This differs from an HTTP retry of a verify for a session the worker still remembers.

## Generating Additional Training Data

To generate your own training JSONL for this environment using NeMo Data Designer, see the [synthetic data generation example](notebooks/synthetic-data-generation/).

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0