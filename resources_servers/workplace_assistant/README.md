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

An Environment Server, such as `single_agent_turn`, seeds the session with a typed `ResourcesSeedSessionRequest` and closes it with `/close_session`. New configs should use this path. The databases are keyed by the caller's `resources_session_id`: a repeated seed returns the same databases, `/verify` leaves them for the close, and the close releases them even without the seed's cookie. A seed after its session closed gets 409, and a seed whose `task_data` is not a workplace_assistant task gets 422.

A tool call for a session the server no longer holds (never seeded, closed, or lost with a restart) gets 404 with `failure_kind: session_lost`, and its `/verify` returns a masked sample (`reward: 0`, `mask_sample: true`).

The `/run` seed is kept for compatibility. The configs in `configs/` still front `simple_agent` with the `legacy_agent` relay, so the agent's `/run` seeds with its run body under the session cookie. The server keeps accepting that body so these configs run unchanged until they switch. That caller never closes the session, so `/verify` releases its databases. To run the configs through `single_agent_turn_legacy` instead, switch them:

```bash
python scripts/add_legacy_agent_environment_servers.py --switch-simple-agent resources_servers/workplace_assistant/configs/
```

## Generating Additional Training Data

To generate your own training JSONL for this environment using NeMo Data Designer, see the [synthetic data generation example](notebooks/synthetic-data-generation/).

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0