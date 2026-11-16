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
    --environment workplace_assistant
```

Collect trajectories:
```
gym eval run --no-serve \
    --agent workplace_assistant_simple_agent \
    --input environments/workplace_assistant/data/example.jsonl \
    --output results/workplace_assistant_trajectory_collection.jsonl \
   --limit 1
```

## Agent response budget

The bundled `workplace_assistant_simple_agent` configuration permits up to 10
model responses per task. This counts model responses, including a final answer,
rather than individual tool calls. For example, validation task `251` requires
eight customer updates and a search to discover the customer IDs; with
serial tool calls, that workflow uses 10 responses including the final answer.

This replaces the previous limit of six. Record the configured `max_steps` when
comparing results: runs with the two budgets are not directly comparable.
Ten responses provide more room for multi-record tasks but do not guarantee
completion when extra lookups, pagination or recovery are needed.

## Generating Additional Training Data

To generate your own training JSONL for this environment using NeMo Data Designer, see the [synthetic data generation example](../../resources_servers/workplace_assistant/notebooks/synthetic-data-generation/).

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0