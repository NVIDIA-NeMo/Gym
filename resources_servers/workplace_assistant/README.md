# Description

1. Environment: This is a tool use - multi step agentic environment that tests the agents ability to execute tasks in a workplace setting. Workplace assistant contains a sandbox environment with five databases, 26 tools, and 690 tasks. These tasks represent common business activities, such as sending emails and scheduling meetings.
2. Domain: Business activities
3. Source of prompts: 
- Full set of prompts (1260): https://huggingface.co/datasets/nvidia/Nemotron-RL-agent-workplace_assistant
4. Example prompt: Reply to carlos's last email about 'Task Update on Develop prototype for report generation' with 'Thanks for the update - I will get back to you tomorrow.'


Commands - 
Spin up server:

```
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/workplace_assistant/configs/workplace_assistant.yaml"
ng_run "+config_paths=[$config_paths]"
```

Collect trajectories:
```
ng_collect_rollouts +agent_name=workplace_assistant_simple_agent \
    +input_jsonl_fpath=resources_servers/workplace_assistant/data/train.jsonl \
    +output_jsonl_fpath=results/workplace_assistant_trajectory_collection.jsonl \
   +limit=1
```

Data links: 
Nemogym prompt datasets: https://gitlab-master.nvidia.com/bxyu/nemo-gym/-/ml/models/55/versions/98#/

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0