# Description

1. Environment: This is a tool use - multi step agentic environment that tests the agents ability to execute tasks in a workplace setting. Workbench contains a sandbox environment with five databases, 26 tools, and 690 tasks. These tasks represent common business activities, such as sending emails and scheduling meetings.
2. Domain: Business activities
3. Source of prompts: 
- Full set of prompts (1260): https://huggingface.co/datasets/Nexusflow/250319-workbench-fulleval/viewer/default/train?row=0
4. Example prompt: Reply to carlos's last email about 'Task Update on Develop prototype for report generation' with 'Thanks for the update - I will get back to you tomorrow.'
5. Verifier: The verifier accuracy is checked by running the same benchmark on the corresponding already implemented Workbench environment in (VERL)[https://gitlab-master.nvidia.com/nexus-team/verl/-/blob/jk/reasoning-adherence-wip/verl/third_party/environments/workbench/workben_env.py?ref_type=heads]. 
6. Legal Approval Status: TBD

Rollouts - 
Link: https://huggingface.co/datasets/Nexusflow/abhibha-traj-coll-workbench

Commands - 
Spin up server:

```
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/workbench/configs/workbench.yaml"
ng_run "+config_paths=[$config_paths]"
```

Collect trajectories:
```
ng_collect_rollouts +agent_name=workbench_simple_agent \
    +input_jsonl_fpath=resources_servers/workbench/data/train.jsonl \
    +output_jsonl_fpath=results/workbench_trajectory_collection.jsonl \
   +limit=1
```

Data links: https://gitlab-master.nvidia.com/bxyu/nemo-gym/-/ml/models/55/versions/69#/

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
