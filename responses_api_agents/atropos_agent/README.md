# Atropos Agent

Integrates Nous Research RL Environment framework [Atropos](https://github.com/NousResearch/atropos) into NeMo Gym. This agent loads any Atropos `BaseEnv`, routes its inference through Gym's atropos model server, and returns rollouts with token IDs formatted for NeMo RL and other training framework integrations. 

## Setup

```
policy_base_url: "http://localhost:8000/v1"
policy_api_key: EMPTY
policy_model_name: "Qwen/Qwen3-4B-Instruct-2507"
```

## Example: Blackjack (multi-turn game)

```bash
ng_run "+config_paths=[responses_api_agents/atropos_agent/configs/atropos_blackjack.yaml,responses_api_models/atropos_model/configs/atropos_model.yaml]"

ng_collect_rollouts \
    +agent_name=atropos_blackjack_agent \
    +input_jsonl_fpath=responses_api_agents/atropos_agent/data/example_blackjackenvnothinking.jsonl \
    +output_jsonl_fpath=results/blackjack_rollouts.jsonl \
    +num_repeats=1
```

## Example: GSM8K (cohort-buffered math)

Some atropos envs, like GSM8K, require group size > 1 for proper reward calculation etc. In this case, we buffer requests in NeMo Gym in a cohort, so this requires NeMo Gym's `num_repeats >= atropos_group_size` so the agent can collect a full group for scoring. 
```bash
ng_run "+config_paths=[responses_api_agents/atropos_agent/configs/atropos_gsm8k.yaml,responses_api_models/atropos_model/configs/atropos_model.yaml]"

ng_collect_rollouts \
    +agent_name=atropos_gsm8k_agent \
    +input_jsonl_fpath=responses_api_agents/atropos_agent/data/example_gsm8kenv.jsonl \
    +output_jsonl_fpath=results/gsm8k_rollouts.jsonl \
    +num_repeats=4
```

## Adding a new environment

Create a config in `configs/`, pointing at the Atropos env module and class:

```yaml
atropos_myenv_agent:
  responses_api_agents:
    atropos_agent:
      entrypoint: app.py
      model_server:
        type: responses_api_models
        name: policy_model
      model_name: ${policy_model_name}
      atropos_env_module: environments.my_env
      atropos_env_class: MyEnv
      atropos_group_size: 1          # use >1 for envs with collect_trajectories only (i.e., those that require cohort apporach)
      atropos_env_config:
        max_token_length: 4096
        include_messages: true
      reward_function: atroposlib.envs.reward_fns.accuracy_reward:AccuracyReward
      max_tokens: 2048
      temperature: 1.0
```

Add example data in `data/`, then collect rollouts.

## Environment compatibility

Envs implementing `collect_trajectory` (singular) are called directly — one rollout per `/run` request. Envs implementing only `collect_trajectories` (plural) are cohort-buffered: requests accumulate until `atropos_group_size` arrive for the same prompt, then `collect_trajectories` runs once and results fan out. Set `num_repeats >= atropos_group_size` in `ng_collect_rollouts`.

Tested envs: BlackjackEnvNoThinking, GymTaxiEnv, GSM8kEnv, MathEnv, MCQAThinkingEnv, AnswerFormatEnv, LeanEnv, LeanProofEnv, SigmaRuleEnv, MetricCardEnv, PhilosophicalRLAIFEnv, RLAIFEnv, InterleavedInlineEnv.

# Licensing information
Code: Apache 2.0
Data: see individual env datasets
