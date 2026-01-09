# GRL Sokoban Resource Server

Single-box Sokoban puzzle environment. The environment is implemented under `resources_servers/grl_sokoban/env`, mirroring the sokoban implementation in GRL repo (https://github.com/lmgame-org/GRL). The implementation uses gym-sokoban package (https://github.com/mpSchrader/gym-sokoban) which implements DeepMind's paper Imagination Augmented Agents for Deep Reinforcement Learning following the standard of https://gymnasium.farama.org.

## Why it exists
- **Domain**: Deterministic Sokoban puzzles.
- **Evaluation**: Agents must push a box onto its target with minimal invalid moves.
- **Verifier**: `/verify` rewards the cumulative Sokoban score only when `success` is reported by the environment.

## Setup

Please follow the setup instructions as outlined in: https://github.com/NVIDIA-NeMo/Gym/blob/main/docs/tutorials/02-setup.md#step-1-clone-and-install.

## Running
Spin up the server alongside a compatible agent:
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/grl_sokoban/configs/grl_sokoban.yaml"
ng_run "+config_paths=[$config_paths]"
```

Collect trajectories:
```bash
ng_collect_rollouts +agent_name=grl_sokoban_simple_agent \
    +input_jsonl_fpath=resources_servers/grl_sokoban/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/grl_sokoban/data/example_rollouts.jsonl \
    +limit=5
```

Launch the rollout viewer:
```bash
ng_viewer +jsonl_fpath=resources_servers/grl_sokoban/data/example_rollouts.jsonl
```

## Tests
```bash
pytest resources_servers/grl_sokoban/tests
```

## Licensing
- Code: Apache 2.0
- Data: Apache 2.0

---

## Reward Profiling Results

### Qwen3-4B

**Dataset**: 3,200 rollouts (200 prompts × 16 repeats)

**Performance Metrics**:
- **Success Rate**: 13.47% (431/3,200 rollouts)
- **Mean Reward**: 0.93 (range: -8.90 to 10.90)
- **Median Reward**: 0.00

**Key Findings**:
- Most rollouts (66.7%) received reward of 0.00 (no valid actions taken)
- Successful puzzle solutions achieved rewards of ~10.5-10.9
- Average 2.64 tool calls per rollout
- Moderate negative correlation between tool calls and reward (-0.23)

**Top Reward Distribution**:
- `0.0`: 2,134 rollouts (66.7%) - no valid actions or early termination
- `10.8`: 206 rollouts (6.4%) - successful puzzle completion
- `10.9`: 72 rollouts (2.2%) - successful puzzle completion
- `10.7`: 51 rollouts (1.6%) - successful puzzle completion
- Negative rewards: Invalid moves or non-optimal solutions

The moderate success rate (13.47%) indicates that Sokoban puzzle-solving requires spatial planning and understanding of box-pushing mechanics. Most failures result from the model not taking valid actions (reward 0.0), while successful completions achieve consistent high rewards (~10.5-10.9). The negative correlation between tool calls and reward suggests that longer sequences often lead to invalid moves or dead-end states.

### Qwen3-30B-A3B

**Dataset**: 3,200 rollouts (200 prompts × 16 repeats)

**Performance Metrics**:
- **Success Rate**: 38.56% (1,234/3,200 rollouts)
- **Mean Reward**: 4.00 (range: -5.40 to 10.90)
- **Median Reward**: 0.00

**Key Findings**:
- Most rollouts (43.9%) received reward of 0.00 (no valid actions taken)
- Successful puzzle solutions achieved rewards of ~10.5-10.9
- Average 2.10 tool calls per rollout
- Moderate positive correlation between tool calls and reward (0.22)

**Top Reward Distribution**:
- `0.0`: 1,405 rollouts (43.9%) - no valid actions or early termination
- `10.8`: 477 rollouts (14.9%) - successful puzzle completion
- `10.6`: 183 rollouts (5.7%) - successful puzzle completion
- `10.7`: 172 rollouts (5.4%) - successful puzzle completion
- `10.9`: 157 rollouts (4.9%) - successful puzzle completion
- Negative rewards: Invalid moves or non-optimal solutions

The higher success rate (38.56%) compared to Qwen3-4B indicates that the larger model performs significantly better at spatial planning and understanding box-pushing mechanics. While the majority of failures still result from not taking valid actions (reward 0.0), the model achieves nearly 3x the success rate of the 4B variant. The positive correlation between tool calls and reward suggests that the model can effectively use longer action sequences to solve puzzles.
