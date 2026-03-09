# GRL Tetris Resource Server

FastAPI-based resource server that exposes the GRL Tetris environment through NeMo Gym conventions. The environment logic lives under `resources_servers/grl_tetris/tetris_env` and is a standalone adaptation of the upstream GRL implementation.

## Why it exists
- **Domain**: Classic falling-block Tetris on a configurable grid.
- **Evaluation**: Agents must clear at least one line; `/verify` rewards the cumulative score only when the environment reports success.
- **Independence**: No runtime dependency on the GRL repository—the environment is vendored and self-contained.

## Setup

Please follow the setup instructions as outlined in: https://github.com/NVIDIA-NeMo/Gym/blob/main/docs/tutorials/02-setup.md#step-1-clone-and-install.

## Running
Spin up the server alongside a compatible agent:
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/grl_tetris/configs/grl_tetris.yaml"
ng_run "+config_paths=[$config_paths]"
```

Collect trajectories:
```bash
ng_collect_rollouts +agent_name=grl_tetris_simple_agent \
    +input_jsonl_fpath=resources_servers/grl_tetris/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/grl_tetris/data/example_rollouts.jsonl \
    +limit=5
```

Launch the rollout viewer:
```bash
ng_viewer +jsonl_fpath=resources_servers/grl_tetris/data/example_rollouts.jsonl
```

## Tests
```bash
pytest resources_servers/grl_tetris/tests
```

## Licensing
- Code: Apache 2.0
- Data: Apache 2.0

---

## Reward Profiling Results

### Qwen3-4B

**Dataset**: 3,200 rollouts (200 prompts × 16 repeats)

**Performance Metrics**:
- **Success Rate**: 5.09% (163/3,200 rollouts)
- **Mean Reward**: -0.29 (range: -2.00 to 19.20)
- **Median Reward**: -0.80

**Key Findings**:
- Most rollouts (21%) received reward of -0.90 (piece dropped without clearing lines)
- Successful line clears achieved rewards of ~9.0-9.2
- Average 7.48 tool calls per rollout
- Weak negative correlation between tool calls and reward (-0.06)

**Top Reward Distribution**:
- `-0.9`: 672 rollouts (21.0%) - piece dropped, no line clear
- `-0.8`: 603 rollouts (18.8%)
- `-0.7`: 495 rollouts (15.5%)
- `9.1`: 29 rollouts (0.9%) - successful line clear
- `8.9`: 26 rollouts (0.8%)

The relatively low success rate (5.09%) suggests that Tetris line-clearing is challenging for the model, requiring precise spatial reasoning and action sequencing. Most rollouts result in pieces dropping without clearing lines (negative rewards from -0.1 per action step).
