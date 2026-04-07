#!/usr/bin/env bash
source /lustre/fsw/portfolios/llmservice/users/cmunley/gym-envclass/.venv/bin/activate
export RAY_TMPDIR=/tmp

CONFIGS="resources_servers/reasoning_gym_env/configs/reasoning_gym_env.yaml,\
resources_servers/workplace_assistant_env/configs/workplace_assistant_env.yaml,\
resources_servers/example_multi_turn_env/configs/example_multi_turn_env.yaml,\
resources_servers/blackjack_env/configs/blackjack_env.yaml,\
resources_servers/tictactoe_env/configs/tictactoe_env.yaml,\
responses_api_models/vllm_model/configs/vllm_model_training.yaml"

ng_run "+config_paths=[$CONFIGS]"
