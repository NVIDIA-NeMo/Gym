#!/bin/bash
# Start SWE Agent Server for Miles integration

set -e

cd /workspace/swe-agent/Gym

CONFIG_PATHS="resources_servers/mini_swe_agent/configs/mini_swe_agent.yaml,responses_api_models/openai_model/configs/openai_model.yaml"

ng_run "+config_paths=[$CONFIG_PATHS]" \
    '+mini_swe_simple_agent.responses_api_agents.mini_swe_agent.env=docker' \
    '+mini_swe_simple_agent.responses_api_agents.mini_swe_agent.cache_dir_template=/tmp/unused_for_docker.sif' \
    '+mini_swe_simple_agent.responses_api_agents.mini_swe_agent.concurrency=16' \
    '+mini_swe_simple_agent.responses_api_agents.mini_swe_agent.step_timeout=600' \
    '+mini_swe_simple_agent.responses_api_agents.mini_swe_agent.eval_timeout=1800' \
    '+mini_swe_simple_agent.responses_api_agents.mini_swe_agent.run_golden=False' \
    '+mini_swe_simple_agent.responses_api_agents.mini_swe_agent.skip_if_exists=False'
