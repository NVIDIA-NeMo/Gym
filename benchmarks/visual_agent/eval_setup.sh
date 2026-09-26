#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sourced in the eval container before Gym starts (EVAL_SETUP_SCRIPT, see sbatch_external_vllm.sh).
# The visual_agent resources server and the OpenCode visual agent need exactly what the prebuilt
# terminal_bench_2_1 / opencode_sandboxed_agent venvs hold (nemo-gym[dev,sandbox] installed
# editable from /opt/Gym), so point Gym at those. Building them at job start means an editable
# install of the whole repo from a network filesystem; on 2026-09-24 that hit uv's 300 s cache-lock timeout and
# killed the smoke job before any rollout ran.
_visual_agent_link_venv() {
    local source_venv=/opt/uv_venvs/$1/.venv target_dir=/opt/uv_venvs/$2
    if [[ -x "$source_venv/bin/python" && ! -e "$target_dir/.venv" ]]; then
        mkdir -p "$target_dir"
        ln -s "$source_venv" "$target_dir/.venv"
        echo "visual_agent eval setup: $target_dir/.venv -> $source_venv"
    fi
}
_visual_agent_link_venv resources_servers/terminal_bench_2_1 resources_servers/visual_agent
_visual_agent_link_venv responses_api_agents/opencode_sandboxed_agent responses_api_agents/opencode_visual_sandboxed_agent
_visual_agent_link_venv responses_api_agents/opencode_sandboxed_agent responses_api_agents/visual_replay_agent
unset -f _visual_agent_link_venv
