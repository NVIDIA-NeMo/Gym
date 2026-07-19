#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
set -x  # Enable debug output

# Variables
setup_dir=$SETUP_DIR
miniforge_dir=$MINIFORGE_DIR
openhands_dir=$OPENHANDS_DIR
agent_framework_repo=$AGENT_FRAMEWORK_REPO
agent_framework_commit=$AGENT_FRAMEWORK_COMMIT
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
streaming_tool_call_patch="$script_dir/../patches/streaming_tool_call.patch"
streaming_tool_call_observability_patch="$script_dir/../patches/streaming_tool_call_observability.patch"
streaming_tool_call_admission_observability_patch="$script_dir/../patches/streaming_tool_call_admission_observability.patch"
streaming_tool_call_tokenizer_only_patch="$script_dir/../patches/streaming_tool_call_tokenizer_only.patch"
streaming_tool_call_valid_action_metrics_patch="$script_dir/../patches/streaming_tool_call_valid_action_metrics.patch"
openhands_runtime_breakdown_patch="$script_dir/../patches/openhands_runtime_breakdown.patch"
streaming_tool_call_prompt_reuse_patch="$script_dir/../patches/streaming_tool_call_prompt_reuse.patch"
streaming_tool_call_exact_incremental_tokenizer_patch="$script_dir/../patches/streaming_tool_call_exact_incremental_tokenizer.patch"
streaming_tool_call_action_timeout_patch="$script_dir/../patches/streaming_tool_call_action_timeout.patch"
streaming_tool_call_counterfactual_tokenizer_metrics_patch="$script_dir/../patches/streaming_tool_call_counterfactual_tokenizer_metrics.patch"
streaming_tool_call_final_only_incremental_tokenizer_patch="$script_dir/../patches/streaming_tool_call_final_only_incremental_tokenizer.patch"
streaming_tool_call_final_only_prefill_patch="$script_dir/../patches/streaming_tool_call_final_only_prefill.patch"
streaming_tool_call_prefill_race_patch="$script_dir/../patches/streaming_tool_call_prefill_race.patch"
streaming_tool_call_prefix_seed_metrics_patch="$script_dir/../patches/streaming_tool_call_prefix_seed_metrics.patch"
streaming_tool_call_prefill_after_admission_patch="$script_dir/../patches/streaming_tool_call_prefill_after_admission.patch"
streaming_tool_call_bucketed_long_poll_patch="$script_dir/../patches/streaming_tool_call_bucketed_long_poll.patch"
streaming_tool_call_server_timing_patch="$script_dir/../patches/streaming_tool_call_server_timing.patch"
streaming_tool_call_deferred_admission_patch="$script_dir/../patches/streaming_tool_call_deferred_admission.patch"
streaming_tool_call_fallback_metrics_patch="$script_dir/../patches/streaming_tool_call_fallback_metrics.patch"
streaming_tool_call_error_observation_patch="$script_dir/../patches/streaming_tool_call_error_observation.patch"
streaming_tool_call_request_timing_patch="$script_dir/../patches/streaming_tool_call_request_timing.patch"
streaming_tool_call_compact_request_context_patch="$script_dir/../patches/streaming_tool_call_compact_request_context.patch"
streaming_tool_call_model_call_attribution_patch="$script_dir/../patches/streaming_tool_call_model_call_attribution.patch"
streaming_tool_call_cancellable_long_poll_patch="$script_dir/../patches/streaming_tool_call_cancellable_long_poll.patch"
streaming_tool_call_effective_prefill_patch="$script_dir/../patches/streaming_tool_call_effective_prefill.patch"
streaming_tool_call_background_prefill_metrics_patch="$script_dir/../patches/streaming_tool_call_background_prefill_metrics.patch"
streaming_tool_call_cached_token_metrics_patch="$script_dir/../patches/streaming_tool_call_cached_token_metrics.patch"
streaming_tool_call_skip_unadmitted_finalization_patch="$script_dir/../patches/streaming_tool_call_skip_unadmitted_finalization.patch"
streaming_tool_call_deferred_abort_patch="$script_dir/../patches/streaming_tool_call_deferred_abort.patch"
streaming_tool_call_same_request_metrics_patch="$script_dir/../patches/streaming_tool_call_same_request_metrics.patch"
streaming_tool_call_event_driven_snapshot_patch="$script_dir/../patches/streaming_tool_call_event_driven_snapshot.patch"
streaming_tool_call_event_driven_snapshot_toggle_patch="$script_dir/../patches/streaming_tool_call_event_driven_snapshot_toggle.patch"
streaming_tool_call_snapshot_query_bool_patch="$script_dir/../patches/streaming_tool_call_snapshot_query_bool.patch"
streaming_tool_call_prefill_start_priority_patch="$script_dir/../patches/streaming_tool_call_prefill_start_priority.patch"

cd $setup_dir

# Install miniforge if not properly installed
if [ ! -f "$miniforge_dir/bin/conda" ] || [ ! -f "$miniforge_dir/bin/mamba" ]; then
    echo "Installing miniforge..."
    # Clean up any partial installation
    rm -rf "$miniforge_dir"
    rm -f Miniforge3-*.sh

    echo "Downloading miniforge..."
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

    echo "Running miniforge installer..."
    bash Miniforge3-$(uname)-$(uname -m).sh -b -p $miniforge_dir

    echo "Cleaning up installer..."
    rm Miniforge3-$(uname)-$(uname -m).sh
else
    echo "Miniforge already installed at $miniforge_dir"
fi

# Add conda to PATH and source conda setup
echo "Setting up conda environment..."
export PATH="$miniforge_dir/bin:$PATH"
source $miniforge_dir/etc/profile.d/conda.sh
conda activate base

# Verify conda and mamba are available
echo "Verifying conda installation..."
which conda
which mamba
conda --version
mamba --version

# Install required packages
echo "Installing conda packages (this may take 5-10 minutes)..."
mamba install -y --override-channels conda-forge::python=3.12 conda-forge::nodejs conda-forge::poetry conda-forge::tmux conda-forge::git

$miniforge_dir/bin/python -m pip install -q 'packaging==26.0'

# Install jq as a static binary (avoid conda solver changing other package versions)
if [ ! -f "$miniforge_dir/bin/jq" ]; then
    echo "Installing jq static binary..."
    curl -fsSL https://github.com/jqlang/jq/releases/download/jq-1.8.1/jq-linux-amd64 -o "$miniforge_dir/bin/jq"
    chmod +x "$miniforge_dir/bin/jq"
fi

echo "Verifying jq installation..."
which jq
jq --version || true


# Verify installations
echo "Verifying package installations..."
which python
which node
which poetry
which jq

# Clone OpenHands
if [ ! -d "$openhands_dir/.git" ]; then
    echo "Cloning OpenHands..."
    # Clean up any partial clone
    rm -rf "$openhands_dir"
    git clone $agent_framework_repo $openhands_dir
else
    echo "OpenHands already cloned at $openhands_dir"
fi

cd $openhands_dir
echo "Checking out $agent_framework_commit..."
git checkout $agent_framework_commit

echo "Applying NeMo Gym streaming tool-call integration..."
git apply --check "$streaming_tool_call_patch"
git apply "$streaming_tool_call_patch"
git apply --check "$streaming_tool_call_observability_patch"
git apply "$streaming_tool_call_observability_patch"
git apply --check "$streaming_tool_call_admission_observability_patch"
git apply "$streaming_tool_call_admission_observability_patch"
git apply --check "$streaming_tool_call_tokenizer_only_patch"
git apply "$streaming_tool_call_tokenizer_only_patch"
git apply --check "$streaming_tool_call_valid_action_metrics_patch"
git apply "$streaming_tool_call_valid_action_metrics_patch"
git apply --check "$openhands_runtime_breakdown_patch"
git apply "$openhands_runtime_breakdown_patch"
git apply --check "$streaming_tool_call_prompt_reuse_patch"
git apply "$streaming_tool_call_prompt_reuse_patch"
git apply --check "$streaming_tool_call_exact_incremental_tokenizer_patch"
git apply "$streaming_tool_call_exact_incremental_tokenizer_patch"
git apply --check "$streaming_tool_call_action_timeout_patch"
git apply "$streaming_tool_call_action_timeout_patch"
git apply --check "$streaming_tool_call_counterfactual_tokenizer_metrics_patch"
git apply "$streaming_tool_call_counterfactual_tokenizer_metrics_patch"
git apply --check "$streaming_tool_call_final_only_incremental_tokenizer_patch"
git apply "$streaming_tool_call_final_only_incremental_tokenizer_patch"
git apply --check "$streaming_tool_call_final_only_prefill_patch"
git apply "$streaming_tool_call_final_only_prefill_patch"
git apply --check "$streaming_tool_call_prefill_race_patch"
git apply "$streaming_tool_call_prefill_race_patch"
git apply --check "$streaming_tool_call_prefix_seed_metrics_patch"
git apply "$streaming_tool_call_prefix_seed_metrics_patch"
git apply --check "$streaming_tool_call_prefill_after_admission_patch"
git apply "$streaming_tool_call_prefill_after_admission_patch"
git apply --check "$streaming_tool_call_bucketed_long_poll_patch"
git apply "$streaming_tool_call_bucketed_long_poll_patch"
git apply --check "$streaming_tool_call_server_timing_patch"
git apply "$streaming_tool_call_server_timing_patch"
git apply --check "$streaming_tool_call_deferred_admission_patch"
git apply "$streaming_tool_call_deferred_admission_patch"
git apply --check "$streaming_tool_call_fallback_metrics_patch"
git apply "$streaming_tool_call_fallback_metrics_patch"
git apply --check "$streaming_tool_call_error_observation_patch"
git apply "$streaming_tool_call_error_observation_patch"
git apply --check "$streaming_tool_call_request_timing_patch"
git apply "$streaming_tool_call_request_timing_patch"
git apply --check "$streaming_tool_call_compact_request_context_patch"
git apply "$streaming_tool_call_compact_request_context_patch"
git apply --check "$streaming_tool_call_model_call_attribution_patch"
git apply "$streaming_tool_call_model_call_attribution_patch"
git apply --check "$streaming_tool_call_cancellable_long_poll_patch"
git apply "$streaming_tool_call_cancellable_long_poll_patch"
git apply --check "$streaming_tool_call_effective_prefill_patch"
git apply "$streaming_tool_call_effective_prefill_patch"
git apply --check "$streaming_tool_call_background_prefill_metrics_patch"
git apply "$streaming_tool_call_background_prefill_metrics_patch"
git apply --check "$streaming_tool_call_cached_token_metrics_patch"
git apply "$streaming_tool_call_cached_token_metrics_patch"
git apply --check "$streaming_tool_call_skip_unadmitted_finalization_patch"
git apply "$streaming_tool_call_skip_unadmitted_finalization_patch"
git apply --check "$streaming_tool_call_deferred_abort_patch"
git apply "$streaming_tool_call_deferred_abort_patch"
git apply --check "$streaming_tool_call_same_request_metrics_patch"
git apply "$streaming_tool_call_same_request_metrics_patch"
git apply --check "$streaming_tool_call_event_driven_snapshot_patch"
git apply "$streaming_tool_call_event_driven_snapshot_patch"
git apply --check "$streaming_tool_call_event_driven_snapshot_toggle_patch"
git apply "$streaming_tool_call_event_driven_snapshot_toggle_patch"
git apply --check "$streaming_tool_call_snapshot_query_bool_patch"
git apply "$streaming_tool_call_snapshot_query_bool_patch"
git apply --check "$streaming_tool_call_prefill_start_priority_patch"
git apply "$streaming_tool_call_prefill_start_priority_patch"

# Build OpenHands
echo "Building OpenHands (this may take 5-10 minutes)..."
export INSTALL_DOCKER=0


# Remove any cached virtualenvs from previous runs
# Use poetry's actual cache dir (respects XDG_CACHE_HOME) instead of hardcoded ~/.cache
echo "Removing any cached poetry virtualenvs..."
poetry_cache_dir="$(poetry config cache-dir 2>/dev/null || echo ~/.cache/pypoetry)"
rm -rf "$poetry_cache_dir"/virtualenvs/openhands-* || true

# CRITICAL: Unset any active virtualenv from the host .venv
# This prevents poetry from getting confused about which venv to use
echo "Unsetting host virtualenv to avoid poetry confusion..."
unset VIRTUAL_ENV
unset PYTHONHOME
# Remove any venv paths from PATH to ensure clean environment
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv' | tr '\n' ':' | sed 's/:$//')

# Configure poetry to create virtualenv in the project directory (so it's mounted in container)
export POETRY_VIRTUALENVS_IN_PROJECT=true

# Retry `make build` with a timeout guard on the first attempt
MAX_MAKE_BUILD_ATTEMPTS=2
MAKE_BUILD_TIMEOUT_SECONDS=$((2 * 60))
MAKE_BUILD_TIMEOUT_MINUTES=$((MAKE_BUILD_TIMEOUT_SECONDS / 60))

attempt=1
while [ "$attempt" -le "$MAX_MAKE_BUILD_ATTEMPTS" ]; do
    echo "Running make build (attempt $attempt/$MAX_MAKE_BUILD_ATTEMPTS)..."

    if [ "$attempt" -lt "$MAX_MAKE_BUILD_ATTEMPTS" ]; then
        if timeout "$MAKE_BUILD_TIMEOUT_SECONDS" make build; then
            echo "make build completed successfully."
            break
        fi

        exit_code=$?
        if [ "$exit_code" -eq 124 ]; then
            echo "make build timed out after $MAKE_BUILD_TIMEOUT_MINUTES minutes."
        else
            echo "make build failed with exit code $exit_code."
        fi

        echo "Retrying make build after cleanup..."
        make clean || true
        attempt=$((attempt + 1))
        continue
    fi

    if make build; then
        echo "make build completed successfully."
        break
    fi

    exit_code=$?
    echo "make build failed on the final attempt with exit code $exit_code."
done

# Install Python dependencies with poetry
echo "Installing Python dependencies (creating .venv in OpenHands directory)..."
poetry install --no-interaction --no-root

# Install datasets package
echo "Installing datasets package..."

poetry run python -m pip install datasets huggingface_hub packaging==26.0

mkdir -p evaluation/oh
mkdir -p logs
mkdir -p .eval_sessions

echo "Verifying .venv was created..."
if [ -d .venv ]; then
    echo "✓ .venv created at $(pwd)/.venv"
else
    echo "✗ ERROR: .venv was not created!"
    exit 1
fi

echo "OpenHands setup complete!"
