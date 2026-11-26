#!/bin/bash
set -e
set -x  # Enable debug output

# Variables
setup_dir=$SETUP_DIR
miniforge_dir=$MINIFORGE_DIR
openhands_dir=$OPENHANDS_DIR
agent_framework_repo=$AGENT_FRAMEWORK_REPO
agent_framework_commit=$AGENT_FRAMEWORK_COMMIT

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
# poetry pinned below 2.1: poetry>=2.1 has a real regression (python-poetry/poetry#10184) where
# marker evaluation crashes with "Could not parse version constraint: <value>" on any Linux
# kernel release string that isn't valid PEP440. poetry 2.0.1 confirmed not affected.
echo "Installing conda packages (this may take 5-10 minutes)..."
mamba install -y --override-channels conda-forge::python=3.12 conda-forge::nodejs "conda-forge::poetry=2.0.1" conda-forge::tmux conda-forge::git

# [CUSTOM-PEP668 2026-10-14] conda-forge's python=3.12 package now ships the PEP 668
# EXTERNALLY-MANAGED marker (a recent conda-forge change -- this plain pip install used to
# work fine), so a bare `pip install` into this conda env is now refused. Safe to override
# here: this is a throwaway, OpenHands-only miniforge env, not a shared system Python.
$miniforge_dir/bin/python -m pip install -q --break-system-packages 'packaging==26.0'

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

# [CUSTOM-PATCHES] This vendored OpenHands checkout is patched in place (literal
# find-and-replace against the OpenHands code). Each patch script is idempotent, all applied
# unconditionally here, regardless of which dataset(s) are actually used during training:
#   - swesmith_dataset_type: adds a dedicated 'SWE-smith' DATASET_TYPE
#   - swerebench_v1_skip_entry: adds plain SWE-rebench (V1) to SKIP_ENTRY_SCRIPT_TYPES, so
#     OpenHands' default entry script doesn't destroy app.py's pre-populated workspace bind mount.
#     Must run after the SWE-smith patch above -- both touch the same tuple.
#   - isolate_shared_venv_by_default: makes every dataset (not just language=="python" ones)
#     isolate its agent actions from the shared, persistent OpenHands orchestrator venv --
#     fixes a real cross-job venv corruption incident (one rollout's pip activity leaking into
#     another's).
#   - swerebench_path_order: fixes that isolation branch's PATH order (/usr/bin was put ahead of
#     /usr/local/bin) for any dataset still falling through to it. Must run after
#     isolate_shared_venv_by_default, which is what introduces the text this one edits.
#   - swerebench_conda_activate: SWE-rebench (V1) and SWE-rebench-V2 were wrongly assumed not to
#     use a conda testbed-style environment and fell through to the generic isolation branch
#     above with no task-env activation at all -- confirmed directly against real pulled images
#     that V1 ships a "testbed" env (same convention as SWE-Gym) and V2 ships one named
#     "swebench_matterhorn". Adds dedicated activation branches for both. Must run after
#     isolate_shared_venv_by_default (same reason as swerebench_path_order).
#   - pythonunbuffered: forces PYTHONUNBUFFERED=1 so a crash during run_infer.py's own startup
#     surfaces a real traceback
#   - swebench_pin: fixes poetry.lock's `swebench` git dependency pin, which points at a commit
#     that's no longer fetchable. Must run before `make build` below.
echo "Applying CUSTOM-SWESMITH dataset-type patch to OpenHands' run_infer.py..."
python3 "$(dirname "${BASH_SOURCE[0]}")/patch_openhands_swesmith_dataset_type.py" "$openhands_dir"

echo "Applying CUSTOM-SWEREBENCH-V1-SKIP-ENTRY patch to OpenHands' run_infer.py..."
python3 "$(dirname "${BASH_SOURCE[0]}")/patch_openhands_swerebench_v1_skip_entry.py" "$openhands_dir"

echo "Applying CUSTOM-ISOLATE-BY-DEFAULT patch to OpenHands' run_infer.py..."
python3 "$(dirname "${BASH_SOURCE[0]}")/patch_openhands_isolate_shared_venv_by_default.py" "$openhands_dir"

echo "Applying CUSTOM-SWEREBENCH-PATH-ORDER patch to OpenHands' run_infer.py..."
python3 "$(dirname "${BASH_SOURCE[0]}")/patch_openhands_swerebench_path_order.py" "$openhands_dir"

echo "Applying CUSTOM-SWEREBENCH-CONDA-ACTIVATE patch to OpenHands' run_infer.py..."
python3 "$(dirname "${BASH_SOURCE[0]}")/patch_openhands_swerebench_conda_activate.py" "$openhands_dir"

echo "Applying CUSTOM-PYTHONUNBUFFERED patch to OpenHands' run_infer.sh..."
python3 "$(dirname "${BASH_SOURCE[0]}")/patch_openhands_pythonunbuffered.py" "$openhands_dir"

echo "Applying CUSTOM-SWEBENCH-PIN patch to OpenHands' poetry.lock..."
python3 "$(dirname "${BASH_SOURCE[0]}")/patch_openhands_swebench_pin.py" "$openhands_dir"

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
#
# [CUSTOM-MAKE-BUILD-RETRY-FIX 2026-10-14] The final-attempt branch below used to never
# increment `attempt` or exit the loop on failure, so `while [ attempt -le MAX ]` stayed true
# forever. Now `attempt` always increments and the loop always terminates; on total failure
# it exits 1 so `_run_setup_command`'s `assert return_code == 0` (app.py) catches it instead
# of silently limping forward into a broken OpenHands build.
MAX_MAKE_BUILD_ATTEMPTS=2
MAKE_BUILD_TIMEOUT_SECONDS=$((2 * 60))
MAKE_BUILD_TIMEOUT_MINUTES=$((MAKE_BUILD_TIMEOUT_SECONDS / 60))

attempt=1
build_ok=false
while [ "$attempt" -le "$MAX_MAKE_BUILD_ATTEMPTS" ]; do
    echo "Running make build (attempt $attempt/$MAX_MAKE_BUILD_ATTEMPTS)..."

    if [ "$attempt" -lt "$MAX_MAKE_BUILD_ATTEMPTS" ]; then
        if timeout "$MAKE_BUILD_TIMEOUT_SECONDS" make build; then
            echo "make build completed successfully."
            build_ok=true
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
        build_ok=true
        break
    fi

    exit_code=$?
    echo "make build failed on the final attempt with exit code $exit_code."
    attempt=$((attempt + 1))
done

if [ "$build_ok" != "true" ]; then
    echo "ERROR: make build did not succeed after $MAX_MAKE_BUILD_ATTEMPTS attempts." >&2
    exit 1
fi

# [CUSTOM-OPENHANDS-POETRY-RETRY-LOOP 2026-10-16] `poetry install` here can report exit code 0
# while silently leaving the venv incomplete. Loop install+verify until the SAME multi-package
# import check used by app.py's OpenHandsHarnessProcessor.setup() and
# scripts/prepare-env/setup_swe_agents_harnesses.sh actually passes, instead of trusting poetry's
# exit code alone. Keep this package list in sync with those two call sites.
echo "Installing Python dependencies (creating .venv in OpenHands directory)..."
MAX_POETRY_INSTALL_ATTEMPTS=5
poetry_attempt=1
poetry_install_ok=false
while [ "$poetry_attempt" -le "$MAX_POETRY_INSTALL_ATTEMPTS" ]; do
    echo "poetry install attempt $poetry_attempt/$MAX_POETRY_INSTALL_ATTEMPTS..."
    poetry install --no-interaction --no-root
    if .venv/bin/python -c "
from openhands.core.main import create_runtime
import PIL
import browsergym
import pandas
import reportlab
import gevent
import pptx
" 2>/dev/null; then
        echo "poetry install verified complete on attempt $poetry_attempt."
        poetry_install_ok=true
        break
    fi
    echo "poetry install attempt $poetry_attempt left the venv incomplete (failed real import check), retrying..."
    poetry_attempt=$((poetry_attempt + 1))
done

if [ "$poetry_install_ok" != "true" ]; then
    echo "ERROR: poetry install did not produce a working OpenHands venv after $MAX_POETRY_INSTALL_ATTEMPTS attempts." >&2
    .venv/bin/python -c "
from openhands.core.main import create_runtime
import PIL
import browsergym
import pandas
import reportlab
import gevent
import pptx
" 2>&1 | tail -10 >&2 || true
    exit 1
fi

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
