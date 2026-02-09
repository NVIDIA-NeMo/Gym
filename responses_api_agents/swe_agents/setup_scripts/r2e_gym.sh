#!/bin/bash
set -e
set -x

# Variables
setup_dir=$SETUP_DIR
uv_dir=$UV_DIR
python_dir=$PYTHON_DIR
r2e_gym_dir=$R2E_GYM_DIR
eval_harness_repo=$EVAL_HARNESS_REPO
eval_harness_commit=$EVAL_HARNESS_COMMIT

cd $setup_dir

export UV_INSTALL_DIR="$uv_dir"
export UV_PYTHON_INSTALL_DIR="$python_dir"
if [ ! -f "$uv_dir/bin/uv" ]; then
    echo "Installing uv to $uv_dir..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "uv already installed at $uv_dir"
fi

export PATH="$uv_dir/bin:$PATH"
echo "Verifying uv installation..."
which uv
uv --version

# Clone R2E-Gym
if [ ! -d "$r2e_gym_dir/.git" ]; then
    echo "Cloning R2E-Gym..."
    # Clean up any partial clone
    rm -rf "$r2e_gym_dir"
    git clone $eval_harness_repo $r2e_gym_dir
else
    echo "R2E-Gym already cloned at $r2e_gym_dir"
fi

cd $r2e_gym_dir
echo "Checking out $eval_harness_commit..."
git checkout $eval_harness_commit

echo "Installing Python 3.12 to portable location..."
uv python install 3.12

echo "Python installations:"
uv python list

echo "Creating virtual environment with uv..."
rm -rf venv
uv venv --python 3.12 venv

echo "Installing R2E-Gym in editable mode..."
uv pip install -p $r2e_gym_dir/venv/bin/python -e . --no-cache

echo "Verifying installation..."
$r2e_gym_dir/venv/bin/python -c "import r2egym; print('✓ r2egym installed successfully')"

if [ -d venv ] && [ -f venv/bin/python ]; then
    echo "✓ venv created at $(pwd)/venv"
    echo "✓ Python version: $(venv/bin/python --version)"
else
    echo "✗ ERROR: venv was not created properly!"
    exit 1
fi

echo "R2E-Gym setup complete!"
