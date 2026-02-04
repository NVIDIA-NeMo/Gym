#!/bin/bash
set -e
set -x

# Variables
setup_dir=$SETUP_DIR
uv_dir=$UV_DIR
python_dir=$PYTHON_DIR
swebench_dir=$SWEBENCH_DIR
swebench_repo=$SWEBENCH_REPO
swebench_commit=$SWEBENCH_COMMIT

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

# Clone SWE-bench
if [ ! -d "$swebench_dir/.git" ]; then
    echo "Cloning SWE-bench..."
    # Clean up any partial clone
    rm -rf "$swebench_dir"
    git clone $swebench_repo $swebench_dir
else
    echo "SWE-bench already cloned at $swebench_dir"
fi

cd $swebench_dir
echo "Checking out $swebench_commit..."
git checkout $swebench_commit

echo "Installing Python 3.12 to portable location..."
uv python install 3.12

echo "Python installations:"
uv python list

echo "Creating virtual environment with uv..."
rm -rf venv
uv venv --python 3.12 venv

echo "Installing SWE-bench..."
uv pip install -p $swebench_dir/venv/bin/python -e .

if [ -d venv ] && [ -f venv/bin/python ]; then
    echo "✓ venv created at $(pwd)/venv"
    echo "✓ Python version: $(venv/bin/python --version)"
else
    echo "✗ ERROR: venv was not created properly!"
    exit 1
fi

echo "SWE-bench setup complete!"
