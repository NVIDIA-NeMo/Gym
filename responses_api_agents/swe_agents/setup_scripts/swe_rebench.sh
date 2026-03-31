#!/bin/bash
set -e
set -x

setup_dir=$SETUP_DIR
rebench_dir=$REBENCH_DIR

cd $setup_dir

# Clone SWE-rebench-V2 (contains log parsers needed for evaluation)
if [ ! -d "$rebench_dir/.git" ]; then
    echo "Cloning SWE-rebench-V2..."
    rm -rf "$rebench_dir"
    git clone https://github.com/SWE-rebench/SWE-rebench-V2.git "$rebench_dir"
else
    echo "SWE-rebench-V2 already cloned at $rebench_dir"
fi

cd "$rebench_dir"
echo "SWE-rebench-V2 setup complete!"
