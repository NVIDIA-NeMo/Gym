#!/bin/bash

command='
    source .venv/bin/activate
    gym --version
'
sbatch \
    --nodes=2 \
    --account=nemotron_n4_post \
    --partition=batch \
    --gres=gpu:4 \
    --time=04:00:00 \
    --job-name=nemotron_3_ultra-evals-$USER \
    --exclusive \
    scripts/sbatch_base.sh bash -lc $command
