#!/usr/bin/env bash
set -euo pipefail
# Each scorer is a fresh namespace without the evaluator's credentials,
# network, or user filesystem. The model answer is supplied through stdin.
: "${BWRAP:?}" "${AUTHOR_ROOT:?}" "${DATASET_ROOT:?}" "${AUTHOR_VENV:?}" "${AUTHOR_PYTHON_ROOT:?}" "${ADAPTER_ROOT:?}" "${VS_BENCH_GAME:?}"
case "$VS_BENCH_GAME" in
  hanabi|overcooked|knights_archers_zombies|breakthrough|kuhn_poker|atari_pong|simple_push|coin_dilemma|monster_hunt|battle_of_the_colors) ;;
  *) exit 2 ;;
esac
exec "$BWRAP" --unshare-all --new-session --die-with-parent --clearenv \
  --ro-bind /usr /usr --ro-bind-try /lib /lib --ro-bind-try /lib64 /lib64 \
  --ro-bind "$AUTHOR_ROOT" /reference \
  --ro-bind "$AUTHOR_VENV" /venv \
  --ro-bind "$AUTHOR_PYTHON_ROOT" /python \
  --ro-bind "$ADAPTER_ROOT" /adapter \
  --dir /data --dir /data/perception \
  --ro-bind "$DATASET_ROOT/perception/$VS_BENCH_GAME.parquet" "/data/perception/$VS_BENCH_GAME.parquet" \
  --proc /proc --dev /dev --tmpfs /tmp --chdir /reference \
  --setenv PATH /usr/bin --setenv HOME /tmp --setenv MPLCONFIGDIR /tmp/matplotlib \
  --setenv PYTHONHOME /python --setenv PYTHONPATH /venv/lib/python3.10/site-packages \
  --setenv OPENBLAS_NUM_THREADS 1 --setenv OMP_NUM_THREADS 1 --setenv MPLBACKEND Agg \
  -- /python/bin/python3.10 /adapter/reference_perception_score.py /reference /data
