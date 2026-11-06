# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SEGALE Ray actor pool for the longmt_eval resource server.

Mirrors the wmt_translation CometActor pattern. Each _SegaleActor holds
LASER2 and COMETKiwi resident in GPU memory across calls.

Call _build_segale_actor_class() once after Ray.init() to get the remote
class, then instantiate one actor per GPU on the gym node.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict

import ray

from nemo_gym import CACHE_DIR
from nemo_gym.config import CACHE_DIR_KEY_NAME, maybe_get_global_config_dict


LOG = logging.getLogger(__name__)


def _cache_dir() -> Path:
    """Return the configured Gym cache directory without triggering a config parse."""
    global_config_dict = maybe_get_global_config_dict()
    configured = global_config_dict.get(CACHE_DIR_KEY_NAME) if global_config_dict is not None else None
    return Path(configured) if configured else CACHE_DIR


def _mirror_python(cache_env_var: str, default_cache: str | Path) -> Path:
    """Mirror the venv's uv Python install to a shared-FS path for Ray workers.

    Identical strategy to wmt_translation/app.py: uv ships python-build-
    standalone binaries whose absolute paths change across containers. Copying
    the whole python root to a stable shared-FS location lets remote Ray
    workers find a py_executable that resolves at runtime.
    """
    venv_python = Path(sys.executable).resolve()
    if not venv_python.exists():
        raise RuntimeError(f"sys.executable not found: {venv_python}")
    uv_python_root = venv_python.parent.parent

    cache_root = Path(os.environ.get(cache_env_var, default_cache))
    mirrored_root = cache_root / uv_python_root.name
    mirrored_bin = mirrored_root / "bin" / venv_python.name

    if not mirrored_bin.exists():
        LOG.info("Mirroring uv Python %s -> %s for cross-node Ray actors", uv_python_root, mirrored_root)
        mirrored_root.parent.mkdir(parents=True, exist_ok=True)
        tmp = mirrored_root.with_suffix(".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(uv_python_root, tmp, symlinks=True)
        tmp.rename(mirrored_root)

    return mirrored_bin


def _build_segale_actor_class(actors_per_gpu: int = 1, use_extra_gpu: bool = False):
    """Build the _SegaleActor @ray.remote class. Must be called after Ray.init().

    Built lazily so importing this module does not require Ray to be
    initialised (mirrors _build_comet_actor_class in wmt_translation/app.py).

    actors_per_gpu controls how many actors share one physical GPU.

    use_extra_gpu selects the Ray resource mode:
      False (default): actors claim fractional num_gpus so Ray manages
        CUDA_VISIBLE_DEVICES. Use this when the gym runs its own Ray cluster
        with dedicated GPU nodes (HTTP-separated from vLLM).
      True: actors claim the custom extra_gpu resource (num_gpus=0) and manage
        GPU visibility themselves via RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES.
        Use this when the gym joins the vLLM Ray cluster and a separate node has
        been registered with `ray start --resources='{"extra_gpu": N}'`.
    """
    from setup_scoring import ensure_scoring_python

    cache_dir = _cache_dir()
    scoring_python = ensure_scoring_python(cache_dir / "longmt-scoring")
    mirrored_bin = _mirror_python(
        "LONGMT_EVAL_PY_CACHE",
        cache_dir / "longmt-python",
    )

    venv_dir = Path(sys.executable).parent.parent
    site_packages = venv_dir / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"

    env_vars: Dict[str, str] = {
        "PYTHONPATH": f"{site_packages}:{os.environ.get('PYTHONPATH', '')}",
        "LONGMT_CACHE_DIR": str(cache_dir),
    }
    if use_extra_gpu:
        # Ray thinks this node has 0 GPUs; preserve physical CUDA_VISIBLE_DEVICES
        # so the actor can still access its assigned GPU directly.
        env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
    # HF_HOME: root HuggingFace cache dir; hub cache derives from it.
    # HF_HUB_OFFLINE: when 1, blocks all HF Hub network calls.
    # HF_HUB_CACHE: overrides just the hub model-cache location.
    # TRANSFORMERS_CACHE: legacy transformers cache dir (deprecated; forwarded for compat).
    # LASER_HOME: dir for LASER2 weights
    # ERSATZ: dir for ersatz segmenter weights
    for key in ("HF_HOME", "HF_HUB_OFFLINE", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        if os.environ.get(key):
            env_vars[key] = os.environ[key]
    env_vars["LASER_HOME"] = os.environ.get("LASER_HOME", str(cache_dir / "longmt-laser"))
    env_vars["ERSATZ"] = os.environ.get("ERSATZ", str(cache_dir / "longmt-ersatz"))
    env_vars["LONGMT_COMET_CACHE"] = os.environ.get("LONGMT_COMET_CACHE", str(cache_dir / "longmt-comet"))

    gpu_fraction = 1 / actors_per_gpu
    if use_extra_gpu:
        ray_kwargs = dict(num_gpus=0, resources={"extra_gpu": gpu_fraction})
    else:
        ray_kwargs = dict(num_gpus=gpu_fraction)

    class _SegaleActorImpl:
        def __init__(self, gpu_idx: int, comet_model: str, comet_batch_size: int, embed_batch_size: int) -> None:
            from scoring_process import ScoringProcess

            self._scorer = ScoringProcess(
                scoring_python,
                Path(__file__).parent / "scoring" / "worker.py",
                config={
                    "gpu_idx": gpu_idx,
                    "comet_model": comet_model,
                    "comet_batch_size": comet_batch_size,
                    "embed_batch_size": embed_batch_size,
                },
            )

        def ping(self) -> bool:
            return True

        def score(self, source_text: str, mt_text: str, target_language: str) -> dict:
            return self._scorer.score(source_text, mt_text, target_language)

        def close(self) -> None:
            self._scorer.close()

    _SegaleActor = ray.remote(
        **ray_kwargs,
        runtime_env={"py_executable": str(mirrored_bin), "env_vars": env_vars},
    )(_SegaleActorImpl)
    return _SegaleActor
