# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prebuild staged Harbor task images with bounded concurrency on one backend."""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from environments.unified_video.prepare import backend_environment, client_environment


def build_task_images(folder: Path, *, workers: int = 8) -> None:
    """Build independent contexts concurrently, retaining Harbor's validated records."""
    from nemo_gym.data.tasks.harbor import ImageBuilder, build_images, plan_builds, refused_before_build

    if workers < 1:
        raise ValueError("Image build workers must be positive")
    conflicts = {}
    services = plan_builds(folder, conflicts=conflicts)
    refused = refused_before_build(services)
    if conflicts or refused:
        raise RuntimeError(f"Refusing invalid task image plan: conflicts={conflicts}, refused={refused}")
    # All services of one context stay together. Distinct contexts have distinct
    # image names; Harbor merges their records atomically under per-file locks.
    contexts = {entry.digest: entry for entry in services}

    def build(entry):
        tasks = set(entry.tasks)
        return build_images(
            folder,
            builder=ImageBuilder(tool="docker", platform="linux/arm64"),
            select=lambda name: name in tasks,
            contexts={entry.digest},
        )

    started = time.monotonic()
    print(f"IMAGE_PREPARATION_START contexts={len(contexts)} workers={workers}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(build, entry): digest for digest, entry in contexts.items()}
        try:
            for completed, future in enumerate(as_completed(futures), 1):
                future.result()
                print(
                    f"IMAGE_PREPARATION_PROGRESS completed={completed}/{len(contexts)} "
                    f"elapsed_seconds={time.monotonic() - started:.1f}",
                    flush=True,
                )
        except Exception:
            for pending in futures:
                pending.cancel()
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--docker-binary", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    docker = args.docker_binary.resolve()
    values = backend_environment(args.backend) | {
        "PATH": str(docker.parent) + os.pathsep + os.environ.get("PATH", ""),
        "DOCKER_CONFIG": str(docker.parent),
    }
    with client_environment(values):
        build_task_images(args.task_dir, workers=args.workers)


if __name__ == "__main__":
    main()
