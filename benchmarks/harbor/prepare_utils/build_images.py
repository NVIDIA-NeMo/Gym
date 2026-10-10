# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the environment images a Harbor benchmark's tasks start from.

Most Harbor tasks build their sandbox image from environment/Dockerfile. The harbor_tasks server never builds: it
starts the image `image_template` names for the environment's content hash. This command builds those images ahead
of a run, from the same benchmark config, with `docker buildx`. Identical environments share one image, and images
that already exist are skipped, so reruns only build what is missing.

    # Into the local Docker daemon, for the Docker sandbox provider:
    python -m benchmarks.harbor.prepare_utils.build_images --config benchmarks/harbor/hello_world/config.yaml --load
    # To the registry in image_template, for remote providers; a few tasks only:
    python -m benchmarks.harbor.prepare_utils.build_images --config <config.yaml> --push --task-names a b --jobs 4

`docker buildx` may use a remote builder (`docker buildx create --driver remote|kubernetes`), so this needs the
Docker CLI but not necessarily a local daemon for `--push`.
"""

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from benchmarks.harbor.prepare_utils.provisioning import describe_tasks, load_harbor_tasks_settings


def plan_builds(records: list[dict]) -> dict[str, str]:
    """Image reference -> environment directory for every supported task whose image is built, one per image."""
    builds: dict[str, str] = {}
    for record in records:
        if record["builds_image"] and not record["unsupported"]:
            builds.setdefault(record["image"], record["environment_dir"])
    return builds


def image_exists(image: str, *, push: bool) -> bool:
    command = ["docker", "buildx", "imagetools", "inspect", image] if push else ["docker", "image", "inspect", image]
    return subprocess.run(command, capture_output=True).returncode == 0


def build_image(image: str, environment_dir: str, *, push: bool) -> Optional[str]:
    """Build one image; return an error message instead of raising so one failure does not stop the rest."""
    command = ["docker", "buildx", "build", "--push" if push else "--load", "--tag", image, environment_dir]
    result = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        return f"{image}: {(result.stderr or result.stdout)[-2000:]}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="Harbor benchmark config, as passed to gym")
    parser.add_argument("--resources-server", help="harbor_tasks instance, when the config has several")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--load", action="store_true", help="load images into the local Docker daemon")
    destination.add_argument("--push", action="store_true", help="push images to the registry in image_template")
    parser.add_argument("--task-names", nargs="*", help="only these tasks")
    parser.add_argument("--limit", type=int, help="only the first N tasks of each dataset")
    parser.add_argument("--jobs", type=int, default=1, help="concurrent builds")
    parser.add_argument("--dry-run", action="store_true", help="list the images to build without building them")
    args = parser.parse_args()

    settings = load_harbor_tasks_settings(args.config, args.resources_server)
    if settings.image_template is None:
        parser.error(f"{settings.resources_server} sets no image_template, so no task image is built")
    builds = plan_builds(describe_tasks(settings, task_names=args.task_names, limit=args.limit))
    missing = {image: path for image, path in builds.items() if not image_exists(image, push=args.push)}
    print(f"{len(builds)} environment image(s) needed; {len(builds) - len(missing)} already built.")
    if args.dry_run or not missing:
        for image in missing:
            print(f"  to build: {image}")
        return

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {image: executor.submit(build_image, image, path, push=args.push) for image, path in missing.items()}
        errors = []
        for image, future in futures.items():
            error = future.result()
            print(f"  {'FAILED' if error else 'built'}: {image}")
            if error:
                errors.append(error)
    if errors:
        raise SystemExit("Failed builds:\n" + "\n".join(errors))


if __name__ == "__main__":
    main()
