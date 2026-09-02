# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build Apptainer SIF images from OCI references, ahead of any sandbox create."""

import asyncio
import logging
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path

from nemo_gym.sandbox.providers.apptainer.provider import (
    _apptainer_subprocess_env,
    _require_apptainer,
    _resolve_image,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_BUILD_TIMEOUT_S = 3600.0


class ApptainerImageBuildError(RuntimeError):
    """Raised when a SIF cannot be built from an image."""


def is_usable_sif(path: Path) -> bool:
    """Whether ``path`` is a complete SIF to reuse rather than rebuild.

    A bare ``exists()`` would accept a truncated file from an interrupted build,
    which then fails at instance start where the cause is far less obvious.
    """
    try:
        st = path.stat()
    except OSError:
        return False
    return stat.S_ISREG(st.st_mode) and st.st_size > 0


async def build_sif(
    image: str,
    target: Path,
    *,
    binary: str,
    subprocess_env: Mapping[str, str],
    build_timeout_s: float = DEFAULT_BUILD_TIMEOUT_S,
    attempts: int = 3,
    retry_delay_s: float = 2.0,
    skip_existing: bool = True,
) -> Path:
    """Build one SIF at ``target`` and return its path."""
    source_image = _resolve_image(image.strip())
    if skip_existing and is_usable_sif(target):
        LOGGER.info("apptainer image %s already built, skipping", target)
        return target

    failures: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            detail = await _build_once(source_image, target, binary, subprocess_env, build_timeout_s)
        except OSError as exc:
            # Staging and install errors belong on the retry path.
            detail = str(exc)
        if detail is None:
            LOGGER.info("Built apptainer image %s from %s", target, source_image)
            return target
        failures.append(f"attempt {attempt}/{attempts}: {detail[-500:]}")
        if attempt < attempts:
            LOGGER.warning("apptainer build failed for %s, retrying: %s", target, detail)
            await asyncio.sleep(retry_delay_s * attempt)

    raise ApptainerImageBuildError(
        f"Failed to build apptainer image {str(target)!r} from {source_image!r}:\n" + "\n".join(failures)
    )


async def _build_once(
    source_image: str,
    target: Path,
    binary: str,
    subprocess_env: Mapping[str, str],
    build_timeout_s: float,
) -> str | None:
    """Run one build attempt. Returns None on success, else a failure detail.

    Staged then moved with :func:`os.replace` so a reader never sees a
    half-written SIF and a failed rebuild leaves any existing image untouched.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    build_dir = Path(tempfile.mkdtemp(prefix=f".{target.stem}-", dir=target.parent))
    staged = build_dir / target.name
    try:
        proc = await asyncio.create_subprocess_exec(
            binary,
            "build",
            "--force",
            str(staged),
            source_image,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=dict(subprocess_env),
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=build_timeout_s)
        except asyncio.TimeoutError:
            # A registry can accept the connection then stall on layer data.
            proc.kill()
            await proc.wait()
            return f"timed out after {build_timeout_s}s"

        if proc.returncode != 0:
            return err.decode(errors="replace").strip() or out.decode(errors="replace").strip() or "build failed"
        if not is_usable_sif(staged):
            return f"apptainer reported success without producing a usable {staged.name}"
        os.replace(staged, target)
        return None
    finally:
        try:
            shutil.rmtree(build_dir)
        except OSError as exc:
            # Leaked staging dirs are invisible otherwise, and they add up on a shared image_dir.
            LOGGER.warning("failed to clean staging dir %s: %s", build_dir, exc)


async def build_sifs(
    images: Mapping[str, str],
    image_dir: Path,
    *,
    concurrency: int = 4,
    continue_on_error: bool = False,
    **kwargs: object,
) -> dict[str, Path]:
    """Build ``{name: image}`` into ``image_dir``, returning ``{name: path}``.

    Raises ``RuntimeError`` before any work when apptainer is missing, so callers
    report that once rather than once per image. With ``continue_on_error`` a bad
    image is logged and omitted rather than discarding the batch.
    """
    if concurrency < 1:
        # Semaphore(0) would deadlock silently rather than error.
        raise ValueError(f"concurrency must be >= 1, got {concurrency!r}")
    for name in images:
        # Names come from dataset metadata, so a separator would escape image_dir.
        if not name.strip() or "/" in name or os.sep in name or name.strip(".") == "":
            raise ValueError(f"image name must be a single path component, got {name!r}")

    binary = _require_apptainer()
    env = _apptainer_subprocess_env(None)
    image_dir = Path(image_dir)

    semaphore = asyncio.Semaphore(concurrency)
    built: dict[str, Path] = {}

    async def _one(name: str, image: str) -> None:
        async with semaphore:
            built[name] = await build_sif(
                image, image_dir / f"{name}.sif", binary=binary, subprocess_env=env, **kwargs
            )

    # return_exceptions so one failure never orphans the in-flight builds.
    results = await asyncio.gather(*(_one(n, i) for n, i in images.items()), return_exceptions=True)
    for name, result in zip(images, results):
        if isinstance(result, BaseException):
            if not continue_on_error:
                raise result
            LOGGER.error("apptainer image build failed for %s: %s", name, result)
    return built
