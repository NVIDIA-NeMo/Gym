# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build Luxor sandbox images from OCI images and emit a provider ``image_map``.

Luxor starts a sandbox from an image in its own registry, not from an OCI
reference, and a sandbox is a microVM on Luxor's base OS rather than a
container. This helper builds one Luxor template per OCI image: the build
downloads the image's filesystem and moves the chosen directories onto the
base image. It suits task images whose environment lives in a few directories,
such as SWE-bench (``/testbed`` and ``/opt/miniconda3``, the defaults here).
System packages installed elsewhere in the OCI image are not carried over.

    python -m nemo_gym.sandbox.providers.luxor.build \\
        --images-file images.txt --output image_map.yaml

Paste the generated mapping under the provider's ``create`` block. Builds are
provisioning operations that create Luxor templates and images; the provider
never runs them implicitly.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import math
import posixpath
import re
import shlex
from collections.abc import Iterable, Sequence
from typing import Any


LOGGER = logging.getLogger(__name__)

DEFAULT_PATHS = ("/testbed", "/opt/miniconda3")
DEFAULT_BASE_IMAGE = "sandbox"
DEFAULT_CPU_CORES = 2
DEFAULT_MEMORY_MB = 8192
DEFAULT_DISK_SIZE_GB = 30
DEFAULT_BUILD_TIMEOUT_S = 3600.0
TEMPLATE_TAG = "nemo-gym"

# A build with one run step at these defaults needs no Git context.
_BUILD_WORKDIR = "/home/tenki"
_STEP_TIMEOUT_S = 1800
_NAME_UNSAFE_RE = re.compile(r"[^a-z0-9-]")
_CRANE_VERSION = "v0.20.2"
_CRANE_SHA256 = {
    "x86_64": "c14340087103ba9dadf61d45acd20675490fd0ccbd56ac7901fc1b502137f44b",
    "aarch64": "aff0db48825124c9331ea310057214bd4e92c01aa2e414d539e9659841d9422a",
}


class LuxorImageBuildError(RuntimeError):
    """Raised when a Luxor image cannot be built from an OCI image."""


def _require_luxor_sdk() -> Any:
    try:
        import luxor_ai
    except ImportError as exc:
        raise ModuleNotFoundError(
            "The Luxor AI SDK (luxor-ai) is required to build Luxor images. Install nemo-gym[sandbox]."
        ) from exc
    return luxor_ai


def _validate(image: str, paths: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(image, str) or not image.strip() or any(char.isspace() for char in image):
        raise ValueError("image must be a non-empty OCI reference without whitespace")
    if isinstance(paths, str) or not paths:
        raise ValueError("paths must be a non-empty list of absolute directories")
    normalized = []
    for path in paths:
        if not isinstance(path, str) or not path.startswith("/") or posixpath.normpath(path) != path or path == "/":
            raise ValueError(f"path {path!r} must be an absolute, normalized directory other than /")
        normalized.append(path)
    return tuple(dict.fromkeys(normalized))


def derive_name(image: str, paths: Sequence[str] = DEFAULT_PATHS) -> str:
    """Return a deterministic Luxor template name for an image and its imported paths."""
    paths = _validate(image, paths)
    digest = hashlib.sha256(f"{image}|{'|'.join(paths)}".encode()).hexdigest()[:12]
    stem = image.rsplit("/", 1)[-1].split("@", 1)[0].lower()
    stem = _NAME_UNSAFE_RE.sub("-", stem).strip("-") or "image"
    # Luxor names match ^[a-z][a-z0-9-]{0,63}$.
    return f"gym-{stem[:46].strip('-')}-{digest}"


def import_script(image: str, paths: Sequence[str] = DEFAULT_PATHS) -> str:
    """Return the build step that copies ``paths`` out of ``image`` onto the base image."""
    paths = _validate(image, paths)
    moves = "\n".join(
        f"[ ! -e {shlex.quote(path)} ] || {{ echo 'destination {path} already exists' >&2; exit 1; }}\n"
        f"$SUDO mkdir -p {shlex.quote(posixpath.dirname(path))}\n"
        f"$SUDO mv {shlex.quote('/opt/.gym-oci-root' + path)} {shlex.quote(path)}"
        for path in paths
    )
    return f"""set -eu
SUDO=; [ "$(id -u)" = 0 ] || SUDO=sudo
arch=$(uname -m)
case "$arch" in
  x86_64) sum={_CRANE_SHA256["x86_64"]}; asset=x86_64 ;;
  aarch64) sum={_CRANE_SHA256["aarch64"]}; asset=arm64 ;;
  *) echo "unsupported architecture $arch" >&2; exit 1 ;;
esac
work=$(mktemp -d)
curl -fsSL -o "$work/crane.tgz" \\
  "https://github.com/google/go-containerregistry/releases/download/{_CRANE_VERSION}/go-containerregistry_Linux_$asset.tar.gz"
echo "$sum  $work/crane.tgz" | sha256sum -c -
tar -xzf "$work/crane.tgz" -C "$work" crane
$SUDO mkdir -p /opt/.gym-oci-root
"$work/crane" export --platform linux/amd64 {shlex.quote(image)} - | $SUDO tar -x -C /opt/.gym-oci-root
{moves}
$SUDO rm -rf /opt/.gym-oci-root "$work"
"""


async def _find_template(client: Any, name: str) -> Any | None:
    for template in await client.templates.list(tags=[TEMPLATE_TAG]):
        if template.name == name:
            return template
    return None


def _image_ref(build: Any) -> str | None:
    image = getattr(build, "image", None)
    if str(getattr(build, "state", "")).upper() != "READY" or image is None:
        return None
    return getattr(image, "digest_ref", None) or getattr(image, "ref", None)


async def build_image(
    image: str,
    *,
    client: Any,
    paths: Sequence[str] = DEFAULT_PATHS,
    base_image: str = DEFAULT_BASE_IMAGE,
    cpu_cores: int = DEFAULT_CPU_CORES,
    memory_mb: int = DEFAULT_MEMORY_MB,
    disk_size_gb: int = DEFAULT_DISK_SIZE_GB,
    build_timeout_s: float = DEFAULT_BUILD_TIMEOUT_S,
    skip_existing: bool = True,
    workspace_id: str | None = None,
) -> str:
    """Build one Luxor image from an OCI image and return its registry ref."""
    paths = _validate(image, paths)
    if isinstance(build_timeout_s, bool) or not math.isfinite(build_timeout_s) or build_timeout_s <= 0:
        raise ValueError("build_timeout_s must be a positive finite number")
    luxor_ai = _require_luxor_sdk()
    name = derive_name(image, paths)
    try:
        template = await _find_template(client, name)
        if template is not None and skip_existing:
            ref = _image_ref(template.latest_build)
            if ref:
                LOGGER.info("Luxor image %s already exists for %s; skipping build", ref, image)
                return ref
        if template is None:
            spec = (
                luxor_ai.TemplateSpec()
                .from_image(base_image)
                .workdir(_BUILD_WORKDIR)
                .run(import_script(image, paths), workdir=_BUILD_WORKDIR, timeout=_STEP_TIMEOUT_S)
                .resources(cpu_cores=cpu_cores, memory_mb=memory_mb, disk_size_gb=disk_size_gb)
            )
            template = await client.templates.create(
                workspace_id=workspace_id, name=name, spec=spec, tags=[TEMPLATE_TAG]
            )
        LOGGER.info("Building Luxor template %s from image %s", name, image)
        build = await client.templates.build(template, wait_for_completion=True, timeout=build_timeout_s)
    except asyncio.TimeoutError as exc:
        raise LuxorImageBuildError(
            f"Timed out after {build_timeout_s:g}s building Luxor template {name!r} from image {image!r}. "
            "The remote build may still be running; check Luxor before retrying."
        ) from exc
    except Exception as exc:
        raise LuxorImageBuildError(f"Failed to build Luxor template {name!r} from image {image!r}: {exc}") from exc
    ref = _image_ref(build)
    if not ref:
        raise LuxorImageBuildError(
            f"Luxor template {name!r} built from image {image!r} ended in state "
            f"{getattr(build, 'state', None)!r} without an image"
        )
    return ref


async def build_images(
    images: Iterable[str],
    *,
    client: Any | None = None,
    concurrency: int = 4,
    continue_on_error: bool = False,
    **kwargs: Any,
) -> dict[str, str]:
    """Build images for many OCI references, returning the provider's ``create.image_map``.

    With ``continue_on_error`` the failures are logged and omitted, so one bad
    image does not discard a long batch.
    """
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency <= 0:
        raise ValueError("concurrency must be a positive integer")
    owned = client is None
    if owned:
        client = _require_luxor_sdk().AsyncClient()

    unique_images = list(dict.fromkeys(images))
    semaphore = asyncio.Semaphore(concurrency)
    abort = asyncio.Event()
    mapping: dict[str, str] = {}

    async def _one(image: str) -> None:
        async with semaphore:
            # Set before the semaphore is released, so queued builds do not start after a failure.
            if abort.is_set():
                return
            try:
                mapping[image] = await build_image(image, client=client, **kwargs)
            except Exception as exc:  # noqa: BLE001 - reported per image below
                if not continue_on_error:
                    abort.set()
                    raise
                LOGGER.error("Luxor image build failed for %s: %s", image, exc)

    tasks = [asyncio.create_task(_one(image)) for image in unique_images]
    try:
        # A remote build Luxor already accepted cannot be recalled, so let started builds finish.
        results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        if owned:
            await client.close()
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return {image: mapping[image] for image in unique_images if image in mapping}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", action="append", default=[], help="OCI image to build (repeatable).")
    parser.add_argument("--images-file", help="File with one OCI image per line ('#' comments allowed).")
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help=f"Directory to import from each image (repeatable). Defaults to {' '.join(DEFAULT_PATHS)}.",
    )
    parser.add_argument("--base-image", default=DEFAULT_BASE_IMAGE)
    parser.add_argument("--cpu-cores", type=int, default=DEFAULT_CPU_CORES)
    parser.add_argument("--memory-mb", type=int, default=DEFAULT_MEMORY_MB)
    parser.add_argument("--disk-size-gb", type=int, default=DEFAULT_DISK_SIZE_GB)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--build-timeout-s", type=float, default=DEFAULT_BUILD_TIMEOUT_S)
    parser.add_argument("--workspace-id", default=None)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild even if a ready image already exists.")
    parser.add_argument("--continue-on-error", action="store_true", help="Skip failures instead of aborting.")
    parser.add_argument(
        "--output",
        help="Write the image -> Luxor ref mapping here (.yaml or .json). Defaults to stdout as JSON.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)

    images = list(args.image)
    if args.images_file:
        with open(args.images_file) as handle:
            images.extend(line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#"))
    if not images:
        raise SystemExit("No images given: pass --image and/or --images-file.")

    mapping = asyncio.run(
        build_images(
            images,
            paths=tuple(args.path) or DEFAULT_PATHS,
            base_image=args.base_image,
            cpu_cores=args.cpu_cores,
            memory_mb=args.memory_mb,
            disk_size_gb=args.disk_size_gb,
            concurrency=args.concurrency,
            build_timeout_s=args.build_timeout_s,
            workspace_id=args.workspace_id,
            skip_existing=not args.rebuild,
            continue_on_error=args.continue_on_error,
        )
    )

    if args.output and args.output.endswith((".yaml", ".yml")):
        import yaml

        with open(args.output, "w") as handle:
            yaml.safe_dump({"image_map": mapping}, handle, sort_keys=True)
    elif args.output:
        with open(args.output, "w") as handle:
            json.dump(mapping, handle, indent=2, sort_keys=True)
    else:
        print(json.dumps(mapping, indent=2, sort_keys=True))

    failed = len(dict.fromkeys(images)) - len(mapping)
    if failed:
        LOGGER.error("%d image(s) failed to build", failed)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
