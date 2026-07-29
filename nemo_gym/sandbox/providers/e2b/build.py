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

"""Build E2B templates from OCI images.

E2B cannot start a sandbox from an OCI reference: ``POST /sandboxes`` accepts
only a template ID, and an image reference is rejected outright (*"snapshot
alias only supports ASCII letters, digits, hyphens, and underscores"*).
Building a template from the image is the only path from an OCI reference to a
running sandbox.

This module is deliberately **outside** the sandbox public API and the
:class:`SandboxProvider` protocol. Building a template is a *provisioning*
step -- slow, one-off, and shared across runs -- whereas the provider API is
about starting and driving sandboxes. Keeping them apart means
:meth:`E2BProvider.create` never blocks on an image build, and provisioning can
run ahead of time from CI, a notebook, or the CLI below.

Typical use: build templates once, then feed the resulting mapping into the
provider's ``create.template_map``.

    python -m nemo_gym.sandbox.providers.e2b.build \\
        --image ghcr.io/acme/task-a:1.0 --image ghcr.io/acme/task-b:1.0 \\
        --cpu-count 8 --memory-mb 16384 --output template_map.yaml
"""

import argparse
import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Iterable, Sequence
from typing import Any


LOGGER = logging.getLogger(__name__)

# E2B template aliases accept ASCII letters, digits, hyphens and underscores.
_ALIAS_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]")

DEFAULT_CPU_COUNT = 2
DEFAULT_MEMORY_MB = 1024
DEFAULT_BUILD_TIMEOUT_S = 3600.0


class E2BTemplateBuildError(RuntimeError):
    """Raised when a template cannot be built from an image."""


def _require_e2b_sdk() -> Any:
    try:
        import e2b
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise ImportError(
            "Building e2b templates requires the 'e2b' package. Install it with `pip install 'e2b>=2.25.0'`."
        ) from exc
    return e2b


def derive_alias(image: str, cpu_count: int = DEFAULT_CPU_COUNT, memory_mb: int = DEFAULT_MEMORY_MB) -> str:
    """Return a deterministic, charset-safe alias for an image + build size.

    The digest covers the resources as well as the image because E2B bakes
    cpu/memory into the template: two sandboxes wanting different sizes must
    not share one, or the second silently inherits the first's sizing.
    """
    digest = hashlib.sha256(f"{image}|cpu={cpu_count}|mem={memory_mb}".encode()).hexdigest()[:12]
    stem = image.rsplit("/", 1)[-1].split("@", 1)[0].replace(":", "-")
    stem = _ALIAS_SAFE_RE.sub("-", stem).strip("-") or "image"
    return f"{stem[:48]}__{digest}"


async def template_exists(alias: str, **api_params: Any) -> bool:
    """Whether ``alias`` already exists on the target deployment."""
    e2b = _require_e2b_sdk()
    try:
        return bool(await e2b.AsyncTemplate.alias_exists(alias, **api_params))
    except Exception as exc:  # noqa: BLE001 - existence check is best-effort
        LOGGER.debug("e2b alias_exists(%s) failed: %s", alias, exc)
        return False


async def build_template(
    image: str,
    *,
    alias: str | None = None,
    cpu_count: int = DEFAULT_CPU_COUNT,
    memory_mb: int = DEFAULT_MEMORY_MB,
    build_timeout_s: float = DEFAULT_BUILD_TIMEOUT_S,
    registry_username: str | None = None,
    registry_password: str | None = None,
    skip_existing: bool = True,
    on_build_logs: Any = None,
    **api_params: Any,
) -> str:
    """Build one template from an OCI image and return its alias."""
    e2b = _require_e2b_sdk()
    resolved_alias = alias or derive_alias(image, cpu_count, memory_mb)

    if skip_existing and await template_exists(resolved_alias, **api_params):
        LOGGER.info("e2b template %s already exists; skipping build", resolved_alias)
        return resolved_alias

    LOGGER.info(
        "Building e2b template %s from image %s (cpu_count=%d, memory_mb=%d)",
        resolved_alias,
        image,
        cpu_count,
        memory_mb,
    )
    builder = e2b.AsyncTemplate().from_image(image, username=registry_username, password=registry_password)
    build_kwargs: dict[str, Any] = {
        "alias": resolved_alias,
        "cpu_count": cpu_count,
        "memory_mb": memory_mb,
        **api_params,
    }
    if on_build_logs is not None:
        build_kwargs["on_build_logs"] = on_build_logs
    try:
        await asyncio.wait_for(e2b.AsyncTemplate.build(builder, **build_kwargs), timeout=build_timeout_s)
    except asyncio.TimeoutError as exc:
        raise E2BTemplateBuildError(
            f"Timed out after {build_timeout_s}s building e2b template {resolved_alias!r} from image {image!r}."
        ) from exc
    except Exception as exc:
        raise E2BTemplateBuildError(
            f"Failed to build e2b template {resolved_alias!r} from image {image!r}: {exc}"
        ) from exc
    return resolved_alias


async def build_templates(
    images: Iterable[str],
    *,
    cpu_count: int = DEFAULT_CPU_COUNT,
    memory_mb: int = DEFAULT_MEMORY_MB,
    concurrency: int = 4,
    continue_on_error: bool = False,
    **kwargs: Any,
) -> dict[str, str]:
    """Build templates for many images, returning an image -> alias mapping.

    The result is exactly the shape of the provider's ``create.template_map``.
    With ``continue_on_error`` the failures are logged and omitted, so one bad
    image does not discard a long batch.
    """
    unique_images = list(dict.fromkeys(images))
    semaphore = asyncio.Semaphore(max(1, concurrency))
    mapping: dict[str, str] = {}

    async def _one(image: str) -> None:
        async with semaphore:
            try:
                mapping[image] = await build_template(image, cpu_count=cpu_count, memory_mb=memory_mb, **kwargs)
            except Exception as exc:  # noqa: BLE001 - reported per image below
                if not continue_on_error:
                    raise
                LOGGER.error("e2b template build failed for %s: %s", image, exc)

    await asyncio.gather(*(_one(image) for image in unique_images))
    return {image: mapping[image] for image in unique_images if image in mapping}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", action="append", default=[], help="OCI image to build (repeatable).")
    parser.add_argument("--images-file", help="File with one OCI image per line ('#' comments allowed).")
    parser.add_argument("--cpu-count", type=int, default=DEFAULT_CPU_COUNT)
    parser.add_argument("--memory-mb", type=int, default=DEFAULT_MEMORY_MB)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--build-timeout-s", type=float, default=DEFAULT_BUILD_TIMEOUT_S)
    parser.add_argument("--registry-username", default=None)
    parser.add_argument("--registry-password", default=None)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild even if the alias already exists.")
    parser.add_argument("--continue-on-error", action="store_true", help="Skip failures instead of aborting.")
    parser.add_argument(
        "--output",
        help="Write the image -> alias mapping here (.yaml or .json). Defaults to stdout as JSON.",
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
        build_templates(
            images,
            cpu_count=args.cpu_count,
            memory_mb=args.memory_mb,
            concurrency=args.concurrency,
            build_timeout_s=args.build_timeout_s,
            registry_username=args.registry_username,
            registry_password=args.registry_password,
            skip_existing=not args.rebuild,
            continue_on_error=args.continue_on_error,
        )
    )

    if args.output and args.output.endswith((".yaml", ".yml")):
        import yaml

        with open(args.output, "w") as handle:
            yaml.safe_dump({"template_map": mapping}, handle, sort_keys=True)
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
