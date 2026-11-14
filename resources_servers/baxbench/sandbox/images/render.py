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
"""Write the build context of each of the 14 BaxBench framework images from upstream's Dockerfiles.

Each image is the upstream framework Dockerfile up to the COPY of the generated code, with upstream's
common packages and every scenario's extra packages for that language, plus a harness layer under
``/opt/baxbench`` (Python 3.12, the test dependencies and the upstream source at the pinned commit) that
the app does not see on its PATH. Run from the repository root:

    python -m resources_servers.baxbench.sandbox.images.render --out resources_servers/baxbench/sandbox/images
"""

import argparse
from pathlib import Path

from resources_servers.baxbench.dockerfile_split import split_dockerfile
from resources_servers.baxbench.upstream import UPSTREAM_SHA, fetch_upstream, image_slug, import_upstream


IMAGES_DIR = Path(__file__).parent
CACHE_DIR = Path(__file__).parents[2] / ".upstream_cache"
UV_IMAGE = "ghcr.io/astral-sh/uv:0.13.5"
HARNESS_PYTHON = "3.12"
HARNESS_REQUIREMENTS = ("requests==2.34.2", "PyYAML==6.0.3", "imageio==2.38.1", "pdfplumber==0.11.10")

_HARNESS_LAYER = f"""
# ---- BaxBench harness for NeMo Gym: test code and its interpreter, off the app's PATH ----
COPY --from={UV_IMAGE} /uv /opt/baxbench/bin/uv
RUN UV_PYTHON_INSTALL_DIR=/opt/baxbench/python /opt/baxbench/bin/uv venv --python {HARNESS_PYTHON} /opt/baxbench/venv \\
    && /opt/baxbench/bin/uv pip install --python /opt/baxbench/venv/bin/python --no-cache {" ".join(HARNESS_REQUIREMENTS)}
ADD https://codeload.github.com/logic-star-ai/baxbench/tar.gz/{UPSTREAM_SHA} /opt/baxbench/upstream.tar.gz
RUN mkdir -p /opt/baxbench/upstream \\
    && tar -xzf /opt/baxbench/upstream.tar.gz --strip-components=1 -C /opt/baxbench/upstream \\
    && rm /opt/baxbench/upstream.tar.gz \\
    && echo {UPSTREAM_SHA} > /opt/baxbench/upstream/SHA \\
    && printenv -0 > /opt/baxbench/image.env0
"""

# Reproducibility fixes: upstream leaves these unpinned, so a rebuild today differs from upstream's
# leaderboard run (2025-12-09). Each replaces one upstream line and fails loudly if that line is gone.
_RUST_LOCK = (
    "# Lock crates that still build on Rust 1.83 (the newest actix dependencies need a newer rustc): a newer\n"
    "# cargo resolves with rust-version 1.83, the image keeps 1.83. Upstream's prebuild of a dummy main that\n"
    "# does not compile follows unchanged; it caches the compiled dependencies.\n"
    "RUN cp Cargo.toml Cargo.toml.orig \\\n"
    "    && sed -i '/^\\[package\\]/a rust-version = \"1.83\"' Cargo.toml \\\n"
    "    && rustup toolchain install 1.86.0 --profile minimal \\\n"
    "    && CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback cargo +1.86.0 generate-lockfile \\\n"
    "    && rustup toolchain uninstall 1.86.0 \\\n"
    "    && mv Cargo.toml.orig Cargo.toml\n"
    'RUN cargo build || echo "Build failed"'
)
# Grading has no network, but upstream's per-sample build replaces the generated Gemfile with its own and
# runs bundler install; install that Gemfile here so the step finds every gem. Lumen needs nothing: upstream's
# composer.json is not valid JSON, so its per-sample composer install always fails (hidden by `|| echo`) and
# the app runs on the create-project vendor, here as upstream.
_RAILS_GEMS = (
    "RUN rails new . --api\n"
    "# Upstream's Gemfile replaces this one in every sample before bundler install: install its gems now.\n"
    "COPY Gemfile Gemfile\n"
    "RUN bundler install"
)
IMAGE_PATCHES = {
    "Rust-Actix": [('RUN cargo build || echo "Build failed"', _RUST_LOCK)],
    # Rails from upstream's Gemfile (~> 8.0.1), newest patch on 2025-12-09; Lumen's last skeleton release.
    "Ruby-Rails": [
        ("RUN gem install rails", "RUN gem install rails -v 8.0.4"),
        ("RUN rails new . --api", _RAILS_GEMS),
    ],
    "PHP-Lumen": [
        (
            "RUN composer create-project --prefer-dist laravel/lumen .",
            "RUN composer create-project --prefer-dist laravel/lumen . 10.0.0",
        )
    ],
}


def _patch(env_id: str, text: str) -> str:
    for old, new in IMAGE_PATCHES.get(env_id, []):
        if old not in text:
            raise ValueError(f"{env_id}: upstream line to patch is gone: {old}")
        text = text.replace(old, new, 1)
    return text


def image_dockerfile(environment: object, scenarios: list[object], common_commands: list[str]) -> str:
    """Upstream's Dockerfile for ``environment`` up to the code COPY, with every scenario's packages."""
    additional = list(common_commands)
    for scenario in scenarios:
        for command in scenario.needed_packages.get("_all_", []) + scenario.needed_packages.get(
            environment.language, []
        ):
            if command not in additional:
                additional.append(command)
    split = split_dockerfile(environment.build_only_docker_image_file(additional))
    header = f"# Generated by render.py from upstream BaxBench {UPSTREAM_SHA} ({environment.id}). Do not edit.\n"
    return header + _patch(environment.id, "\n".join(split.image_lines).strip()) + "\n" + _HARNESS_LAYER


def render(out_dir: Path) -> list[Path]:
    bx_env, bx_scenarios = import_upstream(fetch_upstream(CACHE_DIR))
    from env.base import COMMON_DOCKER_RUN_COMMANDS

    written = []
    for environment in bx_env.all_envs:
        context = out_dir / image_slug(environment.id)
        context.mkdir(parents=True, exist_ok=True)
        (context / "Dockerfile").write_text(
            image_dockerfile(environment, bx_scenarios.all_scenarios, COMMON_DOCKER_RUN_COMMANDS)
        )
        for name, content in environment.manifest_files.items():
            (context / name).write_text(content)
        written.append(context)
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=IMAGES_DIR)
    for path in render(parser.parse_args().out):
        print(path)
