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

"""Prepare SWE-bench Pro data and pin its per-instance evaluator assets."""

import json
import tarfile
import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path
from time import sleep
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote

from datasets import load_dataset

from nemo_gym.global_config import get_hf_token


UPSTREAM_COMMIT = "ca10a60a5fcae51e6948ffe1485d4153d421e6c5"  # pragma: allowlist secret
UPSTREAM_ARCHIVE_URL = f"https://github.com/scaleapi/SWE-bench_Pro-os/archive/{UPSTREAM_COMMIT}.tar.gz"
DATASET_REVISION = "7ab5114912baf22bb098818e604c02fe7ad2c11f"  # pragma: allowlist secret
EXPECTED_INSTANCE_COUNT = 731
BENCHMARK_DIR = Path(__file__).parent.parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "swebench_pro_benchmark.jsonl"
UPSTREAM_CACHE_DIR = DATA_DIR / "swebench_pro_upstream"


def fetch_upstream_assets(cache_dir: Path = UPSTREAM_CACHE_DIR) -> Path:
    """Download and extract the pinned evaluator source once."""
    root = cache_dir / f"SWE-bench_Pro-os-{UPSTREAM_COMMIT}"
    if root.is_dir():
        return root

    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / f"{UPSTREAM_COMMIT}.tar.gz"
    urllib.request.urlretrieve(UPSTREAM_ARCHIVE_URL, archive_path)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(cache_dir, filter="data")
    if not root.is_dir():
        raise FileNotFoundError(f"Expected extracted upstream directory at {root}")
    return root


def _read_asset(upstream_root: Path, relative_path: str) -> str:
    path = upstream_root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Missing SWE-bench Pro evaluator asset: {path}")
    return path.read_text(encoding="utf-8")


def fetch_image_digest(dockerhub_tag: str, max_attempts: int = 8) -> str:
    """Resolve a case-sensitive Docker Hub tag to an immutable digest."""
    url = f"https://hub.docker.com/v2/repositories/jefzda/sweap-images/tags/{quote(dockerhub_tag, safe='')}"
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                metadata = json.load(response)
            break
        except HTTPError as exc:
            if exc.code != 429 or attempt == max_attempts - 1:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(2**attempt, 60)
            print(f"Docker Hub rate-limited {dockerhub_tag}; retrying in {delay:g}s")
            sleep(delay)
    digest = metadata.get("digest")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise ValueError(f"Docker Hub did not return a digest for {dockerhub_tag}")
    return digest


def _load_digest_cache(*paths: Path) -> dict[str, str]:
    cache: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            continue
        try:
            if path.suffix == ".json":
                values = json.loads(path.read_text(encoding="utf-8"))
                cache.update({str(tag): str(digest) for tag, digest in values.items()})
            else:
                for line in path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    if row.get("dockerhub_tag") and row.get("image_digest"):
                        cache[str(row["dockerhub_tag"])] = str(row["image_digest"])
        except (json.JSONDecodeError, OSError):
            continue
    return cache


def _write_digest_cache(path: Path, cache: Mapping[str, str]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def enrich_row(row: Mapping[str, Any], upstream_root: Path, image_digest: str) -> dict[str, Any]:
    """Attach verifier-only assets to one public dataset row."""
    instance_id = str(row["instance_id"])
    enriched = dict(row)
    enriched.update(
        {
            "run_script": _read_asset(upstream_root, f"run_scripts/{instance_id}/run_script.sh"),
            "parser_script": _read_asset(upstream_root, f"run_scripts/{instance_id}/parser.py"),
            "base_dockerfile": _read_asset(upstream_root, f"dockerfiles/base_dockerfile/{instance_id}/Dockerfile"),
            "instance_dockerfile": _read_asset(
                upstream_root, f"dockerfiles/instance_dockerfile/{instance_id}/Dockerfile"
            ),
            "image_digest": image_digest,
            "responses_create_params": {
                "input": [
                    {
                        "role": "user",
                        "content": row["problem_statement"],
                    }
                ],
            },
            "subset": "pro",
            "split": "test",
            "evaluator_commit": UPSTREAM_COMMIT,
            "dataset_revision": DATASET_REVISION,
        }
    )
    return enriched


def prepare(
    dataset: Iterable[Mapping[str, Any]] | None = None,
    upstream_root: Path | None = None,
    output_fpath: Path = OUTPUT_FPATH,
    image_digest_resolver=fetch_image_digest,
    image_digest_cache_fpath: Path | None = None,
) -> Path:
    """Materialize the public Pro split as self-contained NeMo Gym JSONL."""
    if dataset is None:
        dataset = load_dataset(
            "ScaleAI/SWE-bench_Pro",
            split="test",
            revision=DATASET_REVISION,
            token=get_hf_token(),
        )
    if upstream_root is None:
        upstream_root = fetch_upstream_assets()

    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    if image_digest_cache_fpath is None:
        image_digest_cache_fpath = output_fpath.parent / "swebench_pro_image_digests.json"
    output_tmp_fpath = output_fpath.with_suffix(output_fpath.suffix + ".tmp")
    digest_cache = _load_digest_cache(image_digest_cache_fpath, output_fpath, output_tmp_fpath)
    count = 0
    with output_tmp_fpath.open("w", encoding="utf-8") as output:
        for row in dataset:
            dockerhub_tag = str(row["dockerhub_tag"])
            image_digest = digest_cache.get(dockerhub_tag)
            if image_digest is None:
                image_digest = image_digest_resolver(dockerhub_tag)
                digest_cache[dockerhub_tag] = image_digest
                _write_digest_cache(image_digest_cache_fpath, digest_cache)
            output.write(json.dumps(enrich_row(row, upstream_root, image_digest)) + "\n")
            count += 1
    if count == 0:
        raise ValueError("SWE-bench Pro preparation produced no rows")
    if dataset is not None and hasattr(dataset, "num_rows") and count != EXPECTED_INSTANCE_COUNT:
        raise ValueError(f"Expected {EXPECTED_INSTANCE_COUNT} SWE-bench Pro rows, got {count}")

    output_tmp_fpath.replace(output_fpath)
    print(f"Wrote {count} SWE-bench Pro problems to {output_fpath}")
    return output_fpath


if __name__ == "__main__":
    prepare()
