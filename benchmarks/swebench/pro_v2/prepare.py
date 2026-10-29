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

"""Prepare SWE-bench Pro V2 (642 tasks) from the pinned HF revision and the upstream v2.0.0 Harbor tasks."""

import hashlib
import json
import tarfile
import tomllib
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from io import BytesIO
from pathlib import Path
from time import sleep
from typing import Any
from urllib.error import HTTPError

from datasets import load_dataset

from nemo_gym.global_config import get_hf_token


UPSTREAM_TAG = "v2.0.0"
UPSTREAM_COMMIT = "66f92766bba642462d4bbe5479e83f91f9211862"  # pragma: allowlist secret
UPSTREAM_ARCHIVE_URL = f"https://github.com/scaleapi/SWE-bench_Pro-os/archive/{UPSTREAM_COMMIT}.tar.gz"
DATASET_REVISION = "2d52cb3df914a3fcf80c7f66738b3a88ae37fc50"  # pragma: allowlist secret
IMAGE_REPOSITORY = "ghcr.io/scaleapi/swe-bench_pro-v2"
EXPECTED_INSTANCE_COUNTS = {"default": 642, "hard": 51}
SUBSETS = {"default": "pro_v2", "hard": "pro_v2_hard"}
VERIFIER_FILES = ("test.sh", "run_script.sh", "parser.py", "config.json", "test_patch.patch")
BENCHMARK_DIR = Path(__file__).parent.parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "swebench_pro_v2_benchmark.jsonl"
UPSTREAM_CACHE_DIR = DATA_DIR / "swebench_pro_v2_upstream"
IMAGE_DIGEST_CACHE_FPATH = DATA_DIR / "swebench_pro_v2_image_digests.json"
# OpenCode's glob and grep tools download rg at run time, which the locked sandbox cannot do; the agent uploads this
# pinned static build instead (task images are linux/amd64).
RIPGREP_RELEASE = "ripgrep-14.1.1-x86_64-unknown-linux-musl"
RIPGREP_URL = f"https://github.com/BurntSushi/ripgrep/releases/download/14.1.1/{RIPGREP_RELEASE}.tar.gz"
RIPGREP_SHA256 = "4cf9f2741e6c465ffdb7c26f38056a59e2a2544b51f7cc128ef28337eeae4d8e"  # pragma: allowlist secret
RIPGREP_FPATH = DATA_DIR / RIPGREP_RELEASE / "rg"
MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)


def fetch_upstream_v2(cache_dir: Path = UPSTREAM_CACHE_DIR) -> Path:
    """Download the pinned upstream tree once and return its checksum-verified ``v2/`` directory."""
    root = cache_dir / f"SWE-bench_Pro-os-{UPSTREAM_COMMIT}"
    v2_root = root / "v2"
    if not v2_root.is_dir():
        cache_dir.mkdir(parents=True, exist_ok=True)
        archive_path = cache_dir / f"{UPSTREAM_COMMIT}.tar.gz"
        urllib.request.urlretrieve(UPSTREAM_ARCHIVE_URL, archive_path)
        prefix = f"{root.name}/v2/"
        with tarfile.open(archive_path, "r:gz") as archive:
            members = [member for member in archive.getmembers() if member.name.startswith(prefix)]
            archive.extractall(cache_dir, members=members, filter="data")
        archive_path.unlink()
    if not v2_root.is_dir():
        raise FileNotFoundError(f"Expected extracted upstream directory at {v2_root}")
    verify_checksums(v2_root)
    return v2_root


def verify_checksums(v2_root: Path) -> None:
    """Check every file listed in upstream's ``v2/SHA256SUMS``."""
    mismatched = []
    for line in (v2_root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative_path = line.split(maxsplit=1)
        path = v2_root / relative_path.strip().removeprefix("*")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            mismatched.append(relative_path)
    if mismatched:
        raise ValueError(f"{len(mismatched)} SWE-bench Pro V2 files fail SHA256SUMS, e.g. {mismatched[:3]}")


def fetch_ripgrep(path: Path = RIPGREP_FPATH) -> Path:
    """Stage the pinned rg binary that the agent uploads into the locked sandbox."""
    if path.is_file():
        return path
    with urllib.request.urlopen(RIPGREP_URL, timeout=120) as response:
        archive_bytes = response.read()
    if hashlib.sha256(archive_bytes).hexdigest() != RIPGREP_SHA256:
        raise ValueError(f"{RIPGREP_URL} does not match its pinned SHA-256")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=BytesIO(archive_bytes), mode="r:gz") as archive:
        binary = archive.extractfile(f"{RIPGREP_RELEASE}/rg")
        if binary is None:
            raise FileNotFoundError(f"{RIPGREP_URL} has no rg binary")
        temporary_path = path.with_suffix(".tmp")
        temporary_path.write_bytes(binary.read())
    temporary_path.chmod(0o755)
    temporary_path.replace(path)
    return path


def is_prepared_data_current(fpath: Path) -> bool:
    """A cached prepared file still needs the staged rg binary next to it."""
    return RIPGREP_FPATH.is_file()


def fetch_image_digest(image: str, max_attempts: int = 8) -> str:
    """Resolve a public GHCR image tag to its immutable manifest digest."""
    repository, tag = image.removeprefix("ghcr.io/").rsplit(":", 1)
    for attempt in range(max_attempts):
        try:
            token_url = f"https://ghcr.io/token?scope=repository:{repository}:pull"
            with urllib.request.urlopen(token_url, timeout=30) as response:
                token = json.load(response)["token"]
            manifest = urllib.request.Request(
                f"https://ghcr.io/v2/{repository}/manifests/{tag}",
                method="HEAD",
                headers={"Authorization": f"Bearer {token}", "Accept": MANIFEST_ACCEPT},
            )
            with urllib.request.urlopen(manifest, timeout=30) as response:
                digest = response.headers.get("Docker-Content-Digest")
            break
        except HTTPError as exc:
            if exc.code != 429 or attempt == max_attempts - 1:
                raise
            delay = min(2**attempt, 60)
            print(f"GHCR rate-limited {tag}; retrying in {delay}s")
            sleep(delay)
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise ValueError(f"GHCR did not return a digest for {image}")
    return digest


def _load_digest_cache(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        return {str(image): str(digest) for image, digest in json.loads(path.read_text(encoding="utf-8")).items()}
    except (json.JSONDecodeError, OSError, AttributeError):
        return {}


def _write_digest_cache(path: Path, cache: Mapping[str, str]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _read_exact(path: Path) -> str:
    """Read a task file without newline translation: some test patches carry CRLF lines."""
    return path.read_bytes().decode("utf-8")


def build_row(row: Mapping[str, Any], task_dir: Path, image_digest: str, config_name: str) -> dict[str, Any]:
    """Attach the upstream Harbor task's prompt, verifier and reference patch to one HF row."""
    instance_id = str(row["instance_id"])
    if not task_dir.is_dir():
        raise FileNotFoundError(f"No upstream V2 task directory for {instance_id}: {task_dir}")
    task = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    if task["environment"]["docker_image"] != row["docker_image"]:
        raise ValueError(f"{instance_id}: task.toml image differs from the HF docker_image column")
    if task["agent"].get("network_mode") != "no-network":
        raise ValueError(f"{instance_id}: upstream task no longer locks the agent's network")
    verifier_files = {name: _read_exact(task_dir / "tests" / name) for name in VERIFIER_FILES}
    enriched = dict(row)
    enriched.update(
        {
            # The verifier and the reference patch come from the Harbor task, which is what upstream grades with;
            # the HF columns have their CRLF line endings normalized away.
            "patch": _read_exact(task_dir / "solution" / "gold_patch.diff"),
            "test_patch": verifier_files["test_patch.patch"],
            "run_script": verifier_files["run_script.sh"],
            "parser_script": verifier_files["parser.py"],
            "verifier_files": verifier_files,
            "image_digest": image_digest,
            "responses_create_params": {
                "input": [{"role": "user", "content": _read_exact(task_dir / "instruction.md")}],
            },
            "subset": SUBSETS[config_name],
            "split": "test",
            "evaluator_commit": UPSTREAM_COMMIT,
            "dataset_revision": DATASET_REVISION,
        }
    )
    return enriched


def prepare(
    config_name: str = "default",
    dataset: Iterable[Mapping[str, Any]] | None = None,
    upstream_v2_root: Path | None = None,
    output_fpath: Path = OUTPUT_FPATH,
    image_digest_resolver: Callable[[str], str] = fetch_image_digest,
    image_digest_cache_fpath: Path = IMAGE_DIGEST_CACHE_FPATH,
    stage_ripgrep: bool = True,
) -> Path:
    """Materialize one SWE-bench Pro V2 config (``default`` or ``hard``) as self-contained NeMo Gym JSONL."""
    if config_name not in EXPECTED_INSTANCE_COUNTS:
        raise ValueError(f"Unknown SWE-bench Pro V2 config {config_name!r}")
    if stage_ripgrep:
        fetch_ripgrep()
    if dataset is None:
        dataset = load_dataset(
            "ScaleAI/SWE-bench_Pro",
            config_name,
            split="test",
            revision=DATASET_REVISION,
            token=get_hf_token(),
        )
    if upstream_v2_root is None:
        upstream_v2_root = fetch_upstream_v2()
    if config_name == "hard":
        hard_ids = set((upstream_v2_root / "hard51_ids.txt").read_text(encoding="utf-8").split())

    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    digest_cache = _load_digest_cache(image_digest_cache_fpath)
    output_tmp_fpath = output_fpath.with_suffix(output_fpath.suffix + ".tmp")
    instance_ids = []
    with output_tmp_fpath.open("w", encoding="utf-8") as output:
        for row in dataset:
            image = str(row["docker_image"])
            if not image.startswith(f"{IMAGE_REPOSITORY}:"):
                raise ValueError(f"{row['instance_id']}: unexpected image {image}")
            image_digest = digest_cache.get(image)
            if image_digest is None:
                image_digest = image_digest_resolver(image)
                digest_cache[image] = image_digest
                _write_digest_cache(image_digest_cache_fpath, digest_cache)
            task_dir = upstream_v2_root / "tasks" / str(row["instance_id"])
            output.write(json.dumps(build_row(row, task_dir, image_digest, config_name)) + "\n")
            instance_ids.append(str(row["instance_id"]))

    expected = EXPECTED_INSTANCE_COUNTS[config_name]
    if len(instance_ids) != expected or len(set(instance_ids)) != expected:
        raise ValueError(f"Expected {expected} unique SWE-bench Pro V2 {config_name} rows, got {len(instance_ids)}")
    if config_name == "hard" and set(instance_ids) != hard_ids:
        raise ValueError("SWE-bench Pro V2 hard rows differ from upstream v2/hard51_ids.txt")

    output_tmp_fpath.replace(output_fpath)
    print(f"Wrote {len(instance_ids)} SWE-bench Pro V2 {config_name} problems to {output_fpath}")
    return output_fpath


if __name__ == "__main__":
    prepare()
