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

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from benchmarks.swebench.pro_v2 import prepare as v2
from benchmarks.swebench.pro_v2_hard import prepare as v2_hard
from resources_servers.swebench_pro.app import SWEBenchProInstanceRequest
from resources_servers.swebench_pro.task_data import TaskData


IMAGE = "ghcr.io/scaleapi/swe-bench_pro-v2:{instance_id}"


def make_task(v2_root: Path, instance_id: str, *, agent_network_mode: str = "no-network") -> None:
    task = v2_root / "tasks" / instance_id
    files = {
        "instruction.md": f"A code repository is available in the `/app` directory. Fix {instance_id}.\n",
        "task.toml": (
            f'[agent]\nnetwork_mode = "{agent_network_mode}"\ntimeout_sec = 3000.0\n\n'
            f'[environment]\ndocker_image = "{IMAGE.format(instance_id=instance_id)}"\n'
        ),
        "solution/gold_patch.diff": "gold\n",
        **{f"tests/{name}": f"{name} of {instance_id}\n" for name in v2.VERIFIER_FILES},
    }
    for relative_path, contents in files.items():
        path = task / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents.encode("utf-8"))


def write_checksums(v2_root: Path) -> None:
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(v2_root)}"
        for path in sorted(v2_root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (v2_root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def hf_row(instance_id: str) -> dict:
    return {
        "repo": "example/repo",
        "instance_id": instance_id,
        "base_commit": "abc123",
        "patch": "hf patch",
        "test_patch": "hf test patch",
        "problem_statement": "Fix it",
        "requirements": "",
        "interface": "",
        "repo_language": "python",
        "fail_to_pass": '["new_test"]',
        "pass_to_pass": "[]",
        "before_repo_set_cmd": "",
        "selected_test_files_to_run": '["tests"]',
        "dockerhub_tag": "v1-tag",
        "docker_image": IMAGE.format(instance_id=instance_id),
        "hard": "False",
        "version": "2.0.0",
    }


@pytest.fixture
def upstream(tmp_path: Path) -> Path:
    v2_root = tmp_path / "v2"
    for instance_id in ("instance_a", "instance_b"):
        make_task(v2_root, instance_id)
    (v2_root / "hard51_ids.txt").write_text("instance_b\n", encoding="utf-8")
    write_checksums(v2_root)
    return v2_root


def test_build_row_uses_the_harbor_task_prompt_verifier_and_reference_patch(upstream: Path) -> None:
    row = v2.build_row(hf_row("instance_a"), upstream / "tasks" / "instance_a", "sha256:abc", "default")

    assert row["responses_create_params"]["input"] == [
        {"role": "user", "content": "A code repository is available in the `/app` directory. Fix instance_a.\n"}
    ]
    assert row["verifier_files"]["test.sh"] == "test.sh of instance_a\n"
    assert row["run_script"] == "run_script.sh of instance_a\n"
    assert row["test_patch"] == "test_patch.patch of instance_a\n"
    assert row["patch"] == "gold\n"
    assert row["image_digest"] == "sha256:abc"
    assert (row["subset"], row["evaluator_commit"], row["dataset_revision"]) == (
        "pro_v2",
        v2.UPSTREAM_COMMIT,
        v2.DATASET_REVISION,
    )
    # The prepared row is a valid task for the resources server.
    SWEBenchProInstanceRequest.model_validate(row)
    TaskData.model_validate(row)


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda row: row.update(docker_image=IMAGE.format(instance_id="other")), "image differs"),
        (lambda row: row.update(instance_id="instance_missing"), "No upstream V2 task directory"),
    ],
)
def test_build_row_rejects_rows_that_do_not_match_upstream(upstream: Path, mutate, message: str) -> None:
    row = hf_row("instance_a")
    mutate(row)
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        v2.build_row(row, upstream / "tasks" / row["instance_id"], "sha256:abc", "default")


def test_build_row_keeps_crlf_lines_of_the_task_files(upstream: Path) -> None:
    test_patch = upstream / "tasks" / "instance_a" / "tests" / "test_patch.patch"
    test_patch.write_bytes(b"+--boundary\r\n+Content-Type: text/plain\r\n")

    row = v2.build_row(hf_row("instance_a"), upstream / "tasks" / "instance_a", "sha256:abc", "default")

    assert row["verifier_files"]["test_patch.patch"] == "+--boundary\r\n+Content-Type: text/plain\r\n"
    assert json.loads(json.dumps(row))["test_patch"] == row["verifier_files"]["test_patch.patch"]


def test_build_row_rejects_a_task_without_the_network_lock(tmp_path: Path) -> None:
    make_task(tmp_path, "instance_a", agent_network_mode="public")
    with pytest.raises(ValueError, match="no longer locks"):
        v2.build_row(hf_row("instance_a"), tmp_path / "tasks" / "instance_a", "sha256:abc", "default")


def test_verify_checksums_rejects_a_changed_file(upstream: Path) -> None:
    v2.verify_checksums(upstream)
    (upstream / "tasks" / "instance_a" / "tests" / "test.sh").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fail SHA256SUMS"):
        v2.verify_checksums(upstream)


def test_prepare_writes_both_configs_and_caches_digests(
    upstream: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v2, "EXPECTED_INSTANCE_COUNTS", {"default": 2, "hard": 1})
    resolved: list[str] = []

    def resolver(image: str) -> str:
        resolved.append(image)
        return "sha256:" + image.rsplit(":", 1)[1]

    cache = tmp_path / "digests.json"
    full = v2.prepare(
        dataset=[hf_row("instance_a"), hf_row("instance_b")],
        upstream_v2_root=upstream,
        output_fpath=tmp_path / "full.jsonl",
        image_digest_resolver=resolver,
        image_digest_cache_fpath=cache,
        stage_ripgrep=False,
    )
    hard = v2_hard.prepare(
        dataset=[hf_row("instance_b")],
        upstream_v2_root=upstream,
        output_fpath=tmp_path / "hard.jsonl",
        image_digest_resolver=resolver,
        image_digest_cache_fpath=cache,
        stage_ripgrep=False,
    )

    full_rows = [json.loads(line) for line in full.read_text().splitlines()]
    hard_rows = [json.loads(line) for line in hard.read_text().splitlines()]
    assert [row["instance_id"] for row in full_rows] == ["instance_a", "instance_b"]
    assert [(row["instance_id"], row["subset"]) for row in hard_rows] == [("instance_b", "pro_v2_hard")]
    assert hard_rows[0]["image_digest"] == "sha256:instance_b"
    assert len(resolved) == 2  # HARD-51 reuses the cached digest
    assert json.loads(cache.read_text()) == {
        IMAGE.format(instance_id="instance_a"): "sha256:instance_a",
        IMAGE.format(instance_id="instance_b"): "sha256:instance_b",
    }


def test_prepare_rejects_a_wrong_row_count_or_hard_subset(
    upstream: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v2, "EXPECTED_INSTANCE_COUNTS", {"default": 2, "hard": 1})
    kwargs = {
        "upstream_v2_root": upstream,
        "image_digest_resolver": lambda image: "sha256:x",
        "image_digest_cache_fpath": tmp_path / "digests.json",
        "stage_ripgrep": False,
    }

    with pytest.raises(ValueError, match="Expected 2 unique"):
        v2.prepare(dataset=[hf_row("instance_a")], output_fpath=tmp_path / "full.jsonl", **kwargs)
    with pytest.raises(ValueError, match="hard51_ids"):
        v2_hard.prepare(dataset=[hf_row("instance_a")], output_fpath=tmp_path / "hard.jsonl", **kwargs)
    assert not (tmp_path / "full.jsonl").exists()


def test_fetch_image_digest_reads_the_ghcr_manifest_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = []

    class Response:
        def __init__(self, body: bytes = b"", headers: dict | None = None) -> None:
            self.body, self.headers = body, headers or {}

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, *args: object) -> bytes:
            return self.body

    def urlopen(request, timeout: float):
        requests.append(request)
        if isinstance(request, str):
            return Response(json.dumps({"token": "anonymous"}).encode())
        return Response(headers={"Docker-Content-Digest": "sha256:feed"})

    monkeypatch.setattr(v2.urllib.request, "urlopen", urlopen)

    assert v2.fetch_image_digest(IMAGE.format(instance_id="instance_a")) == "sha256:feed"
    assert requests[0] == "https://ghcr.io/token?scope=repository:scaleapi/swe-bench_pro-v2:pull"
    manifest = requests[1]
    assert manifest.full_url == "https://ghcr.io/v2/scaleapi/swe-bench_pro-v2/manifests/instance_a"
    assert manifest.get_method() == "HEAD"
    assert manifest.get_header("Authorization") == "Bearer anonymous"


def ripgrep_archive(contents: bytes) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo(f"{v2.RIPGREP_RELEASE}/rg")
        info.size = len(contents)
        archive.addfile(info, io.BytesIO(contents))
    return buffer.getvalue()


@pytest.mark.parametrize("pinned", [True, False])
def test_fetch_ripgrep_stages_only_the_pinned_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned: bool
) -> None:
    archive = ripgrep_archive(b"#!/bin/sh\necho rg\n")
    sha256 = hashlib.sha256(archive).hexdigest() if pinned else "0" * 64
    monkeypatch.setattr(v2, "RIPGREP_SHA256", sha256)
    monkeypatch.setattr(v2.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(archive))
    path = tmp_path / v2.RIPGREP_RELEASE / "rg"

    if not pinned:
        with pytest.raises(ValueError, match="pinned SHA-256"):
            v2.fetch_ripgrep(path)
        assert not path.exists()
        return
    assert v2.fetch_ripgrep(path) == path
    assert path.read_bytes() == b"#!/bin/sh\necho rg\n"
    assert path.stat().st_mode & 0o111


def test_config_uploads_the_rg_that_prepare_stages() -> None:
    """Gym starts the agent server in its own directory, so the configured path is relative to it."""
    import yaml

    config = yaml.safe_load((v2.BENCHMARK_DIR / "pro_v2" / "opencode.yaml").read_text())
    agent = config["swebench_pro_v2_opencode_sandboxed_agent"]["responses_api_agents"]["opencode_sandboxed_agent"]
    agent_dir = v2.BENCHMARK_DIR.parent.parent / "responses_api_agents" / "opencode_sandboxed_agent"

    assert (agent_dir / agent["local_ripgrep_binary_path"]).resolve() == v2.RIPGREP_FPATH.resolve()
