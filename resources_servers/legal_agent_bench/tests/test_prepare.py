# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from resources_servers.legal_agent_bench import prepare


TASK_IDS = ("area/task-one", "area/task-group/scenario-01")


def _configure_small_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(prepare, "EXPECTED_TASK_COUNT", len(TASK_IDS))
    monkeypatch.setattr(prepare, "SMOKE_TASK_IDS", TASK_IDS)
    monkeypatch.setattr(prepare, "DEFAULT_INDEX_FPATH", tmp_path / "all.jsonl")


def _write_source(root: Path, task_ids: tuple[str, ...] = TASK_IDS, *, missing_skill: str | None = None) -> Path:
    source = root / prepare.LAB_SOURCE_ARCHIVE_ROOT
    for task_id in task_ids:
        task_dir = source / "tasks" / task_id
        (task_dir / "documents").mkdir(parents=True)
        (task_dir / "documents" / "input.txt").write_text(f"document for {task_id}\n", encoding="utf-8")
        task = {
            "title": f"Task {task_id}",
            "work_type": "review",
            "instructions": "Review the provided document and write response.docx.",
            "criteria": [
                {
                    "id": "C-001",
                    "title": "Response exists",
                    "match_criteria": "PASS if response.docx exists.",
                    "deliverables": ["response.docx"],
                }
            ],
        }
        (task_dir / "task.json").write_text(json.dumps(task), encoding="utf-8")
    for skill in prepare.REQUIRED_SKILLS:
        if skill == missing_skill:
            continue
        skill_dir = source / "harness" / "skills" / skill
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(f"# {skill}\n", encoding="utf-8")
        (skill_dir / "helper.py").write_text("print('ok')\n", encoding="utf-8")
    return source


def _archive_source(source: Path, archive_path: Path) -> None:
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source, arcname=source.name)


def _build_caches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path, Path]:
    _configure_small_snapshot(monkeypatch, tmp_path)
    source = _write_source(tmp_path / "source")
    prepare._validate_source_tree(source)
    tasks = tmp_path / "tasks"
    skills = tmp_path / "skills"
    prepare._build_task_cache(source, tasks)
    prepare._build_skills_cache(source, skills)
    return source, tasks, skills


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_generated_task_cache_is_deterministic_and_credential_free(monkeypatch, tmp_path) -> None:
    source, first, _skills = _build_caches(monkeypatch, tmp_path)
    second = tmp_path / "tasks-second"

    prepare._build_task_cache(source, second)

    assert _tree_hashes(first) == _tree_hashes(second)
    prepare.validate_harbor_tasks(first)
    rows = [json.loads(line) for line in (first / prepare.INDEX_FILENAME).read_text().splitlines()]
    assert [row["instance_id"] for row in rows] == [
        "legal_agent_bench::area__task-group__scenario-01",
        "legal_agent_bench::area__task-one",
    ]
    assert all(row["agent_ref"]["name"] == "legal_agent_bench_harbor_agent" for row in rows)
    for toml in first.glob("*/task.toml"):
        text = toml.read_text(encoding="utf-8")
        assert "[verifier.env]" not in text
        assert "API_KEY" not in text
        assert "docker_image" not in text


def test_existing_valid_caches_skip_network(monkeypatch, tmp_path) -> None:
    _source, tasks, skills = _build_caches(monkeypatch, tmp_path)
    monkeypatch.setattr(
        prepare,
        "_download_source_archive",
        lambda _path: pytest.fail("valid caches must not access the network"),
    )

    result = prepare.prepare_assets("all", tasks_dir=tasks, skills_dir=skills)

    assert result == {"tasks": tasks, "skills": skills}
    assert (tmp_path / "all.jsonl").read_bytes() == (tasks / prepare.INDEX_FILENAME).read_bytes()


def test_missing_assets_download_extract_and_install(monkeypatch, tmp_path) -> None:
    _configure_small_snapshot(monkeypatch, tmp_path)
    source = _write_source(tmp_path / "source")
    archive = tmp_path / "source.tar.gz"
    _archive_source(source, archive)
    monkeypatch.setattr(prepare, "_download_source_archive", lambda output: output.write_bytes(archive.read_bytes()))

    tasks = tmp_path / "cache" / "tasks"
    skills = tmp_path / "cache" / "skills"
    result = prepare.prepare_assets("all", tasks_dir=tasks, skills_dir=skills)

    assert result == {"tasks": tasks, "skills": skills}
    assert (tmp_path / "all.jsonl").read_bytes() == (tasks / prepare.INDEX_FILENAME).read_bytes()
    assert (tasks / prepare.INDEX_FILENAME).is_file()
    assert (tasks / "area__task-one" / "documents" / "input.txt").is_file()
    assert (tasks / "area__task-group__scenario-01" / "task.toml").is_file()
    for relpath in prepare.VERIFIER_TEMPLATE_SOURCES:
        assert (tasks / "area__task-one" / relpath).is_file()
    assert sorted(path.name for path in skills.iterdir() if path.is_dir()) == list(prepare.REQUIRED_SKILLS)


def test_failed_force_refresh_does_not_replace_valid_cache(monkeypatch, tmp_path) -> None:
    _source, tasks, skills = _build_caches(monkeypatch, tmp_path)
    prepare.prepare_assets("all", tasks_dir=tasks, skills_dir=skills)
    before = _tree_hashes(tasks)
    published_before = (tmp_path / "all.jsonl").read_bytes()
    bad_source = _write_source(tmp_path / "bad-source", task_ids=(TASK_IDS[0],))
    archive = tmp_path / "bad-source.tar.gz"
    _archive_source(bad_source, archive)
    monkeypatch.setattr(prepare, "_download_source_archive", lambda output: output.write_bytes(archive.read_bytes()))

    with pytest.raises(ValueError, match="must contain 2 tasks"):
        prepare.prepare_assets("all", tasks_dir=tasks, skills_dir=skills, force=True)

    assert _tree_hashes(tasks) == before
    assert (tmp_path / "all.jsonl").read_bytes() == published_before
    prepare.validate_harbor_tasks(tasks)


def test_missing_or_modified_generated_index_invalidates_cache(monkeypatch, tmp_path) -> None:
    _source, tasks, _skills = _build_caches(monkeypatch, tmp_path)
    index_path = tasks / prepare.INDEX_FILENAME

    index_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stale or non-deterministic"):
        prepare.validate_harbor_tasks(tasks)

    index_path.unlink()
    with pytest.raises(FileNotFoundError, match="task index is missing"):
        prepare.validate_harbor_tasks(tasks)


@pytest.mark.parametrize("kind", ["traversal", "link", "corrupt"])
def test_unsafe_or_corrupt_archive_is_rejected(kind, tmp_path) -> None:
    archive = tmp_path / "bad.tar.gz"
    if kind == "corrupt":
        archive.write_bytes(b"not a tar archive")
    else:
        with tarfile.open(archive, "w:gz") as tar:
            info = tarfile.TarInfo(
                f"{prepare.LAB_SOURCE_ARCHIVE_ROOT}/../../escape.txt"
                if kind == "traversal"
                else f"{prepare.LAB_SOURCE_ARCHIVE_ROOT}/link"
            )
            if kind == "link":
                info.type = tarfile.SYMTYPE
                info.linkname = "/tmp/target"
                tar.addfile(info)
            else:
                payload = b"bad"
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError):
        prepare._safe_extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_runtime_hydration_keeps_pristine_cache_secret_free(monkeypatch, tmp_path) -> None:
    _source, tasks, _skills = _build_caches(monkeypatch, tmp_path)
    runtime = tmp_path / "runtime"

    prepare.hydrate_runtime_tasks(
        tasks,
        runtime,
        verifier_env={
            "LAB_JUDGE_BASE_URL": "https://judge.example/v1",
            "LAB_JUDGE_API_KEY": "test-secret",  # pragma: allowlist secret
            "LAB_JUDGE_MODEL": "openai-compatible/judge-model",
        },
        reward_mode="full_task",
    )

    cached_toml = (tasks / "area__task-one" / "task.toml").read_text(encoding="utf-8")
    runtime_toml = (runtime / "area__task-one" / "task.toml").read_text(encoding="utf-8")
    assert "test-secret" not in cached_toml
    assert "[verifier.env]" not in cached_toml
    assert 'LAB_JUDGE_API_KEY = "test-secret"' in runtime_toml  # pragma: allowlist secret
    assert 'LEGAL_AGENT_BENCH_REWARD_MODE = "full_task"' in runtime_toml
    assert (tasks / "area__task-one" / "documents" / "input.txt").stat().st_ino == (
        runtime / "area__task-one" / "documents" / "input.txt"
    ).stat().st_ino


def test_runtime_hydration_can_reuse_a_validated_cache(monkeypatch, tmp_path) -> None:
    _source, tasks, _skills = _build_caches(monkeypatch, tmp_path)
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(
        prepare,
        "validate_harbor_tasks",
        lambda _path: pytest.fail("startup must not validate an already validated cache twice"),
    )

    prepare.hydrate_runtime_tasks(
        tasks,
        runtime,
        verifier_env={},
        reward_mode="full_task",
        cache_is_validated=True,
    )

    assert (runtime / "area__task-one" / "task.toml").is_file()


def test_missing_public_skill_fails_clearly(monkeypatch, tmp_path) -> None:
    _configure_small_snapshot(monkeypatch, tmp_path)
    source = _write_source(tmp_path / "source", missing_skill="pptx")

    with pytest.raises(ValueError, match="pptx/SKILL.md"):
        prepare._validate_source_tree(source)


class _FakeResponse:
    def __init__(
        self,
        chunks: list[bytes],
        *,
        error: Exception | None = None,
        stream_error: Exception | None = None,
    ) -> None:
        self.chunks = chunks
        self.error = error
        self.stream_error = stream_error
        self.headers = {"content-length": str(sum(len(chunk) for chunk in chunks))}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def raise_for_status(self) -> None:
        if self.error:
            raise self.error

    def iter_content(self, chunk_size: int):
        del chunk_size
        for chunk in self.chunks:
            yield chunk
        if self.stream_error:
            raise self.stream_error


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response

    def mount(self, *_args) -> None:
        pass

    def get(self, *_args, **_kwargs) -> _FakeResponse:
        return self.response

    def close(self) -> None:
        pass


def test_downloader_verifies_exact_bytes_and_checksum(monkeypatch, tmp_path) -> None:
    requests = pytest.importorskip("requests")
    payload = b"\x00\xff\x80public-binary\x00"
    response = _FakeResponse([payload[:5], payload[5:]])
    monkeypatch.setattr(requests, "Session", lambda: _FakeSession(response))
    monkeypatch.setattr(prepare, "LAB_SOURCE_ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest())
    output = tmp_path / "archive.tar.gz"

    prepare._download_source_archive(output)

    assert output.read_bytes() == payload
    assert not output.with_suffix(output.suffix + ".part").exists()


def test_downloader_failure_leaves_no_completed_file(monkeypatch, tmp_path) -> None:
    requests = pytest.importorskip("requests")
    response = _FakeResponse([], error=RuntimeError("HTTP 503"))
    monkeypatch.setattr(requests, "Session", lambda: _FakeSession(response))
    monkeypatch.setattr(prepare.time, "sleep", lambda _seconds: None)
    output = tmp_path / "archive.tar.gz"

    with pytest.raises(RuntimeError, match="503"):
        prepare._download_source_archive(output)

    assert not output.exists()
    assert not output.with_suffix(output.suffix + ".part").exists()


@pytest.mark.parametrize(
    "response, expected_error",
    [
        (_FakeResponse([b"wrong bytes"]), "checksum mismatch"),
        (_FakeResponse([b"partial"], stream_error=OSError("connection reset")), "connection reset"),
    ],
)
def test_checksum_or_interrupted_download_leaves_no_output(monkeypatch, tmp_path, response, expected_error) -> None:
    requests = pytest.importorskip("requests")
    monkeypatch.setattr(requests, "Session", lambda: _FakeSession(response))
    monkeypatch.setattr(prepare.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(prepare, "LAB_SOURCE_ARCHIVE_SHA256", hashlib.sha256(b"expected").hexdigest())
    output = tmp_path / "archive.tar.gz"

    with pytest.raises((ValueError, OSError), match=expected_error):
        prepare._download_source_archive(output)

    assert not output.exists()
    assert not output.with_suffix(output.suffix + ".part").exists()
