# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import io
import json
import tarfile
from pathlib import Path

import pytest

from benchmarks.wandr.prepare import SOURCE_COMMIT, SUBSET100_FPATH, _extract, _load_tasks, _select_subset


ROOT = f"wandr-{SOURCE_COMMIT}/datasets/wandr"


def _tarball(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return path


def _task_files(task: str, core: bytes = b"core") -> dict[str, bytes]:
    return {
        f"{ROOT}/{task}/instruction.md": b"Find things.",
        f"{ROOT}/{task}/task.toml": b"[task]",
        f"{ROOT}/{task}/tests/verify.py": b"verify",
        f"{ROOT}/{task}/tests/test.sh": b"harbor wrapper",
        f"{ROOT}/{task}/tests/wandr_task/config.py": b"CONFIG = None",
        f"{ROOT}/{task}/tests/wandr_core/src/pipeline.py": core,
        f"{ROOT}/{task}/environment/Dockerfile": b"FROM python",
    }


def test_extract_keeps_task_files_and_one_shared_core(tmp_path: Path) -> None:
    tarball = _tarball(
        tmp_path / "w.tar.gz", {**_task_files("a-task"), **_task_files("b-task"), **_task_files("skip")}
    )
    tasks_dir, core_dir = tmp_path / "tasks", tmp_path / "core"
    _extract(tarball, {"a-task", "b-task"}, tasks_dir, core_dir)
    assert sorted(path.name for path in tasks_dir.iterdir()) == ["a-task", "b-task"]
    assert (tasks_dir / "a-task" / "tests" / "wandr_task" / "config.py").read_bytes() == b"CONFIG = None"
    assert not (tasks_dir / "a-task" / "tests" / "test.sh").exists()
    assert not (tasks_dir / "a-task" / "environment").exists()
    assert (core_dir / "src" / "pipeline.py").read_bytes() == b"core"


def test_extract_refuses_differing_core_copies(tmp_path: Path) -> None:
    tarball = _tarball(tmp_path / "w.tar.gz", {**_task_files("a-task"), **_task_files("b-task", core=b"other")})
    with pytest.raises(ValueError, match="differs"):
        _extract(tarball, {"a-task", "b-task"}, tmp_path / "tasks", tmp_path / "core")


@pytest.mark.parametrize(
    "name",
    [
        f"{ROOT}/a-task/tests/wandr_core/../../../escaped.txt",
        f"{ROOT}/a-task/tests/wandr_task/../../../../escaped.txt",
    ],
)
def test_extract_refuses_members_that_climb_out_of_the_data_dir(tmp_path: Path, name: str) -> None:
    tarball = _tarball(tmp_path / "w.tar.gz", {**_task_files("a-task"), name: b"payload"})
    with pytest.raises(ValueError, match="unsafe path"):
        _extract(tarball, {"a-task"}, tmp_path / "data" / "tasks", tmp_path / "data" / "core")
    assert not list(tmp_path.rglob("escaped.txt"))


def _row(task_id: str, commit: str = SOURCE_COMMIT) -> dict:
    task_dir = task_id.replace("_", "-")
    return {
        "task_id": task_id,
        "source_commit": commit,
        "instruction_source_url": f"https://github.com/perplexityai/wandr/blob/{SOURCE_COMMIT}/datasets/wandr/{task_dir}/instruction.md",
        "required_output_files": [f"results_{task_id}.jsonl"],
        "task_tree_json": json.dumps(
            [{"required_output_file": f"results_{task_id}.jsonl", "item_fields": ["entity"]}]
        ),
    }


def test_load_tasks_reads_task_dirs_files_and_item_fields(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(_row("abm_provider")) + "\n")
    assert _load_tasks(path, 1) == [
        {
            "task_name": "abm-provider",
            "task_id": "abm_provider",
            "required_output_files": ["results_abm_provider.jsonl"],
            "item_fields": {"results_abm_provider.jsonl": ["entity"]},
        }
    ]


@pytest.mark.parametrize(
    ("row", "count", "message"), [(_row("a", commit="0" * 40), 1, "source commit"), (_row("a"), 2, "expected 2")]
)
def test_load_tasks_refuses_another_commit_or_count(tmp_path: Path, row: dict, count: int, message: str) -> None:
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match=message):
        _load_tasks(path, count)


def test_select_subset_keeps_the_benchmark_order() -> None:
    rows = [{"task_name": name} for name in ("a", "b", "c", "d")]
    assert _select_subset(rows, ["d", "b"]) == [{"task_name": "b"}, {"task_name": "d"}]


@pytest.mark.parametrize(("names", "message"), [(["a", "x"], "unknown tasks: \\['x'\\]"), (["a", "a"], "duplicate")])
def test_select_subset_refuses_unknown_or_duplicate_names(names: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _select_subset([{"task_name": "a"}, {"task_name": "b"}], names)


def test_the_subset_prepare_returns_the_subset_configs_jsonl_fpath(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gym requires prepare() to return the dataset's own jsonl_fpath; the full prepare writes both files."""
    import yaml

    from benchmarks.wandr import prepare as full
    from benchmarks.wandr import prepare_subset100

    calls = []
    monkeypatch.setattr(full, "prepare", lambda: calls.append("full") or full.OUTPUT_FPATH)
    root = Path(full.__file__).parents[2]
    for config in ("benchmarks/wandr/config_subset100.yaml", "benchmarks/wandr/configs/wandr_opencode_subset100.yaml"):
        datasets = yaml.safe_load((root / config).read_text())["wandr_resources_server"]["resources_servers"]["wandr"]
        (dataset,) = datasets["datasets"]
        assert dataset["prepare_script"] == "benchmarks/wandr/prepare_subset100.py"
        assert prepare_subset100.prepare() == root / dataset["jsonl_fpath"]
    assert calls == ["full", "full"]


def test_the_committed_subset_lists_100_distinct_tasks() -> None:
    names = SUBSET100_FPATH.read_text().split()
    assert len(names) == len(set(names)) == 100
