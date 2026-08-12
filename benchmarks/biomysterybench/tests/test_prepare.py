# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import json
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.biomysterybench.prepare import (
    RELEASES,
    _download,
    _gym_row,
    _load_problem_rows,
    _select_rows,
    safe_extract,
)


def _row(task_id: str = "hb013", human_solvable: str = "yes") -> dict[str, str]:
    return {
        "id": task_id,
        "question": "Analyze the files and identify the biological condition.",
        "answer_rubric": "Credit if the answer identifies condition A.",
        "allowed_domains": "ncbi.nlm.nih.gov, pypi.org",
        "human_solvable": human_solvable,
    }


class TestPinnedReleases:
    def test_official_release_matches_published_task_split(self) -> None:
        release = RELEASES["official-99"]
        assert release.expected_task_count == 99
        assert release.expected_split_counts == {"yes": 76, "no": 23}

    def test_v11_release_matches_audited_task_split(self) -> None:
        release = RELEASES["v11"]
        assert release.expected_task_count == 90
        assert release.expected_split_counts == {"yes": 73, "no": 17}

    def test_problem_csv_integrity_check(self, tmp_path: Path) -> None:
        release = RELEASES["v11"]
        rows = [_row(f"yes-{index}", "yes") for index in range(73)]
        rows.extend(_row(f"no-{index}", "no") for index in range(17))
        problems = tmp_path / "problems.csv"
        with problems.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        assert len(_load_problem_rows(problems, release)) == 90


class TestSelection:
    def test_requested_order_is_preserved(self) -> None:
        rows = [_row("a"), _row("b"), _row("c")]
        assert [row["id"] for row in _select_rows(rows, ["c", "a"])] == ["c", "a"]

    def test_unknown_task_fails(self) -> None:
        with pytest.raises(ValueError, match="unknown BioMysteryBench task"):
            _select_rows([_row("known")], ["missing"])


class TestDownloadRetries:
    def test_transient_failure_resets_client_and_retries(self, tmp_path: Path) -> None:
        archive = tmp_path / "task.zip"
        download = MagicMock(side_effect=[RuntimeError("client closed"), str(archive)])
        with (
            patch("huggingface_hub.hf_hub_download", download),
            patch("huggingface_hub.close_session") as close_session,
            patch("benchmarks.biomysterybench.prepare.time.sleep") as sleep,
        ):
            assert _download("data/hb013.zip", "token", "revision") == archive
        assert download.call_count == 2
        close_session.assert_called_once_with()
        sleep.assert_called_once_with(1)

    def test_auth_failure_is_not_retried(self) -> None:
        error = RuntimeError("forbidden")
        error.response = SimpleNamespace(status_code=403)
        with (
            patch("huggingface_hub.hf_hub_download", side_effect=error) as download,
            patch("huggingface_hub.close_session") as close_session,
        ):
            with pytest.raises(RuntimeError, match="forbidden"):
                _download("problems.csv", "token", "revision")
        download.assert_called_once()
        close_session.assert_not_called()


class TestSafeExtract:
    def test_extracts_and_reuses_hash_marker(self, tmp_path: Path) -> None:
        archive = tmp_path / "task.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("nested/data.csv", "a,b\n1,2\n")
        destination = tmp_path / "extracted"
        first = safe_extract(archive, destination)
        second = safe_extract(archive, destination)
        assert first == second
        assert (destination / "nested" / "data.csv").read_text() == "a,b\n1,2\n"

    def test_rejects_path_traversal(self, tmp_path: Path) -> None:
        archive = tmp_path / "bad.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("../escape.txt", "no")
        with pytest.raises(ValueError, match="unsafe path"):
            safe_extract(archive, tmp_path / "out")

    def test_rejects_symbolic_link(self, tmp_path: Path) -> None:
        archive = tmp_path / "symlink.zip"
        info = zipfile.ZipInfo("link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr(info, "/etc/passwd")
        with pytest.raises(ValueError, match="Symbolic links|symbolic links"):
            safe_extract(archive, tmp_path / "out")


class TestGymRow:
    def test_rubric_is_not_exposed_to_policy_input_or_metadata(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        row = _gym_row(
            _row(),
            data_dir,
            {"compressed_bytes": 10, "uncompressed_bytes": 20, "file_count": 1},
            "biomysterybench-runtime:v12",
            RELEASES["v11"].revision,
        )
        serialized_policy_params = json.dumps(row["responses_create_params"])
        assert "condition A" not in serialized_policy_params
        assert row["expected_answer"] == "Credit if the answer identifies condition A."
        assert row["responses_create_params"]["metadata"]["data_dir"] == str(data_dir.resolve())
        assert row["allowed_domains"] == ["ncbi.nlm.nih.gov", "pypi.org"]
        assert json.loads(row["responses_create_params"]["metadata"]["allowed_domains"]) == row["allowed_domains"]
        assert "ncbi.nlm.nih.gov, pypi.org" in serialized_policy_params
        assert all(isinstance(value, str) for value in row["responses_create_params"]["metadata"].values())
