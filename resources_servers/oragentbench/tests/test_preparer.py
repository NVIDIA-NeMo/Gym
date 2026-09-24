# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The preparer fails closed: a corpus that does not match the published manifest writes nothing."""

import json
import shutil
import sys
from pathlib import Path

import pytest


SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR / "scripts"))
import prepare_oragentbench as prep  # noqa: E402


FIXTURE_TASK = SERVER_DIR / "tests" / "fixtures" / "toy_assignment"


def synthetic_checkout(tmp_path: Path, counts=prep.EXPECTED_COUNTS) -> Path:
    """A fake upstream checkout with ``counts`` copies of the fixture task per band."""
    checkout = tmp_path / "ORAgentBench"
    tasks = {}
    for band, count in counts.items():
        for i in range(count):
            name = f"{band}_task_{i:03d}"
            shutil.copytree(FIXTURE_TASK, checkout / "harbor_tasks" / name)
            tasks[name] = {"band": band}
    (checkout / "difficulty.json").write_text(json.dumps({"tasks": tasks}))
    return checkout


class TestLoadRows:
    def test_complete_corpus_emits_one_row_per_task_in_the_row_schema(self, tmp_path):
        checkout = synthetic_checkout(tmp_path)
        rows = prep.load_rows(checkout, Path("data/ORAgentBench/harbor_tasks"))
        assert len(rows) == prep.EXPECTED_TOTAL
        row = rows[0]
        assert row["task_name"] == "synthetic/toy_assignment"
        assert row["docker_image"] == f"oragentbench/easy_task_000:{prep.PINNED_COMMIT[:12]}"
        assert row["task_folder"] == "data/ORAgentBench/harbor_tasks/easy_task_000"
        assert row["difficulty"] == "easy" and row["num_steps"] == 1
        assert row["responses_create_params"]["input"][0]["content"].startswith("You are solving")
        assert row["agent_ref"] == {"type": "responses_api_agents", "name": prep.AGENT_NAME}
        assert [r["task_folder"] for r in rows] == sorted(r["task_folder"] for r in rows)

    def test_short_corpus_fails_closed(self, tmp_path):
        checkout = synthetic_checkout(tmp_path, {"easy": 32, "medium": 41, "hard": 33})
        with pytest.raises(RuntimeError, match="expected 107"):
            prep.load_rows(checkout)

    def test_unreadable_task_fails_closed_even_when_directory_count_matches(self, tmp_path):
        checkout = synthetic_checkout(tmp_path)
        (checkout / "harbor_tasks" / "hard_task_000" / "tests" / "test.sh").unlink()
        with pytest.raises(RuntimeError, match="hard_task_000: FileNotFoundError"):
            prep.load_rows(checkout)

    def test_task_missing_from_difficulty_json_fails_closed(self, tmp_path):
        checkout = synthetic_checkout(tmp_path)
        payload = json.loads((checkout / "difficulty.json").read_text())
        del payload["tasks"]["medium_task_000"]
        (checkout / "difficulty.json").write_text(json.dumps(payload))
        with pytest.raises(RuntimeError, match="medium_task_000: KeyError"):
            prep.load_rows(checkout)

    def test_image_tags_are_lowercase_docker_repositories(self):
        assert prep.image_tag("IndustryOR_100") == f"oragentbench/industryor_100:{prep.PINNED_COMMIT[:12]}"


class TestMain:
    def test_writes_nothing_when_the_corpus_is_short(self, tmp_path, monkeypatch):
        checkout = synthetic_checkout(tmp_path, {"easy": 1, "medium": 1, "hard": 1})
        monkeypatch.setattr(prep, "ensure_checkout", lambda path: checkout)
        output = tmp_path / "out" / "benchmark.jsonl"
        with pytest.raises(RuntimeError):
            prep.main(["--output", str(output), "--checkout-dir", str(checkout)])
        assert not output.exists()

    def test_limit_writes_a_prefix_and_reports_the_full_digest(self, tmp_path, monkeypatch, capsys):
        checkout = synthetic_checkout(tmp_path)
        monkeypatch.setattr(prep, "ensure_checkout", lambda path: checkout)
        built = []
        monkeypatch.setattr(prep, "build_images", lambda checkout_dir, names: built.extend(names))
        output = tmp_path / "benchmark.jsonl"
        assert (
            prep.main(["--output", str(output), "--checkout-dir", str(checkout), "--limit", "3", "--build-images"])
            == 0
        )
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert len(rows) == 3 and built == ["easy_task_000", "easy_task_001", "easy_task_002"]
        assert "Loaded 107 tasks" in capsys.readouterr().err

    @pytest.mark.parametrize("limit", ["0", "-1", "two"])
    def test_bad_limit_is_rejected_before_any_fetch(self, monkeypatch, limit):
        def never(path):
            raise AssertionError("ensure_checkout must not run before argument validation")

        monkeypatch.setattr(prep, "ensure_checkout", never)
        with pytest.raises(SystemExit):
            prep.main(["--limit", limit])


class TestEnsureCheckout:
    def test_existing_checkout_at_the_pin_is_accepted_without_network(self, tmp_path, monkeypatch):
        calls = []

        def fake_run(argv, cwd=None):
            calls.append(argv)
            if argv[:2] == ["git", "rev-parse"]:
                return prep.PINNED_COMMIT + "\n"
            raise AssertionError(f"unexpected git call {argv}")

        checkout = tmp_path / "ORAgentBench"
        (checkout / ".git").mkdir(parents=True)
        monkeypatch.setattr(prep, "_run", fake_run)
        assert prep.ensure_checkout(checkout) == checkout
        assert calls == [["git", "rev-parse", "HEAD"]]

    def test_checkout_at_another_commit_is_moved_to_the_pin_or_rejected(self, tmp_path, monkeypatch):
        heads = iter(["deadbeef\n", "deadbeef\n"])

        def fake_run(argv, cwd=None):
            if argv[:2] == ["git", "rev-parse"]:
                return next(heads)
            return ""

        checkout = tmp_path / "ORAgentBench"
        (checkout / ".git").mkdir(parents=True)
        monkeypatch.setattr(prep, "_run", fake_run)
        with pytest.raises(RuntimeError, match="expected c9eb9524"):
            prep.ensure_checkout(checkout)
