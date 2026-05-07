# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
from pathlib import Path

from scripts.build_skill_eval_jsonl import build_jsonl, main


def _write_skill(
    root: Path,
    name: str = "demo",
    skill_md: str = "# demo\nbody\n",
    assertions: tuple[str, ...] = ("a", "b"),
    fixture_bytes: bytes = b"fixture",
) -> Path:
    skill = root / name
    (skill / "evals" / "files").mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(skill_md)
    (skill / "evals" / "evals.json").write_text(
        json.dumps(
            {
                "evals": [
                    {
                        "id": 1,
                        "prompt": "do x",
                        "assertions": list(assertions),
                        "files": ["evals/files/input.txt"],
                        "expected_output": None,
                    }
                ]
            }
        )
    )
    (skill / "evals" / "files" / "input.txt").write_bytes(fixture_bytes)
    return skill


def _load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class TestProvenance:
    def test_emits_four_records_per_scenario_default(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        skills.mkdir()
        _write_skill(skills)

        out = tmp_path / "out.jsonl"
        n = build_jsonl(skills_dir=skills, output=out)
        assert n == 4  # 2×2 default: blind, docs-only, skill-only, skill+docs

        records = _load_records(out)
        cells = sorted(r["verifier_metadata"]["cell"] for r in records)
        assert cells == ["blind", "docs-only", "skill+docs", "skill-only"]

        flag_pairs = sorted(
            (r["verifier_metadata"]["with_skill"], r["verifier_metadata"]["with_references"]) for r in records
        )
        assert flag_pairs == [(False, False), (False, True), (True, False), (True, True)]

    def test_cells_flag_restricts_emitted_records(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        skills.mkdir()
        _write_skill(skills)

        out = tmp_path / "out.jsonl"
        n = build_jsonl(skills_dir=skills, output=out, cells=["blind", "skill+docs"])
        assert n == 2

        cells = {r["verifier_metadata"]["cell"] for r in _load_records(out)}
        assert cells == {"blind", "skill+docs"}

    def test_unknown_cell_raises(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        skills.mkdir()
        _write_skill(skills)
        try:
            build_jsonl(skills_dir=skills, output=tmp_path / "out.jsonl", cells=["bogus"])
        except ValueError as e:
            assert "bogus" in str(e)
        else:
            raise AssertionError("expected ValueError for unknown cell")

    def test_every_record_has_all_provenance_fields(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        skills.mkdir()
        _write_skill(skills)

        out = tmp_path / "out.jsonl"
        build_jsonl(skills_dir=skills, output=out)
        for r in _load_records(out):
            md = r["verifier_metadata"]
            for f in ("skill_md_sha", "evals_sha", "fixtures_sha", "judge_prompt_sha", "harness_version"):
                assert f in md, f"missing {f}"
            # SHAs are 12 hex chars when source is present; judge/harness can be empty if inputs aren't wired.
            assert len(md["skill_md_sha"]) == 12
            assert len(md["evals_sha"]) == 12
            assert len(md["fixtures_sha"]) == 12

    def test_skill_md_sha_changes_with_skill_md_only(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        skills.mkdir()
        _write_skill(skills, skill_md="# v1\n")
        build_jsonl(skills_dir=skills, output=tmp_path / "a.jsonl")
        sha_a = _load_records(tmp_path / "a.jsonl")[0]["verifier_metadata"]["skill_md_sha"]

        (skills / "demo" / "SKILL.md").write_text("# v2\n")
        build_jsonl(skills_dir=skills, output=tmp_path / "b.jsonl")
        rec_b = _load_records(tmp_path / "b.jsonl")[0]["verifier_metadata"]

        assert rec_b["skill_md_sha"] != sha_a
        # Evals and fixtures untouched → those SHAs must not move.
        assert rec_b["evals_sha"] == _load_records(tmp_path / "a.jsonl")[0]["verifier_metadata"]["evals_sha"]
        assert rec_b["fixtures_sha"] == _load_records(tmp_path / "a.jsonl")[0]["verifier_metadata"]["fixtures_sha"]

    def test_evals_sha_changes_when_assertions_change(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        skills.mkdir()
        _write_skill(skills, assertions=("a", "b"))
        build_jsonl(skills_dir=skills, output=tmp_path / "a.jsonl")
        sha_a = _load_records(tmp_path / "a.jsonl")[0]["verifier_metadata"]["evals_sha"]

        _write_skill(skills, assertions=("a", "b", "c"))
        build_jsonl(skills_dir=skills, output=tmp_path / "b.jsonl")
        assert _load_records(tmp_path / "b.jsonl")[0]["verifier_metadata"]["evals_sha"] != sha_a

    def test_fixtures_sha_changes_when_fixture_bytes_change(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        skills.mkdir()
        _write_skill(skills, fixture_bytes=b"v1")
        build_jsonl(skills_dir=skills, output=tmp_path / "a.jsonl")
        sha_a = _load_records(tmp_path / "a.jsonl")[0]["verifier_metadata"]["fixtures_sha"]

        (skills / "demo" / "evals" / "files" / "input.txt").write_bytes(b"v2")
        build_jsonl(skills_dir=skills, output=tmp_path / "b.jsonl")
        assert _load_records(tmp_path / "b.jsonl")[0]["verifier_metadata"]["fixtures_sha"] != sha_a

    def test_judge_prompt_sha_uses_provided_file(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        skills.mkdir()
        _write_skill(skills)
        prompt_a = tmp_path / "prompt_a.txt"
        prompt_a.write_text("judge v1")
        prompt_b = tmp_path / "prompt_b.txt"
        prompt_b.write_text("judge v2")

        build_jsonl(skills_dir=skills, output=tmp_path / "a.jsonl", judge_prompt=prompt_a)
        build_jsonl(skills_dir=skills, output=tmp_path / "b.jsonl", judge_prompt=prompt_b)
        sha_a = _load_records(tmp_path / "a.jsonl")[0]["verifier_metadata"]["judge_prompt_sha"]
        sha_b = _load_records(tmp_path / "b.jsonl")[0]["verifier_metadata"]["judge_prompt_sha"]
        assert sha_a and sha_b and sha_a != sha_b

    def test_harness_version_changes_when_any_harness_file_changes(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        skills.mkdir()
        _write_skill(skills)
        h1 = tmp_path / "workspace.py"
        h1.write_text("v1")
        h2 = tmp_path / "judge.py"
        h2.write_text("v1")

        build_jsonl(skills_dir=skills, output=tmp_path / "a.jsonl", harness_paths=[h1, h2])
        ver_a = _load_records(tmp_path / "a.jsonl")[0]["verifier_metadata"]["harness_version"]

        h1.write_text("v2 edit")
        build_jsonl(skills_dir=skills, output=tmp_path / "b.jsonl", harness_paths=[h1, h2])
        ver_b = _load_records(tmp_path / "b.jsonl")[0]["verifier_metadata"]["harness_version"]
        assert ver_a and ver_b and ver_a != ver_b

    def test_build_is_deterministic_for_identical_inputs(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        skills.mkdir()
        _write_skill(skills)
        build_jsonl(skills_dir=skills, output=tmp_path / "a.jsonl")
        build_jsonl(skills_dir=skills, output=tmp_path / "b.jsonl")
        assert (tmp_path / "a.jsonl").read_text() == (tmp_path / "b.jsonl").read_text()

    def test_skip_scenarios_missing_required_fields(self, tmp_path: Path) -> None:
        """Scenarios without assertions, prompt, or id are silently skipped —
        ensure build_jsonl just emits fewer records rather than crashing."""
        skills = tmp_path / "skills"
        skills.mkdir()
        _write_skill(skills)
        # Append a malformed scenario
        ev = skills / "demo" / "evals" / "evals.json"
        data = json.loads(ev.read_text())
        data["evals"].append({"id": 99, "prompt": "", "assertions": []})
        ev.write_text(json.dumps(data))

        build_jsonl(skills_dir=skills, output=tmp_path / "out.jsonl")
        records = _load_records(tmp_path / "out.jsonl")
        # Still 4 cells × 1 valid scenario = 4 records; malformed scenario skipped.
        assert len(records) == 4

    def test_skips_dirs_without_evals_json(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        skills.mkdir()
        (skills / "not-a-skill").mkdir()  # no evals/ at all
        _write_skill(skills)
        build_jsonl(skills_dir=skills, output=tmp_path / "out.jsonl")
        records = _load_records(tmp_path / "out.jsonl")
        # Only the demo skill contributes — the bare directory is ignored.
        assert {r["verifier_metadata"]["skill_name"] for r in records} == {"demo"}


class TestCLI:
    def test_main_writes_output_and_returns_zero(self, tmp_path: Path, capsys) -> None:
        skills = tmp_path / "skills"
        skills.mkdir()
        _write_skill(skills)
        out = tmp_path / "cli_out.jsonl"
        rc = main(["--skills-dir", str(skills), "--output", str(out)])
        assert rc == 0
        assert out.is_file()
        captured = capsys.readouterr()
        assert "wrote" in captured.out and str(out) in captured.out

    def test_main_cells_flag_parses_comma_list(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        skills.mkdir()
        _write_skill(skills)
        out = tmp_path / "cli_out.jsonl"
        rc = main(["--skills-dir", str(skills), "--output", str(out), "--cells", "blind,skill+docs"])
        assert rc == 0
        records = _load_records(out)
        assert {r["verifier_metadata"]["cell"] for r in records} == {"blind", "skill+docs"}

    def test_main_custom_judge_prompt_and_harness_paths(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        skills.mkdir()
        _write_skill(skills)
        prompt = tmp_path / "judge.txt"
        prompt.write_text("judge prompt body")
        harness = tmp_path / "harness.py"
        harness.write_text("# pretend this is harness source\n")
        out = tmp_path / "cli_out.jsonl"
        rc = main(
            [
                "--skills-dir",
                str(skills),
                "--output",
                str(out),
                "--judge-prompt",
                str(prompt),
                "--harness-path",
                str(harness),
            ]
        )
        assert rc == 0
        md = _load_records(out)[0]["verifier_metadata"]
        assert md["judge_prompt_sha"]  # populated from the file we passed
        assert md["harness_version"]  # populated from the file we passed
