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

import pytest

from nemo_gym import PARENT_DIR
from nemo_gym.skills import (
    _resolve_skills_path,
    hash_skill_dir,
    load_skill_directory,
    parse_skill_md,
    stage_skills,
)


def _write_skill(skills_dir, name, description="A skill.", version=None, body="# Body\n"):
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True)
    frontmatter = f"name: {name}\ndescription: {description}\n"
    if version is not None:
        # The Agent Skills spec nests version under an optional `metadata:` map.
        frontmatter += f"metadata:\n  version: {version}\n"
    (skill_dir / "SKILL.md").write_text(f"---\n{frontmatter}---\n{body}")
    return skill_dir


class TestParseSkillMd:
    def test_parses_name_description_version(self, tmp_path):
        skill_dir = _write_skill(tmp_path, "cot_enhanced", description="Chain of thought.", version="1.2")
        meta = parse_skill_md(skill_dir / "SKILL.md")
        assert meta.name == "cot_enhanced"
        assert meta.description == "Chain of thought."
        assert meta.version == "1.2"

    def test_version_optional(self, tmp_path):
        skill_dir = _write_skill(tmp_path, "baseline")
        meta = parse_skill_md(skill_dir / "SKILL.md")
        assert meta.version is None

    def test_top_level_version_ignored(self, tmp_path):
        # The spec puts version under metadata:, so a stray top-level version: is not read.
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("---\nname: foo\ndescription: bar\nversion: 9.9\n---\n# Body\n")
        meta = parse_skill_md(skill_md)
        assert meta.version is None

    def test_bom_prefixed_frontmatter_parses(self, tmp_path):
        # A SKILL.md saved with a UTF-8 BOM must not be misread as missing frontmatter.
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_bytes(b"\xef\xbb\xbf" + b"---\nname: foo\ndescription: bar\n---\n# Body\n")
        meta = parse_skill_md(skill_md)
        assert meta.name == "foo"
        assert meta.description == "bar"

    def test_missing_frontmatter_raises(self, tmp_path):
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("# Just markdown, no frontmatter\n")
        with pytest.raises(ValueError, match="missing YAML frontmatter"):
            parse_skill_md(skill_md)

    def test_unterminated_frontmatter_raises(self, tmp_path):
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("---\nname: foo\ndescription: bar\n")
        with pytest.raises(ValueError, match="unterminated YAML frontmatter"):
            parse_skill_md(skill_md)

    def test_malformed_yaml_raises(self, tmp_path):
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("---\nname: : : bad\n\t- nope\n---\n")
        with pytest.raises(ValueError, match="malformed YAML frontmatter"):
            parse_skill_md(skill_md)

    def test_non_mapping_frontmatter_raises(self, tmp_path):
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("---\n- a\n- b\n---\n")
        with pytest.raises(ValueError, match="must be a YAML mapping"):
            parse_skill_md(skill_md)

    def test_missing_name_raises(self, tmp_path):
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("---\ndescription: no name here\n---\n")
        with pytest.raises(ValueError, match="missing a required 'name' field"):
            parse_skill_md(skill_md)


class TestHashSkillDir:
    def test_stable_across_calls(self, tmp_path):
        _write_skill(tmp_path, "a")
        assert hash_skill_dir(tmp_path) == hash_skill_dir(tmp_path)

    def test_changes_with_content(self, tmp_path):
        skill_dir = _write_skill(tmp_path, "a", description="original")
        h1 = hash_skill_dir(tmp_path)
        (skill_dir / "SKILL.md").write_text("---\nname: a\ndescription: mutated\n---\n# Body\n")
        h2 = hash_skill_dir(tmp_path)
        assert h1 != h2

    def test_changes_with_file_layout(self, tmp_path):
        skill_dir = _write_skill(tmp_path, "a")
        h1 = hash_skill_dir(tmp_path)
        (skill_dir / "references").mkdir()
        (skill_dir / "references" / "extra.md").write_text("more context")
        h2 = hash_skill_dir(tmp_path)
        assert h1 != h2

    def test_identical_content_same_hash(self, tmp_path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        _write_skill(dir_a, "skill", description="same")
        _write_skill(dir_b, "skill", description="same")
        assert hash_skill_dir(dir_a) == hash_skill_dir(dir_b)

    def test_hash_is_short_hex(self, tmp_path):
        _write_skill(tmp_path, "a")
        h = hash_skill_dir(tmp_path)
        assert len(h) == 12
        int(h, 16)  # parses as hex


class TestLoadSkillDirectory:
    def test_loads_multiple_skills_sorted(self, tmp_path):
        skills_dir = tmp_path / "variant_a"
        _write_skill(skills_dir, "tool_focused", description="Tools.")
        _write_skill(skills_dir, "baseline", description="Baseline.")
        ref = load_skill_directory(str(skills_dir))
        assert ref.path == str(skills_dir)
        assert len(ref.hash) == 12
        assert [s.name for s in ref.skills] == ["baseline", "tool_focused"]
        assert ref.skills[0].description == "Baseline."

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            load_skill_directory(str(tmp_path / "nope"))

    def test_file_instead_of_dir_raises(self, tmp_path):
        f = tmp_path / "skills.txt"
        f.write_text("not a dir")
        with pytest.raises(ValueError, match="must be a directory"):
            load_skill_directory(str(f))

    def test_empty_dir_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="contains no skills"):
            load_skill_directory(str(empty))

    def test_skill_dir_missing_skill_md_raises(self, tmp_path):
        skills_dir = tmp_path / "variant"
        (skills_dir / "broken").mkdir(parents=True)
        with pytest.raises(ValueError, match="missing a SKILL.md"):
            load_skill_directory(str(skills_dir))

    def test_serializable_ref(self, tmp_path):
        skills_dir = tmp_path / "variant"
        _write_skill(skills_dir, "a")
        ref = load_skill_directory(str(skills_dir))
        dumped = ref.model_dump()
        assert dumped["path"] == str(skills_dir)
        assert dumped["skills"][0]["name"] == "a"


class TestResolveSkillsPath:
    def test_absolute_path_unchanged(self, tmp_path):
        assert _resolve_skills_path(str(tmp_path)) == tmp_path

    def test_relative_resolves_to_parent_dir_when_absent_in_cwd(self):
        resolved = _resolve_skills_path("definitely_not_a_real_skills_dir_xyz")
        assert resolved == PARENT_DIR / "definitely_not_a_real_skills_dir_xyz"


class TestStageSkills:
    def test_copies_tree(self, tmp_path):
        src = tmp_path / "variant"
        _write_skill(src, "cot", description="CoT.")
        (src / "cot" / "references").mkdir()
        (src / "cot" / "references" / "ref.md").write_text("ref")

        dest = tmp_path / "config" / "skills"
        stage_skills(str(src), dest)

        assert (dest / "cot" / "SKILL.md").is_file()
        assert (dest / "cot" / "references" / "ref.md").read_text() == "ref"

    def test_missing_source_raises(self, tmp_path):
        with pytest.raises(ValueError, match="is not a directory"):
            stage_skills(str(tmp_path / "nope"), tmp_path / "dest")
