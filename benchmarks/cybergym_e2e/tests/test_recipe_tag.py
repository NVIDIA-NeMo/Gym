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
"""recipe_tag + set_task: the template name tracks the build recipe, so a recipe change can't silently
reuse a stale cached template."""

import validate as vc


IMG = "gcr.io/oss-fuzz/jq@sha256:" + "a" * 64


def _mk_cg(tmp_path, deps=b"echo deps\n"):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "install_validate_deps.sh").write_bytes(deps)
    pdir = tmp_path / "projects" / "jq"
    (pdir / "arvo_1").mkdir(parents=True)
    (pdir / "project.toml").write_text(f'build_image = "{IMG}"\n')
    (pdir / "arvo_1" / "config.toml").write_text('repo_to_patch = "jq"\n')
    return tmp_path


def test_recipe_tag_is_stable(tmp_path):
    cg = _mk_cg(tmp_path)
    assert vc.recipe_tag(cg) == vc.recipe_tag(cg)


def test_recipe_tag_changes_with_deps_script(tmp_path):
    cg = _mk_cg(tmp_path)
    before = vc.recipe_tag(cg)
    (cg / "scripts" / "install_validate_deps.sh").write_bytes(b"echo changed\n")
    assert vc.recipe_tag(cg) != before


def test_recipe_tag_changes_with_apt_list(tmp_path, monkeypatch):
    cg = _mk_cg(tmp_path)
    before = vc.recipe_tag(cg)
    monkeypatch.setattr(vc, "APT_PACKAGES", [*vc.APT_PACKAGES, "strace"])
    assert vc.recipe_tag(cg) != before


def test_recipe_tag_changes_with_revision(tmp_path, monkeypatch):
    cg = _mk_cg(tmp_path)
    before = vc.recipe_tag(cg)
    monkeypatch.setattr(vc, "RECIPE_REVISION", vc.RECIPE_REVISION + "x")
    assert vc.recipe_tag(cg) != before


def test_set_task_bakes_recipe_tag_into_names(tmp_path):
    cg = _mk_cg(tmp_path)
    proj, task = vc.set_task("jq/arvo_1", cg)
    assert (proj, task) == ("jq", "arvo_1")
    rtag = vc.recipe_tag(cg)
    assert vc.TEMPLATE == f"cybergym-score-jq-{'a' * 12}-r{rtag}"
    assert vc.BASE_TEMPLATE == f"cybergym-agent-jq-{'a' * 12}-r{rtag}"
