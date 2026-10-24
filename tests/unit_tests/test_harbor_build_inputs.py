# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The inputs digest of a Dockerfile build: what it covers, how the built image's record carries it, and when a
recorded image is reused or rebuilt under its unchanged Harbor name."""

import json
import os
from pathlib import Path

import pytest

import nemo_gym.tasks.harbor.prepare_builds as prepare_builds
from nemo_gym.tasks.harbor.build_plan import DEFAULT_PLATFORM, BuildError, plan_builds
from nemo_gym.tasks.harbor.build_records import (
    build_inputs_digest,
    build_inputs_mismatch,
    pop_stale_build_inputs,
    settle_entry,
)
from nemo_gym.tasks.harbor.build_tools import ImageBuilder
from nemo_gym.tasks.harbor.digest import context_digest, inputs_digest
from nemo_gym.tasks.harbor.image_build import build_images
from nemo_gym.tasks.harbor.records import IMAGE_CONFIGS_FILE, read_built_images, read_image_configs
from tests.unit_tests.harbor_build_helpers import (
    PULL_DOCKERFILE,
    FakeTool,
    harbor_hash,
    local_docker,
    task_inputs_digest,
    write_base_record,
    write_task,
)


CHANGED = (
    "its build inputs changed since it was built (a file's content or mode, a symbolic link, a folder, or a build "
    "setting) while the context's hash did not"
)
UNRECORDED = "its configuration records no inputs digest (it was written before build inputs were recorded)"
# main's build section takes FOO from the build's environment: a value the tree does not hold.
ENV_ARG_COMPOSE = (
    "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    build:\n      context: .\n      args:\n        - FOO\n"
)
# main's inline Dockerfile and a label interpolate FOO and REV from the build's environment: no file and no build
# argument holds either.
INLINE_COMPOSE = (
    "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    build:\n      context: .\n"
    "      dockerfile_inline: |\n        FROM x\n        RUN echo ${FOO}\n"
    "      labels:\n        rev: ${REV}\n"
)
# main's paths spelt through CONTEXT_DIR, the absolute path of the context Harbor exports to the build.
CONTEXT_DIR_COMPOSE = (
    "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    build:\n"
    "      context: ${CONTEXT_DIR}\n      dockerfile: ${CONTEXT_DIR}/Dockerfile\n"
)


def write_tree(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def recorded_inputs(folder: Path, image: str) -> str | None:
    """The inputs digest ``compose-images.json`` in ``folder`` records for ``image``."""
    return read_image_configs(folder)[image].get("inputs_digest")


class TestInputsDigest:
    FILES = {"Dockerfile": "FROM x\n", "tools/setup.sh": "echo ok\n", "tools/data.txt": "1\n"}

    def test_identical_trees_match_whatever_order_they_were_written_in(self, tmp_path):
        a = write_tree(tmp_path / "a", self.FILES)
        b = write_tree(tmp_path / "b", dict(reversed(list(self.FILES.items()))))
        assert inputs_digest(a) == inputs_digest(b)
        assert len(inputs_digest(a)) == 64 and inputs_digest(a) != context_digest(a)
        (b / "tools" / "data.txt").write_text("2\n")
        assert inputs_digest(a) != inputs_digest(b)

    def test_a_mode_change_alone_changes_it_and_keeps_harbors_name(self, tmp_path):
        a = write_tree(tmp_path / "a", self.FILES)
        name, before = context_digest(a), inputs_digest(a)
        script = a / "tools" / "setup.sh"
        script.chmod(script.stat().st_mode | 0o111)
        assert context_digest(a) == name and inputs_digest(a) != before
        # A folder's mode counts too.
        executable = inputs_digest(a)
        (a / "tools").chmod(0o700)
        assert inputs_digest(a) not in (before, executable)

    def test_an_empty_folder_counts(self, tmp_path):
        a = write_tree(tmp_path / "a", self.FILES)
        name, before = context_digest(a), inputs_digest(a)
        (a / "cache").mkdir()
        assert context_digest(a) == name and inputs_digest(a) != before

    def test_symbolic_links_count_by_target_not_by_what_they_point_at(self, tmp_path):
        a = write_tree(tmp_path / "a", self.FILES)
        name, before = context_digest(a), inputs_digest(a)
        (a / "link").symlink_to("tools/setup.sh")
        linked = inputs_digest(a)
        assert context_digest(a) == name and linked != before
        (a / "link").unlink()
        (a / "link").symlink_to("tools/data.txt")
        assert inputs_digest(a) not in (before, linked)
        # A link to a folder is one entry, its target, not a second copy of the folder's files.
        (a / "link").unlink()
        (a / "link").symlink_to("tools")
        assert inputs_digest(a) not in (before, linked)
        (a / "link").unlink()
        (a / "elsewhere").symlink_to(tmp_path / "missing")
        assert inputs_digest(a) not in (before, linked)  # a dangling link is read, not followed

    def test_the_names_harbors_hash_skips_count(self, tmp_path):
        a = write_tree(tmp_path / "a", self.FILES)
        name, before = context_digest(a), inputs_digest(a)
        write_tree(a, {"tools/__pycache__/setup.cpython-313.pyc": "\x00", ".DS_Store": "junk"})
        assert context_digest(a) == name and inputs_digest(a) != before

    def test_settings_are_part_of_it_in_any_order(self, tmp_path):
        a = write_tree(tmp_path / "a", self.FILES)
        plain = inputs_digest(a)
        assert inputs_digest(a, {}) == plain
        gpu = inputs_digest(a, {"dockerfile": "Dockerfile.gpu", "arg:FOO": "1"})
        assert gpu != plain
        assert inputs_digest(a, {"arg:FOO": "1", "dockerfile": "Dockerfile.gpu"}) == gpu
        assert inputs_digest(a, {"dockerfile": "Dockerfile.gpu", "arg:FOO": "2"}) != gpu

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no named pipes on this platform")
    def test_another_kind_of_entry_counts_by_its_mode(self, tmp_path):
        a = write_tree(tmp_path / "a", self.FILES)
        before = inputs_digest(a)
        os.mkfifo(a / "pipe")
        assert inputs_digest(a) != before

    def test_requires_a_folder(self, tmp_path):
        with pytest.raises(ValueError, match="Not a build context"):
            inputs_digest(tmp_path / "missing")


class TestBuildInputsDigest:
    def test_the_settled_build_settings_are_part_of_it(self, tmp_path, monkeypatch):
        folder = tmp_path / "ds"
        write_task(folder / "a", extra={"tools/setup.sh": "true\n", "docker-compose.yaml": ENV_ARG_COMPOSE})
        (entry,) = plan_builds(folder)
        monkeypatch.setenv("FOO", "1")
        one = settle_entry(entry, repository=None, platform=DEFAULT_PLATFORM)
        assert one.build_args == {"FOO": "1"}
        monkeypatch.setenv("FOO", "2")
        two = settle_entry(entry, repository=None, platform=DEFAULT_PLATFORM)
        # The tree is the same (Harbor's name too); the argument the tool is given is not.
        assert context_digest(entry.context) == one.digest == two.digest
        assert build_inputs_digest(one) != build_inputs_digest(two)
        # The repository and platform a builder is configured with are not inputs of the image unless the file
        # interpolates them: the same settled entry gives the same digest for a pushing builder.
        monkeypatch.setenv("FOO", "1")
        pushing = settle_entry(entry, repository="r.example/x", platform=None)
        assert build_inputs_digest(pushing) == build_inputs_digest(one)

    def test_the_whole_build_section_counts_as_interpolated(self, tmp_path, monkeypatch):
        folder = tmp_path / "ds"
        write_task(folder / "a", extra={"tools/setup.sh": "true\n", "docker-compose.yaml": INLINE_COMPOSE})
        (entry,) = plan_builds(folder)
        monkeypatch.setenv("FOO", "one")
        monkeypatch.setenv("REV", "r1")
        one = settle_entry(entry, repository=None, platform=DEFAULT_PLATFORM)
        assert one.dockerfile_inline is True and one.build_args == {}
        assert one.compose_build["dockerfile_inline"] == "FROM x\nRUN echo one\n"
        assert one.compose_build["labels"] == {"rev": "r1"}
        first = build_inputs_digest(one)
        # The inline Dockerfile's text changes with FOO, the label with REV: no file and no argument changes.
        monkeypatch.setenv("FOO", "two")
        inline = build_inputs_digest(settle_entry(entry, repository=None, platform=DEFAULT_PLATFORM))
        monkeypatch.setenv("FOO", "one")
        monkeypatch.setenv("REV", "r2")
        label = build_inputs_digest(settle_entry(entry, repository=None, platform=DEFAULT_PLATFORM))
        assert context_digest(entry.context) == entry.digest and len({first, inline, label}) == 3
        monkeypatch.setenv("REV", "r1")
        assert build_inputs_digest(settle_entry(entry, repository=None, platform=DEFAULT_PLATFORM)) == first

    def test_the_sections_paths_count_as_resolved_not_as_spelt(self, tmp_path):
        # Two copies of one task whose file spells its paths through CONTEXT_DIR: the text differs between the
        # copies, the build does not, so a datasets folder moved keeps its records.
        digests = []
        for home in ("here", "there"):
            folder = tmp_path / home
            write_task(folder / "a", extra={"tools/setup.sh": "true\n", "docker-compose.yaml": CONTEXT_DIR_COMPOSE})
            (entry,) = plan_builds(folder)
            settled = settle_entry(entry, repository=None, platform=DEFAULT_PLATFORM)
            assert settled.compose_build["context"] == str(entry.context) and settled.dockerfile.startswith("/")
            assert settled.dockerfile_path == entry.context.resolve() / "Dockerfile"
            digests.append(build_inputs_digest(settled))
        assert digests[0] == digests[1]

    def test_any_yaml_scalar_of_the_section_digests(self, tmp_path):
        # YAML reads a date and a bare number where JSON has text only: the section digests all the same.
        compose = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    build:\n      context: .\n      labels:\n"
        folder = tmp_path / "ds"
        write_task(
            folder / "a", extra={"docker-compose.yaml": compose + "        built: 2026-10-19\n        1: one\n"}
        )
        (entry,) = plan_builds(folder)
        dated = build_inputs_digest(settle_entry(entry, repository=None, platform=DEFAULT_PLATFORM))
        assert len(dated) == 64
        write_task(
            folder / "a", extra={"docker-compose.yaml": compose + "        built: 2026-10-20\n        1: one\n"}
        )
        (entry,) = plan_builds(folder)
        assert build_inputs_digest(settle_entry(entry, repository=None, platform=DEFAULT_PLATFORM)) != dated

    def test_the_built_images_record_carries_the_digest_of_the_settled_entry(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        builder, _ = local_docker()
        (result,) = build_images(folder, builder=builder, log=lambda _: None)
        expected = task_inputs_digest(folder / "a")
        assert result.record["inputs_digest"] == expected
        assert recorded_inputs(folder, result.image) == expected
        (entry,) = plan_builds(folder)
        record, configs = read_built_images(folder)[result.digest], read_image_configs(folder)
        assert build_inputs_mismatch(entry, record, configs, repository=None, platform=DEFAULT_PLATFORM) is None


class TestReuse:
    def test_an_identical_context_reuses_the_record(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        builder, tool = local_docker()
        (first,) = build_images(folder, builder=builder, log=lambda _: None)
        lines: list[str] = []
        assert build_images(folder, builder=builder, log=lines.append) == []
        assert lines == [f"Built image recorded for a: {first.image}"]
        assert [call for call in tool.calls if call[-1] == "build"] == [tool.calls[1]]

    @pytest.mark.parametrize("change", ["mode", "empty folder", "symlink target"])
    def test_a_change_the_hash_misses_rebuilds_under_the_same_name(self, tmp_path, change):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        context = folder / "a" / "environment"
        if change == "symlink target":
            (context / "link").symlink_to("tools/setup.sh")
        builder, tool = local_docker()
        (first,) = build_images(folder, builder=builder, log=lambda _: None)
        recorded = read_built_images(folder)[first.digest]
        if change == "mode":
            script = context / "tools" / "setup.sh"
            script.chmod(script.stat().st_mode | 0o111)
        elif change == "empty folder":
            (context / "cache").mkdir()
        else:
            (context / "link").unlink()
            (context / "link").symlink_to("Dockerfile")
        assert harbor_hash(context) == first.digest
        lines: list[str] = []
        (again,) = build_images(folder, builder=builder, log=lines.append)
        assert lines == [
            f"Rebuilding {first.image} for a: {CHANGED}",
            f"Built {first.image} (local, docker) for task(s) a",
        ]
        # Same hash, same hb__ name and tag; a new build, and the image's record replaced with the new digest.
        assert again.digest == first.digest and again.tag == first.tag == f"hb__{first.digest}"
        assert len([call for call in tool.calls if call[-1] == "build"]) == 2
        assert read_built_images(folder)[first.digest] == recorded
        assert recorded_inputs(folder, again.image) == again.record["inputs_digest"] != first.record["inputs_digest"]
        assert again.record["inputs_digest"] == task_inputs_digest(folder / "a")
        # The rebuilt context is recorded as it is now: the next prepare reuses it.
        assert build_images(folder, builder=builder, log=lines.append) == []
        assert lines[-1] == f"Built image recorded for a: {first.image}"

    def test_a_build_argument_from_the_environment_rebuilds(self, tmp_path, monkeypatch):
        folder = tmp_path / "ds"
        write_task(folder / "a", extra={"tools/setup.sh": "true\n", "docker-compose.yaml": ENV_ARG_COMPOSE})
        builder, tool = local_docker()
        monkeypatch.setenv("FOO", "1")
        (first,) = build_images(folder, builder=builder, log=lambda _: None)
        assert build_images(folder, builder=builder, log=lambda _: None) == []
        monkeypatch.setenv("FOO", "2")
        lines: list[str] = []
        (again,) = build_images(folder, builder=builder, log=lines.append)
        assert lines[0] == f"Rebuilding {first.image} for a: {CHANGED}" and again.tag == first.tag
        assert again.record["inputs_digest"] != first.record["inputs_digest"]

    def test_a_variable_interpolated_into_the_build_section_rebuilds(self, tmp_path, monkeypatch):
        folder = tmp_path / "ds"
        write_task(folder / "a", extra={"tools/setup.sh": "true\n", "docker-compose.yaml": INLINE_COMPOSE})
        builder, tool = local_docker()
        monkeypatch.setenv("FOO", "one")
        monkeypatch.setenv("REV", "r1")
        (first,) = build_images(folder, builder=builder, log=lambda _: None)
        assert build_images(folder, builder=builder, log=lambda _: None) == []
        # A label's value changes: no file, no argument and the same Harbor hash, another image.
        monkeypatch.setenv("REV", "r2")
        lines: list[str] = []
        (again,) = build_images(folder, builder=builder, log=lines.append)
        assert harbor_hash(folder / "a" / "environment") == first.digest == again.digest
        assert lines[0] == f"Rebuilding {first.image} for a: {CHANGED}" and again.tag == first.tag
        # Then the inline Dockerfile's text.
        monkeypatch.setenv("FOO", "two")
        lines.clear()
        (third,) = build_images(folder, builder=builder, log=lines.append)
        assert lines[0] == f"Rebuilding {first.image} for a: {CHANGED}" and third.tag == first.tag
        assert len({r.record["inputs_digest"] for r in (first, again, third)}) == 3
        assert len([call for call in tool.calls if call[-1] == "build"]) == 3
        assert build_images(folder, builder=builder, log=lambda _: None) == []

    def test_an_image_recorded_without_the_digest_is_rebuilt_once(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        builder, tool = local_docker()
        (first,) = build_images(folder, builder=builder, log=lambda _: None)
        configs = read_image_configs(folder)
        del configs[first.image]["inputs_digest"]
        (folder / IMAGE_CONFIGS_FILE).write_text(json.dumps(configs))
        (entry,) = plan_builds(folder)
        recorded = read_built_images(folder)
        recorded["not-a-context"] = "junk"  # an entry that is not a mapping is left alone
        stale = pop_stale_build_inputs(recorded, [entry], configs, repository=None, platform=DEFAULT_PLATFORM)
        assert [(e.digest, why) for e, _, why in stale] == [(first.digest, UNRECORDED)]
        assert recorded == {"not-a-context": "junk"}
        lines: list[str] = []
        (again,) = build_images(folder, builder=builder, log=lines.append)
        assert lines[0] == f"Rebuilding {first.image} for a: {UNRECORDED}"
        assert recorded_inputs(folder, again.image) == again.record["inputs_digest"]
        assert build_images(folder, builder=builder, log=lambda _: None) == []

    def test_a_compose_file_the_build_refuses_is_reported_and_the_rebuild_fails_with_it(self, tmp_path):
        folder = tmp_path / "ds"
        compose = (
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    build:\n      context: .\n      ssh: [default]\n"
        )
        write_task(folder / "a", extra={"tools/setup.sh": "true\n", "docker-compose.yaml": compose})
        digest = harbor_hash(folder / "a" / "environment")
        record = {"image": f"hb__{digest}", "tasks": ["a"], "tool": "docker", "pushed": False}
        (folder / "built-images.json").write_text(json.dumps({digest: record}))
        write_base_record(folder, f"hb__{digest}")  # the image's record is there; only the inputs are in question
        (entry,) = plan_builds(folder)
        configs = read_image_configs(folder)
        why = build_inputs_mismatch(entry, record, configs, repository=None, platform=DEFAULT_PLATFORM)
        assert why is not None and why.startswith("its build inputs cannot be checked: ") and "ssh" in why
        builder, _ = local_docker()
        lines: list[str] = []
        with pytest.raises(BuildError, match="ssh"):
            build_images(folder, builder=builder, log=lines.append)
        assert lines == [f"Rebuilding hb__{digest} for a: {why}"]


class TestPrepare:
    @pytest.fixture
    def fake_builder(self, monkeypatch):
        made: list[dict] = []

        def factory(**kwargs):
            made.append(dict(kwargs))
            tool = kwargs.pop("tool", None) or "docker"
            return ImageBuilder(tool=tool, run=FakeTool(tool), **kwargs)

        monkeypatch.setattr(prepare_builds, "ImageBuilder", factory)
        return made

    def _recorded_then_changed(self, folder: Path) -> str:
        write_task(folder / "a")
        write_task(folder / "c", dockerfile=PULL_DOCKERFILE)
        write_base_record(folder)
        (first,) = build_images(folder, builder=local_docker()[0], log=lambda _: None)
        script = folder / "a" / "environment" / "tools" / "setup.sh"
        script.chmod(script.stat().st_mode | 0o111)
        return first.image

    def test_builds_disabled_leaves_out_a_stale_record_with_the_reason(self, tmp_path, fake_builder, capsys):
        from nemo_gym.tasks.harbor.build_options import BuildOptions
        from nemo_gym.tasks.harbor.cli import prepare_target

        folder = tmp_path / "ds"
        image = self._recorded_then_changed(folder)
        prepared = prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(enabled=False))
        assert fake_builder == []
        assert [t.task_id for t in prepared.tasks] == ["c"]
        reason = f"recorded image {image} cannot be used: {CHANGED}; builds are disabled"
        assert prepared.skipped == {"a": reason}
        out = capsys.readouterr().out
        assert f"Skipping a: {reason}" in out and "tasks that need one are left out" not in out
        assert image not in (tmp_path / "out" / "ds" / "tasks.jsonl").read_text() and image not in prepared.images
        # The stale records are left in place for the enabled path to replace.
        assert read_built_images(folder)[harbor_hash(folder / "a" / "environment")]["image"] == image

    def test_builds_enabled_rebuilds_a_stale_record_under_the_same_name(self, tmp_path, fake_builder, capsys):
        from nemo_gym.tasks.harbor.build_options import BuildOptions
        from nemo_gym.tasks.harbor.cli import prepare_target

        folder = tmp_path / "ds"
        image = self._recorded_then_changed(folder)
        prepared = prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(tool="docker"))
        assert fake_builder == [{"tool": "docker", "repository": None, "platform": None}]
        assert [t.task_id for t in prepared.tasks] == ["a", "c"] and prepared.tasks[0].image == image
        out = capsys.readouterr().out
        assert f"Rebuilding {image} for a: {CHANGED}" in out and "Built 1 image(s) with docker, kept local" in out
        assert recorded_inputs(folder, image) == task_inputs_digest(folder / "a")
        # Prepared again, nothing is built.
        fake_builder.clear()
        prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(tool="docker"))
        assert fake_builder == [] and "Built image recorded for a" in capsys.readouterr().out
