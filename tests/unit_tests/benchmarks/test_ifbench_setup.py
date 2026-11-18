# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from resources_servers.ifbench import setup_ifbench


def test_patches_nltk_downloads_without_changing_instruction_logic(tmp_path):
    source = tmp_path / "instructions.py"
    source.write_text(
        "import nltk\n"
        "def check(value):\n"
        "    nltk.download('punkt_tab')\n"
        '    nltk.download("averaged_perceptron_tagger_eng")\n'
        "    return nltk.pos_tag(nltk.word_tokenize(value))\n"
    )
    setup_ifbench._patch_nltk_downloads(source)
    patched = source.read_text()
    assert "nltk.download" not in patched
    assert "nltk.data.find('tokenizers/punkt_tab')" in patched
    assert "nltk.data.find('taggers/averaged_perceptron_tagger_eng')" in patched
    assert "return nltk.pos_tag(nltk.word_tokenize(value))" in patched
    compile(patched, str(source), "exec")
    setup_ifbench._patch_nltk_downloads(source)
    assert source.read_text() == patched


@pytest.fixture
def nltk_mock(monkeypatch, tmp_path):
    nltk = SimpleNamespace(data=MagicMock(), download=MagicMock())
    nltk.data.path = []
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("NEMO_GYM_TEST_DATA_ROOT", raising=False)
    monkeypatch.setitem(sys.modules, "nltk", nltk)
    monkeypatch.setitem(sys.modules, "regex", ModuleType("regex"))
    return nltk


def test_cached_nltk_data_does_not_download(nltk_mock):
    setup_ifbench._ensure_nltk_data()
    nltk_mock.download.assert_not_called()
    assert [call.args[0] for call in nltk_mock.data.find.call_args_list] == list(setup_ifbench.NLTK_RESOURCES.values())


def test_rejected_mounted_nltk_data_is_copied_privately_without_download(nltk_mock, monkeypatch, tmp_path):
    root = tmp_path / "mounted"
    monkeypatch.setenv("NEMO_GYM_TEST_DATA_ROOT", str(root))
    for resource in setup_ifbench.NLTK_RESOURCES.values():
        path = root / "nltk_data" / resource
        path.mkdir(parents=True)
        (path / "data").write_text("fixture")
    cache = tmp_path / "nltk_data"

    def find(resource):
        # NLTK 3.10 rejects the foreign-UID mount, then accepts the private copy.
        if not (cache / resource / "data").is_file():
            raise LookupError(resource)

    nltk_mock.data.find.side_effect = find
    setup_ifbench._ensure_nltk_data()
    nltk_mock.download.assert_not_called()
    for resource in setup_ifbench.NLTK_RESOURCES.values():
        assert (cache / resource / "data").read_text() == "fixture"
        assert (root / "nltk_data" / resource / "data").read_text() == "fixture"


def test_missing_nltk_data_is_downloaded_and_checked(nltk_mock, tmp_path):
    nltk_mock.data.find.side_effect = [LookupError(), None, None, None, None]
    setup_ifbench._ensure_nltk_data()
    nltk_mock.download.assert_called_once_with(
        "punkt", download_dir=str(tmp_path / "nltk_data"), quiet=True, raise_on_error=True
    )
    assert str(tmp_path / "nltk_data") in nltk_mock.data.path
    assert nltk_mock.data.find.call_args_list[0] == nltk_mock.data.find.call_args_list[1]


def test_failed_nltk_download_is_reported(nltk_mock):
    nltk_mock.data.find.side_effect = LookupError("missing punkt")
    nltk_mock.download.side_effect = ValueError("download failed")
    with pytest.raises(ValueError, match="download failed"):
        setup_ifbench._ensure_nltk_data()


def test_existing_clone_gets_nltk_patches(tmp_path, monkeypatch, nltk_mock):
    monkeypatch.setattr(setup_ifbench, "IFBENCH_DIR", tmp_path)
    monkeypatch.setattr(setup_ifbench, "INSTALL_MARKER", tmp_path / ".installed")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    (tmp_path / ".installed").touch()
    for filename in ("instructions.py", "instructions_util.py"):
        (tmp_path / filename).write_text("nltk.download('stopwords')\n")
    setup_ifbench.ensure_ifbench()
    for filename in ("instructions.py", "instructions_util.py"):
        assert (tmp_path / filename).read_text() == "nltk.data.find('corpora/stopwords')\n"
    nltk_mock.download.assert_not_called()


def test_incomplete_nltk_data_is_reported(nltk_mock):
    nltk_mock.data.find.side_effect = LookupError("still missing punkt")
    with pytest.raises(LookupError, match="still missing punkt"):
        setup_ifbench._ensure_nltk_data()


def test_mounted_source_is_copied_and_patched_without_modifying_mount(tmp_path, monkeypatch, nltk_mock):
    source = tmp_path / "mounted"
    source.mkdir()
    (source / "REVISION").write_text(setup_ifbench.IFBENCH_COMMIT)
    (source / "instructions.py").write_text("download('en_core_web_sm')\nnltk.download('punkt_tab')\n")
    (source / "instructions_util.py").write_text("nltk.download('stopwords')\n")
    for path in source.iterdir():
        path.chmod(0o444)
    source.chmod(0o555)
    destination = tmp_path / "private"
    monkeypatch.setattr(setup_ifbench, "IFBENCH_DIR", destination)
    monkeypatch.setattr(setup_ifbench, "INSTALL_MARKER", destination / ".installed")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(setup_ifbench.subprocess, "run", MagicMock(side_effect=AssertionError("unexpected git")))
    setup_ifbench.ensure_ifbench(source_dir=source)
    assert (destination / ".installed").is_file()
    assert "nltk.data.find" in (destination / "instructions.py").read_text()
    assert "nltk.download" in (source / "instructions.py").read_text()
    assert not (source / ".installed").exists()


def test_wrong_mounted_revision_fails_before_import(tmp_path, monkeypatch):
    source = tmp_path / "mounted"
    source.mkdir()
    (source / "REVISION").write_text("wrong")
    monkeypatch.setattr(setup_ifbench, "IFBENCH_DIR", tmp_path / "private")
    monkeypatch.setattr(setup_ifbench, "INSTALL_MARKER", tmp_path / "private/.installed")
    with pytest.raises(ValueError, match="source revision does not match"):
        setup_ifbench.ensure_ifbench(source_dir=source)


def test_absent_source_falls_back_to_git(tmp_path, monkeypatch, nltk_mock):
    destination = tmp_path / "private"
    monkeypatch.setattr(setup_ifbench, "IFBENCH_DIR", destination)
    monkeypatch.setattr(setup_ifbench, "INSTALL_MARKER", destination / ".installed")
    monkeypatch.setattr(sys, "path", sys.path.copy())

    def git(command, *, check):
        if command[1] == "clone":
            destination.mkdir()
            (destination / "instructions.py").write_text("# source\n")
            (destination / "instructions_util.py").write_text("# source\n")

    run = MagicMock(side_effect=git)
    monkeypatch.setattr(setup_ifbench.subprocess, "run", run)
    setup_ifbench.ensure_ifbench(source_dir=tmp_path / "absent")
    assert run.call_args_list[0].args[0][1] == "clone"
    assert run.call_args_list[1].args[0][-1] == setup_ifbench.IFBENCH_COMMIT
    assert (destination / ".installed").is_file()
