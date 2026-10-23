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
def nltk_mock(monkeypatch):
    nltk = SimpleNamespace(data=MagicMock(), download=MagicMock())
    monkeypatch.setitem(sys.modules, "nltk", nltk)
    monkeypatch.setitem(sys.modules, "regex", ModuleType("regex"))
    return nltk


def test_cached_nltk_data_does_not_download(nltk_mock):
    setup_ifbench._ensure_nltk_data()
    nltk_mock.download.assert_not_called()
    assert [call.args[0] for call in nltk_mock.data.find.call_args_list] == list(setup_ifbench.NLTK_RESOURCES.values())


def test_missing_nltk_data_is_downloaded_and_checked(nltk_mock):
    nltk_mock.data.find.side_effect = [LookupError(), None, None, None, None]
    setup_ifbench._ensure_nltk_data()
    nltk_mock.download.assert_called_once_with("punkt", quiet=True, raise_on_error=True)
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
