# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import io
import sys
import zipfile

import pytest

from resources_servers.instruction_following import setup_indicifeval as setup


def test_checksum_rejects_corrupt_download_before_extracting(tmp_path):
    destination = tmp_path / "scorer"
    with pytest.raises(ValueError, match="checksum"):
        setup.install_harness(b"corrupt archive", destination)
    assert not destination.exists()


def test_cold_setup_imports_namespaced_trans_only_then_works_offline(tmp_path, monkeypatch):
    archive = io.BytesIO()
    prefix = f"IndicIFEval-{setup.HARNESS_REVISION}/"
    source = prefix + "lm-evaluation-harness/custom_configs/indicifeval-trans/"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(prefix + "LICENSE", "upstream license text")
        output.writestr(prefix + "indicifeval-ground/ground.py", "raise RuntimeError('must not be imported')")
        output.writestr(
            source + "utils.py",
            'import sys\nsys.path.append("/should-not-leak")\nimport instructions_registry\n',
        )
        output.writestr(
            source + "instructions_registry.py",
            "import importlib\n"
            'def load(lang):\n    return importlib.import_module(f"instructions.{lang}_instructions")\n'
            f"INSTRUCTION_DICT = {{lang: load(lang).instructions_util.LANGUAGE for lang in {setup.LANGUAGES!r}}}\n",
        )
        for language in (*setup.LANGUAGES, "en", "as", "sa"):
            output.writestr(
                source + f"instructions/{language}_instructions.py",
                f"import {language}_instructions_util as instructions_util\n",
            )
            output.writestr(source + f"instructions/{language}_instructions_util.py", f"LANGUAGE = {language!r}\n")
    payload = archive.getvalue()
    monkeypatch.setattr(setup, "ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(setup, "CACHE_DIR", tmp_path / "cache/scorer")
    monkeypatch.setattr(setup, "PACKAGE_NAME", "_gym_indicifeval_setup_test")
    monkeypatch.setattr(setup.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    setup.load_harness.cache_clear()
    original_path = list(sys.path)
    try:
        harness = setup.load_harness()
        assert harness.instructions_registry.INSTRUCTION_DICT == {language: language for language in setup.LANGUAGES}
        assert sys.path == original_path
        assert (setup.CACHE_DIR / "LICENSE").read_text() == "upstream license text"
        assert not list(setup.CACHE_DIR.rglob("en_instructions.py"))
        assert not list(setup.CACHE_DIR.rglob("ground.py"))
        assert (setup.CACHE_DIR / ".complete").read_text() == setup.ARCHIVE_SHA256

        # A partial or stale cache must be rebuilt before it can be imported.
        (setup.CACHE_DIR / ".complete").write_text("stale checksum")
        (setup.CACHE_DIR / "stale.py").write_text("stale code")
        setup.load_harness.cache_clear()
        setup.load_harness()
        assert (setup.CACHE_DIR / ".complete").read_text() == setup.ARCHIVE_SHA256
        assert not (setup.CACHE_DIR / "stale.py").exists()

        def offline(*args, **kwargs):
            raise AssertionError("warm setup must not fetch the archive")

        monkeypatch.setattr(setup.urllib.request, "urlopen", offline)
        setup.load_harness.cache_clear()
        assert setup.load_harness().instructions_registry.INSTRUCTION_DICT["kn"] == "kn"
    finally:
        setup.load_harness.cache_clear()
        for name in list(sys.modules):
            if name == setup.PACKAGE_NAME or name.startswith(setup.PACKAGE_NAME + "."):
                del sys.modules[name]
