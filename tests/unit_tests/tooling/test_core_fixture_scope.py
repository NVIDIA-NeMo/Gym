# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise shared cleanup in fresh pytest processes, without collection-time imports."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("import_at_collection", [False, True])
def test_core_cleanup_does_not_load_unused_modules(tmp_path, import_at_collection):
    root = Path(__file__).resolve().parents[3]
    shutil.copyfile(root / "tests/unit_tests/conftest.py", tmp_path / "conftest.py")
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    preload = ""
    if import_at_collection:
        preload = """
import nemo_gym.config.global_config as config
import nemo_gym.artifacts.exporters as exporters
config._GLOBAL_CONFIG_DICT = {"stale": True}
exporters.mark_run_failed()
"""
    (tmp_path / "test_isolation.py").write_text(
        "import sys\nimport pytest\n"
        + preload
        + f"""
def test_unrelated(request):
    with pytest.raises(pytest.FixtureLookupError):
        request.getfixturevalue("fake_registry")
    if not {import_at_collection!r}:
        assert "nemo_gym.config.global_config" not in sys.modules
        assert "nemo_gym.artifacts.exporters" not in sys.modules
        assert "nemo_gym.data.tasks.harbor.image_configs" not in sys.modules

def test_set_state():
    import nemo_gym.config.global_config as config
    import nemo_gym.artifacts.exporters as exporters
    assert config._GLOBAL_CONFIG_DICT is None
    assert not exporters.run_failed()
    config._GLOBAL_CONFIG_DICT = {{"stale": True}}
    exporters.mark_run_failed()

def test_next_test_is_clean():
    import nemo_gym.config.global_config as config
    import nemo_gym.artifacts.exporters as exporters
    assert config._GLOBAL_CONFIG_DICT is None
    assert not exporters.run_failed()
"""
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-c", str(tmp_path / "pytest.ini"), str(tmp_path)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 passed" in result.stdout
