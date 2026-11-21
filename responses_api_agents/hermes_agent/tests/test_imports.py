# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise Hermes import precedence in fresh Python processes."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def import_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    caller, sdk, hermes = (tmp_path / name for name in ("caller", "sdk", "hermes"))
    for root in (caller, sdk, hermes):
        root.mkdir()
    for root, name, source in (
        (caller, "tools", 'ORIGIN = "caller"\n'),
        (sdk, "openai", 'VERSION = "pinned"\n'),
        (hermes, "openai", 'VERSION = "older"\n'),
        (hermes, "tools", "from .registry import registry\n"),
    ):
        (root / name).mkdir()
        (root / name / "__init__.py").write_text(source)
    (hermes / "tools/registry.py").write_text('registry = "hermes"\n')
    (hermes / "model_tools.py").write_text(
        "import openai\nfrom tools.registry import registry\nSDK_VERSION = openai.VERSION\n"
    )
    return caller, sdk, hermes


def _run(import_tree: tuple[Path, Path, Path], body: str) -> None:
    helper = Path(__file__).resolve().parents[1] / "hermes_imports.py"
    setup = """
import importlib.util, sys
sys.path[:0] = sys.argv[2:5]
spec = importlib.util.spec_from_file_location('hermes_imports_under_test', sys.argv[1])
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
"""
    result = subprocess.run(
        [sys.executable, "-c", setup + body, str(helper), *(str(path) for path in import_tree)],
        cwd=import_tree[0].parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_hermes_tools_wins_without_reordering_sdk_paths(import_tree: tuple[Path, Path, Path]) -> None:
    _run(
        import_tree,
        """
assert importlib.util.find_spec('tools').origin.startswith(sys.argv[2])
paths, finders = list(sys.path), list(sys.meta_path)
helper.import_model_tools()
module = sys.modules["model_tools"]
assert module.registry == 'hermes'
assert module.SDK_VERSION == 'pinned'
assert sys.path == paths and sys.meta_path == finders
helper.import_model_tools()
assert sys.modules["model_tools"] is module
""",
    )


def test_existing_foreign_tools_is_rejected_without_replacement(import_tree: tuple[Path, Path, Path]) -> None:
    _run(
        import_tree,
        """
import tools
foreign = tools
try:
    helper.import_model_tools()
except ImportError as error:
    assert 'already loaded' in str(error)
else:
    raise AssertionError('loaded foreign tools must be rejected')
assert sys.modules['tools'] is foreign
""",
    )


def test_import_failure_restores_finders(import_tree: tuple[Path, Path, Path]) -> None:
    (import_tree[2] / "model_tools.py").write_text(
        "from tools.registry import registry\nraise RuntimeError('Hermes initialization failed')\n"
    )
    _run(
        import_tree,
        """
finders = list(sys.meta_path)
try:
    helper.import_model_tools()
except RuntimeError as error:
    assert str(error) == 'Hermes initialization failed'
else:
    raise AssertionError('Hermes initialization failure must propagate')
assert sys.meta_path == finders
""",
    )
