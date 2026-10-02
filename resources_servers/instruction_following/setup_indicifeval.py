# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install the pinned IndicIFEval-Trans checkers into a local, ignored cache."""

import hashlib
import importlib
import importlib.util
import io
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Literal, get_args

from filelock import FileLock


HARNESS_REVISION = "1bb5f1bc4936cb20b8544e185bfd7cbed8f31464"
ARCHIVE_SHA256 = "01692685e707e039a48b60680ea1436811325a1907f79c69ffb5ce2fd000ba9e"
ARCHIVE_URL = f"https://codeload.github.com/AI4Bharat/IndicIFEval/zip/{HARNESS_REVISION}"
IndicLanguage = Literal["bn", "gu", "hi", "kn", "mr", "ml", "ne", "or", "pa", "ta", "te", "ur"]
LANGUAGES: tuple[IndicLanguage, ...] = get_args(IndicLanguage)
CACHE_DIR = Path(__file__).parent / ".indicifeval_trans" / HARNESS_REVISION
PACKAGE_NAME = f"_gym_indicifeval_trans_{HARNESS_REVISION}"


def install_harness(archive: bytes, destination: Path) -> None:
    """Extract only the selected Trans scorers, retaining upstream license notices.

    Changes to upstream sources are limited to package-relative imports and
    removal of sys.path mutations and the strict scorer's registry debug print.
    Checker and strict/loose scoring bodies are otherwise unchanged.
    """
    if hashlib.sha256(archive).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("IndicIFEval archive checksum mismatch")
    root = f"IndicIFEval-{HARNESS_REVISION}/"
    prefix = root + "lm-evaluation-harness/custom_configs/indicifeval-trans/"
    files = ["utils.py", "instructions_registry.py"] + [
        f"instructions/{language}_instructions{suffix}.py" for language in LANGUAGES for suffix in ("", "_util")
    ]
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        (destination / "LICENSE").write_bytes(source.read(root + "LICENSE"))
        for filename in files:
            text = source.read(prefix + filename).decode("utf-8")
            text = re.sub(r"^sys\.path\.append\(.*\)\s*$", "", text, flags=re.MULTILINE)
            text = re.sub(
                r"^import\s+(\w+_instructions_util) as instructions_util",
                r"from . import \1 as instructions_util",
                text,
                flags=re.MULTILINE,
            )
            if filename == "instructions_registry.py":
                text = text.replace(
                    'f"instructions.{lang}_instructions"', 'f"{__package__}.instructions.{lang}_instructions"'
                )
            elif filename == "utils.py":
                text = text.replace("import instructions_registry", "from . import instructions_registry")
                text = text.replace('    print("INSTRUCTION_DICT", instructions_registry.INSTRUCTION_DICT)\n', "")
            output = destination / filename
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
    for directory in (destination, destination / "instructions"):
        (directory / "__init__.py").write_text("", encoding="utf-8")
    (destination / ".complete").write_text(ARCHIVE_SHA256, encoding="utf-8")


@lru_cache(maxsize=1)
def load_harness() -> ModuleType:
    """Download once, then import a namespaced copy without modifying sys.path."""
    CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(CACHE_DIR) + ".lock"):
        marker = CACHE_DIR / ".complete"
        if not marker.is_file() or marker.read_text(encoding="utf-8") != ARCHIVE_SHA256:
            with urllib.request.urlopen(ARCHIVE_URL, timeout=60) as response:
                archive = response.read()
            with tempfile.TemporaryDirectory(dir=CACHE_DIR.parent) as temporary:
                staging = Path(temporary) / "package"
                install_harness(archive, staging)
                if CACHE_DIR.exists():
                    shutil.rmtree(CACHE_DIR)
                staging.rename(CACHE_DIR)
    spec = importlib.util.spec_from_file_location(PACKAGE_NAME, CACHE_DIR / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    return importlib.import_module(f"{PACKAGE_NAME}.utils")
