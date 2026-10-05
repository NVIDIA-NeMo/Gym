# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load the bundled scorer or an explicitly configured replacement."""

import importlib.util
from pathlib import Path
from types import ModuleType


def load_grader(path: str) -> ModuleType:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"ChemEval scorer missing: {source}. Restore the bundled grading.py or correct grader_path."
        )
    spec = importlib.util.spec_from_file_location("chemeval_local_grading", source)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load ChemEval Python scorer: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "grade", None)):
        raise ValueError("ChemEval scorer must export a callable grade function")
    return module
