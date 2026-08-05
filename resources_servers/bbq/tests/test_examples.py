from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_validator():
    path = ROOT / "scripts/validate_examples.py"
    spec = importlib.util.spec_from_file_location("validate_examples", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_validator() -> None:
    load_validator().main()


@pytest.mark.upstream
def test_upstream_provenance_when_paths_are_configured() -> None:
    sft_root = os.environ.get("BBQ_SFT_ROOT")
    rlvr_root = os.environ.get("BBQ_RLVR_ROOT")
    if not sft_root or not rlvr_root:
        pytest.skip("set BBQ_SFT_ROOT and BBQ_RLVR_ROOT for the upstream provenance audit")
    load_validator().main(
        bbq_sft_root=Path(sft_root),
        bbq_rlvr_root=Path(rlvr_root),
    )
