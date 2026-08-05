from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def example_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in (ROOT / "data/example.jsonl").read_text(encoding="utf-8").splitlines()
    ]
