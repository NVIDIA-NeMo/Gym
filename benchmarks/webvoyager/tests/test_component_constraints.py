# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CONSTRAINTS = REPO_ROOT / "benchmarks/webvoyager/component-constraints.txt"
IMAGE_LOCK = REPO_ROOT / "resources_servers/visual_browser/docker/requirements.lock"
PIN = re.compile(r"^([a-z0-9][a-z0-9.-]*)==([^\s=]+)$")


def _pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        match = PIN.match(line)
        assert match, f"{path.name}: not a normalized exact pin: {line!r}"
        name, version = match.groups()
        assert name not in pins, f"{path.name}: {name} pinned twice"
        pins[name] = version
    return pins


def test_component_constraints_are_exact_pins() -> None:
    pins = _pins(CONSTRAINTS)
    assert not any(name.startswith("nemo-gym") for name in pins)
    # The pins exist to stop drift of the packages that changed what the policy sees.
    assert pins["pillow"] == "12.3.0"
    assert pins["grpcio"] == "1.84.0"


def test_image_lock_agrees_with_component_constraints() -> None:
    constraints = _pins(CONSTRAINTS)
    lock = _pins(IMAGE_LOCK)
    assert set(lock) <= set(constraints), sorted(set(lock) - set(constraints))
    mismatched = {name: (version, constraints[name]) for name, version in lock.items() if constraints[name] != version}
    assert not mismatched, mismatched
