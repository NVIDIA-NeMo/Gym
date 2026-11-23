# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invented public fixture data; no private checkout, credential or network access."""

import json
from pathlib import Path


SOURCE = Path(__file__).parent / "fixtures/replay"
DATA = SOURCE / "assets/ngc-recorded-context"
TASK = json.loads((SOURCE / "tasks/ExampleProject/pif_029.json").read_text())
PARAMS = TASK["responses_create_params"]
RUNTIME = json.loads((DATA / "runtime.json").read_text())
BUNDLE = json.loads((DATA / "bundle.json").read_text()) | {"runtime": RUNTIME}
PROBE = RUNTIME["probe_script"]
