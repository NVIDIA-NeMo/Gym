# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Secondary structure prediction, as an adapter over RibonanzaNet-SS in its own checkout.

No model is imported at module level, so the tools and tests load without a GPU or weights.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..registry import tool
from ..spec import Cost, field
from ..structure import compare
from ..targets import TARGETS
from .paths import resolve as _resolve


@dataclass
class RNetSSIn:
    sequences: list[str] = field("RNA sequences (A/C/G/U), or a selector: @all, @unscored, @top:N.")
    target: str = field("Campaign target to compare against; empty to only predict.", default="")
    batch_size: int = field("Sequences per forward pass", default=8)


@tool(
    name="struct.rnet_ss",
    summary=(
        "Predict RNA secondary structure with RibonanzaNet-SS, which models "
        "pseudoknots. Optionally compares each prediction with a campaign target "
        "and reports pair-level F1 and recall over crossing (pseudoknotted) pairs."
    ),
    cost=Cost.GPU_BATCH,
    model=RNetSSIn,
)
def struct_rnet_ss(inp: RNetSSIn) -> dict:
    import sys

    import torch

    repo = _resolve("GRNADE_REPO", "/opt/geometric-rna-design")
    # prepare.py --setup downloads the checkpoint into the gRNAde checkout, so that is the default home.
    default = repo / "checkpoints" / "ribonanzanet_sec_struct"
    if not default.exists():
        default = Path(os.environ.get("RDS_WEIGHTS", "/weights")) / "ribonanzanet_sec_struct"
    weights = _resolve("RDS_RNET_SS_WEIGHTS", str(default))
    code = repo / "tools" / "ribonanzanet_sec_struct"
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))

    if inp.target and inp.target not in TARGETS:
        raise ValueError(f"unknown target {inp.target!r} (have: {', '.join(sorted(TARGETS))})")
    target = TARGETS[inp.target] if inp.target else None
    if target:
        for seq in inp.sequences:
            if len(seq) != len(target.target_structure):
                raise ValueError(f"sequence length {len(seq)} != target structure {len(target.target_structure)}")

    from network import RibonanzaNetSS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # RibonanzaNetSS points $ARNIEFILE at its own arnie_file.txt, which would cut EternaFold off for the rest
    # of the process, so the original is restored afterwards.
    saved_arniefile = os.environ.get("ARNIEFILE")
    try:
        # RibonanzaNetSS prints a load banner to stdout, where the worker writes its JSON replies.
        import contextlib
        import sys

        with contextlib.redirect_stdout(sys.stderr):
            model = RibonanzaNetSS(
                config_filepath=str(code / "config.yaml"),
                checkpoint_filepath=str(Path(weights) / "ribonanzanet_ss.pt"),
                device=device,
            )
        model.to(device).eval()
        structures: list[str] = []
        for start in range(0, len(inp.sequences), max(1, inp.batch_size)):
            batch = inp.sequences[start : start + max(1, inp.batch_size)]
            _, predicted = model.predict(batch)
            structures.extend(predicted)
    finally:
        if saved_arniefile is None:
            os.environ.pop("ARNIEFILE", None)
        else:
            os.environ["ARNIEFILE"] = saved_arniefile

    results = []
    for seq, structure in zip(inp.sequences, structures, strict=True):
        entry = {"sequence": seq, "structure": structure}
        if target:
            entry["match"] = compare(structure, target.target_structure)
        results.append(entry)
    return {
        "tool": "struct.rnet_ss",
        "device": device,
        "n": len(results),
        "target": target.name if target else None,
        "results": results,
    }
