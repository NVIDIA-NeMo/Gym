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
"""A placeholder PDB for gRNAde's 2d mode, which still loads a PDB of the right length before designing."""

from __future__ import annotations

from pathlib import Path


#: One G-C pair from yeast tRNA-Phe (PDB 1EHZ, residues 1 and 72). biotite returns an empty dot-bracket
#: when it detects no pairs, so one real pair is needed for gRNAde's length check to pass.
_WC_PAIR = {
    "G": [
        ("P", 50.626, 49.730, 50.573),
        ("O5'", 50.161, 49.136, 52.023),
        ("C5'", 50.216, 49.948, 53.210),
        ("C4'", 50.968, 49.231, 54.309),
        ("O4'", 50.450, 47.888, 54.472),
        ("C3'", 52.454, 49.030, 54.074),
        ("O3'", 53.203, 50.177, 54.425),
        ("C2'", 52.781, 47.831, 54.957),
        ("O2'", 53.018, 48.156, 56.313),
        ("C1'", 51.502, 47.007, 54.836),
        ("N9", 51.628, 45.992, 53.798),
        ("C8", 51.064, 46.007, 52.547),
        ("N7", 51.379, 44.966, 51.831),
        ("C5", 52.197, 44.218, 52.658),
        ("C6", 52.848, 42.992, 52.425),
        ("O6", 52.826, 42.291, 51.404),
        ("N1", 53.588, 42.588, 53.534),
        ("C2", 53.685, 43.282, 54.716),
        ("N2", 54.452, 42.733, 55.671),
        ("N3", 53.077, 44.429, 54.946),
        ("C4", 52.356, 44.836, 53.879),
    ],
    "C": [
        ("P", 57.094, 33.402, 52.426),
        ("O5'", 57.017, 34.599, 53.469),
        ("C5'", 57.817, 34.563, 54.660),
        ("C4'", 57.653, 35.834, 55.450),
        ("O4'", 57.964, 36.953, 54.582),
        ("C3'", 56.261, 36.155, 55.979),
        ("O3'", 55.932, 35.502, 57.209),
        ("C2'", 56.302, 37.671, 56.139),
        ("O2'", 56.872, 38.146, 57.338),
        ("C1'", 57.184, 38.082, 54.963),
        ("N1", 56.343, 38.502, 53.829),
        ("C2", 55.796, 39.801, 53.838),
        ("O2", 56.063, 40.568, 54.791),
        ("N3", 54.993, 40.180, 52.815),
        ("C4", 54.734, 39.331, 51.818),
        ("N4", 53.935, 39.749, 50.832),
        ("C5", 55.283, 38.015, 51.780),
        ("C6", 56.077, 37.647, 52.792),
    ],
}


def _placeholder_pdb(target, path: Path) -> Path:
    """A PDB with the target's length. Residues sit 20 A apart, except a real G-C pair at the two ends."""
    sequence = target.wild_type or "A" * target.length
    purine, pyrimidine = ("N9", "C8", "C4"), ("N1", "C2", "C4")
    anchors = {0: "G", target.length - 1: "C"}
    lines, serial = [], 1
    for index, base in enumerate(sequence):
        number = index + 1
        if index in anchors:
            residue, atoms = anchors[index], _WC_PAIR[anchors[index]]
            rows = [(name, x, y, z) for name, x, y, z in atoms]
        else:
            residue = base
            ring = purine if base in ("A", "G") else pyrimidine
            rows = [
                (name, float(offset), 0.0, number * 20.0 + 200.0) for offset, name in enumerate(("P", "C4'", *ring))
            ]
        for name, x, y, z in rows:
            lines.append(
                f"ATOM  {serial:5d} {name:<4s} {residue:>3s} A{number:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {name[0]:>2s}"
            )
            serial += 1
    lines.append("END")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path
