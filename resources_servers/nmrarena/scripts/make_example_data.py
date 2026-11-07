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

"""Write ``data/example.jsonl``: five synthetic rows in the exact shape the preparer emits.

The benchmark's spectra are curated from a third-party database and are not
redistributed here. The five molecules below are textbook compounds, none of which
is in the 105-molecule benchmark, and their peak lists were written for this file
in the benchmark's string format (``H_NMR (<MHz>, <solvent>) δ ...``). They exercise
the same prompt construction as the real corpus.
"""

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_nmrarena import format_row  # noqa: E402
from prompting import NUM_CANDIDATES  # noqa: E402


EXAMPLES = [
    (
        "cls_alcohols",
        {
            "compound_id": "synthetic-ethanol",
            "smiles": "CCO",
            "n_complex": 0.05,
            "h_nmr": "H_NMR (400 MHz, CDCl3) δ 3.69 (q, J = 7.0 Hz, 2H), 2.61 (br s, 1H), 1.22 (t, J = 7.0 Hz, 3H)",
            "c_nmr": "C_NMR (100 MHz, CDCl3) δ 58.3, 18.4",
        },
    ),
    (
        "cls_ketones",
        {
            "compound_id": "synthetic-acetone",
            "smiles": "CC(=O)C",
            "n_complex": 0.05,
            "h_nmr": "H_NMR (400 MHz, CDCl3) δ 2.17 (s, 6H)",
            "c_nmr": "C_NMR (100 MHz, CDCl3) δ 206.7, 30.9",
        },
    ),
    (
        "cls_esters",
        {
            "compound_id": "synthetic-ethyl-acetate",
            "smiles": "CCOC(C)=O",
            "n_complex": 0.12,
            "h_nmr": "H_NMR (400 MHz, CDCl3) δ 4.12 (q, J = 7.1 Hz, 2H), 2.05 (s, 3H), 1.26 (t, J = 7.1 Hz, 3H)",
            "c_nmr": "C_NMR (100 MHz, CDCl3) δ 171.1, 60.4, 21.0, 14.2",
        },
    ),
    (
        "cls_polyarenes",
        {
            "compound_id": "synthetic-toluene",
            "smiles": "Cc1ccccc1",
            "n_complex": 0.10,
            "h_nmr": "H_NMR (400 MHz, CDCl3) δ 7.28-7.24 (m, 2H), 7.19-7.15 (m, 3H), 2.36 (s, 3H)",
            "c_nmr": "C_NMR (100 MHz, CDCl3) δ 137.9, 129.1, 128.3, 125.4, 21.5",
        },
    ),
    (
        "cls_ethers",
        {
            "compound_id": "synthetic-anisole",
            "smiles": "COc1ccccc1",
            "n_complex": 0.14,
            "h_nmr": "H_NMR (400 MHz, CDCl_3) δ 7.31-7.26 (m, 2H), 6.96-6.88 (m, 3H), 3.81 (s, 3H)",
            "c_nmr": "C_NMR (100 MHz, CDCl_3) δ 159.7, 129.5, 120.7, 114.0, 55.1",
        },
    ),
]


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "data" / "example.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for cls, rec in EXAMPLES:
            f.write(json.dumps(format_row(cls, rec, NUM_CANDIDATES), ensure_ascii=False) + "\n")
    print(f"Wrote {len(EXAMPLES)} rows to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
