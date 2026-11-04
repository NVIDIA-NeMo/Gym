# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import patch

from benchmarks.mmlu_pro import prepare as prepare_module


def test_prepare_matches_nemo_skills_mcq_format(tmp_path: Path) -> None:
    fixture = [
        {
            "question": "\n  Which formula\nis water?  \n",
            "options": ["H2O", "CO2", "NaCl"],
            "answer": "A",
            "category": "chemistry",
        }
    ]
    output_fpath = tmp_path / "mmlu_pro_benchmark.jsonl"

    with (
        patch("datasets.load_dataset", return_value=fixture) as load_dataset,
        patch.object(prepare_module, "DATA_DIR", tmp_path),
        patch.object(prepare_module, "OUTPUT_FPATH", output_fpath),
        patch.object(
            prepare_module,
            "get_global_config_dict",
            return_value={prepare_module.HF_TOKEN_KEY_NAME: "hf-token"},
        ),
    ):
        result = prepare_module.prepare()

    assert result == output_fpath
    load_dataset.assert_called_once_with("TIGER-Lab/MMLU-Pro", split="test", token="hf-token")

    row = json.loads(output_fpath.read_text(encoding="utf-8"))
    assert row["options_text"] == "A) H2O\nB) CO2\nC) NaCl"
    assert row["problem"] == "  Which formula\nis water?  \n\nA) H2O\nB) CO2\nC) NaCl"
    assert row["options"] == [{"A": "H2O"}, {"B": "CO2"}, {"C": "NaCl"}]
