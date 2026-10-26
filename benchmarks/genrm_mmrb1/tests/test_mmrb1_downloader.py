# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from benchmarks.genrm_mmrb1 import run_downloader as module


@pytest.mark.parametrize("fails", [False, True])
def test_only_visitbench_uses_parquet_and_loader_is_restored(monkeypatch, fails):
    datasets = pytest.importorskip("datasets")
    loader = Mock(return_value="loaded")
    monkeypatch.setattr(datasets, "load_dataset", loader)

    def run(path, *, run_name):
        assert path == "upstream.py"
        assert run_name == "__main__"
        assert datasets.load_dataset("HuggingFaceM4/NoCaps", split="test") == "loaded"
        loader.assert_called_with("HuggingFaceM4/NoCaps", split="test")
        assert datasets.load_dataset("mlfoundations/VisIT-Bench", split="test") == "loaded"
        loader.assert_called_with("parquet", data_files={"test": module.VISITBENCH_PARQUET}, split="test")
        if fails:
            raise RuntimeError("upstream failed")

    monkeypatch.setattr(module.runpy, "run_path", run)
    if fails:
        with pytest.raises(RuntimeError, match="upstream failed"):
            module.run_downloader("upstream.py")
    else:
        module.run_downloader("upstream.py")
    assert datasets.load_dataset is loader
