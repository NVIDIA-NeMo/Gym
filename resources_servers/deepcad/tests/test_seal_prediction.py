# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import stat
from pathlib import Path
from unittest.mock import patch

import cadquery as cq
import pytest

from resources_servers.deepcad import seal_prediction
from resources_servers.deepcad.geometry import load_solid, score


@pytest.mark.parametrize("offset", [0.5, 3.0])
def test_sealing_fuses_untrusted_compound_and_preserves_iou(tmp_path, monkeypatch, capsys, offset):
    first = cq.Workplane("XY").box(2, 2, 2).val()
    second = first.translate((offset, 0, 0))
    source = tmp_path / "model-output.step"
    cq.exporters.export(cq.Compound.makeCompound([first, second]), str(source))
    monkeypatch.setattr(seal_prediction, "INPUT_PATH", str(source))
    monkeypatch.setattr("sys.argv", ["seal", str(32 * 1024 * 1024)])
    real_mkdtemp = seal_prediction.tempfile.mkdtemp
    monkeypatch.setattr(
        seal_prediction.tempfile, "mkdtemp", lambda **kw: real_mkdtemp(prefix=kw["prefix"], dir=tmp_path)
    )
    seal_prediction.main()
    output = Path(json.loads(capsys.readouterr().out)["path"])
    assert stat.S_IMODE(output.parent.stat().st_mode) == 0o700
    fused = load_solid(output)
    expected = first.fuse(second)
    assert fused.Volume() == pytest.approx(expected.Volume())
    assert len(fused.Solids()) == len(expected.Solids())
    assert score(fused, expected)["reward"] == pytest.approx(1, abs=1e-6)
    # Later writes to the model-owned artifact cannot change the sealed snapshot.
    source.write_text("replaced by background process")
    assert load_solid(output).Volume() == pytest.approx(expected.Volume())


@pytest.mark.parametrize("kind", ["oversized", "symlink", "empty"])
def test_sealing_rejects_bad_artifacts_before_import(tmp_path, monkeypatch, kind):
    source = tmp_path / "prediction.step"
    source.write_text("large" if kind == "oversized" else "")
    if kind == "symlink":
        target = tmp_path / "other.step"
        source.rename(target)
        source.symlink_to(target)
    monkeypatch.setattr(seal_prediction, "INPUT_PATH", str(source))
    monkeypatch.setattr("sys.argv", ["seal", "1"])
    with patch.object(seal_prediction.cq.importers, "importStep") as load:
        with pytest.raises((ValueError, OSError)):
            seal_prediction.main()
    load.assert_not_called()
