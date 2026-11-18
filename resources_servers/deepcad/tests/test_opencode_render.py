# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
from PIL import Image

from resources_servers.deepcad.sandbox_image import cad_render


def setup_renderer(tmp_path, monkeypatch):
    (tmp_path / "settings.json").write_text(json.dumps({"view_mode": "axis", "color": "teal"}))
    monkeypatch.setattr(cad_render, "__file__", str(tmp_path / "cad_render.py"))
    monkeypatch.setattr("sys.argv", ["cad_render.py", str(tmp_path / "model.py"), "--output", str(tmp_path)])


def test_axis_preview_combines_views_in_order(tmp_path, monkeypatch, capsys):
    setup_renderer(tmp_path, monkeypatch)
    paths = []
    for name, color in [("front", "red"), ("side", "green"), ("top", "blue")]:
        path = tmp_path / (name + ".png")
        Image.new("RGB", (16, 16), color).save(path)
        paths.append(path)

    def render(model, output, *, view_mode, color):
        assert view_mode == "axis" and color == "teal"
        return paths

    monkeypatch.setattr(cad_render, "render_model", render)
    cad_render.main()
    with Image.open(tmp_path / "preview.png") as preview:
        assert preview.size == (48, 16)
        assert [preview.getpixel((x, 8)) for x in (8, 24, 40)] == [(255, 0, 0), (0, 128, 0), (0, 0, 255)]
    assert json.loads(capsys.readouterr().out)["preview"] == str(tmp_path / "preview.png")


def test_failed_render_clears_stale_preview(tmp_path, monkeypatch):
    setup_renderer(tmp_path, monkeypatch)
    (tmp_path / "preview.png").write_bytes(b"old preview")

    def fail(*args, **kwargs):
        raise RuntimeError("invalid generated program")

    monkeypatch.setattr(cad_render, "render_model", fail)
    with pytest.raises(RuntimeError, match="invalid generated"):
        cad_render.main()
    assert not (tmp_path / "preview.png").exists()
