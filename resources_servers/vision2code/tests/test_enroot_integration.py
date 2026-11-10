# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in real execution tests. No host execution fallback is permitted."""

import os
import shutil
from pathlib import Path

import pytest

from resources_servers.vision2code.render import render


IMAGE = os.environ.get("VISION2CODE_TEST_SQSH")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not IMAGE or not shutil.which("enroot"), reason="Set VISION2CODE_TEST_SQSH in an Enroot worker allocation"
    ),
]
PROVIDER = {"enroot": {}}
VALID = "from PIL import Image\nImage.new('RGB',(TARGET_WIDTH,TARGET_HEIGHT),'white').save(OUTPUT_PATH)"


@pytest.mark.parametrize(
    "code,status",
    [
        (VALID, "ok"),
        (f"```python\n{VALID}\n```", "ok"),
        ("this is not python !", "code_error"),
        ("raise SystemExit(137)", "code_error"),
        ("while True: pass", "timeout"),
        ("pass", "missing_output"),
        ("open(OUTPUT_PATH,'w').write('not png')", "invalid_output"),
        (
            VALID + "\nfrom pathlib import Path\n"
            "p = Path(OUTPUT_PATH)\ndata = bytearray(p.read_bytes())\n"
            "chunk = data.index(b'IDAT')\nlength = int.from_bytes(data[chunk-4:chunk], 'big')\n"
            "data[chunk+4+length] ^= 1\np.write_bytes(data)",
            "invalid_output",
        ),
        ("import os; os.symlink('/etc/hosts',OUTPUT_PATH)", "invalid_output"),
        ("with open(OUTPUT_PATH,'wb') as f: f.truncate(32*1024*1024+1)", "invalid_output"),
        ("from PIL import Image; Image.new('1',(5001,5000)).save(OUTPUT_PATH)", "invalid_output"),
    ],
)
async def test_real_execution(tmp_path, code, status):
    result = await render(
        code=code, width=80, height=60, image=IMAGE, provider=PROVIDER, artifact_dir=tmp_path, timeout=5
    )
    assert result.status == status, result.model_dump()
    assert result.diagnostics["stopped"]


async def test_real_matplotlib_dependencies_and_dimensions(tmp_path):
    code = """
import errno
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
try:
    open('/etc/vision2code-write-test', 'w')
except OSError as exc:
    assert exc.errno == errno.EROFS
else:
    raise AssertionError('Rootfs must be read-only')
values = pd.Series(np.arange(3))
assert nx.path_graph(3).number_of_edges() == 2
fig, ax = plt.subplots(figsize=(1, 1), dpi=100)
ax.plot(values)
fig.savefig(OUTPUT_PATH)
"""
    result = await render(
        code=code, width=80, height=60, image=IMAGE, provider=PROVIDER, artifact_dir=tmp_path, timeout=10
    )
    assert result.status == "ok", result.model_dump()
    assert not result.dimension_mismatch and (result.width, result.height) == (80, 60)
    assert result.diagnostics["stopped"]


@pytest.mark.parametrize(
    "save",
    [
        "plt.savefig('/missing/incorrect.svg', dpi=250, bbox_inches='tight', pad_inches=2)",
        "fig.savefig(fname='/missing/incorrect.pdf', dpi=250, bbox_inches='tight', pad_inches=2)",
    ],
    ids=["pyplot", "figure"],
)
async def test_real_matplotlib_output_helpers(tmp_path: Path, save: str) -> None:
    from PIL import Image

    code = (
        """
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(5, 5), dpi=300, facecolor='red')
ax = fig.add_axes([0.25, 0.25, 0.5, 0.5])
ax.set_facecolor('blue')
ax.set_xticks([])
ax.set_yticks([])
"""
        + save
    )
    result = await render(
        code=code, width=160, height=120, image=IMAGE, provider=PROVIDER, artifact_dir=tmp_path, timeout=10
    )
    assert result.status == "ok", result.model_dump()
    assert not result.dimension_mismatch and (result.width, result.height) == (160, 120)
    assert result.diagnostics["stopped"]
    with Image.open(tmp_path / "output.png") as rendered:
        assert rendered.format == "PNG"
        assert rendered.info["dpi"] == pytest.approx((100, 100), abs=0.01)
        # Cropping must retain the full figure background, not just the central axes.
        assert rendered.convert("RGB").getpixel((0, 0)) == (255, 0, 0)
        assert rendered.convert("RGB").getpixel((80, 60)) == (0, 0, 255)


async def test_real_concurrent_workspaces(tmp_path):
    import asyncio

    from PIL import Image

    async def one(index):
        code = f"from PIL import Image; Image.new('RGB',(80,60),({index},0,0)).save(OUTPUT_PATH)"
        result = await render(
            code=code,
            width=80,
            height=60,
            image=IMAGE,
            provider=PROVIDER,
            artifact_dir=tmp_path / str(index),
            timeout=10,
        )
        assert result.status == "ok", result.model_dump()
        assert result.diagnostics["stopped"]
        with Image.open(tmp_path / str(index) / "output.png") as im:
            assert im.getpixel((0, 0)) == (index, 0, 0)

    await asyncio.gather(*(one(i) for i in range(4)))


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(
            """
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
drawer = rdMolDraw2D.MolDraw2DCairo(TARGET_WIDTH, TARGET_HEIGHT)
drawer.DrawMolecule(Chem.MolFromSmiles('CC(=O)O'))
drawer.FinishDrawing()
drawer.WriteDrawingText(OUTPUT_PATH)
""",
            id="rdkit-cairo",
        ),
        pytest.param(
            """
import cairosvg
import schemdraw
import schemdraw.elements as elm
drawing = schemdraw.Drawing(canvas='svg', show=False)
drawing.add(elm.Resistor().label('R1'))
cairosvg.svg2png(bytestring=drawing.get_imagedata('svg'), write_to=OUTPUT_PATH,
                output_width=TARGET_WIDTH, output_height=TARGET_HEIGHT, background_color='white')
""",
            id="schemdraw-cairosvg",
        ),
        pytest.param(
            """
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
values = gaussian_filter1d(np.array([0., 1., 4., 1., 0.]), sigma=0.5)
fig, ax = plt.subplots(figsize=(TARGET_WIDTH / 100, TARGET_HEIGHT / 100), dpi=100)
sns.lineplot(x=np.arange(len(values)), y=values, ax=ax)
fig.savefig(OUTPUT_PATH, dpi=100)
""",
            id="scipy-seaborn",
        ),
        pytest.param(
            """
import cairosvg
import svgwrite
from shapely.geometry import Polygon, box
shape = Polygon([(10, 10), (150, 10), (80, 110)]).intersection(box(20, 20, 140, 100))
drawing = svgwrite.Drawing(size=(TARGET_WIDTH, TARGET_HEIGHT))
drawing.add(drawing.polygon(list(shape.exterior.coords), fill='blue'))
cairosvg.svg2png(bytestring=drawing.tostring().encode(), write_to=OUTPUT_PATH,
                background_color='white')
""",
            id="shapely-svgwrite",
        ),
        pytest.param(
            """
import cairosvg
from reportlab.graphics import renderSVG
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib.colors import blue
drawing = Drawing(TARGET_WIDTH, TARGET_HEIGHT)
drawing.add(Rect(10, 10, 100, 60, fillColor=blue))
drawing.add(String(10, 90, 'ReportLab'))
cairosvg.svg2png(bytestring=renderSVG.drawToString(drawing).encode(), write_to=OUTPUT_PATH,
                background_color='white')
""",
            id="reportlab-svg",
        ),
    ],
)
async def test_real_rendering_libraries(tmp_path: Path, code: str) -> None:
    """Exercise native extensions, drawing backends, and SVG rasterization in isolation."""
    from PIL import Image

    result = await render(
        code=code, width=160, height=120, image=IMAGE, provider=PROVIDER, artifact_dir=tmp_path, timeout=30
    )
    assert result.status == "ok", result.model_dump()
    assert (result.width, result.height) == (160, 120)
    assert not result.dimension_mismatch
    assert result.diagnostics["stopped"]
    with Image.open(tmp_path / "output.png") as rendered:
        assert rendered.convert("L").getextrema()[0] < 200
        assert rendered.convert("L").getextrema()[1] > 240


async def test_real_worker_isolation(tmp_path: Path) -> None:
    """The renderer must hide host files, credentials, network, and other processes."""
    host_only = tmp_path / "host-only.txt"
    host_only.write_text("Must not be visible inside the renderer")
    forbidden = [str(Path.home()), str(Path.cwd()), str(host_only), "/root/.aws", "/root/.ssh"]
    code = """
import json, os, socket
from PIL import Image, ImageDraw
allowed = {'PATH','HOME','LANG','TMPDIR','MPLBACKEND','MPLCONFIGDIR','OPENBLAS_NUM_THREADS',
           'OMP_NUM_THREADS','OUTPUT_PATH','TARGET_WIDTH','TARGET_HEIGHT'}
unexpected_env = sorted(set(os.environ) - allowed)
visible_forbidden = [p for p in __FORBIDDEN_PATHS__ if os.path.exists(p)]
network_blocked = False
try:
    with socket.create_connection(('1.1.1.1', 443), timeout=2):
        pass
except OSError:
    network_blocked = True
routes = open('/proc/net/route').read().splitlines()[1:]
processes = [name for name in os.listdir('/proc') if name.isdigit()]
report = dict(network_blocked=network_blocked, routes=routes,
              unexpected_env=unexpected_env, visible_forbidden=visible_forbidden,
              processes_visible=len(processes), environment_keys=sorted(os.environ))
print(json.dumps(report), flush=True)
assert network_blocked and not routes and not unexpected_env and not visible_forbidden, report
assert len(processes) == 1, 'Parent/host processes are visible'
im = Image.new('RGB', (TARGET_WIDTH, TARGET_HEIGHT), 'white')
ImageDraw.Draw(im).rectangle((10, 10, 50, 50), fill='black')
im.save(OUTPUT_PATH)
""".replace("__FORBIDDEN_PATHS__", repr(forbidden))
    result = await render(
        code=code,
        width=80,
        height=60,
        image=IMAGE,
        provider=PROVIDER,
        artifact_dir=tmp_path / "render",
        timeout=10,
    )
    assert result.status == "ok", result.model_dump()
    assert result.diagnostics["stopped"]


@pytest.mark.parametrize("image_format,tight", [("JPEG", False), ("PNG", True)])
async def test_real_matplotlib_preserves_format_and_global_cropping(tmp_path, image_format, tight):
    from PIL import Image

    code = f"""
import matplotlib.pyplot as plt
plt.rcParams['savefig.bbox'] = {"tight" if tight else None!r}
fig = plt.figure(figsize=(5, 5), dpi=300)
ax = fig.add_axes([0.25, 0.25, 0.5, 0.5])
ax.plot([0, 1], [0, 1])
ax.set_axis_off()
fig.savefig('/missing/image.png', format={image_format!r})
"""
    result = await render(
        code=code, width=160, height=120, image=IMAGE, provider=PROVIDER, artifact_dir=tmp_path, timeout=15
    )
    assert result.status == "ok", result.model_dump()
    assert result.diagnostics["stopped"]
    with Image.open(tmp_path / "output.png") as image:
        assert image.format == image_format
        assert (image.size != (160, 120)) == tight
