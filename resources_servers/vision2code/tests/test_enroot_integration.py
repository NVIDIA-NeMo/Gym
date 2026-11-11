# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in real execution tests. No host execution fallback is permitted."""

import hashlib
import json
import os
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from resources_servers.vision2code.app import Vision2CodeConfig, Vision2CodeServer, Vision2CodeVerifyRequest
from resources_servers.vision2code.generic_rubric import GENERIC_RUBRIC
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
        ("raise SystemExit(1)", "code_error"),
        ("import sys; sys.stderr.write('[ERROR] deliberate failure\\n'); raise SystemExit(1)", "code_error"),
        ("import os; os.write(2, b'[ERROR] direct fd write\\n'); raise SystemExit(1)", "code_error"),
        (VALID + "\nimport os; os.chmod(OUTPUT_PATH, 0)", "invalid_output"),
        (VALID + "\nimport os; os.chmod('/workspace', 0)", "ok"),
        (VALID + "\nlabel = '```py``` <think>literal</think>'", "ok"),
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
    if "[ERROR]" in code:
        assert "[ERROR]" in result.diagnostics["execution"]["stdout"]
        assert result.diagnostics["execution"]["error_type"] is None


@pytest.mark.parametrize(
    "code,status",
    [
        ("raise SystemExit(1)", "code_error"),
        ("import sys; sys.stderr.write('[ERROR] deliberate failure\\n'); raise SystemExit(1)", "code_error"),
        (VALID + "\nimport os; os.chmod(OUTPUT_PATH, 0)", "invalid_output"),
        (VALID + "\nimport os; os.chmod('/workspace', 0)", "ok"),
        pytest.param(
            f"Notes:\n```text\nUse a white background.\n```\n```python\n{VALID}\n```", "ok", id="text-before-python"
        ),
        pytest.param(
            f"<think>\n```text\nUse a white background.\n```\n</think>\n```python\n{VALID}\n```",
            "ok",
            id="text-inside-reasoning",
        ),
        pytest.param(f"```python\n{VALID}\n```\n```text\nUse a white background.\n```", "ok", id="text-after-python"),
    ],
)
async def test_real_program_outcomes_are_not_masked(tmp_path: Path, monkeypatch, code: str, status: str) -> None:
    reference = tmp_path / "reference.png"
    Image.new("RGB", (80, 60), "white").save(reference)
    server = Vision2CodeServer(
        config=Vision2CodeConfig(
            name="vision2code",
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            judge_model_server={"type": "responses_api_models", "name": "judge"},
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            sandbox_image=IMAGE,
            artifact_dir=str(tmp_path / "artifacts"),
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    body = Vision2CodeVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=NeMoGymResponse(
            id="response",
            created_at=0,
            model="test",
            object="response",
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
            output=[
                {
                    "id": "message",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": code, "annotations": []}],
                }
            ],
        ),
        verifier_metadata={
            "id": "sample",
            "image_path": str(reference),
            "rubric": "generic",
            "width": 80,
            "height": 60,
            "sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        },
    )
    judgment = body.response.model_copy(deep=True)
    judgment.output[0].content[0].text = json.dumps(
        {"category_scores": {category["id"]: 4 for category in GENERIC_RUBRIC["categories"]}, "rationales": {}}
    )
    judge = AsyncMock(return_value=judgment)
    monkeypatch.setattr("resources_servers.vision2code.app.request_judgment", judge)
    result = await server.verify(body)
    assert not result.mask_sample
    if status != "ok":
        assert result.reward == 0
        judge.assert_not_awaited()
    else:
        judge.assert_awaited_once()
    assert result.render.status == status
    assert len(result.render_attempts) == 1
    assert result.render.diagnostics["stopped"]
    assert not server._pending_cleanup


@pytest.mark.parametrize("literal", ["<think>", "</think>", "<think>visible</think>"])
async def test_real_fenced_literals_match_raw_pixels(tmp_path: Path, literal: str) -> None:
    code = (
        "from PIL import Image, ImageDraw\n"
        "im = Image.new('RGB', (TARGET_WIDTH, TARGET_HEIGHT), 'white')\n"
        f"ImageDraw.Draw(im).text((0, 0), {literal!r}, fill='black')\n"
        "im.save(OUTPUT_PATH)"
    )
    for name, submitted in [("raw", code), ("fenced", f"```python\n{code}\n```")]:
        result = await render(
            code=submitted,
            width=160,
            height=60,
            image=IMAGE,
            provider=PROVIDER,
            artifact_dir=tmp_path / name,
            timeout=10,
        )
        assert result.status == "ok", result.model_dump()
        assert result.diagnostics["stopped"]
    with Image.open(tmp_path / "raw/output.png") as raw, Image.open(tmp_path / "fenced/output.png") as fenced:
        assert raw.size == fenced.size
        assert raw.tobytes() == fenced.tobytes()


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
