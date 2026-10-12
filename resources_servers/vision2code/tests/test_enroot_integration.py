# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in real execution tests. No host execution fallback is permitted."""

import os
import shutil

import pytest

from resources_servers.vision2code.render import render


IMAGE = os.environ.get("VISION2CODE_TEST_SQSH")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not IMAGE or not shutil.which("enroot"), reason="Set VISION2CODE_TEST_SQSH in an Enroot worker allocation"
    ),
]
PROVIDER = {
    "enroot": {
        "create": {"mount_point": "/workspace"},
        "probe": {"command": "test -d /workspace", "expected_stdout": None},
    }
}
VALID = "from PIL import Image\nImage.new('RGB',(TARGET_WIDTH,TARGET_HEIGHT),'white').save(OUTPUT_PATH)"


@pytest.mark.parametrize(
    "code,status",
    [
        (VALID, "ok"),
        (f"```python\n{VALID}\n```", "ok"),
        ("this is not python !", "code_error"),
        ("while True: pass", "timeout"),
        ("pass", "missing_output"),
        ("open(OUTPUT_PATH,'w').write('not png')", "invalid_output"),
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
    assert result.dimension_mismatch and (result.width, result.height) == (100, 100)
    assert result.diagnostics["stopped"]


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
