# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real concurrent filesystem test; enable with GYM_EVAL_SANDBOX_TEST_CONFIG."""

import asyncio
import json
import os
from pathlib import Path

import pytest
from PIL import Image

from responses_api_agents.opencode_multimodal_agent.eval_sandbox import EvalSandboxConfig, sandbox_command


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace_tmp", [False, True])
@pytest.mark.skipif(
    not os.environ.get("GYM_EVAL_SANDBOX_TEST_CONFIG"), reason="requires installed bubblewrap and tool Python"
)
async def test_concurrent_episodes_have_private_tmp_and_inputs(tmp_path, workspace_tmp):
    settings = json.loads(Path(os.environ["GYM_EVAL_SANDBOX_TEST_CONFIG"]).read_text())
    cfg = EvalSandboxConfig.model_validate(settings["sandbox"])
    cfg.workspace_tmp = workspace_tmp
    env = {
        "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": "131072",
        "OPENCODE_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS": "120000",
    }

    hidden = tmp_path / "host-sentinel"
    hidden.write_text("must not be visible in sandbox")

    async def run(marker):
        workspace = tmp_path / marker
        workspace.mkdir()
        (workspace / "input-0.txt").write_text(marker)
        color = (255, 0, 0) if marker == "episode-a" else (0, 0, 255)
        Image.new("RGB", (4, 3), color).save(workspace / "input-0.png")
        code = "\n".join(
            [
                "from pathlib import Path",
                "from PIL import Image",
                "import time, os",
                "p=Path('/tmp/opencode/person_crop.png'); p.parent.mkdir(exist_ok=True)",
                "Image.open('input-0.png').crop((0,0,2,2)).save(p)",
                "time.sleep(0.5)",
                "assert Image.open(p).getpixel((0,0))==Image.open('input-0.png').getpixel((0,0))",
                *(
                    [
                        "assert p.samefile('tmp/opencode/person_crop.png')",
                        "Path('tmp/relative.txt').write_text(Path('input-0.txt').read_text())",
                        "assert Path('/tmp/relative.txt').read_text()==Path('input-0.txt').read_text()",
                        "assert os.environ['TMPDIR']=='/testbed/tmp'",
                        "assert os.environ['TMP']==os.environ['TEMP']==os.environ['TMPDIR']",
                    ]
                    if workspace_tmp
                    else ["assert not Path('tmp').exists()"]
                ),
                f"assert not Path({str(hidden)!r}).exists()",
                "assert 'OPENAI_API_KEY' not in os.environ",
                "print(Path('input-0.txt').read_text())",
            ]
        )
        command = sandbox_command(
            cfg, workspace, Path(settings["command"]), env, [str(cfg.python_env / "bin/python3"), "-c", code]
        )
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ | {"OPENAI_API_KEY": "test-secret"},
        )
        out, err = await asyncio.wait_for(proc.communicate(), 30)
        assert proc.returncode == 0, err.decode(errors="replace")
        return out.decode().strip()

    assert await asyncio.gather(run("episode-a"), run("episode-b")) == ["episode-a", "episode-b"]
