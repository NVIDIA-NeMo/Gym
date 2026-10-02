# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tool results of the OpenSandbox code-exec provider, run against a local bash standing in for the sandbox."""

import asyncio
import shutil

import pytest

from nemo_gym.sandbox.providers import SandboxExecResult
from responses_api_agents.stirrup_agent import opensandbox_provider
from responses_api_agents.stirrup_agent.opensandbox_provider import OpenSandboxCodeExecToolProvider, shell_stdout


class _LocalSandbox:
    """Runs commands with the host's bash; stdout mimics execd's background logs (stdout and stderr merged)."""

    async def exec(self, command, *, timeout_s=None, **_):
        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            raise TimeoutError("timed out")
        return SandboxExecResult(stdout=out.decode(), stderr=None, return_code=proc.returncode)

    async def download(self, remote_path, local_path):
        shutil.copyfile(remote_path, local_path)

    async def upload(self, local_path, remote_path):
        shutil.copyfile(local_path, remote_path)

    async def stop(self):
        pass


@pytest.fixture
def provider(tmp_path, monkeypatch):
    monkeypatch.setattr(opensandbox_provider, "IO_DIR", str(tmp_path / "io"))
    p = OpenSandboxCodeExecToolProvider("img", working_dir=str(tmp_path / "root"))
    (tmp_path / "root").mkdir()
    p._sandbox = _LocalSandbox()
    return p


@pytest.mark.parametrize(
    "raw,expected",
    [(b"", b"\n"), (b"out", b"out\n"), (b"out\n", b"out\n"), (b"a\n\n", b"a\n\n"), (b"\n", b"\n")],
)
def test_shell_stdout_matches_apptainer_marker_echo(raw, expected):
    assert shell_stdout(raw) == expected


def test_streams_stay_separate(provider):
    result = asyncio.run(provider.run_command("echo out; echo err >&2; exit 3"))
    assert (result.exit_code, result.stdout, result.stderr, result.error_kind) == (3, "out\n", "err\n", None)


def test_bytes_are_kept(provider):
    result = asyncio.run(provider.run_command("printf 'a\\r\\n\\nb'"))
    assert result.stdout == "a\r\n\nb\n"


def test_each_call_starts_in_working_dir_with_fresh_shell(provider, tmp_path):
    asyncio.run(provider.run_command("echo hi > f.txt; cd /tmp; export X=1"))
    result = asyncio.run(provider.run_command('pwd; echo "${X:-unset}"; cat f.txt'))
    assert result.stdout == f"{tmp_path / 'root'}\nunset\nhi\n"


def test_timeout_matches_apptainer_message(provider):
    result = asyncio.run(provider.run_command("echo partial; sleep 30", timeout=1))
    assert (result.exit_code, result.stdout, result.stderr) == (1, "partial\n", "Command timed out after 1 seconds")


def test_io_dirs_are_cleaned_up(provider, tmp_path):
    for _ in range(3):
        asyncio.run(provider.run_command("true"))
    assert len(list((tmp_path / "io").iterdir())) == 1


def test_upload_and_save_round_trip(provider, tmp_path):
    src = tmp_path / "refs"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "a.bin").write_bytes(b"\x00\xff")
    uploaded = asyncio.run(provider.upload_files(src))
    assert not uploaded.failed and uploaded.uploaded[0].dest_path == f"{tmp_path / 'root'}/sub/a.bin"

    out = tmp_path / "out"
    saved = asyncio.run(provider.save_output_files(["sub/a.bin", "missing.txt"], out))
    assert (out / "a.bin").read_bytes() == b"\x00\xff"
    assert list(saved.failed) == ["missing.txt"]
