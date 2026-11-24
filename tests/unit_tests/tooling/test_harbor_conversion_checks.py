# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`python -m scripts.harbor_conversion_checks`: each command on clean and on broken input."""

import base64
import bz2
import gzip
import hashlib
import io
import json
import lzma
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import tomllib
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path

import pytest
from scripts.harbor_conversion_checks import check, first_request, scan
from scripts.harbor_conversion_checks.__main__ import main
from scripts.harbor_conversion_checks.fake_model import FakeModelServer
from scripts.harbor_conversion_checks.messages import message_text, unquote_argv

from nemo_gym.data.tasks.harbor.dataset_config import apply_dataset_config, read_dataset_config
from nemo_gym.data.tasks.harbor.models import HarborTaskConfig


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


# messages


def test_message_text_and_unquote_argv():
    assert message_text("plain") == "plain"
    assert message_text([{"type": "text", "text": "a"}, {"type": "image"}, {"type": "text", "text": "b"}]) == "ab"
    assert message_text(None) == ""
    assert unquote_argv('"say \\"hi\\" with \\\\cdots"') == 'say "hi" with \\\\cdots'
    assert unquote_argv("not quoted") is None
    assert unquote_argv('"bare " quote"') is None


# controls


def _task_folders(root):
    for name in ("a", "b"):
        (root / name / "solution").mkdir(parents=True)
        (root / name / "environment").mkdir()
        (root / name / "task.toml").write_text('version = "1.0"\n')
        (root / name / "solution" / "solve.sh").write_text("#!/bin/bash\necho reference\n")
        (root / name / "solution" / "golden.patch").write_text("diff\n")
        (root / name / "environment" / "Dockerfile").write_text("FROM x\n")
    os.symlink("Dockerfile", root / "a" / "environment" / "link")
    return root


def _template(root):
    (root / "solution").mkdir(parents=True)
    (root / "task.toml").write_text('version = "1.0"\n')
    (root / "tasks.jsonl").write_text('{"task_id": "x"}\n{"task_id": "y"}\n{"task_id": "z"}\n')
    (root / "instruction.template.md").write_text("{{ task_data.q }}\n")
    (root / "solution" / "answer.template.md").write_text("{{ task_data.answer }}")
    return root


def _files(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() or p.is_symlink())


@pytest.mark.parametrize("layout", ["folders", "template"])
def test_controls_replace_only_the_solution(tmp_path, capsys, layout):
    dataset = _task_folders(tmp_path / "ds") if layout == "folders" else _template(tmp_path / "ds")
    out = tmp_path / "out"
    assert main(["controls", str(dataset), str(out), "--empty"]) == 0
    solutions = ["a/solution", "b/solution"] if layout == "folders" else ["solution"]
    kept = [f for f in _files(dataset) if not any(f.startswith(s + "/") for s in solutions)]
    assert _files(out) == sorted(kept + [f"{s}/solve.sh" for s in solutions])
    for solution in solutions:
        script = out / solution / "solve.sh"
        assert script.read_text() == "#!/bin/bash\nexit 0\n"
        assert script.stat().st_mode & 0o777 == 0o755
    if layout == "folders":
        assert os.readlink(out / "a" / "environment" / "link") == "Dockerfile"
    count = 2 if layout == "folders" else 3
    assert capsys.readouterr().out.strip() == f"controls: wrote --empty for {count} task(s) to {out}"


def test_controls_solution_dir_is_copied_in(tmp_path):
    dataset = _template(tmp_path / "ds")
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    (wrong / "answer.template.md").write_text("\\boxed{-987654321}")
    assert main(["controls", str(dataset), str(tmp_path / "out"), "--solution-dir", str(wrong)]) == 0
    assert _files(tmp_path / "out" / "solution") == ["answer.template.md"]
    assert (tmp_path / "out" / "solution" / "answer.template.md").read_text() == "\\boxed{-987654321}"


def test_controls_refuse_an_existing_out(tmp_path, capsys):
    dataset = _task_folders(tmp_path / "ds")
    (tmp_path / "out").mkdir()
    assert main(["controls", str(dataset), str(tmp_path / "out"), "--empty"]) == 2
    assert "already exists" in capsys.readouterr().err
    assert list((tmp_path / "out").iterdir()) == []


def test_controls_find_grouped_task_folders_as_gym_does(tmp_path, capsys):
    dataset = tmp_path / "ds"
    _task_folders(dataset / "group" / "sub")
    for name in ("c", ".hidden/d"):
        (dataset / name / "solution").mkdir(parents=True)
        (dataset / name / "task.toml").write_text('version = "1.0"\n')
        (dataset / name / "solution" / "solve.sh").write_text("echo reference\n")
    out = tmp_path / "out"
    assert main(["controls", str(dataset), str(out), "--empty"]) == 0
    assert capsys.readouterr().out.strip() == f"controls: wrote --empty for 3 task(s) to {out}"
    for task in ("group/sub/a", "group/sub/b", "c"):
        assert (out / task / "solution" / "solve.sh").read_text() == "#!/bin/bash\nexit 0\n"
    assert (out / ".hidden" / "d" / "solution" / "solve.sh").read_text() == "echo reference\n"
    (dataset / "group" / "c").mkdir()
    (dataset / "group" / "c" / "task.toml").write_text('version = "1.0"\n')
    assert main(["controls", str(dataset), str(tmp_path / "out2"), "--empty"]) == 2
    assert "two task folders are named c" in capsys.readouterr().err


def test_controls_refuse_a_symlinked_task_folder(tmp_path, capsys):
    real = tmp_path / "real" / "linked"
    (real / "environment").mkdir(parents=True)
    (real / "task.toml").write_text('version = "1.0"\n')
    dataset = _task_folders(tmp_path / "ds")
    os.symlink(real, dataset / "linked")
    before = _files(real)
    assert main(["controls", str(dataset), str(tmp_path / "out"), "--empty"]) == 2
    assert "linked: a task folder must not be a symbolic link" in capsys.readouterr().err
    assert _files(real) == before == ["task.toml"] and not (real / "solution").exists()


def test_controls_drop_the_solutions_of_steps(tmp_path):
    dataset = _task_folders(tmp_path / "ds")
    step = dataset / "a" / "steps" / "one"
    (step / "solution").mkdir(parents=True)
    (step / "solution" / "solve.sh").write_text("echo reference\n")
    (step / "instruction.md").write_text("Step one.\n")
    out = tmp_path / "out"
    assert main(["controls", str(dataset), str(out), "--empty"]) == 0
    assert not (out / "a" / "steps" / "one" / "solution").exists()
    assert (out / "a" / "steps" / "one" / "instruction.md").read_text() == "Step one.\n"
    assert (out / "a" / "solution" / "solve.sh").read_text() == "#!/bin/bash\nexit 0\n"


def _wait_for(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_plant_reward_writes_both_rewards_and_stubs(tmp_path):
    dataset = _task_folders(tmp_path / "ds")
    assert main(["controls", str(dataset), str(tmp_path / "out"), "--plant-reward"]) == 0
    root = tmp_path / "root"
    # solve.sh counts whole seconds, so PROBE_SECONDS=2 keeps the loop running for at least one more second.
    env = {**os.environ, "PROBE_ROOT": str(root), "PROBE_SECONDS": "2"}
    subprocess.run(["bash", str(tmp_path / "out" / "a" / "solution" / "solve.sh")], env=env, check=True, timeout=30)
    # The background loop truncates and rewrites both files while the test reads them: read until a whole write.
    assert _wait_for(lambda: (root / "logs/verifier/reward.json").read_text() == '{"reward": 1.0}\n')
    assert _wait_for(lambda: (root / "logs/verifier/reward.txt").read_text() == "1\n")
    (root / "logs/verifier/reward.txt").unlink()
    assert _wait_for(lambda: (root / "logs/verifier/reward.txt").is_file()), "the loop did not rewrite the reward"
    time.sleep(2.5)  # the loop ends after PROBE_SECONDS; then only the stubs can write the reward
    (root / "logs/verifier/reward.txt").unlink()
    (root / "logs/verifier/reward.json").unlink()
    stub = root / "usr/local/bin/python3"
    assert os.access(stub, os.X_OK) and os.access(root / "usr/local/bin/python", os.X_OK)
    subprocess.run([str(stub), "/tests/grade.py"], check=True, timeout=10)
    assert (root / "logs/verifier/reward.txt").read_text().strip() == "1"
    assert json.loads((root / "logs/verifier/reward.json").read_text()) == {"reward": 1.0}


def _sleeper(argument):
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", argument])


def _probe_env(root, kill_prefix):
    # The shell probe counts whole seconds, so PROBE_SECONDS=2 keeps its loop running for at least one more second.
    # The kill prefix is under the test's tmp_path, so the kill step can match only the test's own sleeper.
    env = {key: value for key, value in os.environ.items() if key != "PROBE_KILL"}
    return {**env, "PROBE_ROOT": str(root), "PROBE_SECONDS": "2", "PROBE_KILL_PREFIX": kill_prefix}


def _run_kill_step(command, tmp_path, **kwargs):
    """Runs a probe with the kill step on: the test's sleeper under PROBE_KILL_PREFIX is killed, and a sleeper with an
    argument under the default /tests/ (any other process on the host) survives."""
    kill_prefix = f"{tmp_path}/tests/"
    target = _sleeper(f"{kill_prefix}nemo_probe_target_{uuid.uuid4().hex}")
    bystander = _sleeper(f"/tests/nemo_probe_bystander_{uuid.uuid4().hex}")
    try:
        subprocess.run(command, env=_probe_env(tmp_path / "root", kill_prefix), check=True, timeout=30, **kwargs)
        assert target.wait(timeout=10) == -signal.SIGKILL
        time.sleep(0.5)
        assert bystander.poll() is None, "the kill step killed a process outside PROBE_KILL_PREFIX"
    finally:
        target.kill()
        bystander.kill()
        bystander.wait(timeout=10)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.skipif(sys.platform != "linux", reason="the kill step reads /proc")
def test_plant_conftest_writes_both_rewards_on_import(tmp_path):
    dataset = _task_folders(tmp_path / "ds")
    assert main(["controls", str(dataset), str(tmp_path / "out"), "--plant-conftest"]) == 0
    repo, root = tmp_path / "repo", tmp_path / "root"
    repo.mkdir()
    subprocess.run(["bash", str(tmp_path / "out" / "b" / "solution" / "solve.sh")], cwd=repo, check=True, timeout=30)
    assert (repo / "conftest.py").is_file() and not root.exists()
    _run_kill_step(["python3", "-c", "import conftest"], tmp_path, cwd=repo)
    assert json.loads((root / "logs/verifier/reward.json").read_text()) == {"reward": 1.0}
    (root / "logs/verifier/reward.txt").unlink()
    assert _wait_for(lambda: (root / "logs/verifier/reward.txt").is_file()), "the daemon did not rewrite the reward"
    time.sleep(2.5)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.skipif(sys.platform != "linux", reason="the kill step reads /proc")
def test_plant_hooks_leaves_go_pytest_and_npm_hooks_that_write_the_reward(tmp_path):
    dataset = _task_folders(tmp_path / "ds")
    assert main(["controls", str(dataset), str(tmp_path / "out"), "--plant-hooks"]) == 0
    repo, root = tmp_path / "repo", tmp_path / "root"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "a_test.go").write_text("package pkg\n")
    (repo / "package.json").write_text('{"name": "p", "scripts": {"test": "jest"}}\n')
    env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(tmp_path)}
    subprocess.run(["bash", str(tmp_path / "out" / "a" / "solution" / "solve.sh")], cwd=repo, env=env, check=True)
    probe = repo / ".nemo_probe.sh"
    assert os.access(probe, os.X_OK) and not root.exists()
    go_hook = (repo / "pkg" / "zz_nemo_probe_test.go").read_text()
    assert go_hook.startswith("package pkg\n") and f'exec.Command("sh", "{probe}")' in go_hook
    assert f'subprocess.run(["sh", "{probe}"])' in (repo / "conftest.py").read_text()
    scripts = json.loads((repo / "package.json").read_text())["scripts"]
    if shutil.which("node"):
        assert scripts["pretest"] == f"sh {probe}" and scripts["test"] == "jest"
    _run_kill_step(["sh", str(probe)], tmp_path)
    assert json.loads((root / "logs/verifier/reward.json").read_text()) == {"reward": 1.0}
    (root / "logs/verifier/reward.txt").unlink()
    assert _wait_for(lambda: (root / "logs/verifier/reward.txt").is_file()), "the loop did not rewrite the reward"
    time.sleep(3.5)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh")
def test_plant_collection_replaces_only_the_named_tools_and_forges_the_reward(tmp_path):
    dataset = _task_folders(tmp_path / "ds")
    assert main(["controls", str(dataset), str(tmp_path / "out"), "--plant-collection"]) == 0
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(tmp_path)}
    subprocess.run(["bash", str(tmp_path / "out" / "a" / "solution" / "solve.sh")], cwd=repo, env=env, check=True)
    probe = repo / ".nemo_collection_probe.sh"
    assert os.access(probe, os.X_OK)
    assert ".nemo_collection_probe.sh" in (repo / "conftest.py").read_text()

    # Every directory and tool name the probe touches is a parameter, so the test points them all at tmp_path and
    # replaces only stand-in programs of its own: no real program on the host is touched. The default in-sandbox KEEP
    # directory must stay absent to prove nothing outside tmp_path was written.
    sandbox = tmp_path / "sandbox"
    stand_in, place, keep, root = (sandbox / name for name in ("stand_in_bin", "place_bin", "keep", "root"))
    stand_in.mkdir(parents=True)
    ran = sandbox / "ran.txt"
    tools = ("probe_tool_one", "probe_tool_two")
    for name in tools:
        program = stand_in / name
        program.write_text(f"#!/bin/sh\necho '{name} original ran' >> {ran}\n")
        program.chmod(0o755)
    # place is searched before stand_in, so `command -v` finds the stand-ins until the probe replaces them there too.
    env = {
        **os.environ,
        "PATH": f"{place}{os.pathsep}{stand_in}{os.pathsep}{os.environ['PATH']}",
        "PROBE_ROOT": str(root),
        "PROBE_COLLECT_TOOLS": " ".join(tools),
        "PROBE_COLLECT_BIN": str(place),
        "PROBE_COLLECT_KEEP": str(keep),
    }
    # Several test processes may import conftest.py, so the probe runs more than once: the later runs must leave the
    # kept programs alone, or a replacement would be kept as the real program and exec itself for ever.
    for _ in range(2):
        subprocess.run(["sh", str(probe)], env=env, check=True, timeout=30)

    for name in tools:
        assert (keep / f"real_{name}").is_file(), "the real program was not kept"
        assert "original ran" in (keep / f"real_{name}").read_text(), "a replacement was kept as the real program"
        assert (place / name).is_file(), "no replacement was placed earlier on PATH"
        assert "reward.json" in (stand_in / name).read_text(), "the stand-in was not rewritten in place"
    # The probe only installs the replacements; nothing writes the reward until a replaced program runs.
    assert not (root / "logs" / "verifier" / "reward.json").exists()
    assert not Path("/usr/local/lib/nemo_collection_probe").exists(), "the probe wrote to a path outside tmp_path"

    # A replaced program (placed on PATH, and the one rewritten in place) forges the reward, then runs the original.
    subprocess.run([str(place / "probe_tool_one")], env=env, check=True, timeout=30)
    assert json.loads((root / "logs" / "verifier" / "reward.json").read_text()) == {"reward": 1.0}
    assert (root / "logs" / "verifier" / "reward.txt").read_text().strip() == "1"
    assert "probe_tool_one original ran" in ran.read_text()
    subprocess.run([str(stand_in / "probe_tool_two")], env=env, check=True, timeout=30)
    assert "probe_tool_two original ran" in ran.read_text()


# fake-model


@pytest.fixture
def fake_model(tmp_path):
    replies = _write_jsonl(tmp_path / "replies.jsonl", [{"match": 'Is "x" odd?', "text": "Answer: \\boxed{no}"}])
    server = FakeModelServer(0, replies=replies, reply="default reply")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", replies
    server.shutdown()
    server.server_close()


def _post(base, body):
    request = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode())
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.headers.get("Content-Type"), response.read()


def _chat(user, **extra):
    return {
        "model": "m1",
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": user}],
        **extra,
    }


def test_fake_model_json_match_default_and_models(fake_model):
    base, _ = fake_model
    _, raw = _post(base, _chat('Please: Is "x" odd?'))
    body = json.loads(raw)
    assert body["object"] == "chat.completion" and body["model"] == "m1"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Answer: \\boxed{no}"}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert json.loads(_post(base, _chat("something else"))[1])["choices"][0]["message"]["content"] == "default reply"
    with urllib.request.urlopen(base + "/models", timeout=10) as response:
        assert json.loads(response.read())["data"][0]["id"] == "m1"


def test_fake_model_matches_the_argv_quoted_prompt(fake_model):
    base, _ = fake_model
    quoted = '"Please: Is \\"x\\" odd?"'
    assert json.loads(_post(base, _chat(quoted))[1])["choices"][0]["message"]["content"] == "Answer: \\boxed{no}"


def test_fake_model_streams_sse_with_usage_last(fake_model):
    base, _ = fake_model
    content_type, raw = _post(base, _chat('Is "x" odd?', stream=True, stream_options={"include_usage": True}))
    assert content_type == "text/event-stream"
    events = [line[len("data: ") :] for line in raw.decode().split("\n") if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(event) for event in events[:-1]]
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "Answer: \\boxed{no}"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop" and "usage" in chunks[-1]
    assert all("usage" not in c for c in chunks[:-1])
    _, raw = _post(base, _chat("q", stream=True))
    assert all("usage" not in json.loads(e[6:]) for e in raw.decode().split("\n") if e.startswith("data: {"))


def test_fake_model_rereads_the_replies_file(fake_model):
    base, replies = fake_model
    _write_jsonl(replies, [{"match": "odd", "text": "changed"}])
    assert json.loads(_post(base, _chat('Is "x" odd?'))[1])["choices"][0]["message"]["content"] == "changed"


def test_fake_model_unknown_route_is_404(fake_model):
    base, _ = fake_model
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(base.removesuffix("/v1") + "/v1/responses", data=b"{}", timeout=10)
    assert error.value.code == 404


def test_fake_model_needs_a_reply_source(capsys):
    assert main(["fake-model", "--port", "0"]) == 2


def test_fake_model_stops_cleanly_on_sigterm():
    root = Path(__file__).resolve().parents[3]
    command = [sys.executable, "-m", "scripts.harbor_conversion_checks", "fake-model", "--port", "0", "--reply", "x"]
    proc = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().startswith("fake-model listening on http://127.0.0.1:")
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=10) == 0
    finally:
        proc.kill()
        proc.stdout.close()


# first-request


def _capture(
    directory,
    name,
    user,
    *,
    model="m",
    tools=("bash", "read"),
    system="sys",
    temperature=0.6,
    title=True,
    extra=None,
    more_messages=(),
):
    rows = []
    if title:
        title_request = {"model": model, "messages": [{"role": "user", "content": "Generate a title"}]}
        rows.append({"started_at": 1.0, "request": title_request})
    request = {
        "model": model,
        "temperature": temperature,
        "tools": [{"type": "function", "function": {"name": tool}} for tool in tools],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}, *more_messages],
        **(extra or {}),
    }
    later = {**request, "messages": request["messages"] + [{"role": "assistant", "content": "x"}]}
    rows += [{"started_at": 3.0, "request": later}, {"started_at": 2.0, "request": request}]
    _write_jsonl(directory / f"{name}.capture.jsonl", rows)


def _first_request(tmp_path, *extra):
    return main(["first-request", str(tmp_path / "native"), str(tmp_path / "harbor"), *extra])


def test_first_request_equal_pairs_pass(tmp_path, capsys):
    for side in ("native", "harbor"):
        _capture(tmp_path / side, "0-0", "task one")
        _capture(tmp_path / side, "1-0", "task two")
    assert _first_request(tmp_path, "--out", str(tmp_path / "out")) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "first-request: 2 paired, 0 unpaired, 0 differing"
    key = hashlib.sha256(b"task one").hexdigest()[:12]
    assert (tmp_path / "out" / key / "harbor.user.txt").read_text() == "task one"
    assert (tmp_path / "out" / key / "native.system.txt").read_text() == "sys"


def test_first_request_skips_the_title_call_and_takes_the_earliest(tmp_path, capsys):
    _capture(tmp_path / "native", "0-0", "task", title=True)
    _capture(tmp_path / "harbor", "0-0", "task", title=False)
    assert _first_request(tmp_path, "--json", str(tmp_path / "r.json")) == 0
    pair = json.loads((tmp_path / "r.json").read_text())["pairs"][0]
    assert pair["native"]["roles"] == ["system", "user"]


def test_first_request_pairs_an_argv_quoted_prompt_and_reports_it(tmp_path, capsys):
    _capture(tmp_path / "native", "0-0", '"Say \\"hi\\""')
    _capture(tmp_path / "harbor", "0-0", 'Say "hi"')
    assert _first_request(tmp_path) == 0
    out = capsys.readouterr().out
    assert "argv_prompt_quoting" in out and "1 paired, 0 unpaired, 0 differing" in out


def test_first_request_pairs_a_crlf_prompt_and_reports_it(tmp_path, capsys):
    _capture(tmp_path / "native", "0-0", 'Fix it.\r\n\r\nSteps:\r\nrun "x"')
    _capture(tmp_path / "native", "1-0", '"Other\r\ntask"')
    _capture(tmp_path / "harbor", "0-0", 'Fix it.\n\nSteps:\nrun "x"')
    _capture(tmp_path / "harbor", "1-0", "Other\ntask")
    assert _first_request(tmp_path) == 0
    out = capsys.readouterr().out
    assert out.count("newlines: the user messages differ by CRLF or CR vs LF") == 2
    assert out.count("argv_prompt_quoting") == 1 and "2 paired, 0 unpaired, 0 differing" in out


def test_first_request_system_line_difference_is_reported_only(tmp_path, capsys):
    _capture(tmp_path / "native", "0-0", "task", system="Working directory: /a\nrules")
    _capture(tmp_path / "harbor", "0-0", "task", system="Working directory: /workspace\nrules")
    assert _first_request(tmp_path) == 0
    assert "+Working directory: /workspace" in capsys.readouterr().out


@pytest.mark.parametrize(
    "harbor_kwargs, problem",
    [
        ({"tools": ("bash",)}, "tools differs"),
        ({"model": "other"}, "model differs"),
        ({"temperature": 1.0}, "sampling differs"),
        ({"extra": {"tool_choice": "required"}}, "sampling differs"),
        ({"more_messages": [{"role": "user", "content": "Also this."}]}, "roles differ"),
        ({"system": "sys\nInstructions from: /repo/AGENTS.md"}, "flag AGENTS.md on a system-prompt line that differs"),
    ],
)
def test_first_request_differences_fail(tmp_path, capsys, harbor_kwargs, problem):
    _capture(tmp_path / "native", "0-0", "task")
    _capture(tmp_path / "harbor", "0-0", "task", **harbor_kwargs)
    assert _first_request(tmp_path) == 1
    out = capsys.readouterr().out
    assert problem in out and out.splitlines()[-1] == "first-request: 1 paired, 0 unpaired, 1 differing"


def _weather(argument="location", *, description="Get the weather", strict=None, key_order=False):
    parameters = {"type": "object", "properties": {argument: {"type": "string"}}, "required": [argument]}
    function = {"name": "weather", "description": description, "parameters": parameters}
    if strict is not None:
        function["strict"] = strict
    if key_order:
        function = dict(reversed(function.items()))
    return {"type": "function", "function": function}


BASH = {"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}


@pytest.mark.parametrize(
    "harbor_tools, problem",
    [
        ([BASH, _weather()], None),
        ([_weather(key_order=True), BASH], None),
        ([BASH, _weather("query")], "tools differs: the definition of weather"),
        ([BASH, _weather(description="Look up the weather")], "tools differs: the definition of weather"),
        ([BASH, _weather(strict=True)], "tools differs: the definition of weather"),
    ],
)
def test_first_request_compares_tool_definitions_not_only_names(tmp_path, capsys, harbor_tools, problem):
    # The native prompt is argv-quoted, so the pair still needs the quoting normalisation.
    _capture(tmp_path / "native", "0-0", '"Say \\"hi\\""', extra={"tools": [BASH, _weather()]})
    _capture(tmp_path / "harbor", "0-0", 'Say "hi"', extra={"tools": harbor_tools})
    assert _first_request(tmp_path) == (1 if problem else 0)
    out = capsys.readouterr().out
    assert "argv_prompt_quoting" in out
    assert out.splitlines()[-1] == f"first-request: 1 paired, 0 unpaired, {1 if problem else 0} differing"
    if problem:
        assert problem in out


# OpenCode's built-in system prompt, as captured on both paths of the scale_swe pilot.
OPENCODE_BUILTIN = (
    "If you are unable to find the correct command, ask the user for the command to run and if they supply it, "
    "proactively suggest writing it to AGENTS.md so that you will know to run it next time.\n"
    "<available_skills>\n  <skill><name>customize-opencode</name></skill>\n</available_skills>"
)


def test_first_request_flags_of_the_builtin_prompt_on_both_sides_are_noted_only(tmp_path, capsys):
    _capture(tmp_path / "native", "0-0", "task", system=f"Working directory: /a\n{OPENCODE_BUILTIN}")
    _capture(tmp_path / "harbor", "0-0", "task", system=f"Working directory: /workspace\n{OPENCODE_BUILTIN}")
    assert _first_request(tmp_path) == 0
    out = capsys.readouterr().out
    assert "flag AGENTS.md: 1 on both sides, on identical lines" in out
    assert "flag <available_skills>: 1 on both sides, on identical lines" in out


@pytest.mark.parametrize(
    "native_extra, harbor_extra, problem",
    [
        ("", "\nInstructions from: /workspace/AGENTS.md", "flag AGENTS.md on a system-prompt line that differs"),
        (
            "\nInstructions from: /home/u/AGENTS.md",
            "\nInstructions from: /workspace/AGENTS.md",
            "flag Instructions from on a system-prompt line that differs (native 1, harbor 1)",
        ),
    ],
)
def test_first_request_flags_beyond_the_builtin_prompt_fail(tmp_path, capsys, native_extra, harbor_extra, problem):
    _capture(tmp_path / "native", "0-0", "task", system=OPENCODE_BUILTIN + native_extra)
    _capture(tmp_path / "harbor", "0-0", "task", system=OPENCODE_BUILTIN + harbor_extra)
    assert _first_request(tmp_path) == 1
    assert problem in capsys.readouterr().out


def test_first_request_ignores_messages_that_are_not_objects(tmp_path, capsys):
    for side in ("native", "harbor"):
        request = {
            "model": "m",
            "tools": [{"name": "bash"}, "junk"],
            "messages": ["junk", {"role": "user", "content": "t"}],
        }
        _write_jsonl(tmp_path / side / "0-0.capture.jsonl", [{"started_at": 1.0, "request": request}])
    assert _first_request(tmp_path) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "first-request: 1 paired, 0 unpaired, 0 differing"


def test_first_request_unpaired_task_fails(tmp_path, capsys):
    _capture(tmp_path / "native", "0-0", "task one")
    _capture(tmp_path / "harbor", "0-0", "task two")
    assert _first_request(tmp_path) == 1
    assert capsys.readouterr().out.splitlines()[-1] == "first-request: 0 paired, 2 unpaired, 0 differing"


CALL = {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}
CONTEXT = [
    {"role": "user", "content": "Constraints: be brief."},
    {"role": "assistant", "content": None, "tool_calls": [CALL]},
    {"role": "tool", "tool_call_id": "c1", "content": "a.txt"},
    {"role": "user", "content": [{"type": "text", "text": "Look at "}, {"type": "text", "text": "a.txt"}]},
]


def _changed(number, **fields):
    """CONTEXT with message `number` (1-based, after the first user message) given `fields`."""
    return [{**m, **fields} if i == number else m for i, m in enumerate(CONTEXT, 1)]


@pytest.mark.parametrize(
    "harbor_context, problem",
    [
        (CONTEXT, None),
        (_changed(1, content="The answer is 42."), "messages differ: non-system message 2 (user)"),
        (_changed(2, content="I will run ls."), "messages differ: non-system message 3 (assistant)"),
        (
            _changed(2, tool_calls=[{**CALL, "function": {"name": "bash", "arguments": '{"cmd": "cat answer"}'}}]),
            "messages differ: non-system message 3 (assistant)",
        ),
        (_changed(3, content="answer.txt"), "messages differ: non-system message 4 (tool)"),
        (
            _changed(4, content=[{"type": "text", "text": "Look at a.txt"}]),
            "messages differ: non-system message 5 (user)",
        ),
        (
            _changed(4, content=list(reversed(CONTEXT[3]["content"]))),
            "messages differ: non-system message 5 (user)",
        ),
    ],
)
def test_first_request_compares_every_later_message_whole(tmp_path, capsys, harbor_context, problem):
    print(first_request.__file__)
    # The first prompt is argv-quoted and CRLF on the native side: the pairing's normalization still applies.
    _capture(tmp_path / "native", "0-0", '"Say \\"hi\\"\r\nnow"', more_messages=CONTEXT)
    _capture(tmp_path / "harbor", "0-0", 'Say "hi"\nnow', more_messages=harbor_context)
    assert _first_request(tmp_path) == (1 if problem else 0)
    out = capsys.readouterr().out
    assert "argv_prompt_quoting" in out and "newlines: the user messages differ" in out
    assert out.splitlines()[-1] == f"first-request: 1 paired, 0 unpaired, {1 if problem else 0} differing"
    assert (problem in out) if problem else "messages differ: " not in out


IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


@pytest.mark.parametrize(
    "harbor_user, problem",
    [
        ([{"type": "text", "text": "task"}, IMAGE], None),
        ([{"type": "text", "text": "task"}, {**IMAGE, "image_url": {"url": "data:image/png;base64,BBBB"}}], True),
        ([IMAGE, {"type": "text", "text": "task"}], True),
        ([{"type": "text", "text": "task"}], True),
        ("task", True),
    ],
)
def test_first_request_compares_the_first_user_messages_parts_but_its_text(tmp_path, capsys, harbor_user, problem):
    print(first_request.__file__)
    _capture(tmp_path / "native", "0-0", [{"type": "text", "text": "task"}, IMAGE])
    _capture(tmp_path / "harbor", "0-0", harbor_user)
    assert _first_request(tmp_path) == (1 if problem else 0)
    out = capsys.readouterr().out
    assert out.splitlines()[-1] == f"first-request: 1 paired, 0 unpaired, {1 if problem else 0} differing"
    assert ("messages differ: non-system message 1 (user)" in out) == bool(problem)


def test_first_request_extra_harbor_user_message_with_instructions_fails(tmp_path, capsys):
    _capture(tmp_path / "native", "0-0", "task")
    extra = {"role": "user", "content": "Instructions from: /workspace/AGENTS.md\nThe answer is 42."}
    _capture(tmp_path / "harbor", "0-0", "task", more_messages=[extra])
    assert _first_request(tmp_path) == 1
    out = capsys.readouterr().out
    assert "roles differ" in out and "flag Instructions from: native 0, harbor 1" in out


def test_first_request_without_an_agent_request_fails(tmp_path, capsys):
    for side in ("native", "harbor"):
        title = {"model": "m", "messages": [{"role": "user", "content": "Generate a title"}]}
        _write_jsonl(tmp_path / side / "0-0.capture.jsonl", [{"started_at": 1.0, "request": title}])
    assert _first_request(tmp_path, "--json", str(tmp_path / "r.json")) == 1
    lines = capsys.readouterr().out.splitlines()
    assert "no agent request: native: 0-0.capture.jsonl" in lines and "first-request: no pair to compare" in lines
    assert lines[-1] == "first-request: 0 paired, 0 unpaired, 0 differing"
    assert json.loads((tmp_path / "r.json").read_text())["no_agent_request"]["harbor"] == ["0-0.capture.jsonl"]


@pytest.mark.parametrize(
    "later_harbor, problem",
    [({}, None), ({"model": "other"}, "model differs"), ({"temperature": 1.0}, "sampling differs")],
)
def test_first_request_compares_every_repeat_of_a_prompt(tmp_path, capsys, later_harbor, problem):
    # Two rollouts of one task per side, each with OpenCode's title call; the native prompt is argv-quoted.
    for name in ("0-0", "0-1"):
        _capture(tmp_path / "native", name, '"Say \\"hi\\""')
    _capture(tmp_path / "harbor", "0-0", 'Say "hi"')
    _capture(tmp_path / "harbor", "0-1", 'Say "hi"', **later_harbor)
    assert _first_request(tmp_path, "--out", str(tmp_path / "out")) == (1 if problem else 0)
    out = capsys.readouterr().out
    assert out.count("argv_prompt_quoting") == 2
    assert out.splitlines()[-1] == f"first-request: 2 paired, 0 unpaired, {1 if problem else 0} differing"
    assert "native 0-1.capture.jsonl / harbor 0-1.capture.jsonl: " + ("DIFFERENT" if problem else "same") in out
    if problem:
        assert problem in out
    key = hashlib.sha256(b'Say "hi"').hexdigest()[:12]
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == [key, f"{key}.2"]


def test_first_request_an_extra_repeat_on_one_side_is_unpaired(tmp_path, capsys):
    for name in ("0-0", "0-1"):
        _capture(tmp_path / "native", name, "task")
    _capture(tmp_path / "harbor", "0-0", "task")
    assert _first_request(tmp_path) == 1
    lines = capsys.readouterr().out.splitlines()
    assert "unpaired: only on native: 0-1.capture.jsonl" in lines
    assert lines[-1] == "first-request: 1 paired, 1 unpaired, 0 differing"


def test_first_request_without_captures_is_an_input_error(tmp_path, capsys):
    (tmp_path / "native").mkdir()
    (tmp_path / "harbor").mkdir()
    assert _first_request(tmp_path) == 2


# subset


def _rows(tmp_path, ids):
    return _write_jsonl(tmp_path / "rows.jsonl", [{"meta": {"id": i}} for i in ids])


def _subset(tmp_path, rows, *extra, count="3", seed="7"):
    out = tmp_path / "ids.txt"
    code = main(
        ["subset", str(rows), "--id-field", "meta.id", "--count", count, "--seed", seed, "--out", str(out), *extra]
    )
    return code, out


def _ids(path):
    return [line for line in path.read_text().splitlines() if not line.startswith("#")]


def test_subset_is_seeded_and_ordered_by_hash(tmp_path, capsys):
    rows = _rows(tmp_path, [f"t{i}" for i in range(10)])
    code, out = _subset(tmp_path, rows)
    assert code == 0
    expected = sorted((f"t{i}" for i in range(10)), key=lambda i: hashlib.sha256(f"7:{i}".encode()).hexdigest())[:3]
    assert _ids(out) == expected
    sha = hashlib.sha256(rows.read_bytes()).hexdigest()
    assert out.read_text().splitlines()[:2] == [
        f"# rows rows.jsonl sha256 {sha} lines 10",
        "# seed 7 count 3 include none exclude none",
    ]
    assert _subset(tmp_path, rows, seed="8")[0] == 0 and _ids(out) != expected
    assert capsys.readouterr().out.splitlines()[0] == f"subset: wrote 3 ids to {out}"


def test_subset_include_and_exclude(tmp_path):
    rows = _rows(tmp_path, [f"t{i}" for i in range(10)])
    (tmp_path / "inc.txt").write_text("# supported\nt1\nt2\nt3\nt4\n")
    (tmp_path / "exc.txt").write_text("t2\n")
    code, out = _subset(tmp_path, rows, "--include", str(tmp_path / "inc.txt"), "--exclude", str(tmp_path / "exc.txt"))
    assert code == 0 and sorted(_ids(out)) == ["t1", "t3", "t4"]
    assert out.read_text().splitlines()[1] == "# seed 7 count 3 include inc.txt exclude exc.txt"


def test_subset_reports_listed_ids_that_are_not_rows(tmp_path, capsys):
    rows = _rows(tmp_path, [f"t{i}" for i in range(10)])
    (tmp_path / "exc.txt").write_text("t2\nT3\nt99\n")
    (tmp_path / "inc.txt").write_text("".join(f"t{i}\n" for i in range(0, 1000, 3)))
    code, out = _subset(tmp_path, rows, "--exclude", str(tmp_path / "exc.txt"), "--include", str(tmp_path / "inc.txt"))
    assert code == 0 and "t2" not in _ids(out)
    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == [
        "subset: 4 of 334 --include id(s) are rows of rows.jsonl",
        "subset: 2 --exclude id(s) are not in rows.jsonl: T3, t99",
    ]


def test_subset_too_few_fails_without_a_file(tmp_path):
    rows = _rows(tmp_path, ["a", "b"])
    code, out = _subset(tmp_path, rows)
    assert code == 1 and not out.exists()


def test_subset_non_utf8_row_is_an_input_error(tmp_path, capsys):
    rows = tmp_path / "rows.jsonl"
    rows.write_bytes(b'{"meta": {"id": "a"}}\n{"meta": {"id": "\xff"}}\n')
    assert _subset(tmp_path, rows)[0] == 2
    assert "line 2" in capsys.readouterr().err


def test_subset_index_ids_and_duplicates(tmp_path):
    rows = _rows(tmp_path, ["same", "same", "x"])
    assert _subset(tmp_path, rows)[0] == 2
    out = tmp_path / "ids.txt"
    assert main(["subset", str(rows), "--id-field", "_index", "--count", "3", "--seed", "1", "--out", str(out)]) == 0
    assert sorted(_ids(out)) == ["0", "1", "2"]


# check


SYSTEM = "You are a careful assistant."
QUESTION = 'What is 6 * 7? Say "done" after.'
ANSWER = "forty-two exactly"
PATCH = (
    "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -1 +1 @@\n-x = 1\n+result = compute_the_secret_value(42)\n"
)


def _dataset(root, *, instruction=None, environment=None):
    """Two task folders converted from two native rows (task id = native `name`)."""
    rows = []
    for name in ("t1", "t2"):
        task = root / "ds" / name
        (task / "environment").mkdir(parents=True)
        (task / "tests").mkdir()
        (task / "task.toml").write_text('version = "1.0"\n')
        (task / "instruction.md").write_text(instruction or f"<!-- canary GUID abc -->\n\n{SYSTEM}\n\n{QUESTION}\n")
        (task / "environment" / "Dockerfile").write_text(environment or "FROM python:3.12\n")
        (task / "tests" / "expected.txt").write_text(ANSWER)
        rows.append(
            {
                "name": name,
                "responses_create_params": {
                    "input": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION}]
                },
                "expected": ANSWER,
                "patch": PATCH,
                "short": "42",
            }
        )
    return root / "ds", _write_jsonl(root / "rows.jsonl", rows)


def _check(dataset, rows, *extra):
    return main(["check", str(dataset), "--native-rows", str(rows), "--native-id", "name", *extra])


def _statuses(capsys):
    lines = capsys.readouterr().out.splitlines()
    return {
        line.split(":")[0].split(" ", 1)[1]: line.split(" ", 1)[0]
        for line in lines
        if line[:4] in ("PASS", "FAIL", "SKIP")
    }


def test_clean_dataset_passes_every_check(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path)
    assert _check(dataset, rows, "--hidden", "expected", "--hidden", "patch") == 0
    assert _statuses(capsys) == {
        "instruction": "PASS",
        "hidden-values": "PASS",
        "hidden-template": "SKIP",
        "secrets": "PASS",
        "local-paths": "PASS",
    }


def test_instruction_one_character_off_fails_with_offset(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path, instruction=f"{SYSTEM}\n\n{QUESTION.replace('7', '8')}")
    assert _check(dataset, rows) == 1
    out = capsys.readouterr().out
    assert "FAIL instruction" in out and "differs at offset" in out


def test_instruction_prefix_file_is_put_before_the_native_prompt(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path, instruction=f"Agent rules.\n\n{SYSTEM}\n\n{QUESTION}")
    prefix = tmp_path / "prefix.txt"
    prefix.write_text("Agent rules.\n\n")
    assert _check(dataset, rows) == 1
    capsys.readouterr()
    assert _check(dataset, rows, "--prefix-file", str(prefix)) == 0


def test_instruction_with_crlf_fails_until_newlines_are_normalized(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path, instruction=f"{SYSTEM}\n\nLine one\nline two\n")
    rows_data = [json.loads(line) for line in rows.read_text().splitlines()]
    rows_data[0]["responses_create_params"]["input"][1]["content"] = "Line one\r\nline two\r\n"
    rows_data[1]["responses_create_params"]["input"][1]["content"] = "Line one\nline two"
    _write_jsonl(rows, rows_data)
    assert _check(dataset, rows) == 1
    out = capsys.readouterr().out
    assert "t1: differs only in line endings" in out and "t2" not in out
    assert _check(dataset, rows, "--normalize-newlines") == 0
    assert "2 task(s) equal their native prompt; 1 only with CRLF/CR read as LF" in capsys.readouterr().out
    rows_data[0]["responses_create_params"]["input"][1]["content"] = "Line one\r\nline 2\r\n"
    _write_jsonl(rows, rows_data)
    assert _check(dataset, rows, "--normalize-newlines") == 1
    assert "t1: differs at offset" in capsys.readouterr().out


def test_instruction_hash_canary_is_dropped_but_a_leading_blank_line_without_one_is_kept(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path, instruction=f"# canary GUID abc\n\n{SYSTEM}\n\n{QUESTION}\n")
    assert _check(dataset, rows) == 0
    capsys.readouterr()
    # Gym keeps the leading newline when there is no canary line, so the model sees it.
    for name in ("t1", "t2"):
        (dataset / name / "instruction.md").write_text(f"\n{SYSTEM}\n\n{QUESTION}\n")
    assert _check(dataset, rows) == 1
    assert "t1: differs at offset 0" in capsys.readouterr().out


def test_two_native_rows_naming_one_task_are_refused(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path)
    first = json.loads(rows.read_text().splitlines()[0])
    _write_jsonl(rows, [first, first])
    assert _check(dataset, rows) == 2
    assert "two rows name task t1" in capsys.readouterr().err


def test_instruction_without_a_native_row_fails(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path)
    _write_jsonl(rows, [json.loads(rows.read_text().splitlines()[0])])
    assert _check(dataset, rows) == 1
    assert "t2: no native row" in capsys.readouterr().out


def test_id_map_and_index_name_the_native_row(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path)
    id_map = tmp_path / "map.json"
    id_map.write_text(json.dumps({"0": "t1", "1": "t2"}))
    assert (
        main(["check", str(dataset), "--native-rows", str(rows), "--native-id", "_index", "--id-map", str(id_map)])
        == 0
    )


def test_hidden_value_in_environment_fails(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path, environment=f"FROM python:3.12\nRUN echo '{ANSWER}' > /tmp/a\n")
    assert _check(dataset, rows, "--hidden", "expected") == 1
    out = capsys.readouterr().out
    assert "FAIL hidden-values" in out and "environment/Dockerfile contains expected value" in out


def test_hidden_value_in_environment_env_or_a_step_instruction_fails(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path)
    (dataset / "t1" / "task.toml").write_text(f'version = "1.0"\n[environment.env]\nEXPECTED = "{ANSWER}"\n')
    assert _check(dataset, rows, "--hidden", "expected") == 1
    assert f"t1/task.toml [environment].env contains expected value {ANSWER!r}" in capsys.readouterr().out
    (dataset / "t1" / "task.toml").write_text('version = "1.0"\n')
    (dataset / "t2" / "steps" / "one").mkdir(parents=True)
    (dataset / "t2" / "steps" / "one" / "instruction.md").write_text(f"Now say {ANSWER}.\n")
    assert _check(dataset, rows, "--hidden", "expected") == 1
    assert "t2/steps/one/instruction.md contains expected value" in capsys.readouterr().out


def _multi_step(dataset):
    """Make t2 a valid two-step task: each step with its instruction, workdir/, tests/ and solution/."""
    task = dataset / "t2"
    (task / "task.toml").write_text('version = "1.0"\n[[steps]]\nname = "one"\n[[steps]]\nname = "two"\n')
    (task / "tests" / "test.sh").write_text("exit 0\n")
    for step in ("one", "two"):
        for part in ("workdir", "tests", "solution"):
            (task / "steps" / step / part).mkdir(parents=True)
        (task / "steps" / step / "instruction.md").write_text(f"Step {step}.\n")
        (task / "steps" / step / "workdir" / "notes.txt").write_text("Some notes.\n")
    return task


def test_hidden_value_in_a_step_workdir_fails_and_step_verifier_files_are_excluded(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path)
    task = _multi_step(dataset)
    # Clean: a hidden value in the steps' (and the task's) tests/ and solution/ stays with the verifier and oracle.
    for folder in (task / "steps" / "one" / "tests", task / "steps" / "two" / "solution", task / "solution"):
        folder.mkdir(exist_ok=True)
        (folder / "answer.txt").write_text(ANSWER)
    assert _check(dataset, rows, "--hidden", "expected") == 0
    assert _statuses(capsys)["hidden-values"] == "PASS"

    plain = task / "steps" / "two" / "workdir" / "answer.txt"
    plain.write_text(ANSWER)
    assert _check(dataset, rows, "--hidden", "expected") == 1
    assert f"t2/steps/two/workdir/answer.txt contains expected value {ANSWER!r}" in capsys.readouterr().out
    plain.write_text("nothing here\n")

    filler = "\n".join(hashlib.sha256(str(i).encode()).hexdigest() for i in range(60))
    member = f"{ANSWER}\n{filler}\n".encode()
    bundle = io.BytesIO()
    with tarfile.open(fileobj=bundle, mode="w:gz") as tar:
        info = tarfile.TarInfo("data/answer.txt")
        info.size = len(member)
        tar.addfile(info, io.BytesIO(member))
    (task / "steps" / "one" / "workdir" / "setup.sh").write_text(
        f"base64 -d > seed.tgz <<'EOF'\n{base64.encodebytes(bundle.getvalue()).decode()}EOF\n"
    )
    assert _check(dataset, rows, "--hidden", "expected") == 1
    assert "t2/steps/one/workdir/setup.sh[base64 at " in capsys.readouterr().out


def test_hidden_value_in_a_dataset_env_override_fails_as_in_task_toml(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path)
    (dataset / "dataset.toml").write_text(f'[gym.tasks.t1.environment.env]\nEXPECTED = "{ANSWER}"\n')
    assert _check(dataset, rows, "--hidden", "expected") == 1
    assert f'dataset.toml [gym.tasks."t1".environment].env contains expected value {ANSWER!r}' in (
        capsys.readouterr().out
    )
    # A benign override passes, and one that replaces (or empties) the task's own variable leaves nothing to find.
    (dataset / "t1" / "task.toml").write_text(f'version = "1.0"\n[environment.env]\nEXPECTED = "{ANSWER}"\n')
    for value in ("benign value", ""):
        (dataset / "dataset.toml").write_text(
            f'[gym.tasks.t1.environment]\ndocker_image = "python:3.12"\n[gym.tasks.t1.environment.env]\n'
            f'EXPECTED = "{value}"\nOTHER = "fine"\n'
        )
        assert _check(dataset, rows, "--hidden", "expected") == 0
        assert _statuses(capsys)["hidden-values"] == "PASS"
    # An override of another variable merges over the task's own, which still reaches the agent.
    (dataset / "dataset.toml").write_text('[gym.tasks.t1.environment.env]\nOTHER = "fine"\n')
    assert _check(dataset, rows, "--hidden", "expected") == 1
    assert "t1/task.toml [environment].env contains expected value" in capsys.readouterr().out


def _encoded(text, *, gzipped=False):
    """`text` after 60 lines of filler, gzipped or not, as one base64 line long enough to be decoded."""
    data = ("\n".join(hashlib.sha256(str(i).encode()).hexdigest() for i in range(60)) + f"\n{text}\n").encode()
    return base64.b64encode(gzip.compress(data, mtime=0) if gzipped else data).decode()


ENV_DECLARATIONS = {
    "task": (
        "t1/task.toml [environment].env",
        lambda body: ("t1/task.toml", f'version = "1.0"\n[environment.env]\n{body}'),
    ),
    "dataset": (
        'dataset.toml [gym.tasks."t1".environment].env',
        lambda body: ("dataset.toml", f"[gym.tasks.t1.environment.env]\n{body}"),
    ),
}


@pytest.mark.parametrize("declaration", ["task", "dataset"])
@pytest.mark.parametrize(
    "body, where",
    [
        (f'EXPECTED = "{_encoded(ANSWER)}"', "[base64 at 0]"),
        (f'EXPECTED = "{_encoded(ANSWER, gzipped=True)}"', "[base64 at 0]"),
        (f'"{ANSWER}" = "1"', " name"),
        (f'"{_encoded(ANSWER)}" = "1"', " name[base64 at 0]"),
    ],
)
def test_encoded_hidden_values_and_names_in_the_effective_env_fail(tmp_path, capsys, declaration, body, where):
    print(check.__file__)
    dataset, rows = _dataset(tmp_path)
    label, write = ENV_DECLARATIONS[declaration]
    path, text = write(body)
    (dataset / path).write_text(text)
    assert _check(dataset, rows, "--hidden", "expected") == 1
    assert f"{label}{where} contains expected value {ANSWER!r}" in capsys.readouterr().out


def test_encoded_hidden_value_in_env_replaced_or_cleared_by_the_dataset_passes(tmp_path, capsys):
    print(check.__file__)
    dataset, rows = _dataset(tmp_path)
    (dataset / "t1" / "task.toml").write_text(f'version = "1.0"\n[environment.env]\nEXPECTED = "{_encoded(ANSWER)}"\n')
    assert _check(dataset, rows, "--hidden", "expected") == 1
    capsys.readouterr()
    for value in (_encoded("nothing hidden here"), ""):
        (dataset / "dataset.toml").write_text(f'[gym.tasks.t1.environment.env]\nEXPECTED = "{value}"\nOTHER = "x"\n')
        assert _check(dataset, rows, "--hidden", "expected") == 0
        assert _statuses(capsys)["hidden-values"] == "PASS"


@pytest.mark.parametrize("declaration", ["task", "dataset"])
@pytest.mark.parametrize("bound", ["zip", "nesting", "budget"])
def test_an_env_value_past_the_scan_bounds_is_not_scanned(tmp_path, capsys, monkeypatch, declaration, bound):
    print(check.__file__)
    dataset, rows = _dataset(tmp_path)
    label, write = ENV_DECLARATIONS[declaration]
    value = _encoded("nothing hidden here")
    if bound == "zip":
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("answer.txt", "x" * 1024)
        value = base64.b64encode(archive.getvalue()).decode()
    elif bound == "nesting":
        for _ in range(scan.MAX_NESTING):
            value = base64.b64encode(value.encode()).decode()
    else:
        monkeypatch.setattr(scan, "MAX_DECODED", 100)
    path, text = write(f'SEED = "{value}"')
    (dataset / path).write_text(text)
    assert _check(dataset, rows, "--hidden", "expected") == 1
    out = capsys.readouterr().out
    assert "FAIL hidden-values" in out and f"{label}[base64 at 0]" in out and "not scanned" in out


MCP = '[[environment.mcp_servers]]\nname = "tools"\ntransport = "stdio"\ncommand = "serve"\n'
SSE = 'name = "tools"\ntransport = "sse"\nurl = "http://h/?q=' + ANSWER + '"\n'


@pytest.mark.parametrize(
    "task_toml, dataset_toml, label",
    [
        (MCP + f'args = ["{ANSWER}"]\n', "", "t1/task.toml [environment].mcp_servers"),
        ("[[environment.mcp_servers]]\n" + SSE, "", "t1/task.toml [environment].mcp_servers"),
        (
            "",
            "[[gym.tasks.t1.environment.mcp_servers]]\n" + SSE,
            'dataset.toml [gym.tasks."t1".environment].mcp_servers',
        ),
        (f'[environment]\nskills_dir = "/skills/{ANSWER}"\n', "", "t1/task.toml [environment].skills_dir"),
        (f'[environment.healthcheck]\ncommand = "echo {ANSWER}"\n', "", "t1/task.toml [environment].healthcheck"),
    ],
)
def test_hidden_value_in_agent_visible_environment_settings_fails(tmp_path, capsys, task_toml, dataset_toml, label):
    print(check.__file__)
    dataset, rows = _dataset(tmp_path)
    (dataset / "t1" / "task.toml").write_text('version = "1.0"\n' + task_toml)
    (dataset / "dataset.toml").write_text(dataset_toml)
    assert _check(dataset, rows, "--hidden", "expected") == 1
    assert f"{label} contains expected value {ANSWER!r}" in capsys.readouterr().out


def test_only_the_image_and_workdir_settings_exempt_a_hidden_value(tmp_path, capsys):
    print(check.__file__)
    dataset, rows = _dataset(tmp_path)
    (dataset / "t1" / "environment" / "answer.txt").write_text(ANSWER)
    (dataset / "t1" / "task.toml").write_text(f'version = "1.0"\n{MCP}args = ["{ANSWER}"]\n')
    assert _check(dataset, rows, "--hidden", "expected") == 1
    assert f"t1/environment/answer.txt contains expected value {ANSWER!r}" in capsys.readouterr().out
    # A hidden value equal to the image or the working directory is visible by design and not scanned.
    for setting in (f'docker_image = "{ANSWER}"', f'workdir = "{ANSWER}"'):
        for name in ("t1", "t2"):
            (dataset / name / "task.toml").write_text(f'version = "1.0"\n[environment]\n{setting}\n')
        assert _check(dataset, rows, "--hidden", "expected") == 0
        assert _statuses(capsys)["hidden-values"] == "PASS"


@pytest.mark.parametrize(
    "dataset_toml",
    [
        "[gym.tasks.t1.environment]\nenv = 3\n",
        "[gym.tasks.t1.environment.env]\nEXPECTED = 3\n",
        "[gym.tasks.t1]\nenvironment = 3\n",
        "[gym]\ntasks = 3\n",
        "[gym.tasks\n",
    ],
)
def test_a_dataset_override_of_a_shape_not_modeled_is_not_scanned(tmp_path, capsys, dataset_toml):
    dataset, rows = _dataset(tmp_path)
    (dataset / "dataset.toml").write_text(dataset_toml)
    assert _check(dataset, rows, "--hidden", "expected") == 1
    assert "not scanned" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("task_toml", "dataset_toml"),
    [
        ('[environment.env]\nA = "task"\nB = "kept"\n', '[gym.tasks.t.environment.env]\nA = "override"\n'),
        ('[environment.env]\nA = "task"\n', '[gym.tasks.t.environment.env]\nA = ""\n'),
        ("", '[gym.tasks.t.environment.env]\nA = "added"\n'),
        ('[environment.env]\nA = "task"\n', "[gym.tasks.t.environment]\ncpus = 2\n"),
        ('[environment.env]\nA = "task"\n', '[gym.tasks.other.environment.env]\nA = "elsewhere"\n'),
        ('[environment.env]\nA = "task"\n', "[gym.defaults]\nresource_multiplier = 2.0\n"),
    ],
)
def test_effective_environment_env_matches_gyms_dataset_merge(tmp_path, task_toml, dataset_toml):
    print(check.__file__)
    (tmp_path / "dataset.toml").write_text(dataset_toml)
    task = tomllib.loads('version = "1.0"\n' + task_toml)
    runtime = apply_dataset_config(HarborTaskConfig.model_validate(task), "t", read_dataset_config(tmp_path))
    overrides, problem = check._task_overrides(tmp_path)
    environment, _, applied = check.effective_environment(task.get("environment", {}), overrides.get("t"))
    assert problem is None and applied
    assert (environment.get("env") or {}) == runtime.environment.env


def test_json_escaped_hidden_values_in_environment_fail(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path)
    rows_data = [json.loads(line) for line in rows.read_text().splitlines()]
    values = {"t1": json.dumps({"recipient": "a@b.com"}), "t2": 'say "forty two" now'}
    for row in rows_data:
        row["arguments"] = values[row["name"]]
    _write_jsonl(rows, rows_data)
    for name, value in values.items():
        (dataset / name / "environment" / "task.json").write_text(json.dumps({"expected": value}))
        assert value.encode() not in (dataset / name / "environment" / "task.json").read_bytes()
    assert _check(dataset, rows, "--hidden", "arguments") == 1
    out = capsys.readouterr().out
    assert "t1/environment/task.json contains arguments value" in out
    assert "t2/environment/task.json contains arguments value" in out


def test_plain_gzip_and_tar_files_in_environment_are_unpacked(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path)
    compressed = gzip.compress(ANSWER.encode(), mtime=0)
    assert ANSWER.encode() not in compressed
    (dataset / "t1" / "environment" / "answer.gz").write_bytes(compressed)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo("data/answer.txt")
        info.size = len(ANSWER)
        tar.addfile(info, io.BytesIO(ANSWER.encode()))
    (dataset / "t2" / "environment" / "data.tar.gz").write_bytes(buffer.getvalue())
    assert _check(dataset, rows, "--hidden", "expected") == 1
    out = capsys.readouterr().out
    assert "t1/environment/answer.gz contains expected value" in out
    assert "t2/environment/data.tar.gz:data/answer.txt contains expected value" in out
    (dataset / "t1" / "environment" / "answer.gz").unlink()
    (dataset / "t2" / "environment" / "data.tar.gz").unlink()
    with zipfile.ZipFile(dataset / "t1" / "environment" / "data.zip", "w") as archive:
        archive.writestr("answer.txt", "nothing hidden")
    assert _check(dataset, rows, "--hidden", "expected") == 1
    assert "t1/environment/data.zip: a zip archive: not scanned" in capsys.readouterr().out


def test_patch_added_line_in_a_base64_tar_gz_fails(tmp_path, capsys):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        padding = "".join(f"# {hashlib.sha256(str(i).encode()).hexdigest()}\n" for i in range(40))
        data = (padding + "result = compute_the_secret_value(42)\n").encode()
        info = tarfile.TarInfo("bundle/m.py")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    blob = base64.b64encode(gzip.compress(buffer.getvalue(), mtime=0)).decode()
    assert len(blob) >= 1024
    dataset, rows = _dataset(tmp_path, environment=f"FROM python:3.12\nRUN echo '{blob}' | base64 -d | tar -xzf -\n")
    assert _check(dataset, rows, "--hidden", "patch") == 1
    assert ":bundle/m.py contains patch value 'result = compute_the_secret_value(42)'" in capsys.readouterr().out


def test_removed_patch_lines_and_values_in_the_prompt_are_not_hidden(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path, environment="FROM python:3.12\nRUN echo 'x = 1' 'What is 6 * 7?'\n")
    rows_data = [json.loads(line) for line in rows.read_text().splitlines()]
    for row in rows_data:
        row["question_copy"] = "What is 6 * 7?"
    _write_jsonl(rows, rows_data)
    assert _check(dataset, rows, "--hidden", "patch", "--hidden", "question_copy") == 0


def test_short_values_are_counted_not_scanned(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path, environment="FROM python:3.12\nRUN echo 42\n")
    assert _check(dataset, rows, "--hidden", "short") == 0
    assert "2 short value(s) not scanned: use --template-hidden" in capsys.readouterr().out


def test_hidden_from_hides_fields_consumed_by_neither_prompt_nor_provenance(tmp_path, capsys):
    environment = f"FROM python:3.12\nRUN echo '{ANSWER}' 'multiplication tables' 'repository-name'\n"
    dataset, rows = _dataset(tmp_path, environment=environment)
    rows_data = [json.loads(line) for line in rows.read_text().splitlines()]
    for row in rows_data:
        row["topic"], row["repo"] = "multiplication tables", "repository-name"
    _write_jsonl(rows, rows_data)
    task_data = tmp_path / "task_data.py"
    task_data.write_text(
        "from pydantic import BaseModel, Field\n\n\nclass TaskData(BaseModel):\n"
        '    topic: str = Field(json_schema_extra={"consumed_by": ["prompt", "metrics"]})\n'
        '    repo: str = Field(json_schema_extra={"consumed_by": ["provenance"]})\n'
        '    expected: str = Field(json_schema_extra={"consumed_by": ["verify"]})\n'
    )
    assert _check(dataset, rows, "--hidden-from", str(task_data)) == 1
    out = capsys.readouterr().out
    assert "contains expected value" in out and "topic value" not in out and "repo value" not in out


def test_environment_settings_and_offered_tools_are_not_hidden(tmp_path, capsys):
    image, workdir = "registry.example/swe:task-one", "/workspace/project"
    environment = f"FROM --platform=linux/amd64 {image}\nRUN cd {workdir} && echo email_send_email\n"
    dataset, rows = _dataset(tmp_path, environment=environment)
    for name in ("t1", "t2"):
        (dataset / name / "task.toml").write_text(f'version = "1.0"\n[environment]\nworkdir = "{workdir}"\n')
    rows_data = [json.loads(line) for line in rows.read_text().splitlines()]
    for row in rows_data:
        row.update(image_url=image, workdir=workdir, ground_truth=[{"name": "email_send_email", "arguments": "{}"}])
        row["responses_create_params"]["tools"] = [{"type": "function", "name": "email_send_email"}]
    _write_jsonl(rows, rows_data)
    hidden = ["--hidden", "image_url", "--hidden", "workdir", "--hidden", "ground_truth"]
    assert _check(dataset, rows, *hidden) == 0
    del rows_data[0]["responses_create_params"]["tools"]
    _write_jsonl(rows, rows_data)
    assert _check(dataset, rows, *hidden) == 1
    assert "t1/environment/Dockerfile contains ground_truth value 'email_send_email'" in capsys.readouterr().out


def test_a_digest_pinned_from_image_names_the_hidden_image(tmp_path, capsys):
    image = "registry.example/swe:task-one"
    dataset, rows = _dataset(tmp_path, environment=f"FROM {image}@sha256:{'0' * 64}\nRUN true\n")
    rows_data = [json.loads(line) for line in rows.read_text().splitlines()]
    for row in rows_data:
        row["image_name"] = image
    _write_jsonl(rows, rows_data)
    assert _check(dataset, rows, "--hidden", "image_name") == 0
    (dataset / "t2" / "environment" / "Dockerfile").write_text(f"FROM python:3.12\nRUN echo {image}\n")
    assert _check(dataset, rows, "--hidden", "image_name") == 1


def test_allow_values_lists_reviewed_matches_without_failing(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path, environment="FROM python:3.12\nRUN for x in a; do continue; done\n")
    rows_data = [json.loads(line) for line in rows.read_text().splitlines()]
    for row in rows_data:
        row["PASS_TO_PASS"] = json.dumps(["continue", "test_secret_behaviour"])
    _write_jsonl(rows, rows_data)
    assert _check(dataset, rows, "--hidden", "PASS_TO_PASS") == 1
    capsys.readouterr()
    allow = tmp_path / "allow.txt"
    allow.write_text("# keyword in the seed script\ncontinue\n")
    assert _check(dataset, rows, "--hidden", "PASS_TO_PASS", "--allow-values", str(allow)) == 0
    out = capsys.readouterr().out
    assert "2 match(es) of --allow-values values" in out
    assert "  allowed: t1/environment/Dockerfile contains PASS_TO_PASS value 'continue'" in out
    (dataset / "t1" / "environment" / "notes.txt").write_text("test_secret_behaviour\n")
    assert _check(dataset, rows, "--hidden", "PASS_TO_PASS", "--allow-values", str(allow)) == 1
    capsys.readouterr()
    # An allowed value still fails in instruction.md: the allow list covers files under environment/ only.
    (dataset / "t1" / "environment" / "notes.txt").unlink()
    (dataset / "t1" / "instruction.md").write_text(f"{SYSTEM}\n\n{QUESTION}\ncontinue\n")
    assert _check(dataset, rows, "--hidden", "PASS_TO_PASS", "--allow-values", str(allow)) == 1
    out = capsys.readouterr().out
    assert "\n  t1/instruction.md contains PASS_TO_PASS value 'continue'" in out


def test_lines_of_a_hidden_script_count_from_40_characters(tmp_path, capsys):
    script = 'import pytest\n\nassert compute_the_hidden_answer(6, 7) == 42  # f2p\n\nif __name__ == "__main__":\n'
    dataset, rows = _dataset(tmp_path, environment="FROM python:3.12\nRUN echo 'if __name__ == \"__main__\":'\n")
    rows_data = [json.loads(line) for line in rows.read_text().splitlines()]
    for row in rows_data:
        row["f2p_script"] = script
    _write_jsonl(rows, rows_data)
    assert _check(dataset, rows, "--hidden", "f2p_script") == 0
    capsys.readouterr()
    (dataset / "t2" / "environment" / "t.py").write_text("assert compute_the_hidden_answer(6, 7) == 42  # f2p\n")
    assert _check(dataset, rows, "--hidden", "f2p_script") == 1
    assert "t2/environment/t.py contains f2p_script value 'assert compute_the_hidden_answer" in capsys.readouterr().out


def test_each_item_of_a_json_encoded_list_is_hidden(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path)
    rows_data = [json.loads(line) for line in rows.read_text().splitlines()]
    for row in rows_data:
        row["FAIL_TO_PASS"] = json.dumps(["tests/test_m.py::test_secret_case", "tests/test_m.py::test_other"])
        row["arguments"] = json.dumps({"recipient": "someone@example.com"})
    _write_jsonl(rows, rows_data)
    (dataset / "t1" / "environment" / "setup.sh").write_text("pytest tests/test_m.py::test_secret_case\n")
    (dataset / "t2" / "environment" / "data.csv").write_text("someone@example.com\n")
    assert _check(dataset, rows, "--hidden", "FAIL_TO_PASS", "--hidden", "arguments") == 1
    out = capsys.readouterr().out
    assert "t1/environment/setup.sh contains FAIL_TO_PASS value 'tests/test_m.py::test_secret_case'" in out
    assert "data.csv" not in out


def test_absent_fields_are_named_once(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path)
    rows_data = [json.loads(line) for line in rows.read_text().splitlines()]
    rows_data[1]["sometimes"] = "a value only one row has"
    _write_jsonl(rows, rows_data)
    assert _check(dataset, rows, "--hidden", "expected", "--hidden", "sometimes") == 0
    assert "; absent from some rows: sometimes" in capsys.readouterr().out
    # An explicit --hidden field no row has is a typo or a wrong path.
    assert _check(dataset, rows, "--hidden", "expected", "--hidden", "missing") == 2
    assert "--hidden missing: no native row has field missing" in capsys.readouterr().err
    # An optional TaskData field may be absent from every row of a subset: noted, not refused.
    task_data = tmp_path / "task_data.py"
    task_data.write_text(
        "from pydantic import BaseModel, Field\n\n\nclass TaskData(BaseModel):\n"
        '    optional: str = Field(json_schema_extra={"consumed_by": ["verify"]})\n'
    )
    assert _check(dataset, rows, "--hidden", "expected", "--hidden-from", str(task_data)) == 0
    assert "; absent from every row: optional" in capsys.readouterr().out


def _schemas(root, child_base, child_import):
    """A parent TaskData in a package and a child task_data.py that inherits it, as Resources Servers do."""
    parent = root / "servers" / "parent_server" / "task_data.py"
    parent.parent.mkdir(parents=True)
    parent.write_text(
        "from pydantic import BaseModel, Field\n\n\nclass TaskData(BaseModel):\n"
        '    topic: str = Field(json_schema_extra={"consumed_by": ["prompt"]})\n'
        '    expected: str = Field(json_schema_extra={"consumed_by": ["verify"]})\n'
    )
    child = root / "servers" / "child_server" / "task_data.py"
    child.parent.mkdir(parents=True)
    child.write_text(
        f"from pydantic import Field\n\n{child_import}\n\n\nclass TaskData({child_base}):\n"
        '    repo: str = Field(json_schema_extra={"consumed_by": ["provenance"]})\n'
    )
    return child


@pytest.mark.parametrize(
    "child_base, child_import",
    [
        ("ParentTaskData", "from servers.parent_server.task_data import TaskData as ParentTaskData"),
        ("ParentTaskData", "from ..parent_server.task_data import TaskData as ParentTaskData"),
    ],
)
def test_hidden_from_includes_inherited_fields(tmp_path, capsys, child_base, child_import):
    environment = "FROM python:3.12\nRUN echo 'multiplication tables' 'repository-name'\n"
    dataset, rows = _dataset(tmp_path, environment=environment)
    rows_data = [json.loads(line) for line in rows.read_text().splitlines()]
    for row in rows_data:
        row["topic"], row["repo"] = "multiplication tables", "repository-name"
    _write_jsonl(rows, rows_data)
    child = _schemas(tmp_path, child_base, child_import)
    # A clean task passes, and the inherited visible field (topic) is not hidden.
    assert _check(dataset, rows, "--hidden-from", str(child)) == 0
    assert "PASS hidden-values: none of 2 hidden value(s) of 2 row(s)" in capsys.readouterr().out
    # The inherited hidden field (expected) is scanned.
    (dataset / "t1" / "environment" / "answer.txt").write_text(ANSWER)
    assert _check(dataset, rows, "--hidden-from", str(child)) == 1
    out = capsys.readouterr().out
    assert "t1/environment/answer.txt contains expected value 'forty-two exactly'" in out
    assert "topic value" not in out and "repo value" not in out


def test_hidden_from_the_math_with_autograder_schema_hides_its_inherited_answer(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path)
    rows_data = [json.loads(line) for line in rows.read_text().splitlines()]
    for row in rows_data:
        row["question"], row["expected_answer"], row["problem_id"] = QUESTION, ANSWER, "imo-bench-algebra-001"
    _write_jsonl(rows, rows_data)
    schema = Path(__file__).resolve().parents[3] / "resources_servers" / "math_with_autograder" / "task_data.py"
    (dataset / "t1" / "environment" / "answer.txt").write_text(ANSWER + " imo-bench-algebra-001")
    assert _check(dataset, rows, "--hidden-from", str(schema)) == 1
    out = capsys.readouterr().out
    assert "t1/environment/answer.txt contains expected_answer value" in out and "problem_id value" not in out


def test_hidden_from_refuses_a_base_it_cannot_follow(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path)
    child = _schemas(tmp_path, "Elsewhere", "from not_in_this_repository.schemas import Elsewhere")
    assert _check(dataset, rows, "--hidden-from", str(child)) == 2
    assert "cannot find the module of Elsewhere" in capsys.readouterr().err
    child.write_text("import other\n\n\nclass TaskData(other.Base):\n    expected: str\n")
    assert _check(dataset, rows, "--hidden-from", str(child)) == 2
    assert "cannot follow base other.Base of TaskData" in capsys.readouterr().err


def _base64_tar(member, text):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        padding = "".join(f"# {hashlib.sha256(str(i).encode()).hexdigest()}\n" for i in range(40))
        data = (padding + text).encode()
        info = tarfile.TarInfo(member)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return base64.b64encode(gzip.compress(buffer.getvalue(), mtime=0)).decode()


def test_base64_wrapped_at_76_columns_is_decoded(tmp_path, capsys):
    blob = _base64_tar("bundle/m.py", "result = compute_the_secret_value(42)\n")
    wrapped = "\n".join(blob[i : i + 76] for i in range(0, len(blob), 76))
    dataset, rows = _dataset(tmp_path)
    (dataset / "t1" / "environment" / "bundle.b64").write_text(wrapped + "\nEOF\n")
    assert _check(dataset, rows, "--hidden", "patch") == 1
    assert "bundle.b64[base64 at 0]:bundle/m.py contains patch value" in capsys.readouterr().out


def test_what_cannot_be_scanned_in_full_fails(tmp_path, capsys, monkeypatch):
    dataset, rows = _dataset(tmp_path)
    (dataset / "t1" / "environment" / "bundle.b64").write_text(_base64_tar("bundle/m.py", "x = 1\n") + "\n")
    assert _check(dataset, rows, "--hidden", "patch") == 0
    capsys.readouterr()
    monkeypatch.setattr(scan, "MAX_DECODED", 256)
    assert _check(dataset, rows, "--hidden", "patch") == 1
    assert "t1/environment/bundle.b64[base64 at 0]: over 256 decoded bytes in all: not scanned" in (
        capsys.readouterr().out
    )
    monkeypatch.undo()
    (dataset / "t2" / "environment" / "bundle.b64").write_text("Q" * 1025 + "\n")
    assert _check(dataset, rows, "--hidden", "patch") == 1
    assert "t2/environment/bundle.b64[base64 at 0]: a 1025-character base64 run does not decode" in (
        capsys.readouterr().out
    )


def _hex_lines(count):
    """Text (no NUL byte, so the secret and path checks read it) that compresses little."""
    return "".join(f"# {hashlib.sha256(str(i).encode()).hexdigest()}\n" for i in range(count)).encode()


def test_decoded_bytes_share_one_bound_across_layers(tmp_path, capsys, monkeypatch):
    dataset, rows = _dataset(tmp_path)
    text = _hex_lines(45)  # 3015 bytes
    (dataset / "t1" / "environment" / "bundle.tar").write_bytes(_tar({"a.txt": text, "b.txt": text}))
    (dataset / "t2" / "environment" / "data.gz").write_bytes(gzip.compress(text * 2))
    assert _check(dataset, rows, "--hidden", "expected") == 0
    capsys.readouterr()
    # Each member fits the bound alone; together they pass it.
    monkeypatch.setattr(scan, "MAX_DECODED", 4096)
    assert _check(dataset, rows, "--hidden", "expected") == 1
    out = capsys.readouterr().out
    assert "t1/environment/bundle.tar:b.txt: over 4096 decoded bytes in all: only the start is scanned" in out
    assert "bundle.tar:a.txt: over" not in out
    assert "t2/environment/data.gz: over 4096 decoded bytes in all: only the start is scanned" in out


def _place(dataset, encoding, data):
    """`data` as t1's environment/data.gz, or base64 in its Dockerfile."""
    if encoding == "raw":
        (dataset / "t1" / "environment" / "data.gz").write_bytes(data)
        return "t1/environment/data.gz"
    blob = base64.b64encode(data).decode()
    assert len(blob) >= 1024
    (dataset / "t1" / "environment" / "Dockerfile").write_text(f"FROM python:3.12\nRUN echo '{blob}' | base64 -d\n")
    return "t1/environment/Dockerfile[base64 at 27]"


@pytest.mark.parametrize("encoding", ["raw", "base64"])
def test_every_member_of_a_gzip_stream_is_scanned(tmp_path, capsys, encoding):
    dataset, rows = _dataset(tmp_path)
    first = gzip.compress(_hex_lines(40), mtime=0)
    label = _place(dataset, encoding, first)
    assert _check(dataset, rows, "--hidden", "expected") == 0
    capsys.readouterr()
    _place(dataset, encoding, first + gzip.compress(b"more harmless text\n", mtime=0))
    assert _check(dataset, rows, "--hidden", "expected") == 0
    capsys.readouterr()
    root = "/" + "scr" + "atch" + "/"
    hidden = f"{ANSWER}\nKEY={SECRET_SAMPLES['aws']}\ncp {root}project/data .\n".encode()
    _place(dataset, encoding, first + gzip.compress(hidden, mtime=0))
    assert _check(dataset, rows, "--hidden", "expected") == 1
    out = capsys.readouterr().out
    assert f"{label} contains expected value" in out
    assert f"{label}: AKIA" in out
    assert f"{label}: {root}" in out


@pytest.mark.parametrize("encoding", ["raw", "base64"])
@pytest.mark.parametrize(
    "wrap", [lambda data: gzip.compress(data, mtime=0), lambda data: _tar({"x.txt": data})], ids=["gzip", "tar"]
)
def test_a_later_gzip_member_is_unpacked_on_its_own(tmp_path, capsys, encoding, wrap):
    # Joined, the members start with member 1's text, so member 2's own gzip or tar is only seen in member 2 alone.
    dataset, rows = _dataset(tmp_path)
    first = gzip.compress(_hex_lines(40), mtime=0)
    _place(dataset, encoding, first + gzip.compress(wrap(b"nothing hidden here\n"), mtime=0))
    assert _check(dataset, rows, "--hidden", "expected") == 0
    capsys.readouterr()
    root = "/" + "scr" + "atch" + "/"
    hidden = f"{ANSWER}\nKEY={SECRET_SAMPLES['aws']}\ncp {root}project/data .\n".encode()
    label = _place(dataset, encoding, first + gzip.compress(wrap(hidden), mtime=0))
    assert _check(dataset, rows, "--hidden", "expected") == 1
    out = capsys.readouterr().out
    assert "FAIL hidden-values" in out and f"{label}[gzip member 2]" in out
    assert "FAIL secrets" in out and ": AKIA" in out
    assert "FAIL local-paths" in out and f": {root}" in out


@pytest.mark.parametrize("encoding", ["raw", "base64"])
@pytest.mark.parametrize(
    "tail, problem",
    [(b"trailing bytes", "14 bytes after the gzip stream: not scanned"), (None, "a gzip stream that ends early")],
)
def test_what_follows_or_cuts_a_gzip_stream_fails_as_not_scanned(tmp_path, capsys, encoding, tail, problem):
    dataset, rows = _dataset(tmp_path)
    data = gzip.compress(_hex_lines(40), mtime=0)
    label = _place(dataset, encoding, data + tail if tail else data[:-100])
    assert _check(dataset, rows, "--hidden", "expected") == 1
    assert f"{label}: {problem}" in capsys.readouterr().out


def _incompressible(size):
    return b"".join(hashlib.sha256(str(i).encode()).digest() for i in range(size // 32))


def _payload(text):
    """`text` repeated (so a compressor codes it rather than storing it) and enough noise for a 1024+ base64 run."""
    return text.encode() * 100 + _incompressible(1024)


def _zip(text):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("answer.txt", _payload(text))
    return buffer.getvalue()


ARCHIVES = {
    "zip": lambda: _zip(ANSWER),
    "xz": lambda: lzma.compress(_payload(ANSWER)),
    "bz2": lambda: bz2.compress(_payload(ANSWER)),
}


@pytest.mark.parametrize("kind", sorted(ARCHIVES))
def test_unsupported_archives_fail_as_not_scanned_raw_in_base64_or_in_a_tar(tmp_path, capsys, kind):
    data = ARCHIVES[kind]()
    assert ANSWER.encode() not in data
    dataset, rows = _dataset(tmp_path)
    (dataset / "t1" / "environment" / f"answer.{kind}").write_bytes(data)
    blob = base64.b64encode(data).decode()
    assert len(blob) >= 1024
    (dataset / "t2" / "environment" / "Dockerfile").write_text(f"FROM python:3.12\nRUN echo '{blob}' | base64 -d\n")
    assert _check(dataset, rows, "--hidden", "expected") == 1
    out = capsys.readouterr().out
    assert f"t1/environment/answer.{kind}: a {kind} archive: not scanned" in out
    assert f"t2/environment/Dockerfile[base64 at 27]: a {kind} archive: not scanned" in out
    (dataset / "t1" / "environment" / f"answer.{kind}").unlink()
    (dataset / "t2" / "environment" / "Dockerfile").write_text("FROM python:3.12\n")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo(f"data/answer.{kind}")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    (dataset / "t1" / "environment" / "data.tar.gz").write_bytes(buffer.getvalue())
    assert _check(dataset, rows, "--hidden", "expected") == 1
    assert f"t1/environment/data.tar.gz:data/answer.{kind}: a {kind} archive: not scanned" in capsys.readouterr().out


def _tar(members, mode="w"):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode=mode) as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_a_tar_gz_inside_a_tar_is_unpacked(tmp_path, capsys, monkeypatch):
    dataset, rows = _dataset(tmp_path)
    inner = _tar({"answer.txt": b"nothing hidden here"}, mode="w:gz")
    (dataset / "t1" / "environment" / "bundle.tar").write_bytes(_tar({"inner.tar.gz": inner}))
    blob = base64.b64encode(_tar({"inner.tar.gz": inner, "noise.bin": _incompressible(1024)})).decode()
    (dataset / "t2" / "environment" / "Dockerfile").write_text(f"FROM python:3.12\nRUN echo '{blob}' | base64 -d\n")
    assert _check(dataset, rows, "--hidden", "expected") == 0
    capsys.readouterr()
    inner = _tar({"answer.txt": ANSWER.encode()}, mode="w:gz")
    assert ANSWER.encode() not in inner
    (dataset / "t1" / "environment" / "bundle.tar").write_bytes(_tar({"inner.tar.gz": inner}))
    assert _check(dataset, rows, "--hidden", "expected") == 1
    assert "t1/environment/bundle.tar:inner.tar.gz:answer.txt contains expected value" in capsys.readouterr().out
    # bundle.tar holds the member inner.tar.gz (one layer), which gunzips (two layers).
    monkeypatch.setattr(scan, "MAX_NESTING", 1)
    assert _check(dataset, rows, "--hidden", "expected") == 1
    assert "t1/environment/bundle.tar:inner.tar.gz: nested over 1 layers deep: not scanned" in capsys.readouterr().out


def test_base64_runs_in_a_tar_gz_member_are_decoded(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path)
    blob = base64.b64encode(_zip(ANSWER)).decode()
    dockerfile = f"FROM python:3.12\nRUN echo '{blob}' | base64 -d\n".encode()
    (dataset / "t1" / "environment" / "ctx.tar.gz").write_bytes(_tar({"Dockerfile": dockerfile}, mode="w:gz"))
    assert _check(dataset, rows, "--hidden", "expected") == 1
    assert "t1/environment/ctx.tar.gz:Dockerfile[base64 at 27]: a zip archive: not scanned" in capsys.readouterr().out


@pytest.mark.parametrize("encoding", ["raw", "base64"])
def test_a_gzip_inside_a_gzip_is_unpacked(tmp_path, capsys, encoding):
    dataset, rows = _dataset(tmp_path)
    _place(dataset, encoding, gzip.compress(gzip.compress(_payload("nothing hidden here"))))
    assert _check(dataset, rows, "--hidden", "expected") == 0
    capsys.readouterr()
    label = _place(dataset, encoding, gzip.compress(gzip.compress(_payload(ANSWER))))
    assert _check(dataset, rows, "--hidden", "expected") == 1
    assert f"{label} contains expected value" in capsys.readouterr().out


def _setup_bundle(text):
    """A .tar.gz whose setup.sh unpacks a base64 gzip of `text`."""
    inner = base64.b64encode(gzip.compress(text.encode() * 3 + _hex_lines(40), mtime=0)).decode()
    assert len(inner) >= 1024
    setup = f"#!/bin/sh\necho '{inner}' | base64 -d | gunzip > /answer\n".encode()
    return _tar({"setup.sh": setup}, mode="w:gz")


def test_a_bundle_finds_the_same_values_raw_or_in_base64(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path)
    for bundle, status in ((_setup_bundle("nothing hidden here"), 0), (_setup_bundle(ANSWER), 1)):
        (dataset / "t1" / "environment" / "bundle.tar.gz").write_bytes(bundle)
        blob = base64.b64encode(bundle).decode()
        (dataset / "t2" / "environment" / "Dockerfile").write_text(
            f"FROM python:3.12\nRUN echo '{blob}' | base64 -d\n"
        )
        assert _check(dataset, rows, "--hidden", "expected") == status
        out = capsys.readouterr().out
        for label in ("t1/environment/bundle.tar.gz", "t2/environment/Dockerfile[base64 at 27]"):
            assert (f"{label}:setup.sh[base64 at 16] contains expected value" in out) == bool(status)


def test_layers_count_toward_one_depth_bound(tmp_path, capsys):
    """A value base64-encoded 8 times is found; 9 times, its innermost layer is past `MAX_NESTING` (8)."""
    dataset, rows = _dataset(tmp_path)
    for times, status in ((8, 1), (9, 1)):
        blob = ANSWER.encode() + b" " + _hex_lines(12)
        for _ in range(times):
            blob = base64.b64encode(blob)
        (dataset / "t1" / "environment" / "deep.b64").write_bytes(blob + b"\n")
        assert _check(dataset, rows, "--hidden", "expected") == status
        out = capsys.readouterr().out
        label = "t1/environment/deep.b64" + "[base64 at 0]" * times
        assert (f"{label} contains expected value" in out) == (times == 8)
        assert (f"{label}: nested over 8 layers deep: not scanned" in out) == (times == 9)


def test_grouped_task_folders_are_found_as_gym_finds_them(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path)
    (dataset / "group").mkdir()
    (dataset / "t2").rename(dataset / "group" / "t2")
    assert _check(dataset, rows, "--hidden", "expected") == 0
    assert "PASS instruction: 2 task(s) equal their native prompt" in capsys.readouterr().out


def test_template_rendering_a_hidden_field_fails(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path)
    template = tmp_path / "template"
    (template / "tests").mkdir(parents=True)
    (template / "solution").mkdir()
    (template / "task.toml").write_text('version = "1.0"\n')
    (template / "tests" / "test.template.sh").write_text("echo {{ task_data.expected_answer }}\n")
    (template / "solution" / "answer.template.md").write_text("{{ expected_answer }}\n")
    (template / "notes.md").write_text("A literal {{ expected_answer }} is copied, not rendered.\n")
    (template / "instruction.template.md").write_text("{{ task_data.question }}\n")
    (template / "steps" / "one" / "tests").mkdir(parents=True)
    (template / "steps" / "one" / "tests" / "x.template.sh").write_text("echo {{ task_data.expected_answer }}\n")
    assert _check(dataset, rows, "--template", str(template), "--template-hidden", "expected_answer") == 0
    cases = [
        ("{{ task_data.expected_answer }}", "{{ task_data.expected_answer }}"),
        ("{{expected_answer}}", "{{expected_answer}}"),
        ("{{\n  task_data.expected_answer\n}}", "{{ task_data.expected_answer }}"),
    ]
    for placeholder, shown in cases:
        capsys.readouterr()
        (template / "instruction.template.md").write_text(f"{{{{ question }}}}\nHint: {placeholder}\n")
        assert _check(dataset, rows, "--template", str(template), "--template-hidden", "expected_answer") == 1
        assert f"instruction.template.md:2: renders {shown}\n" in capsys.readouterr().out


# One sample per secrets pattern, built by concatenation so this file carries no secret-like literal.
SECRET_SAMPLES = {
    "openai": "sk-" + "a" * 30,
    "nvidia": "nv" + "api-x",
    "huggingface": "hf" + "_" + "a" * 30,
    "github": "gh" + "p_" + "A" * 36,
    "github_fine_grained": "github" + "_pat_" + "A" * 30,
    "gitlab": "gl" + "pat-" + "a" * 20,
    "slack": "xo" + "xb-" + "1" * 10,
    "aws": "AK" + "IA" + "A" * 16,
    "private_key": "-----BEGIN " + "RSA PRIVATE KEY-----",
}


@pytest.mark.parametrize("kind", sorted(SECRET_SAMPLES))
def test_secret_key_fails(tmp_path, capsys, kind):
    dataset, rows = _dataset(tmp_path)
    (dataset / "t1" / "tests" / "config.py").write_text(f"KEY = '{SECRET_SAMPLES[kind]}'\n")
    assert _check(dataset, rows) == 1
    assert "FAIL secrets" in capsys.readouterr().out


def test_sk_inside_a_word_is_not_a_secret(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path)
    # From scale_swe cloudtools_troposphere_pr2185/solution/golden.patch.
    (dataset / "t1" / "tests" / "golden.patch").write_text("+# aws-properties-msk-configuration-latestrevision.html\n")
    assert _check(dataset, rows) == 0
    assert "PASS secrets" in capsys.readouterr().out


def test_cwd_path_fails(tmp_path, capsys, monkeypatch):
    dataset, rows = _dataset(tmp_path)
    workdir = tmp_path / "somewhere" / "checkout"
    workdir.mkdir(parents=True)
    monkeypatch.chdir(workdir)
    (dataset / "t2" / "environment" / "notes.txt").write_text(f"built in {workdir}/build\n")
    assert _check(dataset, rows) == 1
    assert f"t2/environment/notes.txt: {workdir}" in capsys.readouterr().out


@pytest.mark.parametrize("root", ["lus" + "tre", "ra" + "id", "scr" + "atch"])
def test_cluster_paths_fail(tmp_path, capsys, root):
    dataset, rows = _dataset(tmp_path)
    (dataset / "t1" / "environment" / "notes.txt").write_text(f"data in /{root}/project/data\n")
    assert _check(dataset, rows) == 1
    assert f"t1/environment/notes.txt: /{root}/" in capsys.readouterr().out


def test_home_counts_only_as_a_whole_path_and_never_as_root(tmp_path, capsys, monkeypatch):
    dataset, rows = _dataset(tmp_path, environment="FROM python:3.12\nENV PATH=/root/.local/bin:$PATH\n")
    monkeypatch.setenv("HOME", "/root")
    assert _check(dataset, rows) == 0
    capsys.readouterr()
    monkeypatch.setenv("HOME", "/home/al")
    (dataset / "t1" / "environment" / "notes.txt").write_text("see /home/alice/data\n")
    assert _check(dataset, rows) == 0
    capsys.readouterr()
    (dataset / "t1" / "environment" / "notes.txt").write_text("see /home/al/data\n")
    assert _check(dataset, rows) == 1
    assert "t1/environment/notes.txt: /home/al" in capsys.readouterr().out


def test_unreadable_inputs_and_report_paths_exit_2(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path)
    assert _check(dataset, rows, "--json", str(tmp_path / "missing" / "report.json")) == 2
    assert "report.json" in capsys.readouterr().err
    (dataset / "t1" / "instruction.md").write_bytes(b"\xff\xfe not UTF-8\n")
    assert _check(dataset, rows) == 2
    capsys.readouterr()
    rows.write_bytes(rows.read_bytes() + b'{"name": "\xff"}\n')
    assert _check(dataset, rows) == 2
    assert "line 3: not UTF-8" in capsys.readouterr().err


def test_json_report(tmp_path, capsys):
    dataset, rows = _dataset(tmp_path)
    report = tmp_path / "report.json"
    assert _check(dataset, rows, "--json", str(report)) == 0
    data = json.loads(report.read_text())
    assert data["dataset"] == str(dataset)
    assert [c["name"] for c in data["checks"]] == [
        "instruction",
        "hidden-values",
        "hidden-template",
        "secrets",
        "local-paths",
    ]


def test_template_dataset_is_refused(tmp_path, capsys):
    (tmp_path / "task.toml").write_text('version = "1.0"\n')
    (tmp_path / "tasks.jsonl").write_text("{}\n")
    assert main(["check", str(tmp_path)]) == 2
    assert "rendered" in capsys.readouterr().err


@pytest.mark.parametrize("args", [["--hidden", "x"], ["--template-hidden", "x"]])
def test_options_without_their_inputs_are_refused(tmp_path, capsys, args):
    dataset, _ = _dataset(tmp_path)
    assert main(["check", str(dataset), *args]) == 2
