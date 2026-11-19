# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib
import hashlib
import json
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from responses_api_agents.arc_agi_3_agent import setup_harness
from responses_api_agents.arc_agi_3_agent.harness_runtime import (
    GYM_ENGINE_ENV,
    compose_harness_env,
    harness_sys_path,
)


HARNESS_ENV_TOML = {
    "use_priority_scheduling": True,
    "notebook": {"ARC3_WARMUP_ACTION_GAMES": "1", "ARC3_HTTP_RETRY_INITIAL_SECONDS": "900"},
    "setup_env": {
        "LOCAL_ANALYZER_TEMPERATURE": "0.7",
        "ARC3_CONTEXT_KEEP_TOKENS": "24576",
        "LOCAL_ANALYZER_CONTEXT_WINDOW": 131072,
        "ARC3_GDN_OWNERSHIP": "1",
        "ARC3_GDN_CPU_RETENTION": "1",
    },
    "priority": {"ARC3_MAX_ACTIVE_STREAMS": 96},
}


class TestComposeEnv:
    def test_deployed_values_and_engine_adaptation(self) -> None:
        env = compose_harness_env(
            HARNESS_ENV_TOML, base_url="http://policy/v1", model_id="policy", hyperparameters={}, overrides={}
        )
        assert env["LOCAL_ANALYZER_TEMPERATURE"] == "0.7"
        assert env["LOCAL_ANALYZER_CONTEXT_WINDOW"] == "131072"
        assert env["ARC3_MAX_ACTIVE_STREAMS"] == "96"
        assert env["ARC3_HTTP_RETRY_INITIAL_SECONDS"] == "900"
        assert env["LOCAL_ANALYZER_BASE_URL"] == "http://policy/v1" and env["LOCAL_ANALYZER_MODEL_ID"] == "policy"
        assert env["ONLY_RESET_LEVELS"] == "true" and env["TAAF_RUN_AS_SUBMISSION"] == "0"
        for key, value in GYM_ENGINE_ENV.items():
            assert env[key] == value

    def test_hyperparameters_then_overrides(self) -> None:
        env = compose_harness_env(
            HARNESS_ENV_TOML,
            base_url="u",
            model_id="m",
            hyperparameters={
                "temperature": 1.0,
                "top_p": None,
                "enable_thinking": False,
                "context_keep_tokens": 12288,
                "request_seed": 7,
            },
            overrides={"ARC3_FORCE_ACT": 0, "ARC3_CONTEXT_KEEP_TOKENS": "49152"},
        )
        assert env["LOCAL_ANALYZER_TEMPERATURE"] == "1.0"
        assert "LOCAL_ANALYZER_TOP_P" not in env
        assert env["LOCAL_ANALYZER_ENABLE_THINKING"] == "false"
        assert env["LOCAL_ANALYZER_SEED"] == "7"
        assert env["ARC3_FORCE_ACT"] == "0"
        assert env["ARC3_CONTEXT_KEEP_TOKENS"] == "49152"

    def test_without_priority_table(self) -> None:
        toml = dict(HARNESS_ENV_TOML, use_priority_scheduling=False)
        env = compose_harness_env(toml, base_url="u", model_id="m", hyperparameters={}, overrides={})
        assert "ARC3_MAX_ACTIVE_STREAMS" not in env

    def test_sys_path_order(self, tmp_path: Path) -> None:
        src = tmp_path / "harness" / "taaf" / "src"
        (src / "tufa-arc-agi-framework" / "src").mkdir(parents=True)
        (src / "ARC3-Inference").mkdir(parents=True)
        entries = harness_sys_path(tmp_path)
        assert entries == [
            str(src / "tufa-arc-agi-framework" / "src"),
            str(src / "ARC3-Inference"),
            str(tmp_path / "harness" / "public110"),
        ]


def make_tree(root: Path) -> dict:
    files = {"harness/a.py": b"print(1)\n", "harness/sub/b.py": b"x = 2\n", "config/harness_env.toml": b"[notebook]\n"}
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    (root / "harness" / "MANIFEST.json").write_text("{}")
    sha = {rel: hashlib.sha256(data).hexdigest() for rel, data in files.items()}
    return {
        "harness_files": {"a.py": sha["harness/a.py"], "sub/b.py": sha["harness/sub/b.py"]},
        "config_files": {"harness_env.toml": sha["config/harness_env.toml"]},
    }


class TestSetupHarness:
    def test_verify_accepts_exact_tree_and_ignores_bytecode(self, tmp_path: Path) -> None:
        manifest = make_tree(tmp_path)
        (tmp_path / "harness" / "__pycache__").mkdir()
        (tmp_path / "harness" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\0")
        setup_harness.verify_install(tmp_path, manifest)

    def test_verify_rejects_changes(self, tmp_path: Path) -> None:
        manifest = make_tree(tmp_path)
        (tmp_path / "harness" / "a.py").write_text("print(2)\n")
        (tmp_path / "harness" / "extra.py").write_text("")
        with pytest.raises(ValueError) as error:
            setup_harness.verify_install(tmp_path, manifest)
        assert "sha256 differs: harness/a.py" in str(error.value) and "unpinned file harness/extra.py" in str(
            error.value
        )
        (tmp_path / "harness" / "sub" / "b.py").unlink()
        with pytest.raises(ValueError, match="missing harness/sub/b.py"):
            setup_harness.verify_install(tmp_path, manifest)

    def test_ensure_from_tarball(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        source = tmp_path / "src" / "arc3-harness"
        manifest = make_tree(source)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        monkeypatch.setattr(setup_harness, "MANIFEST_PATH", manifest_path)
        tarball = tmp_path / "harness.tar.gz"
        with tarfile.open(tarball, "w:gz") as archive:
            archive.add(source, arcname="arc3-harness")
        dest = tmp_path / "install"
        assert setup_harness.ensure_harness(str(tarball), dest) == dest
        setup_harness.verify_install(dest, manifest)
        # Installed already: no source needed.
        assert setup_harness.ensure_harness(None, dest) == dest

    def test_ensure_needs_source(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="harness_source"):
            setup_harness.ensure_harness(None, tmp_path / "absent")
        with pytest.raises(FileNotFoundError):
            setup_harness.ensure_harness(str(tmp_path / "nothing-here.zip"), tmp_path / "absent")

    def test_committed_manifest(self) -> None:
        manifest = setup_harness.load_manifest()
        assert len(manifest["harness_files"]) == 119
        prov = manifest["provenance"]
        assert prov["tool_agent_py_sha256"].startswith("22a2ae7b")
        assert prov["harness_env_toml_sha256"].startswith("a49ab988")
        assert (
            manifest["harness_files"]["taaf/src/ARC3-Inference/inference/agent/tool_agent.py"]
            == prov["tool_agent_py_sha256"]
        )


INSTALLER = """
import sys, time
from pathlib import Path
from responses_api_agents.arc_agi_3_agent import setup_harness as setup

setup.MANIFEST_PATH = Path(sys.argv[1])
stage = setup._stage_source


def slow_stage(source, workdir):  # widen the window in which two installers overlap
    with open(sys.argv[4], "a") as log:
        log.write("staged\\n")
    time.sleep(0.5)
    return stage(source, workdir)


setup._stage_source = slow_stage
print(setup.ensure_harness(sys.argv[2], Path(sys.argv[3])))
"""


def test_concurrent_installs_from_two_processes_share_one(tmp_path: Path) -> None:
    """Two servers starting at once on one checkout: one installs, the other waits and verifies its install."""
    import subprocess
    import sys

    source = tmp_path / "src"
    manifest = make_tree(source)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    dest, log = tmp_path / "install" / ".arc3_harness", tmp_path / "stagings.log"
    repo = Path(__file__).resolve().parents[3]
    args = [sys.executable, "-c", INSTALLER, str(manifest_path), str(source), str(dest), str(log)]
    processes = [
        subprocess.Popen(args, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)
    ]
    outputs = [p.communicate(timeout=60) for p in processes]
    assert [p.returncode for p in processes] == [0, 0], outputs
    assert log.read_text().splitlines() == ["staged"]  # installed once
    setup_harness.verify_install(dest, manifest)
    assert sorted(p.name for p in dest.parent.iterdir()) == [".arc3_harness", ".arc3_harness.lock"]


def test_a_failed_install_leaves_nothing_and_a_later_call_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "src"
    manifest = make_tree(source)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(setup_harness, "MANIFEST_PATH", manifest_path)
    dest = tmp_path / "install" / ".arc3_harness"
    copytree = setup_harness.shutil.copytree

    def failing_copy(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(setup_harness.shutil, "copytree", failing_copy)
    with pytest.raises(OSError, match="disk full"):
        setup_harness.ensure_harness(str(source), dest)
    assert sorted(p.name for p in dest.parent.iterdir()) == [".arc3_harness.lock"]
    monkeypatch.setattr(setup_harness.shutil, "copytree", copytree)
    assert setup_harness.ensure_harness(str(source), dest) == dest


class TestSetupHarnessSources:
    def test_from_directory_and_main(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
        source = tmp_path / "src"
        manifest = make_tree(source)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        monkeypatch.setattr(setup_harness, "MANIFEST_PATH", manifest_path)
        setup_harness.main([str(source), "--dest", str(tmp_path / "install")])
        assert "verified" in capsys.readouterr().out

    def test_from_kaggle(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import io
        import zipfile

        source = tmp_path / "src"
        manifest = make_tree(source)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        monkeypatch.setattr(setup_harness, "MANIFEST_PATH", manifest_path)
        monkeypatch.setenv("KAGGLE_API_TOKEN", "test-token")

        def zipped(kind: str) -> bytes:
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                for path in (source / kind).rglob("*"):
                    if path.is_file():
                        archive.write(path, path.relative_to(source / kind).as_posix())
            return buffer.getvalue()

        class FakeResponse:
            def __init__(self, data: bytes):
                self.data = data

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self) -> bytes:
                return self.data

        urls = []

        def fake_urlopen(request, timeout):
            urls.append((request.full_url, request.get_header("Authorization")))
            return FakeResponse(zipped("harness" if request.full_url.endswith("-harness") else "config"))

        monkeypatch.setattr(setup_harness.urllib.request, "urlopen", fake_urlopen)
        dest = setup_harness.ensure_harness("kaggle:owner/prefix", tmp_path / "install")
        setup_harness.verify_install(dest, manifest)
        assert [u for u, _ in urls] == [
            "https://www.kaggle.com/api/v1/datasets/download/owner/prefix-harness",
            "https://www.kaggle.com/api/v1/datasets/download/owner/prefix-config",
        ]
        assert all(auth is None for _, auth in urls)  # public datasets download anonymously, token or not

    def test_kaggle_token_never_follows_a_redirect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A private dataset (refused anonymously) is fetched again with the token; Kaggle redirects the download to
        another origin (cloud storage), which must not receive the token."""
        for name in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(setup_harness.urllib.request, "_opener", None)  # rebuilt without proxies
        with kaggle_and_storage(private=True) as (kaggle_api, seen):
            monkeypatch.setattr(setup_harness, "KAGGLE_API", kaggle_api)
            monkeypatch.setenv("KAGGLE_API_TOKEN", "dummy-token")
            assert setup_harness._kaggle_get(f"{kaggle_api}/datasets/download/o/p-harness") == b"payload"
        assert seen["kaggle"] == [None, "Bearer dummy-token"]
        assert seen["storage"] == [None]
        with pytest.raises(ValueError, match="refusing"):
            setup_harness._kaggle_request("https://storage.example/x", "dummy-token")

    def test_kaggle_token_is_optional(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # public datasets download without a token; one in ~/.kaggle/access_token is used when present
        monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
        monkeypatch.setattr(setup_harness.Path, "home", lambda: tmp_path)
        assert setup_harness._kaggle_token() is None
        (tmp_path / ".kaggle").mkdir()
        (tmp_path / ".kaggle" / "access_token").write_text("file-token\n")
        assert setup_harness._kaggle_token() == "file-token"

    def test_source_without_harness_dir(self, tmp_path: Path) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        with pytest.raises(FileNotFoundError, match="no harness/"):
            setup_harness._single_root(tmp_path)


@contextlib.contextmanager
def kaggle_and_storage(private: bool):
    """Two local HTTP servers on different origins: a stand-in Kaggle API that redirects every download to a stand-in
    storage server (refusing anonymous downloads when ``private``). Yields the API base URL and the Authorization
    header each server received, per request."""
    seen: dict[str, list[str | None]] = {"kaggle": [], "storage": []}

    def handler(name: str, location: list[str] | None):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:
                auth = self.headers.get("Authorization")
                seen[name].append(auth)
                if location is None:
                    self.send_response(200)
                    self.send_header("Content-Length", "7")
                    self.end_headers()
                    self.wfile.write(b"payload")
                elif private and auth is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                else:
                    self.send_response(302)
                    self.send_header("Location", location[0])
                    self.send_header("Content-Length", "0")
                    self.end_headers()

        return Handler

    storage = ThreadingHTTPServer(("127.0.0.1", 0), handler("storage", None))
    target = [f"http://localhost:{storage.server_address[1]}/bucket/file.zip"]  # another origin than 127.0.0.1
    kaggle = ThreadingHTTPServer(("127.0.0.1", 0), handler("kaggle", target))
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (storage, kaggle)]
    for thread in threads:
        thread.start()
    try:
        yield f"http://127.0.0.1:{kaggle.server_address[1]}/api/v1", seen
    finally:
        for server in (storage, kaggle):
            server.shutdown()
            server.server_close()


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "config, template_kwargs",
    [("nemotron_3_5_super_vl.yaml", {"truncate_history_thinking": False}), ("qwen_vl_arc3_kit.yaml", None)],
)
def test_model_configs_for_the_harness(config: str, template_kwargs) -> None:
    """Both model configs split reasoning and pass context-length errors through; only Nemotron adds template kwargs."""
    import yaml

    text = (REPO_ROOT / "responses_api_models" / "vllm_model" / "configs" / config).read_text()
    model = yaml.safe_load(text)["policy_model"]["responses_api_models"]["vllm_model"]
    assert model["uses_reasoning_parser"] is True and model["propagate_context_overflow_errors"] is True
    assert model["chat_template_kwargs"] == template_kwargs
    env_config = (REPO_ROOT / "resources_servers" / "arc_agi_3" / "configs" / "arc_agi_3.yaml").read_text()
    agent = yaml.safe_load(env_config)["arc_agi_3_agent"]["responses_api_agents"]["arc_agi_3_agent"]
    assert agent["chat_template_kwargs"] == {}
