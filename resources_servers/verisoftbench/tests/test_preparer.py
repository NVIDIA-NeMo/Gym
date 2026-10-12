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
"""The preparer with the network and the upstream checkout replaced by fixtures.

Every path that would produce a silently wrong dataset is exercised: a dataset whose
digest does not match the pin, a short or mis-numbered corpus, the ``--limit`` boundary,
and the shape of an emitted row.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR / "scripts"))
import prepare_verisoftbench as prep  # noqa: E402
import upstream  # noqa: E402


def _row(i: int) -> dict:
    return {
        "id": i,
        "thm_name": f"Demo.t{i}",
        "thm_stmt": f"theorem t{i} : True",
        "lean_root": "Lentil",
        "rel_path": "Lentil/Demo.lean",
        "imports": [],
        "local_ctx": "",
        "target_theorem": f"theorem t{i} : True :=",
        "ground_truth_proof": ":= trivial",
        "category": "Semantics",
        "subset_aristotle": False,
    }


def _write_dataset(tmp_path: Path, rows: list[dict], monkeypatch, pin_digest: bool = True) -> Path:
    path = tmp_path / upstream.DATASET_RELATIVE_PATH
    path.parent.mkdir(parents=True)
    data = "".join(json.dumps(r) + "\n" for r in rows).encode()
    path.write_bytes(data)
    if pin_digest:
        monkeypatch.setattr(prep, "DATASET_SHA256", hashlib.sha256(data).hexdigest())
    return tmp_path


class FakePrompts:
    def system_prompt(self):
        return "SYSTEM"

    def user_prompt(self, row):
        return f"<local_ctx>\n{row['local_ctx']}\n</local_ctx>\n\n<target_theorem>\n{row['target_theorem']}\n</target_theorem>\n\n"


class TestLoadRows:
    def test_digest_mismatch_fails_closed(self, tmp_path, monkeypatch) -> None:
        root = _write_dataset(tmp_path, [_row(i) for i in range(1, 501)], monkeypatch, pin_digest=False)
        with pytest.raises(prep.CorpusError, match="sha256"):
            prep.load_rows(root)

    def test_short_corpus_fails_closed(self, tmp_path, monkeypatch) -> None:
        root = _write_dataset(tmp_path, [_row(i) for i in range(1, 500)], monkeypatch)
        with pytest.raises(prep.CorpusError, match="499 rows parsed, expected 500"):
            prep.load_rows(root)

    def test_wrong_ids_fail_closed(self, tmp_path, monkeypatch) -> None:
        rows = [_row(i) for i in range(1, 501)]
        rows[3]["id"] = 4000
        root = _write_dataset(tmp_path, rows, monkeypatch)
        with pytest.raises(prep.CorpusError, match="ids"):
            prep.load_rows(root)

    def test_malformed_line_names_the_line(self, tmp_path, monkeypatch) -> None:
        root = _write_dataset(tmp_path, [_row(i) for i in range(1, 501)], monkeypatch)
        path = root / upstream.DATASET_RELATIVE_PATH
        data = path.read_bytes().replace(b'{"id": 5,', b'{"id": 5', 1)
        path.write_bytes(data)
        monkeypatch.setattr(prep, "DATASET_SHA256", hashlib.sha256(data).hexdigest())
        with pytest.raises(prep.CorpusError, match=":5 is not JSON"):
            prep.load_rows(root)

    def test_full_corpus_loads(self, tmp_path, monkeypatch) -> None:
        root = _write_dataset(tmp_path, [_row(i) for i in range(1, 501)], monkeypatch)
        assert len(prep.load_rows(root)) == 500


class TestPrepare:
    def test_row_shape(self) -> None:
        [out] = prep.prepare([_row(1)], FakePrompts(), limit=None)
        assert out["agent_ref"] == {"type": "responses_api_agents", "name": "verisoftbench_agent"}
        assert [m["role"] for m in out["responses_create_params"]["input"]] == ["system", "user"]
        assert out["responses_create_params"]["input"][0]["content"] == "SYSTEM"
        assert out["verifier_metadata"]["id"] == 1 and out["verifier_metadata"]["context_mode"] == "filtered_context"
        assert out["verifier_metadata"]["ground_truth_proof"] == ":= trivial"

    def test_limit_takes_the_first_n(self) -> None:
        out = prep.prepare([_row(i) for i in range(1, 11)], FakePrompts(), limit=3)
        assert [r["verifier_metadata"]["id"] for r in out] == [1, 2, 3]

    def test_benchmark_form_carries_the_prompts_but_no_input(self) -> None:
        # Gym rejects a populated responses_create_params.input wherever a prompt_config is
        # declared, so the benchmark form must differ from the explicit form in exactly that key.
        [explicit] = prep.prepare([_row(1)], FakePrompts(), limit=None)
        [out] = prep.prepare([_row(1)], FakePrompts(), limit=None, benchmark_form=True)
        assert "responses_create_params" not in out
        assert out == {k: v for k, v in explicit.items() if k != "responses_create_params"}
        assert out["system_prompt"] == "SYSTEM" and "<target_theorem>" in out["user_prompt"]

    def test_prompt_without_target_theorem_fails_closed(self) -> None:
        class Broken(FakePrompts):
            def user_prompt(self, row):
                return "nothing"

        with pytest.raises(prep.CorpusError, match="target_theorem"):
            prep.prepare([_row(1)], Broken(), limit=None)


class TestArguments:
    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_non_positive_limit_rejected_at_parse_time(self, value) -> None:
        with pytest.raises(SystemExit):
            prep.parse_args(["--limit", value])

    def test_defaults(self) -> None:
        args = prep.parse_args([])
        assert args.limit is None and args.upstream_dir == prep.DEFAULT_UPSTREAM_DIR


def _committed_checkout(path: Path) -> None:
    """A git checkout with ``evaluate.py`` committed, standing in for the upstream clone."""
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "evaluate.py").write_text("")
    subprocess.run(["git", "-C", str(path), "add", "evaluate.py"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x"],
        check=True,
    )


class TestCheckout:
    def test_unpinned_checkout_is_refused(self, tmp_path) -> None:
        _committed_checkout(tmp_path)
        with pytest.raises(upstream.UpstreamError, match="expected d066523f"):
            upstream.require_pinned_checkout(tmp_path)

    def test_missing_checkout_is_refused(self, tmp_path) -> None:
        with pytest.raises(upstream.UpstreamError, match="no upstream checkout"):
            upstream.require_pinned_checkout(tmp_path)

    def test_dirty_tracked_file_at_the_pinned_head_is_refused(self, tmp_path, monkeypatch) -> None:
        # The right HEAD with edited tracked content is a different instrument; the
        # untracked artifacts the setup generates (the extracted repo_index) stay allowed.
        _committed_checkout(tmp_path)
        monkeypatch.setattr(upstream, "UPSTREAM_COMMIT", upstream.checkout_head(tmp_path))
        upstream.require_pinned_checkout(tmp_path)
        (tmp_path / "repo_index.json").write_text("{}")
        upstream.require_pinned_checkout(tmp_path)
        (tmp_path / "evaluate.py").write_text("tampered")
        with pytest.raises(upstream.UpstreamError, match="modified tracked files"):
            upstream.require_pinned_checkout(tmp_path)

    def test_ensure_checkout_does_not_clone_over_an_existing_pinned_checkout(self, tmp_path, monkeypatch) -> None:
        (tmp_path / "evaluate.py").write_text("")
        monkeypatch.setattr(upstream, "checkout_head", lambda d: upstream.UPSTREAM_COMMIT)
        monkeypatch.setattr(upstream, "require_clean_tracked_tree", lambda d: None)

        def no_network(*a, **k):
            raise AssertionError("must not clone")

        monkeypatch.setattr(upstream.subprocess, "run", no_network)
        upstream.ensure_checkout(tmp_path)


class TestBackendFaultClassification:
    """``classify_transport_fault`` reads the typed docker exec result, not message text.

    Its matches are *suspicions*: candidate source can print marker-shaped lines, so a
    suspicion becomes a backend fault only after ``Harness._resolve_fault`` confirms it
    with a candidate-free probe (``test_upstream.TestVerifyCandidate``).
    """

    @staticmethod
    def _result(returncode: int, stdout: str = "", stderr: str = ""):
        import subprocess

        return subprocess.CompletedProcess(["docker", "exec"], returncode, stdout=stdout, stderr=stderr)

    @pytest.mark.parametrize(
        "stderr",
        [
            "Error response from daemon: Container 3f4a is not running",
            "Error response from daemon: No such container: x",
            "Error: No such container: x",
            "Cannot connect to the Docker daemon at unix:///var/run/docker.sock",
            'OCI runtime exec failed: exec failed: unable to start container process: exec: "lake": '
            "executable file not found in $PATH: unknown",
        ],
    )
    def test_docker_transport_stderr_is_a_suspicion(self, stderr) -> None:
        suspicion = upstream.classify_transport_fault(self._result(126, stderr=stderr))
        assert suspicion is not None and stderr.splitlines()[0] in suspicion.message
        assert suspicion.kind == "transport"

    def test_pre_elaboration_lake_git_failure_is_a_suspicion(self) -> None:
        output = "info: Cli: checking out revision 'd9fc8ae'\nerror: external command 'git' exited with code 128"
        suspicion = upstream.classify_transport_fault(self._result(1, stdout=output))
        assert suspicion is not None and "external command 'git'" in suspicion.message
        assert suspicion.kind == "lake"

    @pytest.mark.parametrize(
        "stdout,stderr",
        [
            # An elaborator diagnostic whose text happens to contain a marker is the
            # model's failure, not the backend's: it sits after a source location.
            ("././Lentil/Demo.lean:4:2: error: unknown identifier 'is not running'", ""),
            ("", "././Lentil/Demo.lean:4:2: error: unexpected token; expected 'Error response from daemon'"),
            ("././Lentil/Demo.lean:9:0: error: unsolved goals\n⊢ external command 'git' exited with code 128 = x", ""),
            ("Line 3:2 - unsolved goals", ""),
        ],
    )
    def test_compiler_diagnostics_stay_charged(self, stdout, stderr) -> None:
        assert upstream.classify_transport_fault(self._result(1, stdout=stdout, stderr=stderr)) is None

    def test_success_is_never_a_fault(self) -> None:
        assert upstream.classify_transport_fault(self._result(0, stderr="Error response from daemon: x")) is None


class TestMain:
    def test_main_writes_every_prepared_row(self, tmp_path, monkeypatch) -> None:
        root = _write_dataset(tmp_path, [_row(i) for i in range(1, 501)], monkeypatch)
        monkeypatch.setattr(prep, "ensure_checkout", lambda d: None)
        monkeypatch.setattr(prep, "PromptSource", lambda d: FakePrompts())
        out = tmp_path / "out" / "rows.jsonl"
        prep.main(["--upstream-dir", str(root), "--output", str(out), "--limit", "7"])
        lines = out.read_text().splitlines()
        assert len(lines) == 7 and json.loads(lines[-1])["verifier_metadata"]["id"] == 7

    def test_main_benchmark_form_keeps_the_checks_and_reports_the_output_digest(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        root = _write_dataset(tmp_path, [_row(i) for i in range(1, 501)], monkeypatch)
        monkeypatch.setattr(prep, "ensure_checkout", lambda d: None)
        monkeypatch.setattr(prep, "PromptSource", lambda d: FakePrompts())
        out = tmp_path / "out" / "benchmark.jsonl"
        prep.main(["--upstream-dir", str(root), "--output", str(out), "--limit", "3", "--benchmark-form"])
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert [r["verifier_metadata"]["id"] for r in rows] == [1, 2, 3]
        assert all("responses_create_params" not in r and r["system_prompt"] == "SYSTEM" for r in rows)
        # The printed digest is how a new split's identity gets recorded; it must hash the written bytes.
        assert hashlib.sha256(out.read_bytes()).hexdigest() in capsys.readouterr().out

    def test_main_benchmark_form_fails_closed_on_a_digest_mismatch(self, tmp_path, monkeypatch) -> None:
        root = _write_dataset(tmp_path, [_row(i) for i in range(1, 501)], monkeypatch, pin_digest=False)
        monkeypatch.setattr(prep, "ensure_checkout", lambda d: None)
        out = tmp_path / "out" / "benchmark.jsonl"
        with pytest.raises(prep.CorpusError, match="sha256"):
            prep.main(["--upstream-dir", str(root), "--output", str(out), "--benchmark-form"])
        assert not out.exists()


class TestResolveUpstreamDir:
    def test_relative_is_against_the_gym_root_not_the_cwd(self, monkeypatch, tmp_path) -> None:
        monkeypatch.chdir(tmp_path)
        assert upstream.resolve_upstream_dir("cache/x") == upstream.GYM_ROOT / "cache/x"
        assert (upstream.GYM_ROOT / "nemo_gym").is_dir()

    def test_absolute_is_kept(self, tmp_path) -> None:
        assert upstream.resolve_upstream_dir(tmp_path) == tmp_path
