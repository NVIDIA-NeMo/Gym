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


class TestCheckout:
    def test_unpinned_checkout_is_refused(self, tmp_path) -> None:
        import subprocess

        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        (tmp_path / "evaluate.py").write_text("")
        subprocess.run(["git", "-C", str(tmp_path), "add", "evaluate.py"], check=True)
        subprocess.run(
            ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x"],
            check=True,
        )
        with pytest.raises(upstream.UpstreamError, match="expected d066523f"):
            upstream.require_pinned_checkout(tmp_path)

    def test_missing_checkout_is_refused(self, tmp_path) -> None:
        with pytest.raises(upstream.UpstreamError, match="no upstream checkout"):
            upstream.require_pinned_checkout(tmp_path)

    def test_ensure_checkout_does_not_clone_over_an_existing_pinned_checkout(self, tmp_path, monkeypatch) -> None:
        (tmp_path / "evaluate.py").write_text("")
        monkeypatch.setattr(upstream, "checkout_head", lambda d: upstream.UPSTREAM_COMMIT)

        def no_network(*a, **k):
            raise AssertionError("must not clone")

        monkeypatch.setattr(upstream.subprocess, "run", no_network)
        upstream.ensure_checkout(tmp_path)


class TestBackendFaultMarkers:
    @pytest.mark.parametrize(
        "msg,expected",
        [
            ("Error response from daemon: No such container: x", True),
            ("Cannot connect to the Docker daemon at unix:///var/run/docker.sock", True),
            ("Line 3:2 - unsolved goals", False),
            ("Compilation timeout (>300s)", False),
            (None, False),
            ("", False),
        ],
    )
    def test_markers(self, msg, expected) -> None:
        assert upstream.is_backend_fault(msg) is expected


class TestMain:
    def test_main_writes_every_prepared_row(self, tmp_path, monkeypatch) -> None:
        root = _write_dataset(tmp_path, [_row(i) for i in range(1, 501)], monkeypatch)
        monkeypatch.setattr(prep, "ensure_checkout", lambda d: None)
        monkeypatch.setattr(prep, "PromptSource", lambda d: FakePrompts())
        out = tmp_path / "out" / "rows.jsonl"
        prep.main(["--upstream-dir", str(root), "--output", str(out), "--limit", "7"])
        lines = out.read_text().splitlines()
        assert len(lines) == 7 and json.loads(lines[-1])["verifier_metadata"]["id"] == 7


class TestResolveUpstreamDir:
    def test_relative_is_against_the_gym_root_not_the_cwd(self, monkeypatch, tmp_path) -> None:
        monkeypatch.chdir(tmp_path)
        assert upstream.resolve_upstream_dir("cache/x") == upstream.GYM_ROOT / "cache/x"
        assert (upstream.GYM_ROOT / "nemo_gym").is_dir()

    def test_absolute_is_kept(self, tmp_path) -> None:
        assert upstream.resolve_upstream_dir(tmp_path) == tmp_path
