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
"""``upstream.py`` against the real pinned checkout, with Lean and docker replaced.

These tests need the checkout that ``scripts/prepare_verisoftbench.py`` creates
(``VERISOFTBENCH_UPSTREAM_DIR`` or the default cache path) and skip with a reason when
it is absent; nothing here talks to a container. They prove that upstream's modules
import under our loader, that the prompt and parse functions are upstream's, and that
``verify_candidate`` sequences upstream's renaming and retry logic the way the
evaluator does.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import upstream
from upstream import UPSTREAM_COMMIT, Harness, UpstreamError, UpstreamModules, VerificationOutcome


GYM_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_DIR = Path(
    os.environ.get("VERISOFTBENCH_UPSTREAM_DIR", GYM_ROOT / "cache/verisoftbench/VeriSoftBench-d066523")
)

pytestmark = pytest.mark.skipif(
    not (UPSTREAM_DIR / "evaluate.py").exists(),
    reason=f"pinned upstream checkout not present at {UPSTREAM_DIR}; run scripts/prepare_verisoftbench.py",
)


@pytest.fixture(scope="module")
def harness(module_monkeypatch) -> Harness:
    return Harness(UPSTREAM_DIR, docker_container="verisoftbench-lean-test")


@pytest.fixture(scope="module")
def module_monkeypatch():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


ROW = {
    "id": 42,
    "thm_name": "Demo.Inner.add_zero'",
    "thm_stmt": "theorem add_zero' (n : Nat) : n + 0 = n",
    "lean_root": "Lentil",
    "rel_path": "Lentil/Demo.lean",
    "imports": ["import Lentil.Basic"],
    "local_ctx": "import Lentil.Basic\n\nnamespace Demo\n\nnamespace Inner\n",
    "target_theorem": "theorem add_zero' (n : Nat) : n + 0 = n :=",
    "ground_truth_proof": ":= by\n  simp",
    "used_lib_defs": [{"name": "Nat", "module": "Init.Prelude"}],
    "used_repo_defs": [{"name": "foo", "content": "def foo := 1"}],
    "lib_lemmas": [{"name": "Nat.add_zero", "module": "Init.Core"}],
    "repo_lemmas": [],
    "used_local_defs": [],
    "used_local_lemmas": [],
    "category": "Semantics",
    "subset_aristotle": False,
}

SOURCE_FILE = (
    "import Lentil.Basic\n\nnamespace Demo\n\nnamespace Inner\n\n"
    "theorem earlier (n : Nat) : n = n := rfl\n\n"
    "theorem add_zero' (n : Nat) : n + 0 = n := by\n  simp\n\n"
    "theorem later (n : Nat) : 0 + n = n := by simp\n\nend Inner\n\nend Demo\n"
)


def _shadow_installed_anthropic_with_broken_sdk(monkeypatch, tmp_path: Path) -> None:
    """An installed-but-broken anthropic: ``find_spec`` sees it, importing it raises.

    Clears the cached upstream and provider modules so ``load_upstream`` re-imports the
    whole chain and reaches the broken SDK, and resets the loader cache.
    """
    (tmp_path / "anthropic.py").write_text("import verisoftbench_test_missing_dependency  # broken on purpose\n")
    monkeypatch.setattr(upstream, "_LOADED", None)
    for name in list(sys.modules):
        if name.split(".")[0] in upstream.UPSTREAM_TOP_LEVEL_MODULES + ("google", "anthropic"):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(tmp_path))


class TestLoader:
    def test_modules_come_from_the_checkout(self, harness) -> None:
        for module in (
            harness.modules.utils,
            harness.modules.lean_interface,
            harness.modules.evaluator,
            harness.modules.prompt_builder,
        ):
            assert Path(module.__file__).is_relative_to(UPSTREAM_DIR)
        assert upstream.checkout_head(UPSTREAM_DIR) == UPSTREAM_COMMIT

    def test_second_checkout_path_is_refused(self, harness, tmp_path) -> None:
        with pytest.raises(UpstreamError, match="already loaded"):
            upstream.load_upstream(tmp_path)

    def test_taken_module_name_is_detected(self, harness, monkeypatch) -> None:
        monkeypatch.setattr(upstream, "_LOADED", None)
        monkeypatch.setitem(sys.modules, "core", type(sys)("core"))
        with pytest.raises(UpstreamError, match="already taken"):
            upstream.load_upstream(UPSTREAM_DIR)
        # A lookalike sibling sharing the checkout's path as a string prefix is still
        # taken: provenance is exact path containment, not ``startswith``.
        lookalike = type(sys)("core")
        lookalike.__file__ = f"{UPSTREAM_DIR.resolve()}-evil/core/__init__.py"
        monkeypatch.setitem(sys.modules, "core", lookalike)
        with pytest.raises(UpstreamError, match="already taken"):
            upstream.load_upstream(UPSTREAM_DIR)

    def test_failed_upstream_import_restores_sys_path_and_sys_modules(self, harness, monkeypatch, tmp_path) -> None:
        # A load that fails mid-import must leave the interpreter exactly as it found
        # it: no checkout left on sys.path, no stub or half-imported package published.
        _shadow_installed_anthropic_with_broken_sdk(monkeypatch, tmp_path)
        monkeypatch.setattr(sys, "path", [p for p in sys.path if Path(p) != UPSTREAM_DIR.resolve()])
        path_before = list(sys.path)
        modules_before = dict(sys.modules)
        with pytest.raises(ModuleNotFoundError, match="verisoftbench_test_missing_dependency"):
            upstream.load_upstream(UPSTREAM_DIR)
        assert sys.path == path_before
        assert set(sys.modules) == set(modules_before)
        assert all(sys.modules[name] is module for name, module in modules_before.items())

    def test_installed_but_broken_sdk_is_not_replaced_by_a_stub(self, harness, monkeypatch, tmp_path) -> None:
        # ``find_spec`` proves anthropic is installed, so its own import failure must
        # surface to the caller; a stub standing in for a real SDK would silently
        # weaken every later host import of it.
        _shadow_installed_anthropic_with_broken_sdk(monkeypatch, tmp_path)
        with pytest.raises(ModuleNotFoundError, match="verisoftbench_test_missing_dependency"):
            upstream.load_upstream(UPSTREAM_DIR)
        assert "anthropic" not in sys.modules

    def test_stubs_for_absent_sdks_carry_module_metadata(self, monkeypatch) -> None:
        # Spec-based discovery of a published stub must return its spec, not crash the
        # way a bare ModuleType with ``__spec__ = None`` makes ``find_spec`` do.
        real_find_spec = importlib.util.find_spec
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        before = {name: sys.modules.get(name) for name in upstream._PROVIDER_STUBS}
        for name in upstream._PROVIDER_STUBS:
            monkeypatch.delitem(sys.modules, name, raising=False)
        try:
            upstream._stub_missing_provider_sdks()
            for name in upstream._PROVIDER_STUBS:
                stub = sys.modules[name]
                assert stub.__spec__ is not None and stub.__spec__.name == name
                assert real_find_spec(name) is stub.__spec__
            assert sys.modules["anthropic"].Anthropic is object
            assert sys.modules["google"].genai is sys.modules["google.genai"]
        finally:
            for name, module in before.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_checkout_modified_after_the_cached_load_is_refused(self, harness, monkeypatch, tmp_path) -> None:
        # ``load_upstream`` returns its cached modules on every later call; the tracked
        # tree must still be rechecked there so a checkout edited after the first load
        # is not presented as the pinned instrument.
        repo = tmp_path / "checkout"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        (repo / "tracked.py").write_text("original\n")
        subprocess.run(["git", "-C", str(repo), "add", "tracked.py"], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x"],
            check=True,
        )
        cached = UpstreamModules(None, None, None, None, repo.resolve())
        monkeypatch.setattr(upstream, "_LOADED", cached)
        assert upstream.load_upstream(repo) is cached
        (repo / "tracked.py").write_text("tampered\n")
        with pytest.raises(UpstreamError, match="modified tracked files"):
            upstream.load_upstream(repo)


class TestRepoIndex:
    def test_missing_required_index_fails_closed_at_startup(self, harness, monkeypatch, tmp_path) -> None:
        # A healthy harness records the identity of the index it verifies under; one
        # lost index file (or a never-copied directory) must refuse to score rather
        # than degrade to upstream's silent no-op warning.
        assert re.fullmatch(r"[0-9a-f]{64}", harness.repo_index_digest)
        monkeypatch.setattr(harness.modules.utils, "REPO_INDEX_DIR", tmp_path)
        with pytest.raises(UpstreamError, match="does not cover the benchmark"):
            Harness(UPSTREAM_DIR, docker_container="verisoftbench-lean-test")


class TestPrompts:
    def test_system_prompt_is_init_txt(self, harness) -> None:
        assert harness.system_prompt() == (UPSTREAM_DIR / "prompts/templates/init.txt").read_text()

    def test_user_prompt_has_upstreams_sections_and_the_one_shot_example(self, harness) -> None:
        prompt = harness.user_prompt(dict(ROW))
        assert prompt.startswith("### Input Information")
        for tag in ("<used_lib_defs>", "<used_repo_defs>", "<lib_lemmas>", "<local_ctx>", "<target_theorem>"):
            assert tag in prompt, tag
        task_part = prompt.split("### Your task:")[1]
        assert "<repo_lemmas>\n" not in task_part, "empty sections are omitted from the task"
        assert "Example 1 -------------------" in prompt and "Example 2" not in prompt
        assert prompt.rstrip().endswith("</target_theorem>")
        assert "Nat in Init.Prelude" in prompt and "def foo := 1" in prompt

    def test_repair_prompt_is_fix_txt_with_the_failed_round(self, harness) -> None:
        entry = dict(ROW, thm_stmt=ROW["thm_stmt"] + " :=", verif_local_ctxs="import Lentil.Basic\n")
        prompt = harness.repair_prompt(
            entry, "by\n  omega", "lemma h : True := trivial", "Line 3:2 - unsolved goals\n⊢ False"
        )
        assert prompt.startswith("You are an expert Lean 4 and formal verification agent fixing broken Lean 4 code.")
        assert "<potential_broken_lemmas>\nlemma h : True := trivial\n</potential_broken_lemmas>" in prompt
        assert (
            "<potential_broken_theorem>\ntheorem add_zero' (n : Nat) : n + 0 = n :=\nby\n  omega\n</potential_broken_theorem>"
            in prompt
        )
        assert "<err_msg>" in prompt and "unsolved goals" in prompt
        # `prompts/generated/` is not in the release, so upstream's code_before_error is empty.
        assert "<code_before_error>\n" not in prompt


class TestParsing:
    def test_initial_tags_last_occurrence_wins(self, harness) -> None:
        text = "<reasoning>I will write <lean4_proof> later</reasoning>\n<lean4_invented_lemmas>\nlemma a : True := trivial\n</lean4_invented_lemmas>\n<lean4_proof>\nby simp\n</lean4_proof>"
        proof, lemmas, reasoning = harness.parse_initial(text)
        assert (
            proof == "\nby simp\n" and lemmas == "lemma a : True := trivial" and reasoning.startswith("I will write")
        )

    def test_initial_without_tags_is_empty(self, harness) -> None:
        assert harness.parse_initial("by simp") == ("", "", "")

    def test_repair_falls_back_to_previous_round_per_tag(self, harness) -> None:
        assert harness.parse_repair("<fixed_lean4_thm_proof>by omega</fixed_lean4_thm_proof>", "old", "oldlem") == (
            "by omega",
            "oldlem",
        )
        assert harness.parse_repair("nothing here", "old", "oldlem") == ("old", "oldlem")
        assert harness.parse_repair(
            "<fixed_lean4_lemmas>\nlemma z : True := trivial\n</fixed_lean4_lemmas>", "old", "oldlem"
        ) == ("old", "lemma z : True := trivial")


class TestPrepareEntry:
    def test_context_is_rebuilt_from_the_container_file_and_statement_cleaned(self, harness, monkeypatch) -> None:
        reads = []

        def fake_read(path, timeout=30):
            reads.append(path)
            return SOURCE_FILE

        monkeypatch.setattr(harness.lean_repl, "_docker_read_file", fake_read)
        entry = harness.prepare_entry(dict(ROW))
        assert reads == ["/workspace/lean_repos/Lentil/Lentil/Demo.lean"]
        assert entry["thm_stmt"] == "theorem add_zero' (n : Nat) : n + 0 = n :="
        assert entry["verif_local_ctxs"].endswith("theorem earlier (n : Nat) : n = n := rfl\n\n")
        assert "later" not in entry["verif_local_ctxs"]

    def test_missing_file_falls_back_to_imports_plus_local_ctx(self, harness, monkeypatch) -> None:
        monkeypatch.setattr(harness.lean_repl, "_docker_read_file", lambda path, timeout=30: None)
        entry = harness.prepare_entry(dict(ROW))
        assert entry["verif_local_ctxs"] == "import Lentil.Basic\n" + ROW["local_ctx"]


def _docker_run_fake(compile_result=None, write=(0, ""), compile_raises=None):
    """``subprocess.run`` standing in for docker: ``cat`` reads, ``tee`` writes, ``lake`` compiles.

    ``compile_result`` is ``(returncode, stdout, stderr)``; ``{file}`` in the stdout is
    replaced with the container path of the compiled candidate file.
    """

    def run(cmd, **kwargs):
        if "tee" in cmd:
            rc, stderr = write
            return subprocess.CompletedProcess(cmd, rc, stdout="", stderr=stderr)
        if "cat" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=SOURCE_FILE, stderr="")
        if "rm" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        assert "lake" in cmd, f"unexpected docker command: {cmd}"
        if compile_raises is not None:
            raise compile_raises
        rc, stdout, stderr = compile_result
        return subprocess.CompletedProcess(cmd, rc, stdout=stdout.format(file=cmd[-1]), stderr=stderr)

    return run


class TestVerifyCandidate:
    @pytest.fixture
    def entry(self, harness, monkeypatch):
        monkeypatch.setattr(harness.lean_repl, "_docker_read_file", lambda path, timeout=30: SOURCE_FILE)
        return harness.prepare_entry(dict(ROW))

    def test_success_passes_upstreams_arguments_to_verify_proof(self, harness, entry, monkeypatch) -> None:
        calls = []

        def fake_verify(**kw):
            calls.append(kw)
            return True, ""

        monkeypatch.setattr(harness.lean_repl, "verify_proof", fake_verify)
        out = harness.verify_candidate(entry, "by\n  simp", "", "p1", max_conflict_attempts=1)
        assert out == VerificationOutcome(True, None, "by\n  simp", "", 0)
        [kw] = calls
        assert kw["thm_name"] == ROW["thm_name"] and kw["repo_name"] == "Lentil" and kw["proof_id"] == "p1"
        assert kw["theorem_stmt"] == entry["thm_stmt"] and kw["local_context"] == entry["verif_local_ctxs"]

    def test_axiom_becomes_theorem_and_leaked_identifier_is_renamed(
        self, harness, entry, monkeypatch, tmp_path
    ) -> None:
        # The synthetic theorem is not in upstream's repo index — the deliberate shape of
        # the out-of-corpus example tasks — so `clean_leaked_identifiers` returns early
        # and does not even add the `by` prefix; the axiom rewrite is unconditional.
        calls = []

        def fake_verify(**kw):
            calls.append(kw)
            return False, "Line 9:2 - unsolved goals"

        monkeypatch.setattr(harness.lean_repl, "verify_proof", fake_verify)
        out = harness.verify_candidate(entry, "simp", "axiom h : True", "p2", max_conflict_attempts=1)
        assert not out.verified and out.error_msg == "Line 9:2 - unsolved goals"
        assert calls[0]["aux_lemmas"] == "theorem h : True" and calls[0]["theorem_proof"] == "simp"
        # With the task's file indexed — what every benchmark task has after Setup
        # step 4 — a reference to the earlier same-file theorem is renamed out of the
        # proof, so dropping the anti-leak call is detectable here.
        index = {
            "theorems_by_file": {
                ROW["rel_path"]: [
                    {
                        "name": "Demo.Inner.earlier",
                        "aliases": ["Inner.earlier", "earlier"],
                        "kind": "theorem",
                        "namespace": "Demo.Inner",
                        "line": 7,
                        "end_line": 7,
                        "text": "theorem earlier (n : Nat) : n = n := rfl",
                    },
                    {
                        "name": ROW["thm_name"],
                        "aliases": [],
                        "kind": "theorem",
                        "namespace": "Demo.Inner",
                        "line": 9,
                        "end_line": 10,
                        "text": "theorem add_zero' (n : Nat) : n + 0 = n := by\n  simp",
                    },
                ]
            }
        }
        (tmp_path / "Lentil.json").write_text(json.dumps(index))
        monkeypatch.setattr(harness.modules.utils, "REPO_INDEX_DIR", tmp_path)
        harness.verify_candidate(entry, "by\n  exact earlier n", "", "p2b", max_conflict_attempts=1)
        assert calls[-1]["theorem_proof"] == "by\n  exact my_earlier n"

    def test_declared_clash_is_renamed_and_retried_within_budget(self, harness, entry, monkeypatch) -> None:
        answers = iter(
            [
                (False, "Line 7:8 - 'Demo.Inner.helper' has already been declared"),
                (False, "Line 7:8 - 'Demo.Inner.helper_1' has already been declared"),
                (True, ""),
            ]
        )
        calls = []

        def fake_verify(**kw):
            calls.append(kw["aux_lemmas"])
            return next(answers)

        monkeypatch.setattr(harness.lean_repl, "verify_proof", fake_verify)
        lemmas = "lemma helper : True := trivial"
        out = harness.verify_candidate(entry, "by\n  exact helper", lemmas, "p3", max_conflict_attempts=10)
        assert out.verified and out.conflict_resolution_attempts == 2 and len(calls) == 3
        assert calls[0] == lemmas and calls[1] != lemmas and "helper" in calls[1]

    def test_round_zero_budget_is_one_retry(self, harness, entry, monkeypatch) -> None:
        monkeypatch.setattr(
            harness.lean_repl,
            "verify_proof",
            lambda **kw: (False, "Line 7:8 - 'Demo.Inner.helper' has already been declared"),
        )
        out = harness.verify_candidate(
            entry, "by\n  exact helper", "lemma helper : True := trivial", "p4", max_conflict_attempts=1
        )
        assert not out.verified and out.conflict_resolution_attempts == 1

    # The cases below go through the REAL pinned ``verify_proof``/``compile_file`` with
    # only ``subprocess.run`` mocked: upstream's parser keeps Lean-looking lines only, so
    # a daemon failure with no source location is reduced to the bare "Compilation
    # failed" — the typed capture at the docker exec boundary must see it first.

    @pytest.mark.parametrize(
        "rc,stdout,stderr,expect",
        [
            (
                126,
                "",
                "Error response from daemon: Container verisoftbench-lean-test is not running",
                "Error response from daemon",
            ),
            (
                126,
                "",
                "OCI runtime exec failed: exec failed: unable to start container process: error during container init",
                "OCI runtime exec failed",
            ),
            (
                1,
                "info: Cli: checking out revision 'd9fc8ae'\nerror: external command 'git' exited with code 128",
                "",
                "external command 'git'",
            ),
        ],
    )
    def test_transport_and_lake_faults_are_captured_before_formatting_loses_them(
        self, harness, entry, monkeypatch, rc, stdout, stderr, expect
    ) -> None:
        monkeypatch.setattr(subprocess, "run", _docker_run_fake(compile_result=(rc, stdout, stderr)))
        out = harness.verify_candidate(entry, "by\n  simp", "", "p5", max_conflict_attempts=1)
        assert not out.verified and out.backend_fault
        assert expect in out.error_msg

    def test_marker_text_inside_a_real_diagnostic_stays_charged(self, harness, entry, monkeypatch) -> None:
        # A genuine elaborator diagnostic whose text contains a backend marker: it sits
        # after a source location, so it is the candidate's failure, not infrastructure.
        stdout = "{file}:4:2: error: unknown identifier 'is not running'\n"
        monkeypatch.setattr(subprocess, "run", _docker_run_fake(compile_result=(1, stdout, "")))
        out = harness.verify_candidate(entry, "by\n  simp", "", "p6", max_conflict_attempts=1)
        assert not out.verified and not out.backend_fault
        assert out.error_msg == "Line 4:2 - error: unknown identifier 'is not running'"

    def test_failed_container_write_is_a_backend_fault_not_a_lean_error(self, harness, entry, monkeypatch) -> None:
        # Upstream ignores the write helper's exit code; the candidate file was never
        # materialized, so the subsequent compile failure must not be charged.
        monkeypatch.setattr(
            subprocess,
            "run",
            _docker_run_fake(compile_result=(1, "", ""), write=(1, "tee: /workspace: No space left on device")),
        )
        out = harness.verify_candidate(entry, "by\n  simp", "", "p7", max_conflict_attempts=1)
        assert not out.verified and out.backend_fault
        assert "writing" in out.error_msg and "No space left on device" in out.error_msg

    def test_transport_failure_while_reading_the_source_is_a_backend_fault(self, harness, monkeypatch) -> None:
        # Upstream maps any failed read to None; a daemon failure there must surface
        # rather than silently degrade the verification context.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(harness.lean_repl, "_docker_read_file", lambda path, timeout=30: SOURCE_FILE)
            entry = harness.prepare_entry(dict(ROW))
        daemon = "Error response from daemon: Container verisoftbench-lean-test is not running"

        def run(cmd, **kwargs):
            if "cat" in cmd:
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=daemon)
            if "tee" in cmd or "rm" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", run)
        out = harness.verify_candidate(entry, "by\n  simp", "", "p8", max_conflict_attempts=1)
        assert not out.verified and out.backend_fault
        assert out.error_msg.startswith("reading ") and "Error response from daemon" in out.error_msg

    def test_compile_timeout_stays_charged(self, harness, entry, monkeypatch) -> None:
        # Upstream's released protocol charges the timeout to the candidate.
        monkeypatch.setattr(
            subprocess, "run", _docker_run_fake(compile_raises=subprocess.TimeoutExpired(cmd="lake", timeout=600))
        )
        out = harness.verify_candidate(entry, "by\n  simp", "", "p9", max_conflict_attempts=1)
        assert not out.verified and not out.backend_fault
        assert "timeout" in out.error_msg.lower()


class TestContainerHealth:
    def test_running(self, harness, monkeypatch) -> None:
        monkeypatch.setattr(
            subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="true\n", stderr="")
        )
        assert harness.container_running() is True

    def test_stopped_or_missing(self, harness, monkeypatch) -> None:
        monkeypatch.setattr(
            subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="", stderr="No such object")
        )
        assert harness.container_running() is False

    def test_docker_missing(self, harness, monkeypatch) -> None:
        def boom(*a, **k):
            raise FileNotFoundError("docker")

        monkeypatch.setattr(subprocess, "run", boom)
        assert harness.container_running() is False
