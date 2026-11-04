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
"""Start states on the servers that grade the image's tree as shipped (scale_swe, swemer, swe_next,
swemer_oml)."""

import importlib
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from resources_servers.swebench import start_state
from resources_servers.swebench.tests.test_start_state import LocalSandbox, git, repo  # noqa: F401


SERVERS = {
    "scale_swe": dict(
        server="ScaleSWEResourcesServer",
        seed="ScaleSWESeedSessionRequest",
        request=dict(image_url="img"),
        inputs=dict(pre_commands="git checkout -q abc"),
        init="_ensure_git_repo",
    ),
    "swemer_v1": dict(
        server="SwemerV1ResourcesServer",
        seed="SwemerV1SeedSessionRequest",
        request=dict(image_ref="img", test_framework="pytest", test_command="pytest"),
        inputs=dict(test_patch="", test_framework="pytest", test_command="pytest"),
        init="_ensure_git_repo",
    ),
    "swemer_v2": dict(
        server="SwemerV2ResourcesServer",
        seed="SwemerV2SeedSessionRequest",
        request=dict(image_ref="img", test_framework="pytest", test_command="pytest"),
        inputs=dict(test_patch="", test_framework="pytest", test_command="pytest"),
        init="_ensure_git_repo",
    ),
    "swe_next": dict(
        server="SWENextResourcesServer",
        seed="SWENextSeedSessionRequest",
        request=dict(image_ref="img"),
        inputs=dict(test_patch=""),
        init="_init_git_repo",
    ),
    "swemer_oml": dict(
        server="SwemerOmlResourcesServer",
        seed="SwemerOmlSeedSessionRequest",
        request=dict(image_ref="img", test_sh="exit 0", config_json="{}", grade_py=""),
        inputs=dict(test_sh="exit 0", config_json="{}", grade_py=""),
        init="_ensure_git_repo",
        # Blob-stripped delivery images: the tree is re-added before seeding.
        methods=("_restore_missing_blobs",),
    ),
}


def verification(name):
    return importlib.import_module(f"resources_servers.{name}.verification")


def inputs(name, **kwargs):
    return verification(name).VerificationInputs(
        instance_id="toy", workdir="/w", patch="agent\n", **SERVERS[name]["inputs"], **kwargs
    )


@pytest.mark.parametrize("name", [*SERVERS, "swe_rebench"])
def test_task_data_declares_the_start_state_fields_like_the_request(name):
    """Undeclared, the fields count as unknown keys and fail collation in task_data_validation=error."""
    task_data = importlib.import_module(f"resources_servers.{name}.task_data").TaskData
    seed = SERVERS[name]["seed"] if name in SERVERS else "SWERebenchSeedSessionRequest"
    request = getattr(importlib.import_module(f"resources_servers.{name}.app"), seed)
    for field, bad in (("setup_patch", ["not", "a", "patch"]), ("squash_history", "sometimes")):
        assert task_data.model_fields[field].default == request.model_fields[field].default
        with pytest.raises(ValidationError) as error:
            task_data.model_validate({field: bad})
        assert (field,) in [e["loc"] for e in error.value.errors()]


MINIMAL_ROWS = {
    "swe_rebench": dict(instance_id="t", repo="o/r", base_commit="abc", image_name="img"),
    "scale_swe": dict(instance_id="t", workdir="/w", image_url="img"),
    "swe_next": dict(instance_id="t", image_ref="img"),
    "swemer_v1": dict(instance_id="t", image_ref="img", test_framework="pytest", test_command="pytest"),
    "swemer_v2": dict(instance_id="t", image_ref="img", test_framework="pytest", test_command="pytest"),
    "swemer_oml": dict(instance_id="t", image_ref="img", test_sh="exit 0", config_json="{}", grade_py=""),
}


@pytest.mark.parametrize("name", MINIMAL_ROWS)
def test_collation_accepts_start_state_rows_and_still_flags_unknown_fields(name):
    from nemo_gym.task_data import TaskDataValidator, load_task_data_schema

    adapter = load_task_data_schema(Path(start_state.__file__).parents[1] / name)
    base = {"responses_create_params": {"input": [{"role": "user", "content": "fix"}]}, **MINIMAL_ROWS[name]}
    rows = [
        base,
        {**base, "setup_patch": "diff --git a/x b/x\n", "squash_history": True},
        {**base, "squash_history": True},
    ]
    clean = TaskDataValidator(server_name=name, adapter=adapter, dataset_fpath="d.jsonl")
    for index, row in enumerate(rows):
        clean.validate_row(index, row)
    assert clean.report.clean, clean.report.summary()

    dirty = TaskDataValidator(server_name=name, adapter=adapter, dataset_fpath="d.jsonl")
    dirty.validate_row(0, {**rows[1], "not_a_task_field": 1})
    assert dirty.report.unknown_keys == {"not_a_task_field": 1}


@pytest.mark.parametrize("name", SERVERS)
def test_grading_applies_the_start_state_exactly_before_the_agent_patch(name):
    script = verification(name).build_eval_script(inputs(name, setup_patch="setup\n"))
    setup_line = start_state.apply_setup_patch_command("setup\n")
    assert script.index("cd /w") < script.index(setup_line) < script.index("/tmp/nemo_gym_patch.diff")
    if name == "scale_swe":
        assert script.index("git checkout -q abc") < script.index(setup_line)
    assert verification(name).verification_files(inputs(name, setup_patch="setup\n"))[
        start_state.SETUP_PATCH_PATH
    ] == ("setup\n")
    plain = inputs(name)
    assert start_state.SETUP_PATCH_PATH not in verification(name).verification_files(plain)
    assert start_state.SETUP_PATCH_PATH not in verification(name).build_eval_script(plain)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", SERVERS)
async def test_a_start_state_that_does_not_apply_is_incomplete_not_a_zero(name):
    class Failing:
        async def exec(self, command, timeout_s=None):
            return SimpleNamespace(
                stdout=start_state.SETUP_PATCH_FAILED_MARK + "\n",
                stderr="patch failed",
                return_code=start_state.SETUP_PATCH_FAILED_EXIT,
            )

    result = await verification(name).run_verification(Failing(), inputs(name, setup_patch="s\n"))
    assert result.completed is False and result.error == start_state.SETUP_PATCH_FAILED_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("setup_patch", ["", "s\n"])
@pytest.mark.parametrize("name", SERVERS)
async def test_a_test_command_exiting_96_keeps_the_normal_verdict(name, setup_patch):
    """Grading scripts exit with the test command's code; only the setup step prints the mark, so
    a test exit of 96 must grade exactly like any other failing exit code."""

    class TestsExit:
        def __init__(self, code):
            self.code = code

        async def exec(self, command, timeout_s=None):
            return SimpleNamespace(stdout="FAILED some_test\n", stderr="", return_code=self.code)

    def verdict(result):
        fields = dict(vars(result))
        if isinstance(fields.get("test_results"), dict):
            fields["test_results"] = {k: v for k, v in fields["test_results"].items() if k != "test_exit_code"}
        return fields

    run = verification(name).run_verification
    at_96 = await run(TestsExit(start_state.SETUP_PATCH_FAILED_EXIT), inputs(name, setup_patch=setup_patch))
    at_1 = await run(TestsExit(1), inputs(name, setup_patch=setup_patch))
    assert at_96.error != start_state.SETUP_PATCH_FAILED_ERROR
    assert verdict(at_96) == verdict(at_1)


@pytest.mark.asyncio
async def test_without_a_start_state_only_the_configured_scrub_runs(repo):  # noqa: F811
    head = git(repo, "rev-parse", "HEAD")
    await start_state.seed_start_state(
        LocalSandbox(), str(repo), "toy", "test", setup_patch="", squash_history=False, apply_anti_cheating=False
    )
    assert git(repo, "rev-parse", "HEAD") == head
    assert (repo / "__pycache__").exists()
    assert git(repo, "tag") == "v1\n"

    await start_state.seed_start_state(
        LocalSandbox(), str(repo), "toy", "test", setup_patch="", squash_history=False, apply_anti_cheating=True
    )
    assert git(repo, "rev-parse", "HEAD") == head
    assert git(repo, "tag") == ""


@pytest.mark.asyncio
async def test_squashing_a_snapshot_root_without_a_patch_seeds(repo):  # noqa: F811
    """swe_next's git-init snapshot: squashing it rebuilds the identical commit."""
    Path(start_state.SETUP_PATCH_PATH).unlink()
    git(repo, "checkout", "-q", "--orphan", "snapshot")
    git(repo, "-c", "user.name=nemo-gym", "-c", "user.email=n@e", "commit", "-qm", "nemo_gym: initial snapshot")
    git(repo, "tag", "-d", "v1")
    git(repo, "branch", "-D", git(repo, "for-each-ref", "--format=%(refname:short)", "refs/heads").split()[0])
    head = git(repo, "rev-parse", "HEAD")
    await start_state.seed_start_state(
        LocalSandbox(), str(repo), "toy", "test", setup_patch="", squash_history=True, apply_anti_cheating=False
    )
    assert git(repo, "rev-parse", "HEAD") == head
    assert git(repo, "rev-list", "--all", "--count").strip() == "1"


@pytest.mark.asyncio
@pytest.mark.parametrize("name", SERVERS)
async def test_seeding_keeps_the_image_tree_and_extracts_only_the_agent_work(name, repo, tmp_path, monkeypatch):  # noqa: F811
    app = importlib.import_module(f"resources_servers.{name}.app")
    spec = SERVERS[name]
    server_cls = getattr(app, spec["server"])
    (repo / "a.py").write_text("x = 9\n")  # The image's own uncommitted edit, which grading keeps.
    pristine = tmp_path / "pristine"
    shutil.copytree(repo, pristine)
    setup = (tmp_path / "setup.diff").read_text()

    async def create_sandbox(body, files=None):
        if start_state.SETUP_PATCH_PATH in (files or {}):
            Path(start_state.SETUP_PATCH_PATH).write_text(files[start_state.SETUP_PATCH_PATH])
        sandbox = LocalSandbox()
        sandbox._handle = SimpleNamespace(sandbox_id="local")
        return sandbox

    async def stop_sandbox(sandbox):
        return None

    server = SimpleNamespace(
        config=SimpleNamespace(
            apply_anti_cheating=False, patch_capture_mode="worktree", overwrite_maven_settings=True
        ),
        _session_id_to_sandbox={},
        _session_id_to_pristine_untracked={},
        _session_id_to_base_commit={},
        _create_sandbox=create_sandbox,
        _stop_sandbox=stop_sandbox,
    )
    for method in (spec["init"], *spec.get("methods", ())):
        setattr(server, method, lambda sandbox, workdir, m=method: getattr(server_cls, m)(server, sandbox, workdir))
    server._pristine_untracked_files = lambda sandbox, workdir: server_cls._pristine_untracked_files(
        server, sandbox, workdir
    )
    body = getattr(app, spec["seed"])(
        instance_id="toy", workdir=str(repo), setup_patch=setup, squash_history=True, **spec["request"]
    )
    request = SimpleNamespace(session={app.SESSION_ID_KEY: "s"})
    await server_cls.seed_session(server, request, body)

    assert git(repo, "rev-list", "--all", "--count").strip() == "1"
    assert git(repo, "status", "--porcelain") == ""
    assert (repo / "a.py").read_text() == "x = 9\n"
    assert (repo / "b.py").read_text() == "def f():\n    return 2\n"
    assert not Path(start_state.SETUP_PATCH_PATH).exists()

    (repo / "c.py").write_text("y = 1\n")
    base = server._session_id_to_base_commit.pop("s")
    patch = (await server_cls._extract_model_patch(server, "s", str(repo), base)).patch
    assert "c.py" in patch and "a.py" not in patch and "b.py" not in patch

    # Grading: the image's tree, then the start state, then the agent's patch.
    for diff in (setup, patch):
        subprocess.run(["git", "-C", str(pristine), "apply", "-"], input=diff, text=True, check=True)
    assert (pristine / "a.py").read_text() == "x = 9\n"
    assert (pristine / "b.py").read_text() == "def f():\n    return 2\n"
    assert (pristine / "c.py").read_text() == "y = 1\n"
