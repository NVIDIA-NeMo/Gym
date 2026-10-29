# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym authoring extensions (M31): parameterized tasks, Python verifiers and their shim, answer.md, publish."""

import ast
import fcntl
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import nemo_gym.verifiers as verifiers_package
from nemo_gym.data import DIGEST_KEY, HarborTaskError, VerifierShim, load_task, write_shim
from nemo_gym.data.tasks.harbor import content_hash, parameterized
from nemo_gym.data.tasks.harbor.parameterized import (
    ParameterizedError,
    is_template_folder,
    read_rows,
    render_template_folder,
    render_text,
    rendered_name,
    rendered_task_toml,
)
from nemo_gym.data.tasks.harbor.prepare import prepare_target
from nemo_gym.data.tasks.harbor.publish import PublishError, answer_write_command, publish_harbor, strip_gym_table
from nemo_gym.data.tasks.harbor.python_verifier import (
    PYTHON_FLOOR,
    PythonVerifierError,
    declared_arguments,
    python_verifier_for,
    reads_response,
)
from nemo_gym.data.tasks.harbor.python_verifier import test_script as shim_test_script
from nemo_gym.data.tasks.harbor.records import read_built_images, read_image_configs
from nemo_gym.data.tasks.harbor.scaffold import init_dataset
from nemo_gym.data.tasks.harbor.task_data import TASK_DATA_FILE
from nemo_gym.verifiers import exact_match, final_text, numeric_match
from nemo_gym.verifiers.files import file_exists, read_text, text_file_equals
from nemo_gym.verifiers.runner import main as run_verifier
from nemo_gym.verifiers.runner import place_on_path
from tests.unit_tests.test_harbor_tasks import HELLO_TOML, no_real_registry, write_task  # noqa: F401 (autouse fixture)


REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "environments"

ANSWER_VERIFIER = """
from pathlib import Path


def verify(workspace: Path, task_data: dict) -> float:
    answer = (workspace / "answer.txt").read_text().strip() if (workspace / "answer.txt").is_file() else ""
    return float(answer == task_data["answer"])
"""

# The publishable answer shape: the oracle (Gym's, and Harbor's through the published solve.sh) leaves answer.md in the
# working directory, and the verifier reads it there.
ANSWER_MD_VERIFIER = """
from nemo_gym.verifiers.files import read_text


def verify(workspace, task_data) -> float:
    return float(read_text(workspace, "answer.md") == task_data["answer"])
"""

TASK_DATA_PY = "from pydantic import BaseModel\nclass TaskData(BaseModel):\n    question: str\n    answer: str\n"

RESPONSE_VERIFIER = """
from nemo_gym.verifiers import final_text


def verify(task_data, response) -> float:
    return float(final_text(response).strip() == task_data["answer"])
"""


def write_template(root: Path, *, rows: list[dict] | None = None, verifier: str = ANSWER_VERIFIER) -> Path:
    """A template folder: task.toml at the root, a templated instruction and solution, a Python verifier."""
    root.mkdir(parents=True)
    (root / "task.toml").write_text(HELLO_TOML)
    (root / "environment").mkdir()
    (root / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n\nWORKDIR /app")
    (root / "instruction.template.md").write_text("Solve: {{ task_data.question }}\nWrite the answer to answer.txt.\n")
    (root / "tests").mkdir()
    (root / "tests" / "verifier.py").write_text(verifier)
    (root / "solution").mkdir()
    (root / "solution" / "solve.template.sh").write_text("#!/bin/bash\necho '{{ answer }}' > answer.txt\n")
    (root / "solution" / "solve.template.sh").chmod(0o755)
    (root / "solution" / "notes.txt").write_text("not a template: {{ answer }} stays\n")
    rows = (
        rows
        if rows is not None
        else [
            {"task_id": "q-1", "task_data": {"question": "2+2?", "answer": "4"}},
            {"task_id": "q-2", "question": "6*7?", "answer": "42"},
        ]
    )
    (root / "tasks.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return root


def published_verifier(task: Path) -> Path:
    """The module a published task's test.sh runs (``--verifier /tests/<path>``), as a path under its ``tests/``."""
    match = re.search(r"--verifier /tests/(\S+)", (task / "tests" / "test.sh").read_text())
    assert match is not None, (task / "tests" / "test.sh").read_text()
    return task / "tests" / match.group(1)


def run_published_shim(task: Path, workspace: Path, reward_dir: Path) -> subprocess.CompletedProcess:
    """Run a published task's shim the way its test.sh does: a fresh interpreter running the runner as a plain script
    (no ``-I``), so the interpreter itself puts ``.ng`` on ``sys.path`` and the host's ``nemo_gym`` is importable."""
    return subprocess.run(
        [
            sys.executable,
            str(task / "tests" / ".ng" / "run_verifier.py"),
            "--verifier",
            str(published_verifier(task)),
            "--workspace",
            str(workspace),
            "--task-data",
            str(task / "tests" / ".ng" / "task_data.json"),
            "--reward-dir",
            str(reward_dir),
        ],
        capture_output=True,
        text=True,
    )


class TestTemplateRendering:
    def test_rows_accept_nested_or_flat_fields(self, tmp_path):
        template = write_template(tmp_path / "qa")
        assert is_template_folder(template)
        rows = read_rows(template)
        assert [(row.task_id, row.task_data) for row in rows] == [
            ("q-1", {"question": "2+2?", "answer": "4"}),
            ("q-2", {"question": "6*7?", "answer": "42"}),
        ]

    @pytest.mark.parametrize(
        ("line", "fragment"),
        [
            ("[]", "a row must be a JSON object"),
            ('{"task_data": {"a": 1}}', "needs a non-empty string `task_id`"),
            ('{"task_id": "../x", "a": 1}', "tasks.jsonl:1: "),
            ('{"task_id": "x", "task_data": [1]}', "`task_data` must be a JSON object"),
            ('{"task_id": "x", "task_data": {"a": 1}, "b": 2}', "carries nothing else, got ['b']"),
            ('{"task_id": "x", "_ng_digest": "d"}', "keys ['_ng_digest'] are reserved"),
            ('{"task_id": "x", "a": 1}\n{"task_id": "x", "a": 2}', "duplicate task_id 'x'"),
            ("not json", "tasks.jsonl:1: "),
            ("", "has no rows"),
            # Names the rendered dataset root keeps for itself, and hidden names task discovery would skip.
            ('{"task_id": "dataset.toml", "a": 1}', "tasks.jsonl:1: task_id 'dataset.toml' is reserved"),
            ('{"task_id": "tasks.jsonl", "a": 1}', "task_id 'tasks.jsonl' is reserved"),
            ('{"task_id": "built-images.json.lock", "a": 1}', "task_id 'built-images.json.lock' is reserved"),
            ('{"task_id": "verifier.py", "a": 1}', "task_id 'verifier.py' is reserved"),
            ('{"task_id": ".hidden", "a": 1}', "tasks.jsonl:1: task_id '.hidden' cannot start with '.'"),
            # One byte over NAME_MAX: refused here by line, not at mkdir with an OSError naming the staging path.
            (
                '{"task_id": "%s", "a": 1}' % ("x" * 256),
                "tasks.jsonl:1: task_id 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'... is 256 bytes long; a task folder name holds at most 255",
            ),
        ],
    )
    def test_malformed_rows_are_named_by_line(self, tmp_path, line, fragment):
        template = write_template(tmp_path / "qa")
        (template / "tasks.jsonl").write_text(line + "\n")
        with pytest.raises(ParameterizedError) as excinfo:
            read_rows(template)
        assert fragment in str(excinfo.value)

    def test_placeholders_render_strings_and_json_values(self):
        fields = {"question": "2+2?", "n": 3, "tags": ["a", "b"]}
        assert (
            render_text("Q: {{ task_data.question }} ({{n}}) {{ tags }}", fields, where="t")
            == 'Q: 2+2? (3) ["a", "b"]'
        )
        with pytest.raises(ParameterizedError, match=r"task 'x': placeholder '\{\{ missing \}\}' names no field"):
            render_text("{{ missing }}", fields, where="task 'x'")
        # Any other mustache spelling is a misspelt placeholder: refused naming it, never shipped as raw text.
        for text in (
            "{{ task_data.question.upper }}",
            "{{ task-data.question }}",
            "{{ question | upper }}",
            '{{ task_data["question"] }}',
            "ok {{ n }} then {{ }}",
            "{{\n question | x\n}} spans lines",
        ):
            with pytest.raises(ParameterizedError, match=r"task 'x': '\{\{.*\}\}' is not a placeholder"):
                render_text(text, fields, where="task 'x'")
        # A `{{` that never closes, directly before a valid placeholder whose `}}` would otherwise pass for its end
        # (the valid one is gone when the leftovers are searched), at the end of the text, or alone on a line.
        for text, leftover in (
            ("Q: {{ a }{{ question }}", "{{ a }"),
            ("{{ question }} then {{ b", "{{ b"),
            ("{{ question }}\n{{ c\n{{ n }}", "{{ c"),
            ("{{", "{{"),
        ):
            with pytest.raises(ParameterizedError) as excinfo:
                render_text(text, fields, where="task 'x'")
            assert str(excinfo.value).startswith(
                f"task 'x': {leftover!r} opens a placeholder that never closes; write"
            )
        # A value that contains `{{` is not a placeholder of the template: it renders as it is.
        assert render_text("{{ question }}", {"question": "{{ raw }}"}, where="t") == "{{ raw }}"
        assert render_text("{{{ n }}}", fields, where="t") == "{3}"

    def test_template_names(self):
        assert rendered_name("instruction.template.md") == "instruction.md"
        assert rendered_name("solve.template.sh") == "solve.sh"
        assert rendered_name("answer.template") == "answer"
        assert rendered_name("instruction.md") is None
        assert rendered_name(".template.md") is None

    def test_rendering_writes_one_plain_task_per_row(self, tmp_path):
        template = write_template(tmp_path / "qa")
        (template / "dataset.toml").write_text('[gym]\nlicense = "Apache 2.0"\n')
        (template / "task_data.py").write_text(
            "from pydantic import BaseModel\nclass TaskData(BaseModel):\n    question: str\n    answer: str\n"
        )
        out = render_template_folder(template, tmp_path / "out" / "rendered")
        assert sorted(p.name for p in out.iterdir()) == ["dataset.toml", "q-1", "q-2", "task_data.py"]
        task = out / "q-2"
        assert (task / "instruction.md").read_text() == "Solve: 6*7?\nWrite the answer to answer.txt.\n"
        assert not (task / "instruction.template.md").exists()
        assert (task / "solution" / "solve.sh").read_text() == "#!/bin/bash\necho '42' > answer.txt\n"
        assert (task / "solution" / "solve.sh").stat().st_mode & 0o111
        assert (task / "solution" / "notes.txt").read_text() == "not a template: {{ answer }} stays\n"
        assert (task / "tests" / "verifier.py").read_text() == ANSWER_VERIFIER
        assert json.loads((task / TASK_DATA_FILE).read_text()) == {"question": "6*7?", "answer": "42"}
        assert not (task / "tasks.jsonl").exists() and not (task / "task_data.py").exists()
        # Deterministic: a second render of the same template gives byte-identical folders.
        first = {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*") if p.is_file()}
        render_template_folder(template, out)
        assert {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*") if p.is_file()} == first

    def test_rendered_tasks_carry_their_own_harbor_name(self, tmp_path):
        # Harbor keys a dataset on [task].name (`harbor add` keeps one folder per name), so a rendered task cannot
        # carry the template's: it gets `<org>/<task_id>`, and the rest of task.toml byte for byte.
        template = write_template(tmp_path / "qa")
        out = render_template_folder(template, tmp_path / "out")
        for task_id in ("q-1", "q-2"):
            rendered = (out / task_id / "task.toml").read_text()
            assert rendered == HELLO_TOML.replace('name = "harbor/hello-world"', f'name = "harbor/{task_id}"')
            assert tomllib.loads(rendered)["task"]["name"] == f"harbor/{task_id}"
        assert load_task(out / "q-1").config.task == {"name": "harbor/q-1"}
        # The rewrite keeps a trailing comment, and touches no other table's `name`.
        toml = (
            'schema_version = "1.4"\n\n[task]\nname = "org/t"  # label\n\n[task.extra]\nname = "keep"\n'
            '[other]\nname = "keep"\n'
        )
        assert rendered_task_toml(toml, "row-7", where="w") == toml.replace('"org/t"', '"org/row-7"')
        # Without a [task] table nothing is rewritten: Harbor then names the task after its folder.
        plain_toml = 'schema_version = "1.4"\n[environment]\ndocker_image = "python:3.12"\n'
        (template / "task.toml").write_text(plain_toml)
        render_template_folder(template, out)
        assert (out / "q-1" / "task.toml").read_text() == (template / "task.toml").read_text()
        # A name the line-based rewrite cannot reach, a template name that is not org/name and a task_id Harbor would
        # refuse in a name are each refused with the row named, and the previous render stays.
        for toml, task_id, fragment in (
            ('task = { name = "org/t" }\n', "q-1", "could not be rewritten"),
            ('task.name = "org/t"\n', "q-1", "could not be rewritten"),
            ('[task]\nname = "no-org"\n', "q-1", "is not `org/name`"),
            ('[task]\nname = "org/t"\n', "q$1", "cannot be a Harbor task name"),
        ):
            with pytest.raises(ParameterizedError, match=fragment) as info:
                rendered_task_toml(toml, task_id, where=f"task {task_id!r}")
            assert f"task {task_id!r}" in str(info.value)
        (template / "task.toml").write_text('task = { name = "org/t" }\n')
        with pytest.raises(ParameterizedError, match="task 'q-1'.*could not be rewritten"):
            render_template_folder(template, out)
        assert (out / "q-1" / "task.toml").read_text() == plain_toml

    def test_rendering_copies_hidden_task_files_and_skips_what_the_digest_skips(self, tmp_path):
        # environment/.env and .dockerignore are task content (Compose reads .env, Docker .dockerignore; both enter
        # the task digest and reach the sandbox), so a rendered task carries them. The template root's hidden entries
        # (.git, an editor's folder), the digest's ignored names and the root README.md (about the template) are not;
        # the root .gitignore is, since Harbor's packager reads it as the task's ignore rules and publish ships it on.
        template = write_template(tmp_path / "qa")
        (template / "environment" / ".env").write_text("GREETING=hi\n")
        (template / "environment" / ".dockerignore").write_text("*.log\n")
        (template / "tests" / ".fixture").write_text("hidden fixture\n")
        (template / "tests" / ".fixture").chmod(0o600)
        (template / "tests" / "__pycache__").mkdir()
        (template / "tests" / "__pycache__" / "verifier.cpython-313.pyc").write_bytes(b"\x00")
        (template / "environment" / ".DS_Store").write_bytes(b"\x00")
        (template / "solution" / "notes.txt~").write_text("editor leftover\n")
        (template / ".git").mkdir()
        (template / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        (template / ".gitignore").write_text("node_modules/\n.env\n")
        (template / "README.md").write_text("About this template\n")
        (template / "README.template.md").write_text("Task {{ task_data.question }}\n")
        task = render_template_folder(template, tmp_path / "out") / "q-1"
        assert sorted(p.name for p in (task / "environment").iterdir()) == [".dockerignore", ".env", "Dockerfile"]
        assert sorted(p.name for p in (task / "tests").iterdir()) == [".fixture", "verifier.py"]
        assert sorted(p.name for p in (task / "solution").iterdir()) == ["notes.txt", "solve.sh"]
        assert (task / "environment" / ".env").read_text() == "GREETING=hi\n"
        assert (task / "tests" / ".fixture").read_text() == "hidden fixture\n"
        assert stat.S_IMODE((task / "tests" / ".fixture").stat().st_mode) == 0o600
        assert not (task / ".git").exists()
        assert (task / ".gitignore").read_text() == "node_modules/\n.env\n"
        assert (task / "README.md").read_text() == "Task 2+2?\n"
        # Identity: a hand-written task with the same files digests the same as its rendered twin, and the hidden
        # file is part of that identity.
        # The twin carries the name the render gave the row (`harbor/q-1`): the rewritten task.toml is task content.
        twin = write_task(tmp_path / "twin" / "q-1", toml=HELLO_TOML.replace("harbor/hello-world", "harbor/q-1"))
        (twin / "instruction.md").write_text("Solve: 2+2?\nWrite the answer to answer.txt.\n")
        (twin / "README.md").write_text("Task 2+2?\n")
        (twin / "environment" / ".env").write_text("GREETING=hi\n")
        (twin / "environment" / ".dockerignore").write_text("*.log\n")
        (twin / "tests" / "test.sh").unlink()
        (twin / "tests" / "verifier.py").write_text(ANSWER_VERIFIER)
        (twin / "tests" / ".fixture").write_text("hidden fixture\n")
        (twin / "tests" / ".fixture").chmod(0o600)
        (twin / "solution" / "solve.sh").write_text("#!/bin/bash\necho '4' > answer.txt\n")
        (twin / "solution" / "solve.sh").chmod(0o755)
        (twin / "solution" / "notes.txt").write_text("not a template: {{ answer }} stays\n")
        (twin / TASK_DATA_FILE).write_text(json.dumps({"question": "2+2?", "answer": "4"}, sort_keys=True) + "\n")
        assert content_hash(twin) == content_hash(task)
        (twin / "environment" / ".env").unlink()
        assert content_hash(twin) != content_hash(task)

    def test_a_rendered_name_that_collides_with_another_entry_is_refused(self, tmp_path):
        # Whichever sorts first, two entries that would land under one name are an error naming both, not a silent
        # overwrite (a plain expected.txt would otherwise win over expected.template.txt and keep its placeholders).
        template = write_template(tmp_path / "qa")
        (template / "tests" / "expected.txt").write_text("PLAIN\n")
        (template / "tests" / "expected.template.txt").write_text("{{ question }}\n")
        with pytest.raises(ParameterizedError) as info:
            render_template_folder(template, tmp_path / "out")
        message = str(info.value)
        assert message.startswith("task 'q-1': ") and "would both be written as 'expected.txt'" in message
        assert str(template / "tests" / "expected.template.txt") in message
        assert str(template / "tests" / "expected.txt") in message
        assert list((tmp_path / "out").iterdir()) == [] and sorted(p.name for p in tmp_path.iterdir()) == ["out", "qa"]
        (template / "tests" / "expected.template.txt").unlink()
        (template / "tests" / "expected.txt").unlink()
        (template / "instruction.md").write_text("plain\n")
        with pytest.raises(ParameterizedError, match="instruction.md and .*instruction.template.md would both be"):
            render_template_folder(template, tmp_path / "out")
        (template / "instruction.md").unlink()
        (template / "tests" / "data").mkdir()
        (template / "tests" / "data.template").write_text("{{ answer }}")
        with pytest.raises(ParameterizedError, match="would both be written as 'data'"):
            render_template_folder(template, tmp_path / "out")
        (template / "tests" / "data.template").unlink()
        assert (render_template_folder(template, tmp_path / "out") / "q-1" / "tests" / "data").is_dir()
        # The row's fields are written as task_data.json after the tree is rendered, so a template that ships that
        # name at its root (plain or as a template to render) is refused up front rather than copied and replaced.
        for name in ("task_data.json", "task_data.template.json"):
            (template / name).write_text('{"question": "{{ question }}", "units": "cm"}\n')
            with pytest.raises(ParameterizedError) as info:
                render_template_folder(template, tmp_path / "out")
            assert str(info.value).startswith(f"{template / name}: a template has no task_data.json of its own")
            assert "written from its tasks.jsonl row" in str(info.value)
            (template / name).unlink()
        # Below the root the name is task content (a tests/ fixture), copied like any other file.
        (template / "tests" / "task_data.json").write_text('{"fixture": true}\n')
        rendered = render_template_folder(template, tmp_path / "out") / "q-1"
        assert (rendered / "tests" / "task_data.json").read_text() == '{"fixture": true}\n'
        assert json.loads((rendered / "task_data.json").read_text()) == {"answer": "4", "question": "2+2?"}

    def test_rendering_replaces_stale_folders_and_refuses_the_template_itself(self, tmp_path):
        template = write_template(tmp_path / "qa")
        out = render_template_folder(template, tmp_path / "out")
        (out / "stale").mkdir()
        (out / "verifier.py").write_text("stale dataset-level file\n")
        # Prepare's records live in the rendered folder (its dataset root): a re-render keeps them, and their locks.
        for name in ("manifest.toml", "compose-images.json", "built-images.json", "built-images.json.lock"):
            (out / name).write_text(f"record {name}\n")
        render_template_folder(template, out)
        assert not (out / "stale").exists() and not (out / "verifier.py").exists()
        assert {p.name: p.read_text() for p in out.iterdir() if p.is_file()} == {
            name: f"record {name}\n"
            for name in ("manifest.toml", "compose-images.json", "built-images.json", "built-images.json.lock")
        }
        with pytest.raises(ParameterizedError, match="cannot be written inside the template folder"):
            render_template_folder(template, template / "rendered")
        with pytest.raises(ParameterizedError, match="is not a template folder"):
            render_template_folder(tmp_path / "nope", tmp_path / "out2")

    def test_an_identical_re_render_leaves_the_task_folders_in_place(self, tmp_path):
        # The rendered folder is what a server reads: it copies a task folder at seed. A re-render that deleted and
        # rewrote identical folders would open a window in which they do not exist, so an unchanged entry is left
        # alone (same inode), a changed one is swapped in by rename, a dropped row's folder goes, a failed render
        # changes nothing, and the render's hidden sibling never outlives the call.
        template = write_template(tmp_path / "qa")
        (template / "task_data.py").write_text(TASK_DATA_PY)
        out = render_template_folder(template, tmp_path / "out" / "rendered")
        (out / "manifest.toml").write_text("record\n")

        def inodes() -> dict[str, int]:
            return {p.relative_to(out).as_posix(): p.stat().st_ino for p in out.rglob("*")}

        first = inodes()
        assert render_template_folder(template, out) == out
        assert inodes() == first
        # A render that fails (a placeholder with no field) leaves the previous render as it was.
        original = (template / "instruction.template.md").read_text()
        (template / "instruction.template.md").write_text("{{ nope }}\n")
        with pytest.raises(ParameterizedError, match="placeholder"):
            render_template_folder(template, out)
        assert inodes() == first and sorted(p.name for p in out.parent.iterdir()) == [".rendered.lock", "rendered"]
        (template / "instruction.template.md").write_text(original)
        # One row changes its answer, one is new and the row model is edited: those move, q-2 stays where it is.
        rows = [
            {"task_id": "q-1", "task_data": {"question": "2+2?", "answer": "5"}},
            {"task_id": "q-2", "question": "6*7?", "answer": "42"},
            {"task_id": "q-3", "question": "1+1?", "answer": "2"},
        ]
        (template / "tasks.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (template / "task_data.py").write_text(TASK_DATA_PY + "# edited\n")
        render_template_folder(template, out)
        after = inodes()
        assert sorted(p.name for p in out.iterdir()) == ["manifest.toml", "q-1", "q-2", "q-3", "task_data.py"]
        assert {name: after[name] for name in first if name.startswith("q-2")} == {
            name: inode for name, inode in first.items() if name.startswith("q-2")
        }
        assert (out / "q-1" / "solution" / "solve.sh").read_text() == "#!/bin/bash\necho '5' > answer.txt\n"
        assert (out / "q-3" / "instruction.md").read_text().startswith("Solve: 1+1?")
        assert (out / "task_data.py").read_text().endswith("# edited\n")
        assert (out / "manifest.toml").read_text() == "record\n" and after["manifest.toml"] == first["manifest.toml"]
        # A dropped row's folder goes, and nothing of the render's hidden sibling is left beside the output (the
        # install's lock file, which the next render takes again, is the one sibling that stays).
        (template / "tasks.jsonl").write_text(json.dumps(rows[1]) + "\n")
        render_template_folder(template, out)
        assert sorted(p.name for p in out.iterdir()) == ["manifest.toml", "q-2", "task_data.py"]
        assert sorted(p.name for p in out.parent.iterdir()) == [".rendered.lock", "rendered"]

    def test_overlapping_renders_install_under_a_lock_and_keep_a_record_writers_temporary(self, tmp_path, monkeypatch):
        # `gym dataset validate` and `gym eval run` render one template into one folder, possibly at once, while a
        # prepare records into it. The install runs under an exclusive lock on the output's sibling lock file (two
        # first-time renders racing entry by entry met FileExistsError), and a record writer's in-flight temporary
        # (fsynced, about to be moved over its record) is kept with the records: an overlapping render deleted it,
        # and the writer's os.replace met FileNotFoundError.
        template = write_template(tmp_path / "qa")
        out = tmp_path / "out" / "rendered"
        render_template_folder(template, out)
        (out / "manifest.toml").write_text("record\n")
        # The two writers' names: records.write_json_record's and the manifest's.
        temporaries = [out / f".compose-images.json.{os.getpid()}.tmp", out / f".manifest-{os.getpid()}-1-ab12.tmp"]
        for temporary in temporaries:
            temporary.write_text("in flight\n")
        (out / "stale").mkdir()
        lock_path = out.with_name(".rendered.lock")
        held: list[bool] = []
        install = parameterized._install_rendered

        def observed_install(fresh: Path, output: Path) -> None:
            # Another renderer's view while this install runs: the lock cannot be taken (flock is per open file
            # description, so a second descriptor of this process is refused like another process would be).
            fd = os.open(lock_path, os.O_RDWR)
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    held.append(True)
                else:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    held.append(False)
            finally:
                os.close(fd)
            install(fresh, output)

        monkeypatch.setattr(parameterized, "_install_rendered", observed_install)
        render_template_folder(template, out)
        assert held == [True]
        assert all(temporary.read_text() == "in flight\n" for temporary in temporaries)
        assert not (out / "stale").exists() and (out / "manifest.toml").read_text() == "record\n"
        assert sorted(p.name for p in out.iterdir() if not p.name.startswith(".")) == ["manifest.toml", "q-1", "q-2"]
        # Released once the install is done: the next renderer takes it.
        fd = os.open(lock_path, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def test_a_missing_placeholder_field_names_the_task_and_file(self, tmp_path):
        template = write_template(tmp_path / "qa", rows=[{"task_id": "q-1", "task_data": {"answer": "4"}}])
        with pytest.raises(ParameterizedError, match="task 'q-1', instruction.template.md: placeholder"):
            render_template_folder(template, tmp_path / "out")

    def test_prepare_renders_a_template_target_and_rows_carry_the_fields(self, tmp_path):
        template = write_template(tmp_path / "qa")
        (template / "task_data.py").write_text(
            "from pydantic import BaseModel\nclass TaskData(BaseModel):\n    question: str\n    answer: str\n"
        )
        prepared = prepare_target(str(template), output_root=tmp_path / "out")
        assert prepared.taskset == "qa"
        assert prepared.folder == (tmp_path / "out" / "qa" / "rendered").resolve()
        assert [t.task_id for t in prepared.tasks] == ["q-1", "q-2"]
        assert all(t.python_verifier == t.path / "tests" / "verifier.py" and t.has_solution for t in prepared.tasks)
        rows = [json.loads(line) for line in prepared.rows_path.read_text().splitlines()]
        # The rows are labelled by the template, as a plain dataset's are by its folder, not by the shared `rendered`
        # folder they were loaded from.
        assert prepared.task_source == "qa" and {row["task_source"] for row in rows} == {"qa"}
        assert rows[0]["task_input"]["task_data"] == {
            "question": "2+2?",
            "answer": "4",
            DIGEST_KEY: prepared.tasks[0].digest,
            "_ng_dataset_digest": prepared.dataset_digest,
        }
        assert rows[1]["task_input"]["responses_create_params"]["input"][0]["content"].startswith("Solve: 6*7?")
        # The template's TaskData types the rows: a row the model rejects refuses the dataset at prepare.
        (template / "tasks.jsonl").write_text('{"task_id": "q-1", "question": "2+2?", "answer": 4}\n')
        with pytest.raises(HarborTaskError, match="task 'q-1': task_data does not satisfy TaskData"):
            prepare_target(str(template), output_root=tmp_path / "out")

    def test_a_second_prepare_of_a_template_reads_the_records_the_first_left(self, tmp_path):
        # The rendered folder is the dataset root prepare records into; wiping it on every prepare would rebuild
        # and re-push a template's image, and re-resolve its base images, on every run.
        template = write_template(tmp_path / "qa")
        prepared = prepare_target(str(template), output_root=tmp_path / "out")
        rendered = prepared.folder
        configs = read_image_configs(rendered)
        assert "ubuntu:24.04" in configs, configs
        built = {
            "abc123": {"image": "registry.invalid/hb__abc123", "tasks": ["q-1"], "tool": "buildah", "pushed": True}
        }
        (rendered / "built-images.json").write_text(json.dumps(built))
        manifest = (rendered / "manifest.toml").read_text()
        again = prepare_target(str(template), output_root=tmp_path / "out")
        assert again.folder == rendered and [t.task_id for t in again.tasks] == ["q-1", "q-2"]
        assert read_built_images(rendered) == built
        assert read_image_configs(rendered) == configs
        assert (rendered / "manifest.toml").read_text() == manifest and again.dataset_digest == prepared.dataset_digest

    def test_a_task_id_may_use_the_whole_folder_name_limit(self, tmp_path, monkeypatch):
        # 255 bytes is NAME_MAX on Linux and macOS: a task_id of that length renders, re-renders (the aside a changed
        # folder is renamed to does not grow the name) and publishes; one byte more is refused by line in read_rows,
        # and in bytes, not characters. What read_rows cannot know (a filesystem with a shorter limit, a full disk)
        # names tasks.jsonl and the task, never the hidden staging path.
        long_id = "x" * 255
        row = {"task_id": long_id, "question": "1+1?", "answer": "2"}
        template = write_template(tmp_path / "qa", rows=[row])
        out = render_template_folder(template, tmp_path / "out" / "rendered")
        assert (out / long_id / "task_data.json").is_file() and sorted(p.name for p in out.iterdir()) == [long_id]
        (template / "tasks.jsonl").write_text(json.dumps({**row, "question": "1+2?"}) + "\n")
        assert render_template_folder(template, out) == out
        assert "1+2?" in (out / long_id / "instruction.md").read_text()
        assert sorted(p.name for p in out.iterdir()) == [long_id] and sorted(p.name for p in out.parent.iterdir()) == [
            ".rendered.lock",
            "rendered",
        ]
        (template / "tasks.jsonl").write_text(json.dumps({**row, "task_id": "\u00e9" * 128}) + "\n")
        with pytest.raises(
            ParameterizedError, match="tasks.jsonl:1: task_id '\u00e9{40}'... is 256 bytes long; a task folder"
        ):
            read_rows(template)
        (template / "tasks.jsonl").write_text(json.dumps(row) + "\n")
        real_mkdir = Path.mkdir

        def short_name_limit(self, *args, **kwargs):
            if self.name == long_id:
                raise OSError(36, "File name too long", str(self))
            return real_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", short_name_limit)
        with pytest.raises(ParameterizedError) as excinfo:
            render_template_folder(template, tmp_path / "out2")
        assert str(excinfo.value) == (
            f"{template / 'tasks.jsonl'}: task '{'x' * 40}': its folder cannot be created: File name too long"
        )
        assert list((tmp_path / "out2").iterdir()) == [] and not any(".render-" in p.name for p in tmp_path.iterdir())

    def test_validate_and_eval_run_render_a_template_into_one_folder(self, tmp_path):
        # `gym dataset validate` keeps its rows and results under results/harbor/validation/ but renders a template
        # where `gym eval run` does: the rendered folder is the dataset root prepare records into (manifest.toml,
        # built-images.json, compose-images.json), so two copies would build and push a template's image once per
        # command, where a plain dataset's records live in its own folder and serve both.
        template = write_template(tmp_path / "qa")
        out = tmp_path / "out"
        validated = prepare_target(str(template), output_root=out / "validation", render_root=out)
        assert validated.folder == (out / "qa" / "rendered").resolve()
        assert validated.output_dir == out / "validation" / "qa" and validated.rows_path.is_file()
        assert not (out / "validation" / "qa" / "rendered").exists()
        built = {
            "abc": {"image": "registry.invalid/hb__abc", "tasks": ["q-1", "q-2"], "tool": "buildah", "pushed": True}
        }
        (validated.folder / "built-images.json").write_text(json.dumps(built))
        manifest = (validated.folder / "manifest.toml").read_text()
        inodes = {p.name: p.stat().st_ino for p in validated.folder.iterdir()}
        evaluated = prepare_target(str(template), output_root=out)
        assert evaluated.folder == validated.folder and evaluated.output_dir == out / "qa"
        assert read_built_images(evaluated.folder) == built
        assert (evaluated.folder / "manifest.toml").read_text() == manifest
        assert evaluated.dataset_digest == validated.dataset_digest
        assert {p.name: p.stat().st_ino for p in evaluated.folder.iterdir()} == inodes

    def test_the_example_template_renders(self, tmp_path):
        out = render_template_folder(EXAMPLES / "hello_parameterized_tasks", tmp_path / "rendered")
        assert sorted(p.name for p in out.iterdir() if p.is_dir()) == ["addition", "multiplication"]
        assert "What is 17 + 25?" in (out / "addition" / "instruction.md").read_text()
        # The reference answer is a rendered text file, never a value inside a quoted shell literal.
        assert (out / "addition" / "solution" / "answer.md").read_text() == "42\n"
        assert not (out / "addition" / "solution" / "solve.sh").exists()
        assert json.loads((out / "multiplication" / TASK_DATA_FILE).read_text()) == {
            "question": "What is 9 multiplied by 7?",
            "expected_answer": "63",
        }
        # The RFC's shape: the template ships dataset.toml and task_data.py beside tasks.jsonl, and the rendered
        # dataset carries both, so the rows are typed at prepare.
        assert (out / "dataset.toml").is_file() and (out / "task_data.py").is_file()

    def test_the_example_template_types_its_rows(self, tmp_path):
        import shutil

        example = tmp_path / "hello_parameterized_tasks"
        shutil.copytree(EXAMPLES / "hello_parameterized_tasks", example)
        prepared = prepare_target(str(example), output_root=tmp_path / "out")
        assert [t.task_id for t in prepared.tasks] == ["addition", "multiplication"]
        assert all(t.python_verifier == t.path / "tests" / "verifier.py" and t.has_solution for t in prepared.tasks)
        rows = [json.loads(line) for line in prepared.rows_path.read_text().splitlines()]
        assert rows[1]["task_input"]["task_data"]["expected_answer"] == "63"
        # A number where TaskData wants a string, and an unknown key: each refuses the dataset, naming the task.
        rows_file = example / "tasks.jsonl"
        good = rows_file.read_text()
        rows_file.write_text(good.replace('"expected_answer":"63"', '"expected_answer":63'))
        with pytest.raises(HarborTaskError, match="task 'multiplication': task_data does not satisfy TaskData"):
            prepare_target(str(example), output_root=tmp_path / "out")
        rows_file.write_text(good.replace('"expected_answer":"42"', '"expected_answer":"42","hint":"add"'))
        with pytest.raises(HarborTaskError, match=r"task 'addition': task_data has keys \['hint'\] that TaskData"):
            prepare_target(str(example), output_root=tmp_path / "out")


class TestPythonVerifierDetection:
    def test_test_sh_wins_over_a_verifier_py_beside_it(self, tmp_path):
        # The hub's helper shape (multimedia-terminalbench, osworld-verified, o11y-bench, ...): tests/test.sh runs
        # tests/verifier.py itself. The task is a test.sh task; the module is an ordinary test file.
        folder = write_task(tmp_path / "ds" / "a")
        assert python_verifier_for(folder, tmp_path / "ds") is None
        (folder / "tests" / "test.sh").write_text(
            "#!/bin/bash\npython3 /tests/verifier.py > /logs/verifier/reward.txt\n"
        )
        (folder / "tests" / "verifier.py").write_text("print(1.0)\n")
        assert python_verifier_for(folder, tmp_path / "ds") is None
        task = load_task(folder)
        assert task.python_verifier is None
        # A dataset-level verifier.py changes nothing for it either.
        (tmp_path / "ds" / "verifier.py").write_text(ANSWER_VERIFIER)
        assert load_task(folder).python_verifier is None
        assert load_task(folder).digest == load_task(folder).folder_digest
        # Without test.sh the module is the grader.
        (folder / "tests" / "test.sh").unlink()
        (folder / "tests" / "verifier.py").write_text(ANSWER_VERIFIER)
        assert python_verifier_for(folder, tmp_path / "ds") == folder / "tests" / "verifier.py"
        assert load_task(folder).python_verifier == folder / "tests" / "verifier.py"

    def test_dataset_verifier_grades_tasks_without_tests_and_enters_their_digest(self, tmp_path):
        root = tmp_path / "ds"
        folder = write_task(root / "a")
        import shutil

        shutil.rmtree(folder / "tests")
        with pytest.raises(HarborTaskError, match="has no tests/ folder .and the dataset has no verifier.py"):
            load_task(folder)
        (root / "verifier.py").write_text(ANSWER_VERIFIER)
        task = load_task(folder)
        assert task.python_verifier == root / "verifier.py"
        assert task.digest != task.folder_digest
        (root / "verifier.py").write_text(ANSWER_VERIFIER + "\n# changed\n")
        assert load_task(folder).digest != task.digest
        assert load_task(folder).folder_digest == task.folder_digest
        # A task with its own tests/ keeps grading with them; the dataset verifier does not enter its digest.
        other = write_task(root / "b")
        assert load_task(other).python_verifier is None and load_task(other).digest == load_task(other).folder_digest

    def test_a_tests_folder_with_neither_script_is_left_to_test_sh(self, tmp_path):
        # Harbor admits the shape when a separate verifier image ships /tests/test.sh; the loader keeps admitting
        # it and such a task fails at verification, not at load, when nothing ships the script.
        folder = write_task(tmp_path / "ds" / "a")
        (folder / "tests" / "test.sh").unlink()
        (folder / "tests" / "fixtures.txt").write_text("data\n")
        assert python_verifier_for(folder, tmp_path / "ds") is None
        (tmp_path / "ds" / "verifier.py").write_text(ANSWER_VERIFIER)
        # The dataset verifier grades tasks without a tests/ folder only; this task has one.
        assert python_verifier_for(folder, tmp_path / "ds") is None
        assert load_task(folder).python_verifier is None

    def test_declared_arguments_are_read_without_importing(self, tmp_path):
        module = tmp_path / "verifier.py"
        module.write_text(RESPONSE_VERIFIER)
        assert declared_arguments(module) == {"task_data", "response"} and reads_response(module)
        module.write_text(
            "from functools import partial\nfrom nemo_gym.verifiers.files import text_file_equals\nverify = partial(text_file_equals, path='a', content='b')\n"
        )
        assert declared_arguments(module) is None and not reads_response(module)
        module.write_text("def verify(:\n")
        with pytest.raises(PythonVerifierError):
            declared_arguments(module)


class TestShimAndRunner:
    def _run(self, shim_dir: Path, workspace: Path, reward_dir: Path, *, verifier: Path | None = None, response=False):
        argv = [
            "--verifier",
            str(verifier or shim_dir / ".ng" / "verifier.py"),
            "--workspace",
            str(workspace),
            "--task-data",
            str(shim_dir / ".ng" / "task_data.json"),
            "--reward-dir",
            str(reward_dir),
        ]
        if response:
            argv += ["--response", str(shim_dir / ".ng" / "response.json")]
        # A fresh interpreter started the way test.sh starts the runner (`exec "$PY" /tests/.ng/run_verifier.py`, a
        # plain script, no -I): sys.path[0] is the shim's .ng folder and the host's own nemo_gym is importable from
        # site-packages, so the runner's ordering alone decides that the vendored tree provides `nemo_gym`.
        result = subprocess.run(
            [sys.executable, str(shim_dir / ".ng" / "run_verifier.py"), *argv],
            capture_output=True,
            text=True,
            cwd=str(workspace),
        )
        return result

    def test_shim_layout_and_test_script(self, tmp_path):
        verifier = tmp_path / "verifier.py"
        verifier.write_text(ANSWER_VERIFIER)
        shim = VerifierShim(verifier=verifier, in_task_tests=False, workspace="/workspace", task_data={"answer": "4"})
        out = write_shim(tmp_path / "tests", shim)
        files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
        assert files == [
            ".ng/nemo_gym/__init__.py",
            ".ng/nemo_gym/verifiers/__init__.py",
            ".ng/nemo_gym/verifiers/files.py",
            ".ng/nemo_gym/verifiers/runner.py",
            ".ng/run_verifier.py",
            ".ng/task_data.json",
            ".ng/verifier.py",
            "test.sh",
        ]
        script = (out / "test.sh").read_text()
        assert script.startswith("#!/usr/bin/env bash\n") and (out / "test.sh").stat().st_mode & 0o111
        assert (
            "--verifier /tests/.ng/verifier.py --workspace /workspace --task-data /tests/.ng/task_data.json" in script
        )
        assert "--reward-dir /logs/verifier" in script and "--response" not in script
        assert json.loads((out / ".ng" / "task_data.json").read_text()) == {"answer": "4"}
        # The vendored package never imports the real one: an empty nemo_gym/__init__.py shadows it.
        assert (out / ".ng" / "nemo_gym" / "__init__.py").read_text() == ""
        in_tests = VerifierShim(
            verifier=verifier, in_task_tests=True, workspace=None, task_data={}, response={"output": []}
        )
        script = shim_test_script(in_tests)
        assert (
            '--verifier /tests/verifier.py --workspace "$PWD"' in script
            and "--response /tests/.ng/response.json" in script
        )

    def test_the_shim_is_readable_by_a_non_root_verifier_whatever_the_umask(self, tmp_path):
        # Gym's upload and Harbor's `docker cp` keep the archive's modes and chown nothing, and test.sh runs as the
        # task's [verifier].user: a shim generated under umask 077 (0700 folders, 0600 files) would be unreadable to a
        # non-root verifier user, and grading would fail before verify() ran. The task's own files keep their modes.
        verifier = tmp_path / "verifier.py"
        verifier.write_text(ANSWER_VERIFIER)
        verifier.chmod(0o600)
        previous = os.umask(0o077)
        try:
            out = write_shim(
                tmp_path / "tests",
                VerifierShim(verifier=verifier, in_task_tests=False, workspace=None, task_data={}, response={}),
            )
        finally:
            os.umask(previous)
        modes = {p.relative_to(out).as_posix(): stat.S_IMODE(p.stat().st_mode) for p in out.rglob("*")}
        assert modes["test.sh"] == 0o755
        folders = {name for name, mode in modes.items() if (out / name).is_dir()}
        assert folders == {".ng", ".ng/nemo_gym", ".ng/nemo_gym/verifiers"}
        assert all(modes[name] == 0o755 for name in folders)
        files = set(modes) - folders - {"test.sh"}
        assert {".ng/run_verifier.py", ".ng/verifier.py", ".ng/task_data.json", ".ng/response.json"} <= files
        assert all(modes[name] == 0o644 for name in files), modes
        # The tests/ folder write_shim created here is readable too; the author's module is untouched: the shim's
        # copy is readable, the source keeps its own mode.
        assert stat.S_IMODE(out.stat().st_mode) == 0o755
        assert stat.S_IMODE(verifier.stat().st_mode) == 0o600

    def test_publish_under_umask_077_leaves_the_tests_folder_it_creates_readable(self, tmp_path):
        # publish creates tests/ for a task graded by the dataset's verifier.py (the task ships none), and Harbor
        # copies the published folder into /tests modes and all: under umask 077 that folder was 0700, a root-owned
        # door shut above the readable shim for a non-root [verifier].user. A tests/ the author shipped keeps the
        # mode they gave it, as the rest of the task does.
        root = tmp_path / "ds"
        created = write_task(root / "created")
        shutil.rmtree(created / "tests")
        (created / "solution" / "solve.sh").unlink()
        (created / "solution" / "answer.md").write_text("4\n")
        (created / "task_data.json").write_text('{"answer": "4"}')
        shipped = write_task(root / "shipped", toml=HELLO_TOML.replace("harbor/hello-world", "harbor/shipped"))
        (shipped / "tests" / "test.sh").unlink()
        (shipped / "tests" / "verifier.py").write_text(ANSWER_VERIFIER)
        (shipped / "tests").chmod(0o750)
        (root / "verifier.py").write_text(ANSWER_MD_VERIFIER)
        previous = os.umask(0o077)
        try:
            report = publish_harbor(root, tmp_path / "pub")
        finally:
            os.umask(previous)
        assert sorted(report.wrapped_verifiers) == ["created", "shipped"]

        def mode(path: Path) -> int:
            return stat.S_IMODE(path.stat().st_mode)

        tests = tmp_path / "pub" / "created" / "tests"
        assert mode(tests) == 0o755
        assert mode(tests / ".ng") == 0o755 and mode(tests / "test.sh") == 0o755
        assert mode(tests / ".ng" / "verifier.py") == 0o644 and mode(tests / ".ng" / "run_verifier.py") == 0o644
        shipped_tests = tmp_path / "pub" / "shipped" / "tests"
        assert mode(shipped_tests) == 0o750 and mode(shipped_tests / ".ng") == 0o755
        assert "--verifier /tests/verifier.py" in (shipped_tests / "test.sh").read_text()

    def test_runner_grades_the_workspace_with_the_vendored_helpers(self, tmp_path):
        verifier = tmp_path / "verifier.py"
        verifier.write_text(
            "from nemo_gym.verifiers.files import read_text, text_file_equals\n"
            "def verify(workspace, task_data, state):\n"
            "    assert state['workspace'] == workspace\n"
            "    return text_file_equals(workspace, path='answer.txt', content=task_data['answer'])\n"
        )
        out = write_shim(
            tmp_path / "tests",
            VerifierShim(verifier=verifier, in_task_tests=False, workspace=None, task_data={"answer": "4"}),
        )
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "answer.txt").write_text("4\n")
        result = self._run(out, workspace, tmp_path / "reward")
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "reward" / "reward.txt").read_text() == "1.0\n"
        (workspace / "answer.txt").write_text("5\n")
        self._run(out, workspace, tmp_path / "reward2")
        assert (tmp_path / "reward2" / "reward.txt").read_text() == "0.0\n"

    def test_runner_passes_the_response_and_writes_metric_dicts(self, tmp_path):
        verifier = tmp_path / "verifier.py"
        verifier.write_text(
            "from nemo_gym.verifiers import final_text\n"
            "def verify(task_data, response):\n"
            "    ok = final_text(response).strip() == task_data['answer']\n"
            "    return {'reward': float(ok), 'length': float(len(final_text(response)))}\n"
        )
        response = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "72"}]}]}
        out = write_shim(
            tmp_path / "tests",
            VerifierShim(
                verifier=verifier, in_task_tests=False, workspace=None, task_data={"answer": "72"}, response=response
            ),
        )
        workspace = tmp_path / "ws"
        workspace.mkdir()
        result = self._run(out, workspace, tmp_path / "reward", response=True)
        assert result.returncode == 0, result.stderr
        assert json.loads((tmp_path / "reward" / "reward.json").read_text()) == {"reward": 1.0, "length": 2.0}

    def test_the_vendored_helpers_win_over_a_nemo_gym_in_the_tasks_tests(self, tmp_path, monkeypatch):
        # test.sh starts the runner as a plain script, so the interpreter puts /tests/.ng on sys.path itself; the
        # runner must still rank it ahead of the verifier's folder and of /tests (the task's own tests/, uploaded
        # whole), or a nemo_gym package shipped in, or planted into, tests/ replaces the vendored helpers. Both module
        # locations: the task's tests/verifier.py and a dataset-level module carried under .ng/.
        planted = "def exact_match(a, b):\n    return 42.0\n"
        source = (
            "import nemo_gym.verifiers as helpers\nfrom nemo_gym.verifiers import exact_match\n"
            "def verify(workspace, task_data):\n"
            "    print('helpers from', helpers.__file__)\n"
            "    return exact_match((workspace / 'answer.md').read_text().strip(), task_data['answer'])\n"
        )
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "answer.md").write_text("4\n")
        for in_task_tests in (True, False):
            folder = tmp_path / ("task" if in_task_tests else "dataset")
            tests = folder / "tests"
            (tests / "nemo_gym" / "verifiers").mkdir(parents=True)
            (tests / "nemo_gym" / "__init__.py").write_text("")
            (tests / "nemo_gym" / "verifiers" / "__init__.py").write_text(planted)
            verifier = tests / "verifier.py" if in_task_tests else folder / "verifier.py"
            verifier.write_text(source)
            shim = VerifierShim(
                verifier=verifier, in_task_tests=in_task_tests, workspace=None, task_data={"answer": "4"}
            )
            write_shim(tests, shim)
            result = self._run(tests, workspace, folder / "reward", verifier=verifier if in_task_tests else None)
            assert result.returncode == 0, result.stderr
            assert f"helpers from {tests / '.ng' / 'nemo_gym' / 'verifiers' / '__init__.py'}" in result.stdout
            assert (folder / "reward" / "reward.txt").read_text() == "1.0\n"
        # The ordering itself, in process: an entry the interpreter already placed (sys.path[0] as a script, or one
        # from PYTHONPATH) is moved to its rank, never left ahead of a higher one.
        ng, tests = tmp_path / "task" / "tests" / ".ng", tmp_path / "task" / "tests"
        monkeypatch.setattr(sys, "path", [str(ng), str(tests), "/site-packages", str(tests / "sub")])
        place_on_path(ng, tests / "sub", tests)
        assert sys.path == [str(ng), str(tests / "sub"), str(tests), "/site-packages"]
        monkeypatch.setattr(sys, "path", [str(tests), "/site-packages"])
        place_on_path(ng, tests, tests)
        assert sys.path == [str(ng), str(tests), "/site-packages"]

    def test_the_shim_refuses_a_python_older_than_the_floor(self, tmp_path):
        # The runner and the vendored helpers run under the image's own python3 and keep to Python 3.8 (no 3.9+
        # syntax or string methods: ubuntu:20.04's python3 is 3.8); test.sh checks that floor and says so, instead
        # of letting verify() die in a SyntaxError or AttributeError with no reward and no hint at the cause.
        assert PYTHON_FLOOR == (3, 8)
        for module in sorted(Path(verifiers_package.__file__).parent.glob("*.py")):
            source = module.read_text()
            ast.parse(source, filename=str(module), feature_version=PYTHON_FLOOR)
            assert re.search(r"\.(removesuffix|removeprefix|is_relative_to)\(", source) is None, module.name
        script = shim_test_script(
            VerifierShim(verifier=tmp_path / "v.py", in_task_tests=False, workspace=None, task_data={})
        )
        lines = script.splitlines()
        lookup = next(i for i, line in enumerate(lines) if line.startswith("PY=$(command -v python3"))
        check = lines.index(
            '"$PY" -c \'import sys; sys.exit(sys.version_info < (3, 8))\' 2>/dev/null || { echo "the verifier needs '
            'Python 3.8 or newer, found $("$PY" --version 2>&1) at $PY" >&2; exit 127; }'
        )
        assert lookup < check < next(i for i, line in enumerate(lines) if line.startswith('exec "$PY" /tests/.ng/'))
        old = tmp_path / "python3"
        old.write_text(
            '#!/bin/sh\ncase "$*" in *version_info*) exit 1;; --version) echo "Python 3.6.9"; exit 0;; esac\nexit 0\n'
        )
        old.chmod(0o755)

        def run(python):
            return subprocess.run(
                ["bash", "-c", f"PY={shlex.quote(str(python))}\n{lines[check]}\necho passed"],
                capture_output=True,
                text=True,
            )

        refused = run(old)
        assert (refused.returncode, refused.stdout) == (127, "")
        assert refused.stderr.strip() == f"the verifier needs Python 3.8 or newer, found Python 3.6.9 at {old}"
        accepted = run(sys.executable)
        assert (accepted.returncode, accepted.stdout.strip(), accepted.stderr) == (0, "passed", "")

    def test_planted_reward_files_never_survive_the_shim_or_the_runner(self, tmp_path):
        # Harbor shares /logs/verifier with the agent's container for the whole run and reads reward.json before
        # reward.txt; Gym's server empties the folder before every upload, Harbor never does. So the shim clears both
        # files before the runner starts, and the runner again before verify() runs: the verdict is this run's alone.
        script = shim_test_script(
            VerifierShim(verifier=tmp_path / "v.py", in_task_tests=False, workspace=None, task_data={})
        )
        lines = script.splitlines()
        cleared = lines.index("rm -rf /logs/verifier/reward.json /logs/verifier/reward.txt")
        python_lookup = next(i for i, line in enumerate(lines) if line.startswith("PY=$(command -v python3"))
        assert lines.index("mkdir -p /logs/verifier") < cleared < python_lookup

        def plant(reward_dir: Path) -> None:
            reward_dir.mkdir(parents=True, exist_ok=True)
            (reward_dir / "reward.json").write_text('{"reward": 1.0}\n')
            (reward_dir / "reward.txt").write_text("1.0\n")

        workspace = tmp_path / "ws"
        workspace.mkdir()
        verifier = tmp_path / "verifier.py"
        verifier.write_text("def verify(workspace):\n    return 0.0\n")
        shim = VerifierShim(verifier=verifier, in_task_tests=False, workspace=None, task_data={})
        out = write_shim(tmp_path / "tests", shim)
        plant(tmp_path / "r1")
        result = self._run(out, workspace, tmp_path / "r1")
        assert result.returncode == 0, result.stderr
        # Harbor would have read the planted reward.json first; only the honest reward.txt is left.
        assert sorted(p.name for p in (tmp_path / "r1").iterdir()) == ["reward.txt"]
        assert (tmp_path / "r1" / "reward.txt").read_text() == "0.0\n"
        # verify() raising leaves no reward at all (the missing-reward outcome), not the planted 1.0.
        verifier.write_text("def verify(workspace):\n    raise RuntimeError('boom')\n")
        write_shim(tmp_path / "tests", shim)
        plant(tmp_path / "r2")
        result = self._run(out, workspace, tmp_path / "r2")
        assert result.returncode == 1 and "boom" in result.stderr and list((tmp_path / "r2").iterdir()) == []
        # A folder or a link planted under the names goes too; a link's target is left alone.
        elsewhere = tmp_path / "elsewhere.txt"
        elsewhere.write_text("1.0\n")
        r3 = tmp_path / "r3"
        (r3 / "reward.json").mkdir(parents=True)
        (r3 / "reward.txt").symlink_to(elsewhere)
        verifier.write_text("def verify(workspace):\n    return 0.5\n")
        write_shim(tmp_path / "tests", shim)
        result = self._run(out, workspace, r3)
        assert result.returncode == 0, result.stderr
        assert sorted(p.name for p in r3.iterdir()) == ["reward.txt"] and not (r3 / "reward.txt").is_symlink()
        assert (r3 / "reward.txt").read_text() == "0.5\n" and elsewhere.read_text() == "1.0\n"

    def test_runner_reports_contract_problems_without_a_reward(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        verifier = tmp_path / "verifier.py"
        verifier.write_text("def verify(workspace, secret):\n    return 1.0\n")
        out = write_shim(
            tmp_path / "tests", VerifierShim(verifier=verifier, in_task_tests=False, workspace=None, task_data={})
        )
        result = self._run(out, workspace, tmp_path / "r1")
        assert result.returncode == 2 and "asks for `secret`" in result.stderr and not (tmp_path / "r1").exists()
        verifier.write_text("def verify(workspace):\n    return 'yes'\n")
        write_shim(
            tmp_path / "tests", VerifierShim(verifier=verifier, in_task_tests=False, workspace=None, task_data={})
        )
        result = self._run(out, workspace, tmp_path / "r2")
        assert result.returncode == 2 and "must return a number or a dict" in result.stderr
        verifier.write_text("def verify(workspace):\n    raise ValueError('boom')\n")
        write_shim(
            tmp_path / "tests", VerifierShim(verifier=verifier, in_task_tests=False, workspace=None, task_data={})
        )
        result = self._run(out, workspace, tmp_path / "r3")
        assert result.returncode == 1 and "verify() raised" in result.stderr and "boom" in result.stderr
        verifier.write_text("verify = 3\n")
        write_shim(
            tmp_path / "tests", VerifierShim(verifier=verifier, in_task_tests=False, workspace=None, task_data={})
        )
        assert self._run(out, workspace, tmp_path / "r4").returncode == 2
        # A verifier that requires `response` where there is none (every published Harbor task) is a contract error,
        # not a call with response=None that scores 0 quietly; with a default it is left to the default.
        verifier.write_text("def verify(task_data, response):\n    return 0.0 if response is None else 1.0\n")
        write_shim(
            tmp_path / "tests", VerifierShim(verifier=verifier, in_task_tests=False, workspace=None, task_data={})
        )
        result = self._run(out, workspace, tmp_path / "r5")
        assert result.returncode == 2 and "asks for `response`" in result.stderr and not (tmp_path / "r5").exists()
        verifier.write_text("def verify(task_data, response=None):\n    return 0.5 if response is None else 1.0\n")
        write_shim(
            tmp_path / "tests", VerifierShim(verifier=verifier, in_task_tests=False, workspace=None, task_data={})
        )
        result = self._run(out, workspace, tmp_path / "r6")
        assert result.returncode == 0 and (tmp_path / "r6" / "reward.txt").read_text() == "0.5\n"

    def test_runner_awaits_an_async_verifier(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "a.txt").write_text("x")
        verifier = tmp_path / "verifier.py"
        verifier.write_text(
            "import asyncio\n"
            "async def verify(workspace, task_data):\n"
            "    await asyncio.sleep(0)\n"
            "    return {'reward': float((workspace / 'a.txt').is_file()), 'k': task_data['k']}\n"
        )
        out = write_shim(
            tmp_path / "tests",
            VerifierShim(verifier=verifier, in_task_tests=False, workspace=None, task_data={"k": 2}),
        )
        result = self._run(out, workspace, tmp_path / "r")
        assert result.returncode == 0, result.stderr
        assert "coroutine" not in result.stderr
        assert json.loads((tmp_path / "r" / "reward.json").read_text()) == {"reward": 1.0, "k": 2.0}

    def test_runner_passes_positional_only_parameters_by_position(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "a.txt").write_text("x")
        verifier = tmp_path / "verifier.py"
        verifier.write_text(
            "def verify(workspace, state, /, task_data, *, response=None):\n"
            "    assert state['workspace'] == workspace and response is None\n"
            "    return float((workspace / 'a.txt').is_file() and task_data == {'k': 1})\n"
        )
        out = write_shim(
            tmp_path / "tests",
            VerifierShim(verifier=verifier, in_task_tests=False, workspace=None, task_data={"k": 1}),
        )
        result = self._run(out, workspace, tmp_path / "r")
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "r" / "reward.txt").read_text() == "1.0\n"
        # A positional-only parameter the runner does not know is still the contract error, not a TypeError.
        verifier.write_text("def verify(workspace, secret, /):\n    return 1.0\n")
        write_shim(
            tmp_path / "tests",
            VerifierShim(verifier=verifier, in_task_tests=False, workspace=None, task_data={}),
        )
        result = self._run(out, workspace, tmp_path / "r2")
        assert result.returncode == 2 and "asks for `secret`" in result.stderr and not (tmp_path / "r2").exists()

    def test_runner_main_in_process_with_partial_verifier(self, tmp_path, monkeypatch):
        # The runner puts the shim's .ng (with its vendored nemo_gym) first on sys.path; a later test's spawned
        # worker inherits sys.path and would import that nemo_gym instead of the real one.
        monkeypatch.setattr(sys, "path", list(sys.path))
        verifier = tmp_path / "verifier.py"
        verifier.write_text(
            "from functools import partial\nfrom nemo_gym.verifiers.files import text_file_equals\n"
            "verify = partial(text_file_equals, path='hello.txt', content='hi')\n"
        )
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "hello.txt").write_text("hi\n")
        task_data = tmp_path / "td.json"
        task_data.write_text("{}")
        rc = run_verifier(
            [
                "--verifier",
                str(verifier),
                "--workspace",
                str(workspace),
                "--task-data",
                str(task_data),
                "--reward-dir",
                str(tmp_path / "r"),
            ]
        )
        assert rc == 0 and (tmp_path / "r" / "reward.txt").read_text() == "1.0\n"
        # A partial that requires `response` cannot be seen through on the host (publish lets it through), so the
        # runner is where it fails: loudly, with no reward, instead of grading response=None to 0.0.
        verifier.write_text(
            "from functools import partial\n"
            "def _grade(task_data, response, expected):\n"
            "    return float(response is not None and response['text'] == expected)\n"
            'verify = partial(_grade, expected="72")\n'
        )
        assert reads_response(verifier) is False
        argv = ["--verifier", str(verifier), "--workspace", str(workspace), "--reward-dir", str(tmp_path / "r2")]
        assert run_verifier(argv) == 2 and not (tmp_path / "r2").exists()
        (tmp_path / "resp.json").write_text('{"text": "72"}')
        assert run_verifier([*argv, "--response", str(tmp_path / "resp.json")]) == 0
        assert (tmp_path / "r2" / "reward.txt").read_text() == "1.0\n"


class TestVerifierHelpers:
    def test_final_text_reads_the_last_message(self):
        response = {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "first"}]},
                {"type": "function_call", "name": "x"},
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "The answer is "},
                        {"type": "output_text", "text": "72."},
                    ],
                },
            ]
        }
        assert final_text(response) == "The answer is 72."
        assert final_text({"output": []}) == "" and final_text(None) == ""
        assert final_text({"output": [{"type": "message", "content": "plain"}]}) == "plain"

    def test_matches(self):
        assert (
            exact_match(" 4\n", "4") == 1.0
            and exact_match("A", "a") == 0.0
            and exact_match("A", "a", case_sensitive=False) == 1.0
        )
        assert numeric_match("The answer is 72.", "72") == 1.0
        assert numeric_match("1,000", 1000) == 1.0 and numeric_match("none", "1") == 0.0
        assert numeric_match(0.3333, "1/3", abs_tol=1e-3) == 0.0  # "1/3" reads as 3, not a third
        assert numeric_match("3.14159", 3.1416, abs_tol=1e-4) == 1.0

    def test_file_helpers_stay_inside_the_workspace(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello\n")
        (tmp_path.parent / f"{tmp_path.name}-outside.txt").write_text("secret")
        assert read_text(tmp_path, "a.txt") == "hello" and file_exists(tmp_path, "a.txt")
        assert read_text(tmp_path, "a.txt", strip_trailing_newline=False) == "hello\n"
        assert read_text(tmp_path, "missing.txt") is None and not file_exists(tmp_path, "missing.txt")
        assert read_text(tmp_path, f"../{tmp_path.name}-outside.txt") is None
        assert text_file_equals(tmp_path, path="a.txt", content="hello") == 1.0
        assert text_file_equals(tmp_path, path="a.txt", content="hello", strip_trailing_newline=False) == 0.0
        (tmp_path / "b.bin").write_bytes(b"\xff\xfe")
        assert read_text(tmp_path, "b.bin") == "��"


class TestAnswerAndSolutions:
    def test_answer_md_counts_as_a_solution(self, tmp_path):
        folder = write_task(tmp_path / "ds" / "a")
        (folder / "solution" / "solve.sh").unlink()
        task = load_task(folder)
        assert not task.has_solution and not task.has_answer
        (folder / "solution" / "answer.md").write_text("72\n")
        task = load_task(folder)
        assert task.has_solution and task.has_answer
        assert task.digest != load_task(write_task(tmp_path / "ds2" / "a")).digest  # solution/ enters the digest

    def test_every_example_has_a_solution(self):
        for name in ("hello_world", "hello_python_verifier", "hello_mcp_tool"):
            assert (EXAMPLES / name / "solution" / "solve.sh").is_file(), name
        for name in (
            "hello_dataset/hello",
            "hello_dataset/goodbye",
            "hello_verifier_reuse/exact-greeting",
            "hello_verifier_reuse/uppercase-greeting",
        ):
            assert (EXAMPLES / name / "solution" / "solve.sh").is_file(), name
        assert (EXAMPLES / "hello_parameterized_tasks" / "solution" / "answer.template.md").is_file()


class TestPublish:
    def test_published_parameterized_tasks_have_distinct_harbor_names(self, tmp_path):
        # What Harbor builds from the published folders keeps one task per [task].name, so each published task must
        # carry its own; Harbor's own loader (when installed) sees the names Gym's rewrite gave them.
        template = write_template(tmp_path / "qa")
        report = publish_harbor(template, tmp_path / "pub")
        assert report.tasks == ["q-1", "q-2"] and report.ok
        names = [tomllib.loads((tmp_path / "pub" / t / "task.toml").read_text())["task"]["name"] for t in report.tasks]
        assert names == ["harbor/q-1", "harbor/q-2"]
        harbor_task = pytest.importorskip("harbor.models.task.task")
        assert [harbor_task.Task(tmp_path / "pub" / t).name for t in report.tasks] == ["harbor/q-1", "harbor/q-2"]

    def test_publish_refuses_folders_sharing_a_harbor_name(self, tmp_path):
        # Two plain task folders with the same [task].name (a copied scaffold): Harbor would keep one of them, so the
        # publish is refused naming both, and nothing is left behind.
        root = tmp_path / "ds"
        for task_id in ("alpha", "beta", "gamma"):
            write_task(root / task_id)
        (root / "gamma" / "task.toml").write_text(HELLO_TOML.replace("harbor/hello-world", "harbor/gamma"))
        with pytest.raises(PublishError, match=r"'harbor/hello-world' is the \[task\].name of alpha, beta"):
            publish_harbor(root, tmp_path / "pub")
        assert not (tmp_path / "pub").exists() and sorted(p.name for p in tmp_path.iterdir()) == ["ds"]
        (root / "beta" / "task.toml").write_text(HELLO_TOML.replace("harbor/hello-world", "harbor/beta"))
        assert publish_harbor(root, tmp_path / "pub").tasks == ["alpha", "beta", "gamma"]

    def test_publish_leaves_a_multi_step_task_to_its_test_sh(self, tmp_path):
        # Gym never grades a [[steps]] task with a Python verifier (each step runs a test.sh), so publish writes no
        # shim for one even under a dataset-level verifier.py: a shim tests/test.sh would be Harbor's fallback grader
        # for every step without its own script, and the published copy would grade differently from the source.
        root = tmp_path / "ds"
        write_task(root / "plain")
        (root / "plain" / "tests" / "test.sh").unlink()
        (root / "plain" / "tests").rmdir()
        multi = root / "multi"
        write_task(multi)
        (multi / "tests" / "test.sh").unlink()
        (multi / "tests").rmdir()
        (multi / "task.toml").write_text(
            HELLO_TOML.replace("harbor/hello-world", "harbor/multi") + '\n[[steps]]\nname = "s1"\n'
        )
        (multi / "steps" / "s1" / "tests").mkdir(parents=True)
        (multi / "steps" / "s1" / "instruction.md").write_text("Step one.\n")
        (multi / "steps" / "s1" / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
        (root / "verifier.py").write_text(ANSWER_VERIFIER)
        source = load_task(multi, root=root)
        assert source.python_verifier is None and len(source.turns) == 1
        report = publish_harbor(root, tmp_path / "pub")
        assert report.tasks == ["multi", "plain"] and report.wrapped_verifiers == ["plain"] and report.ok
        published = tmp_path / "pub" / "multi"
        assert not (published / "tests").exists()
        assert sorted(p.relative_to(published).as_posix() for p in published.rglob("*") if p.is_file()) == [
            "environment/Dockerfile",
            "instruction.md",
            "solution/solve.sh",
            "steps/s1/instruction.md",
            "steps/s1/tests/test.sh",
            "task.toml",
        ]
        # It loads back as the source does: one step graded by its own test.sh, no Python verifier.
        back = load_task(published, root=tmp_path / "pub")
        assert back.python_verifier is None and [turn.name for turn in back.turns] == ["s1"]
        assert (tmp_path / "pub" / "plain" / "tests" / "test.sh").is_file()

    def test_publish_expands_rows_and_wraps_verifiers(self, tmp_path):
        template = write_template(tmp_path / "qa")
        (template / "dataset.toml").write_text(
            '[dataset]\nname = "local/qa"\nversion = "0.1.0"\n[gym]\nlicense = "Apache 2.0"\n'
        )
        report = publish_harbor(template, tmp_path / "pub")
        assert (
            report.tasks == ["q-1", "q-2"]
            and report.wrapped_verifiers == ["q-1", "q-2"]
            and report.wrapped_answers == []
        )
        task = tmp_path / "pub" / "q-1"
        # The task's module stays in tests/ and the shim's test.sh, the grader, runs it there (the hub's helper shape).
        assert sorted(p.name for p in (task / "tests").iterdir()) == [".ng", "test.sh", "verifier.py"]
        assert (task / "tests" / "verifier.py").read_text() == ANSWER_VERIFIER
        assert not (task / "tests" / ".ng" / "verifier.py").exists()
        assert "--verifier /tests/verifier.py" in (task / "tests" / "test.sh").read_text()
        assert json.loads((task / "tests" / ".ng" / "task_data.json").read_text()) == {
            "question": "2+2?",
            "answer": "4",
        }
        assert (task / "solution" / "solve.sh").read_text() == "#!/bin/bash\necho '4' > answer.txt\n"
        assert not (task / TASK_DATA_FILE).exists() and not (tmp_path / "pub" / "tasks.jsonl").exists()
        # dataset.toml is Harbor's artifact: the [gym] table stays with the source.
        assert (tmp_path / "pub" / "dataset.toml").read_text() == '[dataset]\nname = "local/qa"\nversion = "0.1.0"\n'
        text = report.text()
        assert "Published 2 task(s)" in text
        if report.harbor_check is not None:
            assert report.harbor_check == {} and report.ok, text
            assert "Harbor format check: 2 task(s) valid" in text
        else:
            assert "install the `harbor` package" in text

    def test_publish_copies_a_test_sh_task_with_a_verifier_py_helper_as_it_is(self, tmp_path):
        # The hub's helper shape: tests/test.sh calls tests/verifier.py. Gym grades it by test.sh, so publish copies
        # the folder verbatim, writes no shim and leaves the helper where test.sh expects it, dataset verifier or not.
        root = tmp_path / "ds"
        folder = write_task(root / "a")
        (folder / "tests" / "test.sh").write_text(
            "#!/bin/bash\npython3 /tests/verifier.py > /logs/verifier/reward.txt\n"
        )
        (folder / "tests" / "verifier.py").write_text("print(1.0)\n")
        (root / "verifier.py").write_text(ANSWER_VERIFIER)
        report = publish_harbor(root, tmp_path / "pub")
        assert report.tasks == ["a"] and report.wrapped_verifiers == [] and report.wrapped_answers == []
        tests = tmp_path / "pub" / "a" / "tests"
        assert sorted(p.name for p in tests.iterdir()) == ["test.sh", "verifier.py"]
        assert (tests / "test.sh").read_text() == (folder / "tests" / "test.sh").read_text()
        assert (tests / "verifier.py").read_text() == "print(1.0)\n"
        assert not (tmp_path / "pub" / "verifier.py").exists()
        assert load_task(tmp_path / "pub" / "a").python_verifier is None
        if report.harbor_check is not None:
            assert report.harbor_check == {} and report.ok, report.text()

    def test_publish_wraps_an_answer_and_the_dataset_verifier_into_a_task_that_grades_under_harbor(self, tmp_path):
        # The round trip of the publishable answer shape: answer.md plus a dataset verifier that reads
        # workspace/answer.md is published as solve.sh plus a test.sh shim; the published solve.sh, run as Harbor's
        # oracle runs it, leaves the file the published shim grades to 1.0. Gym's oracle leaves the same file with
        # the same command, so `gym dataset validate` and Harbor agree on the task.
        root = tmp_path / "ds"
        folder = write_task(root / "a")
        import shutil

        shutil.rmtree(folder / "tests")
        (folder / "solution" / "solve.sh").unlink()
        answer = "it's 4\n"
        (folder / "solution" / "answer.md").write_text(answer)
        (root / "verifier.py").write_text(ANSWER_MD_VERIFIER)
        (folder / "task_data.json").write_text('{"answer": "it\'s 4"}')
        report = publish_harbor(root, tmp_path / "pub")
        assert report.wrapped_verifiers == ["a"] and report.wrapped_answers == ["a"]
        if report.harbor_check is not None:
            assert report.harbor_check == {}, report.text()
        task = tmp_path / "pub" / "a"
        assert (task / "tests" / ".ng" / "verifier.py").read_text() == ANSWER_MD_VERIFIER
        assert "--verifier /tests/.ng/verifier.py" in (task / "tests" / "test.sh").read_text()
        solve = (task / "solution" / "solve.sh").read_text()
        assert solve.endswith(answer_write_command(answer) + "\n") and "printf '%s' 'it'\"'\"'s 4\n'" in solve
        assert not (tmp_path / "pub" / "verifier.py").exists()
        workspace = tmp_path / "ws"
        workspace.mkdir()
        subprocess.run(["bash", str(task / "solution" / "solve.sh")], cwd=workspace, check=True)
        assert sorted(p.name for p in workspace.iterdir()) == ["answer.md"]
        assert (workspace / "answer.md").read_text() == answer
        result = subprocess.run(
            [
                sys.executable,
                str(task / "tests" / ".ng" / "run_verifier.py"),
                "--verifier",
                str(task / "tests" / ".ng" / "verifier.py"),
                "--workspace",
                str(workspace),
                "--task-data",
                str(task / "tests" / ".ng" / "task_data.json"),
                "--reward-dir",
                str(tmp_path / "reward"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "reward" / "reward.txt").read_text() == "1.0\n"
        # Gym's own run of the task: the oracle leaves the same file, and the same verifier grades it.
        gym_workspace = tmp_path / "gym-ws"
        gym_workspace.mkdir()
        subprocess.run(["bash", "-c", answer_write_command(answer)], cwd=gym_workspace, check=True)
        assert (gym_workspace / "answer.md").read_bytes() == (workspace / "answer.md").read_bytes()

    @pytest.mark.parametrize("answer", [b"first\r\nlast\r\n", b"one\rtwo\n", b"no trailing newline"])
    def test_publish_keeps_the_answers_line_endings(self, tmp_path, answer):
        # The oracle leaves answer.md byte for byte; the exported solve.sh must too, or a byte-checking grader scores
        # the source 1.0 and the published task 0.0. `read_text` translated CR and CRLF to LF on the way out.
        root = tmp_path / "ds"
        folder = write_task(root / "a")
        shutil.rmtree(folder / "tests")
        (folder / "solution" / "solve.sh").unlink()
        (folder / "solution" / "answer.md").write_bytes(answer)
        (root / "verifier.py").write_text(ANSWER_MD_VERIFIER)
        (folder / "task_data.json").write_text('{"answer": ""}')
        report = publish_harbor(root, tmp_path / "pub")
        assert report.wrapped_answers == ["a"]
        workspace = tmp_path / "ws"
        workspace.mkdir()
        subprocess.run(["bash", str(tmp_path / "pub" / "a" / "solution" / "solve.sh")], cwd=workspace, check=True)
        assert (workspace / "answer.md").read_bytes() == answer
        assert (workspace / "answer.md").read_bytes() == (folder / "solution" / "answer.md").read_bytes()

    def test_publish_refuses_a_verifier_that_reads_the_response(self, tmp_path):
        template = write_template(tmp_path / "qa", verifier=RESPONSE_VERIFIER)
        with pytest.raises(PublishError, match="task 'q-1': verify\\(\\) in .* reads `response`"):
            publish_harbor(template, tmp_path / "pub")

    def test_publish_refuses_a_python_verifier_in_separate_verifier_mode(self, tmp_path):
        # Harbor never uploads tests/ into a separate verifier container (skip_tests_upload), so the published shim
        # could not grade there; Gym's server uploads it in both modes, which is why the task validates under Gym.
        explicit = HELLO_TOML.replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n')
        template = write_template(tmp_path / "explicit")
        (template / "task.toml").write_text(explicit)
        with pytest.raises(PublishError, match=r"task 'q-1': .*separate verifier environment.*environment_mode"):
            publish_harbor(template, tmp_path / "pub1")
        implied = HELLO_TOML + '\n[verifier.environment]\ndocker_image = "org/verifier:1"\n'
        template = write_template(tmp_path / "implied")
        (template / "task.toml").write_text(implied)
        with pytest.raises(PublishError, match=r"task 'q-1': .*\[verifier.environment\] table"):
            publish_harbor(template, tmp_path / "pub2")
        # Shared mode spelled out, and a test.sh task in separate mode, both publish: only the shim needs the upload.
        shared = HELLO_TOML.replace("[verifier]\n", '[verifier]\nenvironment_mode = "shared"\n')
        template = write_template(tmp_path / "shared")
        (template / "task.toml").write_text(shared)
        assert publish_harbor(template, tmp_path / "pub3").wrapped_verifiers == ["q-1", "q-2"]
        folder = write_task(tmp_path / "ds" / "a", toml=explicit)
        report = publish_harbor(folder, tmp_path / "pub4")
        assert report.tasks == ["a"] and report.wrapped_verifiers == []
        assert 'environment_mode = "separate"' in (tmp_path / "pub4" / "a" / "task.toml").read_text()

    def test_publish_checks_the_rows_against_the_dataset_task_data_before_writing(self, tmp_path):
        # The row model is the contract wherever rows are expanded: a template prepare refuses must not publish as
        # valid with the offending value baked into every shim's task_data.json.
        rows = [
            {"task_id": "q-1", "question": "2+2?", "answer": 4},
            {"task_id": "q-2", "question": "6*7?", "answer": "42", "hint": "x"},
        ]
        template = write_template(tmp_path / "qa", rows=rows)
        (template / "task_data.py").write_text(TASK_DATA_PY)
        with pytest.raises(PublishError, match="task 'q-1': task_data does not satisfy TaskData"):
            publish_harbor(template, tmp_path / "pub")
        assert not (tmp_path / "pub").exists()
        # The same check, in the same words, refuses the template at prepare.
        with pytest.raises(HarborTaskError, match="task 'q-1': task_data does not satisfy TaskData"):
            prepare_target(str(template), output_root=tmp_path / "out")
        # A dataset of plain task folders is checked the same way, and a model that does not import refuses it.
        root = tmp_path / "ds"
        folder = write_task(root / "a")
        (root / "task_data.py").write_text(
            "from pydantic import BaseModel\nclass TaskData(BaseModel):\n    answer: str\n"
        )
        (folder / "task_data.json").write_text('{"answer": "4", "extra": 1}')
        with pytest.raises(PublishError, match=r"task 'a': task_data has keys \['extra'\]"):
            publish_harbor(root, tmp_path / "pub")
        (root / "task_data.py").write_text("import nothing_of_the_sort\n")
        with pytest.raises(PublishError, match="task_data.py: failed to import"):
            publish_harbor(root, tmp_path / "pub")
        assert not (tmp_path / "pub").exists()
        (root / "task_data.py").write_text(
            "from pydantic import BaseModel\nclass TaskData(BaseModel):\n    answer: str\n    extra: int\n"
        )
        assert publish_harbor(root, tmp_path / "pub").tasks == ["a"]
        assert not (tmp_path / "pub" / "task_data.py").exists()

    def test_publish_keeps_task_data_json_below_the_task_root(self, tmp_path):
        # Only the task root's task_data.json stays behind (Harbor does not read it; the shim carries the fields): a
        # fixture of that name under tests/ is the task's own file and is published with it.
        folder = write_task(tmp_path / "ds" / "a")
        (folder / "task_data.json").write_text('{"answer": "4"}')
        (folder / "tests" / "task_data.json").write_text('{"fixture": true}')
        report = publish_harbor(folder, tmp_path / "pub")
        assert report.tasks == ["a"] and report.wrapped_verifiers == []
        assert not (tmp_path / "pub" / "a" / "task_data.json").exists()
        assert (tmp_path / "pub" / "a" / "tests" / "task_data.json").read_text() == '{"fixture": true}'

    def test_a_refused_publish_leaves_no_output_behind_and_a_retry_succeeds(self, tmp_path):
        root = tmp_path / "ds"
        write_task(root / "a-good")
        bad = write_task(root / "b-bad", toml=HELLO_TOML.replace("harbor/hello-world", "harbor/b-bad"))
        (bad / "tests" / "test.sh").unlink()
        (bad / "tests" / "verifier.py").write_text(RESPONSE_VERIFIER)
        with pytest.raises(PublishError, match="task 'b-bad': verify\\(\\) in .* reads `response`"):
            publish_harbor(root, tmp_path / "out" / "pub")
        # `a-good` was published before `b-bad` was refused: nothing of it remains, nor any staging folder.
        assert not (tmp_path / "out" / "pub").exists() and list((tmp_path / "out").iterdir()) == []
        (tmp_path / "empty").mkdir()
        with pytest.raises(PublishError, match="reads `response`"):
            publish_harbor(root, tmp_path / "empty")
        assert (tmp_path / "empty").is_dir() and list((tmp_path / "empty").iterdir()) == []
        (bad / "tests" / "verifier.py").write_text(ANSWER_VERIFIER)
        report = publish_harbor(root, tmp_path / "out" / "pub")
        assert report.tasks == ["a-good", "b-bad"] and report.wrapped_verifiers == ["b-bad"]
        assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["pub"]
        assert publish_harbor(root, tmp_path / "empty").tasks == ["a-good", "b-bad"]
        assert sorted(p.name for p in (tmp_path / "empty").iterdir()) == ["a-good", "b-bad"]

    def test_a_module_named_harbor_that_is_not_harbor_is_an_error_not_a_skipped_check(self, tmp_path, monkeypatch):
        # With environment_servers/ on sys.path, `import harbor` resolves to the environment server package. The
        # check must not then call the package "not installed" and let the publish pass unchecked.
        import types

        for name in [n for n in sys.modules if n == "harbor" or n.startswith("harbor.")]:
            monkeypatch.delitem(sys.modules, name)
        shadow = types.ModuleType("harbor")
        shadow.__file__ = str(tmp_path / "elsewhere" / "harbor" / "__init__.py")
        shadow.__path__ = []
        monkeypatch.setitem(sys.modules, "harbor", shadow)
        folder = write_task(tmp_path / "ds" / "a")
        with pytest.raises(PublishError) as info:
            publish_harbor(folder, tmp_path / "pub")
        message = str(info.value)
        assert "The `harbor` package does not import (ModuleNotFoundError: No module named 'harbor.models'" in message
        assert f"`import harbor` resolves to {shadow.__file__}, which is not the Harbor package" in message
        assert message.endswith("; nothing was published")
        assert not (tmp_path / "pub").exists() and sorted(p.name for p in tmp_path.iterdir()) == ["ds"]

    def test_publish_reports_the_folders_harbor_refuses_and_the_cli_exits_1(self, tmp_path, monkeypatch, capsys):
        pytest.importorskip("harbor.models.task.task")
        from nemo_gym.cli.main import main

        # Publish copies any task.toml folder; Harbor's loader then refuses one without an instruction.
        write_task(tmp_path / "ds" / "good")
        bad = write_task(tmp_path / "ds" / "bad", toml=HELLO_TOML.replace("harbor/hello-world", "harbor/bad"))
        (bad / "instruction.md").unlink()
        report = publish_harbor(tmp_path / "ds", tmp_path / "pub")
        assert report.tasks == ["bad", "good"] and report.ok is False
        assert list(report.harbor_check) == ["bad"] and "missing instruction.md" in report.harbor_check["bad"]
        text = report.text()
        assert text.startswith(
            f"Nothing published into {tmp_path / 'pub'}: Harbor's format check refused 1 of 2 task(s)"
        )
        assert "fix them and publish again\n    bad: FileNotFoundError:" in text
        assert "missing instruction.md" in text and "good:" not in text and "Published" not in text
        # The check ran over the staging folder: nothing appears under --output (nor any staging folder beside it),
        # so the retry into the same --output needs no cleanup.
        assert not (tmp_path / "pub").exists() and sorted(p.name for p in tmp_path.iterdir()) == ["ds"]
        monkeypatch.setattr(
            sys,
            "argv",
            ["gym", "dataset", "publish", str(tmp_path / "ds"), "--harbor", "--output", str(tmp_path / "cli")],
        )
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1
        assert "refused 1 of 2 task(s)" in capsys.readouterr().out and not (tmp_path / "cli").exists()
        (bad / "instruction.md").write_text("Do it.\n")
        assert main() is None  # exit 0, the same --output
        assert "Harbor format check: 2 task(s) valid" in capsys.readouterr().out
        assert (tmp_path / "cli" / "bad" / "instruction.md").is_file() and (tmp_path / "cli" / "good").is_dir()

    def test_publish_strips_the_gym_table_from_dataset_toml_and_keeps_harbors_tables_byte_for_byte(self, tmp_path):
        root = tmp_path / "ds"
        write_task(root / "a")
        harbor_head = (
            "# A dataset manifest\n"
            'schema_version = "1.0"\n\n'
            "[dataset]\n"
            'name = "local/qa"\n'
            'version = "0.1.0"\n'
            'description = "the [gym] table is Gym\'s"\n\n'
        )
        gym = (
            "[gym]\n"
            'license = "Apache 2.0"\n\n'
            "[gym.defaults]\n"
            "timeout_multiplier = 2.0\n\n"
            '[gym.tasks."a".environment]  # per-task override\n'
            'env = { PIP_INDEX_URL = "https://mirror.example/simple" }\n\n'
        )
        harbor_tail = f'[[tasks]]\nname = "local/a"\ndigest = "sha256:{"0" * 64}"\n'
        (root / "dataset.toml").write_text(harbor_head + gym + harbor_tail)
        publish_harbor(root, tmp_path / "pub")
        published = (tmp_path / "pub" / "dataset.toml").read_text()
        assert published == harbor_head + harbor_tail
        original = tomllib.loads(harbor_head + gym + harbor_tail)
        assert "gym" in original and tomllib.loads(published) == {k: v for k, v in original.items() if k != "gym"}
        # Without a [gym] table the file is copied as it is; dotted and inline spellings of the table go too.
        (root / "dataset.toml").write_text(harbor_head)
        publish_harbor(root, tmp_path / "pub2")
        assert (tmp_path / "pub2" / "dataset.toml").read_text() == harbor_head
        assert strip_gym_table('gym.license = "x"\n[dataset]\nname = "n"\n') == '[dataset]\nname = "n"\n'
        assert strip_gym_table('gym = { license = "x" }\n[dataset]\nname = "n"\n') == '[dataset]\nname = "n"\n'
        # A key named `gym` under one of Harbor's tables is Harbor's: only the root table's `gym` entries go.
        with_dataset_gym = harbor_head.replace(
            'version = "0.1.0"\n', 'version = "0.1.0"\ngym = "a value of Harbor\'s"\n'
        )
        assert strip_gym_table(with_dataset_gym + gym + harbor_tail) == with_dataset_gym + harbor_tail
        (root / "dataset.toml").write_text(with_dataset_gym + gym + harbor_tail)
        publish_harbor(root, tmp_path / "pub_dataset_gym")
        assert (tmp_path / "pub_dataset_gym" / "dataset.toml").read_text() == with_dataset_gym + harbor_tail
        # A header inside a multi-line string cannot be stripped by line: refused rather than mangled, nothing written.
        (root / "dataset.toml").write_text(
            '[dataset]\nname = "local/n"\ndescription = """\n[gym]\n"""\n[gym]\nlicense = "x"\n'
        )
        with pytest.raises(PublishError, match=r"the \[gym\] table could not be stripped for publishing \("):
            publish_harbor(root, tmp_path / "pub3")
        assert not (tmp_path / "pub3").exists()
        # A strip that leaves a parseable file with one of Harbor's entries changed names the entry (here a root
        # multi-line string whose `gym.` line the line-based strip took for a root `gym` entry).
        (root / "dataset.toml").write_text(
            'description = """\ngym.note = 1\n"""\n[dataset]\nname = "local/n"\n[gym]\nlicense = "x"\n'
        )
        with pytest.raises(PublishError, match=r"\(Harbor's `description` would not survive it unchanged\)"):
            publish_harbor(root, tmp_path / "pub4")
        assert not (tmp_path / "pub4").exists()
        manifest = pytest.importorskip("harbor.models.dataset.manifest")
        assert manifest.DatasetManifest.from_toml(published).dataset.name == "local/qa"

    def test_publish_leaves_out_a_dataset_toml_that_holds_only_the_gym_table(self, tmp_path):
        # `harbor add` takes the dataset branch as soon as dataset.toml exists and refuses an empty manifest, so a
        # source file that is Gym's [gym] table alone publishes no dataset.toml: a plain folder of task folders.
        root = tmp_path / "ds"
        write_task(root / "a")
        (root / "dataset.toml").write_text(
            '# Gym settings only\n[gym]\nlicense = "Apache 2.0"\n\n[gym.defaults]\ntimeout_multiplier = 2.0\n'
        )
        report = publish_harbor(root, tmp_path / "pub")
        assert report.ok and report.tasks == ["a"]
        assert sorted(p.name for p in (tmp_path / "pub").iterdir()) == ["a"]
        text = report.text()
        assert "Published 1 task(s)" in text
        assert "  dataset.toml not published: it holds Gym's [gym] table alone, and Harbor refuses a manifest" in text
        assert report.harbor_check in (None, {})  # the manifest check has nothing to read

    def test_publish_refuses_a_dataset_toml_harbor_would_with_or_without_harbor(self, tmp_path, monkeypatch):
        # Gym's own rule, so the refusal does not depend on the harbor package: a [dataset] table with an org/name name.
        monkeypatch.setitem(sys.modules, "harbor", None)
        root = tmp_path / "ds"
        write_task(root / "a")
        (root / "dataset.toml").write_text('[dataset]\nname = "hello"\nversion = "1.0.0"\n\n[gym]\nlicense = "x"\n')
        with pytest.raises(PublishError, match=r"dataset.toml: \[dataset\].name 'hello' is not `org/name`"):
            publish_harbor(root, tmp_path / "pub")
        assert not (tmp_path / "pub").exists() and sorted(p.name for p in tmp_path.iterdir()) == ["ds"]
        (root / "dataset.toml").write_text('schema_version = "1.0"\n\n[gym]\nlicense = "Apache 2.0"\n')
        with pytest.raises(PublishError, match=r"needs a \[dataset\] table with name = \"org/name\""):
            publish_harbor(root, tmp_path / "pub")
        (root / "dataset.toml").write_text('[dataset]\nname = "local/../x"\n')
        with pytest.raises(PublishError, match="is not `org/name`"):
            publish_harbor(root, tmp_path / "pub")
        # Harbor's version is optional: org/name alone is a manifest it reads.
        (root / "dataset.toml").write_text('[dataset]\nname = "local/hello"\n')
        report = publish_harbor(root, tmp_path / "pub")
        assert report.ok and (tmp_path / "pub" / "dataset.toml").read_text() == '[dataset]\nname = "local/hello"\n'

    def test_harbors_manifest_check_covers_the_published_dataset_toml(self, tmp_path):
        pytest.importorskip("harbor.models.dataset.manifest")
        root = tmp_path / "ds"
        write_task(root / "a")
        # Gym's own rule passes (org/name); Harbor's DatasetManifest refuses the task reference's digest.
        (root / "dataset.toml").write_text(
            '[dataset]\nname = "local/hello"\nversion = "1.0.0"\n\n[[tasks]]\nname = "local/a"\ndigest = "nope"\n'
        )
        report = publish_harbor(root, tmp_path / "pub")
        assert report.ok is False and list(report.harbor_check) == ["dataset.toml"]
        assert report.harbor_check["dataset.toml"].startswith("ValidationError: tasks.0.digest: Value error, Digest")
        text = report.text()
        assert text.startswith(
            f"Nothing published into {tmp_path / 'pub'}: Harbor's format check refused dataset.toml (harbor "
        )
        assert "\n    dataset.toml: ValidationError: tasks.0.digest" in text
        assert not (tmp_path / "pub").exists() and sorted(p.name for p in tmp_path.iterdir()) == ["ds"]
        # A refused task and the manifest are reported together.
        (root / "a" / "instruction.md").unlink()
        report = publish_harbor(root, tmp_path / "pub")
        assert sorted(report.harbor_check) == ["a", "dataset.toml"]
        assert "Harbor's format check refused 1 of 1 task(s) and dataset.toml" in report.text()
        # Fixed, the same --output takes the publish.
        (root / "a" / "instruction.md").write_text("Do it.\n")
        (root / "dataset.toml").write_text('[dataset]\nname = "local/hello"\nversion = "1.0.0"\n')
        assert publish_harbor(root, tmp_path / "pub").ok and (tmp_path / "pub" / "dataset.toml").is_file()

    def test_publish_refuses_a_symbolic_link_in_a_task(self, tmp_path):
        outside = tmp_path / "outside.txt"
        outside.write_text("host file content\n")
        root = tmp_path / "ds"
        task = write_task(root / "t1")
        (task / "tests" / "data.txt").symlink_to(Path("../../../outside.txt"))
        # The loader refuses the task; publish must not follow the link and ship the host file instead.
        with pytest.raises(HarborTaskError, match="escapes the task folder: tests/data.txt"):
            load_task(task, root=root)
        with pytest.raises(PublishError, match="task 't1': tests/data.txt is a symbolic link, which publish does not"):
            publish_harbor(root, tmp_path / "pub")
        assert not (tmp_path / "pub").exists() and sorted(p.name for p in tmp_path.iterdir()) == ["ds", "outside.txt"]
        # A link inside the task is refused too (the renderer's rule: copy the file in).
        (task / "tests" / "data.txt").unlink()
        (task / "tests" / "data.txt").symlink_to(Path("test.sh"))
        with pytest.raises(PublishError, match="tests/data.txt is a symbolic link"):
            publish_harbor(root, tmp_path / "pub")
        # A link under a name the digest ignores is not a task file; a copied file publishes as a regular file.
        (task / "tests" / "data.txt").unlink()
        (task / "tests" / "data.txt").write_bytes(outside.read_bytes())
        (task / "__pycache__").mkdir()
        (task / "__pycache__" / "x").symlink_to(outside)
        report = publish_harbor(root, tmp_path / "pub")
        published = tmp_path / "pub" / "t1"
        assert report.tasks == ["t1"] and not (published / "__pycache__").exists()
        assert not (published / "tests" / "data.txt").is_symlink()
        assert (published / "tests" / "data.txt").read_text() == "host file content\n"

    def test_publish_refuses_a_task_folder_that_is_a_symbolic_link(self, tmp_path):
        # The folder walk lists a linked task folder so that the loader can refuse it; publish checked the folder's
        # descendants for links but not the folder itself, and copytree followed the link: a task from outside the
        # dataset, marker file included, was published while `load_task` refused it.
        outside = write_task(tmp_path / "outside", toml=HELLO_TOML.replace("harbor/hello-world", "harbor/outside"))
        (outside / "MARKER.txt").write_text("outside the dataset\n")
        root = tmp_path / "ds"
        write_task(root / "real")
        (root / "linked").symlink_to(outside)
        with pytest.raises(HarborTaskError, match="must not be a symlink"):
            load_task(root / "linked", root=root)
        with pytest.raises(PublishError, match="task 'linked': the task folder is a symbolic link"):
            publish_harbor(root, tmp_path / "pub")
        assert not (tmp_path / "pub").exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["ds", "outside"]

    def test_publish_copies_only_what_the_digest_covers(self, tmp_path):
        # An author's working files at the task root (a `.env` holding a key, a `.git/`, a `.venv/` whose interpreter
        # is a link, a scratch folder) are not the task's: copytree used to ship them all (and the `.venv` link was
        # refused as if it were a task file). The published task is what the digest covers, which is also what
        # Harbor's packager reads; a hidden file below the root (`environment/.env`) is task content and is kept.
        root = tmp_path / "ds"
        task = write_task(root / "t1")
        (task / ".env").write_text("OPENAI_API_KEY=sk-not-a-real-key\n")
        (task / ".git").mkdir()
        (task / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        (task / ".venv" / "bin").mkdir(parents=True)
        (task / ".venv" / "bin" / "python").symlink_to(Path(sys.executable))
        (task / "scratch").mkdir()
        (task / "scratch" / "notes.txt").write_text("author notes\n")
        (task / "environment" / ".env").write_text("TASK_VAR=1\n")
        report = publish_harbor(root, tmp_path / "pub")
        published = tmp_path / "pub" / "t1"
        assert report.tasks == ["t1"]
        assert sorted(p.relative_to(published).as_posix() for p in published.rglob("*")) == [
            "environment",
            "environment/.env",
            "environment/Dockerfile",
            "instruction.md",
            "solution",
            "solution/solve.sh",
            "task.toml",
            "tests",
            "tests/test.sh",
        ]
        assert "OPENAI_API_KEY" not in b"".join(p.read_bytes() for p in published.rglob("*") if p.is_file()).decode()
        # A test.sh task is published as it is: the published folder has the source's identity.
        assert content_hash(published) == content_hash(task)

    def test_publish_keeps_the_root_gitignore_that_harbor_packages_by(self, tmp_path):
        # Harbor's packager (`harbor add`, `harbor publish`) reads one root file beyond what the digest covers: the
        # task's `.gitignore`, its ignore rules, which `harbor task init` writes into every task (node_modules/,
        # .venv/, .env at any depth). Left behind, Harbor would package a scaffolded task's node_modules/ from the
        # published folder and the hub content hash would no longer be the source's. Gym's digest does not read the
        # file: the published task keeps the source's Gym identity, node_modules/ included.
        root = tmp_path / "ds"
        task = write_task(root / "t1")
        rules = "# Python\n__pycache__/\n*.pyc\n.venv/\n\n# Node\nnode_modules/\n\n# Secrets\n.env\n"
        (task / ".gitignore").write_text(rules)
        (task / "environment" / "node_modules").mkdir()
        (task / "environment" / "node_modules" / "left-pad.js").write_text("module.exports = 1\n")
        (task / "environment" / ".env").write_text("TASK_VAR=1\n")
        publish_harbor(root, tmp_path / "pub")
        published = tmp_path / "pub" / "t1"
        assert (published / ".gitignore").read_text() == rules
        assert (published / "environment" / "node_modules" / "left-pad.js").is_file()
        assert content_hash(published) == content_hash(task)
        # The file travels, so a linked one is refused like any other published link (copytree would follow it).
        (tmp_path / "outside.gitignore").write_text("*\n")
        other = write_task(tmp_path / "ds2" / "t2")
        (other / ".gitignore").symlink_to(tmp_path / "outside.gitignore")
        with pytest.raises(PublishError, match=r"task 't2': \.gitignore is a symbolic link"):
            publish_harbor(tmp_path / "ds2", tmp_path / "pub2")

    def test_published_task_packages_under_harbor_as_the_source_does(self, tmp_path):
        # The hub parity the root `.gitignore` buys: a task scaffolded by `harbor task init` (its template .gitignore)
        # packages to the same file set and content hash from the published folder as from the source.
        harbor = pytest.importorskip("harbor")
        packager = pytest.importorskip("harbor.publisher.packager")
        template_gitignore = Path(harbor.__file__).parent / "cli" / "template-task" / ".gitignore"
        root = tmp_path / "ds"
        task = write_task(root / "t1")
        (task / ".gitignore").write_text(template_gitignore.read_text())
        (task / "environment" / "node_modules").mkdir()
        (task / "environment" / "node_modules" / "left-pad.js").write_text("module.exports = 1\n")
        (task / "environment" / ".env").write_text("TASK_VAR=1\n")
        (task / "tests" / ".pytest_cache").mkdir()
        (task / "tests" / ".pytest_cache" / "v").write_text("x\n")
        publish_harbor(root, tmp_path / "pub")
        published = tmp_path / "pub" / "t1"
        source_hash, _ = packager.Packager.compute_content_hash(task)
        published_hash, published_files = packager.Packager.compute_content_hash(published)
        assert sorted(p.relative_to(published).as_posix() for p in published_files) == [
            "environment/Dockerfile",
            "instruction.md",
            "solution/solve.sh",
            "task.toml",
            "tests/test.sh",
        ]
        assert published_hash == source_hash

    def test_published_template_instances_carry_the_templates_gitignore(self, tmp_path):
        # publish renders a template before it copies, and the renderer used to drop every hidden root entry, the
        # `.gitignore` with them: a published instance then packaged under Harbor without the rules the template
        # carried (node_modules/ and .env shipped to the hub). The template's `.gitignore` now reaches every instance;
        # Gym's digest does not read it, so it is no part of the instance's identity.
        template = write_template(tmp_path / "qa")
        rules = "node_modules/\n.env\n"
        (template / ".gitignore").write_text(rules)
        (template / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
        (template / "environment" / "node_modules").mkdir()
        (template / "environment" / "node_modules" / "left-pad.js").write_text("module.exports = 1\n")
        (template / "environment" / ".env").write_text("TASK_VAR=1\n")
        report = publish_harbor(template, tmp_path / "pub")
        assert report.tasks == ["q-1", "q-2"]
        for task_id in ("q-1", "q-2"):
            published = tmp_path / "pub" / task_id
            assert (published / ".gitignore").read_text() == rules
            assert (published / "environment" / "node_modules" / "left-pad.js").is_file()
            assert (published / "environment" / ".env").read_text() == "TASK_VAR=1\n"
            identity = content_hash(published)
            (published / ".gitignore").unlink()
            assert content_hash(published) == identity

    def test_a_published_template_instance_packages_under_harbor_by_the_templates_rules(self, tmp_path):
        # Harbor's packager, run on a published instance, applies the template's `.gitignore`: node_modules/ and
        # .env stay out of the package, as they do when Harbor packages the template folder itself.
        harbor = pytest.importorskip("harbor")
        packager = pytest.importorskip("harbor.publisher.packager")
        template_gitignore = Path(harbor.__file__).parent / "cli" / "template-task" / ".gitignore"
        template = write_template(tmp_path / "qa", rows=[{"task_id": "q-1", "question": "2+2?", "answer": "4"}])
        (template / ".gitignore").write_text(template_gitignore.read_text())
        (template / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
        (template / "environment" / "node_modules").mkdir()
        (template / "environment" / "node_modules" / "left-pad.js").write_text("module.exports = 1\n")
        (template / "environment" / ".env").write_text("TASK_VAR=1\n")
        publish_harbor(template, tmp_path / "pub")
        published = tmp_path / "pub" / "q-1"
        _, published_files = packager.Packager.compute_content_hash(published)
        assert sorted(p.relative_to(published).as_posix() for p in published_files) == [
            "environment/Dockerfile",
            "instruction.md",
            "solution/notes.txt",
            "solution/solve.sh",
            "task.toml",
            "tests/test.sh",
            "tests/verifier.py",
        ]

    def test_publish_refuses_bad_output_folders(self, tmp_path):
        template = write_template(tmp_path / "qa")
        with pytest.raises(PublishError, match="must be outside the dataset"):
            publish_harbor(template, template / "out")
        (tmp_path / "busy").mkdir()
        (tmp_path / "busy" / "x").write_text("")
        with pytest.raises(PublishError, match="exists and is not empty"):
            publish_harbor(template, tmp_path / "busy")
        with pytest.raises(PublishError, match="is not a directory"):
            publish_harbor(tmp_path / "nope", tmp_path / "pub")

    def test_publish_says_how_to_install_harbor_when_it_is_missing(self, tmp_path, monkeypatch):
        # The check is not optional: it is skipped, and said so, only when the package cannot be imported.
        monkeypatch.setitem(sys.modules, "harbor", None)
        folder = write_task(tmp_path / "ds" / "a")
        report = publish_harbor(folder, tmp_path / "pub")
        assert report.harbor_check is None and report.ok
        assert "install the `harbor` package" in report.harbor_note and "Harbor format check skipped" in report.text()
        assert (tmp_path / "pub" / "a" / "tests" / "test.sh").read_text() == (folder / "tests" / "test.sh").read_text()

    def test_a_published_task_loads_back_into_gym_and_grades_through_its_shim(self, tmp_path):
        # The published tests/ once held both the shim's test.sh and the copied verifier.py, which the loader
        # refuses: a dataset published to the hub could not be pulled, validated or run by Gym again.
        template = write_template(
            tmp_path / "qa",
            verifier="from helpers import ANSWER_FILE\n\n\ndef verify(workspace, task_data):\n"
            "    path = workspace / ANSWER_FILE\n"
            "    return float(path.is_file() and path.read_text().strip() == task_data['answer'])\n",
        )
        (template / "tests" / "helpers.py").write_text('ANSWER_FILE = "answer.txt"\n')
        report = publish_harbor(template, tmp_path / "pub")
        assert report.wrapped_verifiers == ["q-1", "q-2"]
        task = tmp_path / "pub" / "q-2"
        assert sorted(p.name for p in (task / "tests").iterdir()) == [".ng", "helpers.py", "test.sh", "verifier.py"]
        loaded = load_task(task)
        assert loaded.python_verifier is None and loaded.has_solution and loaded.task_id == "q-2"
        # The published shim grades a workspace the way Harbor runs it: test.sh -> the runner -> verify(), the module
        # where the task kept it and the helper module beside it importable from /tests.
        assert published_verifier(task) == task / "tests" / "verifier.py"
        workspace = tmp_path / "ws"
        workspace.mkdir()
        subprocess.run(["bash", str(task / "solution" / "solve.sh")], cwd=workspace, check=True)
        result = run_published_shim(task, workspace, tmp_path / "reward")
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "reward" / "reward.txt").read_text() == "1.0\n"
        # Publishing the published folder again (a dataset pulled from the hub) copies it as the test.sh task it
        # now is, with the shim's tests/.ng/task_data.json it needs: only the task root's task_data.json stays behind.
        again = publish_harbor(tmp_path / "pub", tmp_path / "pub2")
        assert again.tasks == ["q-1", "q-2"] and again.wrapped_verifiers == []
        republished = tmp_path / "pub2" / "q-2"
        assert (republished / "tests" / "test.sh").read_text() == (task / "tests" / "test.sh").read_text()
        assert sorted(p.name for p in (republished / "tests").iterdir()) == [
            ".ng",
            "helpers.py",
            "test.sh",
            "verifier.py",
        ]
        assert sorted(p.name for p in (republished / "tests" / ".ng").iterdir()) == [
            "nemo_gym",
            "run_verifier.py",
            "task_data.json",
        ]
        assert load_task(republished).python_verifier is None
        result = run_published_shim(republished, workspace, tmp_path / "reward2")
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "reward2" / "reward.txt").read_text() == "1.0\n"

    def test_a_published_task_verifier_reads_its_sibling_fixtures_where_they_were(self, tmp_path):
        # A task's tests/verifier.py may read a fixture beside it by __file__. The published shim runs the module where
        # the task kept it, fixture still beside it; moving the module under .ng/ graded the source 1.0 and the
        # published task not at all (FileNotFoundError, no reward).
        root = tmp_path / "ds"
        task = write_task(root / "t1")
        (task / "tests" / "test.sh").unlink()
        (task / "tests" / "verifier.py").write_text(
            "from pathlib import Path\n\n\ndef verify(workspace):\n"
            "    expected = Path(__file__).with_name('expected.txt').read_text()\n"
            "    return float((workspace / 'out.txt').read_text() == expected)\n"
        )
        (task / "tests" / "expected.txt").write_text("hi\n")
        report = publish_harbor(root, tmp_path / "pub")
        published = tmp_path / "pub" / "t1"
        assert report.wrapped_verifiers == ["t1"]
        assert sorted(p.name for p in (published / "tests").iterdir()) == [
            ".ng",
            "expected.txt",
            "test.sh",
            "verifier.py",
        ]
        assert published_verifier(published) == published / "tests" / "verifier.py"
        assert not (published / "tests" / ".ng" / "verifier.py").exists()
        assert load_task(published).python_verifier is None
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "out.txt").write_text("hi\n")
        result = run_published_shim(published, workspace, tmp_path / "reward")
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "reward" / "reward.txt").read_text() == "1.0\n"

    def test_publish_cli(self, tmp_path, monkeypatch, capsys):
        from nemo_gym.cli.main import main

        template = write_template(tmp_path / "qa")
        monkeypatch.setattr(
            sys, "argv", ["gym", "dataset", "publish", str(template), "--harbor", "--output", str(tmp_path / "pub")]
        )
        main()
        assert "Published 2 task(s)" in capsys.readouterr().out
        monkeypatch.setattr(
            sys, "argv", ["gym", "dataset", "publish", str(template), "--output", str(tmp_path / "pub2")]
        )
        with pytest.raises(SystemExit):
            main()
        assert "pass --harbor" in capsys.readouterr().err
        monkeypatch.setattr(
            sys,
            "argv",
            ["gym", "dataset", "publish", str(template), "--harbor", "--output", str(tmp_path / "pub3"), "--no-check"],
        )
        with pytest.raises(SystemExit):
            main()
        assert "--no-check" in capsys.readouterr().err and not (tmp_path / "pub3").exists()


class TestScaffold:
    def test_parameterized_scaffold_renders_loads_and_publishes(self, tmp_path, monkeypatch):
        # The runner puts the shim's .ng (with its vendored nemo_gym) first on sys.path; a later test's spawned
        # worker inherits sys.path and would import that nemo_gym instead of the real one.
        monkeypatch.setattr(sys, "path", list(sys.path))
        folder = init_dataset(tmp_path, "my-qa", parameterized=True)
        assert is_template_folder(folder)
        assert sorted(p.name for p in folder.iterdir()) == [
            "dataset.toml",
            "environment",
            "instruction.template.md",
            "solution",
            "task.toml",
            "task_data.py",
            "tasks.jsonl",
            "tests",
        ]
        assert [p.name for p in (folder / "solution").iterdir()] == ["answer.template.md"]
        # A reference answer with an apostrophe: inside a quoted shell literal it would break the solution; as a
        # rendered answer.md it is written byte for byte by Gym's oracle and by the published solve.sh.
        with (folder / "tasks.jsonl").open("a") as rows_file:
            rows_file.write(
                json.dumps({"task_id": "hello-3", "task_data": {"question": "Who?", "answer": "it's Bob"}})
            )
        prepared = prepare_target(str(folder), output_root=tmp_path / "out")
        assert [t.task_id for t in prepared.tasks] == ["hello-1", "hello-2", "hello-3"]
        assert all(t.python_verifier is not None and t.has_solution and t.has_answer for t in prepared.tasks)
        rows = [json.loads(line) for line in prepared.rows_path.read_text().splitlines()]
        assert rows[1]["task_input"]["task_data"]["answer"] == "42"
        task = prepared.tasks[2]
        assert (task.path / "solution" / "answer.md").read_text() == "it's Bob\n"
        assert "answer.md" in (task.path / "instruction.md").read_text()
        # What Gym's oracle does with answer.md (the command it shares with the published solve.sh) satisfies the
        # scaffold's verifier through the real runner.
        workspace = tmp_path / "ws"
        workspace.mkdir()
        answer = (task.path / "solution" / "answer.md").read_text()
        subprocess.run(["bash", "-c", answer_write_command(answer)], cwd=workspace, check=True)
        assert (workspace / "answer.md").read_text() == "it's Bob\n"
        shim_dir = write_shim(
            tmp_path / "shim",
            VerifierShim(verifier=task.python_verifier, in_task_tests=False, workspace=None, task_data=task.task_data),
        )
        argv = ["--verifier", str(shim_dir / ".ng" / "verifier.py"), "--workspace", str(workspace)]
        argv += ["--task-data", str(shim_dir / ".ng" / "task_data.json"), "--reward-dir", str(tmp_path / "r")]
        assert run_verifier(argv) == 0 and (tmp_path / "r" / "reward.txt").read_text() == "1.0\n"
        # Published: the answer becomes a solve.sh that parses and leaves the same file for the published shim.
        report = publish_harbor(folder, tmp_path / "pub")
        assert report.tasks == ["hello-1", "hello-2", "hello-3"] and report.ok
        assert report.wrapped_verifiers == report.wrapped_answers == ["hello-1", "hello-2", "hello-3"]
        # Each published task carries its own Harbor name (the template's org, the row's task_id): Harbor keys a
        # dataset on it, so three folders sharing the template's name would collapse into one task under `harbor add`.
        assert tomllib.loads((folder / "task.toml").read_text())["task"]["name"] == "local/hello"
        assert [
            tomllib.loads((tmp_path / "pub" / t / "task.toml").read_text())["task"]["name"] for t in report.tasks
        ] == [
            "local/hello-1",
            "local/hello-2",
            "local/hello-3",
        ]
        assert [t.config.task["name"] for t in prepared.tasks] == ["local/hello-1", "local/hello-2", "local/hello-3"]
        published = tmp_path / "pub" / "hello-3"
        harbor_workspace = tmp_path / "harbor-ws"
        harbor_workspace.mkdir()
        subprocess.run(["bash", str(published / "solution" / "solve.sh")], cwd=harbor_workspace, check=True)
        assert (harbor_workspace / "answer.md").read_text() == "it's Bob\n"
        # The scaffold's verifier is the task's own tests/verifier.py: published where it was, the shim runs it there.
        assert published_verifier(published) == published / "tests" / "verifier.py"
        result = run_published_shim(published, harbor_workspace, tmp_path / "r2")
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "r2" / "reward.txt").read_text() == "1.0\n"
