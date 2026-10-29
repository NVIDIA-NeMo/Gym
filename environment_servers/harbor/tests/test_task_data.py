# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The seed validates a row's ``task_data`` against the dataset's ``task_data.py`` and the task folder (M32)."""

import json
from pathlib import Path

import pytest

from environment_servers.harbor.app import HarborEpisodeRequest
from environment_servers.harbor.tests.conftest import make_server, run_episode
from nemo_gym.data import DIGEST_KEY, TaskDataError, TaskDataSchema, load_task
from nemo_gym.data.tasks.harbor.task_data import TASK_DATA_FILE, TASK_DATA_MODULE


QA_SCHEMA = """
from pydantic import BaseModel


class TaskData(BaseModel):
    question: str
    answer: str
"""

FIELDS = {"question": "2+2?", "answer": "4"}

RAISING_SCHEMA = """
from pydantic import BaseModel, field_validator

UNITS = {"cm": 1.0}


class TaskData(BaseModel):
    answer: str
    unit: str

    @field_validator("unit")
    @classmethod
    def known_unit(cls, v):
        UNITS[v]
        return v
"""


def make_dataset_server(tmp_path, monkeypatch, *, schema: str | None = QA_SCHEMA, fields: dict | None = FIELDS):
    """A server over ``datasets/ds`` whose dataset ships ``task_data.py`` and whose task carries ``task_data.json``.

    Both files exist before the server loads the task and imports the schema, as prepare leaves them.
    """
    folder = tmp_path / "datasets" / "ds"
    folder.mkdir(parents=True)
    if schema is not None:
        (folder / TASK_DATA_MODULE).write_text(schema)
    if fields is not None:
        (folder / "hello").mkdir()
        (folder / "hello" / TASK_DATA_FILE).write_text(json.dumps(fields))
    return make_server(tmp_path, monkeypatch)


def episode_with(task, task_data: dict) -> HarborEpisodeRequest:
    return HarborEpisodeRequest.model_validate(
        {
            "episode_id": {"rollout_id": "r1", "attempt": 0},
            "task": {
                "task_id": {"taskset": "ds", "task_id": "hello"},
                "task_input": {
                    "responses_create_params": {"input": [{"role": "user", "content": "Create hello.txt"}]},
                    "task_data": {DIGEST_KEY: task.digest, **task_data},
                },
            },
        }
    )


class TestSeedValidation:
    async def test_a_row_the_model_accepts_seeds_and_runs(self, tmp_path, monkeypatch):
        server, task, _, created = make_dataset_server(tmp_path, monkeypatch)
        assert task.task_data == FIELDS
        schema = server._task_data_schemas["ds"]
        assert isinstance(schema, TaskDataSchema) and schema.path == tmp_path / "datasets" / "ds" / TASK_DATA_MODULE
        response = await server.run_request(episode_with(task, FIELDS))
        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and created == [("hello", "/app")]

    @pytest.mark.parametrize(
        ("task_data", "fragment"),
        [
            # The row carries a value the model rejects.
            ({"question": "2+2?", "answer": 4}, "task 'hello': task_data does not satisfy TaskData in"),
            # The row carries a field the model does not declare.
            ({**FIELDS, "hint": "even"}, "task_data has keys ['hint'] that TaskData in"),
            # The row lost a field the model requires.
            ({"question": "2+2?"}, "answer: Field required"),
        ],
    )
    async def test_a_row_the_model_rejects_is_a_terminal_seed_failure(
        self, tmp_path, monkeypatch, task_data, fragment
    ):
        server, task, _, created = make_dataset_server(tmp_path, monkeypatch)
        response = await server.run_request(episode_with(task, task_data))
        failure = response.failure
        assert response.result is None and failure.stage == "seed" and failure.terminal
        assert fragment in failure.failure_reason
        assert (
            str(tmp_path / "datasets" / "ds" / TASK_DATA_MODULE) in failure.failure_reason
            or "Field required" in fragment
        )
        # Refused before anything was provisioned or the harness was contacted.
        assert created == [] and server.server_client.calls == []

    async def test_a_validator_that_raises_is_a_terminal_seed_failure_too(self, tmp_path, monkeypatch):
        # A KeyError out of an author's validator is a row defect like a ValidationError: 422, terminal, nothing
        # provisioned. Were it to escape raw, run() would record a non-terminal seed failure and the row be retried.
        server, task, _, created = make_dataset_server(
            tmp_path, monkeypatch, schema=RAISING_SCHEMA, fields={"answer": "4", "unit": "cm"}
        )
        assert (await server.run_request(episode_with(task, {"answer": "4", "unit": "cm"}))).result.reward == 1.0
        response = await server.run_request(episode_with(task, {"answer": "4", "unit": "inch"}))
        failure = response.failure
        assert response.result is None and failure.stage == "seed" and failure.terminal
        assert failure.failure_reason == (
            f"task 'hello': TaskData in {tmp_path / 'datasets' / 'ds' / TASK_DATA_MODULE} raised KeyError: 'inch' "
            "while validating task_data"
        )
        assert created == [("hello", "/app")]

    async def test_a_row_that_differs_from_the_folder_is_refused_without_a_schema(self, tmp_path, monkeypatch):
        # No task_data.py: the dataset is not typed, but the row still has to be the folder's.
        server, task, _, created = make_dataset_server(tmp_path, monkeypatch, schema=None)
        assert server._task_data_schemas == {"ds": None}
        assert (await server.run_request(episode_with(task, FIELDS))).result.reward == 1.0
        response = await server.run_request(episode_with(task, {**FIELDS, "answer": "5"}))
        failure = response.failure
        assert response.result is None and failure.stage == "seed" and failure.terminal
        assert (
            "task 'hello': the row's task_data differs from the task folder's task_data.json" in failure.failure_reason
        )
        assert (
            "values that differ: ['answer']" in failure.failure_reason and "re-materialize" in failure.failure_reason
        )
        assert created == [("hello", "/app")]

    async def test_a_value_lax_validation_would_coerce_is_a_terminal_seed_failure(self, tmp_path, monkeypatch):
        # The verifier reads the folder's task_data.json as written (the shim copies task.task_data), so a row the
        # model only accepts by coercion ("3" for an int, "no" for a bool) is refused rather than graded on a string.
        schema = "from pydantic import BaseModel\n\n\nclass TaskData(BaseModel):\n    count: int\n    flag: bool\n"
        server, task, _, created = make_dataset_server(
            tmp_path, monkeypatch, schema=schema, fields={"count": "3", "flag": "no"}
        )
        assert task.task_data == {"count": "3", "flag": "no"}
        response = await server.run_request(episode_with(task, {"count": "3", "flag": "no"}))
        failure = response.failure
        assert response.result is None and failure.stage == "seed" and failure.terminal
        assert "count: Input should be a valid integer" in failure.failure_reason
        assert "flag: Input should be a valid boolean" in failure.failure_reason
        assert created == [] and server.server_client.calls == []

    async def test_a_task_without_fields_refuses_a_row_that_carries_some(self, tmp_path, monkeypatch):
        server, task, _, created = make_server(tmp_path, monkeypatch)
        response = await server.run_request(episode_with(task, {"answer": "4"}))
        assert response.failure.terminal and "keys not in the folder: ['answer']" in response.failure.failure_reason
        assert created == []
        # Rows of a plain task keep working as before.
        assert (await run_episode(server, task)).result.reward == 1.0

    async def test_a_module_that_does_not_import_refuses_every_episode_of_the_taskset(
        self, tmp_path, monkeypatch, caplog
    ):
        server, task, _, created = make_dataset_server(tmp_path, monkeypatch, schema="raise RuntimeError('boom')\n")
        error = server._task_data_schemas["ds"]
        assert isinstance(error, TaskDataError) and "failed to import: RuntimeError: boom" in str(error)
        assert any("its task_data.py does not load" in record.message for record in caplog.records)
        response = await server.run_request(episode_with(task, FIELDS))
        failure = response.failure
        assert response.result is None and failure.stage == "seed" and failure.terminal
        assert (
            failure.failure_reason.startswith("Taskset 'ds': ")
            and "failed to import: RuntimeError: boom" in failure.failure_reason
        )
        assert created == [] and server.server_client.calls == []

    async def test_an_edited_folder_is_still_caught_by_the_digest_first(self, tmp_path, monkeypatch):
        server, task, _, created = make_dataset_server(tmp_path, monkeypatch)
        (task.path / TASK_DATA_FILE).write_text(json.dumps({"question": "2+2?", "answer": "5"}))
        assert load_task(task.path).digest != task.digest
        response = await server.run_request(episode_with(task, FIELDS))
        assert "changed since materialization" in response.failure.failure_reason and created == []

    def test_schemas_are_imported_once_per_taskset_at_startup(self, tmp_path, monkeypatch):
        server, task, _, _ = make_dataset_server(tmp_path, monkeypatch)
        # A later edit of the module is not seen: the server validates against what it started with.
        (tmp_path / "datasets" / "ds" / TASK_DATA_MODULE).write_text("raise RuntimeError('later')\n")
        schema = server._task_data_schemas["ds"]
        assert isinstance(schema, TaskDataSchema)
        schema.validate({DIGEST_KEY: task.digest, **FIELDS}, task_id="hello")
        assert Path(schema.path).read_text().startswith("raise")
