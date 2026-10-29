# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dataset-level ``task_data.py`` (M32): loading, the loader's ``task_data.json``, and the three validation points."""

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pydantic import BaseModel, ValidationError

from nemo_gym.config.config_types import EnvironmentServerInstanceConfig, ResponsesAPIAgentServerInstanceConfig
from nemo_gym.data import (
    DIGEST_KEY,
    HarborTaskError,
    TaskDataError,
    TaskDataSchema,
    check_row_matches_folder,
    load_task,
)
from nemo_gym.data.dataset_preparation import TrainDataProcessor
from nemo_gym.data.tasks.harbor import content_hash
from nemo_gym.data.tasks.harbor.cli import prepare_target
from nemo_gym.data.tasks.harbor.digest import DATASET_DIGEST_KEY
from nemo_gym.data.tasks.harbor.materialize import (
    agent_instance,
    environment_server_instance,
    materialize_task,
    run_config,
    write_rows,
)
from nemo_gym.data.tasks.harbor.task_data import (
    TASK_DATA_FILE,
    TASK_DATA_MODULE,
    DatasetRowValidator,
    check_task_folders,
    check_tasks,
    load_task_data_schema,
    read_task_fields,
    task_fields,
)
from tests.unit_tests.test_harbor_tasks import no_real_registry, write_task  # noqa: F401 (autouse fixture)


QA_SCHEMA = """
from pydantic import BaseModel, Field


class TaskData(BaseModel):
    question: str
    answer: str = Field(min_length=1)
    difficulty: int = 1
"""

# Fields of every JSON type: strict validation refuses a value of another type where lax mode would coerce it.
TYPED_SCHEMA = """
import datetime
import enum

from pydantic import BaseModel


class Level(enum.Enum):
    EASY = "easy"
    HARD = "hard"


class TaskData(BaseModel):
    flag: bool
    count: int
    ratio: float = 1.0
    level: Level = Level.EASY
    due: datetime.datetime | None = None
    tags: list[str] = []
"""

ALIASED_SCHEMA = """
from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    expected: str = Field(alias="expected-answer")
"""

ALIAS_CHOICES_SCHEMA = """
from pydantic import AliasChoices, AliasPath, BaseModel, Field


class TaskData(BaseModel):
    question: str
    answer: str = Field(validation_alias=AliasChoices("answer", "expected_answer", AliasPath("meta", "answer")))
    level: int = Field(default=1, validation_alias=AliasPath("meta", "level"))
"""

# A field read by its alias alone: pydantic ignores a `difficulty` key and takes the default.
DEFAULTED_ALIAS_SCHEMA = """
from pydantic import BaseModel, Field


class TaskData(BaseModel):
    difficulty: int = Field(default=1, alias="level")
"""

# An author's own strict configuration: in Python mode it refuses an ISO string for a datetime, which the JSON
# pass accepts, so the key check must not let it decide the values.
STRICT_CONFIG_SCHEMA = """
import datetime

from pydantic import BaseModel, ConfigDict


class TaskData(BaseModel):
    model_config = ConfigDict(strict=True)

    due: datetime.datetime
"""

# A nested model the author made strict: an extra key inside `meta` is its error, not a key of TaskData.
NESTED_FORBID_SCHEMA = """
from pydantic import BaseModel, ConfigDict


class Meta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: int = 1


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str


class TaskData(BaseModel):
    question: str
    meta: Meta
    items: list[Item] = []
"""

# A validator that lets a KeyError escape for a unit it does not know: not a pydantic ValidationError.
RAISING_SCHEMA = """
from pydantic import BaseModel, field_validator

UNITS = {"cm": 1.0, "m": 100.0}


class TaskData(BaseModel):
    answer: str
    unit: str

    @field_validator("unit")
    @classmethod
    def known_unit(cls, v):
        UNITS[v]
        return v
"""


def write_dataset(root: Path, *, schema: str | None = QA_SCHEMA, tasks: dict[str, dict | None] | None = None) -> Path:
    """A dataset folder with ``task_data.py`` and one task per entry; an entry's dict is its ``task_data.json``."""
    root.mkdir(parents=True, exist_ok=True)
    if schema is not None:
        (root / TASK_DATA_MODULE).write_text(schema)
    for name, fields in (tasks or {"q1": {"question": "2+2?", "answer": "4"}}).items():
        task = write_task(root / name)
        if fields is not None:
            (task / TASK_DATA_FILE).write_text(json.dumps(fields))
    return root


class TestLoadSchema:
    def test_dataset_without_the_module_has_no_schema(self, tmp_path):
        assert load_task_data_schema(tmp_path) is None
        assert read_task_fields(tmp_path) == {}

    def test_module_exporting_a_model_loads(self, tmp_path):
        (tmp_path / TASK_DATA_MODULE).write_text(QA_SCHEMA)
        schema = load_task_data_schema(tmp_path)
        assert isinstance(schema, TaskDataSchema) and issubclass(schema.model, BaseModel)
        assert schema.path == tmp_path / TASK_DATA_MODULE
        assert set(schema.model.model_fields) == {"question", "answer", "difficulty"}
        validated = schema.validate({"question": "q", "answer": "a", DIGEST_KEY: "d"}, task_id="t")
        assert validated.model_dump() == {"question": "q", "answer": "a", "difficulty": 1}

    def test_two_datasets_with_the_same_folder_name_keep_their_own_models(self, tmp_path):
        a = tmp_path / "one" / "ds"
        b = tmp_path / "two" / "ds"
        a.mkdir(parents=True)
        b.mkdir(parents=True)
        (a / TASK_DATA_MODULE).write_text(QA_SCHEMA)
        (b / TASK_DATA_MODULE).write_text(ALIASED_SCHEMA)
        first = load_task_data_schema(a)
        second = load_task_data_schema(b)
        assert set(first.model.model_fields) == {"question", "answer", "difficulty"}
        assert set(second.model.model_fields) == {"expected"}
        assert first.model is not second.model and first.model.__module__ != second.model.__module__

    def test_import_failure_names_the_file_and_the_error(self, tmp_path):
        (tmp_path / TASK_DATA_MODULE).write_text("from pydantic import BaseModel\nraise RuntimeError('boom')\n")
        with pytest.raises(TaskDataError, match=r"task_data\.py: failed to import: RuntimeError: boom"):
            load_task_data_schema(tmp_path)

    def test_syntax_error_is_an_import_failure(self, tmp_path):
        (tmp_path / TASK_DATA_MODULE).write_text("class TaskData(\n")
        with pytest.raises(TaskDataError, match="failed to import: SyntaxError"):
            load_task_data_schema(tmp_path)

    def test_missing_export_is_refused(self, tmp_path):
        (tmp_path / TASK_DATA_MODULE).write_text("from pydantic import BaseModel\nclass Row(BaseModel):\n    x: int\n")
        with pytest.raises(TaskDataError, match="does not export `TaskData`"):
            load_task_data_schema(tmp_path)

    def test_export_that_is_not_a_model_class_is_refused(self, tmp_path):
        (tmp_path / TASK_DATA_MODULE).write_text("TaskData = {'question': str}\n")
        with pytest.raises(TaskDataError, match="must be a Pydantic BaseModel subclass"):
            load_task_data_schema(tmp_path)

    def test_errors_are_harbor_task_errors(self, tmp_path):
        # The CLI's error handling catches HarborTaskError, so a schema problem exits cleanly like a task problem.
        assert issubclass(TaskDataError, HarborTaskError)


class TestValidate:
    def _schema(self, tmp_path, text=QA_SCHEMA):
        (tmp_path / TASK_DATA_MODULE).write_text(text)
        return load_task_data_schema(tmp_path)

    def test_reserved_keys_are_not_task_fields(self, tmp_path):
        schema = self._schema(tmp_path)
        row = {DIGEST_KEY: "abc", DATASET_DIGEST_KEY: "def", "question": "q", "answer": "a"}
        assert task_fields(row) == {"question": "q", "answer": "a"}
        schema.validate(row, task_id="t")

    def test_violation_names_the_task_the_file_and_the_field(self, tmp_path):
        schema = self._schema(tmp_path)
        with pytest.raises(TaskDataError) as excinfo:
            schema.validate({"question": "q", "answer": ""}, task_id="gsm8k-0001")
        message = str(excinfo.value)
        assert message.startswith("task 'gsm8k-0001': task_data does not satisfy TaskData in ")
        assert str(tmp_path / TASK_DATA_MODULE) in message
        assert "answer: String should have at least 1 character" in message

    def test_missing_required_field_is_a_violation(self, tmp_path):
        schema = self._schema(tmp_path)
        with pytest.raises(TaskDataError, match="question: Field required"):
            schema.validate({"answer": "a"}, task_id="t")

    def test_values_are_validated_strictly_in_json_terms_never_coerced(self, tmp_path):
        # The row and the verifier's task_data.json carry the values as the file holds them, never the model's
        # dump: lax mode would accept {"flag": "no", "count": "3"} (reading False and 3) while a verifier reads the
        # strings, "no" truthy among them. Acceptance has to mean the values already have the fields' JSON types.
        schema = self._schema(tmp_path, TYPED_SCHEMA)
        with pytest.raises(TaskDataError) as excinfo:
            schema.validate({"flag": "no", "count": "3"}, task_id="t")
        message = str(excinfo.value)
        assert message.startswith("task 't': task_data does not satisfy TaskData in ")
        assert "flag: Input should be a valid boolean" in message
        assert "count: Input should be a valid integer" in message
        for fields in (
            {"flag": 0, "count": 3},
            {"flag": False, "count": 3.0},
            {"flag": False, "count": 3, "ratio": "1"},
        ):
            with pytest.raises(TaskDataError, match="task 't': task_data does not satisfy TaskData in"):
                schema.validate(fields, task_id="t")
        # JSON's own spellings of the model's types are the values a JSON file can hold, and pass: an ISO string
        # for a datetime, an enum's value, an integer where a float is declared.
        model = schema.validate(
            {"flag": False, "count": 3, "ratio": 2, "level": "hard", "due": "2026-10-24T00:00:00Z", "tags": ["a"]},
            task_id="t",
        )
        assert (model.flag, model.count, model.ratio, model.level.value, model.due.year) == (
            False,
            3,
            2.0,
            "hard",
            2026,
        )
        # The author's model is untouched: on its own it still coerces, as pydantic does.
        assert schema.model.model_validate({"flag": "no", "count": "3"}).flag is False

    def test_an_authors_strict_config_does_not_refuse_json_spellings_at_the_key_check(self, tmp_path):
        # The key check validates the fields as Python values and inherits the author's model_config; forced lax, an
        # author's strict=True (which refuses an ISO string for a datetime in Python mode) leaves the value types
        # to the strict JSON pass, whose spellings are what a task_data.json can hold.
        schema = self._schema(tmp_path, STRICT_CONFIG_SCHEMA)
        assert schema.validate({"due": "2026-10-24T00:00:00Z"}, task_id="t").due.year == 2026
        # The JSON pass still refuses a value of another JSON type, and the key check still refuses an unknown key.
        with pytest.raises(TaskDataError, match="due: Input should be a valid datetime"):
            schema.validate({"due": 20261024}, task_id="t")
        with pytest.raises(TaskDataError, match=r"task 't': task_data has keys \['dew'\] that TaskData in"):
            schema.validate({"due": "2026-10-24T00:00:00Z", "dew": "x"}, task_id="t")
        # The author's model is untouched: on its own it is as strict as it was written.
        with pytest.raises(ValidationError):
            schema.model.model_validate({"due": "2026-10-24T00:00:00Z"})

    def test_unknown_keys_fail_whatever_the_models_extra_setting(self, tmp_path):
        # extra="allow" would keep `anwser` as an extra; the dataset's model is the complete description.
        schema = self._schema(tmp_path, ALIASED_SCHEMA)
        with pytest.raises(
            TaskDataError, match=r"task 't': task_data has keys \['anwser'\] that TaskData in .* does not declare"
        ):
            schema.validate({"expected-answer": "4", "anwser": "4"}, task_id="t")
        # Both the alias and the field name are declared keys.
        schema.validate({"expected-answer": "4"}, task_id="t")
        schema.validate({"expected": "4"}, task_id="t")

    def test_alias_choices_and_alias_paths_are_read(self, tmp_path):
        # Every name pydantic itself reads for a field is declared: a plain alias, each AliasChoices entry and the
        # top-level key of an AliasPath. A row the model validates is never refused as carrying undeclared keys.
        schema = self._schema(tmp_path, ALIAS_CHOICES_SCHEMA)
        for row in (
            {"question": "q", "answer": "4"},
            {"question": "q", "expected_answer": "4"},
            {"question": "q", "meta": {"answer": "4", "level": 2}},
        ):
            assert schema.model.model_validate(row).answer == "4"
            assert schema.validate(row, task_id="t").answer == "4"
        assert schema.validate({"question": "q", "meta": {"answer": "4", "level": 2}}, task_id="t").level == 2
        with pytest.raises(TaskDataError, match=r"task 't': task_data has keys \['expected_anwser'\]"):
            schema.validate({"question": "q", "expected_anwser": "4"}, task_id="t")

    def test_a_name_the_model_does_not_read_is_undeclared(self, tmp_path):
        # `difficulty` is addressed by its alias alone (no populate_by_name): pydantic would ignore a `difficulty` key
        # and take the default, leaving the value unvalidated while the row keeps it. That key is refused, the alias
        # is read and validated.
        schema = self._schema(tmp_path, DEFAULTED_ALIAS_SCHEMA)
        assert schema.validate({"level": 2}, task_id="t").difficulty == 2
        assert schema.validate({}, task_id="t").difficulty == 1
        with pytest.raises(TaskDataError, match=r"task 't': task_data has keys \['difficulty'\] that TaskData in"):
            schema.validate({"difficulty": "not an integer"}, task_id="t")
        with pytest.raises(TaskDataError, match=r"task 't': task_data has keys \['difficulty'\] that TaskData in"):
            schema.validate({"difficulty": 2}, task_id="t")
        with pytest.raises(TaskDataError, match="level: Input should be a valid integer"):
            schema.validate({"level": "not an integer"}, task_id="t")
        # The author's model is untouched: it still ignores the key, as pydantic does.
        assert schema.model.model_validate({"difficulty": "not an integer"}).difficulty == 1
        assert (
            issubclass(schema.strict_model, schema.model) and schema.strict_model.__module__ == schema.model.__module__
        )

    def test_a_second_spelling_of_a_field_already_given_is_undeclared(self, tmp_path):
        # pydantic reads the first AliasChoices alternative present and never looks at the other one, so a value
        # under it would go unvalidated; likewise a field's name beside its alias under populate_by_name.
        schema = self._schema(tmp_path, ALIAS_CHOICES_SCHEMA)
        with pytest.raises(TaskDataError, match=r"task 't': task_data has keys \['expected_answer'\]"):
            schema.validate({"question": "q", "answer": "4", "expected_answer": 5}, task_id="t")
        with pytest.raises(TaskDataError, match=r"task 't': task_data has keys \['meta'\]"):
            schema.validate({"question": "q", "answer": "4", "meta": {"answer": "5"}}, task_id="t")
        # `meta` read by `level` alone is consumed: not a second spelling of `answer`.
        assert schema.validate({"question": "q", "answer": "4", "meta": {"level": 3}}, task_id="t").level == 3
        (tmp_path / "aliased").mkdir()
        aliased = self._schema(tmp_path / "aliased", ALIASED_SCHEMA)
        with pytest.raises(TaskDataError, match=r"task 't': task_data has keys \['expected'\]"):
            aliased.validate({"expected-answer": "4", "expected": "5"}, task_id="t")
        # Undeclared keys are reported before any value error, as before.
        with pytest.raises(TaskDataError, match=r"task 't': task_data has keys \['typo'\]"):
            schema.validate({"question": "q", "answer": 4, "typo": 1}, task_id="t")

    def test_an_extra_key_inside_a_nested_model_is_that_models_error_not_an_undeclared_key(self, tmp_path):
        # `meta` is declared; the extra lives inside the author's own extra="forbid" model and is reported where
        # pydantic found it, beside the other value errors, instead of claiming TaskData does not declare `meta`.
        schema = self._schema(tmp_path, NESTED_FORBID_SCHEMA)
        assert schema.validate({"question": "q", "meta": {"level": 2}}, task_id="t").meta.level == 2
        with pytest.raises(TaskDataError) as excinfo:
            schema.validate({"question": "q", "meta": {"level": 1, "bogus": 2}}, task_id="t")
        message = str(excinfo.value)
        assert message.startswith("task 't': task_data does not satisfy TaskData in ")
        assert "meta.bogus: Extra inputs are not permitted" in message
        assert "does not declare" not in message and "['meta']" not in message
        with pytest.raises(TaskDataError) as excinfo:
            schema.validate({"question": "q", "meta": {"level": "x", "bogus": 2}}, task_id="t")
        message = str(excinfo.value)
        assert "meta.level: Input should be a valid integer" in message
        assert "meta.bogus: Extra inputs are not permitted" in message
        with pytest.raises(TaskDataError, match=r"items\.0\.junk: Extra inputs are not permitted"):
            schema.validate({"question": "q", "meta": {}, "items": [{"name": "a", "junk": 1}]}, task_id="t")
        # A key of the row itself is still undeclared, and reported first.
        with pytest.raises(TaskDataError, match=r"task 't': task_data has keys \['typo'\] that TaskData in"):
            schema.validate({"question": "q", "meta": {"bogus": 2}, "typo": 1}, task_id="t")

    def test_a_validator_that_raises_is_a_task_data_error_naming_the_task(self, tmp_path):
        # Pydantic wraps only ValueError, AssertionError and PydanticCustomError; a KeyError (an author's `UNITS[v]`)
        # escapes model_validate raw. Every caller handles TaskDataError alone, so it has to be one here.
        schema = self._schema(tmp_path, RAISING_SCHEMA)
        assert schema.validate({"answer": "4", "unit": "cm"}, task_id="q1").unit == "cm"
        with pytest.raises(TaskDataError) as excinfo:
            schema.validate({"answer": "4", "unit": "inch", DIGEST_KEY: "x"}, task_id="q1")
        message = str(excinfo.value)
        assert message == (
            f"task 'q1': TaskData in {tmp_path / TASK_DATA_MODULE} raised KeyError: 'inch' while validating task_data"
        )
        assert isinstance(excinfo.value.__cause__, KeyError)
        folder = tmp_path / "q1"
        folder.mkdir()
        (folder / TASK_DATA_FILE).write_text(json.dumps({"answer": "4", "unit": "inch"}))
        with pytest.raises(TaskDataError, match="task 'q1': TaskData in .* raised KeyError: 'inch'"):
            check_task_folders(schema, [folder])

    def test_check_tasks_without_a_schema_is_a_no_op(self):
        check_tasks(None, [object()])
        check_task_folders(None, [Path("/nope")])

    def test_check_task_folders_leaves_a_file_that_does_not_read_to_the_loader(self, tmp_path):
        # A task_data.json that is not a JSON object is that task's load error (the loader skips it), not a dataset
        # refusal; a folder without the file has no fields, which the model may refuse.
        schema = self._schema(tmp_path)
        broken, missing, fine = tmp_path / "broken", tmp_path / "missing", tmp_path / "fine"
        for folder in (broken, missing, fine):
            folder.mkdir()
        (broken / TASK_DATA_FILE).write_text("[1, 2]")
        (fine / TASK_DATA_FILE).write_text(json.dumps({"question": "q", "answer": "a"}))
        check_task_folders(schema, [broken, fine])
        with pytest.raises(TaskDataError, match="task 'missing': .*question: Field required"):
            check_task_folders(schema, [broken, fine, missing])


class TestLoaderReadsTaskFields:
    def test_task_without_the_file_has_no_fields(self, tmp_path):
        task = load_task(write_task(tmp_path / "ds" / "a"))
        assert task.task_data == {}
        row = materialize_task(task, "ds").model_dump(mode="json", exclude_unset=True)
        assert row["task_input"]["task_data"] == {DIGEST_KEY: task.digest}

    def test_fields_enter_the_row_under_the_reserved_keys(self, tmp_path):
        folder = write_task(tmp_path / "ds" / "a")
        (folder / TASK_DATA_FILE).write_text(json.dumps({"question": "2+2?", "answer": "4"}))
        task = load_task(folder)
        assert task.task_data == {"question": "2+2?", "answer": "4"}
        row = materialize_task(task, "ds", dataset_digest="dd").model_dump(mode="json", exclude_unset=True)
        assert row["task_input"]["task_data"] == {
            "question": "2+2?",
            "answer": "4",
            DIGEST_KEY: task.digest,
            DATASET_DIGEST_KEY: "dd",
        }

    def test_the_file_is_part_of_the_task_digest(self, tmp_path):
        folder = write_task(tmp_path / "ds" / "a")
        before = content_hash(folder)
        (folder / TASK_DATA_FILE).write_text(json.dumps({"answer": "4"}))
        after = content_hash(folder)
        assert before != after
        (folder / TASK_DATA_FILE).write_text(json.dumps({"answer": "5"}))
        assert content_hash(folder) not in (before, after)

    @pytest.mark.parametrize(
        ("text", "fragment"),
        [
            ("[1, 2]", "must hold a JSON object, got list"),
            ("{not json", "task_data.json: "),
            ('{"_ng_digest": "x", "answer": "4"}', "keys ['_ng_digest'] are reserved"),
        ],
    )
    def test_malformed_files_fail_the_task_load(self, tmp_path, text, fragment):
        folder = write_task(tmp_path / "ds" / "a")
        (folder / TASK_DATA_FILE).write_text(text)
        with pytest.raises(HarborTaskError, match=fragment.replace("[", r"\[").replace("]", r"\]")):
            load_task(folder)


class TestMaterializeRefusesTheDataset:
    def test_prepare_writes_the_fields_and_accepts_a_valid_dataset(self, tmp_path):
        folder = write_dataset(tmp_path / "ds", tasks={"q1": {"question": "2+2?", "answer": "4", "difficulty": 2}})
        prepared = prepare_target(str(folder), output_root=tmp_path / "out")
        rows = [json.loads(line) for line in prepared.rows_path.read_text().splitlines()]
        assert rows[0]["task_input"]["task_data"] == {
            "question": "2+2?",
            "answer": "4",
            "difficulty": 2,
            DIGEST_KEY: prepared.tasks[0].digest,
            DATASET_DIGEST_KEY: prepared.dataset_digest,
        }

    def test_a_violating_task_refuses_the_dataset_naming_it(self, tmp_path):
        folder = write_dataset(
            tmp_path / "ds",
            tasks={"q1": {"question": "2+2?", "answer": "4"}, "q2": {"question": "3+3?", "answer": 6}},
        )
        with pytest.raises(
            TaskDataError,
            match="task 'q2': task_data does not satisfy TaskData in .*: answer: Input should be a valid string",
        ):
            prepare_target(str(folder), output_root=tmp_path / "out")
        # The dataset is refused as a whole: no rows were written for the valid task either.
        assert not (tmp_path / "out" / "ds" / "tasks.jsonl").exists()

    def test_a_value_of_another_json_type_refuses_the_dataset(self, tmp_path):
        # `difficulty: int` given "2": lax validation would pass the row while it (and the verifier's copy) keep the
        # string. Strict validation refuses the dataset at prepare, before any row is written.
        folder = write_dataset(tmp_path / "ds", tasks={"q1": {"question": "2+2?", "answer": "4", "difficulty": "2"}})
        with pytest.raises(TaskDataError, match="task 'q1': .*difficulty: Input should be a valid integer"):
            prepare_target(str(folder), output_root=tmp_path / "out")
        assert not (tmp_path / "out" / "ds" / "tasks.jsonl").exists()

    def test_a_validator_that_raises_refuses_the_dataset_as_a_harbor_task_error(self, tmp_path):
        # The CLI catches HarborTaskError and OSError: a validator's KeyError must reach it as one, not as a traceback.
        folder = write_dataset(
            tmp_path / "ds",
            schema=RAISING_SCHEMA,
            tasks={"q1": {"answer": "4", "unit": "cm"}, "q2": {"answer": "4", "unit": "inch"}},
        )
        with pytest.raises(HarborTaskError, match="task 'q2': TaskData in .* raised KeyError: 'inch'") as excinfo:
            prepare_target(str(folder), output_root=tmp_path / "out")
        assert isinstance(excinfo.value, TaskDataError)
        assert not (tmp_path / "out" / "ds" / "tasks.jsonl").exists()

    def test_a_task_without_the_file_is_a_violation_when_the_model_requires_fields(self, tmp_path):
        folder = write_dataset(tmp_path / "ds", tasks={"q1": None})
        with pytest.raises(TaskDataError, match="task 'q1': .*question: Field required"):
            prepare_target(str(folder), output_root=tmp_path / "out")

    def test_an_unselected_task_still_refuses_the_dataset(self, tmp_path):
        folder = write_dataset(tmp_path / "ds", tasks={"q1": {"question": "q", "answer": "a"}, "q2": {"nope": 1}})
        with pytest.raises(TaskDataError, match="task 'q2'"):
            prepare_target(str(folder), output_root=tmp_path / "out", only=["q1"])

    def test_a_module_that_does_not_import_refuses_the_dataset_before_images_are_recorded(self, tmp_path):
        from tests.unit_tests.test_harbor_tasks import FakeRegistryClient

        folder = write_dataset(tmp_path / "ds", schema="import nothing_of_the_sort\n")
        with pytest.raises(TaskDataError, match="failed to import: ModuleNotFoundError"):
            prepare_target(str(folder), output_root=tmp_path / "out")
        assert FakeRegistryClient.resolved == []

    def test_a_violating_task_refuses_the_dataset_before_images_are_recorded_or_built(self, tmp_path, monkeypatch):
        # The fields need no loaded task, so a dataset refused as a whole does not first resolve every base image
        # against the registry and build and push every Dockerfile; nested task folders are walked like the loader's.
        from nemo_gym.data.tasks.harbor import cli as cli_module
        from tests.unit_tests.test_harbor_tasks import FakeRegistryClient

        monkeypatch.setattr(FakeRegistryClient, "resolved", [])
        builds: list[Path] = []
        real_build = cli_module._build_dockerfiles

        def recording_build(folder, *args, **kwargs):
            builds.append(folder)
            return real_build(folder, *args, **kwargs)

        monkeypatch.setattr(cli_module, "_build_dockerfiles", recording_build)
        folder = write_dataset(tmp_path / "ds", tasks={"q1": {"question": "2+2?", "answer": "4"}})
        nested = write_task(folder / "group" / "q2")
        (nested / TASK_DATA_FILE).write_text(json.dumps({"question": "3+3?", "answer": 6}))
        with pytest.raises(
            TaskDataError, match="task 'q2': task_data does not satisfy TaskData .*answer: Input should"
        ):
            prepare_target(str(folder), output_root=tmp_path / "out")
        assert FakeRegistryClient.resolved == [] and builds == []
        assert not (tmp_path / "out" / "ds" / "tasks.jsonl").exists()
        # A task_data.json that is not an object is left to the loader (that task is skipped, named); the rest prepare.
        (nested / TASK_DATA_FILE).write_text("[1]")
        prepared = prepare_target(str(folder), output_root=tmp_path / "out")
        assert [task.task_id for task in prepared.tasks] == ["q1"] and "q2" in prepared.skipped
        assert FakeRegistryClient.resolved and builds == [folder]

    def test_a_dataset_without_the_module_behaves_as_before(self, tmp_path):
        folder = write_dataset(tmp_path / "ds", schema=None, tasks={"q1": {"anything": [1, 2]}, "q2": None})
        prepared = prepare_target(str(folder), output_root=tmp_path / "out")
        rows = {
            json.loads(line)["task_id"]["task_id"]: json.loads(line)
            for line in prepared.rows_path.read_text().splitlines()
        }
        assert rows["q1"]["task_input"]["task_data"]["anything"] == [1, 2]
        assert set(rows["q2"]["task_input"]["task_data"]) == {DIGEST_KEY, DATASET_DIGEST_KEY}

    def test_a_single_task_folder_target_reads_the_schema_beside_it(self, tmp_path):
        folder = write_dataset(tmp_path / "ds", tasks={"q1": {"question": "q", "answer": 4}})
        with pytest.raises(TaskDataError, match="task 'q1'"):
            prepare_target(str(folder / "q1"), output_root=tmp_path / "out")


class TestRowMatchesFolder:
    def test_equal_fields_pass_and_differences_are_named(self, tmp_path):
        folder = write_task(tmp_path / "ds" / "a")
        (folder / TASK_DATA_FILE).write_text(json.dumps({"question": "q", "answer": "4"}))
        task = load_task(folder)
        check_row_matches_folder({DIGEST_KEY: task.digest, "question": "q", "answer": "4"}, task)
        with pytest.raises(TaskDataError) as excinfo:
            check_row_matches_folder({DIGEST_KEY: task.digest, "question": "q", "answer": "5", "hint": 1}, task)
        message = str(excinfo.value)
        assert "task 'a': the row's task_data differs from the task folder's task_data.json" in message
        assert "keys not in the folder: ['hint']" in message and "values that differ: ['answer']" in message
        with pytest.raises(TaskDataError, match=r"keys missing from the row: \['answer'\]"):
            check_row_matches_folder({DIGEST_KEY: task.digest, "question": "q"}, task)


class TestCollate:
    """``gym dataset collate`` validates a Harbor taskset's rows against the dataset's TaskData too."""

    def _instances(self, folder: Path, rows_path: Path, tasks, agent_name: str | None = None):
        """The generated run config's agent and environment instances for the taskset at ``folder``.

        ``agent_name`` replaces the generated per-taskset agent instance by a shared one: the environment server
        then names that agent, whose datasets the caller assembles.
        """
        config = run_config(
            taskset=folder.name,
            folder=folder,
            tasks=tasks,
            rows_path=rows_path,
            agent_instance_name="hermes_agent",
            agent_impl="hermes_agent",
        )
        agent_name = agent_name or agent_instance(folder.name)
        agent_block = config[agent_instance(folder.name)]["responses_api_agents"]
        env_block = dict(config[environment_server_instance(folder.name)]["environment_servers"]["harbor"])
        env_block["agent_server"] = {"type": "responses_api_agents", "name": agent_name}
        agent = ResponsesAPIAgentServerInstanceConfig(
            name=agent_name,
            server_type_config_dict=OmegaConf.create({}),
            responses_api_agents={"hermes_agent": {"entrypoint": "app.py", **agent_block["hermes_agent"]}},
        )
        env = EnvironmentServerInstanceConfig(
            name=environment_server_instance(folder.name),
            server_type_config_dict=OmegaConf.create({}),
            environment_servers={"harbor": {"entrypoint": "app.py", **env_block}},
        )
        return agent, env

    def _validate(self, folder: Path, rows_path: Path, tasks):
        agent, env = self._instances(folder, rows_path, tasks)
        dataset = agent.datasets[0]
        validator = TrainDataProcessor._task_data_validator_for(agent, dataset, [agent, env])
        for index, line in enumerate(rows_path.read_text().splitlines()):
            validator.validate_row(index, json.loads(line))
        return validator

    def test_valid_rows_are_clean_and_the_datasets_fields_are_not_unknown(self, tmp_path, monkeypatch):
        monkeypatch.chdir(Path(__file__).resolve().parents[2])  # find environment_servers/harbor/task_data.py
        folder = write_dataset(tmp_path / "ds", tasks={"q1": {"question": "2+2?", "answer": "4"}})
        tasks = [load_task(folder / "q1", root=folder)]
        rows_path = tmp_path / "out" / "tasks.jsonl"
        write_rows(tasks, "ds", rows_path, dataset_digest="dd")
        validator = self._validate(folder, rows_path, tasks)
        assert isinstance(validator, DatasetRowValidator)
        assert validator.report.rows == 1 and validator.report.clean, validator.report.summary()

    def test_two_environment_servers_sharing_one_agent_each_validate_their_own_taskset(self, tmp_path, monkeypatch):
        # One harness instance serves two tasksets: both environment servers name it as their agent_server and the
        # agent block declares one dataset per taskset. Each dataset's schema is the one beside the taskset that
        # routes it, whichever environment server comes first in the config; the second dataset is not left to the
        # server's validator, which would report the dataset's own fields as unknown keys.
        monkeypatch.chdir(Path(__file__).resolve().parents[2])
        environments, datasets, rows_paths = [], [], {}
        for taskset, answer in (("ds-a", "4"), ("ds-b", "6")):
            folder = write_dataset(tmp_path / taskset, tasks={"q1": {"question": "?", "answer": answer}})
            tasks = [load_task(folder / "q1", root=folder)]
            rows_paths[taskset] = tmp_path / "out" / f"{taskset}.jsonl"
            write_rows(tasks, taskset, rows_paths[taskset], dataset_digest="dd")
            agent, env = self._instances(folder, rows_paths[taskset], tasks, agent_name="shared_agent")
            datasets.extend(d.model_dump(mode="json", exclude_unset=True) for d in agent.datasets)
            environments.append(env)
        agent = ResponsesAPIAgentServerInstanceConfig(
            name="shared_agent",
            server_type_config_dict=OmegaConf.create({}),
            responses_api_agents={"hermes_agent": {"entrypoint": "app.py", "datasets": datasets}},
        )
        assert [d.name for d in agent.datasets] == ["ds-a", "ds-b"]
        for order in (environments, environments[::-1]):
            configs = [agent, *order]
            for dataset in agent.datasets:
                validator = TrainDataProcessor._task_data_validator_for(agent, dataset, configs)
                assert isinstance(validator, DatasetRowValidator), (dataset.name, [e.name for e in order])
                assert validator._schema.path == tmp_path / dataset.name / TASK_DATA_MODULE
                for index, line in enumerate(rows_paths[dataset.name].read_text().splitlines()):
                    validator.validate_row(index, json.loads(line))
                assert validator.report.rows == 1 and validator.report.clean, validator.report.summary()
        # The route: the taskset's folder comes from the environment server carrying it; a dataset no environment
        # server carries has none.
        for dataset in agent.datasets:
            assert (
                TrainDataProcessor._taskset_folder(agent, dataset, [agent, *environments]) == tmp_path / dataset.name
            )
        stray = agent.datasets[0].model_copy(update={"name": "ds-c", "taskset": None})
        assert TrainDataProcessor._taskset_folder(agent, stray, [agent, *environments]) is None

    def test_an_edited_row_fails_collation_with_the_task_named(self, tmp_path, monkeypatch):
        monkeypatch.chdir(Path(__file__).resolve().parents[2])
        folder = write_dataset(tmp_path / "ds", tasks={"q1": {"question": "2+2?", "answer": "4"}})
        tasks = [load_task(folder / "q1", root=folder)]
        rows_path = tmp_path / "out" / "tasks.jsonl"
        rows = write_rows(tasks, "ds", rows_path, dataset_digest="dd")
        rows[0]["task_input"]["task_data"]["answer"] = 4
        rows[0]["task_input"]["task_data"]["anwser"] = "4"
        rows_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        validator = self._validate(folder, rows_path, tasks)
        report = validator.report
        assert report.error_rows == 1 and not report.clean
        assert report.errors == [
            f"0: task 'q1': task_data has keys ['anwser'] that TaskData in {folder / TASK_DATA_MODULE} does not declare"
        ]
        # The server's schema sees the reserved keys alone: the typo is the dataset's to report, once.
        assert report.unknown_keys == {}
        assert "task 'q1'" in report.summary()

    def test_a_row_whose_validator_raises_is_reported_not_a_crash(self, tmp_path, monkeypatch):
        # The folder's fields pass; the row was edited to a unit the validator's lookup has no entry for.
        monkeypatch.chdir(Path(__file__).resolve().parents[2])
        folder = write_dataset(tmp_path / "ds", schema=RAISING_SCHEMA, tasks={"q1": {"answer": "4", "unit": "cm"}})
        tasks = [load_task(folder / "q1", root=folder)]
        rows_path = tmp_path / "out" / "tasks.jsonl"
        rows = write_rows(tasks, "ds", rows_path, dataset_digest="dd")
        rows[0]["task_input"]["task_data"]["unit"] = "inch"
        rows.append(json.loads(json.dumps(rows[0])))
        rows[1]["task_input"]["task_data"]["unit"] = "m"
        rows_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        report = self._validate(folder, rows_path, tasks).report
        assert report.rows == 2 and report.error_rows == 1
        assert report.errors == [
            f"0: task 'q1': TaskData in {folder / TASK_DATA_MODULE} raised KeyError: 'inch' while validating task_data"
        ]

    def test_a_row_both_schemas_reject_is_counted_once_with_both_diagnostics(self, tmp_path, monkeypatch):
        # The server's schema fails the row (a non-string _ng_digest) and so does the dataset's model (a non-string
        # answer): two diagnostics, one failed row, not `2/1 rows failed`.
        monkeypatch.chdir(Path(__file__).resolve().parents[2])
        folder = write_dataset(tmp_path / "ds", tasks={"q1": {"question": "2+2?", "answer": "4"}})
        tasks = [load_task(folder / "q1", root=folder)]
        rows_path = tmp_path / "out" / "tasks.jsonl"
        rows = write_rows(tasks, "ds", rows_path, dataset_digest="dd")
        rows[0]["task_input"]["task_data"][DIGEST_KEY] = 123
        rows[0]["task_input"]["task_data"]["answer"] = 4
        rows.append(json.loads(json.dumps(rows[0])))
        rows[1]["task_input"]["task_data"][DIGEST_KEY] = tasks[0].digest  # the dataset's model alone rejects this one
        rows_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        report = self._validate(folder, rows_path, tasks).report
        assert report.rows == 2 and report.error_rows == 2
        assert [message.split(":")[0] for message in report.errors] == ["0", "0", "1"]
        assert "_ng_digest" in report.errors[0] and "answer: Input should be a valid string" in report.errors[1]
        assert report.summary().startswith(f"{rows_path}: 2/2 rows failed task_data validation")
        assert "more rows" not in report.summary()

    def test_a_dataset_without_the_module_is_checked_for_the_reserved_keys_only(self, tmp_path, monkeypatch):
        # An untyped dataset's tasks may still carry task_data.json: its fields are the dataset's, not keys the
        # server's schema should report as unknown. The reserved keys are still the server's to check.
        monkeypatch.chdir(Path(__file__).resolve().parents[2])
        folder = write_dataset(tmp_path / "ds", schema=None, tasks={"q1": {"question": "2+2?", "answer": "4"}})
        tasks = [load_task(folder / "q1", root=folder)]
        rows_path = tmp_path / "out" / "tasks.jsonl"
        rows = write_rows(tasks, "ds", rows_path, dataset_digest="dd")
        validator = self._validate(folder, rows_path, tasks)
        assert isinstance(validator, DatasetRowValidator) and validator._schema is None
        assert validator.report.rows == 1 and validator.report.clean, validator.report.summary()
        rows[0]["task_input"]["task_data"][DIGEST_KEY] = 123
        validator.validate_row(1, rows[0])
        assert validator.report.error_rows == 1 and validator.report.unknown_keys == {}
        assert validator.report.errors[0].startswith("1: ") and "_ng_digest" in validator.report.errors[0]
        # A malformed task_data is the server's schema's to report, as for any taskset.
        validator.validate_row(2, {**rows[0], "task_input": {**rows[0]["task_input"], "task_data": 5}})
        assert validator.report.error_rows == 2 and validator.report.errors[1] == (
            "2: task_input.task_data must be an object, got int"
        )

    def test_a_task_field_that_is_a_mapping_is_not_spliced_apart(self, tmp_path, monkeypatch):
        # The dataset's model declares `verifier_metadata` as a mapping: the server's legacy-format normalization
        # would splice its contents up and report `answer` as unknown. The dataset's model validates it whole.
        monkeypatch.chdir(Path(__file__).resolve().parents[2])
        schema = (
            "from pydantic import BaseModel\n\n\nclass TaskData(BaseModel):\n    verifier_metadata: dict[str, str]\n"
        )
        folder = write_dataset(tmp_path / "ds", schema=schema, tasks={"q1": {"verifier_metadata": {"answer": "4"}}})
        tasks = [load_task(folder / "q1", root=folder)]
        rows_path = tmp_path / "out" / "tasks.jsonl"
        rows = write_rows(tasks, "ds", rows_path, dataset_digest="dd")
        validator = self._validate(folder, rows_path, tasks)
        assert isinstance(validator, DatasetRowValidator)
        assert validator.report.rows == 1 and validator.report.clean, validator.report.summary()
        rows[0]["task_input"]["task_data"]["verifier_metadata"]["answer"] = 4
        validator.validate_row(1, rows[0])
        assert validator.report.error_rows == 1 and validator.report.unknown_keys == {}
        assert validator.report.errors == [
            f"1: task 'q1': task_data does not satisfy TaskData in {folder / TASK_DATA_MODULE}: "
            "verifier_metadata.answer: Input should be a valid string"
        ]

    def _unloadable_dataset(self, tmp_path, monkeypatch):
        monkeypatch.chdir(Path(__file__).resolve().parents[2])
        folder = write_dataset(tmp_path / "ds", schema="raise ImportError('no')\n", tasks={"q1": None})
        tasks = [load_task(folder / "q1", root=folder)]
        rows_path = tmp_path / "out" / "tasks.jsonl"
        rows = write_rows(tasks, "ds", rows_path)
        agent, env = self._instances(folder, rows_path, tasks)
        return agent, env, rows

    def test_a_module_that_does_not_import_is_a_warning_in_warn_mode_and_the_reserved_keys_are_still_checked(
        self, tmp_path, monkeypatch
    ):
        # The default `warn` collation reports and goes on, as it does for a server schema that does not load: the
        # dataset's fields go unchecked (the seed will refuse every episode of the taskset, with the same error), the
        # server's reserved keys are checked as for an untyped dataset.
        agent, env, rows = self._unloadable_dataset(tmp_path, monkeypatch)
        with pytest.warns(UserWarning, match=r"task_data\.py does not load.*failed to import: ImportError: no"):
            validator = TrainDataProcessor._task_data_validator_for(agent, agent.datasets[0], [agent, env])
        assert isinstance(validator, DatasetRowValidator) and validator._schema is None
        validator.validate_row(0, rows[0])
        assert validator.report.rows == 1 and validator.report.clean
        rows[0]["task_input"]["task_data"][DIGEST_KEY] = 123
        validator.validate_row(1, rows[0])
        assert validator.report.error_rows == 1 and "_ng_digest" in validator.report.errors[0]

    def test_a_module_that_does_not_import_fails_the_collation_in_error_mode(self, tmp_path, monkeypatch):
        agent, env, _ = self._unloadable_dataset(tmp_path, monkeypatch)
        with pytest.raises(TaskDataError, match="failed to import: ImportError: no"):
            TrainDataProcessor._task_data_validator_for(
                agent, agent.datasets[0], [agent, env], task_data_validation="error"
            )

    def test_the_collation_passes_its_mode_to_the_dataset_schema_load(self, tmp_path, monkeypatch):
        # Through the processor's own loop: `warn` prepares the file under a warning, `error` stops the collation.
        agent, env, _ = self._unloadable_dataset(tmp_path, monkeypatch)
        processor = TrainDataProcessor()
        processor._all_server_instance_configs = [agent, env]
        dataset = agent.datasets[0]
        with pytest.warns(UserWarning, match=r"task_data\.py does not load"):
            paths = processor._collate_samples_single_type(dataset.type, [agent], task_data_validation="warn")
        assert len(paths) == 1 and paths[0].is_file()
        assert json.loads(paths[0].read_text())["task_id"] == {"taskset": "ds", "task_id": "q1"}
        with pytest.raises(TaskDataError, match="failed to import: ImportError: no"):
            processor._collate_samples_single_type(dataset.type, [agent], task_data_validation="error")
