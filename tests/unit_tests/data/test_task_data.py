# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-server task-data schema protocol (nemo_gym/data/task_data.py)."""

import ast
import json
import sys
from pathlib import Path
from typing import get_args

import pytest
from omegaconf import OmegaConf
from pydantic import BaseModel

from nemo_gym.config.config_types import (
    EnvironmentServerInstanceConfig,
    ResourcesServerInstanceConfig,
    ResponsesAPIAgentServerInstanceConfig,
)
from nemo_gym.data import normalize_task_fields
from nemo_gym.data.dataset_preparation import TrainDataProcessor
from nemo_gym.data.task_data import (
    RESERVED_ROW_KEYS,
    TaskDataSchemaError,
    TaskDataValidator,
    find_server_dir,
    load_task_data_schema,
    validate_jsonl_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_FILES = sorted(
    list((REPO_ROOT / "resources_servers").glob("*/task_data.py"))
    + list((REPO_ROOT / "responses_api_agents").glob("*/task_data.py"))
)


class TestNormalizeTaskFields:
    def test_strips_reserved_keys(self):
        row = {k: "x" for k in RESERVED_ROW_KEYS} | {"question": "q"}
        fields, conflicts = normalize_task_fields(row)
        assert fields == {"question": "q"}
        assert conflicts == []

    def test_splices_verifier_metadata_contents(self):
        fields, conflicts = normalize_task_fields(
            {"responses_create_params": {}, "verifier_metadata": {"label": "safe", "type": "homonyms"}}
        )
        assert fields == {"label": "safe", "type": "homonyms"}
        assert conflicts == []

    def test_equal_duplicate_is_harmless(self):
        fields, conflicts = normalize_task_fields({"label": "safe", "verifier_metadata": {"label": "safe"}})
        assert fields == {"label": "safe"}
        assert conflicts == []

    def test_conflicting_duplicate_is_reported_and_top_level_wins(self):
        fields, conflicts = normalize_task_fields({"label": "safe", "verifier_metadata": {"label": "unsafe"}})
        assert fields == {"label": "safe"}
        assert conflicts == ["label"]

    def test_non_dict_verifier_metadata_is_kept_for_the_schema_to_reject(self):
        fields, _ = normalize_task_fields({"verifier_metadata": "oops"})
        assert fields == {"verifier_metadata": "oops"}

    def test_splices_task_data_contents(self):
        fields, conflicts = normalize_task_fields(
            {"responses_create_params": {}, "task_data": {"expected_city": "Tokyo"}}
        )
        assert fields == {"expected_city": "Tokyo"}
        assert conflicts == []

    def test_empty_task_data_container_leaves_no_residue(self):
        fields, conflicts = normalize_task_fields({"expected_city": "Tokyo", "task_data": {}})
        assert fields == {"expected_city": "Tokyo"}
        assert conflicts == []

    def test_conflicting_task_data_duplicate_is_reported(self):
        fields, conflicts = normalize_task_fields({"expected_city": "Paris", "task_data": {"expected_city": "Tokyo"}})
        assert fields == {"expected_city": "Paris"}
        assert conflicts == ["expected_city"]

    def test_non_dict_task_data_is_kept_for_the_schema_to_reject(self):
        fields, _ = normalize_task_fields({"task_data": "oops"})
        assert fields == {"task_data": "oops"}


class TestLoadTaskDataSchema:
    def test_missing_module_returns_none(self, tmp_path):
        assert load_task_data_schema(tmp_path) is None

    def test_module_without_export_raises(self, tmp_path):
        (tmp_path / "task_data.py").write_text("x = 1\n")
        with pytest.raises(TaskDataSchemaError, match="does not export"):
            load_task_data_schema(tmp_path)

    def test_broken_module_raises(self, tmp_path):
        (tmp_path / "task_data.py").write_text("import does_not_exist_anywhere\n")
        with pytest.raises(TaskDataSchemaError, match="Failed to import"):
            load_task_data_schema(tmp_path)

    def test_plain_model_loads(self, tmp_path):
        (tmp_path / "task_data.py").write_text(
            "from pydantic import BaseModel, ConfigDict\n"
            "class TaskData(BaseModel):\n"
            "    model_config = ConfigDict(extra='allow')\n"
            "    question: str\n"
        )
        adapter = load_task_data_schema(tmp_path)
        assert adapter.validate_python({"question": "q"})
        with pytest.raises(Exception):
            adapter.validate_python({"question": 3.14})

    def test_discriminated_union_alias_loads(self, tmp_path):
        (tmp_path / "task_data.py").write_text(
            "from typing import Annotated, Literal, Union\n"
            "from pydantic import BaseModel, ConfigDict, Field\n"
            "class A(BaseModel):\n"
            "    model_config = ConfigDict(extra='allow')\n"
            "    kind: Literal['a']\n"
            "    x: int\n"
            "class B(BaseModel):\n"
            "    model_config = ConfigDict(extra='allow')\n"
            "    kind: Literal['b']\n"
            "    y: str\n"
            "TaskData = Annotated[Union[A, B], Field(discriminator='kind')]\n"
        )
        adapter = load_task_data_schema(tmp_path)
        assert adapter.validate_python({"kind": "a", "x": 1})
        assert adapter.validate_python({"kind": "b", "y": "s"})
        with pytest.raises(Exception):
            adapter.validate_python({"kind": "a", "x": "not-an-int"})


class _Schema(BaseModel):
    # extra="allow" mirrors the protocol default; pydantic's "ignore" is banned in real schemas
    # and would also disable unknown-field reporting (extras are read off __pydantic_extra__).
    model_config = {"extra": "allow"}

    question: str
    expected_answer: str


class TestTaskDataValidator:
    def _validator(self):
        from pydantic import TypeAdapter

        return TaskDataValidator(server_name="s", adapter=TypeAdapter(_Schema), dataset_fpath="d.jsonl")

    def test_clean_rows(self):
        v = self._validator()
        v.validate_row(0, {"responses_create_params": {}, "question": "q", "expected_answer": "a"})
        assert v.report.clean and v.report.rows == 1 and v.report.error_rows == 0

    def test_invalid_row_recorded(self):
        v = self._validator()
        v.validate_row(0, {"question": "q"})  # missing expected_answer
        assert not v.report.clean
        assert v.report.error_rows == 1
        assert "expected_answer" in v.report.errors[0]
        assert "0/" not in v.report.summary().splitlines()[0]

    def test_verifier_metadata_rows_validate_against_flat_schema(self):
        v = self._validator()
        v.validate_row(0, {"verifier_metadata": {"question": "q", "expected_answer": "a"}})
        assert v.report.clean

    def test_unknown_keys_counted_and_dirty_but_not_errored(self):
        v = self._validator()
        v.validate_row(0, {"question": "q", "expected_answer": "a", "difficulty": 3})
        # The row still validates (no error), but an undeclared key is likely a typo or a
        # missing schema field, so the report is not clean.
        assert v.report.error_rows == 0
        assert not v.report.clean
        assert v.report.unknown_keys == {"difficulty": 1}
        assert "difficulty" in v.report.summary()

    def test_conflicting_duplicate_marks_report_dirty(self):
        v = self._validator()
        v.validate_row(0, {"question": "q", "expected_answer": "a", "verifier_metadata": {"question": "other"}})
        assert not v.report.clean
        assert v.report.conflicting_keys == {"question": 1}

    def test_materialized_task_validates_its_task_data(self):
        v = self._validator()
        task = {"taskset": "t", "task_id": "0"}
        v.validate_row(
            0,
            {
                "task_id": task,
                "task_input": {"responses_create_params": {}, "task_data": {"question": "q", "expected_answer": "a"}},
            },
        )
        assert v.report.clean
        v.validate_row(
            1, {"task_id": task, "task_input": {"responses_create_params": {}, "task_data": {"question": 1}}}
        )
        assert v.report.error_rows == 1
        # A multi-turn row carries its requests as `turns` beside task_data: framework-owned, not an unknown field.
        turns = [{"name": "plan", "responses_create_params": {}}]
        fields = {"question": "q", "expected_answer": "a"}
        v.validate_row(2, {"task_id": task, "task_input": {"turns": turns, "task_data": fields}})
        assert v.report.unknown_keys == {} and v.report.error_rows == 1

    def test_generic_materialized_task_validates_flat_fields(self):
        v = self._validator()
        v.validate_row(
            0, {"task_id": {"taskset": "t", "task_id": "0"}, "task_input": {"question": "q", "expected_answer": "a"}}
        )
        assert v.report.clean
        v.validate_row(1, {"task_id": {"taskset": "t", "task_id": "1"}, "task_input": {"question": "q"}})
        assert v.report.error_rows == 1

    @pytest.mark.parametrize("nested", [False, True])
    def test_materialized_verifier_metadata_preserves_schema_normalization(self, nested):
        fields = {"verifier_metadata": {"question": "q", "expected_answer": "a"}}
        task_input = {"responses_create_params": {}, **({"task_data": fields} if nested else fields)}
        v = self._validator()
        v.validate_row(0, {"task_id": {"taskset": "t", "task_id": "0"}, "task_input": task_input})
        assert v.report.clean

    def test_mixed_materialized_task_reports_conflicting_fields(self):
        v = self._validator()
        v.validate_row(
            0,
            {
                "task_id": {"taskset": "t", "task_id": "0"},
                "task_input": {"task_data": {"question": "q", "expected_answer": "a"}, "question": "different"},
            },
        )
        assert v.report.conflicting_keys == {"question": 1}
        assert not v.report.clean

    def test_materialized_task_data_that_is_not_an_object_is_an_invalid_row(self):
        v = self._validator()
        v.validate_row(0, {"task_id": {"taskset": "t", "task_id": "0"}, "task_input": {"task_data": "bad"}})
        assert v.report.error_rows == 1
        assert "task_input.task_data must be an object, got str" in v.report.errors[0]

    @pytest.mark.parametrize("task_data", [[], "", 0, False, None])
    def test_falsey_non_object_task_data_is_an_invalid_row(self, task_data):
        v = self._validator()
        v.validate_row(0, {"task_id": {"taskset": "t", "task_id": "0"}, "task_input": {"task_data": task_data}})
        assert v.report.error_rows == 1

    def test_missing_task_data_is_an_invalid_row(self):
        v = self._validator()
        v.validate_row(0, {"task_id": {"taskset": "t", "task_id": "0"}, "task_input": {"responses_create_params": {}}})
        assert v.report.error_rows == 1

    def test_validate_jsonl_rows_wrapper(self):
        from pydantic import TypeAdapter

        lines = [json.dumps({"question": "q", "expected_answer": "a"}), "", json.dumps({"question": 1})]
        report = validate_jsonl_rows("s", TypeAdapter(_Schema), "d.jsonl", lines)
        assert report.rows == 2 and report.error_rows == 1


class TestOwningResourcesServerImpl:
    def _rs(self, name, impl):
        return ResourcesServerInstanceConfig(
            name=name,
            server_type_config_dict=OmegaConf.create({}),
            resources_servers={impl: {"entrypoint": "app.py", "domain": "other"}},
        )

    def _agent(self, name, rs_ref):
        inner = {"entrypoint": "app.py"} | ({"resources_server": rs_ref} if rs_ref else {})
        return ResponsesAPIAgentServerInstanceConfig(
            name=name,
            server_type_config_dict=OmegaConf.create({}),
            responses_api_agents={"simple_agent": inner},
        )

    def test_resources_server_owns_its_own_data(self):
        rs = self._rs("aime24_rs", "math_with_judge")
        assert TrainDataProcessor._owning_resources_server_impl(rs, [rs]) == "math_with_judge"

    def test_agent_declared_data_follows_the_rs_reference(self):
        rs = self._rs("bench_rs", "mcqa")
        agent = self._agent("bench_agent", {"type": "resources_servers", "name": "bench_rs"})
        assert TrainDataProcessor._owning_resources_server_impl(agent, [agent, rs]) == "mcqa"

    def test_self_contained_agent_is_skipped(self):
        agent = self._agent("tau2_agent", None)
        assert TrainDataProcessor._owning_resources_server_impl(agent, [agent]) is None

    def test_dangling_rs_reference_is_skipped(self):
        agent = self._agent("a", {"type": "resources_servers", "name": "missing"})
        assert TrainDataProcessor._owning_resources_server_impl(agent, [agent]) is None

    def _env(self, name, impl, agent_name):
        return EnvironmentServerInstanceConfig(
            name=name,
            server_type_config_dict=OmegaConf.create({}),
            environment_servers={
                impl: {"entrypoint": "app.py", "agent_server": {"type": "responses_api_agents", "name": agent_name}}
            },
        )

    def test_environment_server_that_names_the_agent_owns_its_data(self):
        # The Harbor layout: datasets on the agent block, episodes driven by the environment server.
        agent = self._agent("harbor_oracle", None)
        env = self._env("harbor_env", "harbor", "harbor_oracle")
        other = self._env("other_env", "other", "someone_else")
        assert TrainDataProcessor._owning_environment_server_impl(agent, [other, agent, env]) == "harbor"
        assert TrainDataProcessor._schema_home_for(agent, [other, agent, env]) == ("harbor", "environment_servers")
        # Nothing names the agent: it falls back to its own folder.
        assert TrainDataProcessor._schema_home_for(agent, [other, agent]) == ("simple_agent", "responses_api_agents")

    def test_resources_server_reference_wins_over_an_environment_server(self):
        rs = self._rs("bench_rs", "mcqa")
        agent = self._agent("bench_agent", {"type": "resources_servers", "name": "bench_rs"})
        env = self._env("env", "harbor", "bench_agent")
        assert TrainDataProcessor._schema_home_for(agent, [agent, rs, env]) == ("mcqa", "resources_servers")
        dangling = self._agent("dangling", {"type": "resources_servers", "name": "missing"})
        assert TrainDataProcessor._schema_home_for(dangling, [dangling, env]) is None


class TestShippedSchemas:
    """CI enforcement for every committed task_data.py (resources servers and agents)."""

    ALLOWED_IMPORT_PREFIXES = ("pydantic", "nemo_gym.data.task_data")

    def test_at_least_the_example_schemas_ship(self):
        names = {p.parent.name for p in SCHEMA_FILES}
        assert {"example_multi_step", "example_mcp_weather", "example_single_tool_call"} <= names

    @pytest.mark.parametrize("schema_file", SCHEMA_FILES, ids=lambda p: p.parent.name)
    def test_imports_are_dependency_light(self, schema_file):
        tree = ast.parse(schema_file.read_text())
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = [node.module or ""]
            for module in modules:
                root = module.split(".")[0]
                allowed = (
                    root in sys.stdlib_module_names
                    or module.startswith(self.ALLOWED_IMPORT_PREFIXES)
                    or (module.startswith("resources_servers.") and module.endswith(".task_data"))
                )
                assert allowed, (
                    f"{schema_file}: import of {module!r} is not allowed in task_data.py "
                    "(stdlib, pydantic, nemo_gym.data.task_data, and other servers' task_data only)"
                )

    @pytest.mark.parametrize("schema_file", SCHEMA_FILES, ids=lambda p: p.parent.name)
    def test_loads_and_bans_extra_ignore(self, schema_file):
        adapter = load_task_data_schema(schema_file.parent)
        assert adapter is not None
        core = getattr(adapter, "_type", None)

        def collect_models(tp, out):
            if isinstance(tp, type) and issubclass(tp, BaseModel):
                out.append(tp)
                return out
            for arg in get_args(tp):  # unwraps Annotated[...] and Union[...] alike
                collect_models(arg, out)
            return out

        models = collect_models(core, [])
        assert models, f"{schema_file}: TaskData must resolve to at least one BaseModel"
        for model in models:
            extra = model.model_config.get("extra")
            assert extra in ("allow", "forbid"), (
                f"{schema_file}: {model.__name__} uses extra={extra!r}; pydantic's silent-drop "
                "'ignore' (the default) is banned in task_data schemas"
            )

    @pytest.mark.parametrize(
        "server",
        [
            "example_multi_step",
            "example_mcp_weather",
            "example_session_state_mgmt",
            "example_multi_turn_gymnasium",
            "example_tool_call_multireward",
            "example_single_tool_call",
        ],
    )
    def test_committed_example_data_validates_clean(self, server):
        server_dir = find_server_dir(server)
        assert server_dir is not None
        adapter = load_task_data_schema(server_dir)
        example = server_dir / "data" / "example.jsonl"
        report = validate_jsonl_rows(server, adapter, str(example), example.read_text().splitlines())
        assert report.rows > 0
        assert report.clean, report.summary()

    def test_mcp_weather_rejects_wrong_type_through_the_splice(self):
        server_dir = find_server_dir("example_mcp_weather")
        adapter = load_task_data_schema(server_dir)
        v = TaskDataValidator(server_name="example_mcp_weather", adapter=adapter, dataset_fpath="x")
        v.validate_row(0, {"verifier_metadata": {"expected_city": 123}})
        assert v.report.error_rows == 1


class TestUnionUnknownKeys:
    def test_union_schema_reports_typoed_optional_field(self, tmp_path):
        (tmp_path / "task_data.py").write_text(
            "from typing import Annotated, Literal, Optional, Union\n"
            "from pydantic import BaseModel, ConfigDict, Field\n"
            "class StringMatch(BaseModel):\n"
            "    model_config = ConfigDict(extra='allow')\n"
            "    eval_type: Literal['string_match']\n"
            "    length: Optional[int] = None\n"
            "class Multichoice(BaseModel):\n"
            "    model_config = ConfigDict(extra='allow')\n"
            "    eval_type: Literal['multichoice']\n"
            "TaskData = Annotated[Union[StringMatch, Multichoice], Field(discriminator='eval_type')]\n"
        )
        adapter = load_task_data_schema(tmp_path)
        v = TaskDataValidator(server_name="ruler2", adapter=adapter, dataset_fpath="d.jsonl")
        v.validate_row(0, {"eval_type": "string_match", "lenght": 5})
        # The typo is surfaced as an unknown key and dirties the report.
        assert v.report.error_rows == 0
        assert not v.report.clean
        assert v.report.unknown_keys == {"lenght": 1}


class TestSelfContainedAgentSchemaFallback:
    def _agent(self, impl, rs_ref):
        inner = {"entrypoint": "app.py"} | ({"resources_server": rs_ref} if rs_ref else {})
        return ResponsesAPIAgentServerInstanceConfig(
            name=f"{impl}_instance",
            server_type_config_dict=OmegaConf.create({}),
            responses_api_agents={impl: inner},
        )

    def _dataset(self):
        from types import SimpleNamespace

        return SimpleNamespace(jsonl_fpath="data/example.jsonl")

    def test_agent_without_rs_reference_uses_its_own_schema(self, tmp_path, monkeypatch):
        agent_dir = tmp_path / "responses_api_agents" / "tau2"
        agent_dir.mkdir(parents=True)
        (agent_dir / "task_data.py").write_text(
            "from pydantic import BaseModel, ConfigDict\n"
            "class TaskData(BaseModel):\n"
            "    model_config = ConfigDict(extra='allow')\n"
            "    task_id: str\n"
        )
        monkeypatch.chdir(tmp_path)
        agent = self._agent("tau2", None)
        validator = TrainDataProcessor._task_data_validator_for(agent, self._dataset(), [agent])
        assert validator is not None
        assert validator.report.server_name == "tau2"

    def test_agent_without_rs_reference_and_no_schema_is_skipped(self, tmp_path, monkeypatch):
        (tmp_path / "responses_api_agents" / "tau2").mkdir(parents=True)
        monkeypatch.chdir(tmp_path)
        agent = self._agent("tau2", None)
        assert TrainDataProcessor._task_data_validator_for(agent, self._dataset(), [agent]) is None

    def test_agent_with_dangling_rs_reference_is_skipped(self, tmp_path, monkeypatch):
        agent_dir = tmp_path / "responses_api_agents" / "tau2"
        agent_dir.mkdir(parents=True)
        (agent_dir / "task_data.py").write_text(
            "from pydantic import BaseModel\nclass TaskData(BaseModel):\n    pass\n"
        )
        monkeypatch.chdir(tmp_path)
        agent = self._agent("tau2", {"type": "resources_servers", "name": "missing_rs"})
        assert TrainDataProcessor._task_data_validator_for(agent, self._dataset(), [agent]) is None

    def test_agent_with_valid_rs_reference_uses_that_servers_schema(self, tmp_path, monkeypatch):
        # The benchmark shape: the dataset is declared on the agent, the agent references a
        # resources-server instance by name, and validation must use that server's schema.
        rs_dir = tmp_path / "resources_servers" / "math_with_judge"
        rs_dir.mkdir(parents=True)
        (rs_dir / "task_data.py").write_text(
            "from pydantic import BaseModel, ConfigDict\n"
            "class TaskData(BaseModel):\n"
            "    model_config = ConfigDict(extra='allow')\n"
            "    expected_answer: str\n"
        )
        monkeypatch.chdir(tmp_path)
        rs = ResourcesServerInstanceConfig(
            name="bench_rs",
            server_type_config_dict=OmegaConf.create({}),
            resources_servers={"math_with_judge": {"entrypoint": "app.py", "domain": "math"}},
        )
        agent = self._agent("simple_agent", {"type": "resources_servers", "name": "bench_rs"})
        validator = TrainDataProcessor._task_data_validator_for(agent, self._dataset(), [agent, rs])
        assert validator is not None
        assert validator.report.server_name == "math_with_judge"


class TestMisplacedLegacyFields:
    def _adapter(self, tmp_path):
        (tmp_path / "task_data.py").write_text(
            "from pydantic import BaseModel, ConfigDict, Field\n"
            "class TaskData(BaseModel):\n"
            "    model_config = ConfigDict(extra='allow')\n"
            "    expected_city: str = Field(\n"
            "        default='Paris', json_schema_extra={'legacy_location': 'verifier_metadata'}\n"
            "    )\n"
        )
        return load_task_data_schema(tmp_path)

    def test_marked_field_found_only_top_level_is_flagged(self, tmp_path):
        v = TaskDataValidator(server_name="s", adapter=self._adapter(tmp_path), dataset_fpath="d")
        v.validate_row(0, {"expected_city": "Tokyo"})
        assert not v.report.clean
        assert v.report.misplaced_keys == {"expected_city": 1}
        assert "the server will not see them" in v.report.summary()

    def test_marked_field_in_verifier_metadata_is_correct(self, tmp_path):
        v = TaskDataValidator(server_name="s", adapter=self._adapter(tmp_path), dataset_fpath="d")
        v.validate_row(0, {"verifier_metadata": {"expected_city": "Tokyo"}})
        assert v.report.clean

    def test_duplicated_in_both_places_is_not_misplaced(self, tmp_path):
        v = TaskDataValidator(server_name="s", adapter=self._adapter(tmp_path), dataset_fpath="d")
        v.validate_row(0, {"expected_city": "Tokyo", "verifier_metadata": {"expected_city": "Tokyo"}})
        assert v.report.clean

    def test_migrated_row_format_is_exempt(self, tmp_path):
        v = TaskDataValidator(server_name="s", adapter=self._adapter(tmp_path), dataset_fpath="d")
        v.validate_row(0, {"expected_city": "Tokyo", "task_data": {}})
        assert v.report.misplaced_keys == {}

    def test_unmarked_top_level_fields_are_fine(self, tmp_path):
        (tmp_path / "task_data.py").write_text(
            "from pydantic import BaseModel, ConfigDict\n"
            "class TaskData(BaseModel):\n"
            "    model_config = ConfigDict(extra='allow')\n"
            "    question: str\n"
        )
        v = TaskDataValidator(server_name="s", adapter=load_task_data_schema(tmp_path), dataset_fpath="d")
        v.validate_row(0, {"question": "q"})
        assert v.report.clean


class TestAutoValidationMode:
    def _config(self, mode, task_data_validation="auto"):
        from nemo_gym.data.dataset_preparation import TrainDataProcessorConfig

        return TrainDataProcessorConfig(output_dirpath="out", mode=mode, task_data_validation=task_data_validation)

    def test_auto_is_error_for_example_validation(self):
        assert self._config("example_validation").effective_task_data_validation == "error"

    def test_auto_is_warn_for_train_preparation(self):
        assert self._config("train_preparation").effective_task_data_validation == "warn"

    def test_explicit_setting_wins(self):
        assert self._config("example_validation", "off").effective_task_data_validation == "off"
        assert self._config("train_preparation", "error").effective_task_data_validation == "error"


class TestSchemaPresence:
    def test_every_resources_server_ships_a_schema(self):
        missing = sorted(
            d.name
            for d in (REPO_ROOT / "resources_servers").iterdir()
            if d.is_dir() and (d / "app.py").exists() and not (d / "task_data.py").exists()
        )
        assert not missing, (
            f"resources servers without a task_data.py schema: {missing}. Every server must "
            "describe its dataset rows (see nemo_gym/data/task_data.py for the protocol)."
        )


class TestRepoDataMatchesSchemas:
    """The drift gate: every committed dataset row must validate against its server's schema.

    Fails the PR that changes a schema without fixing the data, or commits data that no longer
    matches the schema. Uses the same mapping collate uses: dataset declarations in tracked
    configs (following config_paths chains and one _inherit_from hop) resolve to the declaring
    resources server.
    """

    @staticmethod
    def _merged_config(path, seen=None):
        import yaml

        seen = seen or set()
        if path in seen:
            return {}
        seen.add(path)
        try:
            cfg = yaml.safe_load(open(path)) or {}
        except Exception:
            return {}
        out = {}
        for p in cfg.get("config_paths") or []:
            out.update(TestRepoDataMatchesSchemas._merged_config(p, seen))
        out.update({k: v for k, v in cfg.items() if k != "config_paths"})
        return out

    def test_all_committed_rows_validate(self, monkeypatch):
        import subprocess

        import yaml

        monkeypatch.chdir(REPO_ROOT)
        configs = [
            c
            for c in subprocess.run(["git", "ls-files", "*.yaml"], capture_output=True, text=True).stdout.split()
            if c.split("/")[0] in ("resources_servers", "responses_api_agents", "benchmarks", "environments")
        ]
        tracked = set(subprocess.run(["git", "ls-files", "*.jsonl"], capture_output=True, text=True).stdout.split())

        server_files = {}
        for cf in configs:
            try:
                local = yaml.safe_load(open(cf)) or {}
            except Exception:
                continue
            if not isinstance(local, dict):
                continue
            full = self._merged_config(cf)
            for _inst, v in local.items():
                if not isinstance(v, dict):
                    continue
                for section in ("resources_servers", "responses_api_agents"):
                    sec = v.get(section)
                    if not isinstance(sec, dict):
                        continue
                    for impl_key, body in sec.items():
                        if not isinstance(body, dict):
                            continue
                        for d in body.get("datasets") or []:
                            if not isinstance(d, dict) or not d.get("jsonl_fpath"):
                                continue
                            if section == "resources_servers":
                                owner = ("resources_servers", impl_key)
                            elif "resources_server" not in body:
                                # Self-contained agent: its schema lives next to the agent
                                # implementation, mirroring collate's fallback.
                                owner = ("responses_api_agents", impl_key)
                            else:
                                ref = body.get("resources_server") or {}
                                name = ref.get("name") if isinstance(ref, dict) else None
                                tgt = full.get(name) if name else None
                                if (
                                    isinstance(tgt, dict)
                                    and "resources_servers" not in tgt
                                    and tgt.get("_inherit_from")
                                ):
                                    tgt = full.get(tgt["_inherit_from"]) or tgt
                                rs = (
                                    next(iter(tgt["resources_servers"]))
                                    if isinstance(tgt, dict) and isinstance(tgt.get("resources_servers"), dict)
                                    else None
                                )
                                owner = ("resources_servers", rs) if rs else None
                            if owner and str(d["jsonl_fpath"]) in tracked:
                                server_files.setdefault(owner, set()).add(str(d["jsonl_fpath"]))

        assert len(server_files) > 50, "mapping looks broken: too few servers with committed data"
        agent_owned = [o for o in server_files if o[0] == "responses_api_agents"]
        assert len(agent_owned) >= 5, "mapping looks broken: self-contained agent datasets not found"
        failures = []
        rows_checked = 0
        for (base_folder, server), files in sorted(server_files.items()):
            server_dir = find_server_dir(server, base_folder=base_folder)
            adapter = load_task_data_schema(server_dir) if server_dir else None
            if adapter is None:
                failures.append(f"{server}: no loadable schema")
                continue
            for f in sorted(files):
                report = validate_jsonl_rows(server, adapter, f, open(f).read().splitlines())
                rows_checked += report.rows
                if not report.clean:
                    failures.append(report.summary())
        assert rows_checked > 500, "sweep looks broken: too few rows checked"
        assert not failures, "\n".join(failures)


class TestHarborLayoutThroughTheProcessor:
    """The lookup must work on the processor's real path, where the in-scope list has no environment server."""

    def _processor_and_configs(self, tmp_path, monkeypatch, rows):
        from omegaconf import DictConfig

        from nemo_gym.config import GlobalConfigDictParser
        from nemo_gym.data.dataset_preparation import TrainDataProcessorConfig

        env_dir = tmp_path / "environment_servers" / "harbor"
        env_dir.mkdir(parents=True)
        (env_dir / "task_data.py").write_text((REPO_ROOT / "environment_servers/harbor/task_data.py").read_text())
        # The agent folder exists but ships no schema: the pre-fix fallback found nothing and validated nothing.
        (tmp_path / "responses_api_agents" / "harbor_oracle_agent").mkdir(parents=True)
        rows_path = tmp_path / "rows.jsonl"
        rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        monkeypatch.chdir(tmp_path)
        global_config_dict = GlobalConfigDictParser().parse_no_environment(
            DictConfig(
                {
                    "harbor_ds_agent": {
                        "responses_api_agents": {
                            "harbor_oracle_agent": {
                                "entrypoint": "app.py",
                                "datasets": [
                                    {
                                        "name": "ds",
                                        "type": "validation",
                                        "license": "TBD",
                                        "jsonl_fpath": str(rows_path),
                                    }
                                ],
                            }
                        }
                    },
                    "harbor_ds_environment": {
                        "environment_servers": {
                            "harbor": {
                                "entrypoint": "app.py",
                                "agent_server": {"type": "responses_api_agents", "name": "harbor_ds_agent"},
                            }
                        }
                    },
                }
            )
        )
        processor = TrainDataProcessor()
        config = TrainDataProcessorConfig(
            output_dirpath=str(tmp_path / "out"), mode="train_preparation", task_data_validation="error"
        )
        in_scope = processor.load_and_validate_server_instance_configs(config, global_config_dict)
        # Only the dataset-bearing agent is in scope; the environment server that owns the schema is not.
        assert [c.name for c in in_scope] == ["harbor_ds_agent"]
        return processor, config, in_scope

    @staticmethod
    def _row(task_data):
        return {
            "task_id": {"taskset": "ds", "task_id": "a"},
            "task_input": {
                "responses_create_params": {"input": [{"role": "user", "content": "do it"}]},
                "task_data": task_data,
            },
        }

    def test_row_without_the_digest_is_rejected(self, tmp_path, monkeypatch):
        processor, config, in_scope = self._processor_and_configs(tmp_path, monkeypatch, [self._row({})])
        with pytest.raises(ValueError, match=r"(?s)against the `harbor` schema.*_ng_digest"):
            processor._collate_samples_single_type(
                "validation", in_scope, task_data_validation=config.effective_task_data_validation
            )

    def test_materialized_row_passes(self, tmp_path, monkeypatch):
        processor, config, in_scope = self._processor_and_configs(
            tmp_path, monkeypatch, [self._row({"_ng_digest": "a" * 64})]
        )
        paths = processor._collate_samples_single_type(
            "validation", in_scope, task_data_validation=config.effective_task_data_validation
        )
        assert len(paths) == 1 and json.loads(paths[0].read_text())["task_input"]["task_data"] == {
            "_ng_digest": "a" * 64
        }


class TestLegacyWrapperLayoutThroughTheProcessor:
    """A self-contained agent wrapped by a schema-less environment server validates against its own schema.

    The shipped tau2 layout: ``responses_api_agents/tau2`` ships ``task_data.py`` and an
    ``environment_servers/legacy_agent`` block names it as ``agent_server`` but has no schema. The
    lookup must not stop at the schema-less wrapper (which silently disabled validation).
    """

    def _processor_and_configs(self, tmp_path, monkeypatch, rows):
        from omegaconf import DictConfig

        from nemo_gym.config import GlobalConfigDictParser
        from nemo_gym.data.dataset_preparation import TrainDataProcessorConfig

        agent_dir = tmp_path / "responses_api_agents" / "tau2"
        agent_dir.mkdir(parents=True)
        (agent_dir / "task_data.py").write_text((REPO_ROOT / "responses_api_agents/tau2/task_data.py").read_text())
        # The wrapper exists but ships no schema.
        (tmp_path / "environment_servers" / "legacy_agent").mkdir(parents=True)
        rows_path = tmp_path / "rows.jsonl"
        rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        monkeypatch.chdir(tmp_path)
        global_config_dict = GlobalConfigDictParser().parse_no_environment(
            DictConfig(
                {
                    "tau2_agent": {
                        "responses_api_agents": {
                            "tau2": {
                                "entrypoint": "app.py",
                                "datasets": [
                                    {
                                        "name": "ds",
                                        "type": "validation",
                                        "license": "TBD",
                                        "jsonl_fpath": str(rows_path),
                                    }
                                ],
                            }
                        }
                    },
                    "tau2_environment_server": {
                        "environment_servers": {
                            "legacy_agent": {
                                "entrypoint": "app.py",
                                "agent_server": {"type": "responses_api_agents", "name": "tau2_agent"},
                            }
                        }
                    },
                }
            )
        )
        processor = TrainDataProcessor()
        config = TrainDataProcessorConfig(
            output_dirpath=str(tmp_path / "out"), mode="train_preparation", task_data_validation="error"
        )
        in_scope = processor.load_and_validate_server_instance_configs(config, global_config_dict)
        assert [c.name for c in in_scope] == ["tau2_agent"]
        # The wrapper names the agent, but without a schema the agent's own folder is the schema home.
        lookup = processor._schema_lookup_configs(in_scope)
        assert TrainDataProcessor._owning_environment_server_impl(in_scope[0], lookup) == "legacy_agent"
        assert TrainDataProcessor._schema_home_for(in_scope[0], lookup) == ("tau2", "responses_api_agents")
        return processor, config, in_scope

    @staticmethod
    def _row():
        # The committed tau2 example row, flat legacy format: the agent's schema requires ``seed``.
        row = json.loads((REPO_ROOT / "responses_api_agents/tau2/data/example.jsonl").read_text().splitlines()[0])
        assert "seed" in row
        return row

    def test_row_missing_a_required_agent_field_is_rejected(self, tmp_path, monkeypatch):
        row = self._row()
        del row["seed"]
        processor, config, in_scope = self._processor_and_configs(tmp_path, monkeypatch, [row])
        with pytest.raises(ValueError, match=r"(?s)against the `tau2` schema.*seed"):
            processor._collate_samples_single_type(
                "validation", in_scope, task_data_validation=config.effective_task_data_validation
            )

    def test_valid_row_passes(self, tmp_path, monkeypatch):
        processor, config, in_scope = self._processor_and_configs(tmp_path, monkeypatch, [self._row()])
        paths = processor._collate_samples_single_type(
            "validation", in_scope, task_data_validation=config.effective_task_data_validation
        )
        assert len(paths) == 1 and len(paths[0].read_text().splitlines()) == 1
