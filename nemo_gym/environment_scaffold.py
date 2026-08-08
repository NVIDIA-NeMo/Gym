# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create minimal environment and benchmark skeletons."""

from __future__ import annotations

import ast
import json
import keyword
import re
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any, Mapping

import yaml

from nemo_gym import component_search_roots
from nemo_gym.config_types import ConfigError
from nemo_gym.environment_manifest import (
    EnvironmentKind,
    EnvironmentManifest,
    IntegrationProfile,
    dump_manifest,
)


ENVIRONMENT_KINDS = tuple(kind.value for kind in EnvironmentKind)
INTEGRATION_PROFILES = tuple(profile.value for profile in IntegrationProfile)
SCAFFOLD_PLACEHOLDER = "GYM_ONBOARDING_TODO"

_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_REUSE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class ScaffoldError(ConfigError):
    """A scaffold request is invalid."""


class ScaffoldConflictError(ScaffoldError):
    """Generated files would replace different content."""

    def __init__(self, paths: tuple[Path, ...]):
        self.paths = paths
        rendered = "\n".join(f"  - {path}" for path in paths)
        super().__init__(f"Scaffolding would overwrite existing content:\n{rendered}")


@dataclass(frozen=True)
class ScaffoldResult:
    """Files handled by a scaffold operation."""

    asset_dir: Path
    created: tuple[Path, ...]
    existing: tuple[Path, ...]


@dataclass(frozen=True)
class _ReusedVerifier:
    selector: str
    config_reference: str
    resource_instance: str
    agent_instance: str | None


def scaffold_environment(
    *,
    kind: EnvironmentKind | str,
    name: str,
    profile: IntegrationProfile | str = IntegrationProfile.STOCK_LOOP,
    reuse_verifier: str | None = None,
    root: str | Path | None = None,
) -> ScaffoldResult:
    """Create a profile-aware skeleton without replacing existing files.

    ``profile`` only selects generated integration points. Runtime dispatch remains
    driven by the resulting Gym config.
    """

    environment_kind = _parse_kind(kind)
    integration_profile = _parse_profile(profile)
    _validate_name(name, integration_profile)
    if reuse_verifier is not None and not _REUSE_PATTERN.fullmatch(reuse_verifier):
        raise ScaffoldError("reuse_verifier must be a resources-server name")
    if reuse_verifier is not None and integration_profile != IntegrationProfile.STOCK_LOOP:
        raise ScaffoldError("reuse_verifier currently supports only the stock-loop profile")

    requested_root = Path.cwd() if root is None else Path(root).expanduser()
    if requested_root.is_symlink():
        raise ScaffoldError(f"scaffold root must not be a symlink: {requested_root}")
    if requested_root.exists() and not requested_root.is_dir():
        raise ScaffoldError(f"scaffold root must be a directory: {requested_root}")
    scaffold_root = requested_root.resolve()
    description = f"{SCAFFOLD_PLACEHOLDER}: Describe the {name} {environment_kind.value}."

    module_name = _python_name(name)
    parent_name = "benchmarks" if environment_kind == EnvironmentKind.BENCHMARK else "environments"
    asset_dir = scaffold_root / parent_name / name
    reused = _resolve_reused_verifier(scaffold_root, reuse_verifier) if reuse_verifier else None

    resource_type = reused.selector if reused else module_name
    resource_instance = reused.resource_instance if reused else f"{module_name}_resources_server"
    generated_agent = integration_profile in {
        IntegrationProfile.MEASURED_LOOP,
        IntegrationProfile.EXTERNAL_LOOP,
    }
    agent_type = f"{module_name}_agent" if generated_agent else "simple_agent"
    agent_instance = f"{module_name}_agent"
    if reused is not None and agent_instance in {reused.resource_instance, reused.agent_instance}:
        agent_instance = f"{module_name}_catalog_agent"
        if agent_instance in {reused.resource_instance, reused.agent_instance}:
            raise ScaffoldError(f"name {name!r} collides with instances in reused verifier {reuse_verifier!r}")
    dataset_path = f"{parent_name}/{name}/data/example.jsonl"
    prompt_path = f"{parent_name}/{name}/prompt.yaml"
    prepare_path = f"{parent_name}/{name}/prepare.py"
    driver = f"{parent_name}.{name}.rollout_driver:run_rollout_collection"

    manifest = _manifest(
        kind=environment_kind,
        name=name,
        profile=integration_profile,
        description=description,
        resource_type=resource_type,
        agent_type=agent_type,
        dataset_path=dataset_path,
        prompt_path=prompt_path,
        prepare_path=prepare_path,
        driver=driver,
    )
    files: dict[Path, str] = {
        asset_dir / "__init__.py": _license_header(),
        asset_dir / "manifest.yaml": dump_manifest(manifest),
        asset_dir / "config.yaml": _asset_config(
            kind=environment_kind,
            name=name,
            profile=integration_profile,
            module_name=module_name,
            resource_instance=resource_instance,
            agent_instance=agent_instance,
            agent_type=agent_type,
            dataset_path=dataset_path,
            prompt_path=prompt_path,
            prepare_path=prepare_path,
            reused=reused,
            driver=driver,
        ),
        asset_dir / "README.md": _asset_readme(environment_kind, name, integration_profile, reused),
    }

    if environment_kind == EnvironmentKind.BENCHMARK:
        source = {
            "question": "What is 6 x 7?",
            "expected_answer": "42",
            "_onboarding": f"{SCAFFOLD_PLACEHOLDER}: replace sample data",
        }
        rendered_source = json.dumps(source) + "\n"
        files.update(
            {
                asset_dir / "data" / "source.jsonl": rendered_source,
                asset_dir / "data" / "example.jsonl": rendered_source,
                asset_dir / "prompt.yaml": _benchmark_prompt(),
                asset_dir / "prepare.py": _benchmark_prepare(name),
            }
        )
    else:
        files[asset_dir / "data" / "example.jsonl"] = _environment_example(agent_instance)

    if reused is None:
        resource_dir = scaffold_root / "resources_servers" / module_name
        files.update(
            _resources_server_files(
                scaffold_root,
                resource_dir,
                module_name,
                "other",
                description,
            )
        )

    if generated_agent:
        agent_dir = scaffold_root / "responses_api_agents" / agent_type
        files.update(_agent_files(scaffold_root, agent_dir, module_name, integration_profile))

    if integration_profile == IntegrationProfile.CUSTOM_DRIVER:
        files[asset_dir / "rollout_driver.py"] = _rollout_driver()

    return _write_files(scaffold_root, asset_dir, files)


def _parse_kind(kind: EnvironmentKind | str) -> EnvironmentKind:
    try:
        return EnvironmentKind(kind)
    except ValueError as error:
        raise ScaffoldError(f"kind must be one of {ENVIRONMENT_KINDS}, got {kind!r}") from error


def _parse_profile(profile: IntegrationProfile | str) -> IntegrationProfile:
    try:
        return IntegrationProfile(profile)
    except ValueError as error:
        raise ScaffoldError(f"profile must be one of {INTEGRATION_PROFILES}, got {profile!r}") from error


def _validate_name(name: str, profile: IntegrationProfile) -> None:
    if not _NAME_PATTERN.fullmatch(name) or keyword.iskeyword(name):
        raise ScaffoldError("name must contain only lowercase letters, digits, '_' or '-', and may not be a keyword")
    if profile == IntegrationProfile.CUSTOM_DRIVER and not name.isidentifier():
        raise ScaffoldError("custom-driver names must be valid Python module names")


def _python_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]", "_", name)
    return f"env_{normalized}" if normalized[0].isdigit() else normalized


def _class_name(module_name: str) -> str:
    return "".join(part.capitalize() for part in module_name.split("_") if part)


def _manifest(
    *,
    kind: EnvironmentKind,
    name: str,
    profile: IntegrationProfile,
    description: str,
    resource_type: str,
    agent_type: str,
    dataset_path: str,
    prompt_path: str,
    prepare_path: str,
    driver: str,
) -> EnvironmentManifest:
    dataset: dict[str, Any] = {
        "name": name,
        "type": "benchmark" if kind == EnvironmentKind.BENCHMARK else "example",
        "jsonl_fpath": dataset_path,
        "num_repeats": 1,
    }
    if kind == EnvironmentKind.BENCHMARK:
        dataset.update(prepare_script=prepare_path, prompt_config=prompt_path)
    data: dict[str, Any] = {
        "name": name,
        "version": "0.1.0",
        "kind": kind,
        "integration_profile": profile,
        "domain": "other",
        "description": description,
        "modality": "text",
        "licensing": "unknown",
        "authors": [SCAFFOLD_PLACEHOLDER],
        "reward": {
            "range": (0.0, 1.0),
            "higher_is_better": True,
        },
        "determinism": "unknown",
        "resources_server": resource_type,
        "agent_server": agent_type,
        "model_server": "policy_model",
        "datasets": [dataset],
    }
    if kind == EnvironmentKind.BENCHMARK:
        data.update(
            canonical_split=SCAFFOLD_PLACEHOLDER,
            standard_prompt_config=prompt_path,
        )
    if profile == IntegrationProfile.CUSTOM_DRIVER:
        data["rollout_driver"] = driver
    return EnvironmentManifest.model_validate(data)


def _server_entries(raw: dict[str, Any], server_type: str) -> list[tuple[str, str, dict[str, Any]]]:
    entries: list[tuple[str, str, dict[str, Any]]] = []
    for instance_name, value in raw.items():
        implementations = value.get(server_type) if isinstance(value, dict) else None
        if not isinstance(implementations, dict):
            continue
        for implementation, config in implementations.items():
            if not isinstance(config, dict):
                raise ScaffoldError(f"{server_type} config for {implementation!r} is not a mapping")
            entries.append((str(instance_name), str(implementation), config))
    return entries


def _resolve_reused_verifier(root: Path, selector: str) -> _ReusedVerifier:
    config_reference = f"resources_servers/{selector}/configs/{selector}.yaml"
    candidates = list(
        dict.fromkeys((search_root / config_reference).resolve() for search_root in (root, *component_search_roots()))
    )
    config_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        searched = "\n".join(f"  - {candidate}" for candidate in candidates)
        raise ScaffoldError(f"reuse_verifier {selector!r} was not found. Looked in:\n{searched}") from error
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ScaffoldError(f"could not read reused verifier config {config_path}: {error}") from error
    if not isinstance(raw, dict):
        raise ScaffoldError(f"reused verifier config {config_path} is not a mapping")

    resources = _server_entries(raw, "resources_servers")
    if len(resources) != 1 or resources[0][1] != selector:
        raise ScaffoldError(
            f"reused verifier {selector!r} config must define exactly one resources-server instance of that type"
        )
    resource_instance, _resource_type, selected_config = resources[0]
    if _server_entries(raw, "responses_api_models"):
        raise ScaffoldError(f"reused verifier {selector!r} config must not bundle model servers")

    entrypoint = selected_config.get("entrypoint")
    if not isinstance(entrypoint, str):
        raise ScaffoldError(f"reused verifier {selector!r} does not define a resources-server entrypoint")
    raw_app_path = Path(entrypoint)
    app_path = raw_app_path if raw_app_path.is_absolute() else config_path.parent.parent / raw_app_path
    _require_verifier_fixture(app_path.resolve(), selector)

    agents = _server_entries(raw, "responses_api_agents")
    if len(agents) > 1 or (agents and agents[0][1] != "simple_agent"):
        raise ScaffoldError(f"reused verifier {selector!r} config may bundle only one simple_agent")
    if agents:
        agent_instance, _agent_type, agent_config = agents[0]
        reference = agent_config.get("resources_server")
        if not isinstance(reference, dict) or reference.get("name") != resource_instance:
            raise ScaffoldError(
                f"reused verifier {selector!r} simple_agent must reference resources instance {resource_instance!r}"
            )
    else:
        agent_instance = None
    return _ReusedVerifier(
        selector=selector,
        config_reference=config_reference,
        resource_instance=resource_instance,
        agent_instance=agent_instance,
    )


def _require_verifier_fixture(app_path: Path, selector: str) -> None:
    try:
        tree = ast.parse(app_path.read_text(encoding="utf-8"), filename=str(app_path))
    except FileNotFoundError as error:
        raise ScaffoldError(f"reused verifier {selector!r} entrypoint was not found: {app_path}") from error
    except (OSError, UnicodeError, SyntaxError) as error:
        raise ScaffoldError(f"could not inspect reused verifier entrypoint {app_path}: {error}") from error

    exports_fixture = False
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            exports_fixture = any(
                isinstance(target, ast.Name) and target.id == "VERIFIER_FIXTURE" for target in targets
            )
        elif isinstance(node, ast.ImportFrom):
            exports_fixture = any((alias.asname or alias.name) == "VERIFIER_FIXTURE" for alias in node.names)
        if exports_fixture:
            return
    raise ScaffoldError(
        f"reused verifier {selector!r} must export VERIFIER_FIXTURE from {app_path} before it can be onboarded"
    )


def _asset_config(
    *,
    kind: EnvironmentKind,
    name: str,
    profile: IntegrationProfile,
    module_name: str,
    resource_instance: str,
    agent_instance: str,
    agent_type: str,
    dataset_path: str,
    prompt_path: str,
    prepare_path: str,
    reused: _ReusedVerifier | None,
    driver: str,
) -> str:
    config_reference = (
        reused.config_reference if reused else f"resources_servers/{module_name}/configs/{module_name}.yaml"
    )
    config: dict[str, Any] = {"config_paths": [config_reference]}

    dataset: dict[str, Any] = {
        "name": name,
        "type": "benchmark" if kind == EnvironmentKind.BENCHMARK else "example",
        "jsonl_fpath": dataset_path,
        "num_repeats": 1,
    }
    if kind == EnvironmentKind.BENCHMARK:
        dataset.update(prompt_config=prompt_path, prepare_script=prepare_path)

    agent_config: dict[str, Any] = {
        "resources_server": {"type": "resources_servers", "name": resource_instance},
        "model_server": {"type": "responses_api_models", "name": "policy_model"},
        "datasets": [dataset],
    }
    if reused is not None and reused.agent_instance is not None:
        agent_entry: dict[str, Any] = {
            "_inherit_from": reused.agent_instance,
            "responses_api_agents": {"simple_agent": agent_config},
        }
    else:
        agent_config = {"entrypoint": "app.py", **agent_config}
        agent_entry = {"responses_api_agents": {agent_type: agent_config}}
    config[agent_instance] = agent_entry
    if profile == IntegrationProfile.CUSTOM_DRIVER:
        config["rollout_collection_driver"] = driver
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def _asset_readme(
    kind: EnvironmentKind,
    name: str,
    profile: IntegrationProfile,
    reused: _ReusedVerifier | None,
) -> str:
    scorer = f"`{reused.selector}`" if reused else f"`resources_servers/{_python_name(name)}`"
    return dedent(
        f"""\
        # {name}

        {SCAFFOLD_PLACEHOLDER}: Describe this {kind.value} and its intended use.

        - Integration profile: `{profile.value}`
        - Scorer: {scorer}

        Replace the sample data and complete all scaffold placeholders before running `gym env publish`.
        """
    )


def _environment_example(agent_instance: str) -> str:
    row = {
        "responses_create_params": {
            "input": [{"role": "user", "content": "What is 6 x 7? Reply with only the answer."}]
        },
        "expected_answer": "42",
        "agent_ref": {"type": "responses_api_agents", "name": agent_instance},
        "_onboarding": f"{SCAFFOLD_PLACEHOLDER}: replace sample data",
    }
    return json.dumps(row) + "\n"


def _benchmark_prompt() -> str:
    return dedent(
        """\
        user: |-
          Answer the question. Return only the final answer.

          {question}
        """
    )


def _benchmark_prepare(name: str) -> str:
    return _license_header() + dedent(
        f'''\
        """Prepare source rows for the {name} benchmark."""

        from __future__ import annotations

        import json
        from pathlib import Path
        from typing import TypedDict


        class SourceRow(TypedDict):
            question: str
            expected_answer: str


        BENCHMARK_DIR = Path(__file__).parent
        SOURCE_PATH = BENCHMARK_DIR / "data" / "source.jsonl"
        OUTPUT_PATH = BENCHMARK_DIR / "data" / "example.jsonl"


        def prepare(source: Path = SOURCE_PATH, output: Path = OUTPUT_PATH) -> Path:
            output.parent.mkdir(parents=True, exist_ok=True)
            with (
                source.open(encoding="utf-8") as source_stream,
                output.open("w", encoding="utf-8") as output_stream,
            ):
                for line_number, line in enumerate(source_stream, start=1):
                    raw = json.loads(line)
                    if (
                        not isinstance(raw, dict)
                        or not isinstance(raw.get("question"), str)
                        or not isinstance(raw.get("expected_answer"), str)
                    ):
                        raise ValueError(f"invalid source row {{line_number}}")
                    row: SourceRow = {{
                        "question": raw["question"],
                        "expected_answer": raw["expected_answer"],
                    }}
                    output_stream.write(json.dumps(row) + "\\n")
            return output


        if __name__ == "__main__":
            prepare()
        '''
    )


def _resources_server_files(
    root: Path,
    resource_dir: Path,
    module_name: str,
    domain: str,
    description: str,
) -> dict[Path, str]:
    return {
        resource_dir / "__init__.py": _license_header(),
        resource_dir / "README.md": _resources_server_readme(module_name),
        resource_dir / "app.py": _resources_server_app(module_name),
        resource_dir / "configs" / f"{module_name}.yaml": _resources_server_config(module_name, domain, description),
        resource_dir / "requirements.txt": _requirements(root),
        resource_dir / "tests" / "__init__.py": _license_header(),
        resource_dir / "tests" / "test_app.py": _resources_server_test(module_name),
        resource_dir / "tests" / "verifier_cases.jsonl": _verifier_cases(),
    }


def _resources_server_readme(module_name: str) -> str:
    return dedent(
        f"""\
        # {module_name} resources server

        `app.py` contains the verifier and its `VERIFIER_FIXTURE`. Replace all
        scaffold placeholders, then run:

        ```bash
        gym env test --resources-server {module_name}
        ```
        """
    )


def _resources_server_config(module_name: str, domain: str, description: str) -> str:
    config = {
        f"{module_name}_resources_server": {
            "resources_servers": {
                module_name: {
                    "entrypoint": "app.py",
                    "domain": domain,
                    "verified": False,
                    "description": description,
                }
            }
        }
    }
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def _resources_server_app(module_name: str) -> str:
    class_name = _class_name(module_name)
    return _license_header() + dedent(
        f'''\
        from pathlib import Path
        from typing import ClassVar

        from pydantic import ConfigDict

        from nemo_gym.base_resources_server import (
            BaseResourcesServerConfig,
            BaseVerifyRequest,
            BaseVerifyResponse,
            ReverifyMode,
            SimpleResourcesServer,
        )
        from nemo_gym.verifier_fixture import VerifierFixture


        class {class_name}ResourcesServerConfig(BaseResourcesServerConfig):
            REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


        class {class_name}VerifyRequest(BaseVerifyRequest):
            model_config = ConfigDict(extra="allow")

            expected_answer: str


        class {class_name}VerifyResponse(BaseVerifyResponse):
            model_config = ConfigDict(extra="allow")


        def _assistant_text(body: {class_name}VerifyRequest) -> str:
            return "".join(
                content.text
                for output in body.response.output
                if output.type == "message" and output.role == "assistant"
                for content in output.content
                if content.type == "output_text"
            ).strip()


        class {class_name}Verifier:
            """{SCAFFOLD_PLACEHOLDER}: Replace this sample exact-match scorer."""

            async def verify(self, body: {class_name}VerifyRequest) -> {class_name}VerifyResponse:
                reward = 1.0 if _assistant_text(body) == body.expected_answer else 0.0
                return {class_name}VerifyResponse(**body.model_dump(), reward=reward)


        class {class_name}ResourcesServer({class_name}Verifier, SimpleResourcesServer):
            config: {class_name}ResourcesServerConfig


        VERIFIER_FIXTURE = VerifierFixture(
            server_factory={class_name}Verifier,
            request_model={class_name}VerifyRequest,
            cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
        )


        if __name__ == "__main__":
            {class_name}ResourcesServer.run_webserver()
        '''
    )


def _resources_server_test(module_name: str) -> str:
    return _license_header() + dedent(
        f"""\
        import asyncio

        from resources_servers.{module_name}.app import VERIFIER_FIXTURE

        from nemo_gym.verifier_fixture import exercise_verifier_fixture


        def test_verifier_fixture() -> None:
            asyncio.run(
                exercise_verifier_fixture(
                    VERIFIER_FIXTURE,
                    reward_range=(0.0, 1.0),
                    determinism="unknown",
                )
            )
        """
    )


def _verifier_cases() -> str:
    def response(text: str) -> dict[str, Any]:
        return {
            "id": "response_fixture",
            "created_at": 0,
            "model": "fixture",
            "object": "response",
            "output": [
                {
                    "id": "message_fixture",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
        }

    def request(text: str) -> dict[str, Any]:
        return {
            "responses_create_params": {"input": [{"role": "user", "content": "What is 6 x 7?"}]},
            "response": response(text),
            "expected_answer": "42",
            "_onboarding": f"{SCAFFOLD_PLACEHOLDER}: replace scorer fixture",
        }

    cases = [
        {"name": "correct", "kind": "full_reward", "request": request("42"), "expected_reward": 1.0},
        {"name": "incorrect", "kind": "zero_reward", "request": request("41"), "expected_reward": 0.0},
        {
            "name": "missing response",
            "kind": "malformed",
            "request": {
                "responses_create_params": {"input": "missing response"},
                "expected_answer": "42",
                "_onboarding": f"{SCAFFOLD_PLACEHOLDER}: replace scorer fixture",
            },
            "expected_error": "response",
            "expected_error_type": "ValidationError",
        },
    ]
    return "".join(json.dumps(case) + "\n" for case in cases)


def _agent_files(root: Path, agent_dir: Path, module_name: str, profile: IntegrationProfile) -> dict[Path, str]:
    class_name = _class_name(module_name)
    if profile == IntegrationProfile.MEASURED_LOOP:
        body = f"""\
        from fastapi import Request, Response

        from nemo_gym.base_responses_api_agent import Body
        from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
        from responses_api_agents.simple_agent.app import SimpleAgent


        class {class_name}Agent(SimpleAgent):
            async def responses(
                self,
                request: Request,
                response: Response,
                body: NeMoGymResponseCreateParamsNonStreaming = Body(),
            ) -> NeMoGymResponse:
                # {SCAFFOLD_PLACEHOLDER}: Implement the measured harness strategy.
                return await super().responses(request, response, body)


        if __name__ == "__main__":
            {class_name}Agent.run_webserver()
        """
    else:
        body = f"""\
        from fastapi import Request

        from responses_api_agents.simple_agent.app import (
            SimpleAgent,
            SimpleAgentRunRequest,
            SimpleAgentVerifyResponse,
        )


        class {class_name}Agent(SimpleAgent):
            async def run(self, request: Request, body: SimpleAgentRunRequest) -> SimpleAgentVerifyResponse:
                # {SCAFFOLD_PLACEHOLDER}: Delegate the episode to the external framework.
                return await super().run(request, body)


        if __name__ == "__main__":
            {class_name}Agent.run_webserver()
        """
    app = _license_header() + dedent(body)
    return {
        agent_dir / "__init__.py": _license_header(),
        agent_dir / "README.md": _agent_readme(module_name, profile),
        agent_dir / "app.py": app,
        agent_dir / "requirements.txt": _requirements(root),
        agent_dir / "tests" / "__init__.py": _license_header(),
        agent_dir / "tests" / "test_app.py": _agent_test(module_name),
    }


def _agent_readme(module_name: str, profile: IntegrationProfile) -> str:
    return dedent(
        f"""\
        # {module_name} agent

        This is the `{profile.value}` integration point for the generated workload.
        Replace all scaffold placeholders before running `gym env publish`.
        """
    )


def _agent_test(module_name: str) -> str:
    class_name = _class_name(module_name)
    return _license_header() + dedent(
        f"""\
        from responses_api_agents.simple_agent.app import SimpleAgent
        from responses_api_agents.{module_name}_agent.app import {class_name}Agent


        def test_agent_extends_simple_agent() -> None:
            assert issubclass({class_name}Agent, SimpleAgent)
        """
    )


def _requirements(root: Path) -> str:
    return "-e nemo-gym[dev] @ ../../\n" if (root / "pyproject.toml").is_file() else "nemo-gym[dev]\n"


def _rollout_driver() -> str:
    return _license_header() + dedent(
        f'''\
        """{SCAFFOLD_PLACEHOLDER}: Add custom rollout orchestration here."""

        from collections.abc import Mapping
        from typing import Any


        async def run_rollout_collection(
            rollout_collection_config: Any,
            _global_config_dict: Mapping[str, Any],
        ) -> None:
            from nemo_gym.rollout_collection import RolloutCollectionHelper

            await RolloutCollectionHelper().run_from_config(rollout_collection_config)
        '''
    )


def _validate_target(root: Path, path: Path) -> None:
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ScaffoldError(f"scaffold target traverses symlink {current}")


def _write_files(root: Path, asset_dir: Path, files: Mapping[Path, str]) -> ScaffoldResult:
    ordered = sorted(files.items(), key=lambda item: str(item[0]))
    conflicts: set[Path] = set()
    existing: list[Path] = []
    for path, content in ordered:
        _validate_target(root, path)
        parent = path.parent
        while parent != root:
            if parent.exists() and not parent.is_dir():
                conflicts.add(parent)
                break
            parent = parent.parent
        if path.exists():
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                conflicts.add(path)
            else:
                existing.append(path)
    if conflicts:
        raise ScaffoldConflictError(tuple(sorted(conflicts, key=str)))

    created: list[Path] = []
    for path, content in ordered:
        if path in existing:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)
        created.append(path)
    return ScaffoldResult(asset_dir, tuple(created), tuple(existing))


def _license_header() -> str:
    return dedent(
        """\
        # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
        # SPDX-License-Identifier: Apache-2.0

        """
    )


__all__ = [
    "ENVIRONMENT_KINDS",
    "INTEGRATION_PROFILES",
    "SCAFFOLD_PLACEHOLDER",
    "ScaffoldConflictError",
    "ScaffoldError",
    "ScaffoldResult",
    "scaffold_environment",
]
