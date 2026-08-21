# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate manifest-backed environment and benchmark skeletons."""

from __future__ import annotations

import ast
import json
import keyword
import re
from dataclasses import dataclass
from os.path import relpath
from pathlib import Path
from textwrap import dedent
from typing import Any, Mapping

import yaml
from pydantic import ValidationError

from nemo_gym import component_search_roots
from nemo_gym.config_types import ConfigError
from nemo_gym.environment.manifest import (
    Determinism,
    EnvironmentKind,
    EnvironmentManifest,
    IntegrationProfile,
    ManifestDataset,
    Reward,
    dump_manifest,
)


_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_TODO = "TODO"


class ScaffoldError(ConfigError):
    """A scaffold request cannot be completed safely."""


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


@dataclass(frozen=True)
class _Composition:
    kind: EnvironmentKind
    name: str
    profile: IntegrationProfile
    module_name: str
    asset_dir: Path
    resource_implementation: str
    resource_instance: str
    agent_implementation: str
    agent_instance: str
    dataset: ManifestDataset
    reward: Reward
    config_reference: str
    prompt_path: str | None
    rollout_driver: str | None
    reused_verifier: _ReusedVerifier | None


def scaffold_environment(
    *,
    kind: EnvironmentKind | str,
    name: str,
    profile: IntegrationProfile | str = IntegrationProfile.CUSTOM_GYM_VERIFIER,
    reuse_verifier: str | None = None,
    reward_range: tuple[float, float] | None = None,
    higher_is_better: bool | None = None,
    root: str | Path | None = None,
) -> ScaffoldResult:
    """Create a profile-aware skeleton without replacing user content.

    Profiles select authoring extension points only. The generated ``config.yaml``
    remains the sole source of runtime wiring. Custom-loop templates delegate to
    default behavior until their generated extension point is replaced.
    """

    environment_kind = _parse_kind(kind)
    integration_profile = _parse_profile(profile)
    creates_python_component = reuse_verifier is None or integration_profile != IntegrationProfile.CUSTOM_GYM_VERIFIER
    _validate_name(name, requires_python_module=creates_python_component)
    scaffold_root = _resolve_root(root)

    reused = None
    if reuse_verifier is not None:
        if not _NAME_PATTERN.fullmatch(reuse_verifier) or keyword.iskeyword(reuse_verifier):
            raise ScaffoldError("reuse_verifier must be a canonical resources-server name")
        reused = _resolve_reused_verifier(scaffold_root, reuse_verifier)

    if reused:
        if reward_range is None or higher_is_better is None:
            raise ScaffoldError("reuse_verifier requires reward_range and higher_is_better")
    elif reward_range is not None or higher_is_better is not None:
        raise ScaffoldError("reward_range and higher_is_better are only accepted with reuse_verifier")
    try:
        reward = Reward(
            range=(0.0, 1.0) if reward_range is None else reward_range,
            higher_is_better=True if higher_is_better is None else higher_is_better,
        )
    except ValidationError as error:
        issue = error.errors(include_url=False, include_context=False, include_input=False)[0]
        raise ScaffoldError(f"invalid reward contract: {issue['msg']}") from error

    composition = _composition(
        root=scaffold_root,
        kind=environment_kind,
        name=name,
        profile=integration_profile,
        reused=reused,
        reward=reward,
    )
    files = _render_manifest_composition(scaffold_root, composition)
    return _write_files(scaffold_root, composition.asset_dir, files)


def scaffold_resources_server(
    *,
    directory: str | Path,
    checkout_root: str | Path | None = None,
) -> ScaffoldResult:
    """Create a standalone resources-server scaffold."""
    resource_dir = Path(directory).expanduser().resolve()
    module_name = resource_dir.name
    _validate_name(module_name, requires_python_module=True)
    resolved_checkout_root = Path(checkout_root).resolve() if checkout_root is not None else None
    files = _standalone_resources_server_files(resource_dir, module_name, resolved_checkout_root)
    return _write_files(resource_dir.parent, resource_dir, files)


def _parse_kind(value: EnvironmentKind | str) -> EnvironmentKind:
    try:
        return EnvironmentKind(value)
    except ValueError as error:
        choices = ", ".join(kind.value for kind in EnvironmentKind)
        raise ScaffoldError(f"kind must be one of: {choices}; got {value!r}") from error


def _parse_profile(value: IntegrationProfile | str) -> IntegrationProfile:
    try:
        return IntegrationProfile(value)
    except ValueError as error:
        choices = ", ".join(profile.value for profile in IntegrationProfile)
        raise ScaffoldError(f"profile must be one of: {choices}; got {value!r}") from error


def _validate_name(name: str, *, requires_python_module: bool) -> None:
    if not _NAME_PATTERN.fullmatch(name) or keyword.iskeyword(name):
        raise ScaffoldError("name must be one lowercase path segment containing letters, digits, '.', '_' or '-'")
    if requires_python_module and not name.isidentifier():
        raise ScaffoldError("a scaffold that creates Python components requires a lowercase Python identifier")


def _resolve_root(root: str | Path | None) -> Path:
    requested = Path.cwd() if root is None else Path(root).expanduser()
    if requested.is_symlink():
        raise ScaffoldError(f"scaffold root must not be a symlink: {requested}")
    if requested.exists() and not requested.is_dir():
        raise ScaffoldError(f"scaffold root must be a directory: {requested}")
    return requested.resolve()


def _class_name(module_name: str) -> str:
    return "".join(part.capitalize() for part in module_name.split("_") if part)


def _composition(
    *,
    root: Path,
    kind: EnvironmentKind,
    name: str,
    profile: IntegrationProfile,
    reused: _ReusedVerifier | None,
    reward: Reward,
) -> _Composition:
    module_name = name
    parent_name = "benchmarks" if kind == EnvironmentKind.BENCHMARK else "environments"
    asset_dir = root / parent_name / name
    resource_implementation = reused.selector if reused else module_name
    resource_instance = reused.resource_instance if reused else f"{module_name}_resources_server"
    custom_agent = profile in {
        IntegrationProfile.CUSTOM_GYM_AGENT_LOOP,
        IntegrationProfile.EXTERNAL_AGENT_LOOP,
    }
    agent_implementation = f"{module_name}_agent" if custom_agent else "simple_agent"
    agent_instance = f"{module_name}_agent"
    if reused and agent_instance in {reused.resource_instance, reused.agent_instance}:
        agent_instance = f"{module_name}_catalog_agent"
    if reused and agent_instance in {reused.resource_instance, reused.agent_instance}:
        raise ScaffoldError(f"name {name!r} collides with instances in reused verifier {reused.selector!r}")

    base_path = f"{parent_name}/{name}"
    prompt_path = f"{base_path}/prompt.yaml" if kind == EnvironmentKind.BENCHMARK else None
    prepare_path = f"{base_path}/prepare.py" if kind == EnvironmentKind.BENCHMARK else None
    rollout_driver = (
        f"{parent_name}.{name}.rollout_driver:run_rollout_collection"
        if profile == IntegrationProfile.EXTERNAL_ROLLOUT_DRIVER
        else None
    )
    dataset = ManifestDataset.model_validate(
        {
            "name": name,
            "type": "benchmark" if kind == EnvironmentKind.BENCHMARK else "example",
            "jsonl_fpath": f"{base_path}/data/example.jsonl",
            "prepare_script": prepare_path,
            "prompt_config": prompt_path,
            "num_repeats": 1,
        }
    )
    return _Composition(
        kind=kind,
        name=name,
        profile=profile,
        module_name=module_name,
        asset_dir=asset_dir,
        resource_implementation=resource_implementation,
        resource_instance=resource_instance,
        agent_implementation=agent_implementation,
        agent_instance=agent_instance,
        dataset=dataset,
        reward=reward,
        config_reference=(
            reused.config_reference if reused else f"resources_servers/{module_name}/configs/{module_name}.yaml"
        ),
        prompt_path=prompt_path,
        rollout_driver=rollout_driver,
        reused_verifier=reused,
    )


def _manifest(composition: _Composition) -> EnvironmentManifest:
    data: dict[str, Any] = {
        "name": composition.name,
        "version": "0.1.0",
        "kind": composition.kind,
        "integration_profile": composition.profile,
        "domain": "other",
        "description": f"{_TODO}: Describe the {composition.name} {composition.kind.value}.",
        "modality": "text",
        "licensing": "unknown",
        "authors": [_TODO],
        "reward": composition.reward,
        "determinism": Determinism.UNKNOWN,
        "resources_server": composition.resource_implementation,
        "agent_server": composition.agent_implementation,
        "model_server": "policy_model",
        "datasets": [composition.dataset],
        "rollout_driver": composition.rollout_driver,
    }
    if composition.kind == EnvironmentKind.BENCHMARK:
        data.update(canonical_split=_TODO, standard_prompt_config=composition.prompt_path)
    return EnvironmentManifest.model_validate(data)


def _asset_config(composition: _Composition) -> str:
    agent_config: dict[str, Any] = {
        "resources_server": {"type": "resources_servers", "name": composition.resource_instance},
        "model_server": {"type": "responses_api_models", "name": "policy_model"},
        "datasets": [composition.dataset.model_dump(mode="json", exclude_none=True)],
    }
    reused_agent = composition.reused_verifier.agent_instance if composition.reused_verifier else None
    if reused_agent:
        agent_entry: dict[str, Any] = {
            "_inherit_from": reused_agent,
            "responses_api_agents": {composition.agent_implementation: agent_config},
        }
    else:
        agent_entry = {
            "responses_api_agents": {composition.agent_implementation: {"entrypoint": "app.py", **agent_config}}
        }
    config: dict[str, Any] = {
        "config_paths": [composition.config_reference],
        composition.agent_instance: agent_entry,
    }
    if composition.rollout_driver:
        config["rollout_collection_driver"] = composition.rollout_driver
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def _render_manifest_composition(root: Path, composition: _Composition) -> dict[Path, str]:
    asset = composition.asset_dir
    files = {
        asset / "__init__.py": _license_header(),
        asset / "manifest.yaml": dump_manifest(_manifest(composition)),
        asset / "config.yaml": _asset_config(composition),
        asset / "README.md": _asset_readme(composition),
    }
    if composition.kind == EnvironmentKind.BENCHMARK:
        source = json.dumps({"question": "What is 6 x 7?", "expected_answer": "42"}) + "\n"
        files.update(
            {
                asset / "data" / "source.jsonl": source,
                asset / "data" / "example.jsonl": source,
                asset / "prompt.yaml": _benchmark_prompt(),
                asset / "prepare.py": _benchmark_prepare(composition.name),
            }
        )
    else:
        files[asset / "data" / "example.jsonl"] = _environment_example(composition.agent_instance)

    if composition.reused_verifier is None:
        resource_dir = root / "resources_servers" / composition.module_name
        files.update(
            _resource_component_files(
                resource_dir,
                module_name=composition.module_name,
                config=_manifest_resources_server_config(composition),
                readme=f"# {composition.module_name} resources server\n\n{_TODO}: Document the verifier.\n",
                requirements=_requirements(resource_dir, _gym_checkout_root(root)),
            )
        )
    if composition.profile in {
        IntegrationProfile.CUSTOM_GYM_AGENT_LOOP,
        IntegrationProfile.EXTERNAL_AGENT_LOOP,
    }:
        agent_dir = root / "responses_api_agents" / composition.agent_implementation
        files.update(_agent_files(root, agent_dir, composition))
    if composition.rollout_driver:
        files[asset / "rollout_driver.py"] = _rollout_driver()
    return files


def _asset_readme(composition: _Composition) -> str:
    scorer = (
        composition.reused_verifier.selector
        if composition.reused_verifier
        else f"resources_servers/{composition.module_name}"
    )
    return dedent(
        f"""\
        # {composition.name}

        {_TODO}: Describe this {composition.kind.value} and replace the sample data.

        - Integration profile: `{composition.profile.value}`
        - Scorer: `{scorer}`
        """
    )


def _environment_example(agent_instance: str) -> str:
    return (
        json.dumps(
            {
                "responses_create_params": {
                    "input": [{"role": "user", "content": "What is 6 x 7? Reply with only the answer."}]
                },
                "expected_answer": "42",
                "agent_ref": {"type": "responses_api_agents", "name": agent_instance},
            }
        )
        + "\n"
    )


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

        import json
        from pathlib import Path


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
                    row = json.loads(line)
                    if (
                        not isinstance(row, dict)
                        or not isinstance(row.get("question"), str)
                        or not isinstance(row.get("expected_answer"), str)
                    ):
                        raise ValueError(f"invalid source row {{line_number}}")
                    output_stream.write(
                        json.dumps({{"question": row["question"], "expected_answer": row["expected_answer"]}}) + "\\n"
                    )
            return output


        if __name__ == "__main__":
            prepare()
        '''
    )


def _resource_component_files(
    directory: Path,
    *,
    module_name: str,
    config: str,
    readme: str,
    requirements: str,
) -> dict[Path, str]:
    return {
        directory / "__init__.py": _license_header(),
        directory / "README.md": readme,
        directory / "app.py": _resources_server_app(module_name),
        directory / "configs" / f"{module_name}.yaml": config,
        directory / "requirements.txt": requirements,
        directory / "tests" / "__init__.py": _license_header(),
        directory / "tests" / "test_app.py": _resources_server_test(),
        directory / "tests" / "verifier_cases.jsonl": _verifier_cases(),
    }


def _manifest_resources_server_config(composition: _Composition) -> str:
    return yaml.safe_dump(
        {
            composition.resource_instance: {
                "resources_servers": {
                    composition.resource_implementation: {
                        "entrypoint": "app.py",
                        "domain": "other",
                        "verified": False,
                    }
                }
            }
        },
        sort_keys=False,
        allow_unicode=True,
    )


def _standalone_resources_server_files(
    directory: Path,
    module_name: str,
    checkout_root: Path | None,
) -> dict[Path, str]:
    files = _resource_component_files(
        directory,
        module_name=module_name,
        config=_standalone_resources_server_config(module_name),
        readme=_standalone_resources_server_readme(),
        requirements=_requirements(directory, checkout_root),
    )
    files[directory / "data" / ".gitignore"] = _standalone_data_gitignore()
    return files


def _standalone_resources_server_config(module_name: str) -> str:
    return dedent(
        f"""\
        # Resources server: owns this environment's task verification (verify()) and reward.
        {module_name}_resources_server:          # instance name — how agents/CLI refer to this server
          resources_servers:                    # server type: resources_servers | responses_api_agents | responses_api_models
            {module_name}:                      # implementation directory under resources_servers/
              entrypoint: app.py                # server entry module
              domain: other                     # task domain; change to the closest supported domain
              verified: false                   # set true once the benchmark has been baselined and reviewed

        # Pair the server with the default agent and dataset slots.
        {module_name}_simple_agent:             # pass this instance as --agent to gym eval run
          responses_api_agents:
            simple_agent:
              entrypoint: app.py
              resources_server:
                type: resources_servers
                name: {module_name}_resources_server
              model_server:
                type: responses_api_models
                name: policy_model
              datasets:
              - name: train
                type: train
                jsonl_fpath: resources_servers/{module_name}/data/train.jsonl
                num_repeats: 1
                license: Apache 2.0
                # To fetch this split from a registry, add a source: block.
              - name: validation
                type: validation
                jsonl_fpath: resources_servers/{module_name}/data/validation.jsonl
                num_repeats: 1
                license: Apache 2.0
              - name: example
                type: example
                jsonl_fpath: resources_servers/{module_name}/data/example.jsonl
                num_repeats: 1
        """
    )


def _standalone_resources_server_readme() -> str:
    return dedent(
        """\
        # Description

        Data links: ?

        # Licensing information
        Code: ?
        Data: ?

        Dependencies
        - nemo_gym: Apache 2.0
        ?
        """
    )


def _standalone_data_gitignore() -> str:
    return dedent(
        """\
        *train.jsonl
        *validation.jsonl
        *train_prepare.jsonl
        *validation_prepare.jsonl
        *example_prepare.jsonl
        """
    )


def _resources_server_app(module_name: str) -> str:
    class_name = _class_name(module_name)
    return _license_header() + dedent(
        f"""\
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


        class {class_name}Verifier:
            async def verify(self, body: {class_name}VerifyRequest) -> BaseVerifyResponse:
                reward = float(body.response.output_text.strip() == body.expected_answer.strip())
                return BaseVerifyResponse(**body.model_dump(), reward=reward)


        class {class_name}ResourcesServer({class_name}Verifier, SimpleResourcesServer):
            config: {class_name}ResourcesServerConfig


        VERIFIER_FIXTURE = VerifierFixture(
            server_factory={class_name}Verifier,
            request_model={class_name}VerifyRequest,
            cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
        )


        if __name__ == "__main__":
            {class_name}ResourcesServer.run_webserver()
        """
    )


def _resources_server_test() -> str:
    return _license_header() + dedent(
        """\
        import asyncio

        from nemo_gym.verifier_fixture import exercise_verifier_fixture

        from ..app import VERIFIER_FIXTURE


        def test_verifier_fixture() -> None:
            asyncio.run(
                exercise_verifier_fixture(
                    VERIFIER_FIXTURE,
                    reward_range=(0.0, 1.0),
                    higher_is_better=True,
                    determinism="unknown",
                )
            )
        """
    )


def _verifier_cases() -> str:
    def response(text: str) -> dict[str, Any]:
        return {
            "id": "fixture_response",
            "created_at": 0,
            "model": "fixture",
            "object": "response",
            "output": [
                {
                    "id": "fixture_message",
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
            "responses_create_params": {"input": "What is 6 x 7?"},
            "response": response(text),
            "expected_answer": "42",
        }

    cases = [
        {"name": "correct", "kind": "full_reward", "request": request("42"), "expected_reward": 1.0},
        {"name": "incorrect", "kind": "zero_reward", "request": request("41"), "expected_reward": 0.0},
        {
            "name": "missing response",
            "kind": "malformed",
            "request": {"responses_create_params": {"input": "missing response"}, "expected_answer": "42"},
            "expected_error": "response",
        },
    ]
    return "".join(json.dumps(case) + "\n" for case in cases)


def _agent_files(root: Path, directory: Path, composition: _Composition) -> dict[Path, str]:
    instruction = f"{_TODO}: Implement the `{composition.profile.value}` agent behavior."
    if composition.profile == IntegrationProfile.EXTERNAL_AGENT_LOOP:
        instruction += (
            " Replace run() with external episode delegation, then make responses() raise NotImplementedError."
        )
    return {
        directory / "__init__.py": _license_header(),
        directory / "README.md": f"# {composition.agent_implementation}\n\n{instruction}\n",
        directory / "app.py": _agent_app(composition.module_name, composition.profile),
        directory / "requirements.txt": _requirements(directory, _gym_checkout_root(root)),
    }


def _agent_app(module_name: str, profile: IntegrationProfile) -> str:
    class_name = _class_name(module_name)
    if profile == IntegrationProfile.CUSTOM_GYM_AGENT_LOOP:
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
                # TODO: Implement the custom Gym agent strategy.
                return await super().responses(request, response, body)
        """
    else:
        body = f"""\
        from fastapi import Request, Response

        from nemo_gym.base_responses_api_agent import Body
        from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
        from responses_api_agents.simple_agent.app import (
            SimpleAgent,
            SimpleAgentRunRequest,
            SimpleAgentVerifyResponse,
        )


        class {class_name}Agent(SimpleAgent):
            async def responses(
                self,
                request: Request,
                response: Response,
                body: NeMoGymResponseCreateParamsNonStreaming = Body(),
            ) -> NeMoGymResponse:
                # TODO: Replace this fallback with raise NotImplementedError after implementing run().
                return await super().responses(request, response, body)

            async def run(self, request: Request, body: SimpleAgentRunRequest) -> SimpleAgentVerifyResponse:
                # TODO: Replace this fallback with external episode delegation.
                return await super().run(request, body)
        """
    return (
        _license_header()
        + dedent(body)
        + dedent(
            f"""\


        if __name__ == "__main__":
            {class_name}Agent.run_webserver()
        """
        )
    )


def _rollout_driver() -> str:
    return _license_header() + dedent(
        '''\
        from collections.abc import Mapping
        from typing import Any


        async def run_rollout_collection(
            rollout_collection_config: Any,
            _global_config_dict: Mapping[str, Any],
        ) -> None:
            """TODO: Replace this delegation with custom rollout coordination."""
            from nemo_gym.rollout_collection import RolloutCollectionHelper

            await RolloutCollectionHelper().run_from_config(rollout_collection_config)
        '''
    )


def _gym_checkout_root(root: Path) -> Path | None:
    return root if (root / "pyproject.toml").is_file() and (root / "nemo_gym").is_dir() else None


def _requirements(directory: Path, checkout_root: Path | None) -> str:
    if checkout_root is None:
        return "nemo-gym[dev]\n"
    return f"-e nemo-gym[dev] @ {relpath(checkout_root, directory)}\n"


def _server_entries(raw: Mapping[str, Any], server_type: str) -> list[tuple[str, str, Mapping[str, Any]]]:
    entries = []
    for instance_name, value in raw.items():
        implementations = value.get(server_type) if isinstance(value, Mapping) else None
        if not isinstance(implementations, Mapping):
            continue
        for implementation, config in implementations.items():
            if not isinstance(config, Mapping):
                raise ScaffoldError(f"{server_type} config for {implementation!r} is not a mapping")
            entries.append((str(instance_name), str(implementation), config))
    return entries


def _resolve_reused_verifier(root: Path, selector: str) -> _ReusedVerifier:
    config_reference = f"resources_servers/{selector}/configs/{selector}.yaml"
    candidates = list(
        dict.fromkeys((search_root / config_reference).resolve() for search_root in (root, *component_search_roots()))
    )
    config_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if config_path is None:
        searched = "\n".join(f"  - {candidate}" for candidate in candidates)
        raise ScaffoldError(f"reuse_verifier {selector!r} was not found. Looked in:\n{searched}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ScaffoldError(f"could not read reused verifier config {config_path}: {error}") from error
    if not isinstance(raw, Mapping):
        raise ScaffoldError(f"reused verifier config {config_path} is not a mapping")

    resources = _server_entries(raw, "resources_servers")
    if len(resources) != 1 or resources[0][1] != selector:
        raise ScaffoldError(
            f"reused verifier {selector!r} config must define exactly one resources-server instance of that type"
        )
    if _server_entries(raw, "responses_api_models"):
        raise ScaffoldError(f"reused verifier {selector!r} config may not bundle a model server")
    resource_instance, _implementation, resource_config = resources[0]
    entrypoint = resource_config.get("entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint:
        raise ScaffoldError(f"reused verifier {selector!r} does not define a resources-server entrypoint")
    app_path = Path(entrypoint)
    if not app_path.is_absolute():
        app_path = config_path.parent.parent / app_path
    _require_fixture_export(app_path.resolve(), selector)

    agents = _server_entries(raw, "responses_api_agents")
    if len(agents) > 1 or (agents and agents[0][1] != "simple_agent"):
        raise ScaffoldError(f"reused verifier {selector!r} config may bundle only one simple_agent")
    agent_instance = None
    if agents:
        agent_instance, _agent_implementation, agent_config = agents[0]
        resource_ref = agent_config.get("resources_server")
        if not isinstance(resource_ref, Mapping) or resource_ref.get("name") != resource_instance:
            raise ScaffoldError(
                f"reused verifier {selector!r} simple_agent must reference resources instance {resource_instance!r}"
            )
    return _ReusedVerifier(selector, config_reference, resource_instance, agent_instance)


def _require_fixture_export(app_path: Path, selector: str) -> None:
    try:
        tree = ast.parse(app_path.read_text(encoding="utf-8"), filename=str(app_path))
    except FileNotFoundError as error:
        raise ScaffoldError(f"reused verifier {selector!r} entrypoint was not found: {app_path}") from error
    except (OSError, UnicodeError, SyntaxError) as error:
        raise ScaffoldError(f"could not inspect reused verifier entrypoint {app_path}: {error}") from error
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if any((alias.asname or alias.name) == "VERIFIER_FIXTURE" for alias in node.names):
                return
            continue
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        constructor = value.func if isinstance(value, ast.Call) else None
        declares_fixture = isinstance(constructor, ast.Name) and constructor.id == "VerifierFixture"
        declares_fixture |= isinstance(constructor, ast.Attribute) and constructor.attr == "VerifierFixture"
        if declares_fixture and any(
            isinstance(target, ast.Name) and target.id == "VERIFIER_FIXTURE" for target in targets
        ):
            return
    raise ScaffoldError(f"reused verifier {selector!r} must declare a VERIFIER_FIXTURE from {app_path}")


def _validate_target(root: Path, path: Path) -> None:
    current = root
    for part in path.relative_to(root).parts:
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
            try:
                matches = path.is_file() and path.read_text(encoding="utf-8") == content
            except (OSError, UnicodeError) as error:
                raise ScaffoldError(f"could not inspect scaffold target {path}: {error}") from error
            if matches:
                existing.append(path)
            else:
                conflicts.add(path)
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
    return ScaffoldResult(asset_dir=asset_dir, created=tuple(created), existing=tuple(existing))


def _license_header() -> str:
    return dedent(
        """\
        # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
        # SPDX-License-Identifier: Apache-2.0

        """
    )


__all__ = [
    "ScaffoldConflictError",
    "ScaffoldError",
    "ScaffoldResult",
    "scaffold_environment",
    "scaffold_resources_server",
]
