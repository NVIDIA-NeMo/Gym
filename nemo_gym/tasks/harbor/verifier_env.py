# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor's separate-verifier rules.

Which environment the verifier runs in, the verifier table or a copy of the task environment,
whether tests/ is a build context of its own, which sidecars a tests/docker-compose.yaml may
declare beside main, and the two refusals the loader raises when a separate verifier has no
image and no definition. The loader and the build planner both read these rules.
"""

from pathlib import Path

import yaml

from nemo_gym.tasks.harbor.errors import HarborTaskError
from nemo_gym.tasks.harbor.models import HarborEnvironment, HarborStep, HarborTaskConfig
from nemo_gym.tasks.harbor.records import BUILT_IMAGES_FILE, MAIN_SERVICE


# The Dockerfile Harbor builds the separate verifier's image from, with ``tests/`` as its build context
# (``_verifier_env_build_context`` in Harbor's trial.py); see :func:`verifier_build_context`.
TESTS_DOCKERFILE = Path("tests") / "Dockerfile"
# The one Compose file Harbor reads for the separate verifier's environment (``environment_dir / "docker-compose.yaml"``
# in its docker.py), stacked on its build or prebuilt template; see :func:`verifier_compose_sidecars`.
TESTS_COMPOSE_FILE = Path("tests") / "docker-compose.yaml"


def effective_verifier_environment(config: HarborTaskConfig) -> HarborEnvironment | None:
    """The environment a separate verifier runs in, as Harbor resolves it, or ``None`` in shared mode.

    Harbor's ``resolve_effective_verifier_env_config`` (verifier_mode.py at v0.22.0) for a single-step task:
    ``[verifier.environment]`` when the task declares one, else a copy of ``[environment]``, so a task that only says
    ``environment_mode = "separate"`` grades in an environment shaped like the agent's, ``docker_image`` included.
    """
    if config.is_shared_verifier:
        return None
    if config.verifier.environment is not None:
        return config.verifier.environment
    return config.environment.model_copy(deep=True)


def step_verifier_mode(config: HarborTaskConfig, step: HarborStep) -> str:
    """``"separate"`` or ``"shared"`` for one step, as Harbor's ``resolve_step_verifier_mode`` (verifier_mode.py at
    v0.22.0) resolves it: the step's explicit ``[steps.verifier].environment_mode``, else ``separate`` when the step
    declares a ``[steps.verifier.environment]``, else the task's own mode (:attr:`HarborTaskConfig.is_shared_verifier`).
    Steps of one task may resolve differently."""
    if step.verifier.environment_mode is not None:
        return step.verifier.environment_mode
    if step.verifier.environment is not None:
        return "separate"
    return "shared" if config.is_shared_verifier else "separate"


def step_verifier_environment(config: HarborTaskConfig, step: HarborStep) -> HarborEnvironment | None:
    """The environment a step's separate verifier runs in, as Harbor's ``resolve_effective_verifier_env_config``
    resolves it with a step: ``[steps.verifier.environment]``, else ``[verifier.environment]``, else a copy of
    ``[environment]``; ``None`` when the step verifies in shared mode."""
    if step_verifier_mode(config, step) != "separate":
        return None
    if step.verifier.environment is not None:
        return step.verifier.environment
    if config.verifier.environment is not None:
        return config.verifier.environment
    return config.environment.model_copy(deep=True)


def step_verifier_build_context(task_folder: Path, step: HarborStep) -> Path:
    """The folder Harbor starts a step's separate verifier environment from (``_verifier_env_build_context`` with a
    step in trial.py at v0.22.0): ``steps/<name>/tests/`` when that folder exists, else the task's ``tests/``."""
    step_tests = Path(task_folder) / "steps" / step.name / "tests"
    return step_tests if step_tests.is_dir() else Path(task_folder) / TESTS_DOCKERFILE.parent


def verifier_build_environment(task_folder: Path, config: HarborTaskConfig) -> HarborEnvironment | None:
    """The effective verifier environment whose settings Harbor's build of ``tests/Dockerfile`` sees, or ``None``
    when nothing builds from ``tests/``: the task's own (:func:`effective_verifier_environment`) when it declares no
    ``docker_image``, else the first step's whose separate verifier declares none and starts from ``tests/``
    (:func:`step_verifier_build_context`; a step with a ``tests/`` folder of its own starts from that folder, which
    this server does not build, and the loader refuses such a step). Steps that would build ``tests/`` under
    different environments share the first one's settings here, where Harbor builds once per step."""
    environment = effective_verifier_environment(config)
    if environment is not None and not environment.docker_image:
        return environment
    tests = Path(task_folder) / TESTS_DOCKERFILE.parent
    for step in config.steps:
        step_environment = step_verifier_environment(config, step)
        if (
            step_environment is not None
            and not step_environment.docker_image
            and step_verifier_build_context(task_folder, step) == tests
        ):
            return step_environment
    return None


def verifier_build_context(task_folder: Path, config: HarborTaskConfig) -> Path | None:
    """The folder Harbor builds the separate verifier's image from, ``tests/``, when it builds one; else ``None``.

    Harbor starts the separate verifier's environment with ``tests/`` as its environment dir
    (``_verifier_env_build_context`` in trial.py at v0.22.0) and the effective verifier environment as its config
    (:func:`effective_verifier_environment`). Its Docker environment then uses the config's ``docker_image`` when there
    is one and otherwise runs ``docker compose build`` on ``tests/Dockerfile`` (``should_use_prebuilt_docker_image``,
    ``start`` in docker.py), whatever that Dockerfile holds: a ``FROM`` plus ``RUN`` file is built, not overlaid. So:
    shared mode, a declared or inherited ``docker_image``, or no ``tests/Dockerfile`` means no build. Without the file
    and without an image Harbor refuses the task (``require_agent_environment_definition`` on ``tests/``, which is
    always its verifier environment dir: there is no fallback to ``environment/``), and so does :func:`load_task`.
    Prepare plans one build entry per returned context (``nemo_gym.tasks.harbor.image_build.plan_builds``) and the
    loader reads its record. A multi-step task whose steps verify separately builds ``tests/`` for the steps that
    declare no image and start from it (:func:`verifier_build_environment`), whatever the task's own mode.
    """
    if verifier_build_environment(task_folder, config) is None:
        return None
    dockerfile = Path(task_folder) / TESTS_DOCKERFILE
    return dockerfile.parent if dockerfile.is_file() else None


def verifier_compose_sidecars(task_folder: Path) -> list[str]:
    """The services other than ``main`` that the task's ``tests/docker-compose.yaml`` declares, in file order; ``[]``
    without the file, or for one that does not parse or has no services mapping.

    Harbor's separate verifier is a whole Docker environment over ``tests/``: it stacks that Compose file on its build
    or prebuilt template (``_docker_compose_paths`` in docker.py at v0.22.0), builds every service with a ``build``
    section and starts them all with ``up``, image or built, prebuilt ``main`` or not. This server starts the
    separate verifier as a single sandbox, so a task that declares any such service is refused: by
    :func:`load_task` and, before any build, by ``ImageBuilder.build_many``. A ``main``-only file (cadir's and
    arc-crm-6's shape, runtime settings and ``build.args`` for ``main``) is read for ``main``'s build section only.
    """
    path = Path(task_folder) / TESTS_COMPOSE_FILE
    if not path.is_file():
        return []
    try:
        document = yaml.safe_load(path.read_text())
    except (yaml.YAMLError, UnicodeDecodeError, OSError):
        return []
    services = document.get("services") if isinstance(document, dict) else None
    if not isinstance(services, dict):
        return []
    return [str(name) for name in services if str(name) != MAIN_SERVICE]


def _verifier_build_required(task_folder: Path, root: Path) -> HarborTaskError:
    """``root`` is the dataset folder whose ``built-images.json`` the loader read (see ``_build_required``)."""
    return HarborTaskError(
        f"{task_folder / TESTS_DOCKERFILE} is the separate verifier's image (Harbor builds it from tests/) and no built "
        f"image is recorded in {root / BUILT_IMAGES_FILE}. `gym eval run` and `gym dataset validate` "
        "build it when buildah, podman or docker is installed (configured by the `builds:` block, see "
        "environment_servers/harbor/README.md), or declare [verifier.environment].docker_image with a prebuilt image."
    )


def _verifier_definition_required(task_folder: Path) -> HarborTaskError:
    """Harbor's refusal of a separate verifier environment with nothing to run: its environment dir is always
    ``tests/`` (``_verifier_env_build_context`` in trial.py), checked by ``require_agent_environment_definition``."""
    tests = task_folder / TESTS_DOCKERFILE.parent
    if (task_folder / TESTS_COMPOSE_FILE).is_file():
        return HarborTaskError(
            f"{task_folder / TESTS_COMPOSE_FILE} without {task_folder / TESTS_DOCKERFILE}: Harbor builds the separate "
            "verifier's image from that Compose file's main service, and this server builds it from tests/Dockerfile "
            "only. Add the Dockerfile under tests/, or declare [verifier.environment].docker_image with a prebuilt image."
        )
    return HarborTaskError(
        f"Task environment directory {tests} has no environment definition: the separate verifier's effective "
        "environment names no docker_image and tests/ has no Dockerfile, so Harbor refuses the task "
        "(require_agent_environment_definition; the verifier environment dir is always tests/). Set "
        "[verifier.environment].docker_image or add tests/Dockerfile."
    )
