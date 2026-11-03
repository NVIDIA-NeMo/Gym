# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor's separate-verifier rules.

Which environment the verifier runs in, the verifier table or a copy of the task environment,
whether tests/ is a build context of its own, which top-level keys, sidecars and main-service keys a
tests/docker-compose.yaml may declare (and which such files cannot be read at all), and the two
refusals the loader raises when a separate verifier has no image and no definition. The loader and
the build planner both read these rules.
"""

from pathlib import Path
from typing import Any

import yaml

from nemo_gym.tasks.harbor.errors import HarborTaskError
from nemo_gym.tasks.harbor.models import HarborEnvironment, HarborTaskConfig, HarborTaskStep
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


def step_verifier_mode(config: HarborTaskConfig, step: HarborTaskStep) -> str:
    """``"separate"`` or ``"shared"`` for one step, as Harbor's ``resolve_step_verifier_mode`` (verifier_mode.py at
    v0.22.0) resolves it: the step's explicit ``[steps.verifier].environment_mode``, else ``separate`` when the step
    declares a ``[steps.verifier.environment]``, else the task's own mode (:attr:`HarborTaskConfig.is_shared_verifier`).
    Steps of one task may resolve differently."""
    if step.verifier.environment_mode is not None:
        return step.verifier.environment_mode
    if step.verifier.environment is not None:
        return "separate"
    return "shared" if config.is_shared_verifier else "separate"


def step_verifier_environment(config: HarborTaskConfig, step: HarborTaskStep) -> HarborEnvironment | None:
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


def step_verifier_build_context(task_folder: Path, step: HarborTaskStep) -> Path:
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


# The keys of ``main`` in a ``tests/docker-compose.yaml`` this server reads when prepare builds the verifier from
# ``tests/Dockerfile``: its ``build`` section (``service_build_settings``) and its ``image``, pinned to Harbor's
# ``${MAIN_IMAGE_NAME}`` by ``validate_task_compose`` before the build; ``pull_policy`` is the build-or-pull policy
# the loader drops from every Compose service. For a prebuilt verifier (an effective verifier environment with a
# ``docker_image``) nothing of ``main`` is read, and its ``image`` and ``build`` are overrides. Harbor's prebuilt
# template says ``image: ${PREBUILT_IMAGE_NAME}`` and the task file comes last (``_docker_compose_paths``), so a
# literal ``image`` there runs in place of the declared image and ``${MAIN_IMAGE_NAME}`` runs the ``hb__<tests hash>``
# nobody built. Harbor's prebuilt ``start`` (docker.py) skips ``compose build`` but runs ``up --detach --wait``, and
# Compose's ``up`` builds a service that has a ``build`` section itself: always under ``pull_policy: build`` (what
# Harbor's own build template relies on), and whenever the declared image is neither cached nor pulled otherwise;
# the result is tagged with the declared name and graded in, while this server grades in the declared image. So a
# ``build`` section is refused there, and so is ``pull_policy: build``, which names that build (with no section to
# name it asks Harbor's Compose for a build this server never runs either way); ``always``, ``missing`` and ``never``
# only say when Harbor pulls the declared image, which this server pulls itself, and stay. Any other key is a
# runtime setting of the verifier's container, see :func:`verifier_compose_main_overrides`.
VERIFIER_MAIN_KEYS = frozenset({"image", "build", "pull_policy"})
PREBUILT_VERIFIER_MAIN_KEYS = frozenset({"pull_policy"})
BUILD_PULL_POLICY = "build"
# The top-level keys of a ``tests/docker-compose.yaml`` a separate verifier may carry: ``services`` is what this
# server reads (``main`` under the rules above, anything beside it a refused sidecar); ``name`` and the obsolete
# ``version`` shape no container (Harbor names the project itself, ``--project-name`` from its session id in
# docker.py at v0.22.0, and Compose ignores ``version`` with a warning); an ``x-`` extension field is inert until a
# service references it, and a reference on ``main`` is a key the rules above judge. Every other key (``networks``,
# ``volumes``, ``configs``, ``secrets``, ``include``, one Compose adds later) is refused: :func:`verifier_compose_refusal`.
VERIFIER_COMPOSE_TOP_LEVEL_KEYS = frozenset({"services", "name", "version"})
COMPOSE_EXTENSION_PREFIX = "x-"


def _verifier_compose_services(task_folder: Path) -> dict[Any, Any]:
    """The ``services`` mapping of the task's ``tests/docker-compose.yaml``; ``{}`` without the file, or for one that
    does not parse or has no services mapping (a file the loader refuses first, :func:`verifier_compose_refusal`)."""
    path = Path(task_folder) / TESTS_COMPOSE_FILE
    if not path.is_file():
        return {}
    try:
        document = yaml.safe_load(path.read_text())
    except (yaml.YAMLError, UnicodeDecodeError, OSError):
        return {}
    services = document.get("services") if isinstance(document, dict) else None
    return services if isinstance(services, dict) else {}


def verifier_compose_refusal(task_folder: Path) -> str | None:
    """Why the task's ``tests/docker-compose.yaml`` is not the file :func:`verifier_compose_sidecars` and
    :func:`verifier_compose_main_overrides` read, as the loader's message, or ``None`` when it is (or there is none).

    Those readers see the file's own ``services`` mapping and nothing else, so what they cannot see must not reach
    Harbor's verifier either. Harbor stacks the whole file on its verifier template (``_docker_compose_paths`` in
    docker.py at v0.22.0) and ``compose up`` applies every top-level key of it to the verifier's project, so only the
    keys this server reads or that cannot shape the verifier's container pass (:data:`VERIFIER_COMPOSE_TOP_LEVEL_KEYS`
    and ``x-`` extension fields) and any other is refused by name. A top-level ``include`` merges the included files'
    services into the project before the stacking, so ``main.environment`` or a ``build`` section written in an
    included file reaches Harbor's verifier container as if written here, while this server reads ``services: {}``
    and grades in the declared image without a word (``validate_task_compose`` refuses ``include`` for a Dockerfile
    build the same way; this is the loader's refusal, for every separate verifier, prebuilt ones, which that
    validator never sees, included). A top-level ``networks`` reconfigures the project's ``default`` network, the one
    Compose attaches ``main`` to because neither of Harbor's templates gives it a ``networks`` or ``network_mode`` key
    and, under Harbor's default ``public`` policy, no egress overlay does either: ``default: {internal: true}`` grades
    Harbor's verifier without egress (``driver_opts``, ``ipam``, ``enable_ipv6`` likewise) while this server's single
    sandbox grades with it. Top-level ``volumes``, ``configs`` and ``secrets`` are inert until a service references
    them, and a reference on ``main`` is a key :func:`verifier_compose_main_overrides` refuses, but the class is closed
    rather than judged key by key, since each is a project setting the readers never see. A file that is not valid
    YAML or not a mapping, or whose ``services`` is not one, fails Harbor's ``compose up`` (and Harbor's own reader of
    the file, ``_egress_controlled_service_names`` in docker.py, parses it with ``yaml.safe_load``), so the trial
    errors under Harbor where the readers above would see ``{}``; refused as ``validate_task_compose`` refuses it. An
    empty file is an empty mapping, as it is there.
    """
    path = Path(task_folder) / TESTS_COMPOSE_FILE
    if not path.is_file():
        return None
    try:
        document = yaml.safe_load(path.read_text()) or {}
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        return f"{path} is not valid YAML: {exc}"
    except OSError as exc:
        return f"{path} cannot be read: {exc}"
    if not isinstance(document, dict):
        return f"{path} is not a mapping"
    if "include" in document:
        return (
            f"{path}: include is refused for the separate verifier: Compose merges the included files' services into "
            "Harbor's verifier project (main's keys included) and this server reads the file's own services alone"
        )
    others = [
        str(key)
        for key in document
        if str(key) not in VERIFIER_COMPOSE_TOP_LEVEL_KEYS and not str(key).startswith(COMPOSE_EXTENSION_PREFIX)
    ]
    if others:
        one = len(others) == 1
        return (
            f"{path}: top-level {', '.join(others)} {'is' if one else 'are'} refused for the separate verifier: "
            f"Harbor's compose up applies {'it' if one else 'them'} to the verifier project and this server reads "
            "services alone"
        )
    services = document.get("services")
    if services is not None and not isinstance(services, dict):
        return f"{path}: services is not a mapping"
    return None


def verifier_compose_sidecars(task_folder: Path) -> list[str]:
    """The services other than ``main`` that the task's ``tests/docker-compose.yaml`` declares, in file order; ``[]``
    without the file, or for one that does not parse or has no services mapping (:func:`verifier_compose_refusal`
    names such a file, and a top-level key other than ``services``, first).

    Harbor's separate verifier is a whole Docker environment over ``tests/``: it stacks that Compose file on its build
    or prebuilt template (``_docker_compose_paths`` in docker.py at v0.22.0), builds every service with a ``build``
    section and starts them all with ``up``, image or built, prebuilt ``main`` or not. This server starts the
    separate verifier as a single sandbox, so a task that declares any such service is refused: by
    :func:`load_task` and, before any build, by ``ImageBuilder.build_many``. A ``main``-only file (cadir's and
    arc-crm-6's shape) is read for ``main``'s build section, and its ``image`` when prepare builds the verifier; a
    runtime setting on ``main`` (every key of arc-crm-6's file: ``network_mode``, ``pids_limit``, ``security_opt``,
    ``cap_drop``) is refused the same way (:func:`verifier_compose_main_overrides`).
    """
    return [str(name) for name in _verifier_compose_services(task_folder) if str(name) != MAIN_SERVICE]


def verifier_compose_main_overrides(task_folder: Path, *, prebuilt: bool) -> list[str]:
    """The keys of ``main`` in the task's ``tests/docker-compose.yaml`` other than :data:`VERIFIER_MAIN_KEYS` (other
    than :data:`PREBUILT_VERIFIER_MAIN_KEYS` when ``prebuilt``: the effective verifier environment names a
    ``docker_image``, so nothing builds from ``tests/``, and a ``pull_policy`` of ``build`` counts too), in file
    order; ``[]`` without the file or without a ``main`` mapping.

    Harbor stacks that file on its verifier template for the build and for ``up`` (``_docker_compose_paths`` in
    docker.py at v0.22.0), so ``environment``, ``env_file``, ``volumes``, ``command`` and every other runtime key of
    ``main`` reach the verifier's container: a ``test.sh`` that reads ``CHECK_MODE`` from ``main.environment`` sees it
    there, an ``image`` beside a prebuilt verifier replaces the declared image Harbor's template names
    (``${PREBUILT_IMAGE_NAME}``), and a ``build`` section there has Harbor's ``up`` build ``tests/Dockerfile`` under
    the declared name, always with ``pull_policy: build`` (see :data:`VERIFIER_MAIN_KEYS`). This server starts the
    separate verifier as a single sandbox from the effective verifier environment of ``task.toml`` and the verifier
    image alone, and reads the file for the build section and pinned ``image`` of a ``main`` it builds only, so such
    a task would grade differently from Harbor without a word. It is refused instead, like one with verifier
    sidecars: by :func:`load_task` and, before any build, by ``ImageBuilder.build_many``.
    """
    main = next(
        (service for name, service in _verifier_compose_services(task_folder).items() if str(name) == MAIN_SERVICE),
        None,
    )
    if not isinstance(main, dict):
        return []
    allowed = PREBUILT_VERIFIER_MAIN_KEYS if prebuilt else VERIFIER_MAIN_KEYS
    return [
        str(key)
        for key, value in main.items()
        if str(key) not in allowed
        or (prebuilt and str(key) == "pull_policy" and str(value).strip().lower() == BUILD_PULL_POLICY)
    ]


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
