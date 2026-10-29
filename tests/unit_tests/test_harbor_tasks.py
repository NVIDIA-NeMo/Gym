# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import re
import shutil
import subprocess
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from omegaconf import OmegaConf

import nemo_gym.tasks.harbor.build_options as build_options
import nemo_gym.tasks.harbor.hub as hub_module
import nemo_gym.tasks.harbor.manifest as manifest_module
import nemo_gym.tasks.harbor.materialize as materialize_module
import nemo_gym.tasks.harbor.prepare_builds as prepare_builds
from nemo_gym.task_data import TaskDataValidator, load_task_data_schema
from nemo_gym.tasks.harbor import DIGEST_KEY, HarborTaskConfig, content_hash, discover_tasks, load_task
from nemo_gym.tasks.harbor.cli import (
    AgentSelection,
    PreparedTaskset,
    build_run,
    prepare_target,
    resolve_agent,
    run_images,
    runs_in_sandbox,
)
from nemo_gym.tasks.harbor.digest import context_digest
from nemo_gym.tasks.harbor.dockerfile import (
    DEFAULT_PATH,
    RECORD_VERSION,
    BaseImageConfig,
    DockerfileNeedsBuild,
    ImageConfigRequired,
    OverlayImage,
    base_image_only,
    has_heredoc,
    overlay_image,
    parse_dockerfile,
    resolve_dockerfile,
    split_words,
)
from nemo_gym.tasks.harbor.errors import HarborTaskError
from nemo_gym.tasks.harbor.hub import (
    HubError,
    HubRef,
    RegistryDataset,
    RegistryTask,
    fetch_dataset,
    load_registry,
    resolve_dataset,
    validate_git_url,
    validate_task_name,
)
from nemo_gym.tasks.harbor.materialize import TASK_SOURCE_KEY, dataset_entry, materialize_task, run_config, write_rows
from nemo_gym.tasks.harbor.models import HarborTaskConfig
from nemo_gym.tasks.harbor.package_store import PackageStoreError
from nemo_gym.tasks.harbor.records import BUILT_IMAGES_FILE, IMAGE_CONFIGS_FILE
from nemo_gym.tasks.harbor.task import task_digest


HELLO_TOML = """
schema_version = "1.4"

[task]
name = "harbor/hello-world"

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 120.0

[environment]
cpus = 1
memory_mb = 2048
storage_mb = 10240
gpus = 0
mcp_servers = []
"""


def write_task(root: Path, *, dockerfile: str = "FROM ubuntu:24.04\n\nWORKDIR /app", toml: str = HELLO_TOML) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "task.toml").write_text(toml)
    (root / "instruction.md").write_text("Create a file called hello.txt.\n")
    (root / "environment").mkdir(exist_ok=True)
    (root / "environment" / "Dockerfile").write_text(dockerfile)
    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    (root / "solution").mkdir(exist_ok=True)
    (root / "solution" / "solve.sh").write_text("#!/bin/bash\necho hi > hello.txt\n")
    return root


IMAGE_PATH = "/usr/local/bin:/usr/bin:/bin"


def write_image_config(parent: Path, image: str, **config) -> None:
    """Record ``image``'s OCI configuration next to the task folders, as prepare does."""
    path = parent / IMAGE_CONFIGS_FILE
    recorded = json.loads(path.read_text()) if path.is_file() else {}
    recorded[image] = {
        "os": "linux",
        "architecture": "amd64",
        "image": image,
        "config": config,
        "record_version": RECORD_VERSION,
    }
    path.write_text(json.dumps(recorded))


def dockerfile(text: str, **config) -> OverlayImage:
    """Parse and resolve ``text`` against a base image whose configuration is ``config`` (``Env`` etc.)."""
    record = {"record_version": RECORD_VERSION, "config": config}
    return resolve_dockerfile(parse_dockerfile(text), BaseImageConfig.from_record(record))


class TestTaskConfig:
    def test_reads_harbor_field_names(self):
        config = HarborTaskConfig.model_validate(
            {
                "schema_version": "1.4",
                "environment": {"docker_image": "img:1", "cpus": 2, "memory_mb": 1024},
                "verifier": {"timeout_sec": 30},
            }
        )
        assert config.environment.docker_image == "img:1"
        assert config.environment.memory_mb == 1024
        assert config.verifier.timeout_sec == 30
        assert config.is_shared_verifier

    def test_deprecated_fields_are_read_but_normalized(self):
        config = HarborTaskConfig.model_validate(
            {
                "version": "1.2",
                "environment": {"docker_image": "img:2", "memory": "2G", "storage": "512M", "allow_internet": False},
            }
        )
        assert config.schema_version == "1.2"
        assert config.environment.docker_image == "img:2"
        assert config.environment.memory_mb == 2048
        assert config.environment.storage_mb == 512
        assert config.environment.network_mode == "no-network"
        assert "image" not in config.environment.model_dump()

    def test_conflicting_deprecated_and_current_values_are_rejected(self):
        with pytest.raises(ValueError, match="Conflicting"):
            HarborTaskConfig.model_validate({"environment": {"memory": "1G", "memory_mb": 512}})

    def test_steps_are_parsed_with_harbor_fallbacks(self):
        config = HarborTaskConfig.model_validate(
            {
                "agent": {"timeout_sec": 600},
                "verifier": {"timeout_sec": 120},
                "multi_step_reward_strategy": "final",
                "steps": [
                    {"name": "plan", "agent": {"timeout_sec": 30}, "min_reward": 0.5},
                    {"name": "build", "verifier": {"timeout_sec": 900, "env": {"K": "v"}}},
                ],
            }
        )
        plan, build = config.steps
        assert config.is_multi_turn and config.multi_step_reward_strategy == "final"
        # The agent's timeout falls back to the task's (Harbor's AgentConfig.timeout_sec is None by default); the
        # verifier's does not: Harbor's step verifier is a VerifierConfig with its own 600 s default, so the task's
        # 120 s never reaches `plan`.
        assert (config.step_agent_timeout_sec(plan), config.step_verifier_timeout_sec(plan)) == (30, 600)
        assert (config.step_agent_timeout_sec(build), config.step_verifier_timeout_sec(build)) == (600, 900)
        assert plan.min_reward == 0.5 and build.verifier.env == {"K": "v"}
        assert HarborTaskConfig.model_validate({}).multi_step_reward_strategy == "mean"

    def test_legacy_multi_step_reward_mode_is_read_as_harbor_reads_it(self):
        """Harbor 0.22.0 has no `multi_step_reward_mode` field: `TaskConfig` drops the key as unknown and a multi-step
        task rolls up with its default, `mean`. The mirror accepts the old spelling and drops it to the same outcome;
        what it adds is that the key is known, so the loader does not report it."""
        steps = [{"name": "a"}, {"name": "b"}]
        legacy = HarborTaskConfig.model_validate(
            {"version": "1.0", "multi_step_reward_mode": "aggregate", "steps": steps}
        )
        assert (legacy.schema_version, legacy.multi_step_reward_strategy) == ("1.0", "mean")
        # Known, so the loader does not report it as an unknown key.
        assert HarborTaskConfig.unknown_keys({"version": "1.0", "multi_step_reward_mode": "aggregate"}) == []
        # The current key beside it wins, as under Harbor, which never reads the legacy one.
        both = HarborTaskConfig.model_validate(
            {"multi_step_reward_mode": "aggregate", "multi_step_reward_strategy": "final", "steps": steps}
        )
        assert both.multi_step_reward_strategy == "final"
        # A value Harbor would drop is dropped here too, never read as a strategy.
        dropped = HarborTaskConfig.model_validate({"multi_step_reward_mode": "final", "steps": steps})
        assert dropped.multi_step_reward_strategy == "mean"

    @pytest.mark.parametrize("steps", [[{"name": "a"}, {"name": "a"}], [{"name": "../up"}]])
    def test_step_names_are_unique_portable_folders(self, steps):
        with pytest.raises(ValueError, match="Step name"):
            HarborTaskConfig.model_validate({"steps": steps})

    def test_docker_image_is_optional(self):
        assert HarborTaskConfig.model_validate({}).environment.docker_image is None

    def test_unknown_keys_are_ignored_and_listed(self):
        data = {
            "allowlist": ["x"],
            "version": "1.3",
            "environment": {
                "dockerfile": "x",
                "image": "img",
                "mcp_servers": [{"name": "t", "url": "http://x", "extra": 1}],
                "healthcheck": {"command": "true", "retries_sec": 1},
            },
            "verifier": {"environment": {"docker_image": "v", "nope": 1}},
        }
        config = HarborTaskConfig.model_validate(data)
        assert config.environment.docker_image is None
        assert "dockerfile" not in config.environment.model_dump()
        assert HarborTaskConfig.unknown_keys(data) == [
            "allowlist",
            "environment.dockerfile",
            "environment.image",
            "environment.mcp_servers[0].extra",
            "environment.healthcheck.retries_sec",
            "verifier.environment.nope",
        ]

    def test_healthcheck_at_harbor_lower_bounds_loads(self):
        # The ``[environment.healthcheck]`` block published by billxbf/mimo-v26-* (hub content hash
        # 00a4f9b1136d…, 564 sampled tasks, six datasets); only the base64 payload inside ``command`` is
        # elided. Harbor's HealthcheckConfig takes plain numbers, so ``retries = 0`` must load.
        block = """
[environment.healthcheck]
# one-time setup before the agent (environment/setup/setup.sh)
command = "bash -c 'test -f /var/lib/mimo/ready || { mkdir -p /var/lib/mimo && echo H4sI... | base64 -d | tar xz -C /var/lib/mimo && bash /var/lib/mimo/setup.sh; }'"
timeout_sec = 1200.0
retries = 0
interval_sec = 5.0
"""
        config = HarborTaskConfig.model_validate(tomllib.loads(block))
        check = config.environment.healthcheck
        assert check.retries == 0 and check.timeout_sec == 1200.0 and check.interval_sec == 5.0
        assert (check.start_period_sec, check.start_interval_sec) == (0.0, 5.0)
        assert check.command.startswith("bash -c 'test -f /var/lib/mimo/ready")

    def test_mcp_server_needs_command_or_url(self):
        with pytest.raises(ValueError, match="stdio needs"):
            HarborTaskConfig.model_validate({"environment": {"mcp_servers": [{"name": "t", "transport": "stdio"}]}})
        config = HarborTaskConfig.model_validate(
            {"environment": {"mcp_servers": [{"name": "t", "transport": "http", "url": "http://x"}]}}
        )
        assert config.environment.mcp_servers[0].transport == "streamable-http"


class TestDockerfileDetection:
    def test_base_image_with_settings(self):
        base = base_image_only(
            "# SPDX header\n\nFROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS app\n"
            'WORKDIR /workspace\nENV A=1 B="two words"\nENV LEGACY value here\nUSER runner\nLABEL x=y\n'
        )
        assert base is not None
        assert base.image == "ghcr.io/astral-sh/uv:python3.13-bookworm-slim"
        assert base.workdir == "/workspace"
        assert base.env == {"A": "1", "B": "two words", "LEGACY": "value here"}
        assert base.user == "runner"

    def test_continuations_and_comments_are_ignored(self):
        base = base_image_only("FROM ubuntu:24.04 \\\n   # inside\n\n AS base\n\nWORKDIR \\\n /app\n")
        assert base is not None and base.image == "ubuntu:24.04" and base.workdir == "/app"

    def test_a_continuation_joins_the_next_line_verbatim(self):
        # Docker removes only the backslash and the newline: no separator is inserted, blanks are kept.
        parsed = parse_dockerfile("FROM img\nRUN echo ab\\\ncd \\   \n    && echo e\\\n\tf\n")
        assert parsed.instructions == (("RUN", "echo abcd     && echo e\tf"),)

    def test_env_values_resolve_as_docker_does(self):
        image = dockerfile(
            "FROM img\n"
            "ENV PATH=/opt/a/bin:$PATH\n"
            'ENV PATH="/opt/b/bin:${PATH}" LIT=\'$PATH\' ESC="\\$HOME/x" UNQ=\\$NOPE\n'
            'ENV MIX="a\\"b" EMPTY= TAIL=${MISSING:-dflt} OPT=${PATH:+set}\n'
            "ENV LEGACY two words $LIT\n",
            Env=[f"PATH={IMAGE_PATH}", "HOME=/root"],
        )
        assert image.env == {
            "PATH": f"/opt/b/bin:/opt/a/bin:{IMAGE_PATH}",
            "LIT": "$PATH",
            "ESC": "$HOME/x",
            "UNQ": "$NOPE",
            "MIX": 'a"b',
            "EMPTY": "",
            "TAIL": "dflt",
            "OPT": "set",
            "LEGACY": "two words $PATH",
        }
        # Only the single-quoted and escaped values keep a `$`, and theirs is a literal character, not a reference.
        assert [key for key, value in image.env.items() if "$" in value] == ["LIT", "ESC", "UNQ", "LEGACY"]

    def test_pairs_of_one_env_line_see_the_environment_before_the_line(self):
        # Docker's reference example: `ENV abc=bye def=$abc` gives def the previous abc; the next line sees the new one.
        image = dockerfile("FROM img\nENV abc=bye def=$abc\nENV ghi=$abc\nRUN env\n", Env=["abc=hello"])
        assert image.env == {"abc": "bye", "def": "hello", "ghi": "bye"}
        assert image.runs[0].env["def"] == "hello"

    def test_legacy_env_splits_on_any_run_of_blanks(self):
        # A tab (or several blanks) between the key and the value is what docker build accepts; the value keeps
        # its interior blanks.
        image = dockerfile("FROM img\nENV APP_HOME\t/srv/app\nWORKDIR $APP_HOME\n", Env=[f"PATH={IMAGE_PATH}"])
        assert image.env == {"APP_HOME": "/srv/app"} and image.workdir == "/srv/app"
        assert dockerfile("FROM img\nENV K  \t  v1  v2\n").env == {"K": "v1  v2"}
        with pytest.raises(DockerfileNeedsBuild, match="ENV ONLYKEY must have two arguments"):
            parse_dockerfile("FROM img\nENV ONLYKEY\n")

    def test_env_modifier_defaults_expand(self):
        image = dockerfile(
            "FROM img\nENV A=${B:-${PATH}} C=${B:-$PATH} D=${PATH:+${PATH}:/x} E=${B-${PATH}} F=${B:-${C:-${PATH}}}\n",
            Env=["PATH=/bin"],
        )
        assert image.env == {"A": "/bin", "C": "/bin", "D": "/bin:/x", "E": "/bin", "F": "/bin"}
        assert not [key for key, value in image.env.items() if "$" in value]

    def test_unterminated_modifier_needs_build(self):
        with pytest.raises(DockerfileNeedsBuild, match="unsupported variable expansion"):
            dockerfile("FROM img\nENV A=${B:-${PATH}\n", Env=["PATH=/bin"])

    def test_from_platform_flag_needs_build(self):
        # Dropping the flag would run the registry's linux/amd64 pick for an image the file asked for on arm64.
        text = "FROM --platform=linux/arm64 ubuntu\nRUN echo hi\n"
        with pytest.raises(DockerfileNeedsBuild, match=r"FROM --platform \(BuildKit flag\)"):
            parse_dockerfile(text)
        assert overlay_image(text) is None and base_image_only(text) is None
        with pytest.raises(DockerfileNeedsBuild, match="FROM --platform"):
            parse_dockerfile("FROM --platform=$BUILDPLATFORM --no-cache ubuntu\n")
        # The stage name after the image still parses.
        assert parse_dockerfile("FROM ubuntu AS base\n").image == "ubuntu"

    def test_escape_directive_needs_build(self):
        # With a backtick escape, the backslash is not a continuation: dropping the directive would read the
        # two RUN lines as the single command `echo foo RUN echo bar`.
        directive = "# escape=`\nFROM ubuntu\nRUN echo foo \\\nRUN echo bar\n"
        with pytest.raises(DockerfileNeedsBuild, match="escape"):
            parse_dockerfile(directive)
        assert overlay_image(directive) is None
        for leading in ("# syntax=docker/dockerfile:1\n\n  #  Escape = `\n", "\n# ESCAPE=`\n# a comment\n"):
            with pytest.raises(DockerfileNeedsBuild, match="escape"):
                parse_dockerfile(leading + "FROM ubuntu\n")
        # After the first instruction it is an ordinary comment (Docker reads directives only at the top), so the
        # backslash still joins the lines as it does for any comment inside a continuation.
        parsed = parse_dockerfile("FROM ubuntu\n# escape=`\nRUN echo foo \\\nRUN echo bar\n")
        assert parsed.instructions == (("RUN", "echo foo RUN echo bar"),)
        assert parse_dockerfile("# syntax=docker/dockerfile:1\n# a comment\nFROM ubuntu\n").image == "ubuntu"

    def test_image_without_path_gets_dockers_default(self):
        assert dockerfile("FROM scratch\nENV PATH=/x:$PATH\n").env == {"PATH": f"/x:{DEFAULT_PATH}"}

    def test_workdir_and_user_expand_variables_and_start_from_the_image(self):
        image = dockerfile(
            "FROM img\nENV APP=/srv/app USR=svc\nWORKDIR $APP\nWORKDIR src/\nUSER ${USR}:$USR\n",
            WorkingDir="/home/svc",
            User="svc",
        )
        assert image.workdir == "/srv/app/src" and image.user == "svc:svc"
        assert dockerfile("FROM img\nWORKDIR rel\n", WorkingDir="/home/svc").workdir == "/home/svc/rel"
        inherited = dockerfile("FROM img\n", WorkingDir="/home/svc", User="svc")
        assert inherited.workdir == "/home/svc" and inherited.user == "svc"
        bare = dockerfile("FROM img\n")
        assert bare.workdir == "/" and bare.user is None

    def test_without_the_image_configuration_only_self_contained_files_resolve(self):
        # Nothing refers to the base image: the result does not depend on it.
        base = base_image_only("FROM img\nENV A=1\nENV B=$A\nWORKDIR /app\nUSER me\n")
        assert base is not None and base.env == {"A": "1", "B": "1"} and base.workdir == "/app"
        for text, fragment in [
            ("FROM img\nENV PATH=/x:$PATH\n", "`$PATH` refers to the base image"),
            # `$A` on the same line as `A=1` is the base image's A, not the new one.
            ("FROM img\nENV A=1 B=$A\n", "`$A` refers to the base image"),
            ("FROM img\nWORKDIR src\n", "WORKDIR src is relative to the base image"),
            ("FROM img\nRUN true\n", "RUN lines run with the base image"),
        ]:
            with pytest.raises(ImageConfigRequired, match=re.escape(fragment)):
                resolve_dockerfile(parse_dockerfile(text))

    def test_split_words_keeps_quotes_and_escapes(self):
        assert split_words("A=1  B=\"two words\" C='x y' D=a\\ b") == ["A=1", 'B="two words"', "C='x y'", "D=a\\ b"]

    @pytest.mark.parametrize(
        ("command", "heredoc"),
        [
            ("cat <<EOF", True),
            ("python3 <<-'PY'", True),
            ('cat <<"EOF" > /x', True),
            ("echo $((1<<2))", False),
            ("cat <<< hello", False),
            ("echo '<<EOF'", False),
            ('echo "a <<EOF b"', False),
            ("bash -lc 'cat <<EOF'", False),
            ("echo it's <<EOF", True),
        ],
    )
    def test_heredoc_detection(self, command, heredoc):
        assert has_heredoc(command) is heredoc

    @pytest.mark.parametrize(
        "text",
        [
            "FROM ubuntu\nRUN apt-get update",
            "FROM ubuntu\nCOPY . /app",
            "FROM a AS one\nFROM b",
            "ARG BASE=ubuntu\nFROM $BASE",
            "WORKDIR /app\nFROM ubuntu",
            "",
        ],
    )
    def test_anything_else_needs_a_build(self, text):
        assert base_image_only(text) is None

    def test_emptied_entrypoint_is_accepted_in_pull_mode_and_recorded_as_cleared(self):
        base = base_image_only("FROM ubuntu:24.04\nENTRYPOINT []\nCMD [ ]\nWORKDIR /app\n")
        assert base is not None and base.image == "ubuntu:24.04" and base.workdir == "/app"
        assert base.entrypoint_cleared is True
        assert base_image_only("FROM ubuntu:24.04\nCMD [ ]\n").entrypoint_cleared is False
        assert base_image_only('FROM ubuntu\nENTRYPOINT ["sleep", "infinity"]') is None
        assert base_image_only('FROM ubuntu\nCMD ["bash"]') is None


TB21_DEBIAN_DOCKERFILE = """\
FROM debian:bookworm-slim

# Terminal-Bench style: apt setup, then the task's own files
RUN apt-get update && apt-get install -y --no-install-recommends \\
    curl \\
    tmux \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PATH="/opt/tools/bin:$PATH" \\
    DEBIAN_FRONTEND=noninteractive
RUN mkdir -p /opt/tools/bin && echo ready > /app/ready.txt
ENTRYPOINT []
"""


class TestDigest:
    def test_stable_and_sensitive_to_covered_files(self, tmp_path):
        task = write_task(tmp_path / "t")
        first = content_hash(task)
        assert first == content_hash(task)
        (task / "README.md").write_text("ignored? no, covered")
        second = content_hash(task)
        assert second != first
        (task / "tests" / "__pycache__").mkdir()
        (task / "tests" / "__pycache__" / "x.pyc").write_bytes(b"x")
        (task / "notes.txt").write_text("not covered")
        assert content_hash(task) == second

    def test_archive_hash_is_the_published_hash_without_exclusions(self, tmp_path):
        import hashlib

        from nemo_gym.tasks.harbor.digest import archive_hash

        task = write_task(tmp_path / "t")
        (task / "notes.txt").write_text("not covered by either digest")
        # Harbor's record format: sorted `relative NUL sha256 LF` over the covered files, exactly what its
        # packager hashes and ships, so an archive without excluded names hashes the same both ways.
        records = {
            "environment/Dockerfile",
            "instruction.md",
            "solution/solve.sh",
            "task.toml",
            "tests/test.sh",
        }
        published = hashlib.sha256()
        for relative in sorted(records):
            file_hash = hashlib.sha256((task / relative).read_bytes()).hexdigest()
            published.update(f"{relative}\0{file_hash}\n".encode())
        assert archive_hash(task) == published.hexdigest() == content_hash(task)

        # A byte-code cache is noise to the folder digest but a file to the archive digest: a published
        # archive holds exactly the files the packager kept, so every file in it counts.
        (task / "tests" / "__pycache__").mkdir()
        (task / "tests" / "__pycache__" / "x.pyc").write_bytes(b"x")
        assert content_hash(task) == published.hexdigest()
        assert archive_hash(task) != published.hexdigest()
        pyc_hash = hashlib.sha256(b"x").hexdigest()
        published.update(f"tests/__pycache__/x.pyc\0{pyc_hash}\n".encode())
        assert archive_hash(task) != published.hexdigest()  # records stay sorted: __pycache__ sorts before test.sh
        with_pyc = hashlib.sha256()
        for relative in sorted(records | {"tests/__pycache__/x.pyc"}):
            file_hash = hashlib.sha256((task / relative).read_bytes()).hexdigest()
            with_pyc.update(f"{relative}\0{file_hash}\n".encode())
        assert archive_hash(task) == with_pyc.hexdigest()

    @pytest.mark.parametrize("linked_path", ["tests/data", "tests", "tests/__pycache__/data"])
    @pytest.mark.parametrize("external", [False, True])
    def test_directory_symlinks_are_refused(self, tmp_path, linked_path, external):
        task = write_task(tmp_path / "t")
        target = (tmp_path if external else task) / "payload"
        link = task / linked_path
        if link.exists():
            link.rename(target)
        else:
            target.mkdir()
            link.parent.mkdir(parents=True, exist_ok=True)
        (target / "data.txt").write_text("test input")
        link.symlink_to(target, target_is_directory=True)

        message = "escapes the task folder" if external else "directory symlinks are not supported"
        with pytest.raises(HarborTaskError, match=message):
            load_task(task)

    def test_retargeted_directory_link_cannot_reuse_materialized_digest(self, tmp_path):
        task = write_task(tmp_path / "t")
        for name in ("one", "two"):
            (task / "tests" / name).mkdir()
            (task / "tests" / name / "data.txt").write_text(name)
        link = task / "tests" / "data"
        for name in ("one", "two"):
            link.symlink_to(name, target_is_directory=True)
            # Neither target may produce a valid row with an untracked directory link.
            with pytest.raises(HarborTaskError, match="directory symlinks are not supported"):
                load_task(task)
            link.unlink()

    def test_internal_file_symlink_keeps_the_file_content_digest(self, tmp_path):
        task = write_task(tmp_path / "t")
        source = task / "tests" / "test.sh"
        link = task / "tests" / "linked-test.sh"
        link.write_bytes(source.read_bytes())
        expected = content_hash(task)
        link.unlink()
        link.symlink_to("test.sh")
        assert load_task(task).digest == expected
        source.write_text("changed test input")
        assert load_task(task).digest != expected


class TestLoadTask:
    def test_hello_world_shape(self, tmp_path):
        task = load_task(write_task(tmp_path / "hello-world"))
        assert task.task_id == "hello-world"
        assert task.image == "ubuntu:24.04"
        assert task.workdir == "/app"
        assert task.user is None
        assert task.overlay == ()
        assert task.needs_sandbox and task.has_solution
        assert task.instruction == "Create a file called hello.txt.\n"
        assert task.digest == content_hash(task.path)

    def test_task_toml_wins_over_dockerfile(self, tmp_path):
        toml = HELLO_TOML + '\n[environment.env]\nA = "toml"\n'
        toml = toml.replace("gpus = 0", 'gpus = 0\nworkdir = "/work"')
        task = load_task(
            write_task(tmp_path / "t", dockerfile="FROM img\nENV A=docker B=keep\nWORKDIR /app", toml=toml)
        )
        assert task.workdir == "/work"
        assert task.env == {"A": "toml", "B": "keep"}

    def test_agent_user_wins_over_dockerfile_user(self, tmp_path):
        toml = HELLO_TOML.replace(
            "timeout_sec = 120.0\n\n[environment]", 'timeout_sec = 120.0\nuser = "agent"\n\n[environment]'
        )
        task = load_task(write_task(tmp_path / "t", dockerfile="FROM img\nUSER other", toml=toml))
        assert task.user == "agent"

    def test_dockerfile_needing_a_build_without_image_is_rejected(self, tmp_path):
        with pytest.raises(HarborTaskError, match=r"needs a build \(COPY\)"):
            load_task(write_task(tmp_path / "t", dockerfile="FROM ubuntu\nRUN true\nCOPY . /app"))
        with pytest.raises(HarborTaskError, match=r"needs a build \(RUN with a heredoc"):
            load_task(write_task(tmp_path / "t", dockerfile="FROM ubuntu\nRUN <<EOF\ntrue\nEOF"))

    def test_overlay_dockerfile_records_the_run_lines(self, tmp_path):
        toml = HELLO_TOML + '\n[environment.env]\nDEBIAN_FRONTEND = "teletype"\n'
        folder = write_task(tmp_path / "t", dockerfile=TB21_DEBIAN_DOCKERFILE, toml=toml)
        write_image_config(tmp_path, "debian:bookworm-slim", Env=[f"PATH={IMAGE_PATH}"])
        task = load_task(folder)
        assert task.image == "debian:bookworm-slim" and task.workdir == "/app" and task.user is None
        # task.toml wins for the sandbox env; the Dockerfile's `$PATH` is the image's, resolved, not a live `$`.
        assert task.env == {"PATH": f"/opt/tools/bin:{IMAGE_PATH}", "DEBIAN_FRONTEND": "teletype"}
        assert [run.command[:10] for run in task.overlay] == ["apt-get up", "mkdir -p /"]
        assert task.overlay[1].env["DEBIAN_FRONTEND"] == "noninteractive"
        assert task.needs_sandbox

    def test_a_resolved_dockerfile_starts_the_pinned_base_image_under_the_written_key(self, tmp_path):
        """The sandbox starts the immutable reference the base configuration was recorded at (the index digest
        behind the tag), not the tag the Dockerfile wrote; the record stays keyed by that tag."""
        pinned = "library/debian@sha256:" + "d" * 64
        overlay = write_task(tmp_path / "overlay", dockerfile=TB21_DEBIAN_DOCKERFILE)
        pull = write_task(tmp_path / "pull", dockerfile="FROM debian:bookworm-slim\nENV A=$PATH\nWORKDIR /w")
        write_image_config(tmp_path, "debian:bookworm-slim", Env=[f"PATH={IMAGE_PATH}"])
        path = tmp_path / IMAGE_CONFIGS_FILE
        recorded = json.loads(path.read_text())
        recorded["debian:bookworm-slim"]["image"] = pinned
        path.write_text(json.dumps(recorded))
        task = load_task(overlay)
        assert task.image == pinned and task.needs_sandbox and len(task.overlay) == 2
        task = load_task(pull)
        assert task.image == pinned and task.env == {"A": IMAGE_PATH} and task.workdir == "/w"
        assert list(json.loads(path.read_text())) == ["debian:bookworm-slim"]
        # A declared image starts as written, as Harbor starts it; the pull-mode Dockerfile beside it lends its settings.
        toml = HELLO_TOML.replace("gpus = 0", 'gpus = 0\ndocker_image = "org/app:1"')
        declared = write_task(tmp_path / "declared", dockerfile="FROM debian:bookworm-slim\nWORKDIR /w", toml=toml)
        task = load_task(declared)
        assert task.image == "org/app:1" and task.workdir == "/w"

    def test_overlay_dockerfile_without_the_recorded_base_image_is_rejected(self, tmp_path):
        folder = write_task(tmp_path / "t", dockerfile=TB21_DEBIAN_DOCKERFILE)
        with pytest.raises(HarborTaskError) as excinfo:
            load_task(folder)
        message = str(excinfo.value)
        assert "RUN lines run with the base image" in message
        assert "'debian:bookworm-slim' is not recorded" in message and IMAGE_CONFIGS_FILE in message
        assert "Prepare the dataset again" in message
        # A pull-mode Dockerfile that refers to the image environment needs it too; one that does not, does not.
        (folder / "environment" / "Dockerfile").write_text("FROM img\nENV PATH=/x:$PATH\n")
        with pytest.raises(HarborTaskError, match=r"`\$PATH` refers to the base image"):
            load_task(folder)
        (folder / "environment" / "Dockerfile").write_text("FROM img\nENV A=1\nWORKDIR /app\n")
        assert load_task(folder).env == {"A": "1"}

    def test_recorded_base_image_fills_in_workdir_and_user(self, tmp_path):
        folder = write_task(tmp_path / "t", dockerfile="FROM img\nENV PATH=/x:$PATH\n")
        write_image_config(tmp_path, "img", Env=["PATH=/bin"], WorkingDir="/srv", User="svc")
        task = load_task(folder)
        assert (task.workdir, task.user, task.env) == ("/srv", "svc", {"PATH": "/x:/bin"})

    def test_sandbox_user_is_the_recorded_base_image_user(self, tmp_path):
        # The sandbox starts from the base image, so its default user is the image's, whatever the Dockerfile
        # or [agent].user make the task's user.
        folder = write_task(tmp_path / "t", dockerfile="FROM img\nUSER root\nRUN true\n")
        write_image_config(tmp_path, "img", Env=["PATH=/bin"], User="jovyan")
        task = load_task(folder)
        assert task.user == "root" and task.sandbox_user == "jovyan"
        toml = HELLO_TOML.replace(
            "timeout_sec = 120.0\n\n[environment]", 'timeout_sec = 120.0\nuser = "agent"\n\n[environment]'
        )
        task = load_task(write_task(tmp_path / "t", dockerfile="FROM img\nRUN true\n", toml=toml))
        assert task.user == "agent" and task.sandbox_user == "jovyan"
        # An image that records no User runs as root.
        write_image_config(tmp_path, "rootimg", Env=["PATH=/bin"])
        task = load_task(write_task(tmp_path / "t", dockerfile="FROM rootimg\nUSER agent\nRUN true\n"))
        assert task.user == "agent" and task.sandbox_user is None
        # Without a recorded base image (a self-contained pull-mode Dockerfile, a prebuilt image), the task's
        # user stands in.
        task = load_task(write_task(tmp_path / "t", dockerfile="FROM unknown\nUSER me\n"))
        assert task.user == "me" and task.sandbox_user == "me"
        toml = HELLO_TOML.replace("cpus = 1", 'docker_image = "org/task:1"\ncpus = 1')
        task = load_task(write_task(tmp_path / "t", dockerfile="FROM unknown\nRUN true\nUSER built\n", toml=toml))
        assert task.user is None and task.sandbox_user is None
        toml = toml.replace(
            "timeout_sec = 120.0\n\n[environment]", 'timeout_sec = 120.0\nuser = "agent"\n\n[environment]'
        )
        task = load_task(write_task(tmp_path / "t", dockerfile="FROM unknown\nRUN true\n", toml=toml))
        assert task.user == "agent" and task.sandbox_user == "agent"

    def test_a_declared_image_takes_its_default_user_from_the_dockerfiles_final_user(self, tmp_path):
        toml = HELLO_TOML.replace("cpus = 1", 'docker_image = "org/app:1"\ncpus = 1')
        task = load_task(write_task(tmp_path / "t", dockerfile="FROM python:3.12\nUSER app\n", toml=toml))
        # The declared image is the one that runs; its default user is the Dockerfile's final USER.
        assert task.image == "org/app:1" and task.user == "app" and task.sandbox_user == "app"
        # The FROM image's record expands ENV and WORKDIR but does not name the started image's user.
        write_image_config(tmp_path, "python:3.12", Env=["PATH=/bin"], WorkingDir="/srv", User="root")
        task = load_task(write_task(tmp_path / "t", dockerfile="FROM python:3.12\nENV P=$PATH\nUSER app\n", toml=toml))
        # The declared image keeps its own working directory (its runtime record, /app say, is the provider's
        # to report): the FROM image's /srv is not what org/app:1 starts in.
        assert (task.workdir, task.env, task.sandbox_user) == (None, {"P": "/bin"}, "app")
        task = load_task(
            write_task(tmp_path / "t", dockerfile="FROM python:3.12\nWORKDIR /work\nUSER app\n", toml=toml)
        )
        assert task.workdir == "/work"  # a WORKDIR the Dockerfile sets is the declared image's too
        task = load_task(write_task(tmp_path / "t", dockerfile="FROM python:3.12\nUSER app\n"))
        assert task.workdir == "/srv"  # without a declared image the FROM record's directory is the task's
        # Without a USER line the declared image runs as the FROM image's user.
        write_image_config(tmp_path, "python:3.12", Env=["PATH=/bin"], User="svc")
        task = load_task(write_task(tmp_path / "t", dockerfile="FROM python:3.12\nENV P=$PATH\n", toml=toml))
        assert task.sandbox_user == "svc"

    def test_a_declared_image_runs_as_is_when_its_dockerfile_needs_a_build_at_resolution_time(self, tmp_path):
        toml = HELLO_TOML.replace("cpus = 1", 'docker_image = "org/app:1"\ncpus = 1')
        write_image_config(tmp_path, "python:3.12", Env=["PATH=/bin"], WorkingDir="/srv")
        dockerfile_text = "FROM python:3.12\nENV X=${PATH%:/bin}\n"
        task = load_task(write_task(tmp_path / "t", dockerfile=dockerfile_text, toml=toml))
        # The same fallback as a parse-time build requirement: the prebuilt image with task.toml's settings.
        assert task.image == "org/app:1" and task.env == {} and task.workdir is None and task.overlay == ()
        assert task.sandbox_user is None
        with pytest.raises(HarborTaskError, match=r"needs a build \(unsupported variable expansion"):
            load_task(write_task(tmp_path / "t2", dockerfile=dockerfile_text))

    def test_a_recorded_base_image_from_an_older_gym_is_refused_with_a_prepare_hint(self, tmp_path):
        folder = write_task(tmp_path / "t", dockerfile="FROM img\nRUN true\n")
        legacy = {"img": {"os": "linux", "architecture": "amd64", "image": "img", "config": {"User": "svc"}}}
        (tmp_path / IMAGE_CONFIGS_FILE).write_text(json.dumps(legacy))
        with pytest.raises(HarborTaskError, match="prepare the dataset again") as excinfo:
            load_task(folder)
        assert IMAGE_CONFIGS_FILE in str(excinfo.value) and "'img'" in str(excinfo.value)

    def test_prebuilt_image_allows_a_real_dockerfile(self, tmp_path):
        toml = HELLO_TOML.replace("cpus = 1", 'docker_image = "org/task:1"\ncpus = 1')
        task = load_task(write_task(tmp_path / "t", dockerfile="FROM ubuntu\nRUN true\nWORKDIR /built", toml=toml))
        # The prebuilt image is the build's result: nothing from the Dockerfile is re-applied.
        assert task.image == "org/task:1" and task.overlay == () and task.workdir is None
        task = load_task(write_task(tmp_path / "t", dockerfile="FROM ubuntu\nCOPY . /app", toml=toml))
        assert task.image == "org/task:1" and task.overlay == ()

    def test_missing_pieces_are_reported(self, tmp_path):
        with pytest.raises(HarborTaskError, match="no task.toml"):
            load_task(tmp_path)
        task = write_task(tmp_path / "t")
        (task / "instruction.md").unlink()
        with pytest.raises(HarborTaskError, match="no instruction.md"):
            load_task(task)

    def test_undecodable_files_are_task_errors(self, tmp_path):
        task = write_task(tmp_path / "t")
        (task / "task.toml").write_bytes(b"\xff\xfe not utf-8")
        with pytest.raises(HarborTaskError, match="task.toml"):
            load_task(task)
        task = write_task(tmp_path / "u")
        (task / "instruction.md").write_bytes(b"\xff\xfe")
        with pytest.raises(HarborTaskError, match="instruction.md"):
            load_task(task)

    def test_unknown_task_toml_key_is_warned_once_per_key(self, tmp_path, caplog):
        toml = HELLO_TOML.replace('schema_version = "1.4"', 'schema_version = "1.4"\nallowlist = ["pypi.org"]')
        toml = toml.replace("gpus = 0", "gpus = 0\nnew_harbor_key = true")
        with caplog.at_level(logging.WARNING, logger="nemo_gym.tasks.harbor.task"):
            task = load_task(write_task(tmp_path / "t", toml=toml))
        assert task.config.environment.gpus == 0
        messages = [record.getMessage() for record in caplog.records]
        assert len(messages) == 2
        assert all(str(task.path / "task.toml") in message for message in messages)
        assert any("`allowlist`" in message for message in messages)
        assert any("`environment.new_harbor_key`" in message for message in messages)

    def test_built_image_record_gives_the_entrypoint_its_user_workdir_and_healthcheck(self, tmp_path):
        from nemo_gym.tasks.harbor.digest import context_digest
        from nemo_gym.tasks.harbor.records import BUILT_IMAGES_FILE

        folder = write_task(tmp_path / "t", dockerfile='FROM ubuntu\nCOPY . /app\nENTRYPOINT ["/entrypoint.sh"]\n')
        built = "registry.example/ds-t:abc"
        (tmp_path / BUILT_IMAGES_FILE).write_text(
            json.dumps({context_digest(folder / "environment"): {"image": built}})
        )
        health = {"Test": ["CMD-SHELL", "test -f /tmp/healthy"], "Interval": 10_000_000, "Retries": 3}
        write_image_config(
            tmp_path,
            built,
            Env=["PATH=/bin"],
            Entrypoint=["/entrypoint.sh"],
            User="app",
            WorkingDir="/srv",
            Healthcheck=health,
            Shell=["/bin/bash", "-o", "pipefail", "-c"],
        )
        toml = HELLO_TOML.replace("gpus = 0", 'gpus = 0\nworkdir = "/work"').replace(
            "timeout_sec = 120.0\n\n[environment]", 'timeout_sec = 120.0\nuser = "agent"\n\n[environment]'
        )
        task = load_task(
            write_task(tmp_path / "t", dockerfile=(folder / "environment" / "Dockerfile").read_text(), toml=toml)
        )
        assert task.image == built and task.entrypoint == ("/entrypoint.sh",)
        # The entrypoint starts as the image records, whatever task.toml makes the task's user and workdir.
        assert (task.user, task.workdir) == ("agent", "/work")
        assert (task.entrypoint_user, task.entrypoint_workdir) == ("app", "/srv")
        assert task.image_healthcheck == health
        # The image's SHELL comes with them: what Docker runs a CMD-SHELL healthcheck through.
        assert task.image_shell == ("/bin/bash", "-o", "pipefail", "-c")

    def test_built_image_without_a_record_or_entrypoint_has_none(self, tmp_path):
        from nemo_gym.tasks.harbor.digest import context_digest
        from nemo_gym.tasks.harbor.records import BUILT_IMAGES_FILE

        folder = write_task(tmp_path / "t", dockerfile="FROM ubuntu\nCOPY . /app\n")
        built = "registry.example/ds-t:abc"
        (tmp_path / BUILT_IMAGES_FILE).write_text(
            json.dumps({context_digest(folder / "environment"): {"image": built}})
        )
        task = load_task(folder)
        assert task.image == built and task.entrypoint is None and task.image_healthcheck is None
        assert task.image_shell is None  # nothing recorded, so no shell is known either
        write_image_config(tmp_path, built, Env=["PATH=/bin"], Entrypoint=[], User="app")
        task = load_task(folder)
        assert task.user == "app" and task.entrypoint is None
        # The image's user and WorkingDir are known without an entrypoint: its HEALTHCHECK runs as them too,
        # through Docker's default shell when the image records no SHELL.
        assert (task.entrypoint_user, task.entrypoint_workdir) == ("app", None)
        assert task.image_shell == ("/bin/sh", "-c")

    def test_pull_and_overlay_tasks_take_the_base_entrypoint_unless_cleared(self, tmp_path):
        bash = ["/bin/bash", "-o", "pipefail", "-c"]
        write_image_config(
            tmp_path, "img", Env=["PATH=/bin"], Entrypoint=["/start.sh"], WorkingDir="/srv", User="svc", Shell=bash
        )
        # Pull mode: the base's entrypoint, started with the Dockerfile's final USER and WORKDIR, and the base's
        # shell (a SHELL instruction of the Dockerfile itself would need a build).
        task = load_task(write_task(tmp_path / "t", dockerfile="FROM img\nWORKDIR /work\nUSER other\n"))
        assert task.entrypoint == ("/start.sh",)
        assert (task.entrypoint_user, task.entrypoint_workdir) == ("other", "/work")
        assert task.image_shell == tuple(bash)
        # [agent].user overrides the task's user only; the entrypoint keeps the image's.
        toml = HELLO_TOML.replace(
            "timeout_sec = 120.0\n\n[environment]", 'timeout_sec = 120.0\nuser = "agent"\n\n[environment]'
        )
        task = load_task(write_task(tmp_path / "t", dockerfile="FROM img\nRUN true\n", toml=toml))
        assert task.user == "agent" and task.overlay
        assert (task.entrypoint, task.entrypoint_user, task.entrypoint_workdir) == (("/start.sh",), "svc", "/srv")
        assert task.image_shell == tuple(bash)
        # ENTRYPOINT [] clears it.
        task = load_task(write_task(tmp_path / "t", dockerfile="FROM img\nRUN true\nENTRYPOINT []\n"))
        assert task.entrypoint is None and (task.entrypoint_user, task.entrypoint_workdir) == ("svc", "/srv")
        # Without a recorded base image, or for a prebuilt image, nothing is known.
        unknown = load_task(write_task(tmp_path / "t", dockerfile="FROM unknown\nWORKDIR /app\n"))
        assert unknown.entrypoint is None and unknown.image_shell is None
        toml = HELLO_TOML.replace("cpus = 1", 'docker_image = "org/task:1"\ncpus = 1')
        task = load_task(write_task(tmp_path / "t", dockerfile="FROM img\nRUN true\n", toml=toml))
        assert task.image == "org/task:1" and task.entrypoint is None and task.image_shell is None


class TestDiscovery:
    def test_single_task_folder(self, tmp_path):
        assert [t.task_id for t in discover_tasks(write_task(tmp_path / "solo"))] == ["solo"]

    def test_folder_of_task_folders_is_searched_recursively(self, tmp_path):
        write_task(tmp_path / "ds" / "b")
        write_task(tmp_path / "ds" / "a")
        write_task(tmp_path / "ds" / "nested" / "deep")
        write_task(tmp_path / "ds" / ".hidden" / "never")
        (tmp_path / "ds" / "README.md").write_text("dataset readme")
        tasks = discover_tasks(tmp_path / "ds")
        assert [t.task_id for t in tasks] == ["a", "b", "deep"]
        # A nested task reads the dataset's own files from the root, not from its parent folder.
        (tmp_path / "ds" / "dataset.toml").write_text("[gym.defaults]\ntimeout_multiplier = 2.0\n")
        deep = discover_tasks(tmp_path / "ds")[2]
        assert deep.path == (tmp_path / "ds" / "nested" / "deep").resolve()
        assert deep.config.agent.timeout_sec == 240.0

    def test_duplicate_task_ids_across_folders_are_rejected(self, tmp_path):
        from nemo_gym.tasks.harbor.build_plan import plan_builds
        from nemo_gym.tasks.harbor.image_configs import dockerfile_base_images
        from nemo_gym.tasks.harbor.task import iter_task_folders

        write_task(tmp_path / "ds" / "a")
        write_task(tmp_path / "ds" / "nested" / "a")
        # The folder walk itself rejects them, so every scan keyed by task name fails before doing work.
        for scan in (discover_tasks, iter_task_folders, plan_builds, dockerfile_base_images):
            with pytest.raises(HarborTaskError, match="Duplicate Harbor task ID 'a'"):
                scan(tmp_path / "ds")

    def test_empty_folder_is_an_error(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(HarborTaskError, match="neither a task folder"):
            discover_tasks(tmp_path / "empty")

    def test_unknown_key_in_one_task_does_not_block_the_dataset(self, tmp_path, caplog):
        write_task(tmp_path / "ds" / "a")
        write_task(tmp_path / "ds" / "b", toml=HELLO_TOML + "\n[environment.allowlist]\nhosts = []\n")
        write_task(tmp_path / "ds" / "c")
        with caplog.at_level(logging.WARNING, logger="nemo_gym.tasks.harbor.task"):
            tasks = discover_tasks(tmp_path / "ds")
        assert [t.task_id for t in tasks] == ["a", "b", "c"]
        assert [r.getMessage() for r in caplog.records] == [
            f"{tmp_path.resolve() / 'ds' / 'b' / 'task.toml'}: ignoring unknown key `environment.allowlist`"
        ]

    def test_malformed_task_is_skipped_by_name(self, tmp_path):
        write_task(tmp_path / "ds" / "a")
        write_task(tmp_path / "ds" / "broken", toml="[task\nname = oops")
        write_task(tmp_path / "ds" / "c")
        with pytest.raises(HarborTaskError, match="broken"):
            discover_tasks(tmp_path / "ds")
        skipped: dict[str, HarborTaskError] = {}
        tasks = discover_tasks(tmp_path / "ds", skipped=skipped)
        assert [t.task_id for t in tasks] == ["a", "c"]
        assert list(skipped) == ["broken"]
        assert "broken/task.toml" in str(skipped["broken"])

    def test_all_tasks_failing_is_still_an_error(self, tmp_path):
        write_task(tmp_path / "ds" / "x", toml="not toml =")
        with pytest.raises(HarborTaskError, match="No task under"):
            discover_tasks(tmp_path / "ds", skipped={})

    def test_discovery_reads_the_shared_image_configurations_once(self, tmp_path, monkeypatch):
        import nemo_gym.tasks.harbor.task as task_module

        for name in ("a", "b", "c"):
            write_task(tmp_path / name, dockerfile="FROM img\nRUN true\n")
        write_image_config(tmp_path, "img", Env=["PATH=/bin"])
        reads: list[Path] = []
        original = task_module.read_image_configs

        def counting(folder):
            reads.append(Path(folder))
            return original(folder)

        monkeypatch.setattr(task_module, "read_image_configs", counting)
        tasks = discover_tasks(tmp_path)
        assert [task.task_id for task in tasks] == ["a", "b", "c"] and all(task.overlay for task in tasks)
        assert reads == [tmp_path.resolve()]
        # A single task loads its parent's file itself.
        assert load_task(tmp_path / "a").overlay and reads == [tmp_path.resolve(), tmp_path.resolve()]


class TestMaterialize:
    def test_row_shape(self, tmp_path):
        task = load_task(write_task(tmp_path / "hello-world"))
        row = materialize_task(task, "hello").model_dump(mode="json", exclude_unset=True)
        assert row["task_id"] == {"taskset": "hello", "task_id": "hello-world"}
        params = row["task_input"]["responses_create_params"]
        assert params["input"][0]["role"] == "user"
        assert params["input"][0]["content"] == "Create a file called hello.txt.\n"
        assert params["metadata"] == {"agent_timeout_sec": "120.0"}
        assert row["task_input"]["task_data"] == {DIGEST_KEY: task.digest}

    def test_a_flat_single_turn_input_is_the_canonical_row(self, tmp_path):
        """Review (CI bot, round 34): `HarborTaskInput` replaced `SingleAgentTurnTaskInput`, whose before-validator
        folded flat task fields into `task_data`, with a model that forbade them, so a single-turn `/run` request
        shaped `{"responses_create_params": ..., "_ng_digest": ...}` was rejected before execution. The flat shape is
        normalized again, with both request-envelope fields reserved; conflicting values and a malformed container
        are still refused (pydantic's ValidationError is a ValueError), and a generated multi-step row carries no
        `responses_create_params` key at all."""
        task = load_task(write_task(tmp_path / "hello-world"))
        canonical = materialize_task(task, "hello").task_input
        params = canonical.responses_create_params.model_dump(mode="json", exclude_unset=True)
        validate = materialize_module.HarborTaskInput.model_validate
        flat = validate({"responses_create_params": params, DIGEST_KEY: task.digest})
        assert flat == canonical and flat.model_fields_set == {"responses_create_params", "task_data"}
        assert flat.model_dump(mode="json", exclude_unset=True) == canonical.model_dump(
            mode="json", exclude_unset=True
        )
        with pytest.raises(ValueError, match="Conflicting task field '_ng_digest'"):
            validate({"responses_create_params": params, "task_data": {DIGEST_KEY: "other"}, DIGEST_KEY: task.digest})
        with pytest.raises(ValueError, match="task_data"):
            validate({"responses_create_params": params, "task_data": "not a mapping"})
        with pytest.raises(ValueError, match="either responses_create_params or turns"):
            validate({DIGEST_KEY: task.digest})
        with pytest.raises(ValueError, match="either responses_create_params or turns"):
            validate(
                {
                    "responses_create_params": params,
                    "turns": [{"name": "a", "responses_create_params": params}],
                    DIGEST_KEY: task.digest,
                }
            )
        multi = materialize_task(load_task(write_multi_step_task(tmp_path / "t")), "ds")
        assert multi.task_input.model_fields_set == {"task_data", "turns"}
        assert "responses_create_params" not in multi.model_dump(mode="json", exclude_unset=True)["task_input"]

    def test_write_rows_and_run_config(self, tmp_path):
        folder = tmp_path / "ds"
        tasks = [load_task(write_task(folder / "a")), load_task(write_task(folder / "b"))]
        rows_path = tmp_path / "out" / "tasks.jsonl"
        rows = write_rows(tasks, "ds", rows_path)
        written = [json.loads(line) for line in rows_path.read_text().splitlines()]
        assert [row["task_id"]["task_id"] for row in written] == ["a", "b"]
        assert len(rows) == 2
        # Every row names its dataset (the taskset) under the key rollout collection copies onto results rows,
        # beside the task folder digest.
        assert all(row[TASK_SOURCE_KEY] == "ds" for row in written)
        assert [row["task_input"]["task_data"][DIGEST_KEY] for row in written] == [tasks[0].digest, tasks[1].digest]

        config = run_config(
            taskset="ds",
            folder=folder,
            tasks=tasks,
            rows_path=rows_path,
            agent_instance_name="hermes_agent",
            agent_impl="hermes_agent",
            agent_overrides={"enabled_toolsets": ["terminal"]},
        )
        assert config["environment_routing_mode"] == "taskset"
        assert config["environment_server_routes"] == {"ds": "harbor_ds_environment"}
        assert config["harbor_ds_environment"]["_inherit_from"] == "harbor_environment_server"
        environment = config["harbor_ds_environment"]["environment_servers"]["harbor"]
        assert environment["tasksets"]["ds"]["folder"] == str(folder.resolve())
        assert environment["tasksets"]["ds"]["tasks"] == {"a": tasks[0].digest, "b": tasks[1].digest}
        assert environment["agent_server"] == {"type": "responses_api_agents", "name": "harbor_ds_agent"}
        # Verifier logs land beside the rollouts, not under the server's own folder.
        assert environment["artifacts_dir"] == str((tmp_path / "out" / "verifier").resolve())
        agent = config["harbor_ds_agent"]
        assert agent["_inherit_from"] == "hermes_agent"
        # Rollout collection finds datasets on agent and resources blocks, so the rows are declared on the agent.
        entry = agent["responses_api_agents"]["hermes_agent"]["datasets"][0]
        assert entry["jsonl_fpath"] == str(rows_path.resolve())
        # Without a dataset.toml the rows are a benchmark: no license, not verified, one rollout per task, and
        # no prepare script (the rows were just written). The entry is what Gym's config types accept.
        assert entry == {"name": "ds", "type": "benchmark", "verified": False, "jsonl_fpath": str(rows_path.resolve())}
        from nemo_gym.config_types import BenchmarkDatasetConfig

        assert BenchmarkDatasetConfig.model_validate(entry).prepare_script is None
        assert "resources_server" not in agent["responses_api_agents"]["hermes_agent"]
        assert agent["responses_api_agents"]["hermes_agent"]["enabled_toolsets"] == ["terminal"]
        assert "resources_servers" not in " ".join(config)


MULTI_STEP_TOML = (
    'multi_step_reward_strategy = "final"\n'
    + HELLO_TOML
    + """
[[steps]]
name = "01_plan"
[steps.agent]
timeout_sec = 2400.0

[[steps]]
name = "02_implement"
"""
)


def write_multi_step_task(root: Path, *, hidden_spec: str = "The full spec only the simulated user may read.") -> Path:
    write_task(root, toml=MULTI_STEP_TOML)
    (root / "instruction.md").write_text(hidden_spec)
    for name, instruction in (("01_plan", "Agree on a plan with the maintainer."), ("02_implement", "Implement it.")):
        (root / "steps" / name / "tests").mkdir(parents=True)
        (root / "steps" / name / "instruction.md").write_text(instruction)
        (root / "steps" / name / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    return root


# The shape of one task of the memory team's MemoryArenaBundledShopping dataset (round 24): its task.toml, Dockerfile
# and Compose file as the adapter writes them (with a placeholder sidecar image), and per step a one-line
# instruction.md and stub solve.sh / test.sh in place of the dataset's own scripts, which the loader never reads.
MEMORY_ARENA_TASK = Path(__file__).parent / "fixtures" / "harbor_memory_arena_bundled_shopping" / "0000-baking-item-0"
SLOPCODEBENCH_TASK = Path(__file__).parent / "fixtures" / "harbor_slopcodebench" / "code_search"
SLOPCODEBENCH_STEPS = ["scaffold", "lexer", "index", "query", "report"]
MEMORY_ARENA_STEPS = [
    "01-select-cake-base",
    "02-select-frosting",
    "03-select-coloring-base",
    "04-select-sprinkles",
    "05-select-theme-topper",
    "06-select-candles",
]


class TestMultiStepTasks:
    def test_memory_arena_bundled_shopping_loads_as_harbor_reads_it(self, tmp_path, caplog):
        """The MemoryArenaBundledShopping shape: legacy `version = "1.0"` and `multi_step_reward_mode = "aggregate"`,
        a streamable-http MCP sidecar declared in a Compose file, six `[[steps]]` with their own agent and verifier
        budgets, each with instruction.md, solution/solve.sh and tests/test.sh, and no top-level instruction, tests or
        solution. Harbor 0.22.0 reads it as schema 1.0 with the `mean` roll-up, without a word about the old keys."""
        dataset = tmp_path / "MemoryArenaBundledShopping"
        shutil.copytree(MEMORY_ARENA_TASK, dataset / MEMORY_ARENA_TASK.name)
        # The overlay-mode Dockerfile (RUN lines on python:3.12-slim) needs the base image's recorded configuration.
        write_image_config(dataset, "python:3.12-slim", Env=[f"PATH={IMAGE_PATH}"], WorkingDir="")
        with caplog.at_level(logging.WARNING, logger="nemo_gym.tasks.harbor.task"):
            task = load_task(dataset / MEMORY_ARENA_TASK.name)
        assert "unknown key" not in caplog.text and "legacy key" not in caplog.text

        config = task.config
        assert (config.schema_version, config.multi_step_reward_strategy) == ("1.0", "mean")
        assert config.is_multi_turn and config.is_shared_verifier
        assert [turn.name for turn in task.turns] == MEMORY_ARENA_STEPS
        budgets = {
            (config.step_agent_timeout_sec(t.config), config.step_verifier_timeout_sec(t.config)) for t in task.turns
        }
        assert budgets == {(1800.0, 1800.0)}
        assert all(turn.instruction.startswith("# MemoryArena Bundled Shopping") for turn in task.turns)
        assert all((task.path / turn.tests_dir / "test.sh").is_file() for turn in task.turns)
        assert all((task.path / "steps" / turn.name / "solution" / "solve.sh").is_file() for turn in task.turns)
        # No top-level instruction, tests or solution: the steps carry them, so the task is validatable through its
        # steps' own solutions, each resolved per turn as Harbor's oracle resolves them (`turn_solution_dir`).
        assert task.instruction == "" and not (task.path / "tests").exists() and not (task.path / "solution").exists()
        assert task.has_solution
        assert [task.turn_solution_dir(turn) for turn in task.turns] == [
            task.path / "steps" / name / "solution" for name in MEMORY_ARENA_STEPS
        ]
        [mcp] = config.environment.mcp_servers
        assert (mcp.name, mcp.transport, mcp.url) == ("webshop", "streamable-http", "http://mcp-server:8000/mcp")
        env = config.environment
        assert (env.cpus, env.memory_mb, env.storage_mb, env.gpus) == (4, 16384, 65536, 0)
        assert (env.network_mode, env.build_timeout_sec) == ("public", 7200.0)  # allow_internet = true
        assert task.needs_compose and task.image == "python:3.12-slim" and task.workdir == "/app"
        assert len(task.overlay) == 1 and task.user is None
        # The rows carry one input per step, in order.
        row = materialize_task(task, "ds").model_dump(mode="json", exclude_unset=True)
        assert [turn["name"] for turn in row["task_input"]["turns"]] == MEMORY_ARENA_STEPS

    def test_a_legacy_reward_mode_harbor_would_drop_is_dropped_with_a_warning(self, tmp_path, caplog):
        root = write_multi_step_task(tmp_path / "t")
        toml = MULTI_STEP_TOML.replace('multi_step_reward_strategy = "final"\n', 'multi_step_reward_mode = "final"\n')
        (root / "task.toml").write_text(toml)
        with caplog.at_level(logging.WARNING, logger="nemo_gym.tasks.harbor.task"):
            task = load_task(root)
        assert task.config.multi_step_reward_strategy == "mean"
        messages = [record.getMessage() for record in caplog.records]
        assert [m for m in messages if "ignoring legacy key `multi_step_reward_mode = 'final'`" in m]
        assert not [m for m in messages if "unknown key" in m]

    def test_steps_load_as_turns_with_their_own_instructions(self, tmp_path):
        task = load_task(write_multi_step_task(tmp_path / "t"))
        assert [(turn.name, turn.instruction) for turn in task.turns] == [
            ("01_plan", "Agree on a plan with the maintainer."),
            ("02_implement", "Implement it."),
        ]
        assert task.config.multi_step_reward_strategy == "final"

    def test_rows_carry_one_input_per_turn_and_never_the_top_level_instruction(self, tmp_path):
        hidden = "The full spec only the simulated user may read."
        task = load_task(write_multi_step_task(tmp_path / "t", hidden_spec=hidden))
        row = materialize_task(task, "ds").model_dump(mode="json", exclude_unset=True)
        assert hidden not in json.dumps(row)
        assert "responses_create_params" not in row["task_input"]
        turns = row["task_input"]["turns"]
        assert [turn["name"] for turn in turns] == ["01_plan", "02_implement"]
        assert [turn["responses_create_params"]["metadata"] for turn in turns] == [
            {"agent_timeout_sec": "2400.0", "harbor_step": "01_plan"},
            # A step without its own budget falls back to the task's [agent].timeout_sec.
            {"agent_timeout_sec": "120.0", "harbor_step": "02_implement"},
        ]

    def test_a_multi_step_rows_turns_are_not_task_fields(self, tmp_path):
        """Review (round 32): the row's `turns` sit beside `task_data`, and the validator merged that sibling into
        the task-owned fields, so every multi-step row reported `turns` as an unknown key: a false warning at
        collation, or a rejected row under `task_data_validation: error`. Through the real Harbor TaskData schema."""
        task = load_task(write_multi_step_task(tmp_path / "t"))
        row = materialize_task(task, "ds").model_dump(mode="json", exclude_unset=True)
        assert [turn["name"] for turn in row["task_input"]["turns"]] == ["01_plan", "02_implement"]
        harbor_dir = Path(__file__).resolve().parents[2] / "environment_servers" / "harbor"
        validator = TaskDataValidator("harbor", load_task_data_schema(harbor_dir), "rows.jsonl")
        validator.validate_row(0, row)
        assert validator.report.clean, validator.report.summary()

    def test_top_level_instruction_and_tests_are_optional_for_steps(self, tmp_path):
        root = write_multi_step_task(tmp_path / "t")
        (root / "instruction.md").unlink()
        (root / "tests" / "test.sh").unlink()
        (root / "tests").rmdir()
        assert [turn.name for turn in load_task(root).turns] == ["01_plan", "02_implement"]

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            (lambda root: (root / "steps" / "02_implement" / "instruction.md").unlink(), "instruction.md"),
        ],
    )
    def test_unsupported_or_incomplete_steps_are_refused_by_name(self, tmp_path, change, message):
        root = write_multi_step_task(tmp_path / "t")
        change(root)
        with pytest.raises(HarborTaskError, match=message):
            load_task(root)

    def test_a_steps_workdir_and_setup_script_load_for_the_server_to_apply_per_turn(self, tmp_path, caplog):
        """Harbor 0.22.0 uploads `steps/<name>/workdir/` into the working directory and runs
        `steps/<name>/workdir/setup.sh` before the step's agent (`_prepare_step`); the loader used to refuse both.
        They load now, named per turn for the server, and a `setup.sh` beside instruction.md (not a Harbor file:
        Harbor never runs it) only warns, naming the path Harbor would run."""
        root = write_multi_step_task(tmp_path / "t")
        workdir = root / "steps" / "01_plan" / "workdir"
        workdir.mkdir()
        (workdir / "setup.sh").write_text("#!/bin/bash\nmake fixtures\n")
        (workdir / "fixture.txt").write_text("x\n")
        (root / "steps" / "02_implement" / "setup.sh").write_text("true\n")
        with caplog.at_level(logging.WARNING, logger="nemo_gym.tasks.harbor.task"):
            task = load_task(root)
        plan, implement = task.turns
        assert plan.workdir_dir == Path("steps/01_plan/workdir") and plan.setup_script == plan.workdir_dir / "setup.sh"
        assert (root / plan.setup_script).is_file() and not (root / implement.workdir_dir).exists()
        assert "steps/02_implement/setup.sh is not run; Harbor runs steps/<name>/workdir/setup.sh" in caplog.text
        assert "01_plan/setup.sh" not in caplog.text and "01_plan/workdir" not in caplog.text
        # The step's files are part of the task's identity, as everything under steps/ is.
        before = task.folder_digest
        (workdir / "fixture.txt").write_text("y\n")
        assert load_task(root).folder_digest != before

    def test_a_setup_script_run_as_root_after_a_less_privileged_step_is_warned_about(self, tmp_path, caplog):
        """`steps/<name>/workdir/setup.sh` runs as the step's agent user in the working directory an earlier step's
        agent owned, and nothing stops what that agent left running: a step that takes root (an explicit root, or
        the image's own user) after a step that had less is where a lingering process can replace the script before
        it runs. The loader warns, naming both steps; a task whose later step keeps the same user, or whose root
        step ships no setup.sh, is not warned about."""
        root = write_multi_step_task(tmp_path / "t")
        (root / "task.toml").write_text(
            MULTI_STEP_TOML.replace("timeout_sec = 2400.0\n", 'timeout_sec = 2400.0\nuser = "tester"\n')
        )
        workdir = root / "steps" / "02_implement" / "workdir"
        workdir.mkdir()
        (workdir / "setup.sh").write_text("#!/bin/bash\nrm -rf /app/cheats\n")
        with caplog.at_level(logging.WARNING, logger="nemo_gym.tasks.harbor.task"):
            load_task(root)
        assert (
            "steps/02_implement/workdir/setup.sh runs as the image's own user (root unless the image sets USER) after "
            "step '01_plan' ran as 'tester'"
        ) in caplog.text

        caplog.clear()
        (root / "task.toml").write_text(
            MULTI_STEP_TOML.replace("timeout_sec = 2400.0\n", 'timeout_sec = 2400.0\nuser = "tester"\n').replace(
                'name = "02_implement"\n', 'name = "02_implement"\n[steps.agent]\nuser = "0"\n'
            )
        )
        with caplog.at_level(logging.WARNING, logger="nemo_gym.tasks.harbor.task"):
            load_task(root)
        assert "steps/02_implement/workdir/setup.sh runs as root after step '01_plan' ran as 'tester'" in caplog.text

        caplog.clear()
        (root / "task.toml").write_text(
            MULTI_STEP_TOML.replace(
                "[agent]\ntimeout_sec = 120.0\n", '[agent]\ntimeout_sec = 120.0\nuser = "tester"\n'
            )
        )
        with caplog.at_level(logging.WARNING, logger="nemo_gym.tasks.harbor.task"):
            load_task(root)
        assert "setup.sh runs as" not in caplog.text
        # A root step without the script has no window to warn about.
        (root / "task.toml").write_text(
            MULTI_STEP_TOML.replace("timeout_sec = 2400.0\n", 'timeout_sec = 2400.0\nuser = "tester"\n')
        )
        shutil.rmtree(workdir)
        with caplog.at_level(logging.WARNING, logger="nemo_gym.tasks.harbor.task"):
            load_task(root)
        assert "setup.sh runs as" not in caplog.text

    def test_a_steps_artifacts_load_for_the_server_to_collect_per_turn(self, tmp_path):
        """Harbor 0.22.0 collects a step's `artifacts` after the step's agent, with the task's (`_collect_step_artifacts`);
        the loader used to refuse them. They load with the task's entry rules: a source under the verifier's own
        folders is refused as it is for the task's, naming the step."""
        root = write_multi_step_task(tmp_path / "t")
        toml = (root / "task.toml").read_text()
        (root / "task.toml").write_text(
            toml + 'artifacts = [{ source = "/app", destination = "app", exclude = [".venv"] }, "/var/log/x"]\n'
        )
        plan, implement = load_task(root).turns
        assert plan.config.artifacts == []
        assert [(a.source, a.destination, a.exclude) for a in implement.config.artifacts] == [
            ("/app", "app", [".venv"]),
            ("/var/log/x", None, []),
        ]
        (root / "task.toml").write_text(toml + 'artifacts = ["/tests/helpers"]\n')
        with pytest.raises(HarborTaskError, match=r"steps\.1\.artifacts\.0[\s\S]*would replace the verifier's /tests"):
            load_task(root)

    def test_a_steps_own_collect_hooks_load_for_the_server_to_run_after_the_tasks(self, tmp_path, caplog):
        """Harbor's `StepConfig.verifier` is a full `VerifierConfig`: a step's `[[steps.verifier.collect]]` hooks are run
        after the task's in that step's collection pass (`_collect_hooks_for`). The table used to be dropped with an
        "unknown key" warning; it loads with the hook's fields now."""
        root = write_multi_step_task(tmp_path / "t")
        (root / "task.toml").write_text(
            (root / "task.toml").read_text()
            + '[[steps.verifier.collect]]\ncommand = "pg_dump app > /logs/artifacts/app.sql"\nservice = "db"\n'
            'timeout_sec = 45.0\nuser = "postgres"\n'
        )
        with caplog.at_level(logging.WARNING, logger="nemo_gym.tasks.harbor.task"):
            plan, implement = load_task(root).turns
        assert "unknown key" not in caplog.text
        assert plan.config.verifier.collect == []
        (hook,) = implement.config.verifier.collect
        assert (hook.command, hook.service, hook.timeout_sec, hook.user) == (
            "pg_dump app > /logs/artifacts/app.sql",
            "db",
            45.0,
            "postgres",
        )

    def test_slopcodebench_code_search_loads_as_harbor_reads_it(self, tmp_path):
        """The shape of the hub's `gabeorlanski/slopcodebench` (36 tasks, pulled 2026-10-22), as the synthetic fixture
        reproduces it with its own step names, budgets and exclude list: every `[[steps]]` entry collects `/app` under
        `app` with excludes and gates two reward keys, on a build-mode Dockerfile with a `SHELL` line. The base loader
        refused all 36 for the per-step artifacts."""
        dataset = tmp_path / "slopcodebench"
        shutil.copytree(SLOPCODEBENCH_TASK, dataset / "code_search")
        # The Dockerfile sets SHELL, so it is built at prepare (not overlaid at seed): the build's record is needed.
        with pytest.raises(HarborTaskError, match="needs a build .SHELL."):
            load_task(dataset / "code_search")
        built = "registry.example/slopcodebench-code-search:abc"
        (dataset / BUILT_IMAGES_FILE).write_text(
            json.dumps({context_digest(dataset / "code_search" / "environment"): {"image": built}})
        )
        write_image_config(dataset, built, Env=[f"PATH={IMAGE_PATH}"], WorkingDir="/app")
        task = load_task(dataset / "code_search")
        assert [turn.name for turn in task.turns] == SLOPCODEBENCH_STEPS
        assert task.config.is_shared_verifier and task.config.artifacts == []
        for turn in task.turns:
            (artifact,) = turn.config.artifacts
            assert (artifact.source, artifact.destination, artifact.service) == ("/app", "app", None)
            assert "node_modules" in artifact.exclude and ".venv" in artifact.exclude
            assert turn.config.min_reward == {"core_pass_rate": 0.5, "strict_pass_rate": 0.25}
            # No [steps.agent]: the task's [agent].timeout_sec; each step names its own [steps.verifier].timeout_sec.
            assert task.config.step_agent_timeout_sec(turn.config) == 7200.0
            assert task.config.step_verifier_timeout_sec(turn.config) == 300.0
        assert task.workdir == "/app" and task.overlay == () and task.image == built == task.built_image

    def test_a_step_without_its_own_verifier_timeout_gets_harbors_600_s_not_the_tasks(self, tmp_path, caplog):
        """Harbor 0.22.0 (and main): `StepConfig.verifier` is a full `VerifierConfig`, whose `timeout_sec` defaults
        to 600.0, so `_step_verifier_timeout_sec`'s fallback to the task's `[verifier].timeout_sec` never runs and a
        step that names none is graded within 600 s whatever the task says. The loader applies that and warns once,
        naming the steps the task's figure does not reach; a task whose figure is 600 or whose steps name their own
        is not warned about."""
        root = write_multi_step_task(tmp_path / "t")
        toml = (root / "task.toml").read_text()
        assert "[verifier]\ntimeout_sec = 120.0\n" in toml  # HELLO_TOML's task-level figure
        (root / "task.toml").write_text(toml + "[steps.verifier]\ntimeout_sec = 90.0\n")
        with caplog.at_level(logging.WARNING, logger="nemo_gym.tasks.harbor.task"):
            task = load_task(root)
        plan, implement = task.turns
        assert task.config.step_verifier_timeout_sec(plan.config) == 600.0
        assert task.config.step_verifier_timeout_sec(implement.config) == 90.0
        assert (
            "step(s) '01_plan' name no [steps.verifier].timeout_sec and are graded within Harbor's step default of "
            "600s, not the task's [verifier].timeout_sec = 120s"
        ) in caplog.text
        assert "'02_implement'" not in caplog.text

        caplog.clear()
        (root / "task.toml").write_text(toml.replace("[verifier]\ntimeout_sec = 120.0\n", "[verifier]\n"))
        with caplog.at_level(logging.WARNING, logger="nemo_gym.tasks.harbor.task"):
            task = load_task(root)
        assert [task.config.step_verifier_timeout_sec(t.config) for t in task.turns] == [600.0, 600.0]
        assert "Harbor's step default" not in caplog.text

    def test_a_dataset_multiplier_scales_the_step_default_a_step_inherits_and_the_hint_follows(self, tmp_path, caplog):
        """Review (round 32): a step without its own `[steps.verifier].timeout_sec` is graded within Harbor's 600 s
        step default times the dataset's `timeout_multiplier`, as Harbor's multi_step.py feeds the VerifierConfig
        default through `_resolve_timeout_sec`; the step keeps naming none, so the loader's hint still sees it, and
        the hint compares the task's scaled figure with the scaled default: it names the steps when the two differ
        and stays silent for a task whose figure is the default, instead of comparing the scaled figure with the
        unscaled constant (which fired for every multiplied dataset)."""
        root = write_multi_step_task(tmp_path / "t")
        (tmp_path / "dataset.toml").write_text("[gym.defaults]\ntimeout_multiplier = 2.0\n")
        with caplog.at_level(logging.WARNING, logger="nemo_gym.tasks.harbor.task"):
            task = load_task(root)
        assert task.config.verifier.timeout_sec == 240.0  # HELLO_TOML's 120 s, scaled
        assert [t.config.verifier.timeout_sec for t in task.turns] == [None, None]
        assert [task.config.step_verifier_timeout_sec(t.config) for t in task.turns] == [1200.0, 1200.0]
        assert (
            "step(s) '01_plan', '02_implement' name no [steps.verifier].timeout_sec and are graded within Harbor's "
            "step default of 1200s (600s times the dataset's timeout_multiplier), not the task's "
            "[verifier].timeout_sec = 240s"
        ) in caplog.text

        caplog.clear()
        toml = (root / "task.toml").read_text()
        (root / "task.toml").write_text(
            toml.replace("[verifier]\ntimeout_sec = 120.0\n", "[verifier]\ntimeout_sec = 600.0\n")
        )
        with caplog.at_level(logging.WARNING, logger="nemo_gym.tasks.harbor.task"):
            task = load_task(root)
        assert task.config.verifier.timeout_sec == 1200.0
        assert [task.config.step_verifier_timeout_sec(t.config) for t in task.turns] == [1200.0, 1200.0]
        assert "Harbor's step default" not in caplog.text

    def test_a_shared_step_with_its_own_verifier_environment_is_refused_as_harbor_refuses_it(self, tmp_path):
        """Harbor's `_validate_mode_env_consistency` is on `VerifierConfig`, and a step's verifier is one, so a step
        that says `environment_mode = "shared"` and declares a `[steps.verifier.environment]` fails Harbor's task.toml
        load. The loader refuses it the same way instead of running the step shared and ignoring the environment."""
        root = write_multi_step_task(tmp_path / "t")
        toml = (root / "task.toml").read_text()
        (root / "task.toml").write_text(
            toml
            + '[steps.verifier]\nenvironment_mode = "shared"\n'
            + '[steps.verifier.environment]\ndocker_image = "org/step-verifier:1"\n'
        )
        with pytest.raises(
            HarborTaskError,
            match=r"steps\.1\.verifier(?s:.*)environment_mode='shared' is incompatible with \[steps\.verifier\.environment\]",
        ):
            load_task(root)
        # Alone, either is fine: the mode without the environment is shared, the environment without the mode is separate.
        (root / "task.toml").write_text(toml + '[steps.verifier]\nenvironment_mode = "shared"\n')
        assert [turn.verifier_mode for turn in load_task(root).turns] == ["shared", "shared"]

    def test_each_steps_verifier_mode_and_image_resolve_as_harbor_resolves_them(self, tmp_path):
        """Harbor's resolve_step_verifier_mode: the step's explicit mode, else `separate` when it declares a
        [steps.verifier.environment], else the task's; the environment is the step's, else the task's
        [verifier.environment], else a copy of [environment] (resolve_effective_verifier_env_config). The loader used
        to refuse a separate step and a multi-step task with a separate task-level verifier."""
        root = write_multi_step_task(tmp_path / "t")
        toml = (root / "task.toml").read_text()
        # A shared task with one step in its own image: that step alone is separate, in that image.
        (root / "task.toml").write_text(toml + '[steps.verifier.environment]\ndocker_image = "org/step-verifier:1"\n')
        write_image_config(tmp_path, "org/step-verifier:1", Env=["PATH=/bin"], User="grader", WorkingDir="/grade")
        plan, implement = load_task(root).turns
        assert (plan.verifier_mode, plan.verifier_image) == ("shared", None)
        assert (implement.verifier_mode, implement.verifier_image) == ("separate", "org/step-verifier:1")
        assert implement.verifier_image_config is not None and implement.verifier_image_config.user == "grader"
        # A separate task (its own [verifier.environment]) whose first step opts back into shared mode.
        separate = toml.replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n').replace(
            "[agent]\n", '[verifier.environment]\ndocker_image = "org/verifier:1"\n\n[agent]\n'
        )
        (root / "task.toml").write_text(
            separate.replace('name = "01_plan"\n', 'name = "01_plan"\n[steps.verifier]\nenvironment_mode = "shared"\n')
        )
        task = load_task(root)
        plan, implement = task.turns
        assert not task.config.is_shared_verifier and task.verifier_image == "org/verifier:1"
        assert (plan.verifier_mode, plan.verifier_image) == ("shared", None)
        assert (implement.verifier_mode, implement.verifier_image) == ("separate", "org/verifier:1")
        assert implement.verifier_image_config is None  # nothing recorded for the declared image: nothing starts
        # A separate task whose effective environment is a copy of [environment] with a declared docker_image: the
        # turns run the task's own image (no verifier image of their own), as a single-step task does.
        task_image = toml.replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n').replace(
            "[environment]\n", '[environment]\ndocker_image = "ubuntu:24.04"\n'
        )
        (root / "task.toml").write_text(task_image)
        task = load_task(root)
        assert task.verifier_image is None and [turn.verifier_image for turn in task.turns] == [None, None]
        assert [turn.verifier_mode for turn in task.turns] == ["separate", "separate"]

    def test_a_separate_step_without_an_image_follows_harbors_build_context(self, tmp_path):
        """Harbor builds a step's separate verifier from steps/<name>/tests/Dockerfile when the step has a tests/ folder,
        else from tests/Dockerfile (_verifier_env_build_context with a step). Prepare builds tests/Dockerfile only, so a
        step that starts from its own folder is refused naming the image to declare; one that starts from tests/ reads
        the recorded build, and is planned by prepare whatever the task's own mode."""
        from nemo_gym.tasks.harbor.verifier_env import verifier_build_context, verifier_build_environment

        root = write_multi_step_task(tmp_path / "t")
        toml = (root / "task.toml").read_text() + '[steps.verifier]\nenvironment_mode = "separate"\n'
        (root / "task.toml").write_text(toml)
        with pytest.raises(HarborTaskError, match="steps/02_implement/tests/Dockerfile.*builds tests/Dockerfile only"):
            load_task(root)
        # Without a tests/ folder of its own the step starts from the task's tests/: a Dockerfile is required there...
        shutil.rmtree(root / "steps" / "02_implement" / "tests")
        with pytest.raises(HarborTaskError, match="tests/ has no Dockerfile"):
            load_task(root)
        # ...and then its recorded build, which prepare plans for the step (the task itself is shared).
        (root / "tests" / "Dockerfile").write_text("FROM python:3.12\nCOPY . /tests/\n")
        config = HarborTaskConfig.model_validate(tomllib.loads(toml))
        assert verifier_build_context(root, config) == root / "tests"
        assert verifier_build_environment(root, config) is not None and config.is_shared_verifier
        with pytest.raises(HarborTaskError, match="no built image is recorded"):
            load_task(root)
        built = "registry.example/t-verifier:abc"
        (tmp_path / BUILT_IMAGES_FILE).write_text(
            json.dumps({context_digest(root / "tests"): {"services": {"verifier": {"image": built}}}})
        )
        write_image_config(tmp_path, built, Env=["PATH=/bin"], WorkingDir="/tests")
        plan, implement = load_task(root).turns
        assert (plan.verifier_mode, plan.verifier_image) == ("shared", None)
        assert (implement.verifier_mode, implement.verifier_image) == ("separate", built)
        assert implement.verifier_image_config is not None and implement.verifier_image_config.workdir == "/tests"

    def test_a_steps_verifier_compose_file_with_runtime_settings_on_main_is_refused_like_the_tasks(self, tmp_path):
        """Harbor stacks the docker-compose.yaml of the folder a step's separate verifier starts from (the step's own
        tests/, else the task's: _verifier_env_build_context, then _docker_compose_paths) on that verifier's template,
        prebuilt image or not, so `main.environment` there reaches the step's verifier container; this server starts
        the step verifier as a single sandbox from task.toml and the image, so a test.sh reading CHECK_MODE would grade
        differently. The per-step path refuses such a file as load_task refuses the task verifier's (and a sidecar);
        the step verifier is prebuilt (a declared docker_image), so main's image is an override too (Harbor's prebuilt
        template names ${PREBUILT_IMAGE_NAME} and the task file comes last), as are its build section and a
        pull_policy of build (Harbor's prebuilt up builds tests/Dockerfile under the declared name), while a pull
        policy on the declared image alone still loads; the task's own verifier is shared here, so only the step's
        check sees the file."""
        root = write_multi_step_task(tmp_path / "t")
        toml = (root / "task.toml").read_text()
        (root / "task.toml").write_text(toml + '[steps.verifier.environment]\ndocker_image = "org/step-verifier:1"\n')
        write_image_config(tmp_path, "org/step-verifier:1", Env=["PATH=/bin"], User="grader", WorkingDir="/grade")
        step_tests = root / "steps" / "02_implement" / "tests"
        (step_tests / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    environment:\n      CHECK_MODE: strict\n"
            "    env_file: grader.env\n"
        )
        with pytest.raises(
            HarborTaskError,
            match=r"step '02_implement': .*steps/02_implement/tests/docker-compose.yaml sets image, environment, "
            r"env_file on main; Harbor applies the file to the step's separate verifier container",
        ):
            load_task(root)
        # Beside the declared step image, main's image (a literal would run in its place under Harbor, ${MAIN_IMAGE_NAME}
        # the hb__<tests hash> nobody built), its build section and the pull_policy naming that build are overrides
        # (Harbor's prebuilt up would build the folder's Dockerfile and grade in it); a pull policy alone is not.
        (step_tests / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    pull_policy: build\n"
            "    build:\n      context: .\n      args:\n        PYTEST: '8.4.1'\n"
        )
        with pytest.raises(
            HarborTaskError, match=r"step '02_implement': .*docker-compose.yaml sets image, pull_policy, build on main"
        ):
            load_task(root)
        (step_tests / "docker-compose.yaml").write_text(
            "services:\n  main:\n    pull_policy: build\n    build:\n      context: .\n      args:\n        PYTEST: '8.4.1'\n"
        )
        with pytest.raises(
            HarborTaskError, match=r"step '02_implement': .*docker-compose.yaml sets pull_policy, build on main"
        ):
            load_task(root)
        (step_tests / "docker-compose.yaml").write_text("services:\n  main:\n    pull_policy: missing\n")
        plan, implement = load_task(root).turns
        assert (plan.verifier_mode, plan.verifier_image) == ("shared", None)
        assert (implement.verifier_mode, implement.verifier_image) == ("separate", "org/step-verifier:1")
        # Without a tests/ folder of its own the step starts from the task's tests/, whose file Harbor stacks the
        # same way: refused naming the step and that file.
        shutil.rmtree(step_tests)
        (root / "tests" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    volumes:\n      - ./fixtures:/fixtures:ro\n"
        )
        with pytest.raises(
            HarborTaskError, match=r"step '02_implement': .*/t/tests/docker-compose.yaml sets volumes on main"
        ):
            load_task(root)
        (root / "tests" / "docker-compose.yaml").write_text("services:\n  main:\n    image: org/other:2\n")
        with pytest.raises(
            HarborTaskError, match=r"step '02_implement': .*/t/tests/docker-compose.yaml sets image on main"
        ):
            load_task(root)
        (root / "tests" / "docker-compose.yaml").write_text("services:\n  main:\n    pull_policy: build\n")
        with pytest.raises(
            HarborTaskError, match=r"step '02_implement': .*/t/tests/docker-compose.yaml sets pull_policy on main"
        ):
            load_task(root)
        (root / "tests" / "docker-compose.yaml").write_text("services:\n  main:\n    pull_policy: always\n")
        assert load_task(root).turns[1].verifier_image == "org/step-verifier:1"

    def test_a_steps_verifier_compose_file_with_an_include_or_invalid_yaml_is_refused_like_the_tasks(self, tmp_path):
        """The per-step checks read the file's own services, so a top-level ``include`` whose file sets
        ``main.environment`` and a ``build`` section passed them beside the declared step image while Compose merges
        it into Harbor's step verifier project; a file that is not valid YAML fails Harbor's ``compose up`` where the
        checks saw ``{}``; a top-level ``networks`` block reshapes the default network Harbor's step verifier main is
        attached to. Each is refused naming the step and the file, for the step's own tests/ and for the task's
        tests/ it falls back to; an empty file still loads."""
        root = write_multi_step_task(tmp_path / "t")
        toml = (root / "task.toml").read_text()
        (root / "task.toml").write_text(toml + '[steps.verifier.environment]\ndocker_image = "org/step-verifier:1"\n')
        write_image_config(tmp_path, "org/step-verifier:1", Env=["PATH=/bin"], User="grader", WorkingDir="/grade")
        step_tests = root / "steps" / "02_implement" / "tests"
        included = "services:\n  main:\n    build:\n      context: .\n    environment:\n      CHECK_MODE: strict\n"
        (step_tests / "other.yaml").write_text(included)
        (step_tests / "docker-compose.yaml").write_text("include:\n  - other.yaml\nservices: {}\n")
        with pytest.raises(
            HarborTaskError,
            match=r"step '02_implement': .*steps/02_implement/tests/docker-compose.yaml: include is refused for the "
            r"separate verifier: Compose merges",
        ):
            load_task(root)
        (step_tests / "docker-compose.yaml").write_text(
            "services:\n  main:\n    environment: [CHECK_MODE=strict\n  :bad\n"
        )
        with pytest.raises(
            HarborTaskError,
            match=r"step '02_implement': .*steps/02_implement/tests/docker-compose.yaml is not valid YAML: ",
        ):
            load_task(root)
        (step_tests / "docker-compose.yaml").write_text(
            "services:\n  main:\n    pull_policy: missing\nnetworks:\n  default:\n    internal: true\n"
        )
        with pytest.raises(
            HarborTaskError,
            match=r"step '02_implement': .*steps/02_implement/tests/docker-compose.yaml: top-level networks is refused "
            r"for the separate verifier: Harbor's compose up applies it to the verifier project",
        ):
            load_task(root)
        (step_tests / "docker-compose.yaml").write_text("")
        plan, implement = load_task(root).turns
        assert (plan.verifier_mode, plan.verifier_image) == ("shared", None)
        assert (implement.verifier_mode, implement.verifier_image) == ("separate", "org/step-verifier:1")
        # The task's tests/ the step falls back to without a folder of its own, checked the same way.
        shutil.rmtree(step_tests)
        (root / "tests" / "other.yaml").write_text(included)
        (root / "tests" / "docker-compose.yaml").write_text("include:\n  - other.yaml\nservices: {}\n")
        with pytest.raises(
            HarborTaskError, match=r"step '02_implement': .*/t/tests/docker-compose.yaml: include is refused"
        ):
            load_task(root)
        (root / "tests" / "docker-compose.yaml").write_text("- main\n")
        with pytest.raises(
            HarborTaskError, match=r"step '02_implement': .*/t/tests/docker-compose.yaml is not a mapping$"
        ):
            load_task(root)

    def test_the_pre_pull_manifest_names_a_steps_separate_verifier_image(self, tmp_path):
        """Review (M24 follow-up): `images.json` claims every image a run starts, but run_images read
        HarborTask.verifier_image only, which a shared task leaves None, while the server starts each separate step's
        own verifier image (`_turn_task`). A step image declared beside a shared task verifier is listed once, next to
        the task image; a step that shares the task's declared image adds nothing new, and shared steps add none."""
        folder = tmp_path / "ds"
        root = write_multi_step_task(folder / "t")
        toml = (root / "task.toml").read_text()
        (root / "task.toml").write_text(toml + '[steps.verifier.environment]\ndocker_image = "org/step-verifier:1"\n')
        write_image_config(folder, "ubuntu:24.04", Env=["PATH=/bin"], WorkingDir="/app")
        write_image_config(folder, "org/step-verifier:1", Env=["PATH=/bin"], User="grader", WorkingDir="/grade")
        task = load_task(root)
        assert [(turn.name, turn.verifier_mode, turn.verifier_image) for turn in task.turns] == [
            ("01_plan", "shared", None),
            ("02_implement", "separate", "org/step-verifier:1"),
        ]
        assert task.verifier_image is None  # the task's own verifier is shared
        assert run_images([task], {}) == ("org/step-verifier:1", "ubuntu:24.04")
        # Two separate steps on one declared task image, inherited by both steps: once, with the task's own entry.
        root2 = write_multi_step_task(folder / "u")
        (root2 / "task.toml").write_text(
            MULTI_STEP_TOML.replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n').replace(
                "[agent]\n", '[verifier.environment]\ndocker_image = "org/step-verifier:1"\n\n[agent]\n'
            )
        )
        both = load_task(root2)
        assert both.verifier_image == "org/step-verifier:1"
        assert [turn.verifier_image for turn in both.turns] == ["org/step-verifier:1"] * 2
        assert run_images([task, both], {}) == ("org/step-verifier:1", "ubuntu:24.04")
        # Prepare writes the same list, with builds disabled as with them (nothing here builds).
        prepared = prepare_target(
            str(folder), output_root=tmp_path / "out", build=build_options.BuildOptions(enabled=False)
        )
        assert [t.task_id for t in prepared.tasks] == ["t", "u"]
        assert prepared.images == ("org/step-verifier:1", "ubuntu:24.04")
        assert json.loads((tmp_path / "out" / "ds" / "images.json").read_text()) == list(prepared.images)

    def test_a_verifier_network_restriction_needs_every_step_separate(self, tmp_path):
        root = write_multi_step_task(tmp_path / "t")
        toml = (root / "task.toml").read_text()
        restricted = toml.replace(
            "[verifier]\n", '[verifier]\nenvironment_mode = "separate"\nnetwork_mode = "no-network"\n'
        ).replace("[agent]\n", '[verifier.environment]\ndocker_image = "org/verifier:1"\n\n[agent]\n')
        (root / "task.toml").write_text(restricted)
        assert [turn.verifier_mode for turn in load_task(root).turns] == ["separate", "separate"]
        (root / "task.toml").write_text(
            restricted.replace(
                'name = "01_plan"\n', 'name = "01_plan"\n[steps.verifier]\nenvironment_mode = "shared"\n'
            )
        )
        with pytest.raises(HarborTaskError, match="on every step of a multi-step task too"):
            load_task(root)

    def test_a_multi_step_tasks_seed_trajectory_is_the_first_steps_and_any_other_is_ignored(self, tmp_path, caplog):
        """Harbor 0.22.0 loads `steps/<first step>/trajectory.json` into the first step's agent session and ignores a
        top-level one on a multi-step task (`Trial._task_trajectory_path`: one path or the other, never both), as it
        ignores a later step's. The loader records the first step's file for the server to refuse per agent and says
        nothing about the others, so a hub task Harbor runs is not dropped at discovery."""
        root = write_multi_step_task(tmp_path / "t")
        (root / "trajectory.json").write_text("{}")
        (root / "steps" / "02_implement" / "trajectory.json").write_text("{}")
        with caplog.at_level(logging.WARNING, logger="nemo_gym.tasks.harbor.task"):
            task = load_task(root)
        assert task.seed_trajectory is None and len(task.turns) == 2 and "seed trajectory" not in caplog.text
        (root / "steps" / "01_plan" / "trajectory.json").write_text("{}")
        with caplog.at_level(logging.WARNING, logger="nemo_gym.tasks.harbor.task"):
            task = load_task(root)
        assert task.seed_trajectory == Path("steps/01_plan/trajectory.json")
        assert "steps/01_plan/trajectory.json is Harbor's seed trajectory" in caplog.text
        assert "runs the task for the oracle alone" in caplog.text

    def test_a_steps_healthcheck_loads_with_harbors_defaults_for_the_server_to_poll_per_turn(self, tmp_path):
        """Harbor 0.22.0 runs `[steps.healthcheck]` after the step's setup and before its agent
        (`_run_step_healthcheck`, Docker's start-period and retries semantics); the loader used to refuse it."""
        root = write_multi_step_task(tmp_path / "t")
        (root / "task.toml").write_text(
            (root / "task.toml").read_text() + '[steps.healthcheck]\ncommand = "curl -sf localhost:8000/health"\n'
            "start_period_sec = 20.0\nretries = 5\n"
        )
        plan, implement = load_task(root).turns
        assert plan.config.healthcheck is None
        check = implement.config.healthcheck
        assert check is not None and check.command == "curl -sf localhost:8000/health"
        assert (check.start_period_sec, check.retries, check.interval_sec, check.timeout_sec) == (20.0, 5, 5.0, 30.0)

    def test_a_step_agent_user_loads_as_written_for_the_server_to_apply_per_turn(self, tmp_path):
        """Harbor runs each step as its own `[steps.agent].user` (else the task's); the loader used to refuse a step
        that named one (G7 round 24, item 1b). The server now switches the sandbox to it for that turn, so the
        loader reads it as written, a numeric uid included, and leaves the other steps' user unset (the task's)."""
        root = write_multi_step_task(tmp_path / "t")
        (root / "task.toml").write_text((root / "task.toml").read_text() + '[steps.agent]\nuser = "tester"\n')
        task = load_task(root)
        assert [turn.config.agent.user for turn in task.turns] == [None, "tester"]
        (root / "task.toml").write_text((root / "task.toml").read_text().replace('user = "tester"', "user = 1000"))
        assert [turn.config.agent.user for turn in load_task(root).turns] == [None, 1000]

    def test_a_task_has_a_solution_when_every_step_has_one_of_its_own_or_the_tasks(self, tmp_path):
        """`gym dataset validate` keeps a task whose solutions are per step (the MemoryArena shape: no top-level
        `solution/`), as Harbor's oracle runs `steps/<name>/solution/solve.sh` per step and falls back to the task's
        `solution/` for a step without one; a step with neither leaves the task unvalidated, and a step folder that
        exists without a `solve.sh` is that step's (missing) answer, never the task's script, as in Harbor."""
        root = write_multi_step_task(tmp_path / "t")
        shutil.rmtree(root / "solution")
        plan, implement = (root / "steps" / name / "solution" for name in ("01_plan", "02_implement"))
        assert not load_task(root).has_solution
        plan.mkdir()
        (plan / "solve.sh").write_text("#!/bin/bash\necho plan\n")
        task = load_task(root)
        assert not task.has_solution and task.turn_solution_dir(task.turns[1]) == root / "solution"
        # The task's top-level solution stands in for the step without one.
        (root / "solution").mkdir()
        (root / "solution" / "solve.sh").write_text("#!/bin/bash\necho task\n")
        task = load_task(root)
        assert task.has_solution
        assert [task.turn_solution_dir(turn) for turn in task.turns] == [plan, root / "solution"]
        # A step folder of its own without solve.sh is not covered by the task's script.
        implement.mkdir()
        task = load_task(root)
        assert not task.has_solution and task.turn_solution_dir(task.turns[1]) == implement

    def test_a_step_needs_a_test_script_of_its_own_or_the_tasks(self, tmp_path):
        root = write_multi_step_task(tmp_path / "t")
        (root / "steps" / "01_plan" / "tests" / "test.sh").unlink()
        assert load_task(root).turns  # the task's tests/test.sh stands in
        (root / "tests" / "test.sh").unlink()
        with pytest.raises(HarborTaskError, match="no test.sh"):
            load_task(root)


class TestTasksetMapping:
    def test_single_task_folder_maps_to_its_parent(self, tmp_path):
        from nemo_gym.tasks.harbor.materialize import taskset_mapping

        task = load_task(write_task(tmp_path / "solo"))
        mapping = taskset_mapping([task], tmp_path / "solo")
        assert mapping == {"folder": str(tmp_path.resolve()), "tasks": {"solo": task.digest}}
        assert Path(mapping["folder"]) / "solo" == task.path

    def test_folder_of_tasks_maps_to_itself(self, tmp_path):
        from nemo_gym.tasks.harbor.materialize import taskset_mapping

        tasks = [load_task(write_task(tmp_path / "ds" / name)) for name in ("a", "b")]
        assert taskset_mapping(tasks, tmp_path / "ds")["folder"] == str((tmp_path / "ds").resolve())


class TestHub:
    def test_ref_parsing(self):
        assert HubRef.parse("harbor:hello-world") == HubRef("hello-world", None)
        assert HubRef.parse("harbor:terminal-bench@2.0") == HubRef("terminal-bench", "2.0")
        package = HubRef.parse("harbor:terminal-bench/terminal-bench@4.0.0")
        assert package == HubRef("terminal-bench/terminal-bench", "4.0.0") and package.is_package
        assert HubRef.parse("harbor:org/name@sha256:" + "a" * 64).version == "sha256:" + "a" * 64
        assert not HubRef.parse("harbor:hello-world").is_package
        with pytest.raises(HubError):
            HubRef.parse("hello-world")
        with pytest.raises(HubError):
            HubRef.parse("harbor:bad name")

    def test_resolve_dataset_picks_latest_version_unless_pinned(self):
        registry = [
            RegistryDataset("tb", "2.0", "", ()),
            RegistryDataset("tb", "2.10", "", ()),
            RegistryDataset("tb", "1.9", "", ()),
        ]
        assert resolve_dataset(HubRef("tb", None), registry).version == "2.10"
        assert resolve_dataset(HubRef("tb", "1.9"), registry).version == "1.9"
        with pytest.raises(HubError, match="no version"):
            resolve_dataset(HubRef("tb", "3.0"), registry)
        with pytest.raises(HubError, match="not in the Harbor registry"):
            resolve_dataset(HubRef("nope", None), registry)

    def test_mixed_numeric_and_named_versions_need_a_pin(self):
        kumo = [RegistryDataset("kumo", "1.0", "", ()), RegistryDataset("kumo", "parity", "", ())]
        with pytest.raises(HubError, match=r"1\.0, parity.*harbor:kumo@<version>"):
            resolve_dataset(HubRef("kumo", None), kumo)
        assert resolve_dataset(HubRef("kumo", "parity"), kumo).version == "parity"
        assert resolve_dataset(HubRef("kumo", "1.0"), kumo).version == "1.0"

    def test_named_versions_are_never_picked_arbitrarily(self):
        lancer = [
            RegistryDataset("swe-lancer-diamond", "diamond", "", ()),
            RegistryDataset("swe-lancer-diamond", "latest", "", ()),
        ]
        with pytest.raises(HubError, match="pin one"):
            resolve_dataset(HubRef("swe-lancer-diamond", None), lancer)
        assert resolve_dataset(HubRef("swe-lancer-diamond", "diamond"), lancer).version == "diamond"
        # One entry is unambiguous whatever its version is called.
        assert resolve_dataset(HubRef("swe-lancer-diamond", None), lancer[:1]).version == "diamond"

    def test_folder_name_includes_the_version(self):
        assert RegistryDataset("kumo", "1.0", "", ()).folder_name == "kumo-1.0"
        assert RegistryDataset("kumo", "parity", "", ()).folder_name == "kumo-parity"
        assert RegistryDataset("kumo", "a/b c", "", ()).folder_name == "kumo-a-b-c"
        assert RegistryDataset("kumo", "", "", ()).folder_name == "kumo"

    def test_load_registry_uses_cache(self, tmp_path):
        cache = tmp_path / ".harbor"
        cache.mkdir()
        (cache / "registry.json").write_text(
            json.dumps(
                [
                    {
                        "name": "hello-world",
                        "version": "1.0",
                        "description": "d",
                        "tasks": [
                            {
                                "name": "hello-world",
                                "git_url": "https://github.com/x/y",
                                "git_commit_id": "HEAD",
                                "path": "p",
                            }
                        ],
                    }
                ]
            )
        )
        [dataset] = load_registry(cache)
        assert dataset.tasks == (RegistryTask("hello-world", "https://github.com/x/y", "HEAD", "p"),)

    def test_registry_task_names_that_are_not_folder_names_are_refused(self, tmp_path):
        cache = tmp_path / ".harbor"
        cache.mkdir()
        task = {"name": "../..", "git_url": "https://github.com/x/y", "git_commit_id": "HEAD", "path": "p"}
        (cache / "registry.json").write_text(json.dumps([{"name": "d", "version": "1.0", "tasks": [task]}]))
        with pytest.raises(HubError, match=r"'\.\./\.\.'"):
            load_registry(cache)

    @pytest.mark.parametrize(
        "url",
        ["https://github.com/x/y.git", "ssh://git@github.com/x/y", "git://host/x", "git@github.com:x/y.git"],
    )
    def test_registry_git_urls_that_git_fetches_over_the_network(self, url):
        assert validate_git_url(url) == url

    @pytest.mark.parametrize(
        "url",
        [
            "-oProxyCommand=touch /tmp/pwned",
            "--upload-pack=touch /tmp/pwned",
            "ext::sh -c 'touch /tmp/pwned'",
            "file:///etc",
            "/etc/passwd",
            "git@-evil:x",
            "git@host:-flag",
            "https://",
        ],
    )
    def test_registry_git_urls_that_are_refused(self, tmp_path, url):
        with pytest.raises(HubError, match="Unsupported git_url"):
            validate_git_url(url)
        cache = tmp_path / ".harbor"
        cache.mkdir()
        entry = {
            "name": "d",
            "version": "1",
            "tasks": [{"name": "t", "git_url": url, "git_commit_id": "HEAD", "path": "p"}],
        }
        (cache / "registry.json").write_text(json.dumps([entry]))
        with pytest.raises(HubError, match="Unsupported git_url"):
            load_registry(cache)

    def test_fetch_dataset_pins_head_and_copies_tasks(self, tmp_path):
        repo = tmp_path / "repo"
        write_task(repo / "examples" / "tasks" / "hello-world")
        write_task(repo / "examples" / "tasks" / "other")
        subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "uploadpack.allowFilter", "true"], check=True)
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@x", "commit", "-q", "-m", "init"],
            check=True,
        )
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        url = repo.as_uri()
        dataset = RegistryDataset(
            "hello-world",
            "1.0",
            "",
            (
                RegistryTask("hello-world", url, "HEAD", "examples/tasks/hello-world"),
                RegistryTask("renamed", url, head, "examples/tasks/other"),
            ),
        )

        folder = fetch_dataset(dataset, tmp_path / "datasets")

        assert folder == tmp_path / "datasets" / "hello-world-1.0"
        assert (folder / "hello-world" / "task.toml").is_file()
        assert (folder / "renamed" / "tests" / "test.sh").is_file()
        manifest = (folder / "manifest.toml").read_text()
        assert f'git_commit_id = "{head}"' in manifest
        assert "HEAD" not in manifest.replace("hello-world", "")
        assert 'ref = "harbor:hello-world@1.0"' in manifest  # the reference fetched, the version resolved (round 32)
        assert [t.task_id for t in discover_tasks(folder)] == ["hello-world", "renamed"]

        # A rerun keeps existing folders and does not need the repository again.
        assert fetch_dataset(dataset, tmp_path / "datasets") == folder

        # HEAD moves. Present folders keep their pin; only the new task resolves HEAD.
        write_task(repo / "examples" / "tasks" / "third")
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@x", "commit", "-q", "-m", "more"],
            check=True,
        )
        new_head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        assert new_head != head
        grown = RegistryDataset(
            dataset.name,
            dataset.version,
            "",
            (*dataset.tasks, RegistryTask("third", url, "HEAD", "examples/tasks/third")),
        )
        assert fetch_dataset(grown, tmp_path / "datasets") == folder
        pins = tomllib.loads((folder / "manifest.toml").read_text())["tasks"]
        assert pins["hello-world"]["git_commit_id"] == head
        assert pins["renamed"]["git_commit_id"] == head
        assert pins["third"]["git_commit_id"] == new_head
        assert (folder / "third" / "task.toml").is_file()

    def test_fetch_resolves_head_once_per_repo_and_shields_urls(self, tmp_path, monkeypatch):
        url = "https://github.com/org/tasks.git"
        sha = "a" * 40
        calls: list[list[str]] = []

        def fake_run(argv, cwd=None, **kwargs):
            calls.append(list(argv))
            if argv[1] == "ls-remote":
                return SimpleNamespace(stdout=f"{sha}\tHEAD\n", stderr="")
            if argv[1] == "clone":
                for task in dataset.tasks:
                    write_task(Path(argv[-1]) / task.path)
            return SimpleNamespace(stdout="", stderr="")

        monkeypatch.setattr(hub_module.subprocess, "run", fake_run)
        dataset = RegistryDataset(
            "big", "2.0", "", tuple(RegistryTask(f"t{i}", url, "HEAD", f"tasks/t{i}") for i in range(12))
        )

        folder = fetch_dataset(dataset, tmp_path / "datasets")

        ls_remotes = [argv for argv in calls if argv[1] == "ls-remote"]
        clones = [argv for argv in calls if argv[1] == "clone"]
        assert len(ls_remotes) == 1 and len(clones) == 1
        assert ls_remotes[0][-3:] == ["--", url, "HEAD"]
        assert clones[0][-3:-1] == ["--", url]
        assert sorted(p.name for p in folder.iterdir() if p.is_dir()) == sorted(t.name for t in dataset.tasks)
        pins = tomllib.loads((folder / "manifest.toml").read_text())["tasks"]
        assert {pin["git_commit_id"] for pin in pins.values()} == {sha}


class TestTaskNames:
    """One validator guards every name that becomes a folder under the dataset folder."""

    @pytest.mark.parametrize("name", ["../..", "a/b", "a\\b", ".", "..", "", "a\0b"])
    def test_names_that_escape_or_are_not_a_folder_are_rejected(self, name):
        with pytest.raises(HubError) as registry_error:
            validate_task_name(name)
        assert repr(name) in str(registry_error.value)
        with pytest.raises(PackageStoreError) as store_error:
            validate_task_name(name, error=PackageStoreError)
        assert repr(name) in str(store_error.value)

    @pytest.mark.parametrize("name", ["task$1", "name-with.dots_ok"])
    def test_real_task_names_are_accepted(self, name):
        assert validate_task_name(name) == name
        assert validate_task_name(name, error=PackageStoreError) == name


class TestSeparateVerifierFields:
    def test_collect_hooks_and_artifacts_parse(self):
        from pathlib import PurePosixPath

        config = HarborTaskConfig.model_validate(
            {
                "artifacts": [
                    "/app/output/report.json",
                    {"source": "/var/log/api", "destination": "api-logs", "service": "api"},
                ],
                "verifier": {
                    "environment_mode": "separate",
                    "environment": {"docker_image": "org/verifier:1", "cpus": 2},
                    "collect": [
                        {"command": "kafka-dump > /logs/artifacts/topics.txt", "service": "kafka", "timeout_sec": 10}
                    ],
                },
            }
        )
        assert [a.host_path for a in config.artifacts] == [
            PurePosixPath("app/output/report.json"),
            PurePosixPath("api-logs"),
        ]
        assert config.artifacts[1].service == "api"
        assert config.verifier.collect[0].service == "kafka" and config.verifier.collect[0].timeout_sec == 10
        assert not config.is_shared_verifier

    def test_verifier_mode_is_inferred_as_harbor_does(self):
        """A [verifier.environment] block alone selects separate mode; only an explicit "shared" keeps it."""
        config = HarborTaskConfig.model_validate({"verifier": {"environment": {"docker_image": "org/verifier:1"}}})
        assert not config.is_shared_verifier
        config = HarborTaskConfig.model_validate(
            {"verifier": {"environment_mode": "separate"}, "artifacts": ["/app/out"]}
        )
        assert not config.is_shared_verifier
        assert HarborTaskConfig.model_validate({"verifier": {"environment_mode": "shared"}}).is_shared_verifier
        assert HarborTaskConfig.model_validate({"verifier": {"timeout_sec": 30}}).is_shared_verifier

    def test_separate_mode_without_artifacts_loads_with_a_warning(self, tmp_path, caplog):
        # A prebuilt image: Harbor's effective verifier environment (a copy of [environment]) carries it, so the task
        # loads without a tests/Dockerfile; without either, the loader refuses it as Harbor does.
        separate = HELLO_TOML.replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n').replace(
            "[environment]\n", '[environment]\ndocker_image = "org/hello:1"\n'
        )
        with caplog.at_level("WARNING", logger="nemo_gym.tasks.harbor.task"):
            task = load_task(write_task(tmp_path / "t", toml=separate))
        assert not task.config.is_shared_verifier
        assert any("declares no artifacts" in record.message for record in caplog.records)
        with_artifacts = separate.replace(
            'schema_version = "1.4"\n', 'schema_version = "1.4"\nartifacts = ["/logs/artifacts"]\n'
        )
        caplog.clear()
        assert not load_task(write_task(tmp_path / "t", toml=with_artifacts)).config.is_shared_verifier
        assert not any("declares no artifacts" in record.message for record in caplog.records)

    def test_artifact_paths_stay_contained(self):
        with pytest.raises(ValueError, match="inside"):
            HarborTaskConfig.model_validate({"artifacts": ["/app/../etc/passwd"]})
        with pytest.raises(ValueError, match="relative"):
            HarborTaskConfig.model_validate({"artifacts": [{"source": "/a", "destination": "/abs"}]})
        # The verifier's own directories are refused as sources, by any spelling, and a source's leading slashes
        # are settled to one at load (a trailing slash stays: tasks declare `/app/output/`).
        for source in ("/tests", "/tests/test.sh", "//tests", "/", "/logs/verifier", "/logs/verifier/reward.json"):
            settled = "/" + source.lstrip("/")
            with pytest.raises(ValueError, match=f"{settled!r} would replace the verifier's /(tests|logs/verifier) "):
                HarborTaskConfig.model_validate({"artifacts": [source]})
        with pytest.raises(ValueError, match="'/logs/verifier/' would replace the verifier's /logs/verifier"):
            HarborTaskConfig.model_validate({"artifacts": [{"source": "//logs/verifier/", "service": "api"}]})
        accepted = HarborTaskConfig.model_validate(
            {"artifacts": ["//logs", "/logs/", "/tests-data", "/logs/verifier-cache", "/app/output/", "relative"]}
        )
        assert [a.source for a in accepted.artifacts] == [
            "/logs",
            "/logs/",
            "/tests-data",
            "/logs/verifier-cache",
            "/app/output/",
            "relative",
        ]


class TestInstruction:
    def test_leading_canary_lines_are_dropped_and_the_rest_kept_verbatim(self, tmp_path):
        from nemo_gym.tasks.harbor.task import read_instruction

        path = tmp_path / "instruction.md"
        path.write_text(
            "<!-- harbor-canary GUID 26b5c67b -->\n# HARBOR-CANARY marker\n\nDo the thing.\n\nKeep  spacing.\n"
        )
        assert read_instruction(path) == "Do the thing.\n\nKeep  spacing.\n"
        path.write_text("Plain task\n")
        assert read_instruction(path) == "Plain task\n"
        # No canary: the file is the prompt verbatim, a leading blank line included (Terminal-Bench 2.1's
        # query-optimize starts that way, and the legacy server passed it through).
        path.write_text("\nStarts after a blank line.\n")
        assert read_instruction(path) == "\nStarts after a blank line.\n"


class FakeRegistryClient:
    """Stands in for ``RegistryClient`` at prepare: records every image, except the ones told to fail."""

    missing: set[str] = set()
    resolved: list[str] = []

    def __init__(self, *args, **kwargs):
        pass

    def image_config(self, reference: str) -> dict:
        from nemo_gym.tasks.harbor.registry import ImageConfigError

        FakeRegistryClient.resolved.append(reference)
        if reference in FakeRegistryClient.missing:
            raise ImageConfigError(
                f"library/{reference.partition(':')[0]}: registry returned HTTP 404 for manifests/x"
            )
        return {
            "architecture": "amd64",
            "config": {"Env": [f"PATH={IMAGE_PATH}"], "WorkingDir": "/"},
            "config_digest": "sha256:c",
            "image": reference,
            "os": "linux",
            "record_version": RECORD_VERSION,
        }


@pytest.fixture
def fake_registry(monkeypatch):
    from nemo_gym.tasks.harbor import image_configs as image_configs_module

    monkeypatch.setattr(image_configs_module, "RegistryClient", FakeRegistryClient)
    monkeypatch.setattr(FakeRegistryClient, "missing", set())
    monkeypatch.setattr(FakeRegistryClient, "resolved", [])
    return FakeRegistryClient


@pytest.fixture(autouse=True)
def no_real_registry(monkeypatch):
    """No test here, or in a module that imports ``write_task``, reaches a registry. Its default Dockerfile is
    ``FROM ubuntu:24.04`` and prepare records base images, so a test that did not ask for ``fake_registry`` went
    to Docker Hub and failed on its rate limit (HTTP 429) whatever the code under test did."""
    from nemo_gym.tasks.harbor import image_configs as image_configs_module

    monkeypatch.setattr(image_configs_module, "RegistryClient", FakeRegistryClient)
    monkeypatch.setattr(FakeRegistryClient, "missing", set())
    monkeypatch.setattr(FakeRegistryClient, "resolved", [])


@pytest.mark.usefixtures("fake_registry")
class TestCli:
    def test_prepare_local_folder(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        prepared = prepare_target(str(folder), output_root=tmp_path / "out")
        assert prepared.taskset == "ds"
        assert prepared.rows_path == tmp_path / "out" / "ds" / "tasks.jsonl"
        assert [t.task_id for t in prepared.tasks] == ["a"]
        assert prepared.skipped == {}

    def test_prepare_records_base_images_before_loading_and_resolves_the_dockerfiles(self, tmp_path, capsys):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        write_task(folder / "b", dockerfile=TB21_DEBIAN_DOCKERFILE)
        prepared = prepare_target(str(folder), output_root=tmp_path / "out")
        # Images are resolved in task order, once each, before any task loads.
        assert FakeRegistryClient.resolved == ["ubuntu:24.04", "debian:bookworm-slim"]
        recorded = json.loads((folder / IMAGE_CONFIGS_FILE).read_text())
        assert set(recorded) == {"debian:bookworm-slim", "ubuntu:24.04"}
        assert recorded["debian:bookworm-slim"]["config"]["Env"] == [f"PATH={IMAGE_PATH}"]
        overlay = next(task for task in prepared.tasks if task.task_id == "b")
        assert overlay.env == {"PATH": f"/opt/tools/bin:{IMAGE_PATH}", "DEBIAN_FRONTEND": "noninteractive"}
        assert overlay.overlay[0].env == {"PATH": IMAGE_PATH} and overlay.overlay[0].workdir == "/"
        assert f"Base image configurations recorded in {folder / IMAGE_CONFIGS_FILE}" in capsys.readouterr().out
        # A second prepare finds everything recorded and asks the registry for nothing.
        prepare_target(str(folder), output_root=tmp_path / "out")
        assert FakeRegistryClient.resolved == ["ubuntu:24.04", "debian:bookworm-slim"]

    def test_prepare_rejects_duplicate_task_ids_before_registry_or_build_work(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "v1" / "x")
        write_task(folder / "v2" / "x", dockerfile=TB21_DEBIAN_DOCKERFILE)
        with pytest.raises(HarborTaskError, match="Duplicate Harbor task ID 'x'"):
            prepare_target(str(folder), output_root=tmp_path / "out")
        assert FakeRegistryClient.resolved == []
        assert not (folder / IMAGE_CONFIGS_FILE).exists()

    def test_prepare_records_compose_images_at_the_dataset_root_for_nested_tasks(self, tmp_path):
        folder = tmp_path / "ds"
        web = write_task(folder / "00-services" / "web")
        (web / "environment" / "docker-compose.yaml").write_text("services:\n  api:\n    image: redis:7-alpine\n")
        write_task(folder / "zeta")
        prepared = prepare_target(str(folder), output_root=tmp_path / "out")
        assert [task.task_id for task in prepared.tasks] == ["web", "zeta"]
        # Where the loader and the environment server read it from: the dataset root, not the group folder.
        recorded = json.loads((folder / "compose-images.json").read_text())
        assert "redis:7-alpine" in recorded
        assert not (folder / "00-services" / "compose-images.json").exists()

    def test_prepare_fails_clearly_when_a_base_image_cannot_be_fetched(self, tmp_path):
        from nemo_gym.tasks.harbor.registry import ImageConfigError

        folder = tmp_path / "ds"
        write_task(folder / "a")
        write_task(folder / "b", dockerfile=TB21_DEBIAN_DOCKERFILE)
        FakeRegistryClient.missing.add("debian:bookworm-slim")
        with pytest.raises(ImageConfigError) as excinfo:
            prepare_target(str(folder), output_root=tmp_path / "out")
        message = str(excinfo.value)
        assert "base image 'debian:bookworm-slim'" in message and "task(s) b" in message
        assert "registry returned HTTP 404" in message
        assert not (tmp_path / "out").exists()

    @pytest.mark.parametrize(
        "kw,reason",
        [({"only": ["a"]}, "not in --only-tasks"), ({"exclude": ["z"]}, "excluded by --exclude-tasks 'z'")],
    )
    def test_prepare_does_not_fetch_the_base_image_of_a_left_out_task(self, tmp_path, capsys, kw, reason):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        write_task(folder / "z", dockerfile="FROM internal.registry/base:dev\nRUN true\n")
        FakeRegistryClient.missing.add("internal.registry/base:dev")
        prepared = prepare_target(str(folder), output_root=tmp_path / "out", **kw)
        assert [t.task_id for t in prepared.tasks] == ["a"]
        # The left-out task did not load (its base image is not recorded) but is left out, not reported broken.
        assert prepared.skipped == {}
        assert FakeRegistryClient.resolved == ["ubuntu:24.04"]
        assert f"Skipping z: {reason}" in capsys.readouterr().out

    def test_prepare_single_task_folder_left_out_by_only_still_says_no_runnable_task(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "z", dockerfile="FROM ubuntu:24.04\nRUN true\n")
        with pytest.raises(ValueError, match="No runnable task"):
            prepare_target(str(folder / "z"), output_root=tmp_path / "out", only=["a"])
        assert FakeRegistryClient.resolved == ["ubuntu:24.04"]

    def test_prepare_does_not_fetch_the_base_image_a_dataset_override_replaces(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        write_task(folder / "t", dockerfile="FROM internal.registry/base:dev\nRUN make\n")
        (folder / "dataset.toml").write_text('[gym.tasks."t".environment]\ndocker_image = "ghcr.io/org/t:prebuilt"\n')
        FakeRegistryClient.missing.add("internal.registry/base:dev")
        prepared = prepare_target(str(folder), output_root=tmp_path / "out")
        assert [(t.task_id, t.image, t.overlay) for t in prepared.tasks] == [
            ("a", "ubuntu:24.04", ()),
            ("t", "ghcr.io/org/t:prebuilt", ()),
        ]
        assert FakeRegistryClient.resolved == ["ubuntu:24.04"]

    def test_prepare_skips_tasks_that_do_not_load(self, tmp_path, caplog):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        write_task(folder / "bad", toml="[[steps]]\nname = 'x'")
        with caplog.at_level(logging.WARNING, logger="nemo_gym.tasks.harbor.cli"):
            prepared = prepare_target(str(folder), output_root=tmp_path / "out")
        assert [t.task_id for t in prepared.tasks] == ["a"]
        assert list(prepared.skipped) == ["bad"]
        assert "step 'x'" in prepared.skipped["bad"]
        assert any("Skipping task bad" in r.getMessage() for r in caplog.records)
        assert len(prepared.rows_path.read_text().splitlines()) == 1

    def test_prepare_refuses_a_dataset_task_key_that_names_no_task(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "nextjs-performance")
        (folder / "dataset.toml").write_text('[gym.tasks."nextjs_performance".environment]\nenv = { A = "1" }\n')
        with pytest.raises(
            ValueError, match=r"dataset.toml \[gym.tasks\] names unknown tasks: \['nextjs_performance'\]"
        ):
            prepare_target(str(folder), output_root=tmp_path / "out")
        # Spelt right, it applies; a single task folder as the target is prepared without its siblings' keys.
        (folder / "dataset.toml").write_text(
            '[gym.tasks."nextjs-performance".environment]\nenv = { A = "1" }\n[gym.tasks."other".environment]\nenv = { B = "2" }\n'
        )
        prepared = prepare_target(str(folder / "nextjs-performance"), output_root=tmp_path / "out2")
        assert prepared.tasks[0].env == {"A": "1"}
        with pytest.raises(ValueError, match=r"\['other'\]"):
            prepare_target(str(folder), output_root=tmp_path / "out3")

    def test_prepare_skips_excluded_tasks_and_keeps_compose_tasks(self, tmp_path, capsys):
        folder = tmp_path / "ds"
        for name in ("keep", "gpu-task", "grouped"):
            write_task(folder / name)
        (folder / "grouped" / "environment" / "docker-compose.yaml").write_text("services: {}\n")
        prepared = prepare_target(str(folder), output_root=tmp_path / "out", exclude=["gpu-*"])
        # Compose tasks run like any other since Compose groups landed.
        assert [task.task_id for task in prepared.tasks] == ["grouped", "keep"]
        assert prepared.tasks[0].needs_compose
        # The Compose task's images were recorded through the (faked) registry, never Docker Hub.
        assert "ubuntu:24.04" in FakeRegistryClient.resolved
        out = capsys.readouterr().out
        assert "Skipping gpu-task: excluded by --exclude-tasks 'gpu-*'" in out
        assert len((prepared.rows_path).read_text().splitlines()) == 2
        with pytest.raises(ValueError, match="No runnable task"):
            prepare_target(str(folder), output_root=tmp_path / "out2", exclude=["*"])
        only = prepare_target(str(folder), output_root=tmp_path / "out3", only=["gpu-*"])
        assert [task.task_id for task in only.tasks] == ["gpu-task"]
        assert "Skipping keep: not in --only-tasks" in capsys.readouterr().out

    def test_prepare_leaves_out_tasks_whose_compose_file_does_not_resolve(self, tmp_path, monkeypatch, caplog):
        # N7: one task's Compose file no longer drops the dataset; the task is left out like one that does not load.
        monkeypatch.delenv("API_KEY", raising=False)
        folder = tmp_path / "ds"
        services = {
            "good": "environment: ['TEST_DIR=${TEST_DIR}']",
            "bad-required": "environment: ['API_KEY=${API_KEY:?API_KEY is needed}']",
            "bad-env-file": "env_file: missing.env",
        }
        for name, line in services.items():
            write_task(folder / name)
            (folder / name / "environment" / "docker-compose.yaml").write_text(f"services:\n  main:\n    {line}\n")
        with caplog.at_level(logging.WARNING, logger="nemo_gym.tasks.harbor.cli"):
            prepared = prepare_target(str(folder), output_root=tmp_path / "out")
        assert [t.task_id for t in prepared.tasks] == ["good"]
        assert sorted(prepared.skipped) == ["bad-env-file", "bad-required"]
        assert prepared.skipped["bad-required"] == (
            "bad-required: docker-compose.yaml: ${API_KEY} is required: API_KEY is needed"
        )
        assert "environment/missing.env' not found" in prepared.skipped["bad-env-file"]
        assert json.loads((prepared.output_dir / "skipped.json").read_text()) == prepared.skipped
        assert len(prepared.rows_path.read_text().splitlines()) == 1
        assert sorted(r.getMessage()[:22] for r in caplog.records) == [
            "Skipping task bad-env-",
            "Skipping task bad-requ",
        ]
        # With nothing left, one error lists every task and its reason.
        (folder / "good" / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    env_file: gone.env\n"
        )
        with pytest.raises(ValueError, match=r"No runnable task left in .*; left out:") as info:
            prepare_target(str(folder), output_root=tmp_path / "out2")
        message = str(info.value)
        assert "\n  bad-required: docker-compose.yaml: ${API_KEY} is required: API_KEY is needed" in message
        assert "\n  bad-env-file: docker-compose.yaml: " in message and "\n  good: docker-compose.yaml: " in message

    def test_prepare_leaves_out_tasks_whose_image_fails_to_build_and_builds_the_rest(
        self, tmp_path, monkeypatch, capsys
    ):
        from nemo_gym.tasks.harbor.build_tools import ImageBuilder

        folder = tmp_path / "ds"
        write_task(folder / "plain")
        write_task(folder / "ok", dockerfile="FROM ubuntu:24.04\nCOPY ok.txt /ok.txt\n")
        write_task(folder / "bad", dockerfile="FROM ubuntu:24.04\nCOPY bad.txt /bad.txt\n")
        failing = str(folder / "bad" / "environment")

        def docker(argv, env=None):
            if failing in argv:
                return subprocess.CompletedProcess(argv, 1, "", "build: boom\n")
            if "inspect" in argv:
                data = [
                    {"Id": "sha256:def456", "Architecture": "amd64", "Os": "linux", "Config": {"WorkingDir": "/srv"}}
                ]
                return subprocess.CompletedProcess(argv, 0, json.dumps(data), "")
            return subprocess.CompletedProcess(argv, 0, "built\n", "")

        monkeypatch.setattr(prepare_builds, "ImageBuilder", lambda **kw: ImageBuilder(tool="docker", run=docker))
        prepared = prepare_target(str(folder), output_root=tmp_path / "out")
        # The context that failed leaves its task out with the build's reason; the other context still built and
        # its task runs from the built image.
        assert [t.task_id for t in prepared.tasks] == ["ok", "plain"]
        ok = prepared.tasks[0]
        assert ok.built_image is not None and ok.image == ok.built_image and "hb__" in ok.image
        assert list(prepared.skipped) == ["bad"]
        assert prepared.skipped["bad"].startswith("Could not build the Dockerfile image(s) for task(s) bad: ")
        assert "boom" in prepared.skipped["bad"]
        assert json.loads((prepared.output_dir / "skipped.json").read_text()) == prepared.skipped
        assert "Built 1 image(s) with docker, kept local" in capsys.readouterr().out
        # The same failure as the only task: the one error carries the build's reason.
        with pytest.raises(ValueError, match=r"No runnable task left.*\n  bad: Could not build the Dockerfile image"):
            prepare_target(str(folder / "bad"), output_root=tmp_path / "out2")

    def test_a_main_build_one_task_gets_wrong_leaves_that_task_out_and_the_rest_prepares(
        self, tmp_path, monkeypatch, capsys
    ):
        # main's build section is read at build time, per context, never at planning: the cadir-ci-benchmark shape
        # (`${VAR:?...}` in services.main.build with the variable unset) and a task env the host does not set leave
        # those tasks out with their reasons and nothing else, with builds enabled or disabled.
        from nemo_gym.tasks.harbor.build_options import BuildOptions
        from nemo_gym.tasks.harbor.build_tools import ImageBuilder

        monkeypatch.delenv("NEMO_GYM_NOT_SET", raising=False)
        folder = tmp_path / "ds"
        write_task(folder / "plain")
        write_task(folder / "ok", dockerfile="FROM ubuntu:24.04\nCOPY ok.txt /ok.txt\n")
        write_task(folder / "bad", dockerfile="FROM ubuntu:24.04\nCOPY bad.txt /bad.txt\n")
        (folder / "bad" / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    build:\n      args:\n        RUNTIME: ${NEMO_GYM_NOT_SET:?set NEMO_GYM_NOT_SET}\n"
        )
        env_toml = HELLO_TOML + '\n[environment.env]\nFOO = "${NEMO_GYM_NOT_SET}"\n'
        write_task(folder / "bad-env", dockerfile="FROM ubuntu:24.04\nCOPY x /x\n", toml=env_toml)
        (folder / "bad-env" / "environment" / "docker-compose.yaml").write_text("services:\n  main:\n    build: .\n")
        # Builds disabled: no Compose file is read, nothing raises, and the loader leaves the three out as before.
        prepared = prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(enabled=False))
        assert [t.task_id for t in prepared.tasks] == ["plain"]
        assert sorted(prepared.skipped) == ["bad", "bad-env", "ok"]
        assert all("needs a build (COPY)" in reason for reason in prepared.skipped.values())

        def docker(argv, env=None):
            if "inspect" in argv:
                data = [
                    {"Id": "sha256:def456", "Architecture": "amd64", "Os": "linux", "Config": {"WorkingDir": "/srv"}}
                ]
                return subprocess.CompletedProcess(argv, 0, json.dumps(data), "")
            return subprocess.CompletedProcess(argv, 0, "built\n", "")

        monkeypatch.setattr(prepare_builds, "ImageBuilder", lambda **kw: ImageBuilder(tool="docker", run=docker))
        prepared = prepare_target(str(folder), output_root=tmp_path / "out2")
        assert [t.task_id for t in prepared.tasks] == ["ok", "plain"]
        assert sorted(prepared.skipped) == ["bad", "bad-env"]
        assert prepared.skipped["bad"].startswith("Could not build the Dockerfile image(s) for task(s) bad: ")
        assert "${NEMO_GYM_NOT_SET} is required: set NEMO_GYM_NOT_SET" in prepared.skipped["bad"]
        assert "[environment].env FOO of task(s) bad-env reads ${NEMO_GYM_NOT_SET}" in prepared.skipped["bad-env"]
        assert json.loads((prepared.output_dir / "skipped.json").read_text()) == prepared.skipped
        assert "Built 1 image(s) with docker, kept local" in capsys.readouterr().out
        # Alone, the one error carries the interpolation reason.
        with pytest.raises(ValueError, match=r"No runnable task left.*\n  bad: Could not build.*set NEMO_GYM_NOT_SET"):
            prepare_target(str(folder / "bad"), output_root=tmp_path / "out3")

    def test_prepare_scopes_rows_to_the_agent(self, tmp_path):
        """Two agents prepared with different selections on one taskset keep their own rows and configs."""
        folder = tmp_path / "ds"
        write_task(folder / "alpha")
        write_task(folder / "beta")
        hermes = AgentSelection(tmp_path / "agent.yaml", "hermes_agent", "hermes_agent")
        terminus = AgentSelection(tmp_path / "t.yaml", "terminus_2_sandboxed_agent", "terminus_2_sandboxed_agent")

        first = prepare_target(
            str(folder), output_root=tmp_path / "out", only=["alpha"], rows_name="tasks_hermes_agent.jsonl"
        )
        hermes_config, _ = build_run(first, hermes, sandbox=None, overrides=[])
        second = prepare_target(
            str(folder),
            output_root=tmp_path / "out",
            only=["beta"],
            rows_name="tasks_terminus_2_sandboxed_agent.jsonl",
        )
        terminus_config, _ = build_run(second, terminus, sandbox=None, overrides=[])

        def referenced_rows(config_path: Path) -> list[str]:
            paths: list[str] = []

            def walk(node):
                if isinstance(node, dict):
                    for key, value in node.items():
                        paths.append(value) if key == "jsonl_fpath" else walk(value)
                elif isinstance(node, list):
                    for item in node:
                        walk(item)

            walk(yaml.safe_load(config_path.read_text()))
            (path,) = paths
            return [json.loads(line)["task_id"]["task_id"] for line in Path(path).read_text().splitlines()]

        assert first.rows_path != second.rows_path and first.output_dir == second.output_dir
        # Each config still references the rows of its own selection after the other agent was prepared.
        assert referenced_rows(hermes_config) == ["alpha"]
        assert referenced_rows(terminus_config) == ["beta"]

    def test_resolve_agent_accepts_short_name(self):
        selection = resolve_agent("simple")
        assert selection.instance_name == "simple_agent"
        assert selection.impl_name == "simple_agent"
        assert selection.config_path.name == "simple_agent.yaml"
        # `hermes` and `terminus_2` each match two harnesses; the caller has to pick.
        for ambiguous in ("hermes", "terminus_2"):
            with pytest.raises(ValueError, match="ambiguous"):
                resolve_agent(ambiguous)
        terminus = resolve_agent("terminus_2_sandboxed_agent")
        assert terminus.instance_name == "terminus_2_sandboxed_agent"
        assert terminus.impl_name == "terminus_2_sandboxed_agent"
        with pytest.raises(ValueError, match="No agent config"):
            resolve_agent("no_such_agent_xyz")

    def test_build_run_writes_config_and_overrides(self, tmp_path):
        folder = tmp_path / "ds"
        tasks = [load_task(write_task(folder / "a"))]
        prepared = PreparedTaskset("ds", folder, tasks, tmp_path / "out" / "tasks.jsonl", tmp_path / "out")
        prepared.output_dir.mkdir(parents=True)
        agent = AgentSelection(tmp_path / "agent.yaml", "hermes_agent", "hermes_agent")

        config_path, tokens = build_run(
            prepared, agent, sandbox="opensandbox", overrides=["+agent_name=x", "+split=train"]
        )

        written = yaml.safe_load(config_path.read_text())
        assert written["environment_server_routes"] == {"ds": "harbor_ds_environment"}
        assert written["harbor_ds_agent"]["responses_api_agents"]["hermes_agent"]["enabled_toolsets"] == ["terminal"]
        terminus = AgentSelection(tmp_path / "t.yaml", "terminus_2_sandboxed_agent", "terminus_2_sandboxed_agent")
        terminus_path, _ = build_run(prepared, terminus, sandbox=None, overrides=[])
        # Each agent gets its own file, so the hermes config above is untouched.
        assert terminus_path != config_path and terminus_path.name == "run_config_terminus_2_sandboxed_agent.yaml"
        written = yaml.safe_load(terminus_path.read_text())
        generated = written["harbor_ds_agent"]["responses_api_agents"]["terminus_2_sandboxed_agent"]
        assert generated["num_workers"] == 1
        # The shipped template leaves `resources_server.name` unset for benchmark composition; the Harbor
        # instance declares none, so it is neither a swap source nor bound to a resources server.
        assert "resources_server" in generated and generated["resources_server"] is None
        assert "hermes_agent" in yaml.safe_load(config_path.read_text())["harbor_ds_agent"]["responses_api_agents"]
        assert "+split=train" in tokens
        assert not any(token.startswith("+agent_name") for token in tokens)
        config_paths = next(token for token in tokens if token.startswith("+config_paths="))
        assert str(config_path.resolve()) in config_paths
        assert "environment_servers/harbor/configs/harbor.yaml" in config_paths
        assert "resources_servers" not in config_paths
        assert "opensandbox/configs/opensandbox.yaml" in config_paths
        assert yaml.safe_load(config_path.read_text())["output_jsonl_fpath"] == str(
            tmp_path / "out" / "rollouts.jsonl"
        )
        assert not any(token.lstrip("+").startswith("use_absolute_ip=") for token in tokens)

    def test_resolve_agent_marks_in_sandbox_harnesses(self):
        assert resolve_agent("terminus_2_sandboxed").runs_in_sandbox is False
        assert resolve_agent("hermes_agent").runs_in_sandbox is True
        assert resolve_agent("codex").runs_in_sandbox is True
        assert resolve_agent("harbor_oracle").runs_in_sandbox is False

    @pytest.mark.parametrize(
        "impl_name",
        [
            "codex_agent",
            "opencode_agent",
            "pi_agent",
            "openclaw_agent",
            "hermes_agent",
            "hermes_sandboxed_agent",
            "miniswe_sandboxed_agent",
            "opencode_sandboxed_agent",
            "pi_sandboxed_agent",
        ],
    )
    def test_shipped_sandboxed_harnesses_declare_model_calls_from_sandbox(self, impl_name):
        """Each harness that calls the model from the task sandbox declares it on its class; the CLI keeps no
        list of names, so a run with it advertises a reachable model server without ++use_absolute_ip=true."""
        assert runs_in_sandbox(impl_name, {"entrypoint": "app.py"}) is True

    @pytest.mark.parametrize("impl_name", ["terminus_2_sandboxed_agent", "terminus_2_agent", "harbor_oracle_agent"])
    def test_host_side_harnesses_declare_nothing(self, impl_name):
        # Terminus 2 drives a tmux session in the sandbox but calls the model from the agent server.
        assert runs_in_sandbox(impl_name, {"entrypoint": "app.py"}) is False

    def test_runs_in_sandbox_from_declaration_or_config_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr("nemo_gym.tasks.harbor.cli.component_search_roots", lambda: [tmp_path])

        def write_agent(name, source, entrypoint="app.py"):
            folder = tmp_path / "responses_api_agents" / name
            folder.mkdir(parents=True)
            (folder / entrypoint).write_text(source)

        declared = (
            "class Agent(SimpleResponsesAPIAgent):\n    calls_model_from_sandbox = True\n\nAgent.run_webserver()\n"
        )
        write_agent("declared_agent", declared)
        write_agent("custom_agent", declared, entrypoint="serve.py")
        write_agent("silent_agent", "class Agent(SimpleResponsesAPIAgent):\n    pass\n\nAgent.run_webserver()\n")
        write_agent(
            "heir_agent",
            "class Base:\n    calls_model_from_sandbox = True\n\nclass Agent(Base):\n    pass\n\nAgent.run_webserver()\n",
        )

        assert runs_in_sandbox("declared_agent", {"entrypoint": "app.py"}) is True
        assert runs_in_sandbox("declared_agent", {}) is True  # the entrypoint defaults to app.py
        assert runs_in_sandbox("custom_agent", {"entrypoint": "serve.py"}) is True
        assert runs_in_sandbox("custom_agent", {}) is False
        # Not declared.
        assert runs_in_sandbox("silent_agent", {"model_server": {"name": "policy_model"}}) is False
        # An inherited declaration is not read from source: each class declares it itself.
        assert runs_in_sandbox("heir_agent", {}) is False
        # The config key alone is enough, whatever the class says.
        assert runs_in_sandbox("silent_agent", {"sandbox_model_base_url": None}) is True
        assert runs_in_sandbox("anyswe_agent", {"sandbox_model_base_url": None}) is True
        # Neither, including an agent folder that does not exist.
        assert runs_in_sandbox("anyswe_agent", {"model_server": {"name": "policy_model"}}) is False
        assert runs_in_sandbox("missing_agent", None) is False

    def _prepared(self, tmp_path):
        folder = tmp_path / "ds"
        tasks = [load_task(write_task(folder / "a"))]
        prepared = PreparedTaskset("ds", folder, tasks, tmp_path / "out" / "tasks.jsonl", tmp_path / "out")
        prepared.output_dir.mkdir(parents=True)
        return prepared

    def test_build_run_advertises_node_ip_for_the_model_server_only(self, tmp_path):
        """The sandboxed harness reaches the model server, so that is the one server the run config moves off
        loopback; the head server (its config endpoint returns the merged configuration), the agent and the
        environment server are not named and keep 127.0.0.1."""
        agent = resolve_agent("hermes_agent")
        path, _ = build_run(self._prepared(tmp_path), agent, sandbox="opensandbox", overrides=[])
        written = yaml.safe_load(path.read_text())
        assert written["use_absolute_ip"] is True
        assert written["absolute_ip_server_types"] == ["responses_api_models"]

    def test_build_run_leaves_host_side_agent_on_loopback(self, tmp_path):
        agent = AgentSelection(tmp_path / "agent.yaml", "harbor_oracle_agent", "harbor_oracle_agent")
        path, _ = build_run(self._prepared(tmp_path), agent, sandbox="docker", overrides=[])
        written = yaml.safe_load(path.read_text())
        assert written.get("use_absolute_ip", False) is False
        assert "absolute_ip_server_types" not in written

    @pytest.mark.parametrize("override", ["+use_absolute_ip=false", "++use_absolute_ip=false"])
    def test_build_run_respects_caller_use_absolute_ip(self, tmp_path, override):
        agent = AgentSelection(tmp_path / "agent.yaml", "hermes_agent", "hermes_agent", runs_in_sandbox=True)
        _, tokens = build_run(self._prepared(tmp_path), agent, sandbox=None, overrides=[override])
        assert [token for token in tokens if "use_absolute_ip" in token] == [override]

    def test_build_run_merges_caller_overlays_after_component_defaults(self, tmp_path):
        """A later config_paths entry wins and replaces lists, so a run overlay must come after the
        component and provider defaults or `setup.commands: []` in opensandbox.yaml would erase it."""
        from nemo_gym.cli.main import _merge_config_paths

        folder = tmp_path / "ds"
        tasks = [load_task(write_task(folder / "a"))]
        prepared = PreparedTaskset("ds", folder, tasks, tmp_path / "out" / "tasks.jsonl", tmp_path / "out")
        prepared.output_dir.mkdir(parents=True)
        agent = AgentSelection(tmp_path / "agent.yaml", "hermes_agent", "hermes_agent")
        overlay = tmp_path / "overlay.yaml"

        _, tokens = build_run(
            prepared, agent, sandbox="opensandbox", overrides=[f"+config_paths=[{overlay}]", "+split=train"]
        )

        merged = [token for token in _merge_config_paths(tokens) if token.startswith("+config_paths=[")]
        assert len(merged) == 1
        # Entries are JSON-quoted when written; the merge may re-emit them bare.
        paths = [json.loads(p) if p.startswith('"') else p for p in merged[0][len("+config_paths=[") : -1].split(",")]
        assert paths[-1] == str(overlay)
        assert any(path.endswith("opensandbox.yaml") for path in paths[:-1])
        assert any(path.endswith("harbor.yaml") for path in paths[:-1])


@pytest.mark.parametrize("failure", ["encoding", "escape", "directory_link", "dangling_link", "io"])
def test_discovery_keeps_file_errors_inside_task_boundary(tmp_path, monkeypatch, failure):
    write_task(tmp_path / "ds" / "good")
    bad = write_task(tmp_path / "ds" / "bad")
    if failure == "encoding":
        (bad / "environment" / "Dockerfile").write_bytes(b"\xff")
    elif failure == "escape":
        marker = tmp_path / "marker"
        marker.write_text("private marker")
        (bad / "tests" / "leak").symlink_to(marker)
    elif failure == "directory_link":
        (bad / "tests" / "data").symlink_to("../solution", target_is_directory=True)
    elif failure == "dangling_link":
        (bad / "tests" / "data").symlink_to("missing", target_is_directory=True)
    else:
        original = Path.open

        def fail_open(path, *args, **kwargs):
            if path == bad / "tests" / "test.sh":
                raise PermissionError("test read denied")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", fail_open)
    skipped = {}
    assert [task.task_id for task in discover_tasks(tmp_path / "ds", skipped=skipped)] == ["good"]
    assert list(skipped) == ["bad"]
    assert isinstance(skipped["bad"], HarborTaskError)


@pytest.mark.parametrize("section", ["environment", "agent", "verifier", "verifier.environment"])
@pytest.mark.parametrize(
    "policy", ['network_mode = "no-network"', 'network_mode = "allowlist"', 'allowed_hosts = ["example.org"]']
)
def test_loader_rejects_unenforced_network_policies(tmp_path, section, policy):
    # A shared-mode task: every phase's restriction is refused, the verifier's for a reason that names the fix.
    with pytest.raises(HarborTaskError, match="network.*not supported") as excinfo:
        load_task(write_task(tmp_path / "t", toml=f"[{section}]\n{policy}\n"))
    if section == "verifier":
        assert "shared verifier mode" in str(excinfo.value) and 'environment_mode = "separate"' in str(excinfo.value)


SEPARATE_VERIFIER_TOML = HELLO_TOML.replace(
    "[verifier]\ntimeout_sec = 120.0", '[verifier]\ntimeout_sec = 120.0\nenvironment_mode = "separate"'
).replace("[environment]\n", '[environment]\ndocker_image = "ubuntu:24.04"\n')


@pytest.mark.parametrize(
    ("policy", "mode", "hosts"),
    [
        ('network_mode = "no-network"', "no-network", None),
        ('network_mode = "allowlist"', "allowlist", None),
        (
            'network_mode = "allowlist"\nallowed_hosts = ["pypi.org", "*.nvidia.com"]',
            "allowlist",
            ["pypi.org", "*.nvidia.com"],
        ),
    ],
)
def test_loader_admits_a_verifier_phase_policy_for_separate_verification(tmp_path, policy, mode, hosts):
    """The restricted policy can be applied to the separate verifier's own sandbox, so the loader admits it there."""
    toml = SEPARATE_VERIFIER_TOML.replace('environment_mode = "separate"', f'environment_mode = "separate"\n{policy}')
    task = load_task(write_task(tmp_path / "t", toml=toml))
    assert not task.config.is_shared_verifier
    assert (task.config.verifier.network_mode, task.config.verifier.allowed_hosts) == (mode, hosts)
    # The agent's own sandbox stays public: the other phases are still refused on the same task.
    for section in ("environment", "agent"):
        restricted = toml.replace(f"[{section}]\n", f'[{section}]\nnetwork_mode = "no-network"\n')
        with pytest.raises(HarborTaskError, match=f"{section} network restrictions are not supported yet"):
            load_task(write_task(tmp_path / section, toml=restricted))


@pytest.mark.parametrize("policy", ['network_mode = "no-network"', 'network_mode = "public"', ""])
def test_loader_refuses_verifier_allowed_hosts_without_an_allowlist(tmp_path, policy):
    toml = SEPARATE_VERIFIER_TOML.replace(
        'environment_mode = "separate"', f'environment_mode = "separate"\n{policy}\nallowed_hosts = ["pypi.org"]'
    )
    with pytest.raises(HarborTaskError, match="allowed_hosts is only valid when network_mode='allowlist'"):
        load_task(write_task(tmp_path / "t", toml=toml))


def test_declared_sidecar_inputs_come_from_dataset_toml_and_are_plain_file_names(tmp_path):
    """`[verifier].inputs` (round 38 review, P1): the names under /logs/verifier a sidecar writes for test.sh, which
    the server's reset of the folder keeps; set from the dataset's `dataset.toml`, part of the task's digest, and
    never a path, a glob pattern (the server keeps them with `find -path`, which matches a pattern), a grader output
    or a repeated name."""
    from nemo_gym.tasks.harbor.models import HarborVerifier

    folder = tmp_path / "ds"
    write_task(folder / "hello")
    plain = load_task(folder / "hello")
    assert plain.config.verifier.inputs == []
    (folder / "dataset.toml").write_text('[gym.tasks."hello".verifier]\ninputs = ["actions.jsonl"]\n')
    declared = load_task(folder / "hello")
    assert declared.config.verifier.inputs == ["actions.jsonl"] and declared.digest != plain.digest
    for inputs, message in (
        (["logs/actions.jsonl"], "plain file name"),
        ([".."], "plain file name"),
        ([""], "plain file name"),
        (["log[1].jsonl"], "not a glob pattern"),
        (["*.jsonl"], "not a glob pattern"),
        (["a?.jsonl"], "not a glob pattern"),
        (["reward.json"], "grader's own output"),
        (["a.jsonl", "a.jsonl"], "names a file twice"),
    ):
        with pytest.raises(ValueError, match=message):
            HarborVerifier(inputs=inputs)
    (folder / "dataset.toml").write_text('[gym.tasks."hello".verifier]\ninputs = ["reward.txt"]\n')
    with pytest.raises(HarborTaskError, match="grader's own output"):
        load_task(folder / "hello")


def _with_steps(root: Path, names: list[str]) -> Path:
    """``root`` with one ``steps/<name>/`` folder per name (an instruction and a stub test.sh), as the loader reads."""
    for name in names:
        (root / "steps" / name / "tests").mkdir(parents=True, exist_ok=True)
        (root / "steps" / name / "instruction.md").write_text(f"Step {name}.\n")
        (root / "steps" / name / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    return root


STEP_GRADER_IMAGE = '[steps.verifier.environment]\ndocker_image = "org/grader:1"\n'


@pytest.mark.parametrize("restriction", ['network_mode = "no-network"', 'allowed_hosts = ["pypi.org"]'])
def test_loader_refuses_a_step_verifier_environment_network_restriction(tmp_path, restriction):
    """Round 38 review (P1): a `[steps.verifier.environment]` with a `docker_image` and `network_mode = "no-network"`
    loaded and reached the verifier sandbox's creation with no egress policy, dropping the restriction the task asked
    for, where the same restriction on `[verifier.environment]` is refused. The step's environment is checked like the
    task's now; without the restriction the step loads in separate mode with its own image."""
    steps = f'\n[[steps]]\nname = "plan"\n{STEP_GRADER_IMAGE}{restriction}\n'
    with pytest.raises(
        HarborTaskError, match="steps.plan.verifier.environment network restrictions are not supported"
    ):
        load_task(_with_steps(write_task(tmp_path / "restricted", toml=HELLO_TOML + steps), ["plan"]))
    task = load_task(
        _with_steps(
            write_task(tmp_path / "plain", toml=HELLO_TOML + '\n[[steps]]\nname = "plan"\n' + STEP_GRADER_IMAGE),
            ["plan"],
        )
    )
    (plan,) = task.turns
    assert (plan.verifier_mode, plan.verifier_image) == ("separate", "org/grader:1")


def test_a_task_wide_verifier_restriction_checks_the_steps_resolved_modes(tmp_path):
    """Round 38 review (P3): a task-wide `[verifier].network_mode` on a shared-default task whose every step grades
    separately (each with its own `[steps.verifier.environment]`) was refused for the unused task-level default,
    although every grader that runs satisfies the restriction. The steps' resolved modes decide for a multi-step task;
    a step left shared is still refused, and a task without steps keeps the top-level check."""
    restricted = HELLO_TOML.replace(
        "[verifier]\ntimeout_sec = 120.0", '[verifier]\ntimeout_sec = 120.0\nnetwork_mode = "no-network"'
    )
    separate_steps = f'\n[[steps]]\nname = "plan"\n{STEP_GRADER_IMAGE}[[steps]]\nname = "build"\n{STEP_GRADER_IMAGE}'
    task = load_task(_with_steps(write_task(tmp_path / "ok", toml=restricted + separate_steps), ["plan", "build"]))
    assert task.config.is_shared_verifier and task.config.verifier.network_mode == "no-network"
    assert [turn.verifier_mode for turn in task.turns] == ["separate", "separate"]
    mixed = restricted + f'\n[[steps]]\nname = "plan"\n{STEP_GRADER_IMAGE}[[steps]]\nname = "build"\n'
    with pytest.raises(HarborTaskError, match="not supported in Harbor's shared verifier mode"):
        load_task(_with_steps(write_task(tmp_path / "mixed", toml=mixed), ["plan", "build"]))
    with pytest.raises(HarborTaskError, match="not supported in Harbor's shared verifier mode"):
        load_task(write_task(tmp_path / "single", toml=restricted))


@pytest.mark.parametrize(
    "toml",
    [
        "[environment]\nallow_internet = false",
        '[environment]\nos = "Windows"',
        '[environment.tpu]\ntype = "v6e"\ntopology = "1x1"',
    ],
)
def test_loader_rejects_unsupported_execution_requirements(tmp_path, toml):
    with pytest.raises(HarborTaskError, match="not supported"):
        load_task(write_task(tmp_path / "t", toml=toml))


def test_schema_matches_harbor_legacy_and_resource_fields(tmp_path):
    config = HarborTaskConfig.model_validate(
        {"environment": {"memory": 4096, "storage": 2048, "cpus": 0, "os": "Linux", "image": "stale"}}
    )
    assert config.environment.memory_mb is None and config.environment.storage_mb is None
    assert config.environment.cpus == 0 and config.environment.os == "linux"
    assert config.environment.docker_image is None
    task = load_task(write_task(tmp_path / "t", toml='[environment]\nimage = "stale"'))
    assert task.image == "ubuntu:24.04"


@pytest.fixture
def registry_repo(tmp_path):
    repo = tmp_path / "repo"
    write_task(repo / "tasks" / "a")
    write_task(repo / "tasks" / "b")
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    commit_repo(repo)
    return repo


def commit_repo(repo: Path) -> str:
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@x", "commit", "-qm", "fixture"], check=True
    )
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def registry_dataset(repo: Path, *, name: str = "a", path: str = "tasks/a") -> RegistryDataset:
    return RegistryDataset("ds", "1.0", "", (RegistryTask(name, repo.as_uri(), "HEAD", path),))


@pytest.mark.parametrize("name", ["../../escaped", "/absolute", ".", "..", "a/b", "a\\b", "manifest.toml"])
def test_fetch_rejects_unsafe_destination_names(tmp_path, name):
    with pytest.raises(HubError, match="Unsafe"):
        fetch_dataset(registry_dataset(tmp_path / "unused", name=name), tmp_path / "datasets")
    assert not (tmp_path / "escaped").exists()


@pytest.mark.parametrize("path", ["../outside", "/absolute", "tasks/../../outside", "--skip-checks"])
def test_fetch_rejects_unsafe_source_paths(tmp_path, path):
    with pytest.raises(HubError, match="Unsafe"):
        fetch_dataset(registry_dataset(tmp_path / "unused", path=path), tmp_path / "datasets")


@pytest.mark.parametrize("link_kind", ["absolute", "relative_escape", "source_root", "directory"])
def test_fetch_rejects_source_symlinks_before_copying(tmp_path, registry_repo, monkeypatch, link_kind):
    marker = tmp_path / "marker"
    marker.write_text("private marker")
    task = registry_repo / "tasks" / "a"
    source_path = "tasks/a"
    if link_kind == "source_root":
        (registry_repo / "alias").symlink_to("tasks/a", target_is_directory=True)
        source_path = "alias"
    elif link_kind == "relative_escape":
        (registry_repo / "private").write_text("private marker")
        (task / "tests" / "leak").symlink_to("../../../private")
    elif link_kind == "directory":
        (tmp_path / "private-dir").mkdir()
        (tmp_path / "private-dir" / "marker").write_text("private marker")
        (task / "tests" / "leak").symlink_to(tmp_path / "private-dir", target_is_directory=True)
    else:
        (task / "tests" / "leak").symlink_to(marker)
    commit_repo(registry_repo)
    copied = []
    original = hub_module.shutil.copytree

    def record_copy(*args, **kwargs):
        copied.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(hub_module.shutil, "copytree", record_copy)
    with pytest.raises(HubError, match="symlink"):
        fetch_dataset(registry_dataset(registry_repo, path=source_path), tmp_path / "datasets")
    assert copied == []
    assert marker.read_text() == "private marker"


def test_fetch_preserves_internal_relative_symlinks(tmp_path, registry_repo):
    (registry_repo / "tasks" / "a" / "tests" / "link").symlink_to("test.sh")
    commit_repo(registry_repo)
    folder = fetch_dataset(registry_dataset(registry_repo), tmp_path / "datasets")
    assert (folder / "a" / "tests" / "link").is_symlink()
    assert (folder / "a" / "tests" / "link").read_bytes() == (folder / "a" / "tests" / "test.sh").read_bytes()


@pytest.mark.parametrize("location", ["dataset", "task", "manifest"])
def test_fetch_refuses_symlinked_destination(tmp_path, registry_repo, location):
    root = tmp_path / "datasets"
    root.mkdir()
    outside = write_task(tmp_path / "outside")
    folder = root / "ds-1.0"
    if location == "dataset":
        folder.symlink_to(outside, target_is_directory=True)
    else:
        folder.mkdir()
        if location == "task":
            (folder / "a").symlink_to(outside, target_is_directory=True)
        else:
            (folder / "manifest.toml").symlink_to(outside / "task.toml")
    before = (outside / "task.toml").read_bytes()
    with pytest.raises(HubError):
        fetch_dataset(registry_dataset(registry_repo), root)
    assert (outside / "task.toml").read_bytes() == before
    assert not (outside / "a").exists()


def test_interrupted_fetch_preserves_pins_when_head_moves(tmp_path, registry_repo, monkeypatch):
    old_head = subprocess.check_output(["git", "-C", str(registry_repo), "rev-parse", "HEAD"], text=True).strip()
    dataset = RegistryDataset(
        "ds",
        "1.0",
        "",
        tuple(RegistryTask(name, registry_repo.as_uri(), "HEAD", f"tasks/{name}") for name in ("a", "b")),
    )
    original = hub_module.shutil.copytree

    def interrupt_second(source, *args, **kwargs):
        if source.name == "b":
            raise OSError("interrupted download")
        return original(source, *args, **kwargs)

    monkeypatch.setattr(hub_module.shutil, "copytree", interrupt_second)
    with pytest.raises(OSError, match="interrupted"):
        fetch_dataset(dataset, tmp_path / "datasets")
    monkeypatch.setattr(hub_module.shutil, "copytree", original)
    for name in ("a", "b"):
        (registry_repo / "tasks" / name / "instruction.md").write_text("new contents")
    new_head = commit_repo(registry_repo)
    folder = fetch_dataset(dataset, tmp_path / "datasets")
    pins = tomllib.loads((folder / "manifest.toml").read_text())["tasks"]
    assert pins["a"]["git_commit_id"] == old_head
    assert pins["b"]["git_commit_id"] == new_head
    assert (folder / "a" / "instruction.md").read_text().startswith("Create a file")
    assert (folder / "b" / "instruction.md").read_text() == "new contents"


@pytest.mark.parametrize("name", ["a", "renamed"])
def test_a_staging_directory_an_interrupted_fetch_left_behind_is_not_a_task(
    tmp_path, registry_repo, monkeypatch, name
):
    """The process died after the pin was published but before the staging directory was cleaned up. The
    retry reuses the task, and discovery, which walks the dataset folder recursively, must not take the
    leftover clone for a second copy of the task (the same ID) or for an extra task (another name)."""
    from nemo_gym.tasks.harbor.task import iter_task_folders

    dataset = registry_dataset(registry_repo, name=name)
    real_tempfile = hub_module.tempfile

    class LeakedStaging:
        """A TemporaryDirectory whose process never reached the cleanup."""

        def __init__(self, *, prefix, dir):
            self.name = real_tempfile.mkdtemp(prefix=prefix, dir=dir)

        def __enter__(self):
            return self.name

        def __exit__(self, *exc):
            return None

    publish = hub_module._write_manifest

    def publish_then_die(*args, **kwargs):
        publish(*args, **kwargs)
        raise RuntimeError("killed after publishing the pin")

    class StagingTempfile:
        """`tempfile` as the fetch sees it: everything real, except the staging directory, which leaks."""

        TemporaryDirectory = LeakedStaging

        def __getattr__(self, name):
            return getattr(real_tempfile, name)

    monkeypatch.setattr(hub_module, "tempfile", StagingTempfile())
    monkeypatch.setattr(hub_module, "_write_manifest", publish_then_die)
    with pytest.raises(RuntimeError, match="killed"):
        fetch_dataset(dataset, tmp_path / "datasets")
    monkeypatch.undo()

    folder = tmp_path / "datasets" / dataset.folder_name
    (leftover,) = [path for path in folder.iterdir() if path.is_dir() and path.name != name]
    assert leftover.name.startswith(".nemo-gym-harbor-")
    assert (leftover / "repo" / "tasks" / "a" / "task.toml").is_file()
    manifest = (folder / "manifest.toml").read_text()

    assert fetch_dataset(dataset, tmp_path / "datasets") == folder

    # The published task was reused as it is, and the leftover is neither a duplicate nor an extra task.
    assert (folder / "manifest.toml").read_text() == manifest and leftover.is_dir()
    assert iter_task_folders(folder) == [folder / name]
    assert [task.task_id for task in discover_tasks(folder)] == [name]


def test_a_legacy_unhidden_staging_directory_is_refused_as_a_duplicate_task(tmp_path):
    """Review (round 14): a `main` build from before this change staged the fetch's clone under an unhidden
    `nemo-gym-harbor-*` name, which first-level discovery never saw. The recursive search does see such a leftover,
    and refuses it loudly (the clone holds the task that was being staged, so its ID is a duplicate) rather than
    running a second copy. The name is not special-cased (a user folder may start that way); the README says to
    delete those directories once."""
    from nemo_gym.tasks.harbor.task import iter_task_folders

    dataset = tmp_path / "datasets" / "ds"
    write_task(dataset / "a")
    write_task(dataset / "nemo-gym-harbor-k3x9q" / "repo" / "tasks" / "a")
    with pytest.raises(HarborTaskError, match=r"Duplicate Harbor task ID 'a'") as excinfo:
        iter_task_folders(dataset)
    assert "nemo-gym-harbor-k3x9q" in str(excinfo.value)
    # Prepare discovers through the same walk, so nothing runs or scores with the leftover in place.
    with pytest.raises(HarborTaskError, match=r"Duplicate Harbor task ID 'a'"):
        discover_tasks(dataset)


def test_existing_unpinned_task_is_never_relabeled(tmp_path, registry_repo):
    existing = write_task(tmp_path / "datasets" / "ds-1.0" / "a")
    (existing / "instruction.md").write_text("local contents")
    with pytest.raises(HubError, match="pin"):
        fetch_dataset(registry_dataset(registry_repo), tmp_path / "datasets")
    assert (existing / "instruction.md").read_text() == "local contents"
    assert not (existing.parent / "manifest.toml").exists()


@pytest.mark.usefixtures("fake_registry")  # prepare records base images; never reach Docker Hub
def test_registry_versions_keep_distinct_rows_routes_and_outputs(tmp_path, monkeypatch):
    import nemo_gym.tasks.harbor.cli as harbor_cli

    for version in ("1.0", "2.0"):
        write_task(tmp_path / f"ds-{version}" / "a")
        # A fetch always leaves the folder's manifest behind; prepare reads it back for the fetch report.
        manifest_module.Manifest.build(tmp_path / f"ds-{version}", source="harbor-registry").write(
            tmp_path / f"ds-{version}"
        )
    monkeypatch.setattr(harbor_cli, "fetch_ref", lambda target, **kw: tmp_path / f"ds-{target.split('@')[1]}")
    first = prepare_target("harbor:ds@1.0", output_root=tmp_path / "out")
    first_rows = first.rows_path.read_bytes()
    second = prepare_target("harbor:ds@2.0", output_root=tmp_path / "out")
    assert first.rows_path != second.rows_path and first.taskset != second.taskset
    assert first.rows_path.read_bytes() == first_rows
    agent = resolve_agent("hermes_agent")
    configs = [build_run(prepared, agent, sandbox=None, overrides=[]) for prepared in (first, second)]
    assert configs[0][0] != configs[1][0]
    for prepared, (path, tokens) in zip((first, second), configs):
        assert json.loads(prepared.rows_path.read_text())["task_id"]["taskset"] == prepared.taskset
        assert list(yaml.safe_load(path.read_text())["environment_server_routes"]) == [prepared.taskset]
        assert yaml.safe_load(path.read_text())["output_jsonl_fpath"] == str(prepared.output_dir / "rollouts.jsonl")


@pytest.mark.usefixtures("fake_registry")  # prepare records base images; never reach Docker Hub
def test_config_overlays_win_after_real_loading_and_inheritance(tmp_path):
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from omegaconf import OmegaConf

    from nemo_gym.cli.main import _merge_config_paths
    from nemo_gym.global_config import GlobalConfigDictParser

    task = write_task(tmp_path / "ds" / "a")
    prepared = prepare_target(str(task.parent), output_root=tmp_path / "out")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        "hermes_agent:\n  responses_api_agents:\n    hermes_agent:\n      max_turns: 71\nharbor_environment_server:\n  environment_servers:\n    harbor:\n      default_episode_timeout_seconds: 1234\n"
    )
    _, tokens = build_run(
        prepared, resolve_agent("hermes_agent"), sandbox=None, overrides=[f"+config_paths=[{overlay}]"]
    )
    paths_token = next(t for t in _merge_config_paths(tokens) if t.startswith("+config_paths="))
    paths = OverridesParser.create().parse_override(paths_token).value()
    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths(paths)
    config = OmegaConf.merge(*configs)
    parser._recursively_swap_keys(config)
    assert config.harbor_ds_agent.responses_api_agents.hermes_agent.max_turns == 71
    environment = config.harbor_ds_environment.environment_servers.harbor
    assert environment.default_episode_timeout_seconds == 1234
    # The generated instance keeps the shipped defaults it does not override.
    assert environment.cleanup_timeout_seconds == 180 and environment.sandbox_provider == "sandbox"


@pytest.mark.parametrize("skip", [False, True])
def test_partial_eval_requires_explicit_opt_in_and_records_reasons(tmp_path, monkeypatch, capsys, skip):
    import sys

    import nemo_gym.cli.main as main_module

    write_task(tmp_path / "ds" / "good")
    write_task(tmp_path / "ds" / "bad", toml="bad toml")
    # The run resolves this host for the servers' addresses; a host that resolves to 127.0.1.1 (Ubuntu /etc/hosts)
    # would be refused as unreachable from a sandbox, so pin a routable address as the other CLI tests do.
    import nemo_gym.global_config as global_config

    monkeypatch.setattr(global_config, "gethostbyname", lambda _: "192.0.2.20")
    monkeypatch.chdir(tmp_path)
    dispatched = []
    monkeypatch.setattr(main_module, "dispatch", lambda *args: dispatched.append(args))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gym",
            "eval",
            "run",
            str(tmp_path / "ds"),
            "--agent",
            "hermes_agent",
            "--model-type",
            "vllm_model",
            "+policy_model_name=test-model",
            "+policy_base_url=http://localhost:1",
            "+policy_api_key=none",
            *(["--skip-unloadable"] if skip else []),
        ],
    )
    if skip:
        main_module.main()
        assert len(dispatched) == 1
    else:
        with pytest.raises(SystemExit) as exc:
            main_module.main()
        assert exc.value.code != 0
        assert not dispatched
    skipped = json.loads((tmp_path / "results" / "harbor" / "ds" / "skipped.json").read_text())
    assert list(skipped) == ["bad"] and "task.toml" in skipped["bad"]
    if not skip:
        # The refusal lists every task left out and why, not only the count.
        assert "1 task(s) could not load" in capsys.readouterr().err


@pytest.mark.parametrize("skip", [False, True])
def test_partial_eval_names_every_task_whose_compose_file_does_not_resolve(tmp_path, monkeypatch, capsys, skip):
    import sys

    import nemo_gym.cli.main as main_module

    monkeypatch.delenv("API_KEY", raising=False)
    write_task(tmp_path / "ds" / "good")
    services = {
        "bad-required": "environment: ['API_KEY=${API_KEY:?API_KEY is needed}']",
        "bad-env-file": "env_file: missing.env",
    }
    for name, line in services.items():
        write_task(tmp_path / "ds" / name)
        (tmp_path / "ds" / name / "environment" / "docker-compose.yaml").write_text(
            f"services:\n  main:\n    {line}\n"
        )
    # The run resolves this host for the servers' addresses; a host that resolves to 127.0.1.1 (Ubuntu /etc/hosts)
    # would be refused as unreachable from a sandbox, so pin a routable address as the other CLI tests do.
    import nemo_gym.global_config as global_config

    monkeypatch.setattr(global_config, "gethostbyname", lambda _: "192.0.2.20")
    monkeypatch.chdir(tmp_path)
    dispatched = []
    monkeypatch.setattr(main_module, "dispatch", lambda *args: dispatched.append(args))
    argv = ["gym", "eval", "run", str(tmp_path / "ds"), "--agent", "hermes_agent", "--model-type", "vllm_model"]
    argv += ["+policy_model_name=test-model", "+policy_base_url=http://localhost:1", "+policy_api_key=none"]
    monkeypatch.setattr(sys, "argv", argv + (["--skip-unloadable"] if skip else []))
    if skip:
        main_module.main()
        assert len(dispatched) == 1
    else:
        with pytest.raises(SystemExit) as exc:
            main_module.main()
        assert exc.value.code != 0 and not dispatched
        err = capsys.readouterr().err
        assert "2 task(s) could not load" in err
        assert "\n  bad-required: docker-compose.yaml: ${API_KEY} is required: API_KEY is needed\n" in err
        assert "\n  bad-env-file: docker-compose.yaml: " in err and "missing.env" in err
        assert "Use --skip-unloadable" in err
    skipped = json.loads((tmp_path / "results" / "harbor" / "ds" / "skipped.json").read_text())
    assert sorted(skipped) == ["bad-env-file", "bad-required"]
    assert skipped["bad-required"].endswith("${API_KEY} is required: API_KEY is needed")


def test_eval_run_resolves_its_own_tokens_over_a_config_this_process_resolved_before(tmp_path, monkeypatch):
    """The resume check reads the configuration the collector dispatches on, run config included. A global
    config dict this process resolved earlier, for other tokens, must not stand in for it."""
    import sys

    from omegaconf import DictConfig

    import nemo_gym.cli.main as main_module
    import nemo_gym.global_config as global_config

    write_task(tmp_path / "ds" / "good")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", DictConfig({"a": 2}))
    # A fixed address: use_absolute_ip resolves the host name, which a stock Ubuntu box answers with 127.0.1.1.
    monkeypatch.setattr(global_config, "gethostbyname", lambda _: "192.0.2.20")
    dispatched = []
    monkeypatch.setattr(main_module, "dispatch", lambda *args: dispatched.append(args))
    argv = ["gym", "eval", "run", str(tmp_path / "ds"), "--agent", "hermes_agent", "--model-type", "vllm_model"]
    argv += ["+policy_model_name=test-model", "+policy_base_url=http://localhost:1", "+policy_api_key=none"]
    monkeypatch.setattr(sys, "argv", argv)
    main_module.main()
    assert len(dispatched) == 1
    resolved = global_config.get_global_config_dict()
    assert "a" not in resolved
    assert Path(resolved["output_jsonl_fpath"]).parts[-4:] == ("results", "harbor", "ds", "rollouts.jsonl")


def test_manifest_publication_failure_keeps_previous_pins(tmp_path, registry_repo, monkeypatch):
    folder = fetch_dataset(registry_dataset(registry_repo), tmp_path / "datasets")
    before = (folder / "manifest.toml").read_bytes()
    dataset = RegistryDataset(
        "ds",
        "1.0",
        "",
        tuple(RegistryTask(name, registry_repo.as_uri(), "HEAD", f"tasks/{name}") for name in ("a", "b")),
    )
    original = manifest_module.os.replace

    def fail_publication(path, target):
        if Path(target) == folder / "manifest.toml":
            raise OSError("interrupted manifest publication")
        return original(path, target)

    monkeypatch.setattr(manifest_module.os, "replace", fail_publication)
    with pytest.raises(OSError, match="interrupted manifest"):
        fetch_dataset(dataset, tmp_path / "datasets")
    assert (folder / "manifest.toml").read_bytes() == before
    assert (folder / "b" / "task.toml").is_file()
    assert not list(folder.glob(".manifest-*"))
    monkeypatch.setattr(manifest_module.os, "replace", original)
    (registry_repo / "tasks" / "b" / "instruction.md").write_text("new contents")
    commit_repo(registry_repo)
    with pytest.raises(HubError, match="no trustworthy source pin"):
        fetch_dataset(dataset, tmp_path / "datasets")
    assert (folder / "manifest.toml").read_bytes() == before
    assert (folder / "b" / "instruction.md").read_text().startswith("Create a file")


def test_all_unloadable_tasks_still_record_reasons(tmp_path):
    write_task(tmp_path / "ds" / "bad", toml="bad toml")
    with pytest.raises(HarborTaskError, match="No task under"):
        prepare_target(str(tmp_path / "ds"), output_root=tmp_path / "out")
    skipped = json.loads((tmp_path / "out" / "ds" / "skipped.json").read_text())
    assert list(skipped) == ["bad"]
    assert "task.toml" in skipped["bad"]


class TestDatasetConfig:
    """`dataset.toml`'s [gym] table beside the task folders shapes the effective task.toml."""

    def test_absent_or_foreign_file_leaves_the_task_alone(self, tmp_path):
        folder = tmp_path / "ds"
        plain = load_task(write_task(folder / "hello"))
        (folder / "dataset.toml").write_text('[dataset]\nname = "x"\nversion = "1.0"\n')
        again = load_task(folder / "hello")
        assert again.config == plain.config and again.digest == plain.digest

    def test_per_task_override_merges_env_and_replaces_fields(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(
            folder / "hello", toml=HELLO_TOML.replace("[environment]", '[environment]\nenv = { A = "1", B = "2" }')
        )
        write_task(folder / "other")
        (folder / "dataset.toml").write_text(
            '[gym.tasks."hello".environment]\nenv = { B = "override", CIRCLE_NODE_TOTAL = "3" }\ngpu_types = ["H100"]\n'
        )
        hello, other = load_task(folder / "hello"), load_task(folder / "other")
        assert hello.env == {"A": "1", "B": "override", "CIRCLE_NODE_TOTAL": "3"}
        assert hello.config.environment.gpu_types == ["H100"]
        assert other.env == {} and other.config.environment.gpu_types is None
        # The digest pins the task folder's content plus the dataset slice applied to it.
        from nemo_gym.tasks.harbor.dataset_config import read_dataset_config

        dataset = read_dataset_config(folder)
        assert hello.digest == task_digest(content_hash(folder / "hello"), dataset, "hello")
        assert hello.digest != content_hash(folder / "hello")

    def test_editing_dataset_toml_changes_the_digest_and_removing_it_restores_the_folder_hash(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "hello")
        (folder / "dataset.toml").write_text("[gym.defaults]\ntimeout_multiplier = 2.0\n")
        scaled = load_task(folder / "hello")
        (folder / "dataset.toml").write_text("[gym.defaults]\ntimeout_multiplier = 3.0\n")
        rescaled = load_task(folder / "hello")
        assert scaled.digest != rescaled.digest != content_hash(folder / "hello")
        (folder / "dataset.toml").unlink()
        assert load_task(folder / "hello").digest == content_hash(folder / "hello")

    def test_legacy_override_spellings_replace_the_tasks_current_fields(self, tmp_path):
        from nemo_gym.tasks.harbor.dataset_config import apply_dataset_config, read_dataset_config

        folder = tmp_path / "ds"
        write_task(folder / "hello")
        (folder / "dataset.toml").write_text(
            '[gym.tasks."hello".environment]\nmemory = "4G"\nstorage = "20G"\nallow_internet = false\n'
        )
        config = HarborTaskConfig.model_validate(tomllib.loads(HELLO_TOML))
        assert config.environment.memory_mb == 2048 and config.environment.network_mode == "public"
        effective = apply_dataset_config(config, "hello", read_dataset_config(folder))
        # `memory` replaces memory_mb instead of conflicting with it; `allow_internet` replaces network_mode.
        assert (effective.environment.memory_mb, effective.environment.storage_mb) == (4096, 20480)
        assert effective.environment.network_mode == "no-network"
        # The loader then reports the unsupported policy as it would for a task.toml that set it.
        with pytest.raises(HarborTaskError, match="network restrictions"):
            load_task(folder / "hello")

    @pytest.mark.parametrize(
        "text",
        ['[gym.tasks."hello".environment]\nenv = 3\n', '[gym.tasks."hello".environment]\nenv = { A = 3 }\n'],
    )
    def test_a_malformed_env_override_is_a_task_error_naming_the_file_and_the_table(self, tmp_path, text):
        folder = tmp_path / "ds"
        write_task(folder / "hello")
        (folder / "dataset.toml").write_text(text)
        with pytest.raises(HarborTaskError) as excinfo:
            load_task(folder / "hello")
        message = str(excinfo.value)
        assert str(folder / "dataset.toml") in message
        assert '[gym.tasks."hello".environment].env must be a table of strings' in message

    @pytest.mark.parametrize(
        "text",
        ['[gym.tasks."bad".verifier]\nenv = 1\n', '[gym.tasks."bad".verifier]\nenv = { A = 3 }\n'],
    )
    def test_a_malformed_verifier_env_override_skips_the_task_by_name_instead_of_aborting_the_dataset(
        self, tmp_path, text
    ):
        """`env = 1` under the verifier table used to reach `dict(1)` in the merge, a TypeError that neither
        `load_task` nor the base-image scan catches: one malformed override aborted the whole dataset's preparation.
        It is refused like the environment table's `env`, so the task is skipped with its table named."""
        from nemo_gym.tasks.harbor.image_configs import dockerfile_base_images

        folder = tmp_path / "ds"
        write_task(folder / "a")
        write_task(folder / "bad")
        (folder / "dataset.toml").write_text(text)
        with pytest.raises(HarborTaskError) as excinfo:
            load_task(folder / "bad")
        assert '[gym.tasks."bad".verifier].env must be a table of strings' in str(excinfo.value)
        skipped: dict[str, HarborTaskError] = {}
        assert [t.task_id for t in discover_tasks(folder, skipped=skipped)] == ["a"]
        assert list(skipped) == ["bad"] and "must be a table of strings" in str(skipped["bad"])
        # The base-image scan skips the task too, instead of aborting the scan of the dataset.
        assert dockerfile_base_images(folder) == {"ubuntu:24.04": ["a"]}

    def test_defaults_scale_timeouts_and_resources(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "hello")
        (folder / "dataset.toml").write_text("[gym.defaults]\ntimeout_multiplier = 2.0\nresource_multiplier = 1.5\n")
        task = load_task(folder / "hello")
        assert (task.config.agent.timeout_sec, task.config.verifier.timeout_sec) == (240.0, 240.0)
        environment = task.config.environment
        assert (environment.cpus, environment.memory_mb, environment.storage_mb, environment.gpus) == (
            2,
            3072,
            15360,
            0,
        )

    def test_defaults_scale_a_steps_own_timeouts_once(self, tmp_path):
        """Review (round 32): the multiplier scaled the task's `[agent]` and `[verifier]` timeouts only, so a step
        with its own `timeout_sec` kept the unscaled value in the row's metadata, the turn's budgets and the
        episode bound, while a step inheriting the task's agent budget was scaled. A step's own values scale once;
        a budget a step does not name stays unnamed (`None`), inherited through the config's step accessors, which
        read the task's scaled agent budget and the scaled step verifier default (the effective config carries the
        multiplier for it), so nothing is double-scaled and the loader's hint about steps graded within the step
        default still sees which steps named none."""
        folder = tmp_path / "ds"
        steps = (
            '\n[[steps]]\nname = "a"\n[steps.agent]\ntimeout_sec = 30.0\n[steps.verifier]\ntimeout_sec = 40.0\n'
            '\n[[steps]]\nname = "b"\n'
        )
        write_task(folder / "hello", toml=HELLO_TOML + steps)
        for name in ("a", "b"):
            (folder / "hello" / "steps" / name).mkdir(parents=True)
            (folder / "hello" / "steps" / name / "instruction.md").write_text(f"Step {name}\n")
        (folder / "dataset.toml").write_text("[gym.defaults]\ntimeout_multiplier = 2.0\n")
        task = load_task(folder / "hello")
        a, b = task.config.steps
        assert (a.agent.timeout_sec, a.verifier.timeout_sec) == (60.0, 80.0)
        assert (task.config.step_agent_timeout_sec(a), task.config.step_verifier_timeout_sec(a)) == (60.0, 80.0)
        assert b.agent.timeout_sec is None and task.config.step_agent_timeout_sec(b) == 240.0
        assert b.verifier.timeout_sec is None and task.config._timeout_multiplier == 2.0
        row = materialize_task(task, "ds").model_dump(mode="json")
        turns = row["task_input"]["turns"]
        assert [turn["responses_create_params"]["metadata"]["agent_timeout_sec"] for turn in turns] == [
            "60.0",
            "240.0",
        ]

    @pytest.mark.parametrize(
        "text,match",
        [
            ('[gym.tasks."hello".environment]\nimage_tag = "x"\n', "image_tag"),
            ('[gym.tasks."hello"]\ncommands = ["rm -rf /"]\n', "commands"),
            ("[gym.defaults]\ntimeout_multiplier = 0\n", "timeout_multiplier"),
            ("[gym\n", "dataset.toml"),
        ],
    )
    def test_bad_dataset_files_are_rejected_with_the_file_named(self, tmp_path, text, match):
        folder = tmp_path / "ds"
        write_task(folder / "hello")
        (folder / "dataset.toml").write_text(text)
        with pytest.raises(HarborTaskError, match=match):
            load_task(folder / "hello")


def test_harbor_task_data_schema_names_the_digest_key(tmp_path):
    """The schema module may import only pydantic, so the key is spelled out; keep it equal to DIGEST_KEY."""
    from pydantic import ValidationError

    from environment_servers.harbor.task_data import TaskData

    assert TaskData.model_fields["ng_digest"].alias == DIGEST_KEY
    data = TaskData.model_validate({DIGEST_KEY: "abc", "other": 1})
    assert data.ng_digest == "abc" and data.model_dump(by_alias=True)[DIGEST_KEY] == "abc"
    # The digest is required: a row without it is refused by the server (422), so the schema refuses it too.
    with pytest.raises(ValidationError):
        TaskData.model_validate({"other": 1})
    # A materialized row's task_data validates as is, so the alias stays aligned with what the loader writes.
    task = load_task(write_task(tmp_path / "hello-world"))
    row = materialize_task(task, "hello").model_dump(mode="json", exclude_unset=True)
    assert TaskData.model_validate(row["task_input"]["task_data"]).ng_digest == task.digest


@pytest.mark.usefixtures("fake_registry")  # prepare records base images; never reach Docker Hub
@pytest.mark.parametrize(
    ("taskset", "agent_name"),
    [
        ("ds", "hermes_agent"),
        ("terminal-bench-2.0", "hermes_agent"),
        ("trial=1", "hermes_agent"),
        ("中文=1", "hermes_agent"),
        # Its template leaves `resources_server.name` unset; the generated instance must still parse as
        # bound to nothing (not as a composition source, which the full parse would reject).
        ("ds", "terminus_2_sandboxed_agent"),
    ],
)
def test_generated_run_passes_full_config_validation(tmp_path, monkeypatch, taskset, agent_name):
    import sys

    import nemo_gym.global_config as global_config
    from nemo_gym import PARENT_DIR
    from nemo_gym.cli.main import _merge_config_paths
    from nemo_gym.global_config import GlobalConfigDictParser, get_global_config_dict
    from nemo_gym.rollout_collection import E2ERolloutCollectionConfig
    from nemo_gym.tasks.harbor.materialize import agent_instance, environment_server_instance

    write_task(tmp_path / taskset / "a")
    prepared = prepare_target(str(tmp_path / taskset), output_root=tmp_path / "out")
    _, tokens = build_run(
        prepared,
        resolve_agent(agent_name),
        sandbox="opensandbox",
        overrides=[
            f"+config_paths=[{PARENT_DIR / 'responses_api_models/openai_model/configs/openai_model.yaml'}]",
            "+policy_model_name=test-model",
            "+policy_base_url=http://127.0.0.1:1/v1",
            "+policy_api_key=unused",
        ],
    )
    monkeypatch.setattr(sys, "argv", ["e2e", *_merge_config_paths(tokens)])
    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
    monkeypatch.setattr(global_config, "gethostbyname", lambda _: "192.0.2.20")
    config = get_global_config_dict()
    servers = GlobalConfigDictParser().filter_for_server_instance_configs(config)
    environment = next(server for server in servers if server.name == environment_server_instance(taskset))
    assert environment.get_inner_run_server_config().agent_server["name"] == agent_instance(taskset)
    agent = next(server for server in servers if server.name == agent_instance(taskset))
    assert agent.datasets[0].name == taskset
    assert agent.get_inner_run_server_config().resources_server is None
    assert agent_instance(taskset) in config and agent_name not in config
    run = E2ERolloutCollectionConfig.model_validate(config)
    assert run.environment_routing_mode == "taskset"
    assert taskset in run.environment_server_routes


def test_interrupted_cache_repair_preserves_pending_pins(tmp_path, registry_repo, monkeypatch):
    import shutil

    write_task(registry_repo / "tasks" / "c")
    old_head = commit_repo(registry_repo)
    dataset = RegistryDataset(
        "ds",
        "1.0",
        "",
        tuple(RegistryTask(name, registry_repo.as_uri(), "HEAD", f"tasks/{name}") for name in ("a", "b", "c")),
    )
    folder = fetch_dataset(dataset, tmp_path / "datasets")
    for name in ("b", "c"):
        shutil.rmtree(folder / name)
        (registry_repo / "tasks" / name / "instruction.md").write_text("new upstream instruction")
    assert commit_repo(registry_repo) != old_head
    original = hub_module.shutil.copytree

    def interrupt_last(source, *args, **kwargs):
        if source.name == "c":
            raise OSError("interrupted cache repair")
        return original(source, *args, **kwargs)

    monkeypatch.setattr(hub_module.shutil, "copytree", interrupt_last)
    with pytest.raises(OSError, match="interrupted cache repair"):
        fetch_dataset(dataset, tmp_path / "datasets")
    pins = tomllib.loads((folder / "manifest.toml").read_text())["tasks"]
    assert pins["c"]["git_commit_id"] == old_head
    monkeypatch.setattr(hub_module.shutil, "copytree", original)
    fetch_dataset(dataset, tmp_path / "datasets")
    pins = tomllib.loads((folder / "manifest.toml").read_text())["tasks"]
    assert {pin["git_commit_id"] for pin in pins.values()} == {old_head}
    assert (folder / "c" / "instruction.md").read_text().startswith("Create a file")


@pytest.mark.usefixtures("fake_registry")  # prepare records base images; never reach Docker Hub
@pytest.mark.parametrize("external", [False, True])
def test_symlinked_child_tasks_are_reported_without_duplicate_rows(tmp_path, external):
    folder = tmp_path / "ds"
    task = write_task(folder / "a")
    target = write_task(tmp_path / "elsewhere" / "renamed") if external else task
    (folder / "b").symlink_to(target, target_is_directory=True)
    with pytest.raises(HarborTaskError, match="symlink"):
        discover_tasks(folder)
    with pytest.raises(HarborTaskError, match="symlink"):
        load_task(folder / "b")
    prepared = prepare_target(str(folder), output_root=tmp_path / "out")
    assert [row["task_id"]["task_id"] for row in map(json.loads, prepared.rows_path.read_text().splitlines())] == ["a"]
    assert "symlink" in prepared.skipped["b"]
    assert "symlink" in json.loads((prepared.output_dir / "skipped.json").read_text())["b"]


def test_duplicate_task_ids_are_rejected_before_rows_or_mapping_are_written(tmp_path):
    from nemo_gym.tasks.harbor.materialize import taskset_mapping

    task = load_task(write_task(tmp_path / "ds" / "a"))
    rows = tmp_path / "tasks.jsonl"
    rows.write_text("previous valid rows\n")
    with pytest.raises(HarborTaskError, match="Duplicate.*a"):
        write_rows([task, task], "ds", rows)
    assert rows.read_text() == "previous valid rows\n"
    with pytest.raises(HarborTaskError, match="Duplicate.*a"):
        taskset_mapping([task, task], task.path.parent)


def test_generated_instance_names_are_override_safe_and_collision_free():
    from hydra.core.override_parser.overrides_parser import OverridesParser

    from nemo_gym.tasks.harbor.materialize import agent_instance, environment_server_instance

    tasksets = ["terminal-bench-2.0", "terminal-bench-2_0", "terminal-bench-2_2e0", "ds-1.0", "ds-2.0", "中文", "a b"]
    parser = OverridesParser.create()
    for instance in (agent_instance, environment_server_instance):
        names = [instance(taskset) for taskset in tasksets]
        assert len(set(names)) == len(tasksets)
        for name in names:
            assert "." not in name
            override = parser.parse_override(f"++{name}.setting=true")
            assert override.key_or_group == f"{name}.setting"


@pytest.mark.usefixtures("fake_registry")  # prepare records base images; never reach Docker Hub
def test_generated_agent_can_be_overridden_for_versioned_taskset(tmp_path, monkeypatch):
    import sys

    import nemo_gym.global_config as global_config
    from nemo_gym import PARENT_DIR
    from nemo_gym.cli.main import _merge_config_paths
    from nemo_gym.global_config import get_global_config_dict
    from nemo_gym.tasks.harbor.materialize import agent_instance

    monkeypatch.setattr(global_config, "gethostbyname", lambda _: "192.0.2.20")
    folder = tmp_path / "ds-2.0"
    write_task(folder / "a")
    prepared = prepare_target(str(folder), output_root=tmp_path / "out")
    name = agent_instance(prepared.taskset)
    _, tokens = build_run(
        prepared,
        resolve_agent("hermes_agent"),
        sandbox=None,
        overrides=[
            f"+config_paths=[{PARENT_DIR / 'responses_api_models/openai_model/configs/openai_model.yaml'}]",
            "+policy_model_name=test-model",
            "+policy_base_url=http://127.0.0.1:1/v1",
            "+policy_api_key=unused",
            f"++{name}.responses_api_agents.hermes_agent.chat_template_kwargs_enabled=true",
        ],
    )
    monkeypatch.setattr(sys, "argv", ["e2e", *_merge_config_paths(tokens)])
    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
    config = get_global_config_dict()
    assert config[name].responses_api_agents.hermes_agent.chat_template_kwargs_enabled is True
    assert list(config.environment_server_routes) == ["ds-2.0"]


@pytest.mark.parametrize(
    ("target", "flags", "message"),
    [
        ("missing-folder", ["--agent", "hermes_agent"], "is not a directory"),
        ("harbor:not-present", ["--agent", "hermes_agent"], "not in the Harbor registry"),
        ("harbor:hello-world", [], "needs `--agent"),
        ("harbor:hello-world", ["--no-serve", "--agent", "hermes_agent"], "drop --no-serve"),
    ],
)
def test_expected_harbor_cli_errors_are_concise(tmp_path, monkeypatch, capsys, target, flags, message):
    import sys

    from nemo_gym.cli.main import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hub_module, "load_registry", lambda *args, **kwargs: [])
    monkeypatch.setattr(sys, "argv", ["gym", "eval", "run", target, *flags])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert message in error and "Traceback" not in error


@pytest.mark.usefixtures("fake_registry")
@pytest.mark.parametrize("command", [["eval", "run"], ["dataset", "validate"]])
def test_a_compose_volume_problem_in_a_task_folder_exits_cleanly(tmp_path, monkeypatch, capsys, command):
    # Review: the planner's ImageConfigError (not a HarborTaskError) reached the user as a traceback at prepare.
    import sys

    from nemo_gym.cli.main import main

    folder = tmp_path / "ds"
    write_task(folder / "aw")
    (folder / "aw" / "environment" / "docker-compose.yaml").write_text(
        "services:\n  api:\n    image: nginx:1.27\n    volumes: [./missing.json:/app/seed.json]\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEMO_GYM_DATASETS_DIR", str(tmp_path / "datasets"))
    flags = ["--agent", "hermes_agent"] if command[0] == "eval" else []
    monkeypatch.setattr(sys, "argv", ["gym", *command, str(folder), *flags])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "aw: environment/docker-compose.yaml: service 'api' binds './missing.json'" in error
    assert "which the task folder does not contain" in error
    assert "Traceback" not in error


@pytest.mark.parametrize(
    "flags", [["--agent", "not-an-agent"], ["--agent", "hermes_agent", "--sandbox", "not-a-sandbox"]]
)
def test_harbor_selectors_fail_before_any_preparation(monkeypatch, capsys, flags):
    import sys

    import nemo_gym.tasks.harbor.cli as harbor_cli
    from nemo_gym.cli.main import main

    def unexpected_preparation(*args, **kwargs):
        pytest.fail("Invalid selectors must be rejected before preparing or fetching tasks")

    monkeypatch.setattr(harbor_cli, "prepare_target", unexpected_preparation)
    monkeypatch.setattr(sys, "argv", ["gym", "eval", "run", "harbor:hello-world", *flags])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "No " in capsys.readouterr().err


def test_unexpected_harbor_programming_errors_remain_visible(monkeypatch):
    import sys

    import nemo_gym.tasks.harbor.cli as harbor_cli
    from nemo_gym.cli.main import main

    def bug(*args, **kwargs):
        raise TypeError("unexpected implementation bug")

    monkeypatch.setattr(harbor_cli, "prepare_target", bug)
    monkeypatch.setattr(sys, "argv", ["gym", "eval", "run", "harbor:hello-world", "--agent", "hermes_agent"])
    with pytest.raises(TypeError, match="unexpected implementation bug"):
        main()


@pytest.mark.parametrize(
    "payload",
    [
        b"[{",
        b"\xff",
        b"{}",
        b"[null]",
        b"[{}]",
        b'[{"name": "ds", "tasks": [{}]}]',
        b'[{"name": "ds", "tasks": null}]',
        b'[{"name": "ds", "tasks": [null]}]',
        b'[{"name": 1}]',
        b'[{"name": "ds", "tasks": [{"name": "a", "git_url": null}]}]',
    ],
)
def test_damaged_registry_cache_reports_recovery_at_cli(tmp_path, monkeypatch, capsys, payload):
    import sys

    from nemo_gym.cli.main import main

    root = tmp_path / "datasets"
    cache = root / ".harbor" / "registry.json"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(payload)
    monkeypatch.setenv("NEMO_GYM_DATASETS_DIR", str(root))
    monkeypatch.setattr(sys, "argv", ["gym", "eval", "run", "harbor:ds", "--agent", "hermes_agent"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    message = capsys.readouterr().err
    assert str(cache) in message
    assert "Remove this file and retry" in message
    assert "Traceback" not in message
    assert cache.read_bytes() == payload


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("payload", [b"[{", b'[{"tasks": []}]'])
def test_invalid_registry_download_does_not_publish(tmp_path, monkeypatch, existing, payload):
    from io import BytesIO

    cache = tmp_path / "registry.json"
    previous = b'[{"name": "previous"}]'
    if existing:
        cache.write_bytes(previous)
    monkeypatch.setattr(hub_module.urllib.request, "urlopen", lambda *args, **kwargs: BytesIO(payload))
    with pytest.raises(HubError, match="registry"):
        load_registry(tmp_path, refresh=True)
    assert cache.read_bytes() == previous if existing else not cache.exists()
    assert not list(tmp_path.glob(".registry-*"))


@pytest.mark.parametrize("existing", [False, True])
def test_registry_publication_is_atomic_and_recovers(tmp_path, monkeypatch, existing):
    from io import BytesIO

    cache = tmp_path / "registry.json"
    previous = b'[{"name": "previous"}]'
    payload = b'[{"name": "replacement"}]'
    if existing:
        cache.write_bytes(previous)
    monkeypatch.setattr(hub_module.urllib.request, "urlopen", lambda *args, **kwargs: BytesIO(payload))
    replace = Path.replace
    attempted = []

    def interrupted(path, target):
        if Path(target) == cache:
            assert path.parent == cache.parent
            assert path.read_bytes() == payload
            assert cache.read_bytes() == previous if existing else not cache.exists()
            attempted.append(path)
            raise OSError("interrupted registry publication")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", interrupted)
    with pytest.raises(HubError, match="interrupted registry publication"):
        load_registry(tmp_path, refresh=True)
    assert attempted
    assert cache.read_bytes() == previous if existing else not cache.exists()
    assert not list(tmp_path.glob(".registry-*"))
    monkeypatch.setattr(Path, "replace", replace)
    assert load_registry(tmp_path, refresh=True)[0].name == "replacement"
    assert cache.read_bytes() == payload
    assert not list(tmp_path.glob(".registry-*"))


@pytest.mark.parametrize("existing", [False, True])
def test_interrupted_registry_cache_write_preserves_previous_file(tmp_path, monkeypatch, existing):
    from io import BytesIO

    cache = tmp_path / "registry.json"
    previous = b'[{"name": "previous"}]'
    payload = b'[{"name": "replacement"}]'
    if existing:
        cache.write_bytes(previous)
    monkeypatch.setattr(hub_module.urllib.request, "urlopen", lambda *args, **kwargs: BytesIO(payload))
    create = hub_module.tempfile.NamedTemporaryFile
    attempted = []

    def interrupted_file(*args, **kwargs):
        stream = create(*args, **kwargs)
        write = stream.write

        def partial_write(data):
            write(data[:5])
            stream.flush()
            attempted.append(Path(stream.name))
            raise OSError("interrupted registry write")

        stream.write = partial_write
        return stream

    monkeypatch.setattr(hub_module.tempfile, "NamedTemporaryFile", interrupted_file)
    with pytest.raises(HubError, match="interrupted registry write"):
        load_registry(tmp_path, refresh=True)
    assert attempted
    assert cache.read_bytes() == previous if existing else not cache.exists()
    assert not list(tmp_path.glob(".registry-*"))


def test_eval_run_resume_refuses_a_changed_selection_or_digest(tmp_path, monkeypatch, capsys):
    """`--resume` resumes the collector's cached inputs, so a narrower selection or an edited task is refused first.
    The guard follows the collector's own resume condition: its effective `resume_from_cache` (flag or config
    file), both cache files present, and the rows it would run after `limit`."""
    import sys

    import nemo_gym.cli.main as main_module
    from nemo_gym import global_config
    from nemo_gym.path_utils import materialized_path_for

    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "ds"
    write_task(folder / "alpha")
    beta = write_task(folder / "beta")
    dispatched = []
    monkeypatch.setattr(main_module, "dispatch", lambda target, overrides: dispatched.append(overrides))
    # A fixed address: use_absolute_ip resolves the host name, which a stock Ubuntu box answers with 127.0.1.1.
    monkeypatch.setattr(global_config, "gethostbyname", lambda _: "192.0.2.20")

    def invoke(*flags):
        monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
        # What any real run passes: the agent config interpolates the model name and the parser checks it at load.
        argv = ["gym", "eval", "run", str(folder), "--agent", "hermes_agent", "--sandbox", "docker"]
        argv += ["--model-type", "vllm_model", "+policy_model_name=test-model", "+policy_base_url=http://localhost:1"]
        argv += ["+policy_api_key=none", *flags]
        monkeypatch.setattr(sys, "argv", argv)
        main_module.main()

    def refused(*flags):
        with pytest.raises(SystemExit) as exc:
            invoke(*flags)
        assert exc.value.code == 2
        return capsys.readouterr().err

    # No cache yet: the run goes ahead. The collector would then save its inputs next to the rollouts.
    invoke("--resume")
    assert len(dispatched) == 1 and "+resume_from_cache=true" in dispatched[-1]
    rows = tmp_path / "results" / "harbor" / "ds" / "tasks_hermes_agent.jsonl"
    rollouts = tmp_path / "results" / "harbor" / "ds" / "rollouts.jsonl"
    shutil.copyfile(rows, materialized_path_for(rollouts))

    # Saved inputs but no results file yet: the collector skips the resume and starts fresh, so a changed
    # selection is not a conflict.
    invoke("--resume", "--exclude-tasks", "beta")
    assert len(dispatched) == 2
    rollouts.touch()

    invoke("--resume")  # the same selection resumes
    assert len(dispatched) == 3
    err = refused("--resume", "--exclude-tasks", "beta")
    assert len(dispatched) == 3
    cached = materialized_path_for(Path("results") / "harbor" / "ds" / "rollouts.jsonl")  # as configured: relative
    assert f"Cannot resume from {cached}" in err and "no longer selected: beta" in err
    assert "+resume_from_cache=false" in err and "+output_jsonl_fpath" in err
    # Without --resume the narrower selection is simply a fresh run.
    invoke("--exclude-tasks", "beta")
    assert len(dispatched) == 4 and not any("resume_from_cache" in token for token in dispatched[-1])
    # `resume_from_cache: true` from a config file is the same resume: the collector reads the merged config,
    # not the flags.
    (tmp_path / "resume.yaml").write_text("resume_from_cache: true\n")
    err = refused("--config", str(tmp_path / "resume.yaml"), "--exclude-tasks", "beta")
    assert f"Cannot resume from {cached}" in err and "no longer selected: beta" in err and len(dispatched) == 4
    # An edited task has another digest than the cached row.
    (beta / "instruction.md").write_text("A different task now.\n")
    assert "changed digest: beta" in refused("--resume")
    # Another output file is another cache, found through the same config the collector reads.
    invoke("--resume", f"+output_jsonl_fpath={tmp_path / 'elsewhere' / 'rollouts.jsonl'}")
    assert len(dispatched) == 5
    # A `--limit 1` run cached the first row alone: the same limited command resumes it, and only the run that
    # would now include beta is refused.
    materialized_path_for(rollouts).write_text(rows.read_text().splitlines(keepends=True)[0])
    invoke("--resume", "--limit", "1")
    assert len(dispatched) == 6 and "+limit=1" in dispatched[-1]
    assert "newly selected: beta" in refused("--resume")
    assert len(dispatched) == 6


def test_eval_run_resume_reports_a_config_error_cleanly(tmp_path, monkeypatch, capsys):
    """The resume check parses the run config before dispatch: a config error is the same one-line message
    the dispatched collector prints, not a traceback."""
    import sys

    import nemo_gym.cli.main as main_module
    from nemo_gym import global_config
    from nemo_gym.path_utils import materialized_path_for

    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "ds"
    write_task(folder / "alpha")
    dispatched = []
    monkeypatch.setattr(main_module, "dispatch", lambda target, overrides: dispatched.append(overrides))
    # A fixed address: use_absolute_ip resolves the host name, which a stock Ubuntu box answers with 127.0.1.1.
    monkeypatch.setattr(global_config, "gethostbyname", lambda _: "192.0.2.20")
    model = ["+policy_model_name=test-model", "+policy_base_url=http://localhost:1", "+policy_api_key=none"]

    def invoke(*flags):
        monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
        argv = ["gym", "eval", "run", str(folder), "--agent", "hermes_agent", "--sandbox", "docker"]
        monkeypatch.setattr(sys, "argv", argv + ["--model-type", "vllm_model", *flags])
        main_module.main()

    invoke("--resume", *model)
    assert len(dispatched) == 1
    rows = tmp_path / "results" / "harbor" / "ds" / "tasks_hermes_agent.jsonl"
    shutil.copyfile(rows, materialized_path_for(tmp_path / "results" / "harbor" / "ds" / "rollouts.jsonl"))

    # The shell lost its model overrides: the agent config's `policy_model_name` interpolation cannot resolve.
    with pytest.raises(SystemExit) as exc:
        invoke("--resume")
    assert exc.value.code == 1 and len(dispatched) == 1
    out = capsys.readouterr().out
    assert "Error:" in out and "policy_model_name" in out and "Provide it via a CLI override" in out


class TestDatasetTaskAdditions:
    """`[gym.tasks."<id>"]` gains `verifier` overrides and `artifacts` additions, applied like `environment`."""

    def test_verifier_fields_override_and_env_merges(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a", toml=HELLO_TOML + '\n[verifier.env]\nMODE = "fast"\nKEEP = "1"\n')
        (folder / "dataset.toml").write_text(
            '[gym.tasks."a".verifier]\ntimeout_sec = 900.0\nuser = "root"\n'
            '[gym.tasks."a".verifier.env]\nMODE = "strict"\nJUDGE = "${JUDGE_KEY}"\n'
        )
        verifier = load_task(folder / "a").config.verifier
        assert verifier.timeout_sec == 900.0 and verifier.user == "root"
        assert verifier.env == {"MODE": "strict", "KEEP": "1", "JUDGE": "${JUDGE_KEY}"}
        # The task's own file is untouched and the digest, which covers it, is unchanged by dataset.toml.
        assert "900" not in (folder / "a" / "task.toml").read_text()

    def test_verifier_environment_override_merges_env_like_the_agents(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(
            folder / "a",
            toml=HELLO_TOML + '\n[verifier.environment]\ndocker_image = "org/verifier:1"\nenv = { A = "1" }\n',
        )
        (folder / "dataset.toml").write_text('[gym.tasks."a".verifier.environment]\ncpus = 4\nenv = { B = "2" }\n')
        environment = load_task(folder / "a").config.verifier.environment
        assert environment.docker_image == "org/verifier:1" and environment.cpus == 4
        assert environment.env == {"A": "1", "B": "2"}

    def test_artifacts_are_added_after_the_tasks_own(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a", toml=HELLO_TOML.replace("[task]\n", 'artifacts = ["/app/out.txt"]\n\n[task]\n', 1))
        (folder / "dataset.toml").write_text(
            '[gym.tasks."a"]\nartifacts = ["/var/log/app", { source = "/data/db", service = "db" }]\n'
        )
        artifacts = load_task(folder / "a").config.artifacts
        assert [(a.source, a.service) for a in artifacts] == [
            ("/app/out.txt", None),
            ("/var/log/app", None),
            ("/data/db", "db"),
        ]
        # Timeout multipliers still apply on top of the per-task verifier override.
        (folder / "dataset.toml").write_text(
            '[gym.defaults]\ntimeout_multiplier = 2.0\n[gym.tasks."a".verifier]\ntimeout_sec = 100.0\n'
        )
        assert load_task(folder / "a").config.verifier.timeout_sec == 200.0

    @pytest.mark.parametrize(
        ("table", "message"),
        [
            ('[gym.tasks."a".verifier]\ntimeout = 5\n', r"verifier\] has unknown keys: timeout"),
            ('[gym.tasks."a".verifier]\ntimeout_sec = -1\n', r"verifier\]:"),
            ('[gym.tasks."a"]\nartifacts = ["../escape"]\n', r"artifacts\]:"),
        ],
    )
    def test_bad_additions_are_named(self, tmp_path, table, message):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        (folder / "dataset.toml").write_text(table)
        with pytest.raises(HarborTaskError, match=message):
            load_task(folder / "a")


def test_taskset_mapping_records_the_relative_path_of_nested_tasks(tmp_path):
    from nemo_gym.tasks.harbor.materialize import taskset_mapping

    folder = tmp_path / "ds"
    write_task(folder / "a")
    write_task(folder / "group" / "sub" / "deep")
    tasks = discover_tasks(folder)
    mapping = taskset_mapping(tasks, folder)
    assert mapping["folder"] == str(folder.resolve())
    assert set(mapping["tasks"]) == {"a", "deep"}
    # Only tasks below the first level need a path; the server resolves the others as <folder>/<task_id>.
    assert mapping["paths"] == {"deep": "group/sub/deep"}
    assert Path(mapping["folder"]) / mapping["paths"]["deep"] == tasks[1].path
    # Prepare writes the same mapping into the run config.
    config = run_config(
        taskset="ds",
        folder=folder,
        tasks=tasks,
        rows_path=tmp_path / "out" / "tasks.jsonl",
        agent_instance_name="hermes_agent",
        agent_impl="hermes_agent",
    )
    assert config["harbor_ds_environment"]["environment_servers"]["harbor"]["tasksets"]["ds"]["paths"] == {
        "deep": "group/sub/deep"
    }


def test_validation_summary_reports_verify_time_and_network_per_task(tmp_path):
    from nemo_gym.tasks.harbor.validation_report import summarize_validation

    rollouts = tmp_path / "rollouts.jsonl"
    rows = [
        {
            "_ng_task_id": {"taskset": "ds", "task_id": "timed"},
            "response": {"metadata": {"oracle": "solved"}},
            "reward": 1.0,
            "verifier_seconds": 12.345,
            "verifier_network": "public",
        },
        {
            "_ng_task_id": {"taskset": "ds", "task_id": "masked"},
            "response": {"metadata": {"oracle": "solved"}},
            "reward": 0.0,
            "mask_sample": True,
            "failure_kind": "provider_unavailable",
            "verifier_seconds": 3,
            "verifier_network": "no-network",
        },
    ]
    rollouts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    lines = summarize_validation(rollouts, []).text.splitlines()
    assert lines[0] == "task_id\tstatus\treward\tverify_s\tnetwork"
    assert "timed\toracle solved\t1.0\t12.3\tpublic" in lines
    assert "masked\tmasked (provider_unavailable)\t-\t3.0\tno-network" in lines


class TestComposeVolumeSourcesAtPrepare:
    """M22: ``volume_sources`` reaches prepare from the ``sandbox:`` block the command names."""

    @pytest.fixture
    def sources(self, tmp_path, monkeypatch):
        import nemo_gym.tasks.harbor.cli as harbor_cli

        dotenv = tmp_path / "env.yaml"
        dotenv.write_text("some_key: 1\n")
        monkeypatch.setattr(build_options, "_resolve_under_cwd_or_install", lambda rel: tmp_path / rel)
        provider = tmp_path / "provider.yaml"
        # Credentials interpolate from the environment, as the real provider files do; reading volume_sources
        # must not need them.
        provider.write_text(
            "sandbox:\n  opensandbox:\n    connection:\n      api_key: ${oc.env:M22_UNSET_KEY}\n"
            "  volume_sources:\n    pgdata: datasets/pgdata\n"
        )
        monkeypatch.setattr(harbor_cli, "resolve_sandbox_config", lambda name: provider)
        monkeypatch.delenv("M22_UNSET_KEY", raising=False)
        return dotenv, provider

    def test_volume_sources_come_from_the_sandbox_file_and_overrides(self, tmp_path, sources):
        from nemo_gym.tasks.harbor.cli import volume_sources_from_overrides

        assert volume_sources_from_overrides([], None) is None
        assert volume_sources_from_overrides([], "opensandbox") == {"pgdata": "datasets/pgdata"}
        tokens = [
            "++sandbox.volume_sources.pgdata=overridden/pg",
            "++sandbox.volume_sources.models=m",
            "+agent_name=a",
        ]
        assert volume_sources_from_overrides(tokens, "opensandbox") == {"pgdata": "overridden/pg", "models": "m"}
        # A --config file sits above the provider file and below the ++ overrides; a host path is a plain key.
        config = tmp_path / "run.yaml"
        config.write_text("sandbox:\n  volume_sources:\n    pgdata: elsewhere/pg\n    /var/lib/models: models\n")
        assert volume_sources_from_overrides([f'+config_paths=["{config}"]'], "opensandbox") == {
            "pgdata": "elsewhere/pg",
            "/var/lib/models": "models",
        }
        with pytest.raises(ValueError, match="volume_sources.*must be a mapping"):
            volume_sources_from_overrides(["++sandbox.volume_sources=[pgdata]"], None)

    def test_shared_storage_is_read_from_the_blocks_provider(self, tmp_path, sources):
        """M35 D6: prepare answers whether the block's provider has shared storage from the one setting the server's
        provider requires (``shared_storage.host_path``), without resolving the block's credentials; unknown (no
        block, or none naming one provider) stays ``None`` and the seed decides."""
        from nemo_gym.tasks.harbor.cli import (
            GPU_SANDBOX_PROVIDER_KEY,
            gpu_shared_storage_from_overrides,
            shared_storage_from_overrides,
        )

        assert shared_storage_from_overrides([], None) is None
        assert shared_storage_from_overrides([], "opensandbox") is False
        assert (
            shared_storage_from_overrides(
                ["++sandbox.opensandbox.shared_storage.host_path=/mnt/shared"], "opensandbox"
            )
            is True
        )
        config = tmp_path / "run.yaml"
        config.write_text("sandbox:\n  opensandbox:\n    shared_storage:\n      host_path: /mnt/efs/data/shared\n")
        assert shared_storage_from_overrides([f'+config_paths=["{config}"]'], "opensandbox") is True
        assert (
            shared_storage_from_overrides(["++sandbox.opensandbox.shared_storage.host_path="], "opensandbox") is False
        )
        gpu_block = f"++{GPU_SANDBOX_PROVIDER_KEY}=sandbox_gpu"
        assert gpu_shared_storage_from_overrides([], "opensandbox") is None
        assert gpu_shared_storage_from_overrides([gpu_block], "opensandbox") is None
        assert (
            gpu_shared_storage_from_overrides([gpu_block, "++sandbox_gpu.opensandbox.connection.api_key=k"], None)
            is False
        )
        assert (
            gpu_shared_storage_from_overrides(
                [gpu_block, "++sandbox_gpu.opensandbox.shared_storage.host_path=/mnt/gpu"], None
            )
            is True
        )

    def test_gpu_volume_sources_come_from_the_gpu_sandbox_provider_block(self, tmp_path, sources):
        # Review: GPU-declaring tasks are seeded from the server's `gpu_sandbox_provider` block, so prepare reads
        # that block's volume_sources too: None without a GPU block, {} for a GPU block without the key.
        from nemo_gym.tasks.harbor.cli import GPU_SANDBOX_PROVIDER_KEY, gpu_volume_sources_from_overrides

        gpu_block = f"++{GPU_SANDBOX_PROVIDER_KEY}=sandbox_gpu"
        assert GPU_SANDBOX_PROVIDER_KEY == "harbor_environment_server.environment_servers.harbor.gpu_sandbox_provider"
        assert gpu_volume_sources_from_overrides([], "opensandbox") is None
        assert gpu_volume_sources_from_overrides([gpu_block], "opensandbox") == {}
        assert gpu_volume_sources_from_overrides([gpu_block, "++sandbox_gpu.volume_sources.pgdata=gpu/pg"], None) == {
            "pgdata": "gpu/pg"
        }
        # The sandbox block's mapping does not leak into the GPU block's.
        assert gpu_volume_sources_from_overrides([gpu_block, "++sandbox.volume_sources.pgdata=pg"], None) == {}
        config = tmp_path / "run.yaml"
        config.write_text(
            "harbor_environment_server:\n  environment_servers:\n    harbor:\n      gpu_sandbox_provider: sandbox_gpu\n"
            "sandbox_gpu:\n  volume_sources:\n    models: gpu/models\n"
        )
        assert gpu_volume_sources_from_overrides([f'+config_paths=["{config}"]'], None) == {"models": "gpu/models"}
        with pytest.raises(ValueError, match="must name a top-level sandbox block"):
            gpu_volume_sources_from_overrides([f"++{GPU_SANDBOX_PROVIDER_KEY}=[a]"], None)

    def test_run_and_validate_hand_volume_sources_to_prepare(self, tmp_path, monkeypatch, sources):
        import argparse

        import nemo_gym.tasks.harbor.cli as harbor_cli

        seen = []

        def fake_prepare(target, **kwargs):
            seen.append((kwargs["volume_sources"], kwargs["gpu_volume_sources"]))
            raise harbor_cli.HubError("stop here")

        def fail(message):
            raise SystemExit(message)

        monkeypatch.setattr(harbor_cli, "prepare_target", fake_prepare)
        # The rows file is scoped to the agent, so the stub has to name one.
        monkeypatch.setattr(
            harbor_cli, "resolve_agent", lambda name: harbor_cli.AgentSelection(Path("a.yaml"), name, name)
        )
        args = argparse.Namespace(
            target=str(tmp_path), agent="a", sandbox="opensandbox", _parser=argparse.Namespace(error=fail)
        )
        with pytest.raises(SystemExit, match="stop here"):
            harbor_cli.run_target(args, [])
        with pytest.raises(SystemExit, match="stop here"):
            harbor_cli.validate_target(
                args,
                [
                    "++sandbox.volume_sources.pgdata=pg",
                    f"++{harbor_cli.GPU_SANDBOX_PROVIDER_KEY}=sandbox_gpu",
                    "++sandbox_gpu.volume_sources.pgdata=gpu/pg",
                ],
            )
        assert seen == [({"pgdata": "datasets/pgdata"}, None), ({"pgdata": "pg"}, {"pgdata": "gpu/pg"})]

    def test_registry_credentials_come_from_the_sandbox_image_auth(self, tmp_path, monkeypatch):
        """G7 round 24: prepare read registries anonymously, so a private Compose sidecar (registry.example.com...) stopped
        it with HTTP 403 though the sandbox block's `images.auth` would pull it. Prepare reads the same block's
        `images.auth`, resolved (so `${gitlab_registry_password}` comes from env.yaml), and records images with it."""
        import nemo_gym.tasks.harbor.build_options as build_options
        import nemo_gym.tasks.harbor.cli as harbor_cli

        dotenv = tmp_path / "env.yaml"
        dotenv.write_text("gitlab_registry_username: bot\ngitlab_registry_password: secret\n")
        monkeypatch.setattr(build_options, "_resolve_under_cwd_or_install", lambda rel: tmp_path / rel)
        provider = tmp_path / "provider.yaml"
        provider.write_text(
            "sandbox:\n  opensandbox:\n    images:\n      auth:\n"
            '        "registry.example.com:5005/":\n'
            "          username: ${gitlab_registry_username}\n"
            "          password: ${gitlab_registry_password}\n"
        )
        monkeypatch.setattr(harbor_cli, "resolve_sandbox_config", lambda name: provider)

        from nemo_gym.tasks.harbor.cli import registry_credentials_from_overrides

        # No block names a provider: anonymous, as before.
        assert registry_credentials_from_overrides([], None) == {}
        assert registry_credentials_from_overrides([], "opensandbox") == {
            "registry.example.com:5005/": ("bot", "secret")
        }

    def test_run_and_validate_hand_registry_credentials_to_prepare(self, tmp_path, monkeypatch, sources):
        import argparse

        import nemo_gym.tasks.harbor.cli as harbor_cli

        # The `sources` fixture's provider.yaml has no images.auth, so credentials are empty; a block with one is
        # covered above. This asserts both entry points read them and pass them on.
        seen = []

        def fake_prepare(target, **kwargs):
            seen.append(kwargs["registry_credentials"])
            raise harbor_cli.HubError("stop here")

        def fail(message):
            raise SystemExit(message)

        monkeypatch.setattr(harbor_cli, "prepare_target", fake_prepare)
        monkeypatch.setattr(
            harbor_cli, "resolve_agent", lambda name: harbor_cli.AgentSelection(Path("a.yaml"), name, name)
        )
        args = argparse.Namespace(
            target=str(tmp_path), agent="a", sandbox="opensandbox", _parser=argparse.Namespace(error=fail)
        )
        with pytest.raises(SystemExit, match="stop here"):
            harbor_cli.run_target(args, [])
        with pytest.raises(SystemExit, match="stop here"):
            harbor_cli.validate_target(args, [])
        assert seen == [{}, {}]

    def test_prepare_fails_naming_an_unmapped_external_volume(self, tmp_path, fake_registry):
        folder = tmp_path / "ds"
        write_task(folder / "pg")
        (folder / "pg" / "environment" / "docker-compose.yaml").write_text(
            "services:\n  db:\n    image: postgres:16\n    volumes: [pgdata:/var/lib/postgresql/data]\n"
            "volumes:\n  pgdata: {external: true}\n"
        )
        # The volume the group cannot run leaves its task out with the reason, as a Compose file that does not
        # resolve does; with no other task, prepare ends naming it.
        with pytest.raises(ValueError, match="external volume 'pgdata' needs a `volume_sources` entry"):
            prepare_target(str(folder), output_root=tmp_path / "out")
        prepared = prepare_target(str(folder), output_root=tmp_path / "out", volume_sources={"pgdata": "pg"})
        assert [task.task_id for task in prepared.tasks] == ["pg"]

    def test_gpu_tasks_prepare_against_the_gpu_block(self, tmp_path, fake_registry):
        # Review: with a GPU block configured, a GPU-declaring task passes or fails prepare in step with the
        # block the server seeds it from, whichever of the two carries the mapping.
        folder = tmp_path / "ds"
        write_task(folder / "pg", toml=HELLO_TOML.replace("gpus = 0", "gpus = 1"))
        (folder / "pg" / "environment" / "docker-compose.yaml").write_text(
            "services:\n  db:\n    image: postgres:16\n    volumes: [pgdata:/var/lib/postgresql/data]\n"
            "volumes:\n  pgdata: {external: true}\n"
        )
        mapped = {"pgdata": "pg"}
        with pytest.raises(ValueError, match="external volume 'pgdata' needs a `volume_sources` entry"):
            prepare_target(str(folder), output_root=tmp_path / "out", volume_sources=mapped, gpu_volume_sources={})
        prepared = prepare_target(str(folder), output_root=tmp_path / "out", gpu_volume_sources=mapped)
        assert [task.task_id for task in prepared.tasks] == ["pg"]


def test_pulled_datasets_default_to_the_working_directory_for_a_package_install(tmp_path) -> None:
    """A package install has no checkout: ``PARENT_DIR`` is site-packages, so pulls go under the working directory.

    The installed layout is rebuilt in a subprocess: the package reached through a ``site/`` folder with no
    ``pyproject.toml`` beside it (what tells the two installs apart in ``nemo_gym/__init__.py``), run from a
    working directory elsewhere. The default must follow that directory, never the install location, which is
    shared and often read-only.
    """
    import os
    import sys

    from nemo_gym import ROOT_DIR

    site = tmp_path / "site"
    site.mkdir()
    (site / "nemo_gym").symlink_to(ROOT_DIR, target_is_directory=True)
    work = tmp_path / "work"
    work.mkdir()
    probe = (
        "import sys\n"
        f"sys.path = [{str(site)!r}, *(p for p in sys.path if '__editable__' not in p and p != '')]\n"
        "sys.meta_path = [f for f in sys.meta_path if 'editable' not in getattr(f, '__module__', '').lower()]\n"
        "import nemo_gym\n"
        "from nemo_gym.tasks.harbor.hub import datasets_dir\n"
        "print(nemo_gym.PARENT_DIR)\n"
        "print(datasets_dir())\n"
    )
    env = {key: value for key, value in os.environ.items() if key != "NEMO_GYM_DATASETS_DIR"}
    completed = subprocess.run(
        [sys.executable, "-c", probe], cwd=work, env=env, capture_output=True, text=True, check=True, timeout=300
    )
    parent_dir, pulled = completed.stdout.splitlines()[-2:]
    assert Path(parent_dir) == site  # the probe ran the installed layout, not this checkout
    assert Path(pulled) == (work / "environments" / "harbor").resolve()


def test_installed_entries_are_pulled_under_the_working_directory_not_the_installation(tmp_path) -> None:
    """A committed entry that ships with a package install lives in site-packages, which is shared and often
    read-only: with ``NEMO_GYM_DATASETS_DIR`` unset, the pull of its name and of the reference it declares both land
    in ``<working directory>/environments/harbor/<entry>/`` (``datasets_dir()``), with the entry's ``dataset.toml``
    copied there, and nothing is created under the installation.

    The installed layout is rebuilt in a subprocess as in
    :func:`test_pulled_datasets_default_to_the_working_directory_for_a_package_install`: the package and the entry
    reached through a ``site/`` folder with no ``pyproject.toml`` beside it, run from a working directory elsewhere.
    The plan is inspected (:func:`~nemo_gym.tasks.harbor.entries.plan_pull`) and the destination prepared; nothing
    is fetched.
    """
    import os
    import sys

    from nemo_gym import ROOT_DIR

    ref = "harbor:terminal-bench/terminal-bench@4.0.0"
    site = tmp_path / "site"
    site.mkdir()
    (site / "nemo_gym").symlink_to(ROOT_DIR, target_is_directory=True)
    installed = site / "environments" / "harbor" / "tb4_installed"
    installed.mkdir(parents=True)
    table = f'[gym]\nsource = "{ref}"\n'
    (installed / "dataset.toml").write_text(table)
    work = tmp_path / "work"
    work.mkdir()
    probe = (
        "import sys\n"
        f"sys.path = [{str(site)!r}, *(p for p in sys.path if '__editable__' not in p and p != '')]\n"
        "sys.meta_path = [f for f in sys.meta_path if 'editable' not in getattr(f, '__module__', '').lower()]\n"
        "import nemo_gym\n"
        "from nemo_gym.tasks.harbor.entries import entry_target, plan_pull, prepare_destination\n"
        "print(nemo_gym.PARENT_DIR)\n"
        f"by_name, by_ref = plan_pull('tb4_installed'), plan_pull({ref!r})\n"
        "assert by_name.entry == by_ref.entry and by_name.ref == by_ref.ref, (by_name, by_ref)\n"
        "print(by_name.entry.folder)\n"
        "print(by_name.folder)\n"
        "print(by_ref.folder)\n"
        "print(entry_target('tb4_installed').destination)\n"
        "print(prepare_destination(by_name.entry))\n"
    )
    env = {
        key: value for key, value in os.environ.items() if key not in ("NEMO_GYM_DATASETS_DIR", "NEMO_GYM_EXTRA_ROOTS")
    }
    completed = subprocess.run(
        [sys.executable, "-c", probe], cwd=work, env=env, capture_output=True, text=True, check=True, timeout=300
    )
    parent_dir, folder, by_name, by_ref, by_target, prepared = completed.stdout.splitlines()[-6:]
    assert Path(parent_dir) == site  # the probe ran the installed layout, not this checkout
    assert Path(folder) == installed.resolve()  # the entry is the installed one
    expected = (work / "environments" / "harbor" / "tb4_installed").resolve()
    assert Path(by_name) == Path(by_ref) == Path(by_target) == Path(prepared) == expected
    # The destination holds a copy of the entry's table; the installation is left as it was.
    assert (expected / "dataset.toml").read_text() == table
    assert sorted(p.name for p in installed.iterdir()) == ["dataset.toml"]
    assert sorted(p.name for p in (site / "environments" / "harbor").iterdir()) == ["tb4_installed"]


def test_installed_entries_in_a_project_local_venv_are_pulled_under_the_working_directory(tmp_path) -> None:
    """A project-local ``.venv`` puts site-packages under the working directory, so an entry the package ships there
    lies under the working root too; it is installed all the same, and with ``NEMO_GYM_DATASETS_DIR`` unset the pull
    of its name and of its reference both land in ``<working directory>/environments/harbor/<entry>/``, with the
    entry's ``dataset.toml`` copied there, and nothing is written under ``.venv`` (Yu He, round 36: both pulls
    targeted the installation and failed on a read-only one).

    The installed layout is rebuilt in a subprocess as in
    :func:`test_installed_entries_are_pulled_under_the_working_directory_not_the_installation`, with the ``site/``
    folder placed at ``work/.venv/lib/python3.X/site-packages`` and the probe run from ``work``.
    """
    import os
    import sys

    from nemo_gym import ROOT_DIR

    ref = "harbor:terminal-bench/terminal-bench@4.0.0"
    work = tmp_path / "work"
    site = work / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    site.mkdir(parents=True)
    (site / "nemo_gym").symlink_to(ROOT_DIR, target_is_directory=True)
    installed = site / "environments" / "harbor" / "tb4_installed"
    installed.mkdir(parents=True)
    table = f'[gym]\nsource = "{ref}"\n'
    (installed / "dataset.toml").write_text(table)
    venv_before = sorted(str(path.relative_to(work)) for path in (work / ".venv").rglob("*"))
    probe = (
        "import sys\n"
        f"sys.path = [{str(site)!r}, *(p for p in sys.path if '__editable__' not in p and p != '')]\n"
        "sys.meta_path = [f for f in sys.meta_path if 'editable' not in getattr(f, '__module__', '').lower()]\n"
        "import nemo_gym\n"
        "from nemo_gym.tasks.harbor.entries import entry_target, plan_pull, prepare_destination\n"
        "print(nemo_gym.PARENT_DIR)\n"
        "print(nemo_gym.WORKING_DIR)\n"
        f"by_name, by_ref = plan_pull('tb4_installed'), plan_pull({ref!r})\n"
        "assert by_name.entry == by_ref.entry and by_name.ref == by_ref.ref, (by_name, by_ref)\n"
        "print(by_name.entry.folder)\n"
        "print(by_name.folder)\n"
        "print(by_ref.folder)\n"
        "print(entry_target('tb4_installed').destination)\n"
        "print(prepare_destination(by_name.entry))\n"
    )
    env = {
        key: value for key, value in os.environ.items() if key not in ("NEMO_GYM_DATASETS_DIR", "NEMO_GYM_EXTRA_ROOTS")
    }
    completed = subprocess.run(
        [sys.executable, "-c", probe], cwd=work, env=env, capture_output=True, text=True, check=True, timeout=300
    )
    parent_dir, working_dir, folder, by_name, by_ref, by_target, prepared = completed.stdout.splitlines()[-7:]
    assert Path(parent_dir) == site  # the probe ran the installed layout, not this checkout
    assert Path(working_dir) == work  # a package install's working root is the working directory, above .venv
    assert Path(folder) == installed.resolve()  # the entry is the installed one
    expected = (work / "environments" / "harbor" / "tb4_installed").resolve()
    assert Path(by_name) == Path(by_ref) == Path(by_target) == Path(prepared) == expected
    # The destination holds a copy of the entry's table; nothing under .venv was written or created.
    assert (expected / "dataset.toml").read_text() == table
    assert sorted(str(path.relative_to(work)) for path in (work / ".venv").rglob("*")) == venv_before
    assert (installed / "dataset.toml").read_text() == table


def test_pulled_datasets_default_to_environments_harbor_in_the_checkout(monkeypatch, tmp_path) -> None:
    """Fetched datasets land in environments/harbor/ of the Gym checkout unless NEMO_GYM_DATASETS_DIR says otherwise."""
    from nemo_gym import PARENT_DIR
    from nemo_gym.tasks.harbor.hub import datasets_dir

    monkeypatch.delenv("NEMO_GYM_DATASETS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)  # in an editable install (this checkout) the working directory plays no part
    assert datasets_dir() == (PARENT_DIR / "environments" / "harbor").resolve()

    monkeypatch.setenv("NEMO_GYM_DATASETS_DIR", str(tmp_path / "elsewhere"))
    assert datasets_dir() == (tmp_path / "elsewhere").resolve()


TRAINING_TOML = """[gym]
license = "Apache 2.0"
type = "validation"
domain = "coding"
verified = true
allowed_agents = ["terminus_2_sandboxed_agent", "hermes_agent"]

[gym.defaults]
num_repeats = 3
"""


class TestTrainingMetadata:
    """The `[gym]` keys of M37 flow from dataset.toml into the generated `datasets:` entry, the rows and the run."""

    def test_task_source_key_is_the_collectors(self):
        from nemo_gym.global_config import TASK_SOURCE_KEY_NAME

        assert TASK_SOURCE_KEY == TASK_SOURCE_KEY_NAME

    def test_entry_carries_the_gym_table(self, tmp_path):
        from nemo_gym.config_types import DatasetConfig
        from nemo_gym.tasks.harbor.dataset_config import read_dataset_config

        folder = tmp_path / "ds"
        write_task(folder / "a")
        (folder / "dataset.toml").write_text(TRAINING_TOML)
        dataset = read_dataset_config(folder)
        tasks = [load_task(folder / "a")]
        rows_path = tmp_path / "out" / "tasks.jsonl"
        write_rows(tasks, "ds", rows_path, dataset_digest="d" * 64)

        config = run_config(
            taskset="ds",
            folder=folder,
            tasks=tasks,
            rows_path=rows_path,
            agent_instance_name="hermes_agent",
            agent_impl="hermes_agent",
            dataset=dataset,
        )
        entry = config["harbor_ds_agent"]["responses_api_agents"]["hermes_agent"]["datasets"][0]
        assert entry == {
            "name": "ds",
            "type": "validation",
            "license": "Apache 2.0",
            "domain": "coding",
            "verified": True,
            "allowed_agents": ["terminus_2_sandboxed_agent", "hermes_agent"],
            "jsonl_fpath": str(rows_path.resolve()),
        }
        # The entry round-trips through Gym's dataset config type with the metadata kept, and is written
        # back without it when unset (existing metrics sidecars dump dataset entries). The repeat count is
        # not the entry's (collate would duplicate by it): build_run puts it on the run config.
        parsed = DatasetConfig.model_validate(entry)
        assert (parsed.domain, parsed.verified, parsed.allowed_agents, parsed.num_repeats) == (
            "coding",
            True,
            ["terminus_2_sandboxed_agent", "hermes_agent"],
            1,
        )
        plain = DatasetConfig(name="x", type="validation", jsonl_fpath="x.jsonl", license="MIT").model_dump(
            mode="json"
        )
        assert not {"domain", "verified", "allowed_agents"} & set(plain)
        # The rows carry the dataset label and both digests.
        row = json.loads(rows_path.read_text().splitlines()[0])
        assert row[TASK_SOURCE_KEY] == "ds"
        assert row["task_input"]["task_data"] == {DIGEST_KEY: tasks[0].digest, "_ng_dataset_digest": "d" * 64}

    def test_entry_defaults_follow_the_rfc(self, tmp_path):
        from nemo_gym.tasks.harbor.dataset_config import GymDatasetConfig

        rows = tmp_path / "tasks.jsonl"
        # A license alone makes a validation set; a type alone (benchmark) needs none; domain is the enum's value.
        assert dataset_entry("ds", rows, GymDatasetConfig(license="MIT"))["type"] == "validation"
        assert dataset_entry("ds", rows, GymDatasetConfig(license="MIT"))["license"] == "MIT"
        assert "license" not in dataset_entry("ds", rows, GymDatasetConfig(type="benchmark"))
        assert dataset_entry("ds", rows, GymDatasetConfig(domain="math"))["domain"] == "math"
        assert dataset_entry("ds", rows, None)["verified"] is False
        assert "allowed_agents" not in dataset_entry("ds", rows, None)

    @pytest.mark.parametrize(
        "table",
        [
            {"defaults": {"num_repeats": 4}},
            {"domain": "coding"},
            {"verified": True},
            {"allowed_agents": ["hermes_agent"]},
            {"license": "MIT"},
            {"type": "benchmark"},
            {"license": "MIT", "type": "train", "domain": "math", "verified": True, "defaults": {"num_repeats": 2}},
        ],
    )
    def test_keys_that_shape_no_task_never_enter_its_digest(self, table):
        """Each key alone, against the empty table: the digest stays the bare folder hash, because nothing about
        the task changed. Only a table that shapes tasks folds a slice in, and the repeat count stays out of it."""
        from nemo_gym.tasks.harbor.dataset_config import GymDatasetConfig

        config = GymDatasetConfig.model_validate(table)
        assert not config.is_empty and not config.shapes_tasks
        assert task_digest("f" * 64, config, "a") == task_digest("f" * 64, GymDatasetConfig(), "a") == "f" * 64
        shaping = GymDatasetConfig.model_validate(table | {"defaults": {"timeout_multiplier": 2.0}})
        with_repeats = GymDatasetConfig.model_validate(
            table | {"defaults": {"timeout_multiplier": 2.0, "num_repeats": 7}}
        )
        assert shaping.shapes_tasks and task_digest("f" * 64, shaping, "a") != "f" * 64
        assert task_digest("f" * 64, with_repeats, "a") == task_digest("f" * 64, shaping, "a")

    def test_prepare_refuses_a_harness_the_dataset_does_not_admit(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        (folder / "dataset.toml").write_text(TRAINING_TOML)
        with pytest.raises(ValueError) as exc:
            prepare_target(str(folder), output_root=tmp_path / "out", agent="openhands_agent")
        message = str(exc.value)
        assert str(folder / "dataset.toml") in message
        assert "terminus_2_sandboxed_agent, hermes_agent" in message and "`--agent openhands_agent`" in message
        # Nothing was prepared: the refusal comes before the rows, the image records and the manifest.
        assert not (tmp_path / "out").exists() and not (folder / "manifest.toml").exists()

        prepared = prepare_target(str(folder), output_root=tmp_path / "out", agent="hermes_agent")
        assert prepared.dataset.allowed_agents == ["terminus_2_sandboxed_agent", "hermes_agent"]
        assert prepared.dataset.defaults.num_repeats == 3
        # The oracle (gym dataset validate) passes no agent and is always admitted.
        assert prepare_target(str(folder), output_root=tmp_path / "out2").dataset.verified is True

    def test_a_single_task_folder_is_checked_against_its_datasets_table(self, tmp_path):
        """A task folder target is prepared against its parent's `[gym]` table, and its rows name the parent:
        `task_source` is the dataset folder's name while the taskset (routing, output paths, instance names)
        stays the target's, so two single-task runs of one dataset carry one dataset label."""
        folder = tmp_path / "ds"
        write_task(folder / "a")
        (folder / "dataset.toml").write_text('[gym]\nallowed_agents = ["hermes_agent"]\n')
        with pytest.raises(ValueError, match="allowed_agents admits hermes_agent, not `--agent terminus_2"):
            prepare_target(str(folder / "a"), output_root=tmp_path / "out", agent="terminus_2_sandboxed_agent")
        prepared = prepare_target(str(folder / "a"), output_root=tmp_path / "out", agent="hermes_agent")
        assert prepared.tasks and (prepared.taskset, prepared.task_source) == ("a", "ds")
        (row,) = [json.loads(line) for line in prepared.rows_path.read_text().splitlines()]
        assert row["task_id"] == {"taskset": "a", "task_id": "a"} and row[TASK_SOURCE_KEY] == "ds"
        whole = prepare_target(str(folder), output_root=tmp_path / "out2", agent="hermes_agent")
        assert (whole.taskset, whole.task_source) == ("ds", "ds")
        assert json.loads(whole.rows_path.read_text().splitlines()[0])[TASK_SOURCE_KEY] == "ds"

    def test_a_standalone_task_folder_is_labelled_by_its_own_name(self, tmp_path):
        """A task folder whose parent is no dataset (no `dataset.toml`, no `manifest.toml`: a scratch directory
        that happens to hold it) is labelled by its own name, as a hand-built taskset is, so two unrelated task
        folders prepared from one parent do not share the parent's arbitrary name as their label. Once the parent
        is a dataset (a `dataset.toml` appears, or a manifest was written for it), the label is the parent's."""
        parent = tmp_path / "scratch_dir"
        write_task(parent / "hello-task")
        write_task(parent / "other-task")
        labels = {}
        for name in ("hello-task", "other-task"):
            prepared = prepare_target(str(parent / name), output_root=tmp_path / "out", agent="hermes_agent")
            (row,) = [json.loads(line) for line in prepared.rows_path.read_text().splitlines()]
            assert (prepared.taskset, prepared.task_source, row[TASK_SOURCE_KEY]) == (name, name, name)
            labels[name] = prepared.task_source
        assert labels == {"hello-task": "hello-task", "other-task": "other-task"}
        # The image records the first preparation wrote beside the task folders do not make the parent a dataset.
        assert not (parent / "dataset.toml").exists() and not (parent / "manifest.toml").exists()
        # A dataset.toml in the parent, even an empty table, makes it the dataset the task belongs to.
        (parent / "dataset.toml").write_text("[gym]\n")
        prepared = prepare_target(str(parent / "hello-task"), output_root=tmp_path / "out2", agent="hermes_agent")
        (row,) = [json.loads(line) for line in prepared.rows_path.read_text().splitlines()]
        assert (prepared.taskset, prepared.task_source, row[TASK_SOURCE_KEY]) == (
            "hello-task",
            "scratch_dir",
            "scratch_dir",
        )
        # So does a manifest alone: a dataset folder prepared as a whole writes one, and its task folders then name it.
        (parent / "dataset.toml").unlink()
        whole = prepare_target(str(parent), output_root=tmp_path / "out3", agent="hermes_agent")
        assert (parent / "manifest.toml").is_file() and whole.task_source == "scratch_dir"
        prepared = prepare_target(str(parent / "other-task"), output_root=tmp_path / "out4", agent="hermes_agent")
        assert (prepared.taskset, prepared.task_source) == ("other-task", "scratch_dir")

    def test_build_run_takes_split_and_repeats_from_the_dataset(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        (folder / "dataset.toml").write_text(TRAINING_TOML)
        prepared = prepare_target(str(folder), output_root=tmp_path / "out")
        agent = AgentSelection(tmp_path / "agent.yaml", "hermes_agent", "hermes_agent")

        def run_repeats(path: Path) -> int:
            config = yaml.safe_load(path.read_text())
            # The count is the run's, not the entry's: collate would duplicate rows by an entry count and the
            # collector would multiply on top, and the metrics sidecar beside the rows records the entry.
            entry = config["harbor_ds_agent"]["responses_api_agents"]["hermes_agent"]["datasets"][0]
            assert "num_repeats" not in entry
            return config["num_repeats"]

        config_path, tokens = build_run(prepared, agent, sandbox=None, overrides=[])
        assert run_repeats(config_path) == 3 and "+split=validation" in tokens

        # Command-line tokens pass through (they outrank the run config at parse time, see the real-parser
        # test below) and no default split is appended when one is given.
        for override in ("+num_repeats=5", "++num_repeats=5"):
            config_path, tokens = build_run(prepared, agent, sandbox=None, overrides=[override, "+split=train"])
            assert run_repeats(config_path) == 3 and override in tokens
            assert [token for token in tokens if token.startswith("+split=")] == ["+split=train"]

        # Without a table the rows are a benchmark run once.
        (folder / "dataset.toml").unlink()
        plain = prepare_target(str(folder), output_root=tmp_path / "out2")
        config_path, tokens = build_run(plain, agent, sandbox=None, overrides=[])
        assert run_repeats(config_path) == 1 and "+split=benchmark" in tokens

    def test_example_datasets_do_not_run(self, tmp_path):
        """An `example` dataset is refused where the harness is admitted, before the manifest, the image records
        and the rows (`gym eval run` passes the harness, `gym dataset validate` none; both prepare first), and
        `build_run` refuses a hand-built taskset of that type as a backstop."""
        from dataclasses import replace

        from nemo_gym.tasks.harbor.dataset_config import GymDatasetConfig

        folder = tmp_path / "ds"
        write_task(folder / "a")
        (folder / "dataset.toml").write_text('[gym]\ntype = "example"\n')
        for agent_name in ("hermes_agent", None):
            with pytest.raises(HarborTaskError, match='type = "example" marks smoke-test samples') as exc:
                prepare_target(str(folder), output_root=tmp_path / "out", agent=agent_name)
            assert str(folder / "dataset.toml") in str(exc.value)
        # Nothing was prepared: no output folder (rows, images.json, skipped.json), no manifest, no image records.
        assert not (tmp_path / "out").exists()
        assert sorted(path.name for path in folder.iterdir()) == ["a", "dataset.toml"]

        (folder / "dataset.toml").write_text('[gym]\ntype = "benchmark"\n')
        prepared = prepare_target(str(folder), output_root=tmp_path / "out", agent="hermes_agent")
        agent = AgentSelection(tmp_path / "agent.yaml", "hermes_agent", "hermes_agent")
        with pytest.raises(HarborTaskError, match='dataset.toml \\[gym\\] type = "example" marks smoke-test samples'):
            build_run(replace(prepared, dataset=GymDatasetConfig(type="example")), agent, sandbox=None, overrides=[])
        assert not list(prepared.output_dir.glob("run_config_*.yaml"))

    @pytest.mark.parametrize("command", [["eval", "run"], ["dataset", "validate"]])
    def test_run_and_validate_refuse_an_example_dataset_before_any_work(self, tmp_path, monkeypatch, capsys, command):
        """Through the real parser, both commands exit with the refusal and leave the dataset folder and the
        results root untouched: no manifest, no image records, no rows, no run config, nothing dispatched."""
        import sys

        import nemo_gym.cli.main as main_module
        import nemo_gym.global_config as global_config

        folder = tmp_path / "ds"
        write_task(folder / "a")
        (folder / "dataset.toml").write_text('[gym]\ntype = "example"\n')
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
        dispatched = []
        monkeypatch.setattr(main_module, "dispatch", lambda *args: dispatched.append(args))
        argv = ["gym", *command, str(folder)]
        if command[0] == "eval":
            argv += ["--agent", "hermes_agent", "--model-type", "vllm_model", "+policy_model_name=test-model"]
            argv += ["+policy_base_url=http://localhost:1", "+policy_api_key=none"]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as exc:
            main_module.main()
        assert exc.value.code == 2
        error = capsys.readouterr().err
        assert f'{folder / "dataset.toml"} [gym] type = "example" marks smoke-test samples' in error
        assert "`gym eval run` and `gym dataset validate` do not run" in error and "Traceback" not in error
        assert not dispatched
        assert sorted(path.name for path in folder.iterdir()) == ["a", "dataset.toml"]
        assert not (tmp_path / "results").exists()

    @pytest.mark.parametrize("override", [None, "+num_repeats=5"])
    def test_eval_run_inherits_the_datasets_repeats_unless_overridden(self, tmp_path, monkeypatch, override):
        """Through the real parser: the run's `num_repeats` is the dataset's, a `+num_repeats=` on the command
        line replaces it, and the entry carries none (so collate duplicates nothing and the count applies once)."""
        import sys

        import nemo_gym.cli.main as main_module
        import nemo_gym.global_config as global_config

        folder = tmp_path / "ds"
        write_task(folder / "good")
        (folder / "dataset.toml").write_text(TRAINING_TOML)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
        # A fixed address: use_absolute_ip resolves the host name, which not every host answers.
        monkeypatch.setattr(global_config, "gethostbyname", lambda _: "192.0.2.20")
        dispatched = []
        monkeypatch.setattr(main_module, "dispatch", lambda *args: dispatched.append(args))
        argv = ["gym", "eval", "run", str(folder), "--agent", "hermes_agent", "--model-type", "vllm_model"]
        argv += ["+policy_model_name=test-model", "+policy_base_url=http://localhost:1", "+policy_api_key=none"]
        if override:
            argv.append(override)
        monkeypatch.setattr(sys, "argv", argv)
        main_module.main()
        assert len(dispatched) == 1
        resolved = global_config.get_global_config_dict()
        entry = resolved["harbor_ds_agent"]["responses_api_agents"]["hermes_agent"]["datasets"][0]
        assert resolved["num_repeats"] == (5 if override else 3) and "num_repeats" not in entry
        assert resolved["split"] == "validation"
        assert entry["type"] == "validation" and entry["license"] == "Apache 2.0" and entry["verified"] is True

    @pytest.mark.parametrize(
        "table",
        [None, '[gym]\ndomain = "coding"\nallowed_agents = ["hermes_agent"]\n'],
        ids=["no-dataset-toml", "metadata-without-a-license"],
    )
    def test_eval_run_runs_a_dataset_without_a_license_as_a_benchmark(self, tmp_path, monkeypatch, table):
        """Through the real parser, the default of every hub dataset without a license: a benchmark entry (no
        `prepare_script`, `verified` false), the benchmark split, one rollout per task, and the typed entry keeps
        the metadata that was declared."""
        import sys

        import nemo_gym.cli.main as main_module
        import nemo_gym.global_config as global_config
        from nemo_gym.config_types import BenchmarkDatasetConfig

        folder = tmp_path / "ds"
        write_task(folder / "good")
        if table is not None:
            (folder / "dataset.toml").write_text(table)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
        # A fixed address: use_absolute_ip resolves the host name, which not every host answers.
        monkeypatch.setattr(global_config, "gethostbyname", lambda _: "192.0.2.20")
        dispatched = []
        monkeypatch.setattr(main_module, "dispatch", lambda *args: dispatched.append(args))
        argv = ["gym", "eval", "run", str(folder), "--agent", "hermes_agent", "--model-type", "vllm_model"]
        argv += ["+policy_model_name=test-model", "+policy_base_url=http://localhost:1", "+policy_api_key=none"]
        monkeypatch.setattr(sys, "argv", argv)
        main_module.main()
        assert len(dispatched) == 1
        resolved = global_config.get_global_config_dict()
        entry = resolved["harbor_ds_agent"]["responses_api_agents"]["hermes_agent"]["datasets"][0]
        assert (resolved["split"], resolved["num_repeats"]) == ("benchmark", 1)
        assert entry["type"] == "benchmark" and entry["verified"] is False and "license" not in entry
        parsed = BenchmarkDatasetConfig.model_validate(OmegaConf.to_container(entry))
        assert parsed.prepare_script is None and parsed.verified is False and parsed.num_repeats == 1
        if table is None:
            assert parsed.domain is None and parsed.allowed_agents is None
        else:
            assert parsed.domain == "coding" and parsed.allowed_agents == ["hermes_agent"]
        row = json.loads(Path(entry["jsonl_fpath"]).read_text().splitlines()[0])
        assert row[TASK_SOURCE_KEY] == "ds"

    @pytest.mark.parametrize("how", ["flag", "env"])
    def test_eval_run_honors_the_pairing_override_at_the_early_admission_check(self, tmp_path, monkeypatch, how):
        """Review (Yu He, M37): the early refusal ignored `--allow-unsupported-pairing`, so an explicit experimental
        run exited 2 before the collector could honor the flag it accepts. The override (the flag, or the
        environment variable the collector's guard reads) lets preparation proceed; the entry still carries the
        list, and the run config carries the override for the guard at dispatch."""
        import sys

        import nemo_gym.cli.main as main_module
        import nemo_gym.global_config as global_config
        from nemo_gym.tasks.harbor.cli import pairing_override_from_overrides

        folder = tmp_path / "ds"
        write_task(folder / "good")
        (folder / "dataset.toml").write_text(TRAINING_TOML)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
        monkeypatch.setattr(global_config, "gethostbyname", lambda _: "192.0.2.20")
        monkeypatch.delenv(global_config.ALLOW_UNSUPPORTED_PAIRING_ENV_VAR_NAME, raising=False)
        assert pairing_override_from_overrides([]) is False
        dispatched = []
        monkeypatch.setattr(main_module, "dispatch", lambda *args: dispatched.append(args))
        argv = ["gym", "eval", "run", str(folder), "--agent", "codex", "--model-type", "vllm_model"]
        argv += ["+policy_model_name=test-model", "+policy_base_url=http://localhost:1", "+policy_api_key=none"]
        if how == "flag":
            argv.append("--allow-unsupported-pairing")
        else:
            monkeypatch.setenv(global_config.ALLOW_UNSUPPORTED_PAIRING_ENV_VAR_NAME, "1")
        monkeypatch.setattr(sys, "argv", argv)
        main_module.main()
        assert len(dispatched) == 1
        resolved = global_config.get_global_config_dict()
        entry = resolved["harbor_ds_agent"]["responses_api_agents"]["codex_agent"]["datasets"][0]
        assert list(entry["allowed_agents"]) == ["terminus_2_sandboxed_agent", "hermes_agent"]
        assert global_config.pairing_override_enabled(resolved) is True

    def test_dispatch_refuses_a_run_config_edited_to_another_harness(self, tmp_path):
        """The entry's `allowed_agents` is not inert: the collector's pairing guard reads it from the generated
        agent block, so a run config edited to another harness after `gym eval run` admitted one is refused at
        dispatch too, naming the harness and the dataset."""
        from nemo_gym.rollout_collection import NG_ENVIRONMENT_SERVER_KEY, RolloutCollectionHelper
        from nemo_gym.tasks.harbor.dataset_config import read_dataset_config
        from nemo_gym.tasks.harbor.materialize import environment_server_instance

        folder = tmp_path / "ds"
        write_task(folder / "a")
        (folder / "dataset.toml").write_text(TRAINING_TOML)
        tasks = [load_task(folder / "a")]
        config = run_config(
            taskset="ds",
            folder=folder,
            tasks=tasks,
            rows_path=tmp_path / "tasks.jsonl",
            agent_instance_name="hermes_agent",
            agent_impl="hermes_agent",
            dataset=read_dataset_config(folder),
        )
        row = {NG_ENVIRONMENT_SERVER_KEY: environment_server_instance("ds")}
        RolloutCollectionHelper._validate_environment_servers([row], OmegaConf.create(config))

        edited = OmegaConf.create(config)
        block = edited["harbor_ds_agent"]["responses_api_agents"]
        block["codex_agent"] = block.pop("hermes_agent")
        with pytest.raises(ValueError) as exc:
            RolloutCollectionHelper._validate_environment_servers([row], edited)
        assert (
            "rows routed to 'harbor_ds_agent' run 'codex_agent', but its dataset 'ds' admits only: "
            "terminus_2_sandboxed_agent, hermes_agent"
        ) in str(exc.value)

    def test_the_oracles_run_config_passes_the_dispatch_guard_and_an_admitted_harness_too(self, tmp_path):
        """`gym dataset validate` is always admitted: the oracle's run config carries the entry without
        `allowed_agents` (the list speaks of model harnesses), so the collector's guard lets the oracle through,
        while an admitted harness's run config keeps the list and passes the same guard."""
        from nemo_gym.rollout_collection import NG_ENVIRONMENT_SERVER_KEY, RolloutCollectionHelper
        from nemo_gym.tasks.harbor.cli import ORACLE_AGENT
        from nemo_gym.tasks.harbor.materialize import environment_server_instance

        folder = tmp_path / "ds"
        write_task(folder / "a")
        (folder / "dataset.toml").write_text(TRAINING_TOML)
        prepared = prepare_target(str(folder), output_root=tmp_path / "out")
        row = {NG_ENVIRONMENT_SERVER_KEY: environment_server_instance("ds")}

        def resolved_entry(agent: AgentSelection) -> tuple[OmegaConf, dict]:
            config_path, _ = build_run(prepared, agent, sandbox=None, overrides=[])
            config = OmegaConf.create(yaml.safe_load(config_path.read_text()))
            RolloutCollectionHelper._validate_environment_servers([row], config)
            return config["harbor_ds_agent"]["responses_api_agents"][agent.impl_name]["datasets"][0]

        oracle = resolved_entry(AgentSelection(tmp_path / "oracle.yaml", ORACLE_AGENT, ORACLE_AGENT))
        assert "allowed_agents" not in oracle and oracle["type"] == "validation" and oracle["verified"] is True
        hermes = resolved_entry(AgentSelection(tmp_path / "agent.yaml", "hermes_agent", "hermes_agent"))
        assert list(hermes["allowed_agents"]) == ["terminus_2_sandboxed_agent", "hermes_agent"]
        # The dataset the preparation read is untouched: the next build_run for a harness sees the list.
        assert prepared.dataset.allowed_agents == ["terminus_2_sandboxed_agent", "hermes_agent"]

    def test_eval_run_refuses_a_disallowed_agent_before_any_work(self, tmp_path, monkeypatch, capsys):
        import sys

        import nemo_gym.cli.main as main_module
        import nemo_gym.global_config as global_config

        folder = tmp_path / "ds"
        write_task(folder / "good")
        (folder / "dataset.toml").write_text(TRAINING_TOML)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
        dispatched = []
        monkeypatch.setattr(main_module, "dispatch", lambda *args: dispatched.append(args))
        argv = ["gym", "eval", "run", str(folder), "--agent", "codex", "--model-type", "vllm_model"]
        argv += ["+policy_model_name=test-model", "+policy_base_url=http://localhost:1", "+policy_api_key=none"]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as exc:
            main_module.main()
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "allowed_agents admits terminus_2_sandboxed_agent, hermes_agent, not `--agent codex_agent`" in err
        # No servers were started and nothing was prepared or recorded for the run.
        assert dispatched == [] and not (tmp_path / "results").exists() and not (folder / "manifest.toml").exists()


def test_a_task_shipping_harbors_seed_trajectory_loads_for_the_oracle_and_the_file_stays_in_its_digest(
    tmp_path, caplog
):
    """Harbor 0.22.0 loads a task's `trajectory.json` (its `--load-trajectory` seed file) into the agent's session before
    the first turn, for every agent but the oracle and nop; Gym hands no prior context to a harness yet, so the loader
    records the file with a warning and the environment server runs the task for the oracle alone, refusing it for
    any other agent (see the server's tests). The file is still part of the task's identity: the folder digest and
    the package digest cover it, so a fetched task verifies against its advertised hash and the manifest records the
    hash with it."""
    from nemo_gym.tasks.harbor.digest import archive_hash, content_hash, is_covered

    root = write_task(tmp_path / "t")
    loaded = load_task(root)
    before = loaded.folder_digest
    assert loaded.seed_trajectory is None
    (root / "trajectory.json").write_text('{"schema_version": "ATIF-v1.7", "steps": []}')
    with caplog.at_level(logging.WARNING, logger="nemo_gym.tasks.harbor.task"):
        task = load_task(root)
    assert task.seed_trajectory == Path("trajectory.json")
    message = caplog.text
    assert "trajectory.json is Harbor's seed trajectory" in message and "not supported yet" in message
    assert "runs the task for the oracle alone" in message and "Remove the file" in message
    assert is_covered("trajectory.json") and content_hash(root) != before
    # The package store's hash (what a fetch verifies) covers the file as the folder digest does.
    assert archive_hash(root) == content_hash(root)
