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

import nemo_gym.tasks.harbor.build_options as build_options
import nemo_gym.tasks.harbor.hub as hub_module
import nemo_gym.tasks.harbor.manifest as manifest_module
import nemo_gym.tasks.harbor.prepare_builds as prepare_builds
from nemo_gym.tasks.harbor import DIGEST_KEY, HarborTaskConfig, content_hash, discover_tasks, load_task
from nemo_gym.tasks.harbor.cli import (
    AgentSelection,
    PreparedTaskset,
    build_run,
    prepare_target,
    resolve_agent,
    runs_in_sandbox,
)
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
from nemo_gym.tasks.harbor.materialize import materialize_task, run_config, write_rows
from nemo_gym.tasks.harbor.package_store import PackageStoreError
from nemo_gym.tasks.harbor.records import IMAGE_CONFIGS_FILE
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

    def test_multi_step_is_rejected_for_now(self):
        with pytest.raises(ValueError, match="Multi-step"):
            HarborTaskConfig.model_validate({"steps": [{"name": "a"}]})

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

    def test_write_rows_and_run_config(self, tmp_path):
        folder = tmp_path / "ds"
        tasks = [load_task(write_task(folder / "a")), load_task(write_task(folder / "b"))]
        rows_path = tmp_path / "out" / "tasks.jsonl"
        rows = write_rows(tasks, "ds", rows_path)
        assert [json.loads(line)["task_id"]["task_id"] for line in rows_path.read_text().splitlines()] == ["a", "b"]
        assert len(rows) == 2

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
        assert agent["responses_api_agents"]["hermes_agent"]["datasets"][0]["jsonl_fpath"] == str(rows_path.resolve())
        assert agent["responses_api_agents"]["hermes_agent"]["datasets"][0]["type"] == "validation"
        assert "resources_server" not in agent["responses_api_agents"]["hermes_agent"]
        assert agent["responses_api_agents"]["hermes_agent"]["enabled_toolsets"] == ["terminal"]
        assert "resources_servers" not in " ".join(config)


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
        assert "Multi-step" in prepared.skipped["bad"]
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
        assert "env_file 'missing.env' not found" in prepared.skipped["bad-env-file"]
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


@pytest.mark.parametrize("failure", ["encoding", "escape", "io"])
def test_discovery_keeps_file_errors_inside_task_boundary(tmp_path, monkeypatch, failure):
    write_task(tmp_path / "ds" / "good")
    bad = write_task(tmp_path / "ds" / "bad")
    if failure == "encoding":
        (bad / "environment" / "Dockerfile").write_bytes(b"\xff")
    elif failure == "escape":
        marker = tmp_path / "marker"
        marker.write_text("private marker")
        (bad / "tests" / "leak").symlink_to(marker)
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
    with pytest.raises(HarborTaskError, match="network.*not supported"):
        load_task(write_task(tmp_path / "t", toml=f"[{section}]\n{policy}\n"))


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
