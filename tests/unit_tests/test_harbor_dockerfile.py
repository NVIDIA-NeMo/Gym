# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of overlay Dockerfiles.

Overlay files resolved against recorded base image configurations, including the SWE-bench and
Terminal Bench shapes seen in the hub.
"""

import pytest

from nemo_gym.data import RECORD_VERSION, OverlayRun
from nemo_gym.data.tasks.harbor.dockerfile import (
    BaseImageConfig,
    DockerfileNeedsBuild,
    OverlayImage,
    base_image_only,
    overlay_image,
    parse_dockerfile,
    resolve_dockerfile,
)
from tests.unit_tests.test_harbor_tasks import IMAGE_PATH, TB21_DEBIAN_DOCKERFILE, dockerfile


SWEBENCH_VERIFIED_DOCKERFILE = """\
FROM swebench/sweb.eval.x86_64.django_1776_django-11099:latest

RUN cd /testbed && git status && git checkout -- . && git clean -fd
RUN /opt/miniconda3/bin/conda run -n testbed pip install pytest
RUN echo 'source /opt/miniconda3/bin/activate testbed' >> /root/.bashrc
"""

SWEBENCH_IMAGE_ENV = [f"PATH=/opt/miniconda3/bin:{IMAGE_PATH}", "LANG=C.UTF-8"]


class TestOverlayDockerfile:
    def test_cleared_entrypoint_is_recorded_and_a_real_one_needs_a_build(self):
        # ``ENTRYPOINT []`` is the one entrypoint instruction the loader accepts: it records that the base
        # image's entrypoint must not run. ``CMD []`` is accepted and changes nothing.
        parsed = parse_dockerfile("FROM ubuntu:24.04\nENTRYPOINT []\nWORKDIR /app\n")
        assert parsed.entrypoint_cleared is True
        assert parse_dockerfile("FROM ubuntu:24.04\nCMD []\n").entrypoint_cleared is False
        assert parse_dockerfile("FROM ubuntu:24.04\nWORKDIR /app\n").entrypoint_cleared is False
        resolved = resolve_dockerfile(
            parsed,
            BaseImageConfig.from_record({"record_version": RECORD_VERSION, "config": {"Entrypoint": ["/start.sh"]}}),
        )
        assert isinstance(resolved, OverlayImage) and resolved.entrypoint_cleared is True
        assert base_image_only("FROM ubuntu:24.04\nENTRYPOINT []\n").entrypoint_cleared is True
        assert base_image_only("FROM ubuntu:24.04\n").entrypoint_cleared is False
        with pytest.raises(DockerfileNeedsBuild, match="ENTRYPOINT"):
            parse_dockerfile('FROM ubuntu:24.04\nENTRYPOINT ["/start.sh"]\n')
        with pytest.raises(DockerfileNeedsBuild, match="ENTRYPOINT"):
            parse_dockerfile("FROM ubuntu:24.04\nENTRYPOINT /start.sh\n")
        with pytest.raises(DockerfileNeedsBuild, match="ENTRYPOINT"):
            parse_dockerfile('FROM ubuntu:24.04\nENTRYPOINT [""]\n')

    def test_base_image_config_records_entrypoint_and_healthcheck(self):
        health = {"Test": ["CMD-SHELL", "curl -f http://localhost/"], "Interval": 10_000_000_000}
        config = BaseImageConfig.from_record(
            {
                "record_version": RECORD_VERSION,
                "config": {"Entrypoint": ["/entrypoint.sh", "--flag"], "Healthcheck": health, "User": "app"},
            }
        )
        assert config.entrypoint == ("/entrypoint.sh", "--flag") and config.healthcheck == health
        assert config.user == "app"
        empty = BaseImageConfig.from_record(
            {"record_version": RECORD_VERSION, "config": {"Entrypoint": [], "Healthcheck": {}, "Env": ["A=1"]}}
        )
        assert empty.entrypoint is None and empty.healthcheck is None and empty.env == {"A": "1"}
        assert BaseImageConfig.from_record({"record_version": RECORD_VERSION}).entrypoint is None

    def test_swebench_verified_shape(self):
        base = BaseImageConfig.from_record(
            {"record_version": RECORD_VERSION, "config": {"Env": SWEBENCH_IMAGE_ENV, "WorkingDir": "/testbed"}}
        )
        overlay = overlay_image(SWEBENCH_VERIFIED_DOCKERFILE, base)
        assert overlay is not None
        assert overlay.image == "swebench/sweb.eval.x86_64.django_1776_django-11099:latest"
        # The Dockerfile sets nothing itself: the image's working directory and user stand.
        assert overlay.workdir == "/testbed" and overlay.env == {} and overlay.user is None
        assert [run.command for run in overlay.runs] == [
            "cd /testbed && git status && git checkout -- . && git clean -fd",
            "/opt/miniconda3/bin/conda run -n testbed pip install pytest",
            "echo 'source /opt/miniconda3/bin/activate testbed' >> /root/.bashrc",
        ]
        # Every RUN sees the whole image environment, resolved, and runs in the image's directory as root.
        image_env = {"PATH": f"/opt/miniconda3/bin:{IMAGE_PATH}", "LANG": "C.UTF-8"}
        assert all(run.workdir == "/testbed" and run.env == image_env and run.user is None for run in overlay.runs)
        # The same file is not pull mode.
        assert base_image_only(SWEBENCH_VERIFIED_DOCKERFILE) is None

    def test_tb21_debian_shape_applies_state_in_order(self):
        overlay = dockerfile(TB21_DEBIAN_DOCKERFILE, Env=[f"PATH={IMAGE_PATH}"])
        assert overlay.image == "debian:bookworm-slim"
        first, second = overlay.runs
        # The apt line runs before WORKDIR and ENV exist; continuations are joined.
        assert " ".join(first.command.split()) == (
            "apt-get update && apt-get install -y --no-install-recommends curl tmux && rm -rf /var/lib/apt/lists/*"
        )
        assert first.workdir == "/" and first.env == {"PATH": IMAGE_PATH}
        # The later RUN sees both, with `$PATH` already expanded, and the final image settings are recorded too.
        assert second == OverlayRun(
            command="mkdir -p /opt/tools/bin && echo ready > /app/ready.txt",
            workdir="/app",
            env={"PATH": f"/opt/tools/bin:{IMAGE_PATH}", "DEBIAN_FRONTEND": "noninteractive"},
        )
        assert overlay.workdir == "/app" and overlay.env == second.env

    def test_user_and_relative_workdir_apply_to_later_runs_only(self):
        overlay = dockerfile(
            "FROM img\nRUN id\nUSER agent\nWORKDIR /home/agent\nWORKDIR src\nRUN make\nENV LATE=1\n",
            Env=[f"PATH={IMAGE_PATH}"],
            WorkingDir="/srv",
            User="svc",
        )
        assert [(r.command, r.user, r.workdir) for r in overlay.runs] == [
            ("id", "svc", "/srv"),
            ("make", "agent", "/home/agent/src"),
        ]
        assert overlay.user == "agent" and overlay.workdir == "/home/agent/src" and overlay.env == {"LATE": "1"}
        # ENV after the last RUN reaches the image, not the RUN.
        assert overlay.runs[1].env == {"PATH": IMAGE_PATH}

    def test_exec_form_run_becomes_a_shell_command(self):
        overlay = dockerfile('FROM img\nRUN ["bash", "-lc", "echo hi there"]\n')
        assert overlay.runs[0].command == "bash -lc 'echo hi there'"
        # Exec form runs its arguments directly: no shell, whatever the image's Shell is.
        assert overlay.runs[0].shell is None

    def test_the_image_shell_runs_shell_form_lines_only(self):
        image = dockerfile(
            'FROM img\nRUN false | true\nRUN ["bash", "-lc", "echo hi"]\n', Shell=["/bin/bash", "-o", "pipefail", "-c"]
        )
        assert image.runs[0].shell == ("/bin/bash", "-o", "pipefail", "-c") and image.runs[1].shell is None
        assert dockerfile("FROM img\nRUN true\n").runs[0].shell == ("/bin/sh", "-c")

    def test_workdir_instructions_are_recorded_in_order_with_the_user_in_effect(self):
        image = dockerfile(
            "FROM img\nWORKDIR /tmp/a\nWORKDIR /\nRUN one\nUSER agent\nWORKDIR /home/agent\nWORKDIR src\n"
            "RUN two\nWORKDIR /last\n"
        )
        assert [(w.path, w.user, w.before) for w in image.workdirs] == [
            ("/tmp/a", None, 0),
            ("/", None, 0),
            ("/home/agent", "agent", 1),
            ("/home/agent/src", "agent", 1),
            ("/last", "agent", 2),
        ]
        assert image.workdir == "/last" and len(image.runs) == 2

    def test_a_record_from_an_older_gym_is_refused(self):
        with pytest.raises(ValueError, match="prepare the dataset again"):
            BaseImageConfig.from_record({"config": {"User": "svc"}})

    @pytest.mark.parametrize(
        ("text", "named"),
        [
            ("FROM ubuntu\nRUN <<EOF\napt-get update\nEOF\n", "RUN with a heredoc"),
            ("FROM ubuntu\nRUN python3 <<-'PY'\nprint(1)\nPY\n", "RUN with a heredoc"),
            ("FROM ubuntu\nRUN --mount=type=cache,target=/root/.cache pip install x", "RUN --mount"),
            ("FROM ubuntu\nCOPY . /app\nRUN true", "COPY"),
            ("FROM ubuntu\nRUN true\nADD x.tar /app", "ADD"),
            ("FROM a AS one\nRUN true\nFROM b\nRUN true", "a second FROM (multi-stage build)"),
            ("ARG BASE=ubuntu\nFROM $BASE\nRUN true", "ARG before FROM"),
            ("FROM $BASE\nRUN true", "FROM $BASE (ARG-templated image)"),
            ('FROM ubuntu\nRUN true\nENTRYPOINT ["/start.sh"]', "ENTRYPOINT"),
            ("FROM ubuntu\nRUN true\nCMD sleep infinity", "CMD"),
            ("FROM ubuntu\nHEALTHCHECK CMD curl -f http://localhost/", "HEALTHCHECK"),
            ("FROM ubuntu\nEXPOSE 8080", "EXPOSE"),
            ("FROM ubuntu\nVOLUME /data", "VOLUME"),
            ('FROM ubuntu\nSHELL ["/bin/bash", "-c"]', "SHELL"),
            ("FROM ubuntu\nSTOPSIGNAL SIGTERM", "STOPSIGNAL"),
            ("RUN true", "RUN before FROM"),
            ("# only comments\n", "no FROM instruction"),
        ],
    )
    def test_everything_else_is_rejected_naming_the_instruction(self, text, named):
        with pytest.raises(DockerfileNeedsBuild) as excinfo:
            parse_dockerfile(text)
        assert str(excinfo.value).startswith(named), str(excinfo.value)
        assert overlay_image(text) is None and base_image_only(text) is None

    def test_here_strings_shifts_and_quoted_arrows_are_not_heredocs(self):
        text = "FROM img\nRUN cat <<< hello\nRUN echo $((1<<2))\nRUN echo '<<EOF' \"<<EOF\"\n"
        assert [command for _, command in parse_dockerfile(text).instructions] == [
            "cat <<< hello",
            "echo $((1<<2))",
            "echo '<<EOF' \"<<EOF\"",
        ]
        with pytest.raises(DockerfileNeedsBuild, match="heredoc"):
            parse_dockerfile("FROM img\nRUN echo $((1<<2)) && cat <<EOF\n4\nEOF\n")
