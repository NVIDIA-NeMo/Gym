# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M30 at the server: the job-level Compose overlay turns an image-only task into a group with the overlay's
services, its MCP entries reach the harness merged with the task's in the task context and the grant list, the
prompt template wraps the first turn's instruction, and the merged document stays private."""

import json
from pathlib import Path

import pytest

from environment_servers.harbor import app as harbor_app
from environment_servers.harbor.tests.conftest import (
    COMPOSE_YAML,
    IMAGE_CONFIGS,
    TASK_TOML,
    FakeCompose,
    FakeSandbox,
    episode,
    make_server,
    run_episode,
    write_image_config,
)
from environment_servers.harbor.tests.test_app import COMPOSE_TOML
from environment_servers.harbor.tests.test_multi_turn import MCP_TOML, THREE_STEPS, multi_turn_server
from nemo_gym.tasks.harbor.compose_overlay import HarborComposeOverlay
from nemo_gym.tasks.harbor.dockerfile import RECORD_VERSION
from nemo_gym.tasks.harbor.image_configs import record_compose_images
from nemo_gym.tasks.harbor.materialize import materialize_task
from nemo_gym.tasks.harbor.task import load_task


# A memory stack the way the memory team ships one: a standalone file with host ports and a global network name,
# then a `.harbor.yaml` glue that resets both and attaches `main`.
STANDALONE = """
services:
  memory:
    image: org/memory:1
    ports:
      - "9050:9000"
    volumes:
      - memory_data:/data
    restart: unless-stopped
    networks:
      - memory_net
volumes:
  memory_data:
networks:
  memory_net:
    name: ${MEMORY_NETWORK_NAME:-memory_net}
"""
GLUE = """
services:
  memory:
    ports: !reset []
  main:
    depends_on:
      memory:
        condition: service_healthy
    networks:
      - default
      - memory_net
networks:
  memory_net:
    name: !reset null
"""
TEMPLATE = "You have a memory server.\n\n## Task Instruction\n\n{{ instruction }}\n"


def memory_overlay(
    tmp_path: Path, *, template: bool = False, mcp: bool = True, files: bool = True
) -> HarborComposeOverlay:
    folder = tmp_path / "overlay"
    folder.mkdir(exist_ok=True)
    (folder / "memory.yaml").write_text(STANDALONE)
    (folder / "memory.harbor.yaml").write_text(GLUE)
    (folder / "memory_prompt.md.j2").write_text(TEMPLATE)
    return HarborComposeOverlay(
        files=[folder / "memory.yaml", folder / "memory.harbor.yaml"] if files else [],
        mcp_servers=[{"name": "memory", "transport": "streamable-http", "url": "http://memory:9000/mcp"}]
        if mcp
        else [],
        prompt_template=folder / "memory_prompt.md.j2" if template else None,
    )


def overlay_server(
    tmp_path, monkeypatch, *, toml: str = TASK_TOML, overlay: HarborComposeOverlay | None = None, **config
):
    """An image-only task (the conftest's pull-mode Dockerfile) with the overlay's sidecar image recorded, and a
    ``_create_compose`` stub that keeps the resolved document and hands back a fake group."""
    server, task, sandbox, created = make_server(tmp_path, monkeypatch, toml=toml, **config)
    write_image_config(task.path.parent, "org/memory:1", Cmd=["memory-server"], User="memory")
    server.config.compose_overlay = overlay if overlay is not None else memory_overlay(tmp_path)
    monkeypatch.setattr(
        "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
    )
    documents: list[dict] = []
    memory = FakeSandbox()
    compose = FakeCompose({"main": sandbox, "memory": memory})

    async def start_compose(task, episode_key):
        documents.append(server._compose_document(task, episode_key))
        sandbox.created_in(task.workdir)
        created.append((task.task_id, task.workdir))
        return compose

    monkeypatch.setattr(server, "_create_compose", start_compose)
    return server, task, sandbox, compose, documents


class TestGroup:
    def test_an_image_only_task_becomes_the_one_service_group_plus_the_overlay_services(self, tmp_path, monkeypatch):
        server, task, _, _, _ = overlay_server(tmp_path, monkeypatch)
        assert not task.needs_compose and server._uses_compose(task)

        document = server._compose_document(task)

        assert list(document["services"]) == ["main", "memory"]
        main = document["services"]["main"]
        # Harbor's own `main`: the task image, Harbor's keep-alive command, the glue's dependency and network.
        assert main["image"] == "ubuntu:24.04" and main["command"] == ["sh", "-c", "sleep infinity"]
        assert main["depends_on"] == {"memory": {"condition": "service_healthy"}}
        assert main["networks"] == {"default": {}, "memory_net": {}}
        memory = document["services"]["memory"]
        # The overlay's service with its recorded start, the standalone file's host ports reset by the glue, its
        # named volume kept for the provider's shared storage, and the network declared without the global name.
        # The non-root sidecar keeps its image user (not copied onto the service) and gets no host injection.
        assert memory["command"] == ["memory-server"] and "user" not in memory
        assert memory["x-sandbox"] == {"hosts": []}
        assert "ports" not in memory and memory["restart"] == "unless-stopped"
        assert memory["volumes"] == [
            {"type": "volume", "source": "memory_data", "target": "/data", "read_only": False}
        ]
        assert document["volumes"] == {"memory_data": {}} and document["networks"] == {"memory_net": {}}

    def test_without_overlay_files_an_image_only_task_stays_a_single_sandbox(self, tmp_path, monkeypatch):
        server, task, _, _, _ = overlay_server(tmp_path, monkeypatch, overlay=memory_overlay(tmp_path, files=False))
        assert not server._uses_compose(task)
        server.config.compose_overlay = None
        assert not server._uses_compose(task)

    async def test_the_episode_seeds_the_group_and_the_overlay_services_stop_with_it(self, tmp_path, monkeypatch):
        server, task, sandbox, compose, documents = overlay_server(tmp_path, monkeypatch)

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0
        assert len(documents) == 1 and set(documents[0]["services"]) == {"main", "memory"}
        # The group, overlay services included, was stopped with the episode and released.
        assert compose.stopped and not server._sandboxes and not server._sessions

    async def test_the_merged_document_stays_private(self, tmp_path, monkeypatch):
        server, task, _, _, _ = overlay_server(tmp_path, monkeypatch)
        started: list[tuple[Path, str]] = []

        class RecordingCompose:
            def __init__(self, provider, path, *, service_specs, timeout_s, **adapter_kwargs):
                self.path = Path(path)
                self.services = {name: object() for name in service_specs}

            async def start(self):
                started.append((self.path, self.path.read_text()))

        monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandboxCompose", RecordingCompose)

        await harbor_app.HarborEnvironmentServer._create_compose(server, task, "r1")

        ((path, text),) = started
        assert "org/memory" in text
        assert not path.exists() and not path.is_relative_to(server._artifacts_dir)
        assert not list(server._artifacts_dir.rglob("compose.yaml"))


class TestSeedPrecheck:
    async def test_an_overlay_image_prepare_did_not_record_is_refused_terminally_before_any_group(
        self, tmp_path, monkeypatch
    ):
        """A dataset prepared without this ``compose_overlay`` has no record for the overlay's images: a configuration
        error the seed refuses (422, terminal) naming the services, not a group start that fails and is retried."""
        server, task, _, compose, documents = overlay_server(tmp_path, monkeypatch)
        extra = tmp_path / "overlay" / "extra.yaml"
        extra.write_text("services:\n  extra:\n    image: org/extra:1\n  other:\n    image: org/other:1\n")
        server.config.compose_overlay = HarborComposeOverlay(files=[*server.config.compose_overlay.files, extra])

        response = await run_episode(server, task)

        assert response.result is None and response.failure.stage == "seed" and response.failure.terminal
        reason = response.failure.failure_reason
        assert "service 'extra' ('org/extra:1')" in reason and "service 'other' ('org/other:1')" in reason
        assert "'memory'" not in reason and "prepare the dataset again with this compose_overlay" in reason
        assert documents == [] and not compose.stopped and not server._sandboxes and not server._sessions

    @pytest.mark.parametrize(
        ("patch", "was"),
        [
            ({"architecture": "arm64"}, f"linux/arm64, record version {RECORD_VERSION}"),
            ({"record_version": RECORD_VERSION - 1}, f"linux/amd64, record version {RECORD_VERSION - 1}"),
        ],
        ids=["other-platform", "stale-record-version"],
    )
    async def test_a_record_this_deployment_cannot_use_is_refused_terminally_before_any_group(
        self, tmp_path, monkeypatch, patch, was
    ):
        """A record that exists but is not reusable for this deployment (another platform: the dataset prepared on x86
        and served on arm64; or an old record version) is refused by prepare's own rule at seed, terminally, naming
        what was recorded, where ``resolve_compose`` would raise inside the group's start and be retried every
        episode."""
        server, task, _, compose, documents = overlay_server(tmp_path, monkeypatch)
        record = task.path.parent / "compose-images.json"
        recorded = json.loads(record.read_text())
        recorded["org/memory:1"].update(patch)
        record.write_text(json.dumps(recorded))

        response = await run_episode(server, task)

        assert response.result is None and response.failure.stage == "seed" and response.failure.terminal
        reason = response.failure.failure_reason
        assert reason.startswith(
            "Compose group of task 'hello': no image configuration this deployment can use recorded in "
        )
        assert (
            f"for service 'memory' ('org/memory:1' is recorded for {was}; this deployment runs linux/amd64 and reads "
            f"record version {RECORD_VERSION}); prepare the dataset again with this compose_overlay, for this platform"
        ) in reason
        assert "'main'" not in reason
        assert documents == [] and not compose.stopped and not server._sandboxes and not server._sessions

    async def test_a_missing_image_record_is_refused_terminally(self, tmp_path, monkeypatch):
        """An image-only dataset prepared before the overlay was set has no record file at all."""
        server, task, _, compose, documents = overlay_server(tmp_path, monkeypatch)
        record = task.path.parent / "compose-images.json"
        record.rename(record.with_suffix(".json.aside"))

        response = await run_episode(server, task)

        assert response.result is None and response.failure.stage == "seed" and response.failure.terminal
        reason = response.failure.failure_reason
        assert reason.startswith("Compose group of task 'hello' needs the image configurations prepare records at ")
        assert reason.endswith(
            "compose-images.json; prepare the dataset again, with this compose_overlay when one is set"
        )
        assert documents == [] and not compose.stopped and not server._sandboxes and not server._sessions

    @pytest.mark.parametrize(
        ("text", "reason"),
        [
            (
                "services:\n  memory:\n    privileged: true\n",
                "Service 'memory': unsupported Compose fields ['privileged']",
            ),
            (
                "services:\n  memory:\n    volumes: ['{outside}/conf:/etc/memory']\n",
                "from outside the task; map it under `volume_sources` in the sandbox block",
            ),
            (
                # Relative to the project directory, the task's environment folder, as under Harbor; not to the
                # overlay file's folder, where a ``conf`` exists.
                "services:\n  memory:\n    volumes: ['./conf:/etc/memory']\n",
                "binds './conf', which the task folder does not contain",
            ),
            ("services: !reset\n", "the merged document has no services mapping"),
            (
                # The recorded org/memory:1 runs as user `memory`: the group starts it without host injection, so
                # prepare refuses host entries on it once the images are recorded; the seed refuses them the same way.
                "services:\n  memory:\n    extra_hosts: ['licence.example:198.51.100.7']\n",
                "service 'memory' extra_hosts cannot be applied without host injection: the image runs as user "
                "'memory', so the sidecar takes no host entries",
            ),
        ],
        ids=["field", "foreign-bind-source", "relative-bind-source", "no-services", "extra-hosts-on-non-root-sidecar"],
    )
    async def test_an_overlay_the_deployment_cannot_run_is_refused_terminally_before_any_group(
        self, tmp_path, monkeypatch, text, reason
    ):
        """The checks prepare ran (field check, volume planner, the non-root sidecar's extra_hosts rule) run at seed
        on the merged document too: an overlay file that no longer passes them (the overlay is a run-config value,
        changed between prepare and serve) is a 422, terminal, naming the overlay files, not a failure inside the
        group's start retried for every episode. Prepare, run on the same overlay, skips the task with the same
        reason."""
        server, task, _, compose, documents = overlay_server(tmp_path, monkeypatch)
        (tmp_path / "overlay" / "conf").mkdir()
        extra = tmp_path / "overlay" / "extra.yaml"
        extra.write_text(text.format(outside=tmp_path / "overlay"))
        server.config.compose_overlay = HarborComposeOverlay(files=[*server.config.compose_overlay.files, extra])

        response = await run_episode(server, task)

        assert response.result is None and response.failure.stage == "seed" and response.failure.terminal
        failure_reason = response.failure.failure_reason
        assert failure_reason.startswith("Compose group of task 'hello': hello: ") or failure_reason.startswith(
            "Compose file of task 'hello': hello: "
        )
        assert reason in failure_reason and str(extra) in failure_reason
        assert documents == [] and not compose.stopped and not server._sandboxes and not server._sessions
        skipped: dict[str, str] = {}
        record_compose_images(
            [task],
            task.path.parent,
            skipped=skipped,
            overlay=server.config.compose_overlay,
            sandbox_provider="opensandbox",
        )
        assert set(skipped) == {"hello"} and failure_reason.endswith(skipped["hello"]), skipped

    async def test_an_overlay_that_resets_a_service_the_tasks_hooks_name_is_refused_as_prepare_refuses_it(
        self, tmp_path, monkeypatch
    ):
        """`services: {kafka: !reset}` against a task whose `[[verifier.collect]]` hook runs in `kafka`: the task is
        valid on its own and the overlay merges, but the group would start without the service the hook needs.
        Prepare skips the task naming the overlay files and the service, and the seed refuses it terminally with the
        same reason, before any group (the check used to live only inside the group's start, retried per episode)."""
        server, task, _, _ = make_server(tmp_path, monkeypatch)
        (task.path / "task.toml").write_text(COMPOSE_TOML)
        (task.path / "environment" / "docker-compose.yaml").write_text(COMPOSE_YAML)
        (task.path.parent / "compose-images.json").write_text(json.dumps(IMAGE_CONFIGS))
        task = load_task(task.path)
        assert [hook.service for hook in task.config.verifier.collect] == ["kafka"]
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict",
            lambda: {"sandbox": {"opensandbox": {}}, "sandbox_gpu": {"opensandbox": {}}},
        )
        reset = tmp_path / "overlay" / "drop_kafka.yaml"
        reset.parent.mkdir()
        reset.write_text("services:\n  kafka: !reset\n")
        server.config.compose_overlay = HarborComposeOverlay(files=[reset])
        documents: list[dict] = []
        compose = FakeCompose({"main": FakeSandbox(), "api": FakeSandbox()})

        async def start_compose(task, episode_key):
            documents.append(server._compose_document(task, episode_key))
            return compose

        monkeypatch.setattr(server, "_create_compose", start_compose)
        skipped: dict[str, str] = {}
        record_compose_images(
            [task],
            task.path.parent,
            skipped=skipped,
            overlay=server.config.compose_overlay,
            sandbox_provider="opensandbox",
        )
        reason = (
            f"hello: environment/docker-compose.yaml merged with Compose overlay files [{reset}]: artifacts or collect "
            "hooks name Compose services the group does not have: ['kafka']"
        )
        assert skipped == {"hello": reason}

        response = await run_episode(server, task)

        assert response.result is None and response.failure.stage == "seed" and response.failure.terminal
        assert response.failure.failure_reason == f"Compose group of task 'hello': {reason}"
        assert documents == [] and not compose.stopped and not server._sandboxes and not server._sessions
        # Without the reset file the task's own group has the service, and the same checks pass the document.
        server.config.compose_overlay = None
        assert set(server._compose_document(task)["services"]) == {"main", "api", "kafka"}


class TestMcpGrants:
    async def test_overlay_entries_reach_the_harness_merged_with_the_tasks_in_the_same_list(
        self, tmp_path, monkeypatch
    ):
        overlay = HarborComposeOverlay(
            mcp_servers=[
                # Replaces the task's `legacy` entry by name (Harbor's rule) and adds `memory`.
                {"name": "legacy", "transport": "streamable-http", "url": "http://legacy:9000/mcp"},
                {"name": "memory", "transport": "sse", "url": "http://memory:9000/sse"},
            ]
        )
        server, task, sandbox, _, _ = overlay_server(
            tmp_path, monkeypatch, toml=TASK_TOML + MCP_TOML, overlay=overlay, task_mcp_grants="optional"
        )

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        context_write = next(c for c in sandbox.execs if "/tmp/.nemo-gym/task.json" in c["command"])
        written = json.loads(context_write["command"].split("printf %s ", 1)[1].split(" > ", 1)[0].strip("'"))
        assert [(s["name"], s["transport"], s["url"]) for s in written["mcp_servers"]] == [
            ("user-server", "streamable-http", "http://user-server:8000/mcp"),
            ("legacy", "streamable-http", "http://legacy:9000/mcp"),
            ("memory", "sse", "http://memory:9000/sse"),
        ]
        (seed,) = server.server_client.seed_bodies()
        # One grant list: the task's streamable-http entry and the overlay's; the sse entry cannot be granted.
        assert [(a["name"], a["connection"]["url"]) for a in seed["tool_accesses"]] == [
            ("user-server", "http://user-server:8000/mcp"),
            ("legacy", "http://legacy:9000/mcp"),
        ]

    async def test_an_overlay_entry_that_cannot_be_granted_is_warned_about_once_per_run(
        self, tmp_path, monkeypatch, caplog
    ):
        """A job-level `sse` entry (the memory team's Redis stack is `sse` by design) is the operator's choice for the
        whole run: one warning per entry per run, not one per turn of every episode. The task's own `sse` entry keeps
        its warning at every seed, the task author's signal. Both still reach the harness in task.json."""
        overlay = HarborComposeOverlay(
            mcp_servers=[{"name": "redis-memory", "transport": "sse", "url": "http://mcp:9000/sse"}]
        )
        server, task, sandbox, _, _ = overlay_server(
            tmp_path, monkeypatch, toml=TASK_TOML + MCP_TOML, overlay=overlay, task_mcp_grants="optional"
        )
        caplog.set_level("WARNING", logger="environment_servers.harbor.app")

        for rollout in ("r1", "r2", "r3"):
            assert (await run_episode(server, task, rollout=rollout)).failure is None

        warnings = [record.getMessage() for record in caplog.records if "cannot be granted" in record.getMessage()]
        assert (
            warnings.count("hello: MCP server 'legacy' (sse) cannot be granted; only streamable-http URL entries can")
            == 3
        )
        assert (
            warnings.count(
                "compose_overlay: MCP server 'redis-memory' (sse) cannot be granted; only streamable-http URL entries can; "
                "it reaches the harness in /tmp/.nemo-gym/task.json (said once per run)"
            )
            == 1
        )
        assert len(warnings) == 4
        for seed in server.server_client.seed_bodies():
            assert [a["name"] for a in seed["tool_accesses"]] == ["user-server"]
        writes = [c["command"] for c in sandbox.execs if "/tmp/.nemo-gym/task.json" in c["command"]]
        assert len(writes) == 3 and all("http://mcp:9000/sse" in w and "http://legacy:9000/sse" in w for w in writes)

    async def test_an_overlay_only_sse_entry_is_refused_under_required_grants(self, tmp_path, monkeypatch):
        """The grant and the refusal read the same merged list: a job-level `sse` entry cannot be granted, so a
        harness that reads no task.json is refused naming it even when `task_mcp_grants: required` granted every
        streamable-http server, and an overlay-only streamable-http entry granted as required lets the task run."""
        overlay = HarborComposeOverlay(
            mcp_servers=[{"name": "redis-memory", "transport": "sse", "url": "http://mcp:9000/sse"}]
        )
        server, task, sandbox, _, _ = overlay_server(
            tmp_path, monkeypatch, overlay=overlay, task_mcp_grants="required"
        )
        assert not task.config.environment.mcp_servers
        server.server_client.supports_task_tool_context = False

        response = await run_episode(server, task)

        failure = response.failure
        assert response.result is None and failure.stage == "agent" and failure.terminal
        assert "MCP server 'redis-memory'" in failure.failure_reason
        assert "task_mcp_grants: required" not in failure.failure_reason
        assert not any("test.sh" in call["command"] for call in sandbox.execs)

        (tmp_path / "ok").mkdir()
        granted = overlay_server(
            tmp_path / "ok",
            monkeypatch,
            overlay=memory_overlay(tmp_path / "ok", files=False),
            task_mcp_grants="required",
        )
        server2, task2, _, _, _ = granted
        server2.server_client.supports_task_tool_context = False
        response = await run_episode(server2, task2, rollout="r2")
        assert response.failure is None, response.failure
        (seed,) = server2.server_client.seed_bodies()
        assert [(a["name"], a["required"]) for a in seed["tool_accesses"]] == [("memory", True)]

    async def test_overlay_entries_alone_make_the_task_one_with_tools(self, tmp_path, monkeypatch):
        server, task, sandbox, _, _ = overlay_server(
            tmp_path, monkeypatch, overlay=memory_overlay(tmp_path, files=False)
        )
        assert not task.config.environment.mcp_servers
        server.server_client.supports_task_tool_context = False

        response = await run_episode(server, task)

        # A harness that cannot drive MCP servers is refused, exactly as for a task's own entries.
        assert response.result is None and response.failure.stage == "agent" and response.failure.terminal
        assert "declares MCP servers or a skills directory" in response.failure.failure_reason
        server.server_client.supports_task_tool_context = True
        response = await run_episode(server, task, rollout="r2")
        assert response.failure is None
        context_write = next(c for c in sandbox.execs if "/tmp/.nemo-gym/task.json" in c["command"])
        assert "http://memory:9000/mcp" in context_write["command"]


class TestPromptTemplate:
    async def test_the_first_turn_instruction_is_rendered_into_the_template(self, tmp_path, monkeypatch):
        server, task, _, _, _ = overlay_server(tmp_path, monkeypatch, overlay=memory_overlay(tmp_path, template=True))

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        rendered = "You have a memory server.\n\n## Task Instruction\n\nCreate hello.txt"
        (turn,) = [call[2]["json"] for call in server.server_client.calls if call[1].endswith("/v1/responses")]
        assert [(item.role, item.content) for item in turn.input] == [("user", rendered)]
        # The request to the harness goes out as `ServerClient.post` serializes it (set fields only): the row's own
        # keys with the rendered content, none of the model's defaults (`tools: []`, `tool_choice`, `phase`, ...)
        # made explicit by the rendering round trip.
        row = episode(task).task.task_input.responses_create_params.model_dump(exclude_unset=True)
        assert row == {"input": [{"role": "user", "content": "Create hello.txt"}]}
        assert turn.model_dump(exclude_unset=True) == {"input": [{"role": "user", "content": rendered}]}
        # The result and the transcript carry what the harness saw.
        assert response.result.responses_create_params.input[0].content == rendered

    async def test_only_the_first_turn_of_a_multi_step_task_is_wrapped(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = multi_turn_server(
            tmp_path, monkeypatch, THREE_STEPS, {"plan": "Plan it.", "build": "Build it.", "check": "Check it."}
        )
        server.config.compose_overlay = memory_overlay(tmp_path, template=True, files=False, mcp=False)
        # The request as the wire carries it (set fields only): each turn's params are `input` and `metadata`.
        request = harbor_app.HarborEpisodeRequest.model_validate(
            {
                "episode_id": {"rollout_id": "r1"},
                "task": materialize_task(task, "ds").model_dump(mode="json", exclude_unset=True),
            }
        )
        rows = [turn.responses_create_params.model_dump(exclude_unset=True) for turn in request.task.task_input.turns]
        assert [set(row) for row in rows] == [{"input", "metadata"}] * 3

        response = await server.run_request(request)

        assert response.failure is None, response.failure
        bodies = [
            call[2]["json"].model_dump(exclude_unset=True)
            for call in server.server_client.calls
            if call[1].endswith("/v1/responses")
        ]
        turns = [body["input"][0]["content"] for body in bodies]
        assert turns == ["You have a memory server.\n\n## Task Instruction\n\nPlan it.", "Build it.", "Check it."]
        # Each turn reaches the harness as the row wrote it; the rendered first turn differs in its content only
        # (no default of the params model made explicit by the rendering round trip).
        assert bodies == [{**rows[0], "input": [{**rows[0]["input"][0], "content": turns[0]}]}, rows[1], rows[2]]
        # The multi-turn record keeps the last turn's params, so the rendered first turn is in its own transcript
        # only (the turn results carry no params; recording them is a follow-up with M34).
        assert response.result.responses_create_params.input[0].content == "Check it."
        assert [turn.name for turn in response.result.turns] == ["plan", "build", "check"]
        assert "memory server" not in json.dumps(response.result.model_dump(mode="json")["turns"])

    async def test_a_template_that_does_not_render_fails_the_episode_before_any_sandbox(self, tmp_path, monkeypatch):
        # The block compiles its template when it is read (an undefined variable passes then, as Harbor's
        # StrictUndefined only fails the render), so the bad text is written before the block is built.
        template = tmp_path / "overlay" / "bad_prompt.md.j2"
        template.parent.mkdir(exist_ok=True)
        template.write_text("{{ instruction }} for {{ user_name }}\n")
        overlay = HarborComposeOverlay(prompt_template=template)
        server, task, _, _, _ = overlay_server(tmp_path, monkeypatch, overlay=overlay)
        created = []
        monkeypatch.setattr(server, "_create_sandbox", lambda *a, **k: created.append(a))

        response = await run_episode(server, task)

        assert response.result is None and response.failure.stage == "seed" and response.failure.terminal
        assert (
            "undefined variables" in response.failure.failure_reason and "user_name" in response.failure.failure_reason
        )
        assert created == [] and server.server_client.paths() == [] and not server._sandboxes

    def test_the_block_is_a_deployment_setting_of_the_server(self, tmp_path, monkeypatch):
        server, _, _, _, _ = overlay_server(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="Prompt template not found"):
            HarborComposeOverlay(prompt_template=tmp_path / "missing.j2")
        assert server.config.compose_overlay.configured
