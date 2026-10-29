# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of Compose volume planning.

Short and long volume forms, bind sources under the task folder, refused options, link
resolution and the plan the adapter receives, with the resolver run over the planned document.
"""

import copy
import hashlib
import shutil
from pathlib import Path

import pytest
import yaml

from nemo_gym.sandbox import adapt_non_root_sidecars, document_volume_sources, resolve_compose
from nemo_gym.sandbox.compose_config import plan_compose_volumes
from nemo_gym.sandbox.compose_volumes import ComposeVolumeError, parse_volume
from tests.unit_tests.compose_testing import IMAGE_CONFIGS, MAIN, OVERLAY, SIDECAR


VOLUME_FIXTURES = Path(__file__).parent / "fixtures" / "harbor_compose_volumes"


class TestComposeVolumes:
    """M22: bind mounts of task files become uploads, named volumes stay for shared storage, external ones map
    through ``volume_sources``; anything else fails naming the field."""

    @pytest.mark.parametrize(
        "entry,expected",
        [
            (
                "./seed.json:/app/seed.json:ro",
                {"type": "bind", "source": "./seed.json", "target": "/app/seed.json", "read_only": True},
            ),
            (
                "/var/lib/models:/models",
                {"type": "bind", "source": "/var/lib/models", "target": "/models", "read_only": False},
            ),
            ("~/cache:/cache:rw,z", {"type": "bind", "source": "~/cache", "target": "/cache", "read_only": False}),
            (
                "pgdata:/var/lib/postgresql/data",
                {"type": "volume", "source": "pgdata", "target": "/var/lib/postgresql/data", "read_only": False},
            ),
            (
                "pgdata:/data:nocopy",
                {
                    "type": "volume",
                    "source": "pgdata",
                    "target": "/data",
                    "read_only": False,
                    "volume": {"nocopy": True},
                },
            ),
            (
                {
                    "type": "bind",
                    "source": "./data",
                    "target": "/data",
                    "read_only": True,
                    "bind": {"create_host_path": True},
                },
                {"type": "bind", "source": "./data", "target": "/data", "read_only": True},
            ),
            (
                {"source": "pgdata", "target": "/data", "volume": {"nocopy": True}, "consistency": "cached"},
                {
                    "type": "volume",
                    "source": "pgdata",
                    "target": "/data",
                    "read_only": False,
                    "volume": {"nocopy": True},
                },
            ),
            # Null and empty option blocks are the mapping's absence (`bind:` with no body is how YAML spells it).
            (
                {"type": "bind", "source": "./data", "target": "/data", "bind": None, "volume": {}},
                {"type": "bind", "source": "./data", "target": "/data", "read_only": False},
            ),
        ],
    )
    def test_parse_volume_handles_short_and_long_syntax(self, entry, expected):
        assert parse_volume(entry, "api") == expected

    @pytest.mark.parametrize(
        "entry,name",
        [
            ("/data", "/data"),  # anonymous volume
            (":/data", ":/data"),
            ("./a:/data:ro:extra", "./a:/data:ro:extra"),
            ("./a:/data:rshared", "rshared"),
            ({"type": "tmpfs", "target": "/tmp"}, "type"),
            ({"type": "volume", "target": "/data"}, "source"),
            ({"type": "bind", "target": "/data"}, "source"),
            ({"type": "bind", "source": "./a", "target": "data"}, "target"),
            ({"type": "bind", "source": "./a", "target": "/data", "bind": {"propagation": "rshared"}}, "propagation"),
            ({"type": "volume", "source": "v", "target": "/data", "volume": {"subpath": "x"}}, "subpath"),
            ({"type": "volume", "source": "v", "target": "/data", "tmpfs": {}}, "tmpfs"),
            (42, "volumes"),
            # Review (M22): non-mapping option blocks escaped as TypeError/AttributeError, aborting prepare.
            ({"type": "bind", "source": "./seed.json", "target": "/seed.json", "bind": 123}, "bind"),
            ({"type": "volume", "source": "v", "target": "/data", "volume": 123}, "volume"),
            ({"type": "volume", "source": "v", "target": "/data", "volume": ["nocopy"]}, "volume"),
        ],
    )
    def test_unsupported_volume_entries_name_the_field(self, entry, name):
        with pytest.raises(ComposeVolumeError) as exc:
            parse_volume(entry, "api")
        assert exc.value.name == name

    def test_a_read_only_task_bind_is_uploaded_with_a_warning_naming_the_service_and_path(self, tmp_path, caplog):
        """Review (M22): a `:ro` task bind becomes a writable copy (no bind mounts in a sandbox), so prepare says so
        once, naming the service, the source and the target (`warn_read_only`); the server, which plans the same
        document again at every seed, and a plain bind stay silent. Issue #5062."""
        env = tmp_path / "task" / "environment"
        env.mkdir(parents=True)
        (env / "seed.json").write_text("{}")
        document = {"services": {"main": {"image": "img", "volumes": ["./seed.json:/app/seed.json:ro"]}}}
        with caplog.at_level("WARNING", logger="nemo_gym.sandbox.compose_config"):
            uploads = plan_compose_volumes(document, env, root=env.parent, warn_read_only=True)
        assert uploads == {"main": [{"source": str(env / "seed.json"), "target": "/app/seed.json"}]}
        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "'main'" in warnings[0] and "./seed.json" in warnings[0] and "/app/seed.json" in warnings[0]
        assert "writable copy" in warnings[0] and "tests/" in warnings[0]
        caplog.clear()
        # The server plans the same document again at every seed, with the default: silent.
        document = {"services": {"main": {"image": "img", "volumes": ["./seed.json:/app/seed.json:ro"]}}}
        with caplog.at_level("WARNING", logger="nemo_gym.sandbox.compose_config"):
            plan_compose_volumes(document, env, root=env.parent)
        assert [r for r in caplog.records if r.levelname == "WARNING"] == []
        document = {"services": {"main": {"image": "img", "volumes": ["./seed.json:/app/seed.json"]}}}
        with caplog.at_level("WARNING", logger="nemo_gym.sandbox.compose_config"):
            plan_compose_volumes(document, env, root=env.parent, warn_read_only=True)
        assert [r for r in caplog.records if r.levelname == "WARNING"] == []

    def test_task_aw_bind_mount_becomes_an_upload_into_the_sidecar(self, tmp_path):
        env = tmp_path / "task-aw" / "environment"
        shutil.copytree(VOLUME_FIXTURES / "task-aw" / "environment", env)
        document = yaml.safe_load((env / "docker-compose.yaml").read_text())
        uploads = plan_compose_volumes(document, env, root=env.parent)
        api = document["services"]["api"]
        assert "volumes" not in api
        assert uploads == {"api": [{"source": str(env / "seed.json"), "target": "/app/seed.json"}]}
        assert api["x-sandbox"] == {"uploads": uploads["api"]}
        assert "x-sandbox" not in document["services"]["main"]
        # Nothing else is needed from the deployment: no shared storage, no volume_sources.
        assert "volumes" not in document

    def test_context_dir_folder_bind_and_task_root_paths_upload(self, tmp_path):
        env = tmp_path / "t" / "environment"
        (env / "data").mkdir(parents=True)
        (env / "data" / "a.txt").write_text("a")
        (tmp_path / "t" / "tests").mkdir()
        document = {
            "services": {
                "main": {"volumes": [f"{env}/data:/data", {"type": "bind", "source": "../tests", "target": "/tests"}]}
            }
        }
        uploads = plan_compose_volumes(document, env, root=env.parent)
        assert uploads["main"] == [
            {"source": str(env / "data"), "target": "/data"},
            {"source": str(tmp_path / "t" / "tests"), "target": "/tests"},
        ]

    def test_named_volumes_stay_in_long_syntax_for_shared_storage(self, tmp_path):
        document = {
            "services": {
                "db": {"volumes": ["pgdata:/var/lib/postgresql/data"]},
                "backup": {
                    "volumes": [{"type": "volume", "source": "pgdata", "target": "/mnt/db", "read_only": True}]
                },
            },
            "volumes": {"pgdata": None},
        }
        assert plan_compose_volumes(document, tmp_path) == {}
        assert document["services"]["db"]["volumes"] == [
            {"type": "volume", "source": "pgdata", "target": "/var/lib/postgresql/data", "read_only": False}
        ]
        assert document["services"]["backup"]["volumes"][0]["read_only"] is True
        # The canonical null declaration is normalised to the mapping the adapter reads (review finding).
        assert document["volumes"] == {"pgdata": {}}

    def test_external_volume_needs_a_volume_sources_entry(self, tmp_path):
        document = {"services": {"db": {"volumes": ["pgdata:/data"]}}, "volumes": {"pgdata": {"external": True}}}
        with pytest.raises(ComposeVolumeError, match="external volume 'pgdata' needs a `volume_sources` entry") as exc:
            plan_compose_volumes(copy.deepcopy(document), tmp_path)
        assert exc.value.name == "pgdata"
        plan_compose_volumes(document, tmp_path, volume_sources={"pgdata": "datasets/pgdata"})
        assert document["services"]["db"]["volumes"][0]["source"] == "pgdata"

    def test_foreign_bind_mount_needs_a_volume_sources_entry(self, tmp_path):
        document = {"services": {"api": {"volumes": ["/var/lib/models:/models:ro"]}}}
        with pytest.raises(ComposeVolumeError, match="outside the task") as exc:
            plan_compose_volumes(copy.deepcopy(document), tmp_path)
        assert exc.value.name == "/var/lib/models"
        plan_compose_volumes(document, tmp_path, volume_sources={"/var/lib/models": "models"})
        assert document["services"]["api"]["volumes"] == [
            {"type": "bind", "source": "/var/lib/models", "target": "/models", "read_only": True}
        ]

    @pytest.mark.parametrize(
        "document,name",
        [
            ({"services": {"db": {"volumes": ["pgdata:/data"]}}}, "pgdata"),
            ({"services": {"db": {"volumes": ["pgdata:/data"]}}, "volumes": {"pgdata": {"driver": "nfs"}}}, "driver"),
            ({"services": {"db": {"volumes": ["./missing.json:/x.json"]}}}, "./missing.json"),
            ({"services": {"db": {"volumes": "./a:/a"}}}, "volumes"),
            ({"services": {"db": {}}, "volumes": []}, "volumes"),
            # Review (round 14): a non-mapping declaration is the planner's own error, naming the volume, not a
            # ``TypeError`` out of ``set(declaration)`` that aborted prepare for every sibling task.
            ({"services": {"db": {"volumes": ["pgdata:/data"]}}, "volumes": {"pgdata": 123}}, "pgdata"),
            ({"services": {"db": {}}, "volumes": {"pgdata": "abc"}}, "pgdata"),
        ],
    )
    def test_undeclared_missing_and_malformed_volumes_fail_naming_them(self, tmp_path, document, name):
        with pytest.raises(ComposeVolumeError) as exc:
            plan_compose_volumes(document, tmp_path)
        assert exc.value.name == name

    def test_a_link_leaving_the_task_folder_is_refused(self, tmp_path):
        env = tmp_path / "t" / "environment"
        env.mkdir(parents=True)
        outside = tmp_path / "outside.json"
        outside.write_text("{}")
        (env / "seed.json").symlink_to(outside)
        with pytest.raises(ComposeVolumeError, match="leaves the task folder") as exc:
            plan_compose_volumes({"services": {"api": {"volumes": ["./seed.json:/seed.json"]}}}, env, root=env.parent)
        assert exc.value.name == "./seed.json"

    def test_a_relative_foreign_bind_resolves_through_volume_sources(self, tmp_path):
        """Review (round 14): ``../../shared:/data`` from ``task/environment`` is ordinary traversal out of the task,
        which Docker binds as a plain host path relative to the Compose file's folder; the symlink-escape check
        compared the unresolved path lexically, took the ``..`` for a link leaving the task, and never reached the
        ``volume_sources`` lookup. It is a foreign source now, like an absolute one, under the literal key."""
        task = tmp_path / "t"
        env = task / "environment"
        env.mkdir(parents=True)
        (tmp_path / "shared").mkdir()
        document = {"services": {"api": {"volumes": ["../../shared:/data"]}}}
        with pytest.raises(ComposeVolumeError, match="outside the task") as exc:
            plan_compose_volumes(copy.deepcopy(document), env, root=task)
        assert exc.value.name == "../../shared"
        assert plan_compose_volumes(document, env, root=task, volume_sources={"../../shared": "shared"}) == {}
        assert document["services"]["api"]["volumes"] == [
            {"type": "bind", "source": "../../shared", "target": "/data", "read_only": False}
        ]
        assert document_volume_sources(document, {"../../shared": "shared", "other": "x"}) == {
            "../../shared": "shared"
        }
        # Traversal that stays inside the task is still an upload, and a link that leaves it is still refused.
        inside = {"services": {"api": {"volumes": ["../environment:/env"]}}}
        assert plan_compose_volumes(inside, env, root=task) == {"api": [{"source": str(env), "target": "/env"}]}
        (tmp_path / "shared" / "x.json").write_text("{}")
        (env / "escape.json").symlink_to(tmp_path / "shared" / "x.json")
        with pytest.raises(ComposeVolumeError, match="leaves the task folder"):
            plan_compose_volumes({"services": {"api": {"volumes": ["./escape.json:/x.json"]}}}, env, root=task)

    def test_a_link_with_an_absolute_target_inside_the_task_is_refused_and_a_relative_one_is_not(self, tmp_path):
        # Review: the server plans on a copy of the task folder (links copied as links). An absolute target still
        # points into the original, so a source reached through one would pass prepare and fail seed; the planner
        # refuses it like the hub refuses such a link in a fetched task. A relative link agrees at both ends.
        task = tmp_path / "t"
        env = task / "environment"
        (env / "realdata" / "sub").mkdir(parents=True)
        (env / "realdata" / "sub" / "a.txt").write_text("a")
        (env / "real.json").write_text("{}")
        (env / "data").symlink_to(env / "realdata", target_is_directory=True)
        (env / "seed.json").symlink_to(env / "real.json")

        def document(*entries):
            return {"services": {"api": {"volumes": list(entries)}}}

        for source, link in [("./data", "data"), ("./data/sub", "data"), ("./seed.json", "seed.json")]:
            with pytest.raises(ComposeVolumeError, match=rf"absolute target: .*environment/{link}; make") as exc:
                plan_compose_volumes(document(f"{source}:/mnt"), env, root=task)
            assert exc.value.name == source
        (env / "data").unlink()
        (env / "seed.json").unlink()
        (env / "data").symlink_to("realdata", target_is_directory=True)
        (env / "seed.json").symlink_to("real.json")
        entries = ("./data:/data", "./data/sub:/sub", "./seed.json:/seed.json")
        uploads = plan_compose_volumes(document(*entries), env, root=task)
        assert uploads == {
            "api": [
                {"source": str(env.resolve() / "realdata"), "target": "/data"},
                {"source": str(env.resolve() / "realdata" / "sub"), "target": "/sub"},
                {"source": str(env.resolve() / "real.json"), "target": "/seed.json"},
            ]
        }
        # The same plan on a copy made the way the server snapshots a task: the links resolve inside the copy.
        copy = tmp_path / "copy"
        shutil.copytree(task, copy, symlinks=True)
        uploads = plan_compose_volumes(document(*entries), copy / "environment", root=copy)
        assert [entry["source"] for entry in uploads["api"]] == [
            str(copy.resolve() / "environment" / "realdata"),
            str(copy.resolve() / "environment" / "realdata" / "sub"),
            str(copy.resolve() / "environment" / "real.json"),
        ]

    def test_a_link_inside_a_bound_folder_is_refused_naming_it(self, tmp_path):
        # Review: a folder is uploaded file by file, so a link inside it would be read through to the host.
        env = tmp_path / "t" / "environment"
        (env / "data" / "sub").mkdir(parents=True)
        (env / "data" / "ok.txt").write_text("ok")
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")
        (env / "data" / "sub" / "leak").symlink_to(outside)
        with pytest.raises(ComposeVolumeError, match=r"contains a link: .*data/sub/leak") as exc:
            plan_compose_volumes({"services": {"api": {"volumes": ["./data:/data"]}}}, env, root=env.parent)
        assert exc.value.name == "./data"
        # A relative link that stays inside the task is refused too: no link is ever uploaded.
        (env / "data" / "sub" / "leak").unlink()
        (env / "data" / "sub" / "alias").symlink_to("../ok.txt")
        with pytest.raises(ComposeVolumeError, match="contains a link"):
            plan_compose_volumes({"services": {"api": {"volumes": ["./data:/data"]}}}, env, root=env.parent)

    @pytest.mark.parametrize("service", ["main", "api"])
    @pytest.mark.parametrize("source", ["/etc/passwd", "OUTSIDE"])
    def test_a_task_authored_x_sandbox_is_refused_naming_the_field(self, tmp_path, service, source):
        # Review blocker: x-sandbox is the planner's output. A task file carrying one could name any host file
        # as an upload source, so it fails before any volume is looked at, naming the field.
        env = tmp_path / "t" / "environment"
        env.mkdir(parents=True)
        outside = tmp_path / "outside" / "secret"
        outside.parent.mkdir()
        outside.write_text("secret")
        source = str(outside) if source == "OUTSIDE" else source
        document = {
            "services": {
                "main": {},
                "api": {"image": "a"},
            }
        }
        document["services"][service]["x-sandbox"] = {"uploads": [{"source": source, "target": "/tmp/stolen"}]}
        with pytest.raises(ComposeVolumeError, match=f"service {service!r} declares x-sandbox") as exc:
            plan_compose_volumes(document, env, root=env.parent)
        assert exc.value.name == "x-sandbox"
        # Any x-sandbox key, not only uploads: the sidecar adaptation's keys are Gym's too.
        document["services"][service]["x-sandbox"] = {"hosts": []}
        with pytest.raises(ComposeVolumeError) as exc:
            plan_compose_volumes(document, env, root=env.parent)
        assert exc.value.name == "x-sandbox"

    @pytest.mark.parametrize(
        "entry",
        [
            "./seed.json:app/seed.json",
            "./seed.json:/app/../x",
            "data:rel/path",
            {"type": "bind", "source": "./seed.json", "target": "/app/../x"},
            "./seed.json:/",
            {"type": "bind", "source": "./seed.json", "target": "//"},
            "data:/",
            # M22 review: `/.` passed as a non-empty name and a folder upload merged into `/`.
            "./seed.json:/.",
            {"type": "bind", "source": "./seed.json", "target": "//./"},
            "data:/././",
        ],
    )
    def test_targets_are_absolute_below_the_root_without_dotdot_in_both_syntaxes(self, tmp_path, entry):
        # Review: the short syntax took any target; a relative or `..` target failed only at seed, and `/` was
        # accepted (chmod and uploads over the whole container).
        (tmp_path / "seed.json").write_text("{}")
        document = {"services": {"api": {"volumes": [entry]}}, "volumes": {"data": {}}}
        with pytest.raises(ComposeVolumeError, match="target") as exc:
            plan_compose_volumes(document, tmp_path)
        assert exc.value.name == "target"
        # Targets below the root, a trailing `/` or `/.` included, are kept as written.
        for target in ("/data", "/data/", "/data/."):
            document = {"services": {"api": {"volumes": [f"./seed.json:{target}"]}}}
            plan_compose_volumes(document, tmp_path)
            assert document["services"]["api"]["x-sandbox"]["uploads"][0]["target"] == target

    def test_an_unknown_home_user_in_a_bind_source_names_the_source(self, tmp_path):
        # Review: Path.expanduser raised a bare RuntimeError for `~user` with no such user.
        document = {"services": {"db": {"volumes": ["~nosuchuser_xyz/x:/x"]}}}
        with pytest.raises(ComposeVolumeError, match="cannot resolve the home directory") as exc:
            plan_compose_volumes(document, tmp_path)
        assert exc.value.name == "~nosuchuser_xyz/x"

    def test_logs_tree_binds_on_main_become_adapter_log_paths_and_are_refused_on_a_sidecar(self, tmp_path):
        # Review (round 13): the binds used to be dropped, but in a Compose group nothing created /logs/* before
        # main's command and healthcheck ran (the server's mkdir comes after the group is up), and a bind with a
        # target elsewhere (`/logs/artifacts:/app/output`) lost its target. They now instruct the adapter, which
        # creates the source and links the target before the command, keeping the real target.
        document = {
            "services": {
                "main": {"volumes": ["/logs/verifier:/logs/verifier", "/logs:/logs", "/logs/artifacts:/app/output"]}
            }
        }
        assert plan_compose_volumes(document, tmp_path) == {}
        main = document["services"]["main"]
        assert "volumes" not in main
        assert main["x-sandbox"] == {
            "logs": [
                {"source": "/logs/verifier", "target": "/logs/verifier"},
                {"source": "/logs", "target": "/logs"},
                {"source": "/logs/artifacts", "target": "/app/output"},
            ]
        }
        # A main that also binds a task file carries both instructions under the one planner key.
        (tmp_path / "environment").mkdir()
        (tmp_path / "environment" / "seed.json").write_text("{}")
        document = {"services": {"main": {"volumes": ["/logs/agent:/logs/agent", "./seed.json:/app/seed.json"]}}}
        uploads = plan_compose_volumes(document, tmp_path / "environment", root=tmp_path)
        assert document["services"]["main"]["x-sandbox"] == {
            "uploads": uploads["main"],
            "logs": [{"source": "/logs/agent", "target": "/logs/agent"}],
        }
        # A logs path with a `..` segment used to pass as written (the adapter would `mkdir -p` and `chmod 777`
        # `/logs/../etc`); it is refused naming the path (repair round 2).
        for source in ("/logs/../etc", "/logs/verifier/.."):
            document = {"services": {"main": {"volumes": [f"{source}:/x"]}}}
            with pytest.raises(ComposeVolumeError, match="a logs path with a `..` segment") as exc:
                plan_compose_volumes(document, tmp_path)
            assert exc.value.name == source
            assert "x-sandbox" not in document["services"]["main"]

    @pytest.mark.parametrize(
        "entry",
        [
            "/logs:/logs/child",
            "/logs/verifier:/logs",
            "/logs/verifier:/logs/verifier/x/",
            # The nesting shows once the target is spelled out: `/logs/.` and `/./logs` and `//logs` are `/logs`.
            "/logs/verifier:/logs/.",
            "/logs/verifier:/logs/./",
            "/logs/verifier:/./logs",
            "/logs/verifier://logs",
            "/logs:/logs//child/.",
        ],
    )
    def test_a_main_logs_target_that_nests_its_source_is_refused_at_prepare(self, tmp_path, entry):
        """M22 review: the planner emitted the instruction and the adapter's nesting guard failed every seed; the
        README promised the refusal at prepare. Refused naming the field and both paths, in whatever spelling the
        target nests the path (the rstrip comparison let `/logs/verifier:/logs/.` through to the seed, where the
        adapter, once it linked the lexical form, removed the whole tree); equal paths, a remap beside the tree
        and a sidecar whose target nests its source (a real mount, not a link) are untouched."""
        source, target = entry.split(":")
        document = {"services": {"main": {"volumes": [entry]}}}
        with pytest.raises(ComposeVolumeError, match="nests it") as exc:
            plan_compose_volumes(document, tmp_path)
        assert exc.value.name == "target"
        assert f"{source!r} at {target!r}" in str(exc.value) and "service 'main'" in str(exc.value)
        assert "x-sandbox" not in document["services"]["main"]
        document = {
            "services": {
                "main": {"volumes": ["/logs/verifier:/logs/verifier", "/logs/artifacts:/app/output", "/logs:/logs/"]},
                "api": {"volumes": ["/logs/verifier:/logs/verifier/nested"]},
            }
        }
        plan_compose_volumes(document, tmp_path, shared_storage=True)
        assert document["services"]["main"]["x-sandbox"]["logs"] == [
            {"source": "/logs/verifier", "target": "/logs/verifier"},
            {"source": "/logs/artifacts", "target": "/app/output"},
            {"source": "/logs", "target": "/logs/"},
        ]
        assert document["services"]["api"]["volumes"][0]["target"] == "/logs/verifier/nested"
        # A remap target spelled with a trailing separator or dot is kept as written, like every other target; the
        # adapter links its lexical form (`/app/output`) at seed. A sibling of the tree spelled with a dot
        # (`/logs/artifacts:/logs/verifier/.`) is the remap `/logs/artifacts:/logs/verifier` is, not a nesting.
        for target in ("/app/output/", "/app/output/.", "/app/output/./", "/logs/verifier/."):
            document = {"services": {"main": {"volumes": [f"/logs/artifacts:{target}"]}}}
            plan_compose_volumes(document, tmp_path)
            assert document["services"]["main"]["x-sandbox"]["logs"] == [
                {"source": "/logs/artifacts", "target": target}
            ]
        # The path bound at a spelling of itself is the path itself (the adapter creates it, no link), where the
        # rstrip comparison refused `/logs/verifier/.` as nesting and accepted `//logs/verifier` as a self-link.
        for target in ("//logs/verifier", "/logs//verifier", "/logs/verifier/.", "/logs/./verifier"):
            document = {"services": {"main": {"volumes": [f"/logs/verifier:{target}"]}}}
            plan_compose_volumes(document, tmp_path)
            assert document["services"]["main"]["x-sandbox"]["logs"] == [
                {"source": "/logs/verifier", "target": target}
            ]

    def test_a_sidecar_logs_bind_is_shared_with_main_through_a_shared_storage_volume(self, tmp_path):
        """M35 D2: kumo's ``verifier`` sidecar binds ``/logs/verifier`` and appends ``actions.jsonl`` there, which
        ``main``'s ``tests/score.py`` reads; under Harbor both bind one host folder. The planner used to refuse the
        sidecar (``only main can mount /logs``), 10,350 tasks. It now shares the path as a named volume on the
        provider's shared storage: declared, ``nocopy``, mounted at the sidecar's target and at the path in ``main``
        (which keeps its own logs instruction for the path, so the adapter still creates and widens it)."""
        document = {
            "services": {
                "verifier": {"volumes": ["/logs/verifier:/logs/verifier"]},
                "main": {"volumes": ["/logs/verifier:/logs/verifier", "/logs/agent:/logs/agent"]},
            }
        }
        assert plan_compose_volumes(document, tmp_path, shared_storage=True) == {}
        mount = {
            "type": "volume",
            "source": "nemo-gym-logs-verifier",
            "target": "/logs/verifier",
            "read_only": False,
            "volume": {"nocopy": True},
        }
        assert document["volumes"] == {"nemo-gym-logs-verifier": {}}
        assert document["services"]["verifier"]["volumes"] == [mount]
        assert document["services"]["verifier"]["x-sandbox"] == {
            "logs": [{"source": "/logs/verifier", "target": "/logs/verifier"}]
        }
        assert document["services"]["main"]["volumes"] == [mount]
        assert document["services"]["main"]["x-sandbox"] == {
            "logs": [
                {"source": "/logs/verifier", "target": "/logs/verifier"},
                {"source": "/logs/agent", "target": "/logs/agent"},
            ]
        }
        # A sidecar target elsewhere (`/logs/artifacts:/data`, read-only) is mounted there; main gets the path itself
        # whether or not it binds it, since that is where the verifier collects from.
        document = {"services": {"main": {}, "api": {"volumes": ["/logs/artifacts:/data:ro"]}}}
        plan_compose_volumes(document, tmp_path)
        assert document["services"]["api"]["volumes"] == [
            {**mount, "source": "nemo-gym-logs-artifacts", "target": "/data", "read_only": True}
        ]
        assert "x-sandbox" not in document["services"]["api"]
        assert document["services"]["main"]["volumes"] == [
            {**mount, "source": "nemo-gym-logs-artifacts", "target": "/logs/artifacts"}
        ]
        # Without shared storage the share cannot be made: refused at prepare, naming the setting.
        document = {"services": {"main": {}, "api": {"volumes": ["/logs/x:/elsewhere"]}}}
        with pytest.raises(ComposeVolumeError, match="shared_storage.host_path") as exc:
            plan_compose_volumes(document, tmp_path, shared_storage=False)
        assert exc.value.name == "/logs/x" and "service 'api' binds the logs tree '/logs/x'" in str(exc.value)
        assert "the `sandbox:` block, where 'api' runs, configures none" in str(exc.value)
        # A path whose characters cannot name a Compose volume (`/logs/my dir` would be `nemo-gym-logs-my dir`) is
        # refused here, not by the adapter at every seed (repair round 3); `/logs/.hidden` names a valid volume.
        space = {"type": "bind", "source": "/logs/my dir", "target": "/x"}
        document = {"services": {"main": {}, "api": {"volumes": [space]}}}
        with pytest.raises(ComposeVolumeError, match="not a Compose volume name") as exc:
            plan_compose_volumes(document, tmp_path, shared_storage=True)
        assert exc.value.name == "/logs/my dir" and "'nemo-gym-logs-my dir'" in str(exc.value)
        document = {"services": {"main": {}, "api": {"volumes": ["/logs/.hidden:/x"]}}}
        plan_compose_volumes(document, tmp_path, shared_storage=True)
        assert document["volumes"] == {"nemo-gym-logs-.hidden": {}}
        # A GPU-reserving binder shares across blocks: the volume is mounted in `main` through the task's block
        # and in the binder through the GPU block, so both need the storage, and the refusal names the one that
        # lacks it (repair round 1: only the binder's block was consulted, and `main` was given a volume its
        # provider could not mount, the D6 seed failure again).
        gpu = {"deploy": {"resources": {"reservations": {"gpus": 1}}}}
        document = {"services": {"main": {}, "trainer": {**gpu, "volumes": ["/logs/verifier:/logs/verifier"]}}}
        with pytest.raises(ComposeVolumeError, match="the `sandbox:` block, where 'main' runs, configures none"):
            plan_compose_volumes(
                copy.deepcopy(document), tmp_path, gpu_volume_sources={}, shared_storage=False, gpu_shared_storage=True
            )
        with pytest.raises(
            ComposeVolumeError, match="the `gpu_sandbox_provider` block, where 'trainer' runs, configures none"
        ):
            plan_compose_volumes(
                copy.deepcopy(document), tmp_path, gpu_volume_sources={}, shared_storage=True, gpu_shared_storage=False
            )
        for storage in ((True, True), (None, True), (True, None)):
            planned = copy.deepcopy(document)
            plan_compose_volumes(
                planned, tmp_path, gpu_volume_sources={}, shared_storage=storage[0], gpu_shared_storage=storage[1]
            )
            assert planned["services"]["main"]["volumes"] == [
                {**mount, "source": "nemo-gym-logs-verifier", "target": "/logs/verifier"}
            ]
        # The volume name is Gym's: a task that declares it is told so.
        document = {"services": {"api": {"volumes": ["/logs/x:/x"]}}, "volumes": {"nemo-gym-logs-x": {}}}
        with pytest.raises(ComposeVolumeError, match="is the name Gym gives the shared logs path '/logs/x'"):
            plan_compose_volumes(document, tmp_path)
        # M22 review: two paths that spell one name (`/logs/a-b`, `/logs/a/b`) were refused as that declaration;
        # the second keeps a distinct, deterministic name and each stays its own share with `main`, while repeated
        # binds of one path share one volume.
        document = {
            "services": {
                "main": {},
                "api": {"volumes": ["/logs/a-b:/one", "/logs/a/b:/two", "/logs/a-b:/one-again:ro"]},
            }
        }
        plan_compose_volumes(document, tmp_path, shared_storage=True)
        twin = "nemo-gym-logs-a-b-" + hashlib.sha256(b"/logs/a/b").hexdigest()[:8]
        assert document["volumes"] == {"nemo-gym-logs-a-b": {}, twin: {}}
        assert [(v["source"], v["target"], v["read_only"]) for v in document["services"]["api"]["volumes"]] == [
            ("nemo-gym-logs-a-b", "/one", False),
            ("nemo-gym-logs-a-b", "/one-again", True),
            (twin, "/two", False),
        ]
        assert [(v["source"], v["target"]) for v in document["services"]["main"]["volumes"]] == [
            ("nemo-gym-logs-a-b", "/logs/a-b"),
            (twin, "/logs/a/b"),
        ]
        again = {"services": {"main": {}, "api": {"volumes": ["/logs/a-b:/one", "/logs/a/b:/two"]}}}
        plan_compose_volumes(again, tmp_path, shared_storage=True)
        assert list(again["volumes"]) == ["nemo-gym-logs-a-b", twin]
        # Repair round 2: a sidecar's `/logs/../etc:/x` passed prepare as the volume `nemo-gym-logs-..-etc` mounted
        # at `/logs/../etc` in `main`, which the provider refuses at every seed (`Shared storage target must be an
        # absolute container path without traversal`): the prepare-vs-seed gap this round closes, refused here.
        for source in ("/logs/../etc", "/logs/verifier/.."):
            document = {"services": {"main": {}, "api": {"volumes": [f"{source}:/x"]}}}
            with pytest.raises(ComposeVolumeError, match="a logs path with a `..` segment") as exc:
                plan_compose_volumes(document, tmp_path, shared_storage=True)
            assert exc.value.name == source and f"service 'api' binds {source!r}" in str(exc.value)
            assert "volumes" not in document and "volumes" not in document["services"]["main"]

    @pytest.mark.parametrize("main", ["omitted", None, {}], ids=["omitted", "null", "empty"])
    def test_a_shared_logs_bind_reaches_an_implicit_or_null_main(self, tmp_path, main):
        """M22 review: a valid overlay may declare only the sidecar, or ``main:`` with no body (Harbor's base
        ``main`` unchanged); the share was added to ``main`` only when it was written as a mapping, so the ``main``
        the resolver then made held a private ``/logs/verifier`` the sidecar's ``actions.jsonl`` never reached."""
        document = {"services": {"verifier": {"image": SIDECAR, "volumes": ["/logs/verifier:/logs/verifier"]}}}
        if main != "omitted":
            document["services"]["main"] = copy.deepcopy(main)
        assert plan_compose_volumes(document, tmp_path, shared_storage=True) == {}
        resolved = resolve_compose(document, MAIN, IMAGE_CONFIGS)
        mount = {
            "type": "volume",
            "source": "nemo-gym-logs-verifier",
            "target": "/logs/verifier",
            "read_only": False,
            "volume": {"nocopy": True},
        }
        assert resolved["volumes"] == {"nemo-gym-logs-verifier": {}}
        assert resolved["services"]["verifier"]["volumes"] == [mount]
        assert resolved["services"]["main"]["volumes"] == [mount]
        assert resolved["services"]["main"]["image"] == IMAGE_CONFIGS[MAIN]["image"]
        assert "x-sandbox" not in resolved["services"]["main"]
        # Without a shared logs bind nothing is normalised: the document is left as written.
        document = {"services": {"api": {"image": SIDECAR}}}
        if main != "omitted":
            document["services"]["main"] = copy.deepcopy(main)
        plan_compose_volumes(document, tmp_path, shared_storage=True)
        assert document["services"].get("main", "omitted") == main

    def test_a_named_volume_without_shared_storage_fails_prepare_naming_the_setting(self, tmp_path):
        """M35 D6: hil-bench's ``ask-human-metrics`` volume (shared by ``main`` and the sidecar) materialized on a
        deployment without ``shared_storage`` and 100/100 seeds failed with ``OpenSandbox shared storage requires
        shared_storage.host_path``. With the deployment's answer, prepare leaves the task out naming the setting;
        without it (``None``) the seed decides, as before."""
        document = {
            "services": {
                "main": {"volumes": ["metrics:/harbor_shared"]},
                "ask-human": {"volumes": ["metrics:/harbor_shared"]},
            },
            "volumes": {"metrics": None},
        }
        with pytest.raises(ComposeVolumeError, match="shared_storage.host_path") as exc:
            plan_compose_volumes(copy.deepcopy(document), tmp_path, shared_storage=False)
        assert exc.value.name == "metrics" and "service 'main' mounts 'metrics'" in str(exc.value)
        for storage in (True, None):
            planned = copy.deepcopy(document)
            assert plan_compose_volumes(planned, tmp_path, shared_storage=storage) == {}
            assert planned["services"]["ask-human"]["volumes"][0]["source"] == "metrics"
        # A mapped external volume and a mapped foreign bind need the storage as well.
        for document in (
            {"services": {"db": {"volumes": ["pgdata:/data"]}}, "volumes": {"pgdata": {"external": True}}},
            {"services": {"db": {"volumes": ["/var/lib/models:/models"]}}},
        ):
            with pytest.raises(ComposeVolumeError, match="needs the sandbox provider's shared storage"):
                plan_compose_volumes(
                    document,
                    tmp_path,
                    volume_sources={"pgdata": "pg", "/var/lib/models": "models"},
                    shared_storage=False,
                )
        # A GPU-reserving sidecar is answered for the GPU block, as its volumes are mounted through that block.
        gpu = {"deploy": {"resources": {"reservations": {"gpus": 1}}}}
        document = {
            "services": {"main": {}, "trainer": {**gpu, "volumes": ["ckpt:/ckpt"]}},
            "volumes": {"ckpt": {}},
        }
        plan_compose_volumes(
            copy.deepcopy(document), tmp_path, gpu_volume_sources={}, shared_storage=False, gpu_shared_storage=True
        )
        with pytest.raises(ComposeVolumeError, match="service 'trainer' mounts 'ckpt'") as exc:
            plan_compose_volumes(
                document, tmp_path, gpu_volume_sources={}, shared_storage=True, gpu_shared_storage=False
            )
        assert "the `gpu_sandbox_provider` block, where 'trainer' runs, configures none" in str(exc.value)

    def test_a_bind_source_the_task_digest_skips_is_refused(self, tmp_path):
        # Review (round 13): the planner accepted any bind source under the task root, but content_hash covers
        # only Harbor's packager set (environment/, tests/, solution/, steps/ and the top-level files), so an
        # edit to <task>/data/seed.json changed neither the digest nor the manifest while the server uploaded the
        # edited bytes. The hash stays the publisher's (parity); the planner is the guard.
        task = tmp_path / "t"
        env = task / "environment"
        (env / "data" / "__pycache__").mkdir(parents=True)
        (env / "data" / "__pycache__" / "x.pyc").write_bytes(b"")
        (env / "data" / "a.txt").write_text("a")
        (env / "seed.json").write_text("{}")
        (task / "data").mkdir()
        (task / "data" / "seed.json").write_text("{}")
        (task / "notes.txt").write_text("n")
        (task / "tests").mkdir()
        for source in ("../data/seed.json", "../notes.txt", "./data/__pycache__/x.pyc", ".."):
            document = {"services": {"api": {"volumes": [f"{source}:/x"]}}}
            with pytest.raises(ComposeVolumeError, match="which the task digest does not cover") as exc:
                plan_compose_volumes(document, env, root=task)
            assert exc.value.name == source
        # Covered sources plan as before: a file under environment/, the tests/ folder, the top-level task file,
        # and a covered folder that holds a byte-code cache after a run (the leftover does not change identity).
        document = {
            "services": {
                "api": {"volumes": ["./seed.json:/app/seed.json", "../tests:/tests", "./data:/data"]},
                "main": {"volumes": ["../instruction.md:/instruction.md"]},
            }
        }
        (task / "instruction.md").write_text("do it")
        uploads = plan_compose_volumes(document, env, root=task)
        assert [u["source"] for u in uploads["api"]] == [
            str(env / "seed.json"),
            str(task / "tests"),
            str(env / "data"),
        ]
        assert uploads["main"] == [{"source": str(task / "instruction.md"), "target": "/instruction.md"}]

    def test_a_gpu_reserving_sidecar_resolves_external_volumes_through_the_gpu_mapping(self, tmp_path):
        # Review (round 13): the server routes a GPU-reserving sidecar of a CPU task to the GPU block, so its
        # external volumes resolve through that block's mapping, not the task's; main follows the task.
        gpu_sidecar = {
            "image": "img",
            "deploy": {"resources": {"reservations": {"gpus": 1}}},
            "volumes": ["models:/models"],
        }
        document = {
            "services": {"main": {"image": "img"}, "model": copy.deepcopy(gpu_sidecar)},
            "volumes": {"models": {"external": True}},
        }
        # Mapped only in the GPU block: passes once the GPU mapping is given.
        with pytest.raises(ComposeVolumeError, match="external volume 'models'"):
            plan_compose_volumes(copy.deepcopy(document), tmp_path, volume_sources={})
        plan_compose_volumes(copy.deepcopy(document), tmp_path, volume_sources={}, gpu_volume_sources={"models": "g"})
        # A GPU block without the mapping does not fall back to the task's block.
        with pytest.raises(ComposeVolumeError, match="external volume 'models'"):
            plan_compose_volumes(
                copy.deepcopy(document), tmp_path, volume_sources={"models": "c"}, gpu_volume_sources={}
            )
        # A sidecar that reserves no GPUs, and main, stay on the task's mapping whatever the GPU block maps.
        cpu_document = copy.deepcopy(document)
        del cpu_document["services"]["model"]["deploy"]
        cpu_document["services"]["main"]["volumes"] = ["models:/models"]
        with pytest.raises(ComposeVolumeError, match="external volume 'models'"):
            plan_compose_volumes(
                copy.deepcopy(cpu_document), tmp_path, volume_sources={}, gpu_volume_sources={"models": "g"}
            )
        plan_compose_volumes(cpu_document, tmp_path, volume_sources={"models": "c"}, gpu_volume_sources={})

    def test_document_volume_sources_keeps_only_what_the_document_resolves_externally(self, tmp_path):
        # Review (round 13): the adapter treats any key of its mapping as an externally managed volume, so the
        # deployment-wide mapping must be narrowed to what this document declares external or binds from outside.
        document = {
            "services": {
                "db": {"volumes": ["pgdata:/var/lib/postgresql/data"]},
                "api": {"volumes": ["models:/models", "/var/lib/weights:/weights:ro"]},
                "main": {},
            },
            "volumes": {"pgdata": None, "models": {"external": True}},
        }
        mapping = {
            "pgdata": "datasets/pgdata",  # another task's external volume: this one declares pgdata ordinary
            "models": "shared/models",
            "/var/lib/weights": "weights",
            "unused": "elsewhere",
        }
        plan_compose_volumes(document, tmp_path, volume_sources=mapping)
        assert document_volume_sources(document, mapping) == {"models": "shared/models", "/var/lib/weights": "weights"}
        assert (
            document_volume_sources({}, mapping) == {} and document_volume_sources({"services": None}, mapping) == {}
        )

    def test_documents_without_services_or_volumes_are_left_alone(self, tmp_path):
        for document in ({}, {"services": None}, {"services": {"main": None, "api": {"volumes": None}}}):
            assert plan_compose_volumes(copy.deepcopy(document), tmp_path) == {}

    def test_non_root_sidecar_adaptation_keeps_uploads(self, tmp_path):
        seed = tmp_path / "seed.json"
        seed.write_text("{}")
        overlay = copy.deepcopy(OVERLAY)
        overlay["services"]["browser"]["volumes"] = ["./seed.json:/app/seed.json:ro"]
        uploads = plan_compose_volumes(overlay, tmp_path)
        document = adapt_non_root_sidecars(resolve_compose(overlay, MAIN, IMAGE_CONFIGS), IMAGE_CONFIGS)
        browser = document["services"]["browser"]
        assert browser["x-sandbox"] == {
            "uploads": uploads["browser"],
            "hosts": [],
            "resolve_environment": ["BROWSER_URL"],
        }
